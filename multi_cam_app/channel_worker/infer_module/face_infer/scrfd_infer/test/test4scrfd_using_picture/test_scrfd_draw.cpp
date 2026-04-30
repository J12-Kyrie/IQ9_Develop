/// @file test_scrfd_draw.cpp
/// @brief SCRFD 可视化测试: 将检测框 + 关键点绘制到原图上
///
/// 完整管线复用:
///   JPEG -> OpenCV(BGR->RGB) -> RGB->NV12(BT.601) -> DmaBuffer
///   -> FacePreprocess::RunScrfd(GPU letterbox)
///   -> QnnInferencer::Execute(QNN HTP)
///   -> ScrfdDecode(锚点解码 + NMS)
///   -> cv::rectangle + cv::circle + cv::putText -> imwrite
///
/// 用法:
///   ./test_scrfd_draw <image_dir> [kernel_dir] [model_path] [conf] [nms]
///
/// 输出:
///   <image_dir>/output/*.jpg (标注后的图片)

#include "../../QnnInferencer.hpp"
#include "../../FaceTypes.hpp"
#include "../../../face_preprocess/FacePreprocess.hpp"
#include "../../../mem_management/dma_buffer.hpp"
#include "../../../mem_management/opencl_loader.hpp"
#include "../../../mem_management/dma_sync_guard.hpp"
#include "../../../mem_management/mem_types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace face_infer;

// ============================================================
// 全局共享对象
// ============================================================
static std::shared_ptr<OpenClLoader> g_ocl;
static FacePreprocess  g_gpu;
static QnnInferencer   g_qnn;

// ============================================================
// NV12Frame 辅助 (复用 test_scrfd_accuracy.cpp 模式)
// ============================================================
struct NV12Frame {
    DmaBuffer buf;
    FramePlaneInfo plane;
    bool valid = false;

    bool Create(uint32_t w, uint32_t h, uint32_t stride,
                cl_context ctx, std::shared_ptr<OpenClLoader> ocl) {
        uint32_t nv12_size = stride * h * 3 / 2;
        if (!buf.Init(nv12_size)) {
            fprintf(stderr, "[NV12Frame] DmaBuffer::Init(%u) failed\n", nv12_size);
            return false;
        }
        if (!buf.BindOpenCl(ctx, ocl)) {
            fprintf(stderr, "[NV12Frame] BindOpenCl failed\n");
            return false;
        }
        plane.fd        = buf.fd();
        plane.frame_len = nv12_size;
        plane.y_offset  = 0;
        plane.uv_offset = stride * h;
        plane.y_stride  = stride;
        plane.uv_stride = stride;
        plane.width     = w;
        plane.height    = h;
        valid = true;
        return true;
    }

    void Destroy() {
        buf.Destroy();
        valid = false;
    }
};

// ============================================================
// RGB -> NV12 (BT.601 limited-range)
// ============================================================
static void rgb_to_nv12(const cv::Mat& rgb, uint8_t* nv12,
                        int w, int h, int stride) {
    for (int y = 0; y < h; y++) {
        const uint8_t* row = rgb.ptr<uint8_t>(y);
        for (int x = 0; x < w; x++) {
            int R = row[x * 3 + 0];
            int G = row[x * 3 + 1];
            int B = row[x * 3 + 2];
            int Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16;
            nv12[y * stride + x] = static_cast<uint8_t>(std::clamp(Y, 16, 235));
        }
    }

    uint8_t* uv = nv12 + h * stride;
    for (int y = 0; y < h / 2; y++) {
        for (int x = 0; x < w / 2; x++) {
            int R = 0, G = 0, B = 0;
            for (int dy = 0; dy < 2; dy++) {
                const uint8_t* row = rgb.ptr<uint8_t>(y * 2 + dy);
                for (int dx = 0; dx < 2; dx++) {
                    R += row[(x * 2 + dx) * 3 + 0];
                    G += row[(x * 2 + dx) * 3 + 1];
                    B += row[(x * 2 + dx) * 3 + 2];
                }
            }
            R /= 4; G /= 4; B /= 4;
            int U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128;
            int V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128;
            uv[y * stride + x * 2 + 0] = static_cast<uint8_t>(std::clamp(U, 16, 240));
            uv[y * stride + x * 2 + 1] = static_cast<uint8_t>(std::clamp(V, 16, 240));
        }
    }
}

// ============================================================
// 目录扫描: 查找所有 .jpg/.jpeg 文件
// ============================================================
static std::vector<std::string> scan_jpg(const std::string& dir) {
    std::vector<std::string> files;
    DIR* d = opendir(dir.c_str());
    if (!d) {
        fprintf(stderr, "[scan] Cannot open directory: %s\n", dir.c_str());
        return files;
    }
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name(ent->d_name);
        if (name.size() > 4) {
            std::string ext = name.substr(name.size() - 4);
            for (auto& c : ext) c = static_cast<char>(tolower(c));
            if (ext == ".jpg") {
                files.push_back(name);
            }
        }
        if (name.size() > 5) {
            std::string ext = name.substr(name.size() - 5);
            for (auto& c : ext) c = static_cast<char>(tolower(c));
            if (ext == ".jpeg") {
                files.push_back(name);
            }
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

// ============================================================
// Landmark 颜色 (BGR for OpenCV)
// ============================================================
static const cv::Scalar kLmkColors[5] = {
    {0, 0, 255},    // 左眼: 红
    {255, 0, 0},    // 右眼: 蓝
    {0, 255, 0},    // 鼻子: 绿
    {0, 255, 255},  // 左嘴角: 黄
    {255, 0, 255}   // 右嘴角: 品红
};

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_dir> [kernel_dir] [model_path] "
                        "[conf_thresh] [nms_thresh]\n", argv[0]);
        return 1;
    }

    const char* image_dir  = argv[1];
    const char* kernel_dir = (argc > 2) ? argv[2]
                             : "../../../face_preprocess/kernels";
    const char* model_path = (argc > 3) ? argv[3]
                             : "../../../../data/models/face/det_10g_qcs9075.bin";
    float conf_thresh = (argc > 4) ? std::atof(argv[4]) : 0.5f;
    float nms_thresh  = (argc > 5) ? std::atof(argv[5]) : 0.4f;

    fprintf(stderr, "=== SCRFD Draw Test ===\n");
    fprintf(stderr, "  Image dir:   %s\n", image_dir);
    fprintf(stderr, "  Kernel dir:  %s\n", kernel_dir);
    fprintf(stderr, "  Model path:  %s\n", model_path);
    fprintf(stderr, "  Conf thresh: %.2f\n", conf_thresh);
    fprintf(stderr, "  NMS thresh:  %.2f\n", nms_thresh);

    // ---- 扫描图片 ----
    auto images = scan_jpg(image_dir);
    if (images.empty()) {
        fprintf(stderr, "FATAL: No .jpg files found in %s\n", image_dir);
        return 1;
    }
    fprintf(stderr, "  Found %zu images\n\n", images.size());

    // ---- 创建输出目录 ----
    std::string out_dir = std::string(image_dir) + "/output";
    mkdir(out_dir.c_str(), 0755);
    fprintf(stderr, "  Output dir:  %s\n\n", out_dir.c_str());

    // ---- 初始化 OpenClLoader ----
    fprintf(stderr, "[Init] OpenClLoader...\n");
    g_ocl = OpenClLoader::Get();
    if (!g_ocl) {
        fprintf(stderr, "FATAL: OpenClLoader::Get() failed\n");
        return 1;
    }

    // ---- 初始化 FacePreprocess (OpenCL GPU) ----
    fprintf(stderr, "[Init] FacePreprocess...\n");
    if (!g_gpu.Init(kernel_dir, g_ocl)) {
        fprintf(stderr, "FATAL: FacePreprocess::Init failed\n");
        return 1;
    }

    // ---- 初始化 QnnInferencer (QNN HTP) ----
    fprintf(stderr, "[Init] QnnInferencer...\n");
    if (!g_qnn.Init("/usr/lib/libQnnHtp.so",
                     "/usr/lib/libQnnSystem.so",
                     model_path)) {
        fprintf(stderr, "FATAL: QnnInferencer::Init failed\n");
        g_gpu.Destroy();
        return 1;
    }
    fprintf(stderr, "[Init] Complete: %u output tensors\n\n",
            g_qnn.GetNumOutputTensors());

    // ---- 推理 + 绘图循环 ----
    int total_faces = 0;
    double total_ms = 0;
    int failed_images = 0;
    int saved_images = 0;

    for (size_t idx = 0; idx < images.size(); idx++) {
        const auto& fname = images[idx];
        std::string fpath = std::string(image_dir) + "/" + fname;

        // 1. 加载图片 (BGR)
        cv::Mat bgr = cv::imread(fpath, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            fprintf(stderr, "[%zu/%zu] %s: FAILED to load, skipping\n",
                    idx + 1, images.size(), fname.c_str());
            failed_images++;
            continue;
        }

        // 2. BGR -> RGB
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        // 3. 确保宽高为偶数
        int w = rgb.cols & ~1;
        int h = rgb.rows & ~1;
        if (w != rgb.cols || h != rgb.rows) {
            rgb = rgb(cv::Rect(0, 0, w, h)).clone();
            bgr = bgr(cv::Rect(0, 0, w, h)).clone();
        }
        int stride = w;

        // 4. 创建 DmaBuffer + NV12 转换
        NV12Frame frame;
        if (!frame.Create(static_cast<uint32_t>(w),
                          static_cast<uint32_t>(h),
                          static_cast<uint32_t>(stride),
                          g_gpu.GetContext(), g_ocl)) {
            fprintf(stderr, "[%zu/%zu] %s: DmaBuffer create failed (%dx%d), skipping\n",
                    idx + 1, images.size(), fname.c_str(), w, h);
            failed_images++;
            continue;
        }

        {
            DmaSyncGuard sync(frame.buf.fd());
            rgb_to_nv12(rgb, static_cast<uint8_t*>(frame.buf.data()),
                        w, h, stride);
        }

        // 5. GPU 预处理 + QNN 推理 + 解码
        auto t0 = std::chrono::high_resolution_clock::now();

        float scale = 0;
        int new_w = 0, new_h = 0;
        const float* preproc = g_gpu.RunScrfd(frame.buf.cl_mem_handle(),
                                               frame.plane,
                                               &scale, &new_w, &new_h);
        if (!preproc) {
            fprintf(stderr, "[%zu/%zu] %s: RunScrfd failed, skipping\n",
                    idx + 1, images.size(), fname.c_str());
            frame.Destroy();
            failed_images++;
            continue;
        }

        size_t input_bytes = FacePreprocess::kScrfdOutputFloats * sizeof(float);
        if (!g_qnn.Execute(preproc, input_bytes)) {
            fprintf(stderr, "[%zu/%zu] %s: QNN Execute failed, skipping\n",
                    idx + 1, images.size(), fname.c_str());
            frame.Destroy();
            failed_images++;
            continue;
        }

        const float* tensors[9];
        for (uint32_t i = 0; i < 9; i++) {
            tensors[i] = g_qnn.GetOutputData(i);
        }

        FaceDetection dets[kFaceMaxDetections];
        int count = ScrfdDecode(tensors, scale, w, h,
                                conf_thresh, nms_thresh,
                                dets, kFaceMaxDetections);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;

        // 6. 在 BGR 原图上绘制检测结果
        for (int i = 0; i < count; i++) {
            const auto& d = dets[i];

            // bbox: 绿色矩形
            cv::rectangle(bgr,
                          cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                          cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)),
                          cv::Scalar(0, 255, 0), 2);

            // score: 白色文字
            char score_buf[32];
            snprintf(score_buf, sizeof(score_buf), "%.2f", d.score);
            cv::putText(bgr, score_buf,
                        cv::Point(static_cast<int>(d.x1),
                                  std::max(static_cast<int>(d.y1) - 5, 12)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 255, 255), 1);

            // 5 landmarks: 彩色圆点
            for (int j = 0; j < 5; j++) {
                cv::circle(bgr,
                           cv::Point(static_cast<int>(d.landmarks[j][0]),
                                     static_cast<int>(d.landmarks[j][1])),
                           3, kLmkColors[j], -1);
            }
        }

        // 7. 保存标注图片
        std::string out_path = out_dir + "/" + fname;
        if (cv::imwrite(out_path, bgr)) {
            saved_images++;
        } else {
            fprintf(stderr, "[%zu/%zu] %s: imwrite failed!\n",
                    idx + 1, images.size(), fname.c_str());
        }

        total_faces += count;

        // 8. 进度日志
        fprintf(stderr, "[%zu/%zu] %s: %d face(s), %.1fms, scale=%.2f (%dx%d->%dx%d) -> %s\n",
                idx + 1, images.size(), fname.c_str(),
                count, ms, scale, w, h, new_w, new_h, out_path.c_str());

        frame.Destroy();
    }

    // ---- 清理 ----
    g_qnn.Destroy();
    g_gpu.Destroy();

    // ---- 统计摘要 ----
    int processed = static_cast<int>(images.size()) - failed_images;
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "  Total images:   %zu\n", images.size());
    fprintf(stderr, "  Processed:      %d\n", processed);
    fprintf(stderr, "  Failed:         %d\n", failed_images);
    fprintf(stderr, "  Total faces:    %d\n", total_faces);
    fprintf(stderr, "  Saved images:   %d\n", saved_images);
    if (processed > 0) {
        fprintf(stderr, "  Avg faces/img:  %.1f\n",
                static_cast<double>(total_faces) / processed);
        fprintf(stderr, "  Avg time/img:   %.1f ms\n",
                total_ms / processed);
        fprintf(stderr, "  Total time:     %.0f ms\n", total_ms);
    }
    fprintf(stderr, "  Output dir:     %s\n", out_dir.c_str());

    return 0;
}