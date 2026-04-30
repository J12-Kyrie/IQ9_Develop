/// @file test_gallery_consistency.cpp
/// @brief Gallery 一致性测试: 静态图片 → FaceProcessor 全链路 → embedding 注册/验证
///
/// 验证目标: 同一张图片在两次独立运行中产生的 embedding 是否能通过余弦相似度正确匹配
///
/// 用法:
///   Phase 1 (注册):
///     ./test_gallery_consistency <face_config.json> --enroll <pic_dir> <gallery.json>
///
///   Phase 2 (验证):
///     ./test_gallery_consistency <face_config.json> --verify <pic_dir> <gallery.json>
///
///   可选: 输出 N*N 交叉相似度矩阵:
///     ./test_gallery_consistency <face_config.json> --verify <pic_dir> <gallery.json> --matrix
///
/// 编译: bash build_test_gallery_consistency.sh
/// 一键运行: bash run_gallery_test.sh [face_config.json]

#include "../FaceProcessor.hpp"
#include "../config/FaceConfigLoader.hpp"
#include "../mem_management/dma_buffer.hpp"
#include "../mem_management/opencl_loader.hpp"
#include "../mem_management/dma_sync_guard.hpp"
#include "../mem_management/mem_types.hpp"

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
#include <fstream>
#include <string>
#include <vector>

using namespace face_infer;

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
// 匹配 scrfd_preprocess.cl 中的 BT.601 逆转换系数
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
// 目录扫描: 查找所有 .jpg/.jpeg 文件 (排序)
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
        auto to_lower = [](std::string s) {
            for (auto& c : s) c = static_cast<char>(tolower(c));
            return s;
        };
        if (name.size() > 4) {
            std::string ext = to_lower(name.substr(name.size() - 4));
            if (ext == ".jpg") { files.push_back(name); continue; }
        }
        if (name.size() > 5) {
            std::string ext = to_lower(name.substr(name.size() - 5));
            if (ext == ".jpeg") { files.push_back(name); }
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

// ============================================================
// Gallery 数据结构
// ============================================================
struct GalleryEntry {
    std::string filename;
    float embedding[kFaceEmbeddingDim];
};

// ============================================================
// Gallery JSON 写入 (不依赖 json-glib)
// ============================================================
static bool save_gallery(const std::string& path,
                         const std::vector<GalleryEntry>& entries) {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "[gallery] Cannot open for writing: %s\n", path.c_str());
        return false;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"count\": %zu,\n", entries.size());
    fprintf(fp, "  \"entries\": [\n");

    for (size_t i = 0; i < entries.size(); i++) {
        const auto& e = entries[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"filename\": \"%s\",\n", e.filename.c_str());
        fprintf(fp, "      \"embedding\": [");
        for (int j = 0; j < kFaceEmbeddingDim; j++) {
            if (j > 0) fprintf(fp, ",");
            fprintf(fp, "%.8f", e.embedding[j]);
        }
        fprintf(fp, "]\n");
        fprintf(fp, "    }%s\n", (i + 1 < entries.size()) ? "," : "");
    }

    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    fclose(fp);
    return true;
}

// ============================================================
// Gallery JSON 读取 (逐字符解析, 不依赖 json-glib)
// ============================================================
static bool load_gallery(const std::string& path,
                         std::vector<GalleryEntry>& entries) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "[gallery] Cannot open for reading: %s\n", path.c_str());
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    ifs.close();
    entries.clear();

    // 查找每个 entry 对象
    size_t pos = 0;
    while (true) {
        // 查找 "filename"
        size_t fn_key = content.find("\"filename\"", pos);
        if (fn_key == std::string::npos) break;

        // 提取 filename 值
        size_t q1 = content.find('"', content.find(':', fn_key) + 1);
        size_t q2 = content.find('"', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos) break;
        std::string filename = content.substr(q1 + 1, q2 - q1 - 1);

        // 查找 "embedding" 数组
        size_t emb_key = content.find("\"embedding\"", q2);
        if (emb_key == std::string::npos) break;
        size_t arr_start = content.find('[', emb_key);
        size_t arr_end   = content.find(']', arr_start);
        if (arr_start == std::string::npos || arr_end == std::string::npos) break;

        // 解析 embedding 浮点数组
        GalleryEntry entry;
        entry.filename = filename;
        memset(entry.embedding, 0, sizeof(entry.embedding));

        std::string arr_str = content.substr(arr_start + 1, arr_end - arr_start - 1);
        int idx = 0;
        const char* p = arr_str.c_str();
        while (idx < kFaceEmbeddingDim) {
            while (*p && (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t'))
                p++;
            if (!*p) break;
            char* end = nullptr;
            float val = strtof(p, &end);
            if (end == p) break;
            entry.embedding[idx++] = val;
            p = end;
        }

        if (idx == kFaceEmbeddingDim) {
            entries.push_back(entry);
        } else {
            fprintf(stderr, "[gallery] Warning: %s has %d dims (expected %d), skipping\n",
                    filename.c_str(), idx, kFaceEmbeddingDim);
        }

        pos = arr_end + 1;
    }

    fprintf(stderr, "[gallery] Loaded %zu entries from %s\n",
            entries.size(), path.c_str());
    return !entries.empty();
}

// ============================================================
// 单张图片推理: 加载图片 → NV12 → FaceProcessor → embedding
// ============================================================
static bool process_image(const std::string& image_path,
                          FaceProcessor& fp,
                          std::shared_ptr<OpenClLoader> ocl,
                          std::vector<FaceResult>& results) {
    results.clear();

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        fprintf(stderr, "  [WARN] Cannot load image: %s\n", image_path.c_str());
        return false;
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // 确保宽高为偶数 (NV12 2x2 chroma subsampling)
    int w = rgb.cols & ~1;
    int h = rgb.rows & ~1;
    if (w != rgb.cols || h != rgb.rows) {
        rgb = rgb(cv::Rect(0, 0, w, h)).clone();
    }
    int stride = w;

    NV12Frame frame;
    if (!frame.Create(static_cast<uint32_t>(w),
                      static_cast<uint32_t>(h),
                      static_cast<uint32_t>(stride),
                      fp.GetContext(), ocl)) {
        fprintf(stderr, "  [WARN] DmaBuffer create failed for %s (%dx%d)\n",
                image_path.c_str(), w, h);
        return false;
    }

    {
        DmaSyncGuard sync(frame.buf.fd());
        rgb_to_nv12(rgb, static_cast<uint8_t*>(frame.buf.data()), w, h, stride);
    }

    results = fp.ProcessFrame(frame.buf.cl_mem_handle(), frame.plane);
    frame.Destroy();
    return true;
}

// ============================================================
// Phase 1: Enroll — 对每张图片提取 embedding, 保存 gallery
// ============================================================
static int do_enroll(const FaceProcessorConfig& config,
                     const std::string& pic_dir,
                     const std::string& gallery_path) {
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  Phase 1: ENROLL\n");
    fprintf(stderr, "  pic_dir:      %s\n", pic_dir.c_str());
    fprintf(stderr, "  gallery_path: %s\n", gallery_path.c_str());
    fprintf(stderr, "========================================\n\n");

    auto images = scan_jpg(pic_dir);
    if (images.empty()) {
        fprintf(stderr, "FATAL: No .jpg files found in %s\n", pic_dir.c_str());
        return 1;
    }
    fprintf(stderr, "Found %zu images\n\n", images.size());

    // 初始化 FaceProcessor
    auto ocl = OpenClLoader::Get();
    if (!ocl) { fprintf(stderr, "FATAL: OpenClLoader::Get() failed\n"); return 1; }

    FaceProcessor fp;
    if (!fp.Init(config)) {
        fprintf(stderr, "FATAL: FaceProcessor::Init failed\n");
        return 1;
    }
    if (!fp.HasArcFace()) {
        fprintf(stderr, "FATAL: ArcFace not enabled (arcface_model required for gallery test)\n");
        fp.Destroy();
        return 1;
    }

    std::vector<GalleryEntry> gallery;
    int enrolled = 0, no_face = 0, failed = 0;

    for (size_t i = 0; i < images.size(); i++) {
        const auto& fname = images[i];
        std::string fpath = pic_dir + "/" + fname;

        std::vector<FaceResult> results;
        if (!process_image(fpath, fp, ocl, results)) {
            fprintf(stderr, "[%zu/%zu] %s: FAILED to process\n",
                    i + 1, images.size(), fname.c_str());
            failed++;
            continue;
        }

        if (results.empty()) {
            fprintf(stderr, "[%zu/%zu] %s: no face detected\n",
                    i + 1, images.size(), fname.c_str());
            no_face++;
            continue;
        }

        // 取第一个人脸 (最高 score)
        int best_idx = 0;
        for (size_t j = 1; j < results.size(); j++) {
            if (results[j].det.score > results[best_idx].det.score)
                best_idx = static_cast<int>(j);
        }

        GalleryEntry entry;
        entry.filename = fname;
        memcpy(entry.embedding, results[best_idx].embedding,
               sizeof(float) * kFaceEmbeddingDim);
        gallery.push_back(entry);
        enrolled++;

        // 检查 embedding 有效性
        float norm_sq = 0;
        for (int j = 0; j < kFaceEmbeddingDim; j++)
            norm_sq += entry.embedding[j] * entry.embedding[j];

        fprintf(stderr, "[%zu/%zu] %s: %zu face(s), score=%.4f, "
                "bbox=[%.0f,%.0f,%.0f,%.0f], ||emb||^2=%.6f\n",
                i + 1, images.size(), fname.c_str(),
                results.size(), results[best_idx].det.score,
                results[best_idx].det.x1, results[best_idx].det.y1,
                results[best_idx].det.x2, results[best_idx].det.y2,
                norm_sq);
    }

    fp.Destroy();

    // 保存 gallery
    if (!save_gallery(gallery_path, gallery)) {
        fprintf(stderr, "FATAL: Failed to save gallery\n");
        return 1;
    }

    fprintf(stderr, "\n=== Enroll Summary ===\n");
    fprintf(stderr, "  Total images:  %zu\n", images.size());
    fprintf(stderr, "  Enrolled:      %d\n", enrolled);
    fprintf(stderr, "  No face:       %d\n", no_face);
    fprintf(stderr, "  Failed:        %d\n", failed);
    fprintf(stderr, "  Gallery saved: %s\n", gallery_path.c_str());

    return 0;
}

// ============================================================
// Phase 2: Verify — 重新推理, 对比 gallery, 输出统计表
// ============================================================
static int do_verify(const FaceProcessorConfig& config,
                     const std::string& pic_dir,
                     const std::string& gallery_path,
                     bool print_matrix) {
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  Phase 2: VERIFY\n");
    fprintf(stderr, "  pic_dir:      %s\n", pic_dir.c_str());
    fprintf(stderr, "  gallery_path: %s\n", gallery_path.c_str());
    fprintf(stderr, "========================================\n\n");

    // 加载 gallery
    std::vector<GalleryEntry> gallery;
    if (!load_gallery(gallery_path, gallery)) {
        fprintf(stderr, "FATAL: Failed to load gallery from %s\n", gallery_path.c_str());
        return 1;
    }

    auto images = scan_jpg(pic_dir);
    if (images.empty()) {
        fprintf(stderr, "FATAL: No .jpg files found in %s\n", pic_dir.c_str());
        return 1;
    }
    fprintf(stderr, "Found %zu images, gallery has %zu entries\n\n",
            images.size(), gallery.size());

    // 初始化 FaceProcessor (新实例, 独立于 enroll)
    auto ocl = OpenClLoader::Get();
    if (!ocl) { fprintf(stderr, "FATAL: OpenClLoader::Get() failed\n"); return 1; }

    FaceProcessor fp;
    if (!fp.Init(config)) {
        fprintf(stderr, "FATAL: FaceProcessor::Init failed\n");
        return 1;
    }
    if (!fp.HasArcFace()) {
        fprintf(stderr, "FATAL: ArcFace not enabled\n");
        fp.Destroy();
        return 1;
    }

    // 收集本次推理的 embedding (用于可选的交叉矩阵)
    struct VerifyResult {
        std::string filename;
        bool has_face;
        float embedding[kFaceEmbeddingDim];
        std::string best_match;
        float best_sim;
        float self_sim;     // 与 gallery 中同名条目的相似度
        bool self_is_best;  // best_match == filename
        std::string status; // PASS / WARN / FAIL / NO_FACE / NOT_IN_GALLERY
    };

    std::vector<VerifyResult> verify_results;
    int pass = 0, warn = 0, fail = 0, no_face = 0, not_in_gallery = 0;

    for (size_t i = 0; i < images.size(); i++) {
        const auto& fname = images[i];
        std::string fpath = pic_dir + "/" + fname;

        VerifyResult vr;
        vr.filename = fname;
        vr.has_face = false;
        vr.best_sim = -1.0f;
        vr.self_sim = -1.0f;
        vr.self_is_best = false;
        memset(vr.embedding, 0, sizeof(vr.embedding));

        std::vector<FaceResult> results;
        if (!process_image(fpath, fp, ocl, results) || results.empty()) {
            vr.status = "NO_FACE";
            no_face++;
            verify_results.push_back(vr);
            continue;
        }

        vr.has_face = true;

        // 取最高 score 的人脸
        int best_idx = 0;
        for (size_t j = 1; j < results.size(); j++) {
            if (results[j].det.score > results[best_idx].det.score)
                best_idx = static_cast<int>(j);
        }
        memcpy(vr.embedding, results[best_idx].embedding,
               sizeof(float) * kFaceEmbeddingDim);

        // 与 gallery 中所有条目计算余弦相似度
        vr.best_sim = -2.0f;
        for (size_t g = 0; g < gallery.size(); g++) {
            float sim = FaceProcessor::Similarity(vr.embedding, gallery[g].embedding);
            if (sim > vr.best_sim) {
                vr.best_sim = sim;
                vr.best_match = gallery[g].filename;
            }
            if (gallery[g].filename == fname) {
                vr.self_sim = sim;
            }
        }

        vr.self_is_best = (vr.best_match == fname);

        // 判定状态
        if (vr.self_sim < 0) {
            // 当前图片不在 gallery 中 (enroll 时可能未检测到人脸)
            vr.status = "NOT_IN_GALLERY";
            not_in_gallery++;
        } else if (vr.self_is_best && vr.best_sim > 0.90f) {
            vr.status = "PASS";
            pass++;
        } else if (vr.self_is_best && vr.best_sim > 0.70f) {
            vr.status = "WARN";
            warn++;
        } else {
            vr.status = "FAIL";
            fail++;
        }

        verify_results.push_back(vr);
    }

    fp.Destroy();

    // ---- 输出统计表 ----
    fprintf(stderr, "\n");
    fprintf(stderr, "==========================================================================\n");
    fprintf(stderr, "  VERIFICATION RESULTS\n");
    fprintf(stderr, "==========================================================================\n");
    fprintf(stderr, "%-16s %-10s %-16s %-10s %-10s %-8s %s\n",
            "IMAGE", "DETECTED", "GALLERY_MATCH", "BEST_SIM", "SELF_SIM",
            "SELF_IS", "STATUS");
    fprintf(stderr, "%-16s %-10s %-16s %-10s %-10s %-8s %s\n",
            "-----", "--------", "-------------", "--------", "--------",
            "------", "------");

    for (const auto& vr : verify_results) {
        if (!vr.has_face) {
            fprintf(stderr, "%-16s %-10s %-16s %-10s %-10s %-8s %s\n",
                    vr.filename.c_str(), "0 faces", "-", "-", "-", "-",
                    vr.status.c_str());
        } else {
            char best_sim_str[16], self_sim_str[16];
            snprintf(best_sim_str, sizeof(best_sim_str), "%.6f", vr.best_sim);
            snprintf(self_sim_str, sizeof(self_sim_str),
                     vr.self_sim >= 0 ? "%.6f" : "-", vr.self_sim);

            fprintf(stderr, "%-16s %-10s %-16s %-10s %-10s %-8s %s\n",
                    vr.filename.c_str(),
                    "YES",
                    vr.best_match.c_str(),
                    best_sim_str,
                    self_sim_str,
                    vr.self_is_best ? "YES" : "NO",
                    vr.status.c_str());
        }
    }

    // ---- 可选: 交叉相似度矩阵 ----
    if (print_matrix) {
        // 收集有 embedding 的验证结果
        std::vector<const VerifyResult*> with_face;
        for (const auto& vr : verify_results) {
            if (vr.has_face) with_face.push_back(&vr);
        }

        if (with_face.size() >= 2) {
            fprintf(stderr, "\n=== Cross-Similarity Matrix ===\n");
            fprintf(stderr, "%16s", "");
            for (const auto* vr : with_face)
                fprintf(stderr, " %10.10s", vr->filename.c_str());
            fprintf(stderr, "\n");

            for (size_t r = 0; r < with_face.size(); r++) {
                fprintf(stderr, "%16.16s", with_face[r]->filename.c_str());
                for (size_t c = 0; c < with_face.size(); c++) {
                    float sim = FaceProcessor::Similarity(
                        with_face[r]->embedding, with_face[c]->embedding);
                    fprintf(stderr, " %10.6f", sim);
                }
                fprintf(stderr, "\n");
            }
        }
    }

    // ---- 汇总 ----
    fprintf(stderr, "\n=== Verify Summary ===\n");
    fprintf(stderr, "  Total images:     %zu\n", images.size());
    fprintf(stderr, "  Gallery entries:  %zu\n", gallery.size());
    fprintf(stderr, "  PASS:             %d  (self_match=YES, best_sim > 0.90)\n", pass);
    fprintf(stderr, "  WARN:             %d  (self_match=YES, best_sim 0.70~0.90)\n", warn);
    fprintf(stderr, "  FAIL:             %d  (self_match=NO or self_sim < 0.70)\n", fail);
    fprintf(stderr, "  NO_FACE:          %d\n", no_face);
    fprintf(stderr, "  NOT_IN_GALLERY:   %d\n", not_in_gallery);

    // 统计 self_sim 分布 (仅 PASS/WARN/FAIL)
    float sim_min = 2.0f, sim_max = -2.0f, sim_sum = 0;
    int sim_count = 0;
    for (const auto& vr : verify_results) {
        if (vr.has_face && vr.self_sim >= 0) {
            if (vr.self_sim < sim_min) sim_min = vr.self_sim;
            if (vr.self_sim > sim_max) sim_max = vr.self_sim;
            sim_sum += vr.self_sim;
            sim_count++;
        }
    }
    if (sim_count > 0) {
        fprintf(stderr, "\n  Self-similarity stats (N=%d):\n", sim_count);
        fprintf(stderr, "    min=%.6f  max=%.6f  avg=%.6f\n",
                sim_min, sim_max, sim_sum / sim_count);
    }

    fprintf(stderr, "\n==========================================================================\n");
    if (fail == 0 && warn == 0) {
        fprintf(stderr, "  ALL MATCHED FACES PASS\n");
    } else if (fail == 0) {
        fprintf(stderr, "  ALL MATCHED, %d WARNINGS (low similarity)\n", warn);
    } else {
        fprintf(stderr, "  %d FAILURES DETECTED\n", fail);
    }
    fprintf(stderr, "==========================================================================\n");

    return (fail > 0) ? 1 : 0;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    fprintf(stderr, "=========================================\n");
    fprintf(stderr, "  Gallery Consistency Test\n");
    fprintf(stderr, "=========================================\n");

    if (argc < 5) {
        fprintf(stderr, "\nUsage:\n");
        fprintf(stderr, "  %s <face_config.json> --enroll <pic_dir> <gallery.json>\n", argv[0]);
        fprintf(stderr, "  %s <face_config.json> --verify <pic_dir> <gallery.json> [--matrix]\n", argv[0]);
        return 1;
    }

    const char* config_path = argv[1];
    const char* mode        = argv[2];
    const char* pic_dir     = argv[3];
    const char* gallery_path = argv[4];

    bool print_matrix = false;
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--matrix") == 0) print_matrix = true;
    }

    // 加载配置
    FaceProcessorConfig config;
    if (!LoadFaceConfig(config_path, config)) {
        fprintf(stderr, "FATAL: Failed to load config: %s\n", config_path);
        return 1;
    }

    if (strcmp(mode, "--enroll") == 0) {
        return do_enroll(config, pic_dir, gallery_path);
    } else if (strcmp(mode, "--verify") == 0) {
        return do_verify(config, pic_dir, gallery_path, print_matrix);
    } else {
        fprintf(stderr, "Unknown mode: %s (use --enroll or --verify)\n", mode);
        return 1;
    }
}
