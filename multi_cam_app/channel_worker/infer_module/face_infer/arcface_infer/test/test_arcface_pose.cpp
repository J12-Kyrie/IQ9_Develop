/// @file test_arcface_pose.cpp
/// @brief SCRFD + ArcFace 完整流水线: 已裁剪人脸 → padding → SCRFD 检测 → 对齐 → ArcFace 推理
///
/// 流水线:
///   imread → pad 640×640 → SCRFD CPU 预处理 → SCRFD QNN → ScrfdDecode
///   → ComputeInverseAffine (Umeyama) → CPU 仿射裁剪 112×112
///   → ArcFace 归一化 → ArcFace QNN → 512-dim embedding
///
/// 用法:
///   Enroll (注册 baseline):
///     ./test_arcface_pose <scrfd_model> <arcface_model> --enroll <image_path> <gallery.json>
///
///   Verify (姿态对比):
///     ./test_arcface_pose <scrfd_model> <arcface_model> --verify <pic_dir> <gallery.json> [--matrix]
///
/// 编译: bash build_test_arcface_pose.sh
/// 一键运行: bash run_arcface_pose_test.sh

#include "../ArcFaceInfer.hpp"

// SCRFD
#include "../../scrfd_infer/FaceTypes.hpp"
#include "../../scrfd_infer/QnnInferencer.hpp"

// Umeyama inverse affine
#include "../../face_preprocess/inverse_affine.hpp"

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

using namespace arcface_infer;

static constexpr int kScrfdSize     = 640;
static constexpr int kArcFaceSize   = 112;
static constexpr int kEmbDim        = 512;
static constexpr int kScrfdInputFloats  = 3 * kScrfdSize * kScrfdSize;   // 1,228,800
static constexpr int kArcFaceInputFloats = 3 * kArcFaceSize * kArcFaceSize; // 37,632

// ============================================================
// Pad image to 640×640 with gray (128,128,128) background, centered
// ============================================================
static cv::Mat pad_to_640(const cv::Mat& bgr) {
    if (bgr.cols == kScrfdSize && bgr.rows == kScrfdSize)
        return bgr.clone();

    cv::Mat padded(kScrfdSize, kScrfdSize, CV_8UC3, cv::Scalar(128, 128, 128));
    int off_x = (kScrfdSize - bgr.cols) / 2;
    int off_y = (kScrfdSize - bgr.rows) / 2;

    // clamp source region if image is larger than 640
    int src_w = std::min(bgr.cols, kScrfdSize);
    int src_h = std::min(bgr.rows, kScrfdSize);
    off_x = std::max(off_x, 0);
    off_y = std::max(off_y, 0);

    cv::Rect dst_roi(off_x, off_y, src_w, src_h);
    cv::Rect src_roi(0, 0, src_w, src_h);
    bgr(src_roi).copyTo(padded(dst_roi));

    return padded;
}

// ============================================================
// SCRFD CPU preprocess: BGR 640×640 → float32 NCHW RGB (pixel-127.5)/128.0
// ============================================================
static void preprocess_for_scrfd(const cv::Mat& bgr_640, float* output) {
    cv::Mat rgb;
    cv::cvtColor(bgr_640, rgb, cv::COLOR_BGR2RGB);

    const int plane_size = kScrfdSize * kScrfdSize;
    for (int y = 0; y < kScrfdSize; y++) {
        const uint8_t* row = rgb.ptr<uint8_t>(y);
        for (int x = 0; x < kScrfdSize; x++) {
            int idx = y * kScrfdSize + x;
            output[0 * plane_size + idx] = (row[x * 3 + 0] - 127.5f) / 128.0f;  // R
            output[1 * plane_size + idx] = (row[x * 3 + 1] - 127.5f) / 128.0f;  // G
            output[2 * plane_size + idx] = (row[x * 3 + 2] - 127.5f) / 128.0f;  // B
        }
    }
}

// ============================================================
// CPU affine warp: inverse affine from 112×112 → source BGR image
// Output: float32 NCHW RGB, pixel/127.5-1.0
// Replicates arcface_preprocess.cl logic on CPU
// ============================================================
static void cpu_affine_warp(const cv::Mat& bgr_src, const float inv[6], float* output) {
    const int src_w = bgr_src.cols;
    const int src_h = bgr_src.rows;
    const int plane_size = kArcFaceSize * kArcFaceSize;

    for (int dy = 0; dy < kArcFaceSize; dy++) {
        for (int dx = 0; dx < kArcFaceSize; dx++) {
            int idx = dy * kArcFaceSize + dx;

            // inverse affine: dst → src
            float sx = inv[0] * dx + inv[1] * dy + inv[2];
            float sy = inv[3] * dx + inv[4] * dy + inv[5];

            // out-of-bounds: fill -1.0 (black in ArcFace normalization)
            if (sx < 0.0f || sx >= (float)(src_w - 1) ||
                sy < 0.0f || sy >= (float)(src_h - 1)) {
                output[0 * plane_size + idx] = -1.0f;
                output[1 * plane_size + idx] = -1.0f;
                output[2 * plane_size + idx] = -1.0f;
                continue;
            }

            // bilinear interpolation
            int x0 = (int)std::floor(sx);
            int y0 = (int)std::floor(sy);
            int x1 = std::min(x0 + 1, src_w - 1);
            int y1 = std::min(y0 + 1, src_h - 1);
            float fx = sx - (float)x0;
            float fy = sy - (float)y0;

            const uint8_t* p00 = bgr_src.ptr<uint8_t>(y0) + x0 * 3;
            const uint8_t* p01 = bgr_src.ptr<uint8_t>(y0) + x1 * 3;
            const uint8_t* p10 = bgr_src.ptr<uint8_t>(y1) + x0 * 3;
            const uint8_t* p11 = bgr_src.ptr<uint8_t>(y1) + x1 * 3;

            // interpolate each BGR channel, then convert to RGB + ArcFace normalize
            for (int c = 0; c < 3; c++) {
                float v = (1.0f - fy) * ((1.0f - fx) * p00[c] + fx * p01[c])
                        + fy * ((1.0f - fx) * p10[c] + fx * p11[c]);
                // BGR→RGB channel mapping: OpenCV BGR c=0→B, c=1→G, c=2→R
                // NCHW output: plane 0=R, plane 1=G, plane 2=B
                int out_c = 2 - c;  // B(0)→2, G(1)→1, R(2)→0
                output[out_c * plane_size + idx] = v / 127.5f - 1.0f;
            }
        }
    }
}

// ============================================================
// Directory scan: .jpg / .jpeg / .png (sorted)
// ============================================================
static std::vector<std::string> scan_images(const std::string& dir) {
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
        std::string lower = to_lower(name);
        if (lower.size() > 4) {
            std::string ext4 = lower.substr(lower.size() - 4);
            if (ext4 == ".jpg" || ext4 == ".png") {
                files.push_back(name);
                continue;
            }
        }
        if (lower.size() > 5) {
            std::string ext5 = lower.substr(lower.size() - 5);
            if (ext5 == ".jpeg") {
                files.push_back(name);
            }
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

// ============================================================
// Gallery data structure + JSON read/write
// ============================================================
struct GalleryEntry {
    std::string filename;
    float embedding[kEmbDim];
};

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
        for (int j = 0; j < kEmbDim; j++) {
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

    size_t pos = 0;
    while (true) {
        size_t fn_key = content.find("\"filename\"", pos);
        if (fn_key == std::string::npos) break;

        size_t q1 = content.find('"', content.find(':', fn_key) + 1);
        size_t q2 = content.find('"', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos) break;
        std::string filename = content.substr(q1 + 1, q2 - q1 - 1);

        size_t emb_key = content.find("\"embedding\"", q2);
        if (emb_key == std::string::npos) break;
        size_t arr_start = content.find('[', emb_key);
        size_t arr_end   = content.find(']', arr_start);
        if (arr_start == std::string::npos || arr_end == std::string::npos) break;

        GalleryEntry entry;
        entry.filename = filename;
        memset(entry.embedding, 0, sizeof(entry.embedding));

        std::string arr_str = content.substr(arr_start + 1, arr_end - arr_start - 1);
        int idx = 0;
        const char* p = arr_str.c_str();
        while (idx < kEmbDim) {
            while (*p && (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t'))
                p++;
            if (!*p) break;
            char* end = nullptr;
            float val = strtof(p, &end);
            if (end == p) break;
            entry.embedding[idx++] = val;
            p = end;
        }

        if (idx == kEmbDim) {
            entries.push_back(entry);
        } else {
            fprintf(stderr, "[gallery] Warning: %s has %d dims (expected %d), skipping\n",
                    filename.c_str(), idx, kEmbDim);
        }

        pos = arr_end + 1;
    }

    fprintf(stderr, "[gallery] Loaded %zu entries from %s\n",
            entries.size(), path.c_str());
    return !entries.empty();
}

// ============================================================
// Full pipeline: image → SCRFD → align → ArcFace → embedding
// ============================================================
static bool process_image(const std::string& image_path,
                          face_infer::QnnInferencer& scrfd_qnn,
                          ArcFaceInfer& arcface,
                          float out_embedding[kEmbDim]) {
    // Step 1: Load image
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        fprintf(stderr, "  [WARN] Cannot load image: %s\n", image_path.c_str());
        return false;
    }
    fprintf(stderr, "  [info] Loaded %dx%d: %s\n", bgr.cols, bgr.rows, image_path.c_str());

    // Step 2: Pad to 640×640
    cv::Mat bgr_640 = pad_to_640(bgr);

    // Step 3: SCRFD preprocess (CPU)
    std::vector<float> scrfd_input(kScrfdInputFloats);
    preprocess_for_scrfd(bgr_640, scrfd_input.data());

    // Step 4: SCRFD QNN inference
    if (!scrfd_qnn.Execute(scrfd_input.data(),
                           kScrfdInputFloats * sizeof(float))) {
        fprintf(stderr, "  [WARN] SCRFD Execute failed for: %s\n", image_path.c_str());
        return false;
    }

    // Step 5: Collect SCRFD output tensors
    uint32_t n_out = scrfd_qnn.GetNumOutputTensors();
    if (n_out < 9) {
        fprintf(stderr, "  [WARN] SCRFD has %u outputs (expected 9)\n", n_out);
        return false;
    }
    const float* output_tensors[9];
    for (int i = 0; i < 9; i++)
        output_tensors[i] = scrfd_qnn.GetOutputData(i);

    // Step 6: Decode (scale=1.0 since input is already 640×640)
    face_infer::FaceDetection dets[32];
    int n_dets = face_infer::ScrfdDecode(
        output_tensors, 1.0f, kScrfdSize, kScrfdSize,
        0.5f, 0.4f, dets, 32);

    if (n_dets == 0) {
        fprintf(stderr, "  [WARN] SCRFD detected 0 faces in: %s\n", image_path.c_str());
        return false;
    }

    // Use highest-score detection
    int best_idx = 0;
    for (int i = 1; i < n_dets; i++) {
        if (dets[i].score > dets[best_idx].score) best_idx = i;
    }
    const auto& det = dets[best_idx];

    fprintf(stderr, "  [info] SCRFD: %d face(s), best score=%.4f bbox=[%.0f,%.0f,%.0f,%.0f]\n",
            n_dets, det.score, det.x1, det.y1, det.x2, det.y2);
    fprintf(stderr, "  [info] landmarks: ");
    for (int i = 0; i < 5; i++)
        fprintf(stderr, "(%.1f,%.1f) ", det.landmarks[i][0], det.landmarks[i][1]);
    fprintf(stderr, "\n");

    // Step 7: Compute inverse affine (Umeyama)
    float inv[6];
    face_infer::ComputeInverseAffine(det.landmarks, inv);

    // Step 8: CPU affine warp → ArcFace input
    std::vector<float> arcface_input(kArcFaceInputFloats);
    cpu_affine_warp(bgr_640, inv, arcface_input.data());

    // Step 9: ArcFace inference
    if (!arcface.Infer(arcface_input.data(), out_embedding)) {
        fprintf(stderr, "  [WARN] ArcFace Infer failed for: %s\n", image_path.c_str());
        return false;
    }

    return true;
}

// ============================================================
// Enroll: single baseline image → gallery (1 entry)
// ============================================================
static int do_enroll(const std::string& scrfd_model,
                     const std::string& arcface_model,
                     const std::string& image_path,
                     const std::string& gallery_path) {
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  ENROLL (SCRFD + ArcFace aligned)\n");
    fprintf(stderr, "  scrfd:   %s\n", scrfd_model.c_str());
    fprintf(stderr, "  arcface: %s\n", arcface_model.c_str());
    fprintf(stderr, "  image:   %s\n", image_path.c_str());
    fprintf(stderr, "  gallery: %s\n", gallery_path.c_str());
    fprintf(stderr, "========================================\n\n");

    // Init SCRFD
    face_infer::QnnInferencer scrfd_qnn;
    if (!scrfd_qnn.Init("/usr/lib/libQnnHtp.so", "/usr/lib/libQnnSystem.so", scrfd_model)) {
        fprintf(stderr, "FATAL: SCRFD QnnInferencer Init failed\n");
        return 1;
    }

    // Init ArcFace
    ArcFaceInfer arcface;
    if (!arcface.Init("/usr/lib/libQnnHtp.so", "/usr/lib/libQnnSystem.so", arcface_model)) {
        fprintf(stderr, "FATAL: ArcFaceInfer Init failed\n");
        scrfd_qnn.Destroy();
        return 1;
    }

    float embedding[kEmbDim];
    if (!process_image(image_path, scrfd_qnn, arcface, embedding)) {
        fprintf(stderr, "FATAL: Failed to process baseline image\n");
        arcface.Destroy();
        scrfd_qnn.Destroy();
        return 1;
    }

    // Validate embedding
    float norm_sq = 0;
    for (int i = 0; i < kEmbDim; i++)
        norm_sq += embedding[i] * embedding[i];
    fprintf(stderr, "  ||embedding||^2 = %.6f\n", norm_sq);

    // Extract filename
    std::string filename = image_path;
    size_t slash = filename.find_last_of("/\\");
    if (slash != std::string::npos) filename = filename.substr(slash + 1);

    GalleryEntry entry;
    entry.filename = filename;
    memcpy(entry.embedding, embedding, sizeof(float) * kEmbDim);

    std::vector<GalleryEntry> gallery;
    gallery.push_back(entry);

    if (!save_gallery(gallery_path, gallery)) {
        fprintf(stderr, "FATAL: Failed to save gallery\n");
        arcface.Destroy();
        scrfd_qnn.Destroy();
        return 1;
    }

    arcface.Destroy();
    scrfd_qnn.Destroy();

    fprintf(stderr, "\n=== Enroll Summary ===\n");
    fprintf(stderr, "  Baseline: %s\n", filename.c_str());
    fprintf(stderr, "  ||emb||^2: %.6f\n", norm_sq);
    fprintf(stderr, "  Gallery saved: %s (1 entry)\n", gallery_path.c_str());

    return 0;
}

// ============================================================
// Verify: all images in directory vs gallery → similarity table + optional matrix
// ============================================================
static int do_verify(const std::string& scrfd_model,
                     const std::string& arcface_model,
                     const std::string& pic_dir,
                     const std::string& gallery_path,
                     bool print_matrix) {
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  VERIFY (SCRFD + ArcFace aligned)\n");
    fprintf(stderr, "  scrfd:   %s\n", scrfd_model.c_str());
    fprintf(stderr, "  arcface: %s\n", arcface_model.c_str());
    fprintf(stderr, "  pic_dir: %s\n", pic_dir.c_str());
    fprintf(stderr, "  gallery: %s\n", gallery_path.c_str());
    fprintf(stderr, "========================================\n\n");

    // Load gallery
    std::vector<GalleryEntry> gallery;
    if (!load_gallery(gallery_path, gallery)) {
        fprintf(stderr, "FATAL: Failed to load gallery from %s\n", gallery_path.c_str());
        return 1;
    }

    auto images = scan_images(pic_dir);
    if (images.empty()) {
        fprintf(stderr, "FATAL: No images found in %s\n", pic_dir.c_str());
        return 1;
    }
    fprintf(stderr, "Found %zu images, gallery has %zu entries\n\n",
            images.size(), gallery.size());

    // Init SCRFD
    face_infer::QnnInferencer scrfd_qnn;
    if (!scrfd_qnn.Init("/usr/lib/libQnnHtp.so", "/usr/lib/libQnnSystem.so", scrfd_model)) {
        fprintf(stderr, "FATAL: SCRFD QnnInferencer Init failed\n");
        return 1;
    }

    // Init ArcFace
    ArcFaceInfer arcface;
    if (!arcface.Init("/usr/lib/libQnnHtp.so", "/usr/lib/libQnnSystem.so", arcface_model)) {
        fprintf(stderr, "FATAL: ArcFaceInfer Init failed\n");
        scrfd_qnn.Destroy();
        return 1;
    }

    struct VerifyResult {
        std::string filename;
        bool ok;
        float embedding[kEmbDim];
        std::string best_match;
        float best_sim;
        float self_sim;
        bool self_is_best;
    };

    std::vector<VerifyResult> results;

    for (size_t i = 0; i < images.size(); i++) {
        const auto& fname = images[i];
        std::string fpath = pic_dir + "/" + fname;

        VerifyResult vr;
        vr.filename = fname;
        vr.ok = false;
        vr.best_sim = -2.0f;
        vr.self_sim = -1.0f;
        vr.self_is_best = false;
        memset(vr.embedding, 0, sizeof(vr.embedding));

        if (!process_image(fpath, scrfd_qnn, arcface, vr.embedding)) {
            fprintf(stderr, "[%zu/%zu] %s: FAILED\n", i + 1, images.size(), fname.c_str());
            results.push_back(vr);
            continue;
        }

        vr.ok = true;

        // Compare against all gallery entries
        for (size_t g = 0; g < gallery.size(); g++) {
            float sim = ArcFaceInfer::Similarity(vr.embedding, gallery[g].embedding);
            if (sim > vr.best_sim) {
                vr.best_sim = sim;
                vr.best_match = gallery[g].filename;
            }
            if (gallery[g].filename == fname) {
                vr.self_sim = sim;
            }
        }

        vr.self_is_best = (vr.best_match == fname);

        fprintf(stderr, "[%zu/%zu] %s: sim=%.6f match=%s\n",
                i + 1, images.size(), fname.c_str(),
                vr.best_sim, vr.best_match.c_str());

        results.push_back(vr);
    }

    arcface.Destroy();
    scrfd_qnn.Destroy();

    // ---- Results table ----
    fprintf(stderr, "\n");
    fprintf(stderr, "==========================================================================\n");
    fprintf(stderr, "  POSE SIMILARITY RESULTS (SCRFD + ArcFace aligned)\n");
    fprintf(stderr, "==========================================================================\n");
    fprintf(stderr, "%-20s %-8s %-20s %-10s %-10s %-8s\n",
            "IMAGE", "STATUS", "GALLERY_MATCH", "BEST_SIM", "SELF_SIM", "SELF_IS");
    fprintf(stderr, "%-20s %-8s %-20s %-10s %-10s %-8s\n",
            "-----", "------", "-------------", "--------", "--------", "-------");

    float sim_min = 2.0f, sim_max = -2.0f, sim_sum = 0;
    int sim_count = 0;
    int ok_count = 0, fail_count = 0;

    for (const auto& vr : results) {
        if (!vr.ok) {
            fprintf(stderr, "%-20s %-8s %-20s %-10s %-10s %-8s\n",
                    vr.filename.c_str(), "FAIL", "-", "-", "-", "-");
            fail_count++;
        } else {
            char best_sim_str[16], self_sim_str[16];
            snprintf(best_sim_str, sizeof(best_sim_str), "%.6f", vr.best_sim);
            snprintf(self_sim_str, sizeof(self_sim_str),
                     vr.self_sim >= 0 ? "%.6f" : "-", vr.self_sim);

            fprintf(stderr, "%-20s %-8s %-20s %-10s %-10s %-8s\n",
                    vr.filename.c_str(),
                    "OK",
                    vr.best_match.c_str(),
                    best_sim_str,
                    self_sim_str,
                    vr.self_is_best ? "YES" : "NO");
            ok_count++;

            if (vr.best_sim < sim_min) sim_min = vr.best_sim;
            if (vr.best_sim > sim_max) sim_max = vr.best_sim;
            sim_sum += vr.best_sim;
            sim_count++;
        }
    }

    // ---- Optional: cross-similarity matrix ----
    if (print_matrix) {
        std::vector<const VerifyResult*> with_emb;
        for (const auto& vr : results)
            if (vr.ok) with_emb.push_back(&vr);

        if (with_emb.size() >= 2) {
            fprintf(stderr, "\n=== Cross-Similarity Matrix ===\n");

            // strip extension from filenames for display
            std::vector<std::string> names;
            size_t max_len = 0;
            for (const auto* vr : with_emb) {
                std::string name = vr->filename;
                size_t dot = name.find_last_of('.');
                if (dot != std::string::npos) name = name.substr(0, dot);
                names.push_back(name);
                if (name.size() > max_len) max_len = name.size();
            }
            // column width: at least filename length + 1, minimum 10
            int col_w = static_cast<int>(std::max(max_len + 1, (size_t)10));
            int row_w = col_w;  // row label width = same

            // header
            fprintf(stderr, "%*s", row_w, "");
            for (const auto& n : names)
                fprintf(stderr, " %*s", col_w, n.c_str());
            fprintf(stderr, "\n");

            for (size_t r = 0; r < with_emb.size(); r++) {
                fprintf(stderr, "%*s", row_w, names[r].c_str());

                for (size_t c = 0; c < with_emb.size(); c++) {
                    float sim = ArcFaceInfer::Similarity(
                        with_emb[r]->embedding, with_emb[c]->embedding);
                    fprintf(stderr, " %*.6f", col_w, sim);
                }
                fprintf(stderr, "\n");
            }
        }
    }

    // ---- Summary ----
    fprintf(stderr, "\n=== Verify Summary ===\n");
    fprintf(stderr, "  Total images:     %zu\n", images.size());
    fprintf(stderr, "  Gallery entries:  %zu\n", gallery.size());
    fprintf(stderr, "  OK:               %d\n", ok_count);
    fprintf(stderr, "  FAIL:             %d\n", fail_count);

    if (sim_count > 0) {
        fprintf(stderr, "\n  Similarity to baseline (N=%d):\n", sim_count);
        fprintf(stderr, "    min=%.6f  max=%.6f  avg=%.6f\n",
                sim_min, sim_max, sim_sum / sim_count);
    }

    fprintf(stderr, "\n==========================================================================\n");
    fprintf(stderr, "  POSE TEST COMPLETE\n");
    fprintf(stderr, "==========================================================================\n");

    return (fail_count > 0) ? 1 : 0;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    fprintf(stderr, "=========================================\n");
    fprintf(stderr, "  ArcFace Pose Similarity Test\n");
    fprintf(stderr, "  (SCRFD aligned pipeline)\n");
    fprintf(stderr, "=========================================\n");

    if (argc < 5) {
        fprintf(stderr, "\nUsage:\n");
        fprintf(stderr, "  %s <scrfd_model> <arcface_model> --enroll <image_path> <gallery.json>\n", argv[0]);
        fprintf(stderr, "  %s <scrfd_model> <arcface_model> --verify <pic_dir> <gallery.json> [--matrix]\n", argv[0]);
        fprintf(stderr, "\nAll paths should be absolute.\n");
        return 1;
    }

    const char* scrfd_model  = argv[1];
    const char* arcface_model = argv[2];
    const char* mode          = argv[3];
    const char* path_arg      = argv[4];
    const char* gallery_arg   = (argc > 5) ? argv[5] : nullptr;

    bool print_matrix = false;
    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "--matrix") == 0) print_matrix = true;
    }

    if (strcmp(mode, "--enroll") == 0) {
        if (!gallery_arg) {
            fprintf(stderr, "FATAL: --enroll requires <image_path> <gallery.json>\n");
            return 1;
        }
        return do_enroll(scrfd_model, arcface_model, path_arg, gallery_arg);
    } else if (strcmp(mode, "--verify") == 0) {
        if (!gallery_arg) {
            fprintf(stderr, "FATAL: --verify requires <pic_dir> <gallery.json>\n");
            return 1;
        }
        return do_verify(scrfd_model, arcface_model, path_arg, gallery_arg, print_matrix);
    } else {
        fprintf(stderr, "Unknown mode: %s (use --enroll or --verify)\n", mode);
        return 1;
    }
}
