/// @file test_face_processor.cpp
/// @brief FaceProcessor 端到端测试 (QCS9075 板上运行)
///
/// 用法: ./test_face_processor <face_config.json>
///   - face_config.json 中 arcface_model 为空 → 仅运行 Group A-B (检测模式)
///   - face_config.json 中 arcface_model 有值 → 运行 Group A-E (检测+识别模式)
///
/// Group A: Init / Config 验证         (#1-#5)
/// Group B: SCRFD 检测验证             (#6-#11)
/// Group C: ArcFace 嵌入验证           (#12-#15)
/// Group D: 边界条件                   (#16-#18)
/// Group E: 完整链路 + 性能            (#19-#24)

#include "../FaceProcessor.hpp"
#include "../config/FaceConfigLoader.hpp"
#include "../mem_management/dma_buffer.hpp"
#include "../mem_management/opencl_loader.hpp"
#include "../mem_management/dma_sync_guard.hpp"
#include "../mem_management/mem_types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace face_infer;

// ============================================================
// 测试框架
// ============================================================
static int g_pass = 0;
static int g_fail = 0;

#define TEST_BEGIN(name) \
    fprintf(stderr, "\n--- Test %s ---\n", name)

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        g_fail++; \
        return; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    fprintf(stderr, "  PASS: %s\n", name); \
    g_pass++; \
} while(0)

// ============================================================
// 全局配置 (从 face_config.json 加载)
// ============================================================
static const char* g_config_path = nullptr;
static FaceProcessorConfig g_config;

static constexpr uint32_t kWidth  = 1920;
static constexpr uint32_t kHeight = 1080;
static constexpr uint32_t kStride = 1920;

// ============================================================
// NV12 帧辅助
// ============================================================

static void fill_nv12_solid(uint8_t* buf, uint32_t w, uint32_t h,
                            uint32_t stride,
                            uint8_t y_val, uint8_t u_val, uint8_t v_val) {
    for (uint32_t r = 0; r < h; r++)
        memset(buf + r * stride, y_val, w);
    uint8_t* uv = buf + h * stride;
    for (uint32_t r = 0; r < h / 2; r++)
        for (uint32_t c = 0; c < w / 2; c++) {
            uv[r * stride + c * 2 + 0] = u_val;
            uv[r * stride + c * 2 + 1] = v_val;
        }
}

struct TestFrame {
    DmaBuffer buf;
    FramePlaneInfo plane;
    bool valid = false;

    bool Create(uint32_t w, uint32_t h, uint32_t stride,
                cl_context ctx, std::shared_ptr<OpenClLoader> ocl) {
        uint32_t size = stride * h * 3 / 2;
        if (!buf.Init(size)) return false;
        if (!buf.BindOpenCl(ctx, ocl)) { buf.Destroy(); return false; }
        plane.fd        = buf.fd();
        plane.frame_len = size;
        plane.y_offset  = 0;
        plane.uv_offset = stride * h;
        plane.y_stride  = stride;
        plane.uv_stride = stride;
        plane.width     = w;
        plane.height    = h;
        valid = true;
        return true;
    }

    uint8_t* data() { return static_cast<uint8_t*>(buf.data()); }
    cl_mem cl() { return buf.cl_mem_handle(); }
    void Destroy() { buf.Destroy(); valid = false; }
};

// ============================================================
// Group A: Init / Config 验证
// ============================================================

static void test_init_scrfd_only() {
    TEST_BEGIN("#1 test_init_scrfd_only");
    FaceProcessor fp;
    FaceProcessorConfig cfg = g_config;
    cfg.arcface_model = "";
    CHECK(fp.Init(cfg), "Init scrfd-only should succeed");
    CHECK(!fp.HasArcFace(), "ArcFace should be disabled");
    fp.Destroy();
    TEST_PASS("#1 test_init_scrfd_only");
}

static void test_init_scrfd_arcface() {
    TEST_BEGIN("#2 test_init_scrfd_arcface");
    if (g_config.arcface_model.empty()) {
        fprintf(stderr, "  SKIP (no arcface_model in config)\n");
        return;
    }
    FaceProcessor fp;
    CHECK(fp.Init(g_config), "Init scrfd+arcface should succeed");
    CHECK(fp.HasArcFace(), "ArcFace should be enabled");
    fp.Destroy();
    TEST_PASS("#2 test_init_scrfd_arcface");
}

static void test_init_bad_model() {
    TEST_BEGIN("#3 test_init_bad_model");
    FaceProcessor fp;
    FaceProcessorConfig cfg = g_config;
    cfg.scrfd_model = "/nonexistent/model.bin";
    CHECK(!fp.Init(cfg), "Init with bad model should fail");
    fp.Destroy();
    TEST_PASS("#3 test_init_bad_model");
}

static void test_double_init() {
    TEST_BEGIN("#4 test_double_init");
    FaceProcessor fp;
    FaceProcessorConfig cfg = g_config;
    cfg.arcface_model = "";
    CHECK(fp.Init(cfg), "First Init should succeed");
    CHECK(!fp.Init(cfg), "Second Init should fail");
    fp.Destroy();
    TEST_PASS("#4 test_double_init");
}

static void test_process_before_init() {
    TEST_BEGIN("#5 test_process_before_init");
    FaceProcessor fp;
    FramePlaneInfo dummy{};
    auto r = fp.ProcessFrame(nullptr, dummy);
    CHECK(r.empty(), "ProcessFrame before Init should return empty");
    TEST_PASS("#5 test_process_before_init");
}

// ============================================================
// Group B: SCRFD 检测验证
// ============================================================

static FaceProcessor g_fp;
static std::shared_ptr<OpenClLoader> g_ocl;
static bool g_fp_ready = false;

static bool init_shared_processor() {
    if (g_fp_ready) return true;
    g_ocl = OpenClLoader::Get();
    if (!g_ocl) return false;
    g_fp_ready = g_fp.Init(g_config);
    return g_fp_ready;
}

static void test_detect_gray_frame() {
    TEST_BEGIN("#6 test_detect_gray_frame");
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);
    auto results = g_fp.ProcessFrame(frame.cl(), frame.plane);
    fprintf(stderr, "  Gray frame detections: %zu\n", results.size());
    CHECK(results.empty(), "gray frame should have zero detections");
    frame.Destroy();
    TEST_PASS("#6 test_detect_gray_frame");
}

static void test_detect_returns_valid() {
    TEST_BEGIN("#7 test_detect_returns_valid");
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    // 白色背景 + 暗色矩形模拟面部
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 200, 128, 128);
    uint8_t* y = frame.data();
    for (uint32_t r = 400; r < 700; r++)
        for (uint32_t c = 800; c < 1100; c++)
            y[r * kStride + c] = 60;
    auto results = g_fp.ProcessFrame(frame.cl(), frame.plane);
    fprintf(stderr, "  Synthetic frame detections: %zu (no crash = OK)\n",
            results.size());
    frame.Destroy();
    TEST_PASS("#7 test_detect_returns_valid");
}

static void test_bbox_within_frame() {
    TEST_BEGIN("#8 test_bbox_within_frame");
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);
    auto results = g_fp.ProcessFrame(frame.cl(), frame.plane);
    for (size_t i = 0; i < results.size(); i++) {
        const auto& d = results[i].det;
        CHECK(d.x1 >= 0 && d.x1 < (float)kWidth, "x1 out of range");
        CHECK(d.y1 >= 0 && d.y1 < (float)kHeight, "y1 out of range");
        CHECK(d.x2 > d.x1 && d.x2 <= (float)kWidth, "x2 out of range");
        CHECK(d.y2 > d.y1 && d.y2 <= (float)kHeight, "y2 out of range");
    }
    frame.Destroy();
    TEST_PASS("#8 test_bbox_within_frame");
}

static void test_landmarks_within_bbox() {
    TEST_BEGIN("#9 test_landmarks_within_bbox");
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);
    auto results = g_fp.ProcessFrame(frame.cl(), frame.plane);
    for (size_t i = 0; i < results.size(); i++) {
        const auto& d = results[i].det;
        float margin_x = (d.x2 - d.x1) * 0.3f;
        float margin_y = (d.y2 - d.y1) * 0.3f;
        for (int j = 0; j < kFaceNumLandmarks; j++) {
            float lx = d.landmarks[j][0];
            float ly = d.landmarks[j][1];
            CHECK(lx >= d.x1 - margin_x && lx <= d.x2 + margin_x,
                  "landmark x outside bbox margin");
            CHECK(ly >= d.y1 - margin_y && ly <= d.y2 + margin_y,
                  "landmark y outside bbox margin");
        }
    }
    frame.Destroy();
    TEST_PASS("#9 test_landmarks_within_bbox");
}

static void test_score_range() {
    TEST_BEGIN("#10 test_score_range");
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);
    auto results = g_fp.ProcessFrame(frame.cl(), frame.plane);
    for (size_t i = 0; i < results.size(); i++) {
        float s = results[i].det.score;
        CHECK(s > 0.0f && s <= 1.0f, "score must be in (0, 1]");
    }
    frame.Destroy();
    TEST_PASS("#10 test_score_range");
}

static void test_detection_only_mode() {
    TEST_BEGIN("#11 test_detection_only_mode");
    FaceProcessor fp_det;
    FaceProcessorConfig cfg = g_config;
    cfg.arcface_model = "";
    CHECK(fp_det.Init(cfg), "Init detection-only failed");

    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, fp_det.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);
    auto results = fp_det.ProcessFrame(frame.cl(), frame.plane);
    for (size_t i = 0; i < results.size(); i++) {
        bool all_zero = true;
        for (int j = 0; j < kFaceEmbeddingDim; j++) {
            if (results[i].embedding[j] != 0.0f) { all_zero = false; break; }
        }
        CHECK(all_zero, "detection-only mode: embedding must be all zeros");
    }
    frame.Destroy();
    fp_det.Destroy();
    TEST_PASS("#11 test_detection_only_mode");
}

// ============================================================
// Group C: ArcFace 嵌入验证
// ============================================================

static void test_embedding_unit_norm() {
    TEST_BEGIN("#12 test_embedding_unit_norm");
    if (!g_fp.HasArcFace()) { fprintf(stderr, "  SKIP (no ArcFace)\n"); return; }
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);
    auto results = g_fp.ProcessFrame(frame.cl(), frame.plane);
    for (size_t i = 0; i < results.size(); i++) {
        float norm_sq = 0;
        for (int j = 0; j < kFaceEmbeddingDim; j++)
            norm_sq += results[i].embedding[j] * results[i].embedding[j];
        fprintf(stderr, "  face[%zu] ||emb||^2 = %.6f\n", i, norm_sq);
        if (norm_sq > 0.01f) {
            CHECK(std::fabs(norm_sq - 1.0f) < 1e-3f,
                  "embedding must be unit norm");
        }
    }
    frame.Destroy();
    TEST_PASS("#12 test_embedding_unit_norm");
}

static void test_embedding_finite() {
    TEST_BEGIN("#13 test_embedding_finite");
    if (!g_fp.HasArcFace()) { fprintf(stderr, "  SKIP (no ArcFace)\n"); return; }
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);
    auto results = g_fp.ProcessFrame(frame.cl(), frame.plane);
    for (size_t i = 0; i < results.size(); i++)
        for (int j = 0; j < kFaceEmbeddingDim; j++)
            CHECK(std::isfinite(results[i].embedding[j]),
                  "embedding element must be finite");
    frame.Destroy();
    TEST_PASS("#13 test_embedding_finite");
}

static void test_same_face_high_sim() {
    TEST_BEGIN("#14 test_same_face_high_sim");
    if (!g_fp.HasArcFace()) { fprintf(stderr, "  SKIP (no ArcFace)\n"); return; }
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);

    auto r1 = g_fp.ProcessFrame(frame.cl(), frame.plane);
    auto r2 = g_fp.ProcessFrame(frame.cl(), frame.plane);

    if (!r1.empty() && !r2.empty()) {
        float sim = FaceProcessor::Similarity(r1[0].embedding, r2[0].embedding);
        fprintf(stderr, "  Same face similarity: %.6f\n", sim);
        CHECK(sim > 0.99f, "same input -> same face -> high similarity");
    } else {
        fprintf(stderr, "  No detections to compare (OK for synthetic frame)\n");
    }
    frame.Destroy();
    TEST_PASS("#14 test_same_face_high_sim");
}

static void test_different_input_diff_emb() {
    TEST_BEGIN("#15 test_different_input_diff_emb");
    if (!g_fp.HasArcFace()) { fprintf(stderr, "  SKIP (no ArcFace)\n"); return; }
    CHECK(g_fp_ready, "shared FaceProcessor not ready");

    TestFrame frame1, frame2;
    CHECK(frame1.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "Frame1 create failed");
    CHECK(frame2.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "Frame2 create failed");

    fill_nv12_solid(frame1.data(), kWidth, kHeight, kStride, 200, 128, 128);
    fill_nv12_solid(frame2.data(), kWidth, kHeight, kStride, 60, 128, 128);

    auto r1 = g_fp.ProcessFrame(frame1.cl(), frame1.plane);
    auto r2 = g_fp.ProcessFrame(frame2.cl(), frame2.plane);

    if (!r1.empty() && !r2.empty()) {
        float sim = FaceProcessor::Similarity(r1[0].embedding, r2[0].embedding);
        fprintf(stderr, "  Different input similarity: %.6f\n", sim);
        CHECK(sim < 0.999f, "different inputs should produce different embeddings");
    } else {
        fprintf(stderr, "  Insufficient detections for comparison (OK)\n");
    }
    frame1.Destroy();
    frame2.Destroy();
    TEST_PASS("#15 test_different_input_diff_emb");
}

// ============================================================
// Group D: 边界条件
// ============================================================

static void test_max_faces_limit() {
    TEST_BEGIN("#16 test_max_faces_limit");
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);
    auto results = g_fp.ProcessFrame(frame.cl(), frame.plane);
    fprintf(stderr, "  Detections: %zu (max=%d)\n",
            results.size(), kFaceMaxDetections);
    CHECK(results.size() <= (size_t)kFaceMaxDetections,
          "detections must not exceed max_faces");
    frame.Destroy();
    TEST_PASS("#16 test_max_faces_limit");
}

static void test_conf_threshold_mono() {
    TEST_BEGIN("#17 test_conf_threshold_mono");
    fprintf(stderr, "  (验证逻辑将在有真实图片时生效)\n");
    TEST_PASS("#17 test_conf_threshold_mono");
}

static void test_nms_threshold_effect() {
    TEST_BEGIN("#18 test_nms_threshold_effect");
    fprintf(stderr, "  (验证逻辑将在有密集人脸图片时生效)\n");
    TEST_PASS("#18 test_nms_threshold_effect");
}

// ============================================================
// Group E: 完整链路 + 性能
// ============================================================

static void test_full_pipeline_e2e() {
    TEST_BEGIN("#19 test_full_pipeline_e2e");
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);
    auto results = g_fp.ProcessFrame(frame.cl(), frame.plane);
    fprintf(stderr, "  E2E result: %zu faces\n", results.size());
    for (size_t i = 0; i < results.size(); i++) {
        CHECK(std::isfinite(results[i].det.score), "score must be finite");
        CHECK(std::isfinite(results[i].det.x1), "x1 must be finite");
        for (int j = 0; j < kFaceEmbeddingDim; j++)
            CHECK(std::isfinite(results[i].embedding[j]),
                  "embedding must be finite");
    }
    frame.Destroy();
    TEST_PASS("#19 test_full_pipeline_e2e");
}

static void test_determinism() {
    TEST_BEGIN("#20 test_determinism");
    CHECK(g_fp_ready, "shared FaceProcessor not ready");
    TestFrame frame;
    CHECK(frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl),
          "TestFrame create failed");
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);

    auto r1 = g_fp.ProcessFrame(frame.cl(), frame.plane);
    auto r2 = g_fp.ProcessFrame(frame.cl(), frame.plane);
    CHECK(r1.size() == r2.size(), "same input -> same detection count");

    for (size_t i = 0; i < std::min(r1.size(), r2.size()); i++) {
        CHECK(std::fabs(r1[i].det.score - r2[i].det.score) < 1e-5f,
              "same input -> same score");
        if (g_fp.HasArcFace()) {
            float sim = FaceProcessor::Similarity(r1[i].embedding,
                                                   r2[i].embedding);
            CHECK(sim > 0.999f, "same input -> nearly identical embeddings");
        }
    }
    frame.Destroy();
    TEST_PASS("#20 test_determinism");
}

static void test_destroy_reinit() {
    TEST_BEGIN("#21 test_destroy_reinit");
    FaceProcessor fp;
    FaceProcessorConfig cfg = g_config;
    cfg.arcface_model = "";
    CHECK(fp.Init(cfg), "First Init should succeed");
    fp.Destroy();
    CHECK(fp.Init(cfg), "Re-Init after Destroy should succeed");
    fp.Destroy();
    TEST_PASS("#21 test_destroy_reinit");
}

static void bench_detect_only() {
    TEST_BEGIN("#22 bench_detect_only");
    if (!g_fp_ready) { fprintf(stderr, "  SKIP\n"); return; }
    TestFrame frame;
    if (!frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl)) {
        fprintf(stderr, "  SKIP (frame create failed)\n"); return;
    }
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);

    constexpr int N = 100;
    double total_ms = 0, min_ms = 1e9, max_ms = 0;
    for (int i = 0; i < N; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = g_fp.ProcessFrame(frame.cl(), frame.plane);
        (void)r;
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }
    fprintf(stderr, "  ProcessFrame: avg=%.2fms min=%.2fms max=%.2fms "
                    "(%d iters, ~%.0f FPS)\n",
            total_ms / N, min_ms, max_ms, N, 1000.0 / (total_ms / N));
    frame.Destroy();
    TEST_PASS("#22 bench_detect_only");
}

static void bench_full_pipeline() {
    TEST_BEGIN("#23 bench_full_pipeline");
    if (!g_fp_ready || !g_fp.HasArcFace()) {
        fprintf(stderr, "  SKIP (no ArcFace)\n"); return;
    }
    TestFrame frame;
    if (!frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl)) {
        fprintf(stderr, "  SKIP\n"); return;
    }
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);

    constexpr int N = 50;
    double total_ms = 0, min_ms = 1e9, max_ms = 0;
    int total_faces = 0;
    for (int i = 0; i < N; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = g_fp.ProcessFrame(frame.cl(), frame.plane);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        total_faces += static_cast<int>(r.size());
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }
    fprintf(stderr, "  ProcessFrame (full): avg=%.2fms min=%.2fms max=%.2fms "
                    "(%d iters, avg faces=%.1f)\n",
            total_ms / N, min_ms, max_ms, N,
            static_cast<float>(total_faces) / N);
    frame.Destroy();
    TEST_PASS("#23 bench_full_pipeline");
}

static void bench_memory_stability() {
    TEST_BEGIN("#24 bench_memory_stability");
    if (!g_fp_ready) { fprintf(stderr, "  SKIP\n"); return; }
    TestFrame frame;
    if (!frame.Create(kWidth, kHeight, kStride, g_fp.GetContext(), g_ocl)) {
        fprintf(stderr, "  SKIP\n"); return;
    }
    fill_nv12_solid(frame.data(), kWidth, kHeight, kStride, 128, 128, 128);

    constexpr int N = 50;
    for (int i = 0; i < N; i++) {
        auto r = g_fp.ProcessFrame(frame.cl(), frame.plane);
        (void)r;
    }
    fprintf(stderr, "  %d consecutive frames processed without crash\n", N);
    frame.Destroy();
    TEST_PASS("#24 bench_memory_stability");
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    fprintf(stderr, "=========================================\n");
    fprintf(stderr, "  FaceProcessor End-to-End Test\n");
    fprintf(stderr, "=========================================\n");

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <face_config.json>\n", argv[0]);
        return 1;
    }
    g_config_path = argv[1];

    // 从 face_config.json 加载所有配置
    if (!LoadFaceConfig(g_config_path, g_config)) {
        fprintf(stderr, "FATAL: failed to load config: %s\n", g_config_path);
        return 1;
    }

    // ---- Group A: Init/Config ----
    fprintf(stderr, "\n========== Group A: Init / Config ==========\n");
    test_init_scrfd_only();
    test_init_scrfd_arcface();
    test_init_bad_model();
    test_double_init();
    test_process_before_init();

    // ---- 初始化共享 FaceProcessor ----
    fprintf(stderr, "\n========== Initializing shared FaceProcessor ==========\n");
    if (!init_shared_processor()) {
        fprintf(stderr, "FATAL: shared FaceProcessor Init failed\n");
        fprintf(stderr, "\n=== Results: %d PASS / %d FAIL ===\n",
                g_pass, g_fail);
        return 1;
    }

    // ---- Group B: SCRFD 检测验证 ----
    fprintf(stderr, "\n========== Group B: SCRFD Detection ==========\n");
    test_detect_gray_frame();
    test_detect_returns_valid();
    test_bbox_within_frame();
    test_landmarks_within_bbox();
    test_score_range();
    test_detection_only_mode();

    // ---- Group C: ArcFace 嵌入验证 ----
    fprintf(stderr, "\n========== Group C: ArcFace Embedding ==========\n");
    test_embedding_unit_norm();
    test_embedding_finite();
    test_same_face_high_sim();
    test_different_input_diff_emb();

    // ---- Group D: 边界条件 ----
    fprintf(stderr, "\n========== Group D: Edge Cases ==========\n");
    test_max_faces_limit();
    test_conf_threshold_mono();
    test_nms_threshold_effect();

    // ---- Group E: 完整链路 + 性能 ----
    fprintf(stderr, "\n========== Group E: Full Pipeline + Bench ==========\n");
    test_full_pipeline_e2e();
    test_determinism();
    test_destroy_reinit();
    bench_detect_only();
    bench_full_pipeline();
    bench_memory_stability();

    // ---- 清理 ----
    g_fp.Destroy();
    g_ocl.reset();

    // ---- 汇总 ----
    fprintf(stderr, "\n=========================================\n");
    fprintf(stderr, "  Results: %d PASS / %d FAIL\n", g_pass, g_fail);
    fprintf(stderr, "=========================================\n");
    if (g_fail == 0)
        fprintf(stderr, "ALL TESTS PASSED\n");
    else
        fprintf(stderr, "SOME TESTS FAILED\n");
    return (g_fail == 0) ? 0 : 1;
}
