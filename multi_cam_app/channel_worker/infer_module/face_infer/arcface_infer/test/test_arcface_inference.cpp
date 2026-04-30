/// @file test_arcface_inference.cpp
/// @brief ArcFace 推理端到端测试 (QCS9075 板上运行)
///
/// Group A (纯数学, 始终运行):
///   #1  test_l2_norm_unit_vector      — 单位向量归一化不变
///   #2  test_l2_norm_known_vector     — 已归一化向量不变
///   #3  test_l2_norm_scale_invariance — 缩放不变性
///   #4  test_l2_norm_zero_vector      — 零向量退化保护
///   #5  test_similarity_identity      — 自相似度 = 1.0
///   #6  test_similarity_orthogonal    — 正交 = 0.0
///   #7  test_similarity_antiparallel  — 反向 = -1.0
///   #8  test_similarity_symmetry      — 对称性
///
/// Group B (QNN 模型验证, 需要 argv[2]=model_path):
///   #9  test_arcface_init             — 模型加载 + tensor 验证
///   #10 test_arcface_execute_zero     — 全零输入不崩溃
///   #11 test_arcface_output_unit_norm — 输出为单位向量
///
/// Group C (完整链路, 需要 argv[1]+argv[2]):
///   #12 test_full_pipeline            — GPU warp + QNN + L2 完整链路
///   #13 test_determinism              — 确定性验证
///
/// Benchmarks:
///   #14 bench_qnn_execute             — 100 次 Infer 性能
///   #15 bench_full_pipeline           — 100 次 RunArcface+Infer 性能
///
/// 编译: bash build_test_arcface.sh
/// 运行: ./test_arcface_inference [kernel_dir] [model_path]

#include "../ArcFaceInfer.hpp"
#include "../../face_preprocess/FacePreprocess.hpp"
#include "../../face_preprocess/inverse_affine.hpp"
#include "../../mem_management/dma_buffer.hpp"
#include "../../mem_management/opencl_loader.hpp"
#include "../../mem_management/dma_sync_guard.hpp"
#include "../../mem_management/mem_types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace arcface_infer;
using face_infer::OpenClLoader;
using face_infer::DmaBuffer;
using face_infer::FramePlaneInfo;

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
// 全局共享对象
// ============================================================
static std::shared_ptr<OpenClLoader> g_ocl;
static face_infer::FacePreprocess g_gpu;
static ArcFaceInfer               g_arc;
static const char*                g_kernel_dir = nullptr;
static const char*                g_model_path = nullptr;
static bool                       g_gpu_ready  = false;
static bool                       g_qnn_ready  = false;

// 测试用 NV12 帧参数
static constexpr uint32_t kTestWidth  = 1920;
static constexpr uint32_t kTestHeight = 1080;
static constexpr uint32_t kTestStride = 1920;
static constexpr uint32_t kTestNv12Size = kTestStride * kTestHeight * 3 / 2;

// ============================================================
// NV12 辅助
// ============================================================
static void fill_nv12_solid(uint8_t* buf, uint32_t w, uint32_t h, uint32_t stride,
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

struct NV12Frame {
    DmaBuffer buf;
    FramePlaneInfo plane;
    bool valid = false;

    bool Create(uint32_t w, uint32_t h, uint32_t stride, cl_context ctx,
                std::shared_ptr<OpenClLoader> ocl) {
        uint32_t nv12_size = stride * h * 3 / 2;
        if (!buf.Init(nv12_size)) {
            fprintf(stderr, "[NV12Frame] DmaBuffer::Init failed\n");
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
// L2 归一化辅助 (测试用独立实现, 不依赖 ArcFaceInfer)
// ============================================================
static void l2_normalize_512(const float* in, float* out) {
    float norm = 0.0f;
    for (int i = 0; i < 512; i++)
        norm += in[i] * in[i];
    norm = sqrtf(norm + 1e-10f);
    for (int i = 0; i < 512; i++)
        out[i] = in[i] / norm;
}

// ============================================================
// Group A: 纯数学测试 (无 QNN/GPU, 始终运行)
// ============================================================

static void test_l2_norm_unit_vector() {
    TEST_BEGIN("test_l2_norm_unit_vector");
    float v[512] = {0};
    v[0] = 1.0f;
    float out[512];
    l2_normalize_512(v, out);
    CHECK(std::fabs(out[0] - 1.0f) < 1e-6f, "out[0] should be ~1.0");
    for (int i = 1; i < 512; i++)
        CHECK(std::fabs(out[i]) < 1e-6f, "out[i] should be ~0.0");
    TEST_PASS("test_l2_norm_unit_vector");
}

static void test_l2_norm_known_vector() {
    TEST_BEGIN("test_l2_norm_known_vector");
    float v[512], out[512];
    float val = 1.0f / sqrtf(512.0f);
    for (int i = 0; i < 512; i++) v[i] = val;
    l2_normalize_512(v, out);
    for (int i = 0; i < 512; i++)
        CHECK(std::fabs(out[i] - val) < 1e-5f, "pre-normalized vector should be unchanged");
    TEST_PASS("test_l2_norm_known_vector");
}

static void test_l2_norm_scale_invariance() {
    TEST_BEGIN("test_l2_norm_scale_invariance");
    float a[512], b[512], norm_a[512], norm_b[512];
    for (int i = 0; i < 512; i++) {
        a[i] = sinf(i * 0.1f) + 0.5f;
        b[i] = a[i] * 5.0f;
    }
    l2_normalize_512(a, norm_a);
    l2_normalize_512(b, norm_b);
    for (int i = 0; i < 512; i++)
        CHECK(std::fabs(norm_a[i] - norm_b[i]) < 1e-5f,
              "L2 norm should be scale-invariant");
    TEST_PASS("test_l2_norm_scale_invariance");
}

static void test_l2_norm_zero_vector() {
    TEST_BEGIN("test_l2_norm_zero_vector");
    float v[512] = {0}, out[512];
    l2_normalize_512(v, out);
    for (int i = 0; i < 512; i++)
        CHECK(std::isfinite(out[i]), "zero vector output must be finite (no NaN/Inf)");
    TEST_PASS("test_l2_norm_zero_vector");
}

static void test_similarity_identity() {
    TEST_BEGIN("test_similarity_identity");
    float v[512], u[512];
    for (int i = 0; i < 512; i++) v[i] = sinf(i * 0.2f);
    l2_normalize_512(v, u);
    float sim = ArcFaceInfer::Similarity(u, u);
    fprintf(stderr, "  Similarity(U, U) = %.6f\n", sim);
    CHECK(std::fabs(sim - 1.0f) < 1e-5f, "self-similarity should be ~1.0");
    TEST_PASS("test_similarity_identity");
}

static void test_similarity_orthogonal() {
    TEST_BEGIN("test_similarity_orthogonal");
    float e0[512] = {0}, e1[512] = {0};
    e0[0] = 1.0f;
    e1[1] = 1.0f;
    float sim = ArcFaceInfer::Similarity(e0, e1);
    fprintf(stderr, "  Similarity(e0, e1) = %.6f\n", sim);
    CHECK(std::fabs(sim) < 1e-5f, "orthogonal vectors similarity should be ~0.0");
    TEST_PASS("test_similarity_orthogonal");
}

static void test_similarity_antiparallel() {
    TEST_BEGIN("test_similarity_antiparallel");
    float u[512], neg_u[512];
    float v[512];
    for (int i = 0; i < 512; i++) v[i] = cosf(i * 0.3f);
    l2_normalize_512(v, u);
    for (int i = 0; i < 512; i++) neg_u[i] = -u[i];
    float sim = ArcFaceInfer::Similarity(u, neg_u);
    fprintf(stderr, "  Similarity(U, -U) = %.6f\n", sim);
    CHECK(std::fabs(sim - (-1.0f)) < 1e-5f, "anti-parallel similarity should be ~-1.0");
    TEST_PASS("test_similarity_antiparallel");
}

static void test_similarity_symmetry() {
    TEST_BEGIN("test_similarity_symmetry");
    float a[512], b[512], na[512], nb[512];
    for (int i = 0; i < 512; i++) {
        a[i] = sinf(i * 0.1f);
        b[i] = cosf(i * 0.2f);
    }
    l2_normalize_512(a, na);
    l2_normalize_512(b, nb);
    float sim_ab = ArcFaceInfer::Similarity(na, nb);
    float sim_ba = ArcFaceInfer::Similarity(nb, na);
    fprintf(stderr, "  Similarity(A,B)=%.6f  Similarity(B,A)=%.6f\n", sim_ab, sim_ba);
    CHECK(std::fabs(sim_ab - sim_ba) < 1e-6f, "similarity should be symmetric");
    TEST_PASS("test_similarity_symmetry");
}

// ============================================================
// Group B: QNN 模型验证 (需要 g_qnn_ready)
// ============================================================

static void test_arcface_init() {
    TEST_BEGIN("test_arcface_init");
    CHECK(g_qnn_ready, "ArcFaceInfer Init must have succeeded");
    uint32_t n_out = g_arc.GetInferencer().GetNumOutputTensors();
    fprintf(stderr, "  ArcFace output tensors: %u\n", n_out);
    CHECK(n_out == 1, "expected 1 output tensor");
    uint32_t out_size = g_arc.GetInferencer().GetOutputTensorSize(0);
    std::string name = g_arc.GetInferencer().GetOutputTensorName(0);
    fprintf(stderr, "  Tensor 0: \"%s\" size=%u\n", name.c_str(), out_size);
    CHECK(out_size == 512, "expected output size 512");
    TEST_PASS("test_arcface_init");
}

static void test_arcface_execute_zero() {
    TEST_BEGIN("test_arcface_execute_zero");
    CHECK(g_qnn_ready, "ArcFaceInfer not ready");
    std::vector<float> zeros(ArcFaceInfer::kInputFloats, 0.0f);
    float emb[512];
    bool ok = g_arc.Infer(zeros.data(), emb);
    CHECK(ok, "Infer with zeros should succeed");
    int finite_count = 0;
    for (int i = 0; i < 512; i++)
        if (std::isfinite(emb[i])) finite_count++;
    fprintf(stderr, "  Finite elements: %d / 512\n", finite_count);
    CHECK(finite_count == 512, "all 512 elements must be finite");
    TEST_PASS("test_arcface_execute_zero");
}

static void test_arcface_output_unit_norm() {
    TEST_BEGIN("test_arcface_output_unit_norm");
    CHECK(g_qnn_ready, "ArcFaceInfer not ready");
    // 用 sin(i*0.1) 填充输入 (非零非均匀)
    std::vector<float> input(ArcFaceInfer::kInputFloats);
    for (uint32_t i = 0; i < ArcFaceInfer::kInputFloats; i++)
        input[i] = sinf(i * 0.1f);
    float emb[512];
    bool ok = g_arc.Infer(input.data(), emb);
    CHECK(ok, "Infer should succeed");
    float norm_sq = 0.0f;
    for (int i = 0; i < 512; i++)
        norm_sq += emb[i] * emb[i];
    fprintf(stderr, "  ||embedding||^2 = %.6f\n", norm_sq);
    CHECK(std::fabs(norm_sq - 1.0f) < 1e-4f, "embedding must be unit norm");
    TEST_PASS("test_arcface_output_unit_norm");
}

// ============================================================
// Group C: 完整链路集成 (需要 g_gpu_ready && g_qnn_ready)
// ============================================================

static void test_full_pipeline() {
    TEST_BEGIN("test_full_pipeline");
    CHECK(g_gpu_ready && g_qnn_ready, "GPU and QNN must be ready");

    NV12Frame frame;
    CHECK(frame.Create(kTestWidth, kTestHeight, kTestStride,
                       g_gpu.GetContext(), g_ocl),
          "NV12Frame create failed");

    // 填充灰色
    fill_nv12_solid(static_cast<uint8_t*>(frame.buf.data()),
                    kTestWidth, kTestHeight, kTestStride, 128, 128, 128);

    // 合成 landmarks (模拟 1920×1080 图中正脸)
    float landmarks[5][2] = {
        {460.0f, 300.0f},  // left eye
        {540.0f, 300.0f},  // right eye
        {500.0f, 370.0f},  // nose
        {470.0f, 440.0f},  // left mouth
        {530.0f, 440.0f}   // right mouth
    };
    float inv[6];
    face_infer::ComputeInverseAffine(landmarks, inv);

    // GPU warp: NV12 → NCHW 112×112
    const float* preproc = g_gpu.RunArcface(frame.buf.cl_mem_handle(),
                                             frame.plane, inv);
    CHECK(preproc != nullptr, "RunArcface returned nullptr");

    // QNN + L2 归一化
    float emb[512];
    bool ok = g_arc.Infer(preproc, emb);
    CHECK(ok, "Infer failed");

    // 验证: unit norm
    float norm_sq = 0.0f;
    for (int i = 0; i < 512; i++)
        norm_sq += emb[i] * emb[i];
    fprintf(stderr, "  ||embedding||^2 = %.6f\n", norm_sq);
    CHECK(std::fabs(norm_sq - 1.0f) < 1e-4f, "embedding must be unit norm");

    // 验证: 全部 finite
    for (int i = 0; i < 512; i++)
        CHECK(std::isfinite(emb[i]), "embedding element must be finite");

    frame.Destroy();
    TEST_PASS("test_full_pipeline");
}

static void test_determinism() {
    TEST_BEGIN("test_determinism");
    CHECK(g_gpu_ready && g_qnn_ready, "GPU and QNN must be ready");

    NV12Frame frame;
    CHECK(frame.Create(kTestWidth, kTestHeight, kTestStride,
                       g_gpu.GetContext(), g_ocl),
          "NV12Frame create failed");

    fill_nv12_solid(static_cast<uint8_t*>(frame.buf.data()),
                    kTestWidth, kTestHeight, kTestStride, 128, 128, 128);

    float landmarks[5][2] = {
        {460.0f, 300.0f}, {540.0f, 300.0f}, {500.0f, 370.0f},
        {470.0f, 440.0f}, {530.0f, 440.0f}
    };
    float inv[6];
    face_infer::ComputeInverseAffine(landmarks, inv);

    // 执行两次
    float emb1[512], emb2[512];

    const float* preproc1 = g_gpu.RunArcface(frame.buf.cl_mem_handle(),
                                              frame.plane, inv);
    CHECK(preproc1 != nullptr, "RunArcface #1 failed");
    CHECK(g_arc.Infer(preproc1, emb1), "Infer #1 failed");

    const float* preproc2 = g_gpu.RunArcface(frame.buf.cl_mem_handle(),
                                              frame.plane, inv);
    CHECK(preproc2 != nullptr, "RunArcface #2 failed");
    CHECK(g_arc.Infer(preproc2, emb2), "Infer #2 failed");

    float sim = ArcFaceInfer::Similarity(emb1, emb2);
    fprintf(stderr, "  Similarity(run1, run2) = %.6f\n", sim);
    CHECK(sim > 0.999f, "same input must produce nearly identical embeddings");

    frame.Destroy();
    TEST_PASS("test_determinism");
}

// ============================================================
// Benchmarks
// ============================================================

static void bench_qnn_execute() {
    TEST_BEGIN("bench_qnn_execute");
    if (!g_qnn_ready) { fprintf(stderr, "  SKIP (QNN not ready)\n"); return; }

    std::vector<float> zeros(ArcFaceInfer::kInputFloats, 0.0f);
    float emb[512];
    constexpr int N = 100;
    double total_ms = 0, min_ms = 1e9, max_ms = 0;

    for (int i = 0; i < N; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        g_arc.Infer(zeros.data(), emb);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }
    fprintf(stderr, "  ArcFace QNN: avg=%.2fms min=%.2fms max=%.2fms (%d iters)\n",
            total_ms / N, min_ms, max_ms, N);
    TEST_PASS("bench_qnn_execute");
}

static void bench_full_pipeline() {
    TEST_BEGIN("bench_full_pipeline");
    if (!g_gpu_ready || !g_qnn_ready) {
        fprintf(stderr, "  SKIP (GPU or QNN not ready)\n");
        return;
    }

    NV12Frame frame;
    if (!frame.Create(kTestWidth, kTestHeight, kTestStride,
                      g_gpu.GetContext(), g_ocl)) {
        fprintf(stderr, "  SKIP (NV12Frame creation failed)\n");
        return;
    }
    fill_nv12_solid(static_cast<uint8_t*>(frame.buf.data()),
                    kTestWidth, kTestHeight, kTestStride, 128, 128, 128);

    float landmarks[5][2] = {
        {460.0f, 300.0f}, {540.0f, 300.0f}, {500.0f, 370.0f},
        {470.0f, 440.0f}, {530.0f, 440.0f}
    };
    float inv[6];
    face_infer::ComputeInverseAffine(landmarks, inv);

    float emb[512];
    constexpr int N = 100;
    double total_ms = 0, min_ms = 1e9, max_ms = 0;

    for (int i = 0; i < N; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const float* preproc = g_gpu.RunArcface(frame.buf.cl_mem_handle(),
                                                 frame.plane, inv);
        if (preproc) g_arc.Infer(preproc, emb);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }
    fprintf(stderr, "  ArcFace full: avg=%.2fms min=%.2fms max=%.2fms (%d iters)\n",
            total_ms / N, min_ms, max_ms, N);
    frame.Destroy();
    TEST_PASS("bench_full_pipeline");
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    fprintf(stderr, "=== ArcFace Inference Test ===\n");

    if (argc > 1) g_kernel_dir = argv[1];
    if (argc > 2) g_model_path = argv[2];

    if (!g_kernel_dir)
        g_kernel_dir = "../../face_preprocess/kernels";
    if (!g_model_path)
        g_model_path = "../../../data/models/face/w600k_r50_qcs9075.bin";

    fprintf(stderr, "  kernel_dir: %s\n", g_kernel_dir);
    fprintf(stderr, "  model_path: %s\n", g_model_path);

    // --- Group A: 纯数学 (始终运行) ---
    test_l2_norm_unit_vector();
    test_l2_norm_known_vector();
    test_l2_norm_scale_invariance();
    test_l2_norm_zero_vector();
    test_similarity_identity();
    test_similarity_orthogonal();
    test_similarity_antiparallel();
    test_similarity_symmetry();

    // --- 初始化 GPU ---
    g_ocl = OpenClLoader::Get();
    if (g_ocl) {
        g_gpu_ready = g_gpu.Init(g_kernel_dir, g_ocl);
        if (!g_gpu_ready)
            fprintf(stderr, "[WARN] FacePreprocess Init failed, GPU tests will be skipped\n");
    } else {
        fprintf(stderr, "[WARN] OpenClLoader failed, GPU tests will be skipped\n");
    }

    // --- 初始化 QNN ---
    g_qnn_ready = g_arc.Init("/usr/lib/libQnnHtp.so",
                              "/usr/lib/libQnnSystem.so",
                              g_model_path);
    if (!g_qnn_ready)
        fprintf(stderr, "[WARN] ArcFaceInfer Init failed, QNN tests will be skipped\n");

    // --- Group B: QNN 模型验证 ---
    test_arcface_init();
    test_arcface_execute_zero();
    test_arcface_output_unit_norm();

    // --- Group C: 完整链路 ---
    test_full_pipeline();
    test_determinism();

    // --- Benchmarks ---
    bench_qnn_execute();
    bench_full_pipeline();

    // --- 清理 ---
    g_arc.Destroy();
    g_gpu.Destroy();

    // --- 汇总 ---
    fprintf(stderr, "\n=== Results: %d PASS / %d FAIL ===\n", g_pass, g_fail);
    if (g_fail == 0)
        fprintf(stderr, "ALL TESTS PASSED\n");
    else
        fprintf(stderr, "SOME TESTS FAILED\n");
    return (g_fail == 0) ? 0 : 1;
}
