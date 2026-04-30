/// @file test_inverse_affine.cpp
/// @brief Umeyama 逆仿射 独立测试 (QCS9075 板上运行)
///
/// 6 个测试用例:
///   #1  test_identity_landmarks       — 参考点自身 → 恒等变换
///   #2  test_scaled_landmarks         — 2× 缩放+平移 → 验证 round-trip
///   #3  test_realistic_landmarks      — 模拟真实检测 → 数值合理性
///   #4  test_forward_inverse_roundtrip — 正向→逆向 round-trip 误差 < 0.01px
///   #5  test_arcface_preprocess_integration — inv + RunArcface → 有效输出
///   #6  test_degenerate_landmarks     — 退化输入不崩溃
///
/// 编译: bash build_test_affine.sh
/// 运行: ./test_inverse_affine [kernel_dir]

#include "../inverse_affine.hpp"
#include "../FacePreprocess.hpp"
#include "../../mem_management/opencl_loader.hpp"
#include "../../mem_management/dma_buffer.hpp"
#include "../../mem_management/dma_sync_guard.hpp"
#include "../../mem_management/mem_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace face_infer;

// ============================================================
// 测试框架 (与 test_gpu_preprocess.cpp 相同模式)
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
// 全局共享对象 (Test 5 的 RunArcface 集成测试需要)
// ============================================================
static std::shared_ptr<OpenClLoader> g_ocl;
static FacePreprocess g_gpu;
static const char* g_kernel_dir = nullptr;

// 测试帧参数
static constexpr uint32_t kTestWidth  = 1920;
static constexpr uint32_t kTestHeight = 1080;
static constexpr uint32_t kTestStride = 1920;

// ============================================================
// 辅助函数
// ============================================================

/// 检查 float 是否有限 (非 NaN/Inf)
static bool is_finite(float v) {
    return std::isfinite(v);
}

/// 打印 inv[6] 矩阵
static void print_inv(const float inv[6]) {
    fprintf(stderr, "  inv = {%.6f, %.6f, %.6f, %.6f, %.6f, %.6f}\n",
            inv[0], inv[1], inv[2], inv[3], inv[4], inv[5]);
}

/// 应用逆仿射: dst(dx,dy) → src(sx,sy)
static void apply_inv(const float inv[6], float dx, float dy,
                      float* sx, float* sy) {
    *sx = inv[0] * dx + inv[1] * dy + inv[2];
    *sy = inv[3] * dx + inv[4] * dy + inv[5];
}

// ============================================================
// Test 1: 恒等变换 — 输入 = 参考点本身
// ============================================================
static void test_identity_landmarks() {
    TEST_BEGIN("test_identity_landmarks");

    float src[5][2];
    for (int i = 0; i < 5; i++) {
        src[i][0] = kArcfaceRefLandmarks[i][0];
        src[i][1] = kArcfaceRefLandmarks[i][1];
    }

    float inv[6];
    ComputeInverseAffine(src, inv);
    print_inv(inv);

    // 恒等: inv ≈ {1, 0, 0, 0, 1, 0}
    float eps = 1e-4f;
    fprintf(stderr, "  Expected: {1, 0, 0, 0, 1, 0}\n");
    CHECK(std::fabs(inv[0] - 1.0f) < eps, "inv[0] (m00) should be ~1.0");
    CHECK(std::fabs(inv[1]) < eps,         "inv[1] (m01) should be ~0.0");
    CHECK(std::fabs(inv[2]) < eps,         "inv[2] (m02) should be ~0.0");
    CHECK(std::fabs(inv[3]) < eps,         "inv[3] (m10) should be ~0.0");
    CHECK(std::fabs(inv[4] - 1.0f) < eps, "inv[4] (m11) should be ~1.0");
    CHECK(std::fabs(inv[5]) < eps,         "inv[5] (m12) should be ~0.0");

    // 验证 round-trip: 参考点 → 逆映射 → 应回到参考点自身
    for (int i = 0; i < 5; i++) {
        float sx, sy;
        apply_inv(inv, kArcfaceRefLandmarks[i][0], kArcfaceRefLandmarks[i][1], &sx, &sy);
        float dx = std::fabs(sx - src[i][0]);
        float dy = std::fabs(sy - src[i][1]);
        CHECK(dx < 0.01f && dy < 0.01f,
              "Identity round-trip error too large");
    }

    TEST_PASS("test_identity_landmarks");
}

// ============================================================
// Test 2: 缩放+平移 — 参考点 × 2 + (100, 50)
// ============================================================
static void test_scaled_landmarks() {
    TEST_BEGIN("test_scaled_landmarks");

    float src[5][2];
    float offset_x = 100.0f, offset_y = 50.0f;
    for (int i = 0; i < 5; i++) {
        src[i][0] = kArcfaceRefLandmarks[i][0] * 2.0f + offset_x;
        src[i][1] = kArcfaceRefLandmarks[i][1] * 2.0f + offset_y;
    }

    float inv[6];
    ComputeInverseAffine(src, inv);
    print_inv(inv);

    // 正向 scale ≈ 0.5 (从 src → 112×112)
    // 逆向 scale ≈ 2.0 (从 112×112 → src)
    fprintf(stderr, "  Expected: inv[0] ≈ 2.0, inv[1] ≈ 0, inv[4] ≈ 2.0\n");
    CHECK(std::fabs(inv[0] - 2.0f) < 0.01f, "inv[0] should be ~2.0 (2× scale)");
    CHECK(std::fabs(inv[1]) < 0.1f,          "inv[1] should be ~0 (no rotation)");
    CHECK(std::fabs(inv[4] - 2.0f) < 0.01f, "inv[4] should be ~2.0 (2× scale)");

    // round-trip: 对每个参考点 (dx,dy), 逆映射应回到 src[i]
    for (int i = 0; i < 5; i++) {
        float sx, sy;
        apply_inv(inv, kArcfaceRefLandmarks[i][0], kArcfaceRefLandmarks[i][1], &sx, &sy);
        float err_x = std::fabs(sx - src[i][0]);
        float err_y = std::fabs(sy - src[i][1]);
        fprintf(stderr, "  ref[%d] → src: (%.2f,%.2f) expected (%.2f,%.2f) err=(%.4f,%.4f)\n",
                i, sx, sy, src[i][0], src[i][1], err_x, err_y);
        CHECK(err_x < 0.01f && err_y < 0.01f,
              "Scaled round-trip error too large");
    }

    TEST_PASS("test_scaled_landmarks");
}

// ============================================================
// Test 3: 真实场景 — 模拟 1920×1080 中的正脸
// ============================================================
static void test_realistic_landmarks() {
    TEST_BEGIN("test_realistic_landmarks");

    float src[5][2] = {
        {400.0f, 270.0f},   // left eye
        {600.0f, 270.0f},   // right eye
        {500.0f, 350.0f},   // nose
        {420.0f, 460.0f},   // left mouth
        {580.0f, 460.0f}    // right mouth
    };

    float inv[6];
    ComputeInverseAffine(src, inv);
    print_inv(inv);

    // 基本合理性检查
    for (int i = 0; i < 6; i++) {
        CHECK(is_finite(inv[i]), "inv[i] must be finite");
    }
    CHECK(inv[0] > 0, "inv[0] (scale x) must be positive");
    CHECK(inv[4] > 0, "inv[4] (scale y) must be positive");

    // 参考点鼻子 (56.0, 71.7) 逆映射应在原图范围内
    float sx, sy;
    apply_inv(inv, 56.0f, 71.7f, &sx, &sy);
    fprintf(stderr, "  Nose ref (56.0, 71.7) → src (%.1f, %.1f)\n", sx, sy);
    CHECK(sx >= 0 && sx <= 1920, "Nose x must be in [0, 1920]");
    CHECK(sy >= 0 && sy <= 1080, "Nose y must be in [0, 1080]");

    // 鼻子逆映射应接近 src 鼻子 (500, 350)
    CHECK(std::fabs(sx - 500.0f) < 20.0f, "Nose x should be near 500");
    CHECK(std::fabs(sy - 350.0f) < 20.0f, "Nose y should be near 350");

    // 打印所有参考点的逆映射
    for (int i = 0; i < 5; i++) {
        apply_inv(inv, kArcfaceRefLandmarks[i][0], kArcfaceRefLandmarks[i][1], &sx, &sy);
        fprintf(stderr, "  ref[%d] (%.1f, %.1f) → src (%.1f, %.1f) [expected (%.0f, %.0f)]\n",
                i, kArcfaceRefLandmarks[i][0], kArcfaceRefLandmarks[i][1],
                sx, sy, src[i][0], src[i][1]);
    }

    TEST_PASS("test_realistic_landmarks");
}

// ============================================================
// Test 4: round-trip 精度 — 正向→逆向 < 0.01px
// ============================================================
static void test_forward_inverse_roundtrip() {
    TEST_BEGIN("test_forward_inverse_roundtrip");

    float src[5][2] = {
        {400.0f, 270.0f},
        {600.0f, 270.0f},
        {500.0f, 350.0f},
        {420.0f, 460.0f},
        {580.0f, 460.0f}
    };

    // 计算正向变换参数 (a, b, tx, ty)
    // 复制 Umeyama 正向计算
    double sx_c = 0, sy_c = 0, dx_c = 0, dy_c = 0;
    for (int i = 0; i < 5; i++) {
        sx_c += src[i][0]; sy_c += src[i][1];
        dx_c += kArcfaceRefLandmarks[i][0]; dy_c += kArcfaceRefLandmarks[i][1];
    }
    sx_c /= 5; sy_c /= 5; dx_c /= 5; dy_c /= 5;

    double num_a = 0, num_b = 0, denom = 0;
    for (int i = 0; i < 5; i++) {
        double sxi = src[i][0] - sx_c;
        double syi = src[i][1] - sy_c;
        double dxi = kArcfaceRefLandmarks[i][0] - dx_c;
        double dyi = kArcfaceRefLandmarks[i][1] - dy_c;
        num_a += dxi * sxi + dyi * syi;
        num_b += dxi * syi - dyi * sxi;
        denom += sxi * sxi + syi * syi;
    }
    double a = num_a / denom;
    double b = num_b / denom;
    double tx = dx_c - a * sx_c + b * sy_c;
    double ty = dy_c - b * sx_c - a * sy_c;

    fprintf(stderr, "  Forward: a=%.6f b=%.6f tx=%.6f ty=%.6f\n", a, b, tx, ty);

    // 计算逆矩阵
    float inv[6];
    ComputeInverseAffine(src, inv);
    print_inv(inv);

    // 对每个 src 点: 正向映射到 dst, 再逆向映射回来
    float max_err = 0;
    for (int i = 0; i < 5; i++) {
        // 正向: dst = [a, -b, tx; b, a, ty] * src
        double dst_x = a * src[i][0] - b * src[i][1] + tx;
        double dst_y = b * src[i][0] + a * src[i][1] + ty;

        // 逆向
        float rec_x, rec_y;
        apply_inv(inv, (float)dst_x, (float)dst_y, &rec_x, &rec_y);

        float err_x = std::fabs(rec_x - src[i][0]);
        float err_y = std::fabs(rec_y - src[i][1]);
        float err = std::max(err_x, err_y);
        max_err = std::max(max_err, err);

        fprintf(stderr, "  src[%d] (%.1f,%.1f) → dst (%.2f,%.2f) → rec (%.4f,%.4f) err=%.6f\n",
                i, src[i][0], src[i][1], dst_x, dst_y, rec_x, rec_y, err);
    }

    fprintf(stderr, "  Max round-trip error: %.6f px\n", max_err);
    CHECK(max_err < 0.01f, "Round-trip error must be < 0.01 px");

    TEST_PASS("test_forward_inverse_roundtrip");
}

// ============================================================
// Test 5: RunArcface 集成 — inv + GPU warp → 有效输出
// ============================================================
static void test_arcface_preprocess_integration() {
    TEST_BEGIN("test_arcface_preprocess_integration");

    // 1. 创建 NV12 DmaBuffer, 填充渐变条纹
    uint32_t nv12_size = kTestStride * kTestHeight * 3 / 2;
    DmaBuffer buf;
    CHECK(buf.Init(nv12_size), "DmaBuffer::Init failed");
    CHECK(buf.BindOpenCl(g_gpu.GetContext(), g_ocl), "BindOpenCl failed");

    {
        DmaSyncGuard sync(buf.fd());
        uint8_t* data = static_cast<uint8_t*>(buf.data());
        // Y = (x + y) % 256 → 渐变条纹 (非纯灰色)
        for (uint32_t r = 0; r < kTestHeight; r++)
            for (uint32_t c = 0; c < kTestWidth; c++)
                data[r * kTestStride + c] = static_cast<uint8_t>((c + r) % 256);
        // UV = 128 (中性色度)
        uint8_t* uv = data + kTestHeight * kTestStride;
        for (uint32_t r = 0; r < kTestHeight / 2; r++)
            for (uint32_t c = 0; c < kTestWidth / 2; c++) {
                uv[r * kTestStride + c * 2 + 0] = 128;
                uv[r * kTestStride + c * 2 + 1] = 128;
            }
    }

    FramePlaneInfo plane{};
    plane.fd        = buf.fd();
    plane.frame_len = nv12_size;
    plane.y_offset  = 0;
    plane.uv_offset = kTestStride * kTestHeight;
    plane.y_stride  = kTestStride;
    plane.uv_stride = kTestStride;
    plane.width     = kTestWidth;
    plane.height    = kTestHeight;

    // 2. 计算逆仿射 (真实场景 landmarks)
    float src[5][2] = {
        {400.0f, 270.0f},
        {600.0f, 270.0f},
        {500.0f, 350.0f},
        {420.0f, 460.0f},
        {580.0f, 460.0f}
    };
    float inv[6];
    ComputeInverseAffine(src, inv);
    print_inv(inv);

    // 3. GPU warp
    const float* output = g_gpu.RunArcface(buf.cl_mem_handle(), plane, inv);
    CHECK(output != nullptr, "RunArcface must succeed");

    // 4. 验证输出
    int total_pixels = 112 * 112;
    int valid_count = 0;       // 非 -1.0 的像素数
    int in_range_count = 0;    // 在 [-1.0, 1.0] 范围内的像素数
    float min_val = 1e9f, max_val = -1e9f;

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < total_pixels; i++) {
            float v = output[c * total_pixels + i];
            CHECK(is_finite(v), "Output pixel must be finite");
            if (v > -0.99f) valid_count++;    // -1.0 是 out-of-bounds 值
            if (v >= -1.0f && v <= 1.0f) in_range_count++;
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
    }

    int total_elements = 3 * total_pixels;
    float valid_ratio = (float)valid_count / total_elements;
    float range_ratio = (float)in_range_count / total_elements;

    fprintf(stderr, "  Output range: [%.4f, %.4f]\n", min_val, max_val);
    fprintf(stderr, "  Valid pixels (non -1.0): %d / %d (%.1f%%)\n",
            valid_count, total_elements, valid_ratio * 100);
    fprintf(stderr, "  In-range pixels [-1,1]: %d / %d (%.1f%%)\n",
            in_range_count, total_elements, range_ratio * 100);

    CHECK(valid_ratio > 0.5f,
          "At least 50% of output pixels must be valid (not out-of-bounds)");
    CHECK(range_ratio > 0.95f,
          "At least 95% of output pixels must be in [-1.0, 1.0]");

    buf.Destroy();
    TEST_PASS("test_arcface_preprocess_integration");
}

// ============================================================
// Test 6: 退化输入 — 所有 5 点重合
// ============================================================
static void test_degenerate_landmarks() {
    TEST_BEGIN("test_degenerate_landmarks");

    float src[5][2] = {
        {500.0f, 500.0f},
        {500.0f, 500.0f},
        {500.0f, 500.0f},
        {500.0f, 500.0f},
        {500.0f, 500.0f}
    };

    float inv[6];
    ComputeInverseAffine(src, inv);
    print_inv(inv);

    // 不崩溃, 且所有值有限
    for (int i = 0; i < 6; i++) {
        CHECK(is_finite(inv[i]), "Degenerate case: inv[i] must be finite");
    }

    TEST_PASS("test_degenerate_landmarks");
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    g_kernel_dir = (argc > 1) ? argv[1] : "../kernels";

    fprintf(stderr, "=== Inverse Affine Test ===\n");
    fprintf(stderr, "  Kernel dir: %s\n", g_kernel_dir);

    // 初始化 OpenCL + FacePreprocess (Test 5 需要)
    fprintf(stderr, "\n[Init] OpenClLoader...\n");
    g_ocl = OpenClLoader::Get();
    if (!g_ocl) {
        fprintf(stderr, "FATAL: OpenClLoader::Get() failed\n");
        return 1;
    }

    fprintf(stderr, "[Init] FacePreprocess...\n");
    if (!g_gpu.Init(g_kernel_dir, g_ocl)) {
        fprintf(stderr, "FATAL: FacePreprocess::Init failed\n");
        return 1;
    }

    // 运行测试
    fprintf(stderr, "\n=== Running Tests ===\n");
    test_identity_landmarks();
    test_scaled_landmarks();
    test_realistic_landmarks();
    test_forward_inverse_roundtrip();
    test_arcface_preprocess_integration();
    test_degenerate_landmarks();

    // 清理
    fprintf(stderr, "\n[Cleanup]\n");
    g_gpu.Destroy();

    // 结果摘要
    fprintf(stderr, "\n=== Results ===\n");
    fprintf(stderr, "  PASS: %d\n", g_pass);
    fprintf(stderr, "  FAIL: %d\n", g_fail);
    fprintf(stderr, "  %s\n", (g_fail == 0) ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return (g_fail == 0) ? 0 : 1;
}
