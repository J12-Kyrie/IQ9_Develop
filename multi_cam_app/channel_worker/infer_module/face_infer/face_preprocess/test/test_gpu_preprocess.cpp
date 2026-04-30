/// @file test_gpu_preprocess.cpp
/// @brief FacePreprocess 独立测试程序 (QCS9075 板上运行)
///
/// 7 个测试用例 + 2 个性能基准:
///   #1  SCRFD letterbox 尺寸 + 填充区
///   #2  SCRFD 归一化精度 (GPU vs CPU)
///   #3  BT.601 已知值 (5 组纯色)
///   #4  ArcFace 恒等变换
///   #5  ArcFace 平移变换
///   #6  ArcFace 边界处理
///   #7  SCRFD 性能基准
///   #8  ArcFace 性能基准
///
/// 依赖: Step 1 mem_management (OpenClLoader, DmaBuffer, DmaSyncGuard)
/// 编译: bash build_test.sh (链接 opencl_loader.cpp + dma_buffer.cpp)
/// 运行: ./test_gpu_preprocess [kernel_dir]

#include "../FacePreprocess.hpp"
#include "../../mem_management/opencl_loader.hpp"
#include "../../mem_management/dma_buffer.hpp"
#include "../../mem_management/dma_sync_guard.hpp"
#include "../../mem_management/mem_types.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace face_infer;

// ============================================================
// 测试框架 (与 test_dma_buffer.cpp 相同模式)
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
static FacePreprocess g_gpu;
static const char* g_kernel_dir = nullptr;

// ============================================================
// NV12 测试图案生成器
// ============================================================

/// 水平渐变 NV12: Y 从左到右 0→255, U 上到下渐变, V=128
static void fill_nv12_gradient(uint8_t* buf, uint32_t w, uint32_t h, uint32_t stride) {
    // Y plane
    for (uint32_t r = 0; r < h; r++)
        for (uint32_t c = 0; c < w; c++)
            buf[r * stride + c] = (uint8_t)(c * 255 / (w > 1 ? w - 1 : 1));
    // UV plane (NV12: interleaved U V)
    uint8_t* uv = buf + h * stride;
    for (uint32_t r = 0; r < h / 2; r++)
        for (uint32_t c = 0; c < w / 2; c++) {
            uv[r * stride + c * 2 + 0] = (uint8_t)(r * 255 / (h / 2 > 1 ? h / 2 - 1 : 1)); // U
            uv[r * stride + c * 2 + 1] = 128;                                                  // V
        }
}

/// 纯色 NV12
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

// ============================================================
// CPU 参考实现 (Ground Truth)
// 移植自 facealign.c, 修正了 line 109 双线性 bug
// ============================================================

/// CPU Y 平面双线性插值 (修正版)
static float cpu_sample_y(const uint8_t* y_plane, uint32_t y_stride,
                          uint32_t w, uint32_t h, float x, float y) {
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = std::clamp(x0, 0, (int)w - 1);
    y0 = std::clamp(y0, 0, (int)h - 1);
    x1 = std::clamp(x1, 0, (int)w - 1);
    y1 = std::clamp(y1, 0, (int)h - 1);
    float fx = x - floorf(x), fy = y - floorf(y);
    float v00 = y_plane[y0 * y_stride + x0];
    float v01 = y_plane[y0 * y_stride + x1];
    float v10 = y_plane[y1 * y_stride + x0];
    float v11 = y_plane[y1 * y_stride + x1];
    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01)
         + fy * ((1.0f - fx) * v10 + fx * v11);
}

/// CPU UV 平面双线性插值 (匹配 facealign.c:113-151)
static void cpu_sample_uv(const uint8_t* uv_plane, uint32_t uv_stride,
                          uint32_t w, uint32_t h, float x, float y,
                          float* u_out, float* v_out) {
    float ux = x * 0.5f, uy = y * 0.5f;
    int ux0 = (int)floorf(ux), uy0 = (int)floorf(uy);
    int ux1 = ux0 + 1, uy1 = uy0 + 1;
    ux0 = std::clamp(ux0, 0, (int)(w / 2) - 1);
    uy0 = std::clamp(uy0, 0, (int)(h / 2) - 1);
    ux1 = std::clamp(ux1, 0, (int)(w / 2) - 1);
    uy1 = std::clamp(uy1, 0, (int)(h / 2) - 1);
    float ufx = ux - floorf(ux), ufy = uy - floorf(uy);

    float u00 = uv_plane[uy0 * uv_stride + ux0 * 2];
    float u01 = uv_plane[uy0 * uv_stride + ux1 * 2];
    float u10 = uv_plane[uy1 * uv_stride + ux0 * 2];
    float u11 = uv_plane[uy1 * uv_stride + ux1 * 2];
    *u_out = (1 - ufy) * ((1 - ufx) * u00 + ufx * u01)
           + ufy * ((1 - ufx) * u10 + ufx * u11);

    float v00 = uv_plane[uy0 * uv_stride + ux0 * 2 + 1];
    float v01 = uv_plane[uy0 * uv_stride + ux1 * 2 + 1];
    float v10 = uv_plane[uy1 * uv_stride + ux0 * 2 + 1];
    float v11 = uv_plane[uy1 * uv_stride + ux1 * 2 + 1];
    *v_out = (1 - ufy) * ((1 - ufx) * v00 + ufx * v01)
           + ufy * ((1 - ufx) * v10 + ufx * v11);
}

/// CPU BT.601 limited-range (参考 facealign.c:624-627)
static void cpu_bt601(float Y, float U, float V, float* R, float* G, float* B) {
    *R = std::clamp(1.164f * (Y - 16.0f) + 1.596f * (V - 128.0f), 0.0f, 255.0f);
    *G = std::clamp(1.164f * (Y - 16.0f) - 0.813f * (V - 128.0f) - 0.391f * (U - 128.0f), 0.0f, 255.0f);
    *B = std::clamp(1.164f * (Y - 16.0f) + 2.018f * (U - 128.0f), 0.0f, 255.0f);
}

/// CPU SCRFD 完整预处理 (letterbox + BT.601 + normalize → NCHW)
static void cpu_scrfd_preprocess(const uint8_t* nv12, uint32_t w, uint32_t h, uint32_t stride,
                                 float* output, int new_w, int new_h, float scale) {
    const int DST = 640;
    const int plane_size = DST * DST;
    const uint8_t* y_plane = nv12;
    const uint8_t* uv_plane = nv12 + h * stride;

    for (int dy = 0; dy < DST; dy++) {
        for (int dx = 0; dx < DST; dx++) {
            int idx = dy * DST + dx;

            if (dx >= new_w || dy >= new_h) {
                output[0 * plane_size + idx] = -0.99609375f;
                output[1 * plane_size + idx] = -0.99609375f;
                output[2 * plane_size + idx] = -0.99609375f;
                continue;
            }

            float src_x = (dx + 0.5f) * scale - 0.5f;
            float src_y = (dy + 0.5f) * scale - 0.5f;
            src_x = std::clamp(src_x, 0.0f, (float)(w - 1));
            src_y = std::clamp(src_y, 0.0f, (float)(h - 1));

            float Y = cpu_sample_y(y_plane, stride, w, h, src_x, src_y);
            float U, V;
            cpu_sample_uv(uv_plane, stride, w, h, src_x, src_y, &U, &V);

            float R, G, B;
            cpu_bt601(Y, U, V, &R, &G, &B);

            output[0 * plane_size + idx] = (R - 127.5f) / 128.0f;
            output[1 * plane_size + idx] = (G - 127.5f) / 128.0f;
            output[2 * plane_size + idx] = (B - 127.5f) / 128.0f;
        }
    }
}

/// CPU ArcFace 完整预处理 (逆仿射 + BT.601 + normalize → NCHW)
static void cpu_arcface_preprocess(const uint8_t* nv12, uint32_t w, uint32_t h, uint32_t stride,
                                   float* output, const float inv[6]) {
    const int DST = 112;
    const int plane_size = DST * DST;
    const uint8_t* y_plane = nv12;
    const uint8_t* uv_plane = nv12 + h * stride;

    for (int dy = 0; dy < DST; dy++) {
        for (int dx = 0; dx < DST; dx++) {
            int idx = dy * DST + dx;

            float sx = inv[0] * (float)dx + inv[1] * (float)dy + inv[2];
            float sy = inv[3] * (float)dx + inv[4] * (float)dy + inv[5];

            if (sx < 0.0f || sx >= (float)(w - 1) || sy < 0.0f || sy >= (float)(h - 1)) {
                output[0 * plane_size + idx] = -1.0f;
                output[1 * plane_size + idx] = -1.0f;
                output[2 * plane_size + idx] = -1.0f;
                continue;
            }

            float Y = cpu_sample_y(y_plane, stride, w, h, sx, sy);
            float U, V;
            cpu_sample_uv(uv_plane, stride, w, h, sx, sy, &U, &V);

            float R, G, B;
            cpu_bt601(Y, U, V, &R, &G, &B);

            output[0 * plane_size + idx] = R / 127.5f - 1.0f;
            output[1 * plane_size + idx] = G / 127.5f - 1.0f;
            output[2 * plane_size + idx] = B / 127.5f - 1.0f;
        }
    }
}

// ============================================================
// 辅助: 分配 NV12 DmaBuffer 并绑定到 GPU context
// ============================================================
struct NV12Frame {
    DmaBuffer buf;
    FramePlaneInfo plane;

    bool Alloc(uint32_t w, uint32_t h) {
        uint32_t stride = w;
        uint32_t nv12_size = stride * h * 3 / 2;

        if (!buf.Init(nv12_size)) return false;
        if (!buf.BindOpenCl(g_gpu.GetContext(), g_ocl)) {
            buf.Destroy();
            return false;
        }

        plane.fd = buf.fd();
        plane.frame_len = nv12_size;
        plane.y_offset = 0;
        plane.uv_offset = stride * h;
        plane.y_stride = stride;
        plane.uv_stride = stride;
        plane.width = w;
        plane.height = h;
        return true;
    }

    uint8_t* data() { return static_cast<uint8_t*>(buf.data()); }
    ::cl_mem mem_handle() { return buf.cl_mem_handle(); }

    void Destroy() { buf.Destroy(); }
};

/// 比较 GPU 和 CPU 输出, 返回最大绝对误差
static float compare_outputs(const float* gpu, const float* cpu, int count,
                             int* max_err_idx = nullptr) {
    float max_err = 0.0f;
    for (int i = 0; i < count; i++) {
        float err = fabsf(gpu[i] - cpu[i]);
        if (err > max_err) {
            max_err = err;
            if (max_err_idx) *max_err_idx = i;
        }
    }
    return max_err;
}

// ============================================================
// #1 SCRFD letterbox 尺寸 + 填充区
// ============================================================
static void test_1_scrfd_letterbox_dims() {
    TEST_BEGIN("#1 SCRFD letterbox dims+padding");

    NV12Frame frame;
    CHECK(frame.Alloc(1920, 1080), "NV12Frame alloc 1920x1080");
    fill_nv12_solid(frame.data(), 1920, 1080, 1920, 128, 128, 128);

    float scale;
    int new_w, new_h;
    const float* result = g_gpu.RunScrfd(frame.mem_handle(), frame.plane,
                                          &scale, &new_w, &new_h);
    CHECK(result != nullptr, "RunScrfd should succeed");

    // 1920x1080: im_ratio=0.5625, new_w=640, new_h=360, scale=3.0
    CHECK(new_w == 640, "new_w should be 640");
    CHECK(new_h == 360, "new_h should be 360");
    CHECK(fabsf(scale - 3.0f) < 0.001f, "scale should be 3.0");

    // 填充区 (row 360-639) 应为 -0.99609375
    const float pad_val = -0.99609375f;
    const int plane_size = 640 * 640;
    bool pad_ok = true;
    for (int dy = 360; dy < 640; dy += 40) {
        for (int dx = 0; dx < 640; dx += 40) {
            int idx = dy * 640 + dx;
            for (int c = 0; c < 3; c++) {
                if (fabsf(result[c * plane_size + idx] - pad_val) > 1e-5f) {
                    fprintf(stderr, "  pad mismatch at (%d,%d) c=%d: got %f expect %f\n",
                            dx, dy, c, result[c * plane_size + idx], pad_val);
                    pad_ok = false;
                    break;
                }
            }
            if (!pad_ok) break;
        }
        if (!pad_ok) break;
    }
    CHECK(pad_ok, "padding region should be -0.99609375");

    // 有效区 (Y=128, U=128, V=128) → BT.601 → R=G=B≈130.1 → SCRFD norm ≈ 0.02031
    float expected_R, expected_G, expected_B;
    cpu_bt601(128.0f, 128.0f, 128.0f, &expected_R, &expected_G, &expected_B);
    float expected_norm = (expected_R - 127.5f) / 128.0f;
    int center_idx = 180 * 640 + 320;  // center of valid area
    float got = result[0 * plane_size + center_idx];
    CHECK(fabsf(got - expected_norm) < 0.02f,
          "center pixel R channel should match BT.601(128,128,128)");

    frame.Destroy();
    TEST_PASS("#1 SCRFD letterbox dims+padding");
}

// ============================================================
// #2 SCRFD 归一化精度 (GPU vs CPU)
// ============================================================
static void test_2_scrfd_norm_accuracy() {
    TEST_BEGIN("#2 SCRFD normalization accuracy");

    NV12Frame frame;
    CHECK(frame.Alloc(1920, 1080), "NV12Frame alloc 1920x1080");
    fill_nv12_gradient(frame.data(), 1920, 1080, 1920);

    float scale;
    int new_w, new_h;
    const float* gpu_out = g_gpu.RunScrfd(frame.mem_handle(), frame.plane,
                                           &scale, &new_w, &new_h);
    CHECK(gpu_out != nullptr, "RunScrfd should succeed");

    // CPU 参考
    const int total = FacePreprocess::kScrfdOutputFloats;
    std::vector<float> cpu_out(total);
    cpu_scrfd_preprocess(frame.data(), 1920, 1080, 1920,
                         cpu_out.data(), new_w, new_h, scale);

    int max_idx = 0;
    float max_err = compare_outputs(gpu_out, cpu_out.data(), total, &max_idx);
    fprintf(stderr, "  max error: %.6f at index %d (gpu=%.6f cpu=%.6f)\n",
            max_err, max_idx, gpu_out[max_idx], cpu_out[max_idx]);

    // 容差 ±1/128 ≈ 0.0078 (8-bit 精度下的浮点舍入)
    CHECK(max_err < 0.01f, "max error should be < 0.01 (tolerance 1/128)");

    frame.Destroy();
    TEST_PASS("#2 SCRFD normalization accuracy");
}

// ============================================================
// #3 BT.601 已知值 (5 组纯色)
// ============================================================
static void test_3_bt601_known_values() {
    TEST_BEGIN("#3 BT.601 known values");

    struct KnownYUV {
        uint8_t y, u, v;
        float exp_r, exp_g, exp_b;
        const char* name;
    };

    // 预计算期望 RGB (BT.601 limited-range)
    KnownYUV cases[] = {
        {235, 128, 128, 255.0f, 255.0f, 255.0f, "white"},
        { 16, 128, 128,   0.0f,   0.0f,   0.0f, "black"},
        { 81,  90, 240,   0.0f,   0.0f,   0.0f, "red"},    // 值在下面重新计算
        {145,  54,  34,   0.0f,   0.0f,   0.0f, "green"},
        { 41, 240, 110,   0.0f,   0.0f,   0.0f, "blue"},
    };

    // 用 CPU BT.601 计算精确期望值
    for (auto& c : cases) {
        cpu_bt601((float)c.y, (float)c.u, (float)c.v, &c.exp_r, &c.exp_g, &c.exp_b);
    }

    NV12Frame frame;
    CHECK(frame.Alloc(1920, 1080), "NV12Frame alloc 1920x1080");

    const int plane_size = 640 * 640;
    bool all_ok = true;

    for (const auto& c : cases) {
        fill_nv12_solid(frame.data(), 1920, 1080, 1920, c.y, c.u, c.v);

        float scale;
        int new_w, new_h;
        const float* gpu_out = g_gpu.RunScrfd(frame.mem_handle(), frame.plane,
                                               &scale, &new_w, &new_h);
        if (!gpu_out) {
            fprintf(stderr, "  FAIL: RunScrfd failed for %s\n", c.name);
            all_ok = false;
            continue;
        }

        // SCRFD 归一化: (pixel - 127.5) / 128.0
        float exp_r_norm = (c.exp_r - 127.5f) / 128.0f;
        float exp_g_norm = (c.exp_g - 127.5f) / 128.0f;
        float exp_b_norm = (c.exp_b - 127.5f) / 128.0f;

        // 检查有效区域中心像素 (所有像素同色, 取几个采样点)
        int sample_points[] = { 100 * 640 + 100, 200 * 640 + 300, 50 * 640 + 500 };
        for (int idx : sample_points) {
            float err_r = fabsf(gpu_out[0 * plane_size + idx] - exp_r_norm);
            float err_g = fabsf(gpu_out[1 * plane_size + idx] - exp_g_norm);
            float err_b = fabsf(gpu_out[2 * plane_size + idx] - exp_b_norm);
            float max_ch_err = std::max({err_r, err_g, err_b});

            // 容差 ±2/128 ≈ 0.016
            if (max_ch_err > 0.02f) {
                fprintf(stderr, "  FAIL: %s at idx=%d: err_r=%.4f err_g=%.4f err_b=%.4f\n",
                        c.name, idx, err_r, err_g, err_b);
                fprintf(stderr, "    gpu: R=%.4f G=%.4f B=%.4f\n",
                        gpu_out[0 * plane_size + idx],
                        gpu_out[1 * plane_size + idx],
                        gpu_out[2 * plane_size + idx]);
                fprintf(stderr, "    exp: R=%.4f G=%.4f B=%.4f\n",
                        exp_r_norm, exp_g_norm, exp_b_norm);
                all_ok = false;
            }
        }
    }

    CHECK(all_ok, "all 5 BT.601 known values should match");

    frame.Destroy();
    TEST_PASS("#3 BT.601 known values");
}

// ============================================================
// #4 ArcFace 恒等变换
// ============================================================
static void test_4_arcface_identity() {
    TEST_BEGIN("#4 ArcFace identity transform");

    NV12Frame frame;
    CHECK(frame.Alloc(1920, 1080), "NV12Frame alloc 1920x1080");
    fill_nv12_gradient(frame.data(), 1920, 1080, 1920);

    // 恒等变换: dst → src identity [1,0,0, 0,1,0]
    float inv_affine[6] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
    const float* gpu_out = g_gpu.RunArcface(frame.mem_handle(), frame.plane, inv_affine);
    CHECK(gpu_out != nullptr, "RunArcface should succeed");

    // CPU 参考
    const int total = FacePreprocess::kArcfaceOutputFloats;
    std::vector<float> cpu_out(total);
    cpu_arcface_preprocess(frame.data(), 1920, 1080, 1920,
                           cpu_out.data(), inv_affine);

    int max_idx = 0;
    float max_err = compare_outputs(gpu_out, cpu_out.data(), total, &max_idx);
    fprintf(stderr, "  max error: %.6f at index %d\n", max_err, max_idx);

    // 容差 ±1/127.5 ≈ 0.0078
    CHECK(max_err < 0.01f, "max error should be < 0.01");

    frame.Destroy();
    TEST_PASS("#4 ArcFace identity transform");
}

// ============================================================
// #5 ArcFace 平移变换
// ============================================================
static void test_5_arcface_translation() {
    TEST_BEGIN("#5 ArcFace translation");

    NV12Frame frame;
    CHECK(frame.Alloc(1920, 1080), "NV12Frame alloc 1920x1080");
    fill_nv12_gradient(frame.data(), 1920, 1080, 1920);

    // 平移变换: dst pixel (dx,dy) → src (dx+500, dy+300)
    float inv_affine[6] = { 1.0f, 0.0f, 500.0f, 0.0f, 1.0f, 300.0f };
    const float* gpu_out = g_gpu.RunArcface(frame.mem_handle(), frame.plane, inv_affine);
    CHECK(gpu_out != nullptr, "RunArcface should succeed");

    // CPU 参考
    const int total = FacePreprocess::kArcfaceOutputFloats;
    std::vector<float> cpu_out(total);
    cpu_arcface_preprocess(frame.data(), 1920, 1080, 1920,
                           cpu_out.data(), inv_affine);

    int max_idx = 0;
    float max_err = compare_outputs(gpu_out, cpu_out.data(), total, &max_idx);
    fprintf(stderr, "  max error: %.6f at index %d\n", max_err, max_idx);

    CHECK(max_err < 0.01f, "max error should be < 0.01");

    frame.Destroy();
    TEST_PASS("#5 ArcFace translation");
}

// ============================================================
// #6 ArcFace 边界处理
// ============================================================
static void test_6_arcface_boundary() {
    TEST_BEGIN("#6 ArcFace boundary handling");

    NV12Frame frame;
    CHECK(frame.Alloc(256, 256), "NV12Frame alloc 256x256");
    fill_nv12_gradient(frame.data(), 256, 256, 256);

    // 平移: dst → src (dx+200, dy+200), 所以 dx>=55 或 dy>=55 时 src >= 255 → 越界
    float inv_affine[6] = { 1.0f, 0.0f, 200.0f, 0.0f, 1.0f, 200.0f };
    const float* gpu_out = g_gpu.RunArcface(frame.mem_handle(), frame.plane, inv_affine);
    CHECK(gpu_out != nullptr, "RunArcface should succeed");

    const int DST = 112;
    const int plane_size = DST * DST;

    // 越界像素应为 (-1, -1, -1)
    bool boundary_ok = true;
    for (int dy = 56; dy < DST; dy += 10) {
        for (int dx = 56; dx < DST; dx += 10) {
            int idx = dy * DST + dx;
            for (int c = 0; c < 3; c++) {
                if (fabsf(gpu_out[c * plane_size + idx] - (-1.0f)) > 1e-5f) {
                    fprintf(stderr, "  boundary fail at (%d,%d) c=%d: got %f expect -1.0\n",
                            dx, dy, c, gpu_out[c * plane_size + idx]);
                    boundary_ok = false;
                }
            }
        }
    }
    CHECK(boundary_ok, "out-of-bounds pixels should be -1.0");

    // 有效像素 (dx<55 && dy<55) 应匹配 CPU 参考
    const int total = FacePreprocess::kArcfaceOutputFloats;
    std::vector<float> cpu_out(total);
    cpu_arcface_preprocess(frame.data(), 256, 256, 256, cpu_out.data(), inv_affine);

    float max_err = 0.0f;
    for (int dy = 0; dy < 55; dy++) {
        for (int dx = 0; dx < 55; dx++) {
            int idx = dy * DST + dx;
            for (int c = 0; c < 3; c++) {
                float err = fabsf(gpu_out[c * plane_size + idx] - cpu_out[c * plane_size + idx]);
                max_err = std::max(max_err, err);
            }
        }
    }
    fprintf(stderr, "  valid region max error: %.6f\n", max_err);
    CHECK(max_err < 0.01f, "valid pixels should match CPU reference");

    frame.Destroy();
    TEST_PASS("#6 ArcFace boundary handling");
}

// ============================================================
// #7 SCRFD 性能基准
// ============================================================
static void bench_scrfd(int iterations) {
    fprintf(stderr, "\n--- Bench: SCRFD 1920x1080 (%d iterations) ---\n", iterations);

    NV12Frame frame;
    if (!frame.Alloc(1920, 1080)) {
        fprintf(stderr, "  SKIP: NV12Frame alloc failed\n");
        return;
    }
    fill_nv12_gradient(frame.data(), 1920, 1080, 1920);

    // Warmup
    float scale;
    int new_w, new_h;
    g_gpu.RunScrfd(frame.mem_handle(), frame.plane, &scale, &new_w, &new_h);

    double total_ms = 0.0;
    double min_ms = 1e9, max_ms = 0.0;

    for (int i = 0; i < iterations; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        g_gpu.RunScrfd(frame.mem_handle(), frame.plane, &scale, &new_w, &new_h);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    fprintf(stderr, "  [PERF] SCRFD 1920x1080: avg=%.3f min=%.3f max=%.3f ms (%d iters)\n",
            total_ms / iterations, min_ms, max_ms, iterations);

    frame.Destroy();
}

// ============================================================
// #8 ArcFace 性能基准
// ============================================================
static void bench_arcface(int iterations) {
    fprintf(stderr, "\n--- Bench: ArcFace 112x112 (%d iterations) ---\n", iterations);

    NV12Frame frame;
    if (!frame.Alloc(1920, 1080)) {
        fprintf(stderr, "  SKIP: NV12Frame alloc failed\n");
        return;
    }
    fill_nv12_gradient(frame.data(), 1920, 1080, 1920);

    float inv_affine[6] = { 1.0f, 0.0f, 500.0f, 0.0f, 1.0f, 300.0f };

    // Warmup
    g_gpu.RunArcface(frame.mem_handle(), frame.plane, inv_affine);

    double total_ms = 0.0;
    double min_ms = 1e9, max_ms = 0.0;

    for (int i = 0; i < iterations; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        g_gpu.RunArcface(frame.mem_handle(), frame.plane, inv_affine);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    fprintf(stderr, "  [PERF] ArcFace 112x112: avg=%.3f min=%.3f max=%.3f ms (%d iters)\n",
            total_ms / iterations, min_ms, max_ms, iterations);

    frame.Destroy();
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    fprintf(stderr, "=== FacePreprocess Test Suite ===\n");

    g_kernel_dir = (argc > 1) ? argv[1] : "../kernels";
    fprintf(stderr, "Kernel directory: %s\n", g_kernel_dir);

    // 初始化 OpenCL 加载器
    g_ocl = std::make_shared<OpenClLoader>();
    if (!g_ocl->Init()) {
        fprintf(stderr, "FATAL: OpenClLoader::Init() failed\n");
        return 1;
    }

    // 初始化 FacePreprocess (包含 context/queue/kernels/output buffers)
    if (!g_gpu.Init(g_kernel_dir, g_ocl)) {
        fprintf(stderr, "FATAL: FacePreprocess::Init() failed\n");
        return 1;
    }

    // 运行全部测试
    test_1_scrfd_letterbox_dims();
    test_2_scrfd_norm_accuracy();
    test_3_bt601_known_values();
    test_4_arcface_identity();
    test_5_arcface_translation();
    test_6_arcface_boundary();

    // 性能基准
    bench_scrfd(100);
    bench_arcface(100);

    // 清理
    g_gpu.Destroy();
    g_ocl.reset();

    fprintf(stderr, "\n=== Results: %d PASS, %d FAIL ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
