/// @file test_dma_buffer.cpp
/// @brief DmaBuffer 独立测试程序 (QCS9075 板上运行)
///
/// 7 个测试用例:
///   #1  DMA 分配 + mmap
///   #2  OpenCL 导入
///   #3  CPU 写 → CPU 读
///   #4  GPU 写 → CPU 读 (零拷贝核心验证)
///   #5  多 buffer 独立
///   #6  Destroy 清理
///   #7  移动语义

#include "../dma_buffer.hpp"
#include "../dma_sync_guard.hpp"
#include "../opencl_loader.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <CL/cl.h>

using namespace face_infer;

// ============================================================
// 全局 OpenCL 环境 (所有 GPU 测试共享)
// ============================================================
static std::shared_ptr<OpenClLoader> g_ocl;
static cl_context      g_ctx    = nullptr;
static cl_device_id    g_dev    = nullptr;
static cl_command_queue g_queue = nullptr;
static cl_kernel       g_fill_kernel = nullptr;
static cl_program      g_fill_prog   = nullptr;

static bool setup_opencl() {
    g_ocl = std::make_shared<OpenClLoader>();
    if (!g_ocl->Init()) {
        fprintf(stderr, "  [setup] OpenClLoader::Init() failed\n");
        return false;
    }

    cl_platform_id platform;
    cl_int rc = g_ocl->GetPlatformIDs(1, &platform, nullptr);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "  [setup] clGetPlatformIDs failed: %d\n", rc);
        return false;
    }

    rc = g_ocl->GetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &g_dev, nullptr);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "  [setup] clGetDeviceIDs(GPU) failed: %d\n", rc);
        return false;
    }

    g_ctx = g_ocl->CreateContext(nullptr, 1, &g_dev, nullptr, nullptr, &rc);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "  [setup] clCreateContext failed: %d\n", rc);
        return false;
    }

    g_queue = g_ocl->CreateCommandQueue(g_ctx, g_dev, 0, &rc);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "  [setup] CreateCommandQueue failed: %d\n", rc);
        return false;
    }

    // 编译 fill kernel (硬编码, 不依赖外部 .cl 文件)
    const char* src =
        "__kernel void fill(__global float* out, float val) {\n"
        "  out[get_global_id(0)] = val;\n"
        "}\n";
    g_fill_prog = g_ocl->CreateProgramWithSource(g_ctx, 1, &src, nullptr, &rc);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "  [setup] clCreateProgramWithSource failed: %d\n", rc);
        return false;
    }

    rc = g_ocl->BuildProgram(g_fill_prog, 1, &g_dev, nullptr, nullptr, nullptr);
    if (rc != CL_SUCCESS) {
        // 打印编译错误日志
        char log[4096];
        g_ocl->GetProgramBuildInfo(g_fill_prog, g_dev, CL_PROGRAM_BUILD_LOG,
                                    sizeof(log), log, nullptr);
        fprintf(stderr, "  [setup] clBuildProgram failed: %d\n%s\n", rc, log);
        return false;
    }

    g_fill_kernel = g_ocl->CreateKernel(g_fill_prog, "fill", &rc);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "  [setup] clCreateKernel failed: %d\n", rc);
        return false;
    }

    fprintf(stderr, "  [setup] OpenCL environment ready\n");
    return true;
}

static void teardown_opencl() {
    if (g_fill_kernel) { g_ocl->ReleaseKernel(g_fill_kernel); g_fill_kernel = nullptr; }
    if (g_fill_prog)   { g_ocl->ReleaseProgram(g_fill_prog);  g_fill_prog = nullptr; }
    if (g_queue)        { g_ocl->ReleaseCommandQueue(g_queue); g_queue = nullptr; }
    if (g_ctx)          { g_ocl->ReleaseContext(g_ctx);        g_ctx = nullptr; }
    g_ocl.reset();
}

// ============================================================
// 测试辅助
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
// #1  DMA 分配 + mmap
// ============================================================
static void test_1_dma_alloc() {
    TEST_BEGIN("#1 DMA alloc + mmap");

    DmaBuffer buf;
    bool ok = buf.Init(4 * 1024 * 1024);  // 4MB
    CHECK(ok, "Init(4MB) should succeed");
    CHECK(buf.fd() >= 0, "fd should be valid");
    CHECK(buf.data() != nullptr, "data() should be non-null");
    CHECK(buf.size() == 4 * 1024 * 1024, "size should be 4MB");
    CHECK(buf.cl_mem_handle() == nullptr, "cl_mem should be null before BindOpenCl");

    buf.Destroy();
    TEST_PASS("#1 DMA alloc + mmap");
}

// ============================================================
// #2  OpenCL 导入
// ============================================================
static void test_2_opencl_bind() {
    TEST_BEGIN("#2 OpenCL bind");

    DmaBuffer buf;
    bool ok = buf.Init(1024 * sizeof(float));
    CHECK(ok, "Init should succeed");

    ok = buf.BindOpenCl(g_ctx, g_ocl);
    CHECK(ok, "BindOpenCl should succeed");
    CHECK(buf.cl_mem_handle() != nullptr, "cl_mem should be non-null after bind");

    buf.Destroy();
    TEST_PASS("#2 OpenCL bind");
}

// ============================================================
// #3  CPU 写 → CPU 读
// ============================================================
static void test_3_cpu_rw() {
    TEST_BEGIN("#3 CPU write -> CPU read");

    DmaBuffer buf;
    bool ok = buf.Init(4096);
    CHECK(ok, "Init should succeed");

    // 写入 0xAB
    memset(buf.data(), 0xAB, buf.size());

    // 验证
    const uint8_t* p = static_cast<const uint8_t*>(buf.data());
    bool all_ok = true;
    for (size_t i = 0; i < buf.size(); i++) {
        if (p[i] != 0xAB) { all_ok = false; break; }
    }
    CHECK(all_ok, "all bytes should be 0xAB");

    buf.Destroy();
    TEST_PASS("#3 CPU write -> CPU read");
}

// ============================================================
// #4  GPU 写 → CPU 读 (核心零拷贝验证)
// ============================================================
static void test_4_gpu_write_cpu_read() {
    TEST_BEGIN("#4 GPU write -> CPU read (zero-copy)");

    const size_t N = 1024;
    DmaBuffer buf;
    bool ok = buf.Init(N * sizeof(float));
    CHECK(ok, "Init should succeed");

    ok = buf.BindOpenCl(g_ctx, g_ocl);
    CHECK(ok, "BindOpenCl should succeed");

    // GPU: fill kernel 将每个 float 设为 42.0f
    cl_mem mem = buf.cl_mem_handle();
    float val = 42.0f;
    cl_int rc;
    rc = g_ocl->SetKernelArg(g_fill_kernel, 0, sizeof(cl_mem), &mem);
    CHECK(rc == CL_SUCCESS, "SetKernelArg(0) should succeed");
    rc = g_ocl->SetKernelArg(g_fill_kernel, 1, sizeof(float), &val);
    CHECK(rc == CL_SUCCESS, "SetKernelArg(1) should succeed");

    size_t global = N;
    rc = g_ocl->EnqueueNDRangeKernel(g_queue, g_fill_kernel, 1,
                                       nullptr, &global, nullptr, 0, nullptr, nullptr);
    CHECK(rc == CL_SUCCESS, "EnqueueNDRangeKernel should succeed");

    rc = g_ocl->Finish(g_queue);
    CHECK(rc == CL_SUCCESS, "clFinish should succeed");

    // CPU 零拷贝读取: DMA sync → 直接读 mmap vaddr
    {
        DmaSyncGuard sync(buf.fd());
        const float* p = static_cast<const float*>(buf.data());
        CHECK(p[0] == 42.0f, "p[0] should be 42.0f");
        CHECK(p[N / 2] == 42.0f, "p[N/2] should be 42.0f");
        CHECK(p[N - 1] == 42.0f, "p[N-1] should be 42.0f");
    }

    buf.Destroy();
    TEST_PASS("#4 GPU write -> CPU read (zero-copy)");
}

// ============================================================
// #5  多 buffer 独立
// ============================================================
static void test_5_multi_buffer() {
    TEST_BEGIN("#5 multi-buffer independence");

    const size_t BUF_SIZE = 4096;
    const uint8_t patterns[] = { 0x11, 0x22, 0x33, 0x44 };
    DmaBuffer bufs[4];

    // 创建 4 个独立 buffer
    for (int i = 0; i < 4; i++) {
        bool ok = bufs[i].Init(BUF_SIZE);
        CHECK(ok, "Init should succeed for each buffer");
    }

    // 各自写入不同 pattern
    for (int i = 0; i < 4; i++) {
        memset(bufs[i].data(), patterns[i], BUF_SIZE);
    }

    // 验证各 buffer 数据互不干扰
    for (int i = 0; i < 4; i++) {
        const uint8_t* p = static_cast<const uint8_t*>(bufs[i].data());
        bool ok = true;
        for (size_t j = 0; j < BUF_SIZE; j++) {
            if (p[j] != patterns[i]) { ok = false; break; }
        }
        char msg[64];
        snprintf(msg, sizeof(msg), "buf[%d] should contain 0x%02X", i, patterns[i]);
        CHECK(ok, msg);
    }

    for (int i = 0; i < 4; i++) bufs[i].Destroy();
    TEST_PASS("#5 multi-buffer independence");
}

// ============================================================
// #6  Destroy 清理
// ============================================================
static void test_6_destroy() {
    TEST_BEGIN("#6 Destroy cleanup");

    DmaBuffer buf;
    bool ok = buf.Init(4096);
    CHECK(ok, "Init should succeed");

    ok = buf.BindOpenCl(g_ctx, g_ocl);
    CHECK(ok, "BindOpenCl should succeed");

    // 记录 fd 用于后续检查
    int old_fd = buf.fd();
    CHECK(old_fd >= 0, "fd should be valid before Destroy");

    buf.Destroy();
    CHECK(buf.fd() == -1, "fd should be -1 after Destroy");
    CHECK(buf.data() == nullptr, "data() should be null after Destroy");
    CHECK(buf.cl_mem_handle() == nullptr, "cl_mem should be null after Destroy");
    CHECK(buf.size() == 0, "size should be 0 after Destroy");

    TEST_PASS("#6 Destroy cleanup");
}

// ============================================================
// #7  移动语义
// ============================================================
static void test_7_move() {
    TEST_BEGIN("#7 move semantics");

    DmaBuffer b1;
    bool ok = b1.Init(4096);
    CHECK(ok, "Init should succeed");

    int orig_fd = b1.fd();
    void* orig_data = b1.data();
    CHECK(orig_fd >= 0, "b1 fd should be valid");
    CHECK(orig_data != nullptr, "b1 data should be non-null");

    // move construct
    DmaBuffer b2(std::move(b1));
    CHECK(b1.fd() == -1, "b1 fd should be -1 after move");
    CHECK(b1.data() == nullptr, "b1 data should be null after move");
    CHECK(b2.fd() == orig_fd, "b2 should have b1's original fd");
    CHECK(b2.data() == orig_data, "b2 should have b1's original data");

    // move assign
    DmaBuffer b3;
    b3 = std::move(b2);
    CHECK(b2.fd() == -1, "b2 fd should be -1 after move assign");
    CHECK(b3.fd() == orig_fd, "b3 should have original fd");
    CHECK(b3.data() == orig_data, "b3 should have original data");

    b3.Destroy();
    TEST_PASS("#7 move semantics");
}

// ============================================================
// main
// ============================================================
int main() {
    fprintf(stderr, "=== DmaBuffer Test Suite ===\n");

    // OpenCL 环境初始化
    if (!setup_opencl()) {
        fprintf(stderr, "FATAL: OpenCL setup failed, skipping GPU tests\n");
        // 仍然运行不需要 GPU 的测试
        test_1_dma_alloc();
        test_3_cpu_rw();
        test_5_multi_buffer();
        test_7_move();
        fprintf(stderr, "\n=== Results: %d PASS, %d FAIL (GPU tests skipped) ===\n",
                g_pass, g_fail);
        return g_fail > 0 ? 1 : 0;
    }

    // 运行全部 7 个测试
    test_1_dma_alloc();
    test_2_opencl_bind();
    test_3_cpu_rw();
    test_4_gpu_write_cpu_read();
    test_5_multi_buffer();
    test_6_destroy();
    test_7_move();

    teardown_opencl();

    fprintf(stderr, "\n=== Results: %d PASS, %d FAIL ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
