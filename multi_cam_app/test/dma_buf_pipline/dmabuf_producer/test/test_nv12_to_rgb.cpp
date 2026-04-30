#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nv12_to_rgb.h"

static int g_pass = 0;
static int g_fail = 0;

static void CheckPixel(const uint8_t* rgb, int x, int y, int width,
                       uint8_t exp_r, uint8_t exp_g, uint8_t exp_b,
                       int tolerance, const char* label) {
    int idx = (y * width + x) * 3;
    int dr = std::abs(rgb[idx] - exp_r);
    int dg = std::abs(rgb[idx + 1] - exp_g);
    int db = std::abs(rgb[idx + 2] - exp_b);

    if (dr <= tolerance && dg <= tolerance && db <= tolerance) {
        ++g_pass;
    } else {
        ++g_fail;
        std::fprintf(stderr,
            "FAIL [%s] pixel(%d,%d): got(%d,%d,%d) expected(%d,%d,%d) "
            "delta(%d,%d,%d) tol=%d\n",
            label, x, y, rgb[idx], rgb[idx + 1], rgb[idx + 2],
            exp_r, exp_g, exp_b, dr, dg, db, tolerance);
    }
}

// Build an NV12 frame: Y plane + UV plane (interleaved U,V)
// y_stride may be > width (simulates Qualcomm padding)
static void BuildNv12(std::vector<uint8_t>* out, int width, int height,
                      int y_stride, int* out_uv_offset,
                      uint8_t y_val, uint8_t u_val, uint8_t v_val) {
    int uv_height = (height + 1) / 2;
    int uv_offset = y_stride * height;
    int total = uv_offset + y_stride * uv_height;
    out->resize(static_cast<size_t>(total), 0);

    // Fill Y plane
    for (int row = 0; row < height; ++row) {
        std::memset(out->data() + row * y_stride, y_val,
                    static_cast<size_t>(width));
    }

    // Fill UV plane (interleaved U, V)
    for (int row = 0; row < uv_height; ++row) {
        uint8_t* uv_row = out->data() + uv_offset + row * y_stride;
        for (int col = 0; col < width; col += 2) {
            uv_row[col] = u_val;
            uv_row[col + 1] = v_val;
        }
    }

    *out_uv_offset = uv_offset;
}

// Build NV12 frame with 4 color quadrants (top-left, top-right, bottom-left, bottom-right)
static void BuildNv12Quadrants(std::vector<uint8_t>* out, int width, int height,
                               int y_stride, int* out_uv_offset,
                               uint8_t y0, uint8_t u0, uint8_t v0,
                               uint8_t y1, uint8_t u1, uint8_t v1,
                               uint8_t y2, uint8_t u2, uint8_t v2,
                               uint8_t y3, uint8_t u3, uint8_t v3) {
    int uv_height = (height + 1) / 2;
    int uv_offset = y_stride * height;
    int total = uv_offset + y_stride * uv_height;
    out->resize(static_cast<size_t>(total), 0);

    int half_w = width / 2;
    int half_h = height / 2;

    // Fill Y plane per quadrant
    for (int row = 0; row < height; ++row) {
        uint8_t* y_row = out->data() + row * y_stride;
        bool top = (row < half_h);
        for (int col = 0; col < width; ++col) {
            bool left = (col < half_w);
            if (top && left)       y_row[col] = y0;
            else if (top && !left) y_row[col] = y1;
            else if (!top && left) y_row[col] = y2;
            else                   y_row[col] = y3;
        }
    }

    // Fill UV plane per quadrant
    for (int row = 0; row < uv_height; ++row) {
        uint8_t* uv_row = out->data() + uv_offset + row * y_stride;
        bool top = (row < half_h / 2);
        for (int col = 0; col < width; col += 2) {
            bool left = (col < half_w);
            uint8_t u, v;
            if (top && left)       { u = u0; v = v0; }
            else if (top && !left) { u = u1; v = v1; }
            else if (!top && left) { u = u2; v = v2; }
            else                   { u = u3; v = v3; }
            uv_row[col] = u;
            uv_row[col + 1] = v;
        }
    }

    *out_uv_offset = uv_offset;
}

// Test 1: Uniform gray (Y=128, U=128, V=128) -> RGB(128,128,128)
static void TestGray() {
    const int W = 16, H = 16, STRIDE = 16;
    std::vector<uint8_t> nv12;
    int uv_off = 0;
    BuildNv12(&nv12, W, H, STRIDE, &uv_off, 128, 128, 128);

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 0, 0, W, 128, 128, 128, 1, "gray-topleft");
    CheckPixel(rgb.data(), 8, 8, W, 128, 128, 128, 1, "gray-center");
    CheckPixel(rgb.data(), 15, 15, W, 128, 128, 128, 1, "gray-bottomright");
    std::fprintf(stdout, "[test] gray: done\n");
}

// Test 2: Black (Y=0, U=128, V=128) -> RGB(0,0,0)
static void TestBlack() {
    const int W = 8, H = 8, STRIDE = 8;
    std::vector<uint8_t> nv12;
    int uv_off = 0;
    BuildNv12(&nv12, W, H, STRIDE, &uv_off, 0, 128, 128);

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 0, 0, W, 0, 0, 0, 1, "black");
    std::fprintf(stdout, "[test] black: done\n");
}

// Test 3: White (Y=255, U=128, V=128) -> RGB(255,255,255)
static void TestWhite() {
    const int W = 8, H = 8, STRIDE = 8;
    std::vector<uint8_t> nv12;
    int uv_off = 0;
    BuildNv12(&nv12, W, H, STRIDE, &uv_off, 255, 128, 128);

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 0, 0, W, 255, 255, 255, 1, "white");
    std::fprintf(stdout, "[test] white: done\n");
}

// Test 4: Known red (Y=76, U=84, V=255) -> ~RGB(254,1,0)
static void TestRed() {
    const int W = 8, H = 8, STRIDE = 8;
    std::vector<uint8_t> nv12;
    int uv_off = 0;
    BuildNv12(&nv12, W, H, STRIDE, &uv_off, 76, 84, 255);

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 0, 0, W, 254, 1, 0, 3, "red");
    std::fprintf(stdout, "[test] red: done\n");
}

// Test 5: Known green (Y=150, U=44, V=21) -> ~RGB(0,255,0)
static void TestGreen() {
    const int W = 8, H = 8, STRIDE = 8;
    std::vector<uint8_t> nv12;
    int uv_off = 0;
    BuildNv12(&nv12, W, H, STRIDE, &uv_off, 150, 44, 21);

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 0, 0, W, 0, 255, 0, 5, "green");
    std::fprintf(stdout, "[test] green: done\n");
}

// Test 6: Known blue (Y=29, U=255, V=107) -> ~RGB(0,0,254)
static void TestBlue() {
    const int W = 8, H = 8, STRIDE = 8;
    std::vector<uint8_t> nv12;
    int uv_off = 0;
    BuildNv12(&nv12, W, H, STRIDE, &uv_off, 29, 255, 107);

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 0, 0, W, 0, 0, 254, 5, "blue");
    std::fprintf(stdout, "[test] blue: done\n");
}

// Test 7: Stride padding (width=12, stride=16, simulating Qualcomm alignment)
static void TestStridePadding() {
    const int W = 12, H = 8, STRIDE = 16;
    std::vector<uint8_t> nv12;
    int uv_off = 0;
    BuildNv12(&nv12, W, H, STRIDE, &uv_off, 200, 128, 128);

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 0, 0, W, 200, 200, 200, 1, "stride-first");
    CheckPixel(rgb.data(), 11, 7, W, 200, 200, 200, 1, "stride-last");
    std::fprintf(stdout, "[test] stride_padding: done\n");
}

// Test 8: 4 color quadrants (verifies UV subsampling boundaries)
static void TestQuadrants() {
    const int W = 16, H = 16, STRIDE = 16;
    std::vector<uint8_t> nv12;
    int uv_off = 0;
    // TL=gray, TR=bright, BL=dark, BR=mid
    BuildNv12Quadrants(&nv12, W, H, STRIDE, &uv_off,
                       128, 128, 128,   // TL: gray
                       220, 128, 128,   // TR: bright gray
                       40,  128, 128,   // BL: dark gray
                       180, 128, 128);  // BR: light gray

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 2, 2, W, 128, 128, 128, 1, "quad-TL");
    CheckPixel(rgb.data(), 12, 2, W, 220, 220, 220, 1, "quad-TR");
    CheckPixel(rgb.data(), 2, 12, W, 40, 40, 40, 1, "quad-BL");
    CheckPixel(rgb.data(), 12, 12, W, 180, 180, 180, 1, "quad-BR");
    std::fprintf(stdout, "[test] quadrants: done\n");
}

// Test 9: 1080p allocation (verify no crash on real-world size)
static void Test1080p() {
    const int W = 1920, H = 1080, STRIDE = 1920;
    std::vector<uint8_t> nv12;
    int uv_off = 0;
    BuildNv12(&nv12, W, H, STRIDE, &uv_off, 128, 128, 128);

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 0, 0, W, 128, 128, 128, 1, "1080p-origin");
    CheckPixel(rgb.data(), 1919, 1079, W, 128, 128, 128, 1, "1080p-last");
    CheckPixel(rgb.data(), 960, 540, W, 128, 128, 128, 1, "1080p-center");
    std::fprintf(stdout, "[test] 1080p: done\n");
}

// Test 10: Qualcomm 1080→1088 scanline padding (stride=1920, height=1080, uv_offset at 1088 rows)
static void TestQualcommPadding() {
    const int W = 1920, H = 1080, STRIDE = 1920;
    int padded_height = 1088;
    int uv_off = STRIDE * padded_height;
    int uv_height = (padded_height + 1) / 2;
    int total = uv_off + STRIDE * uv_height;

    std::vector<uint8_t> nv12(static_cast<size_t>(total), 0);

    // Fill Y plane (only first 1080 rows matter)
    for (int row = 0; row < H; ++row) {
        std::memset(nv12.data() + row * STRIDE, 200, static_cast<size_t>(W));
    }

    // Fill UV plane at padded offset
    for (int row = 0; row < uv_height; ++row) {
        uint8_t* uv_row = nv12.data() + uv_off + row * STRIDE;
        for (int col = 0; col < W; col += 2) {
            uv_row[col] = 128;
            uv_row[col + 1] = 128;
        }
    }

    std::vector<uint8_t> rgb(static_cast<size_t>(W * H * 3));
    dmabuf_producer::Nv12ToRgb24(nv12.data(), uv_off, rgb.data(), W, H, STRIDE);

    CheckPixel(rgb.data(), 960, 540, W, 200, 200, 200, 1, "qcom-pad-center");
    CheckPixel(rgb.data(), 0, 1079, W, 200, 200, 200, 1, "qcom-pad-lastrow");
    std::fprintf(stdout, "[test] qualcomm_padding: done\n");
}

int main() {
    std::fprintf(stdout, "=== NV12 -> RGB24 Unit Tests ===\n\n");

    TestGray();
    TestBlack();
    TestWhite();
    TestRed();
    TestGreen();
    TestBlue();
    TestStridePadding();
    TestQuadrants();
    Test1080p();
    TestQualcommPadding();

    std::fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n",
                 g_pass, g_fail);

    return (g_fail > 0) ? 1 : 0;
}
