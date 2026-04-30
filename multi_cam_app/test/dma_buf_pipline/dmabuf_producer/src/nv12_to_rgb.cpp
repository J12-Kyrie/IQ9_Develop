#include "nv12_to_rgb.h"

namespace dmabuf_producer {

// Clamp int to [0, 255]
static inline uint8_t ClampU8(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return static_cast<uint8_t>(val);
}

void Nv12ToRgb24(const uint8_t* nv12_data, int uv_offset,
                 uint8_t* rgb24_out, int width, int height, int y_stride) {
    // Y plane starts at nv12_data[0]
    // UV plane (interleaved U,V) starts at nv12_data[uv_offset]
    // UV stride = y_stride (same as Y plane on Qualcomm)
    const uint8_t* y_plane = nv12_data;
    const uint8_t* uv_plane = nv12_data + uv_offset;
    const int uv_stride = y_stride;

    // BT.601 conversion using integer math (multiply by 256, shift by 8)
    // R = Y + 1.402 * (V - 128)   -> Y + (359 * (V-128)) >> 8
    // G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)
    //   -> Y - (88 * (U-128) + 183 * (V-128)) >> 8
    // B = Y + 1.772 * (U - 128)   -> Y + (454 * (U-128)) >> 8

    for (int row = 0; row < height; ++row) {
        const uint8_t* y_row = y_plane + row * y_stride;
        // UV plane is half-height: row/2; each UV pair covers 2 horizontal pixels
        const uint8_t* uv_row = uv_plane + (row / 2) * uv_stride;
        uint8_t* out_row = rgb24_out + row * width * 3;

        for (int col = 0; col < width; ++col) {
            int y_val = static_cast<int>(y_row[col]);

            // UV is subsampled 2:1 horizontally, interleaved U,V
            int uv_col = (col / 2) * 2;  // index into UV row
            int u_val = static_cast<int>(uv_row[uv_col]) - 128;
            int v_val = static_cast<int>(uv_row[uv_col + 1]) - 128;

            int r = y_val + ((359 * v_val) >> 8);
            int g = y_val - ((88 * u_val + 183 * v_val) >> 8);
            int b = y_val + ((454 * u_val) >> 8);

            int out_idx = col * 3;
            out_row[out_idx]     = ClampU8(r);
            out_row[out_idx + 1] = ClampU8(g);
            out_row[out_idx + 2] = ClampU8(b);
        }
    }
}

}  // namespace dmabuf_producer
