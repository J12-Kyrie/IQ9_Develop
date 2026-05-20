#include "vlm/jpeg_encoder.h"
#include <turbojpeg.h>
#include <algorithm>

namespace vlm {

JpegEncoder::JpegEncoder(const Config& cfg) : config_(cfg) {}
JpegEncoder::JpegEncoder() : config_(Config{}) {}

std::vector<uint8_t> JpegEncoder::resizeRgb24(const uint8_t* src,
                                                uint32_t src_w, uint32_t src_h,
                                                uint32_t src_row_bytes,
                                                uint32_t dst_w, uint32_t dst_h) {
    std::vector<uint8_t> dst(dst_w * dst_h * 3);
    for (uint32_t dy = 0; dy < dst_h; ++dy) {
        uint32_t sy = dy * src_h / dst_h;
        const uint8_t* src_row = src + sy * src_row_bytes;
        uint8_t* dst_row = dst.data() + dy * dst_w * 3;
        for (uint32_t dx = 0; dx < dst_w; ++dx) {
            uint32_t sx = dx * src_w / dst_w;
            const uint8_t* src_px = src_row + sx * 3;
            uint8_t* dst_px = dst_row + dx * 3;
            dst_px[0] = src_px[0];
            dst_px[1] = src_px[1];
            dst_px[2] = src_px[2];
        }
    }
    return dst;
}

bool JpegEncoder::encode(const uint8_t* rgb_data, uint32_t width, uint32_t height,
                          uint32_t row_bytes, std::vector<uint8_t>& jpeg_bytes,
                          std::string* error) {
    const uint8_t* encode_src = rgb_data;
    uint32_t encode_w = width;
    uint32_t encode_h = height;
    uint32_t encode_row_bytes = row_bytes;

    std::vector<uint8_t> resized;
    if (width != config_.target_width || height != config_.target_height) {
        resized = resizeRgb24(rgb_data, width, height, row_bytes,
                              config_.target_width, config_.target_height);
        encode_src = resized.data();
        encode_w = config_.target_width;
        encode_h = config_.target_height;
        encode_row_bytes = encode_w * 3;
    }

    tjhandle handle = tjInitCompress();
    if (!handle) {
        if (error) *error = "tjInitCompress failed";
        return false;
    }

    unsigned char* buf = nullptr;
    unsigned long buf_size = 0;
    int ret = tjCompress2(handle, encode_src, encode_w, encode_row_bytes, encode_h,
                          TJPF_RGB, &buf, &buf_size, TJSAMP_444,
                          config_.quality, 0);
    if (ret != 0) {
        if (error) *error = std::string("tjCompress2 failed: ") + tjGetErrorStr();
        tjFree(buf);
        tjDestroy(handle);
        return false;
    }

    jpeg_bytes.assign(buf, buf + buf_size);
    tjFree(buf);
    tjDestroy(handle);
    return true;
}

} // namespace vlm
