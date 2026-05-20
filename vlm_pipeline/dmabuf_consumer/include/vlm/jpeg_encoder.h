#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vlm {

class JpegEncoder {
public:
    struct Config {
        int quality = 80;
        uint32_t target_width = 448;
        uint32_t target_height = 448;
    };

    explicit JpegEncoder(const Config& cfg);
    JpegEncoder();

    bool encode(const uint8_t* rgb_data, uint32_t width, uint32_t height,
                uint32_t row_bytes, std::vector<uint8_t>& jpeg_bytes,
                std::string* error = nullptr);

private:
    Config config_;

    std::vector<uint8_t> resizeRgb24(const uint8_t* src,
                                      uint32_t src_w, uint32_t src_h,
                                      uint32_t src_row_bytes,
                                      uint32_t dst_w, uint32_t dst_h);
};

} // namespace vlm
