#ifndef DMABUF_PRODUCER_NV12_TO_RGB_H
#define DMABUF_PRODUCER_NV12_TO_RGB_H

#include <cstdint>

namespace dmabuf_producer {

// Convert NV12 frame to packed RGB24.
// nv12_data: pointer to the start of the NV12 frame (Y plane)
// uv_offset: byte offset from nv12_data to the interleaved UV plane
//            (from GstVideoMeta::offset[1], NOT y_stride*height)
// rgb24_out: output buffer, must be at least width*height*3 bytes
// width/height: actual image dimensions (not padded)
// y_stride: row stride of Y plane in bytes (may be > width due to alignment)
void Nv12ToRgb24(const uint8_t* nv12_data, int uv_offset,
                 uint8_t* rgb24_out, int width, int height, int y_stride);

}  // namespace dmabuf_producer

#endif  // DMABUF_PRODUCER_NV12_TO_RGB_H
