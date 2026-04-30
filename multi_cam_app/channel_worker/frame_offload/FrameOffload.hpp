#ifndef MULTI_CAM_APP_FRAME_OFFLOAD_HPP
#define MULTI_CAM_APP_FRAME_OFFLOAD_HPP

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>

#include <gst/gst.h>

#include "utils/FaceGallery.hpp"

namespace multi_cam_app::frame_offload {

struct FrameOffloadConfig {
    uint32_t channel_id {0};
    // When true, emit one scene_update source object (for qtimsgagg merge + MQTT).
    bool scene_update_mqtt {false};
    uint32_t meta_width {0};
    uint32_t meta_height {0};
    // Face gallery for gallery matching (optional, owned externally)
    gallery::FaceGallery* gallery {nullptr};
    float gallery_threshold {0.3f};
    uint32_t gallery_min_face_size {40U};
    float gallery_min_score {0.6f};
    // If non-empty, open scene_update_ch{N}.jsonl here (per-channel local JSONL)
    std::string jsonl_out_dir {};
};

class FrameOffload {
public:
    explicit FrameOffload(const FrameOffloadConfig& config);

    std::string ExtractAndSerialize(GstBuffer* buffer,
                                    uint32_t frame_w, uint32_t frame_h,
                                    int32_t image_path);

    uint64_t FramesProcessed() const { return frames_processed_; }

    void WriteSceneJsonl(const std::string& json);

private:
    FrameOffloadConfig config_;
    uint64_t frames_processed_ {0};
    uint64_t local_frame_counter_ {0};
    uint64_t last_timestamp_ns_ {0};
    std::unordered_map<uint32_t, int> track_id_to_face_;
    std::unordered_map<uint32_t, uint64_t> track_id_last_seen_;
    std::ofstream jsonl_file_;
};

}  // namespace multi_cam_app::frame_offload

#endif  // MULTI_CAM_APP_FRAME_OFFLOAD_HPP
