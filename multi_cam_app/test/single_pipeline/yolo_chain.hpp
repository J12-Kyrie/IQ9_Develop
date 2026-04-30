#ifndef SINGLE_PIPELINE_YOLO_CHAIN_HPP
#define SINGLE_PIPELINE_YOLO_CHAIN_HPP

#include "yolo_latency_config.hpp"

#include <cstdint>
#include <string>

#include <gst/gst.h>

namespace single_pipeline {

// Adds one file decode → qtimlvconverter → qtitimingmark(in/out) → qtimlqnn → fakesink chain.
bool AddYoloQnnLatencyChain(const YoloQnnLatencyConfig& config,
                            uint32_t channel_id,
                            const std::string& video_path,
                            GstBin* pipeline_bin,
                            std::string* out_error);

}  // namespace single_pipeline

#endif  // SINGLE_PIPELINE_YOLO_CHAIN_HPP
