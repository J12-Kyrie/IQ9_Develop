#ifndef SINGLE_PIPELINE_YOLO_LATENCY_CONFIG_HPP
#define SINGLE_PIPELINE_YOLO_LATENCY_CONFIG_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace single_pipeline {

// Mirrors AppConfig YOLO/QNN + LatencyTest fields; JSON is separate from project config.json.
struct YoloQnnLatencyConfig {
  std::vector<std::string> videos {};
  std::string output_dir {};
  std::string model_path {};
  std::string qnn_backend {"/usr/lib/libQnnHtp.so"};
  std::string qnn_system {"/usr/lib/libQnnSystem.so"};
  uint32_t qnn_backend_device_id {0U};
  std::vector<std::string> qnn_tensors {"boxes", "scores", "class_idx"};
  std::vector<std::string> qtimlvconverter_engine_order {"fcv", "c2d"};
  uint32_t sample_every_n {1U};
  bool flush_per_sample {true};
  uint32_t max_channels {6U};
};

bool LoadYoloQnnLatencyConfigFromFile(const std::string& path,
                                      YoloQnnLatencyConfig* out_config,
                                      std::string* out_error);

}  // namespace single_pipeline

#endif  // SINGLE_PIPELINE_YOLO_LATENCY_CONFIG_HPP
