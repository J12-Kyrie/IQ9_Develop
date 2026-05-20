#ifndef RULES_DATA_TYPES_H
#define RULES_DATA_TYPES_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace core {

struct Detection {
  std::string class_name_;
  std::vector<float> bbox_;
  int track_id_;
  /// scene_update events[].confidence; default 1 if absent (no LeftItem owner filter skip).
  float confidence_ = 1.0f;
  std::vector<float> embedding_;
};

struct EventMeta {
  std::string source_id_;
  /// Numeric source_id from JSON (for ImageMeta.sourceId / prompts).
  int source_id_num_ = 0;
  /// Monotonic time from scene_update.sources[].timestamp (multi_cam / DriveWorks: microseconds).
  uint64_t timestamp_;
  std::vector<int> removed_track_ids_;
  std::vector<Detection> detections_;
  /// From scene_update.sources[].image_meta only (DMA-BUF producer-published geometry).
  uint32_t slot_index_ = 0;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t channels_ = 3;
  /// Raw nanosecond timestamp for IQ9_Task_Packet construction (separate from timestamp_).
  uint64_t timestamp_ns_ = 0;
};

struct Alert {
  std::string rule_name_;
  int priority_;
  std::string dedup_key_;
  int track_id_;
  /// Copied from EventMeta::timestamp_ (microseconds).
  uint64_t timestamp_;
};

struct RuleConfig {
  std::string name_;
  int priority_;
  std::vector<std::string> target_sources_;
  std::vector<std::string> target_classes_;
  std::map<std::string, float> params_;
  std::map<std::string, std::string> str_params_;
};

struct Status {
  bool ok_ = true;
  std::string message_;

  Status() = default;
  Status(bool ok, const std::string &msg) : ok_(ok), message_(msg) {}

  static Status Ok() { return Status(true, ""); }
  static Status Error(const std::string &msg) { return Status(false, msg); }

  bool ok() const { return ok_; }
};

} // namespace core

#endif // RULES_DATA_TYPES_H
