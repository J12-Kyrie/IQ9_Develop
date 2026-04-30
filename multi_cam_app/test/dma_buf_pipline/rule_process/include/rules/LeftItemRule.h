#pragma once

#include "core/Rule.h"
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace rules {

class LeftItemRule : public core::Rule {
public:
  explicit LeftItemRule(const core::RuleConfig& config);
  core::Status validate() const override;
  std::vector<core::Alert> apply(const core::EventMeta& event) override;

private:
  struct ItemState {
    std::vector<float> bbox;
    uint64_t start_ts = 0;
    bool alerted = false;
  };

  struct PersonState {
    float last_cx = -1.0f;
    float last_cy = -1.0f;
    int static_frames = 0;
  };

  bool isStaticPerson(int track_id, const std::vector<float>& bbox);

  static constexpr float kMinPersonDisplacement = 0.008f;
  static constexpr int   kStaticPersonThreshold  = 1;

  uint64_t grace_period_us_ = 0;
  float item_owner_iou_threshold_ = 0.0f;
  bool merge_handbag_backpack_ = false;
  float item_owner_h_expand_ = 0.0f;
  float item_owner_v_expand_ = 0.0f;
  std::set<std::string> item_classes_;

  std::mutex state_mutex_;
  std::map<int, ItemState> item_states_;
  std::map<int, PersonState> person_states_;
  std::set<int> cached_unified_bag_track_ids_;
};

} // namespace rules
