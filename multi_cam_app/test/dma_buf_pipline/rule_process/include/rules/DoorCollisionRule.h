#pragma once

#include "core/Rule.h"
#include <string>
#include <vector>

namespace rules {

class DoorCollisionRule : public core::Rule {
public:
  explicit DoorCollisionRule(const core::RuleConfig& config);
  core::Status validate() const override;
  std::vector<core::Alert> apply(const core::EventMeta& event) override;

private:
  std::string class_a_;
  std::string class_b_;
  float iou_threshold_ = 0.0f;
  bool use_danger_zone_ = false;
  float h_expand_ = 0.0f;
  float v_expand_ = 0.0f;
};

} // namespace rules
