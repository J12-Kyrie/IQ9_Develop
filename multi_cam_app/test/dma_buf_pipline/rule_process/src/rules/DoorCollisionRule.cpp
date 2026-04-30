#include "rules/DoorCollisionRule.h"
#include "MathUtils.h"

namespace {
constexpr const char* kParamIoU = "iou_threshold";
constexpr const char* kParamDangerZone = "use_danger_zone";
constexpr const char* kParamHExpand = "h_expand";
constexpr const char* kParamVExpand = "v_expand";
}  // namespace

namespace rules {

DoorCollisionRule::DoorCollisionRule(const core::RuleConfig &config)
    : core::Rule(config),
      iou_threshold_(getParam(kParamIoU)),
      use_danger_zone_(getParam(kParamDangerZone) > 0.5f),
      h_expand_(getParam(kParamHExpand)),
      v_expand_(getParam(kParamVExpand)) {
  if (config.target_classes_.size() >= 2) {
    class_a_ = config.target_classes_[0];
    class_b_ = config.target_classes_[1];
  }
}

core::Status DoorCollisionRule::validate() const {
  if (target_classes_.size() != 2 || target_classes_[0].empty() ||
      target_classes_[1].empty() || target_classes_[0] == target_classes_[1]) {
    return core::Status::Error(
        "DoorCollisionRule requires exactly 2 distinct target_classes");
  }
  for (const char* p : {kParamIoU, kParamDangerZone, kParamHExpand, kParamVExpand}) {
    if (!hasParam(p))
      return core::Status::Error(std::string("Missing param: ") + p);
  }
  if (iou_threshold_ <= 0.0f || iou_threshold_ >= 1.0f) {
    return core::Status::Error("IoU threshold must be between 0 and 1");
  }
  if (h_expand_ < 0.0f || v_expand_ < 0.0f) {
    return core::Status::Error("Danger zone expansion must be non-negative");
  }
  return core::Status::Ok();
}

std::vector<core::Alert> DoorCollisionRule::apply(const core::EventMeta &event) {
  std::vector<core::Alert> alerts;
  std::vector<core::Detection> objects_a;
  std::vector<core::Detection> objects_b;

  for (const auto &d : event.detections_) {
    if (d.class_name_ == class_a_) {
      objects_a.push_back(d);
    } else if (d.class_name_ == class_b_) {
      objects_b.push_back(d);
    }
  }

  for (const auto &obj_a : objects_a) {
    std::vector<float> expanded_bbox;
    const std::vector<float>* a_bbox = &obj_a.bbox_;
    if (use_danger_zone_) {
      expanded_bbox = utils::calculate_danger_zone(obj_a.bbox_, h_expand_, v_expand_);
      a_bbox = &expanded_bbox;
    }

    for (const auto &obj_b : objects_b) {
      if (utils::calculate_iou(*a_bbox, obj_b.bbox_) > iou_threshold_) {
        const std::string dedup_key = event.source_id_ + "_" + name_ + "_" +
                                      std::to_string(obj_a.track_id_) + "_" +
                                      std::to_string(obj_b.track_id_);
        alerts.push_back(buildAlert(event, obj_a.track_id_, dedup_key));
      }
    }
  }
  return alerts;
}

}  // namespace rules
