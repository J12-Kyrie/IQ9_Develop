#include "rules/LeftItemRule.h"
#include "MathUtils.h"

#include <algorithm>

namespace {
constexpr const char* kParamGracePeriod = "grace_period_sec";
constexpr const char* kParamItemOwnerIoU = "item_owner_iou_threshold";
constexpr const char* kParamMergeHandbagBackpack = "merge_handbag_backpack";
constexpr const char* kParamItemOwnerHExpand = "item_owner_h_expand";
constexpr const char* kParamItemOwnerVExpand = "item_owner_v_expand";

/// Persons below this confidence are ignored for "owner zone" IoU.
constexpr float kMinPersonOwnerConfidence = 0.4f;

// Synthetic track id for merged handbag+backpack state.
constexpr int kUnifiedBagTrackId = -1000;

bool isHandbagOrBackpack(const std::string& c) {
  return c == "handbag" || c == "backpack";
}

std::vector<float> unionBboxXYXY(const std::vector<std::vector<float>>& boxes) {
  if (boxes.empty()) return {};
  float x1 = boxes[0][0], y1 = boxes[0][1], x2 = boxes[0][2], y2 = boxes[0][3];
  for (size_t i = 1; i < boxes.size(); ++i) {
    if (boxes[i].size() < 4) continue;
    x1 = std::min(x1, boxes[i][0]);
    y1 = std::min(y1, boxes[i][1]);
    x2 = std::max(x2, boxes[i][2]);
    y2 = std::max(y2, boxes[i][3]);
  }
  return {x1, y1, x2, y2};
}
}  // namespace

namespace rules {

LeftItemRule::LeftItemRule(const core::RuleConfig &config)
    : core::Rule(config),
      grace_period_us_(static_cast<uint64_t>(getParam(kParamGracePeriod) * 1000000.0f)),
      item_owner_iou_threshold_(getParam(kParamItemOwnerIoU)),
      merge_handbag_backpack_(getParam(kParamMergeHandbagBackpack) >= 0.5f),
      item_owner_h_expand_(getParam(kParamItemOwnerHExpand)),
      item_owner_v_expand_(getParam(kParamItemOwnerVExpand)),
      item_classes_(config.target_classes_.begin(), config.target_classes_.end()) {}

core::Status LeftItemRule::validate() const {
  if (!hasParam(kParamGracePeriod))
    return core::Status::Error("Missing param: grace_period_sec");
  if (!hasParam(kParamItemOwnerIoU))
    return core::Status::Error("Missing param: item_owner_iou_threshold");
  if (grace_period_us_ == 0)
    return core::Status::Error("Grace period must be positive");
  if (item_owner_iou_threshold_ < 0.0f || item_owner_iou_threshold_ > 1.0f)
    return core::Status::Error("item_owner_iou_threshold must be between 0 and 1");
  if (item_owner_h_expand_ < 0.0f || item_owner_v_expand_ < 0.0f)
    return core::Status::Error("item_owner_h_expand / item_owner_v_expand must be non-negative");
  return core::Status::Ok();
}

std::vector<core::Alert> LeftItemRule::apply(const core::EventMeta &event) {
  std::vector<core::Alert> alerts;

  std::vector<const std::vector<float>*> person_bboxes;
  person_bboxes.reserve(event.detections_.size());

  struct ItemUpdate {
    int track_id;
    std::vector<float> bbox;
  };
  std::vector<ItemUpdate> item_updates;
  item_updates.reserve(event.detections_.size());

  std::vector<std::vector<float>> mergeable_bag_boxes;
  std::vector<int> mergeable_bag_track_ids;
  mergeable_bag_track_ids.reserve(4);

  for (const auto& det : event.detections_) {
    if (det.class_name_ == "person") {
      if (det.confidence_ >= kMinPersonOwnerConfidence) {
        person_bboxes.push_back(&det.bbox_);
      }
      continue;
    }
    if (item_classes_.count(det.class_name_) == 0U) continue;
    if (merge_handbag_backpack_ && isHandbagOrBackpack(det.class_name_)) {
      if (det.bbox_.size() >= 4) {
        mergeable_bag_boxes.push_back(det.bbox_);
        mergeable_bag_track_ids.push_back(det.track_id_);
      }
      continue;
    }
    item_updates.push_back({det.track_id_, det.bbox_});
  }

  if (merge_handbag_backpack_ && !mergeable_bag_boxes.empty()) {
    item_updates.push_back({kUnifiedBagTrackId, unionBboxXYXY(mergeable_bag_boxes)});
  }

  struct WorkItem {
    int track_id;
    ItemState state;
  };
  std::vector<WorkItem> work_items;

  {
    std::lock_guard<std::mutex> lock(state_mutex_);

    for (int rid : event.removed_track_ids_) {
      item_states_.erase(rid);
    }

    if (merge_handbag_backpack_) {
      bool drop_unified_bag = false;
      for (int rid : event.removed_track_ids_) {
        if (cached_unified_bag_track_ids_.count(rid) != 0U) {
          drop_unified_bag = true;
          break;
        }
      }
      if (drop_unified_bag) {
        item_states_.erase(kUnifiedBagTrackId);
        cached_unified_bag_track_ids_.clear();
      }
      if (!mergeable_bag_boxes.empty()) {
        cached_unified_bag_track_ids_.clear();
        for (int tid : mergeable_bag_track_ids) {
          cached_unified_bag_track_ids_.insert(tid);
        }
      }
    }

    for (const auto& upd : item_updates) {
      ItemState& state = item_states_[upd.track_id];
      state.bbox = upd.bbox;
    }

    work_items.reserve(item_states_.size());
    for (const auto& kv : item_states_) {
      work_items.push_back({kv.first, kv.second});
    }
  }

  std::vector<std::vector<float>> person_owner_zones;
  person_owner_zones.reserve(person_bboxes.size());
  for (const auto* person_bbox : person_bboxes) {
    if (person_bbox != nullptr && person_bbox->size() >= 4) {
      person_owner_zones.push_back(utils::calculate_danger_zone(
          *person_bbox, item_owner_h_expand_, item_owner_v_expand_));
    }
  }

  for (auto& item : work_items) {
    bool has_owner = false;
    for (const auto& zone : person_owner_zones) {
      if (utils::calculate_iou(zone, item.state.bbox) > item_owner_iou_threshold_) {
        has_owner = true;
        break;
      }
    }

    if (has_owner) {
      item.state.start_ts = 0;
      item.state.alerted = false;
      continue;
    }

    if (item.state.start_ts == 0) {
      item.state.start_ts = event.timestamp_;
    }

    if (!item.state.alerted &&
        (event.timestamp_ - item.state.start_ts >= grace_period_us_)) {
      if (item.track_id == kUnifiedBagTrackId) {
        alerts.push_back(buildAlert(
            event, kUnifiedBagTrackId,
            event.source_id_ + "_" + name_ + "_merged_bag"));
      } else {
        alerts.push_back(buildAlert(event, item.track_id));
      }
      item.state.alerted = true;
    }
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (const auto& item : work_items) {
      auto it = item_states_.find(item.track_id);
      if (it == item_states_.end()) continue;
      it->second.start_ts = item.state.start_ts;
      it->second.alerted = item.state.alerted;
    }
  }

  return alerts;
}

}  // namespace rules
