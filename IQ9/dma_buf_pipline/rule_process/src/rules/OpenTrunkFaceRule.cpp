#include "rules/OpenTrunkFaceRule.h"

namespace {
constexpr const char* kParamSimilarity = "similarity_threshold";
constexpr const char* kParamEmbeddingSize = "embedding_size";
constexpr const char* kParamEmbeddingFile = "embedding_file";

static size_t getEmbeddingSize(const core::RuleConfig& config) {
  auto it = config.params_.find(kParamEmbeddingSize);
  if (it == config.params_.end() || it->second <= 0.0f) return 0;
  return static_cast<size_t>(it->second);
}
}  // namespace

namespace rules {

OpenTrunkFaceRule::OpenTrunkFaceRule(const core::RuleConfig &config)
    : core::Rule(config),
      face_db_(getEmbeddingSize(config)) {
  if (config.target_classes_.size() == 1) {
    target_class_ = config.target_classes_[0];
  }
  threshold_ = getParam(kParamSimilarity);
  embedding_file_ = getStrParam(kParamEmbeddingFile);

  if (getEmbeddingSize(config) > 0 && !embedding_file_.empty()) {
    auto result = face_db_.loadFromFile(embedding_file_);
    face_db_loaded_ = result.ok();
    if (!face_db_loaded_) {
      face_db_error_ = result.message_;
    }
  }
}

core::Status OpenTrunkFaceRule::validate() const {
  if (target_classes_.size() != 1 || target_classes_[0].empty())
    return core::Status::Error("OpenTrunkFaceRule requires exactly 1 target_class");
  if (!hasParam(kParamSimilarity))
    return core::Status::Error("Missing param: similarity_threshold");
  if (!hasParam(kParamEmbeddingSize) || getParam(kParamEmbeddingSize) <= 0)
    return core::Status::Error("Missing param: embedding_size");
  if (getStrParam(kParamEmbeddingFile).empty())
    return core::Status::Error("Missing param: embedding_file");
  if (!face_db_loaded_)
    return core::Status::Error(face_db_error_.empty() ? "Face database load failed" : face_db_error_);
  if (threshold_ <= 0.0f || threshold_ > 1.0f)
    return core::Status::Error("Similarity threshold must be (0,1]");
  return core::Status::Ok();
}

std::vector<core::Alert> OpenTrunkFaceRule::apply(const core::EventMeta &event) {
  std::vector<core::Alert> alerts;

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (int rid : event.removed_track_ids_) {
      face_cache_.erase(rid);
    }
  }

  for (const auto &det : event.detections_) {
    if (det.class_name_ != target_class_) continue;

    int tid = det.track_id_;

    FaceState cached_state;
    bool found_in_cache = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      auto it = face_cache_.find(tid);
      if (it != face_cache_.end()) {
        cached_state = it->second;
        found_in_cache = true;
      }
    }

    if (found_in_cache) {
      if (cached_state.is_known) {
        alerts.push_back(buildAlert(event, tid, name_ + "_" + cached_state.user_id));
      }
      continue;
    }

    if (det.embedding_.empty()) continue;

    auto result = face_db_.compare(det.embedding_);
    bool is_known = (result.second >= threshold_);

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      face_cache_[tid] = {result.first, is_known};
    }

    if (is_known) {
      alerts.push_back(buildAlert(event, tid, name_ + "_" + result.first));
    }
  }

  return alerts;
}

}  // namespace rules
