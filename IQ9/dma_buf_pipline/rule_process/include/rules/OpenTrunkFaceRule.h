#pragma once

#include "core/Rule.h"
#include "FaceDatabase.h"
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace rules {

class OpenTrunkFaceRule : public core::Rule {
public:
  explicit OpenTrunkFaceRule(const core::RuleConfig& config);
  core::Status validate() const override;
  std::vector<core::Alert> apply(const core::EventMeta& event) override;

private:
  struct FaceState {
    std::string user_id;
    bool is_known = false;
  };

  std::string target_class_;
  float threshold_ = 0.0f;
  std::string embedding_file_;
  bool face_db_loaded_ = false;
  std::string face_db_error_;

  utils::FaceDatabase face_db_;
  std::mutex state_mutex_;
  std::map<int, FaceState> face_cache_;
};

} // namespace rules
