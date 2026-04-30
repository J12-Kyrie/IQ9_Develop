#ifndef CORE_RULE_H
#define CORE_RULE_H

#include "core/DataTypes.h"
#include <map>
#include <string>
#include <vector>

namespace core {

class Rule {
public:
  explicit Rule(const RuleConfig &config)
      : name_(config.name_), priority_(config.priority_),
        target_sources_(config.target_sources_),
        target_classes_(config.target_classes_), params_(config.params_),
        str_params_(config.str_params_) {}

  virtual ~Rule() = default;

  // Validate configuration params. Returns Status::Ok() if valid.
  virtual Status validate() const { return Status::Ok(); }

  virtual std::vector<Alert> apply(const EventMeta &event) = 0;

  const std::vector<std::string> &targetSources() const { return target_sources_; }

protected:
  Alert buildAlert(const EventMeta &event, int track_id = -1,
                   const std::string &custom_dedup_key = "") {
    Alert alert;
    alert.rule_name_ = name_;
    alert.priority_ = priority_;
    alert.dedup_key_ =
        custom_dedup_key.empty()
            ? event.source_id_ + "_" + name_ + "_" + std::to_string(track_id)
            : custom_dedup_key;
    alert.timestamp_ = event.timestamp_;
    alert.track_id_ = track_id;
    return alert;
  }

  float getParam(const std::string& key, float fallback = 0.0f) const {
    auto it = params_.find(key);
    return it != params_.end() ? it->second : fallback;
  }

  bool hasParam(const std::string& key) const {
    return params_.count(key) > 0;
  }

  std::string getStrParam(const std::string& key, const std::string& fallback = "") const {
    auto it = str_params_.find(key);
    return it != str_params_.end() ? it->second : fallback;
  }

  std::string name_;
  int priority_;
  std::vector<std::string> target_sources_;
  std::vector<std::string> target_classes_;
  std::map<std::string, float> params_;
  std::map<std::string, std::string> str_params_;
};

} // namespace core

#endif // CORE_RULE_H
