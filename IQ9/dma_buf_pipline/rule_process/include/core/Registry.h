#ifndef CORE_REGISTRY_H
#define CORE_REGISTRY_H

#include "core/DataTypes.h"
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace core {

class Rule;

class RuleRegistry {
public:
  using Creator = std::function<std::shared_ptr<Rule>(const RuleConfig &)>;

  static RuleRegistry &instance() {
    static RuleRegistry inst;
    return inst;
  }

  void registerRule(const std::string &name, Creator creator) {
    creators[name] = creator;
  }

  std::shared_ptr<Rule> createRule(const RuleConfig &config) {
    ensureBuiltinRulesRegistered();
    auto it = creators.find(config.name_);
    if (it != creators.end()) {
      return it->second(config);
    }
    return nullptr;
  }

private:
  void ensureBuiltinRulesRegistered();
  void registerBuiltinRules();

  std::once_flag builtin_rules_once_;
  std::map<std::string, Creator> creators;
};

} // namespace core

#endif // CORE_REGISTRY_H
