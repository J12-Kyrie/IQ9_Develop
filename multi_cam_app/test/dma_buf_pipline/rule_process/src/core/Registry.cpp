#include "core/Registry.h"

#include "rules/DoorCollisionRule.h"
#include "rules/LeftItemRule.h"
#include "rules/OpenTrunkFaceRule.h"

namespace core {

void RuleRegistry::ensureBuiltinRulesRegistered() {
  std::call_once(builtin_rules_once_, [this]() { registerBuiltinRules(); });
}

void RuleRegistry::registerBuiltinRules() {
  registerRule("DoorCollisionRule", [](const RuleConfig &config) {
    return std::make_shared<::rules::DoorCollisionRule>(config);
  });

  registerRule("LeftItemRule", [](const RuleConfig &config) {
    return std::make_shared<::rules::LeftItemRule>(config);
  });

  registerRule("OpenTrunkFaceRule", [](const RuleConfig &config) {
    return std::make_shared<::rules::OpenTrunkFaceRule>(config);
  });
}

} // namespace core
