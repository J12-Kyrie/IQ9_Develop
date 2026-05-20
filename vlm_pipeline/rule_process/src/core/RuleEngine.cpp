#include "core/RuleEngine.h"
#include "infra/Logger.h"
#include "core/AlertManager.h"
#include "core/Dispatcher.h"
#include "core/Registry.h"
#include "core/Rule.h"
#include "core/DataTypes.h"

namespace core {

// Actual Implementation Class
class RuleEngine::Impl {
public:
  Impl(const std::vector<RuleConfig> &configs,
       const AlertManagerParams &alert_params) {
    std::vector<std::shared_ptr<Rule>> rules;
    for (const auto &config : configs) {
      auto rule = RuleRegistry::instance().createRule(config);
      if (rule) {
        Status st = rule->validate();
        if (st.ok()) {
          rules.push_back(rule);
        } else {
          utils::Logger::error("RuleEngine", "Validation failed [" + config.name_ +
                               "]: " + st.message_);
        }
      } else {
        utils::Logger::error("RuleEngine", "Failed to create rule: " + config.name_);
      }
    }
    dispatcher.reset(new Dispatcher(rules));
    alert_manager.reset(new AlertManager(alert_params));
  }

  std::vector<Alert> processEvent(const EventMeta &event) {
    auto candidates = dispatcher->dispatch(event);
    auto final_alerts = alert_manager->processAlerts(candidates);
    return final_alerts;
  }

private:
  std::unique_ptr<Dispatcher> dispatcher;
  std::unique_ptr<AlertManager> alert_manager;
};

// Forwarding Calls to Pimpl
RuleEngine::RuleEngine(const std::vector<RuleConfig> &configs)
    : RuleEngine(configs, AlertManagerParams{}) {
} // C++11 make_unique not available in all compilers, new is safe here

RuleEngine::RuleEngine(const std::vector<RuleConfig> &configs,
                       const AlertManagerParams &alert_params)
    : pImpl(new Impl(configs, alert_params)) {
} // C++11 make_unique not available in all compilers, new is safe here

RuleEngine::~RuleEngine() = default; // Defined here where Impl is complete

RuleEngine::RuleEngine(RuleEngine &&) noexcept = default;
RuleEngine &RuleEngine::operator=(RuleEngine &&) noexcept = default;

std::vector<Alert> RuleEngine::processEvent(const EventMeta &event) {
  return pImpl->processEvent(event);
}

} // namespace core
