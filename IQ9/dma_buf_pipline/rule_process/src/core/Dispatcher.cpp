#include "core/Dispatcher.h"

#include "core/Rule.h"

#include <cstddef>
#include <iterator>

namespace core {

Dispatcher::Dispatcher(const std::vector<std::shared_ptr<Rule>> &rules) {
  for (const auto &rule : rules) {
    if (!rule) {
      continue;
    }

    const auto &targets = rule->targetSources();
    for (const auto &source : targets) {
      routes_[source].push_back(rule);
    }
  }
}

std::vector<Alert> Dispatcher::dispatch(const EventMeta &event) const {
  std::vector<Alert> all_alerts;

  auto it = routes_.find(event.source_id_);
  const size_t reserve_count =
      (it != routes_.end()) ? it->second.size() : 0U;
  if (reserve_count > 0) {
    all_alerts.reserve(reserve_count);
  }

  auto append = [&all_alerts](std::vector<Alert> alerts) {
    if (alerts.empty()) {
      return;
    }
    all_alerts.insert(all_alerts.end(),
                      std::make_move_iterator(alerts.begin()),
                      std::make_move_iterator(alerts.end()));
  };

  if (it != routes_.end()) {
    for (const auto &rule : it->second) {
      append(rule->apply(event));
    }
  }

  return all_alerts;
}

} // namespace core
