#ifndef CORE_DISPATCHER_H
#define CORE_DISPATCHER_H

#include "core/DataTypes.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace core {

class Rule;

/**
 * @class Dispatcher
 * @brief Routes events to matching rules based on target_sources.
 */
class Dispatcher {
public:
  explicit Dispatcher(const std::vector<std::shared_ptr<Rule>> &rules);

  std::vector<Alert> dispatch(const EventMeta &event) const;

private:
  std::unordered_map<std::string, std::vector<std::shared_ptr<Rule>>> routes_;
};

} // namespace core

#endif // CORE_DISPATCHER_H
