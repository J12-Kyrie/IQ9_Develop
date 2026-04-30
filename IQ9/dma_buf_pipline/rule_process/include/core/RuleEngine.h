/**
 * @file RuleEngine.h
 * @brief Public Interface for the Vision Rule Engine.
 *
 * This file defines the RuleEngine facade using the Pimpl idiom for ABI
 * stability. It is the main entry point for the "Rule Engine" service.
 */

#ifndef RULE_ENGINE_H
#define RULE_ENGINE_H

#include <memory>
#include <vector>

namespace core {

// Forward Declarations (Compilation Firewall)
struct Alert;
struct EventMeta;
struct RuleConfig;
struct AlertManagerParams;

/**
 * @class RuleEngine
 * @brief The facade for managing and executing business logic rules.
 *
 * The RuleEngine orchestrates the loading of rules, dispatching of events,
 * and management of alerts. It is thread-safe for event processing.
 */
class RuleEngine {
public:
  /**
   * @brief Constructs the RuleEngine with a set of configurations.
   * @param configs A vector of RuleConfig objects defining the rules to enable.
   *
   * @note Uses the Pimpl idiom. The implementation details are hidden in
   * RuleEngine::Impl.
   */
  RuleEngine(const std::vector<RuleConfig> &configs);
  RuleEngine(const std::vector<RuleConfig> &configs,
             const AlertManagerParams &alert_params);

  /**
   * @brief Destructor. Required for Pimpl unique_ptr.
   */
  ~RuleEngine();

  // Prevent Copying to avoid ambiguous pointer ownership
  RuleEngine(const RuleEngine &) = delete;
  RuleEngine &operator=(const RuleEngine &) = delete;

  // Move is allowed
  RuleEngine(RuleEngine &&) noexcept;
  RuleEngine &operator=(RuleEngine &&) noexcept;

  /**
   * @brief Processes a single vision event.
   * @param event The event metadata containing detections and timestamp.
   * @return A vector of generated Alerts (if any).
   *
   * @note This method is thread-safe if the underlying Dispatcher is
   * thread-safe (framework dependent). In this implementation, it delegates to
   * Pimpl for logic.
   */
  std::vector<Alert> processEvent(const EventMeta &event);

private:
  class Impl; // Opaque Pointer
  std::unique_ptr<Impl> pImpl;
};

} // namespace core

#endif // RULE_ENGINE_H
