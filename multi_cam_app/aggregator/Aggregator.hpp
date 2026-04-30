#ifndef MULTI_CAM_APP_AGGREGATOR_AGGREGATOR_HPP
#define MULTI_CAM_APP_AGGREGATOR_AGGREGATOR_HPP

#include "aggregator/QueueItem.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace multi_cam_app::mqtt {
class MqttPublisher;
}

namespace multi_cam_app::aggregator {

struct AggregatorConfig {
  bool enabled {false};
  uint32_t batch_size {6U};
  uint32_t queue_capacity {6U};
  uint32_t drain_timeout_ms {1000U};
};

class Aggregator {
public:
  explicit Aggregator(const AggregatorConfig& config);
  ~Aggregator();

  Aggregator(const Aggregator&) = delete;
  Aggregator& operator=(const Aggregator&) = delete;

  void SetMqttPublisher(mqtt::MqttPublisher* publisher);
  bool Start(std::string* out_error);
  void Stop();

  bool TryEnqueue(QueueItem&& item);

  uint64_t EnqueuedCount() const;
  uint64_t DroppedCount() const;
  uint64_t BatchesSentCount() const;

  static std::string SerializeBatchJson(const std::vector<QueueItem>& batch);

private:
  void ConsumerLoop();
  void PublishBatch(const std::vector<QueueItem>& batch);

  AggregatorConfig config_;
  mqtt::MqttPublisher* mqtt_publisher_ {nullptr};

  struct QueueImpl;
  std::unique_ptr<QueueImpl> queue_;
  std::atomic<size_t> current_size_ {0};

  std::thread consumer_thread_;
  std::atomic<bool> stop_requested_ {false};

  std::atomic<uint64_t> enqueued_count_ {0};
  std::atomic<uint64_t> dropped_count_ {0};
  std::atomic<uint64_t> batches_sent_ {0};
};

}  // namespace multi_cam_app::aggregator

#endif  // MULTI_CAM_APP_AGGREGATOR_AGGREGATOR_HPP
