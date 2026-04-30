#include "aggregator/Aggregator.hpp"
#include "utils/JsonlWriter.hpp"
#include "utils/mqtt/MqttPublisher.hpp"

#include "blockingconcurrentqueue.h"

#include <chrono>
#include <cstdio>
#include <sstream>

namespace multi_cam_app::aggregator {

struct Aggregator::QueueImpl {
  moodycamel::BlockingConcurrentQueue<QueueItem> queue;

  explicit QueueImpl(size_t initial_capacity)
      : queue(initial_capacity) {}
};

Aggregator::Aggregator(const AggregatorConfig& config)
    : config_(config),
      queue_(std::make_unique<QueueImpl>(config.queue_capacity)) {}

Aggregator::~Aggregator() {
  Stop();
}

void Aggregator::SetMqttPublisher(mqtt::MqttPublisher* publisher) {
  mqtt_publisher_ = publisher;
}

bool Aggregator::Start(std::string* out_error) {
  if (consumer_thread_.joinable()) {
    if (out_error != nullptr) {
      *out_error = "Aggregator already started";
    }
    return false;
  }

  stop_requested_.store(false, std::memory_order_relaxed);
  consumer_thread_ = std::thread(&Aggregator::ConsumerLoop, this);

  std::printf("[Aggregator] Started (batch_size=%u, capacity=%u, drain_timeout=%ums)\n",
              config_.batch_size, config_.queue_capacity, config_.drain_timeout_ms);
  std::fflush(stdout);
  return true;
}

void Aggregator::Stop() {
  if (!consumer_thread_.joinable()) {
    return;
  }

  stop_requested_.store(true, std::memory_order_release);
  consumer_thread_.join();

  std::printf("[Aggregator] Stopped (enqueued=%lu, dropped=%lu, batches_sent=%lu)\n",
              static_cast<unsigned long>(enqueued_count_.load()),
              static_cast<unsigned long>(dropped_count_.load()),
              static_cast<unsigned long>(batches_sent_.load()));
  std::fflush(stdout);
}

bool Aggregator::TryEnqueue(QueueItem&& item) {
  size_t cur = current_size_.load(std::memory_order_acquire);
  while (true) {
    if (cur >= config_.queue_capacity) {
      dropped_count_.fetch_add(1ULL, std::memory_order_relaxed);
      return false;
    }
    if (current_size_.compare_exchange_weak(cur, cur + 1,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      break;
    }
  }
  queue_->queue.enqueue(std::move(item));
  enqueued_count_.fetch_add(1ULL, std::memory_order_relaxed);
  return true;
}

uint64_t Aggregator::EnqueuedCount() const {
  return enqueued_count_.load(std::memory_order_relaxed);
}

uint64_t Aggregator::DroppedCount() const {
  return dropped_count_.load(std::memory_order_relaxed);
}

uint64_t Aggregator::BatchesSentCount() const {
  return batches_sent_.load(std::memory_order_relaxed);
}

void Aggregator::ConsumerLoop() {
  std::vector<QueueItem> batch;
  batch.reserve(config_.batch_size);

  const auto timeout = std::chrono::milliseconds(config_.drain_timeout_ms);

  while (!stop_requested_.load(std::memory_order_acquire)) {
    QueueItem item;
    if (queue_->queue.wait_dequeue_timed(item, timeout)) {
      batch.push_back(std::move(item));
      current_size_.fetch_sub(1, std::memory_order_release);

      while (batch.size() < config_.batch_size) {
        QueueItem more;
        if (!queue_->queue.try_dequeue(more)) {
          break;
        }
        batch.push_back(std::move(more));
        current_size_.fetch_sub(1, std::memory_order_release);
      }
    }

    if (batch.size() >= config_.batch_size) {
      PublishBatch(batch);
      batch.clear();
    } else if (stop_requested_.load(std::memory_order_acquire) && !batch.empty()) {
      PublishBatch(batch);
      batch.clear();
    }
  }

  // Drain remaining items after stop
  QueueItem remaining;
  while (queue_->queue.try_dequeue(remaining)) {
    batch.push_back(std::move(remaining));
    current_size_.fetch_sub(1, std::memory_order_release);
  }
  if (!batch.empty()) {
    PublishBatch(batch);
    batch.clear();
  }
}

void Aggregator::PublishBatch(const std::vector<QueueItem>& batch) {
  std::string json = SerializeBatchJson(batch);

  if (mqtt_publisher_ != nullptr) {
    std::string err;
    if (!mqtt_publisher_->Publish(json, &err)) {
      std::fprintf(stderr, "[Aggregator] MQTT publish failed: %s\n", err.c_str());
    }
  }

  batches_sent_.fetch_add(1ULL, std::memory_order_relaxed);
}

std::string Aggregator::SerializeBatchJson(const std::vector<QueueItem>& batch) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < batch.size(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << output::JsonlWriter::Serialize(batch[i].record);
  }
  oss << "]";
  return oss.str();
}

}  // namespace multi_cam_app::aggregator
