#include "aggregator/Aggregator.hpp"
#include "aggregator/QueueItem.hpp"
#include "utils/mqtt/MqttPublisher.hpp"
#include "utils/mqtt/MqttSubscriber.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace multi_cam_app::aggregator;
using namespace multi_cam_app::output;

struct TestArgs {
  std::string group = "all";
  std::string broker_ip = "127.0.0.1";
  int broker_port = 1883;
};

TestArgs ParseArgs(int argc, char** argv) {
  TestArgs args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--group") == 0 && (i + 1) < argc) {
      args.group = argv[++i];
    } else if (std::strcmp(argv[i], "--broker-ip") == 0 && (i + 1) < argc) {
      args.broker_ip = argv[++i];
    } else if (std::strcmp(argv[i], "--broker-port") == 0 && (i + 1) < argc) {
      args.broker_port = std::atoi(argv[++i]);
    }
  }
  return args;
}

int passed = 0;
int failed = 0;

void Pass(const char* name) {
  std::printf("  [PASS] %s\n", name);
  std::fflush(stdout);
  ++passed;
}

void Fail(const char* name, const std::string& reason) {
  std::printf("  [FAIL] %s: %s\n", name, reason.c_str());
  std::fflush(stdout);
  ++failed;
}

QueueItem MakeItem(uint32_t ch, uint64_t frame, uint64_t ts = 0) {
  QueueItem item;
  item.record.channel_id = ch;
  item.record.frame_id = frame;
  item.record.timestamp_ns = ts;
  return item;
}

QueueItem MakeItemWithDetection(uint32_t ch, uint64_t frame) {
  QueueItem item;
  item.record.channel_id = ch;
  item.record.frame_id = frame;
  item.record.timestamp_ns = frame * 1000000ULL;

  DetectionRecord det;
  det.class_id = 0;
  det.track_id = 1;
  det.label = "person";
  det.score = 0.95;
  det.left = 0.1;
  det.top = 0.2;
  det.right = 0.3;
  det.bottom = 0.4;
  item.record.detections.push_back(std::move(det));
  return item;
}

// =============================================================================
// Group A: Pure C++ unit tests (no MQTT broker)
// =============================================================================

void test_a1_enqueue_consume() {
  std::printf("\n--- A1: Enqueue + Consume ---\n");

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 6;
  cfg.queue_capacity = 6;
  cfg.drain_timeout_ms = 500;

  Aggregator agg(cfg);
  std::string err;
  if (!agg.Start(&err)) {
    Fail("A1_start", err);
    return;
  }

  for (uint32_t i = 0; i < 6; ++i) {
    if (!agg.TryEnqueue(MakeItem(i, i * 10))) {
      Fail("A1_enqueue", "enqueue " + std::to_string(i) + " failed");
      agg.Stop();
      return;
    }
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  if (agg.EnqueuedCount() != 6) {
    Fail("A1_enqueued_count",
         "expected 6, got " + std::to_string(agg.EnqueuedCount()));
    agg.Stop();
    return;
  }
  if (agg.DroppedCount() != 0) {
    Fail("A1_dropped_count",
         "expected 0, got " + std::to_string(agg.DroppedCount()));
    agg.Stop();
    return;
  }
  if (agg.BatchesSentCount() != 1) {
    Fail("A1_batches_sent",
         "expected 1, got " + std::to_string(agg.BatchesSentCount()));
    agg.Stop();
    return;
  }

  agg.Stop();
  Pass("A1_enqueue_consume");
}

void test_a2_capacity_overflow() {
  std::printf("\n--- A2: Capacity Overflow ---\n");

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 100;
  cfg.queue_capacity = 3;
  cfg.drain_timeout_ms = 2000;

  Aggregator agg(cfg);
  std::string err;
  if (!agg.Start(&err)) {
    Fail("A2_start", err);
    return;
  }

  int enqueued = 0;
  int dropped = 0;
  for (uint32_t i = 0; i < 6; ++i) {
    if (agg.TryEnqueue(MakeItem(i, i))) {
      ++enqueued;
    } else {
      ++dropped;
    }
  }

  if (agg.EnqueuedCount() + agg.DroppedCount() != 6) {
    Fail("A2_total",
         "enqueued+dropped should be 6, got " +
         std::to_string(agg.EnqueuedCount() + agg.DroppedCount()));
    agg.Stop();
    return;
  }
  if (agg.DroppedCount() < 1) {
    Fail("A2_dropped", "should have dropped at least 1 item");
    agg.Stop();
    return;
  }

  agg.Stop();
  Pass("A2_capacity_overflow");
}

void test_a3_batch_trigger() {
  std::printf("\n--- A3: Batch Trigger ---\n");

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 3;
  cfg.queue_capacity = 10;
  cfg.drain_timeout_ms = 500;

  Aggregator agg(cfg);
  std::string err;
  if (!agg.Start(&err)) {
    Fail("A3_start", err);
    return;
  }

  for (uint32_t i = 0; i < 3; ++i) {
    agg.TryEnqueue(MakeItem(0, i));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  if (agg.BatchesSentCount() != 1) {
    Fail("A3_first_batch",
         "expected 1 batch, got " + std::to_string(agg.BatchesSentCount()));
    agg.Stop();
    return;
  }

  for (uint32_t i = 0; i < 2; ++i) {
    agg.TryEnqueue(MakeItem(0, 10 + i));
  }

  agg.Stop();

  if (agg.BatchesSentCount() != 2) {
    Fail("A3_drain_batch",
         "expected 2 batches after Stop, got " +
         std::to_string(agg.BatchesSentCount()));
    return;
  }

  Pass("A3_batch_trigger");
}

void test_a4_timeout_drain() {
  std::printf("\n--- A4: Timeout Drain ---\n");

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 6;
  cfg.queue_capacity = 10;
  cfg.drain_timeout_ms = 200;

  Aggregator agg(cfg);
  std::string err;
  if (!agg.Start(&err)) {
    Fail("A4_start", err);
    return;
  }

  agg.TryEnqueue(MakeItem(0, 1));
  agg.TryEnqueue(MakeItem(1, 2));

  // Wait longer than drain_timeout_ms so timeout fires
  // but batch_size not reached, so timeout alone doesn't send
  // Only stop will drain
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  agg.Stop();

  if (agg.EnqueuedCount() != 2) {
    Fail("A4_enqueued", "expected 2, got " + std::to_string(agg.EnqueuedCount()));
    return;
  }
  if (agg.BatchesSentCount() < 1) {
    Fail("A4_drain", "expected at least 1 batch after stop drain");
    return;
  }

  Pass("A4_timeout_drain");
}

void test_a5_stop_drain() {
  std::printf("\n--- A5: Stop Drain ---\n");

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 10;
  cfg.queue_capacity = 10;
  cfg.drain_timeout_ms = 5000;

  Aggregator agg(cfg);
  std::string err;
  if (!agg.Start(&err)) {
    Fail("A5_start", err);
    return;
  }

  for (uint32_t i = 0; i < 4; ++i) {
    agg.TryEnqueue(MakeItem(i, i));
  }

  agg.Stop();

  if (agg.BatchesSentCount() != 1) {
    Fail("A5_drain_batch",
         "expected 1 drain batch, got " + std::to_string(agg.BatchesSentCount()));
    return;
  }
  if (agg.EnqueuedCount() != 4) {
    Fail("A5_enqueued", "expected 4, got " + std::to_string(agg.EnqueuedCount()));
    return;
  }

  Pass("A5_stop_drain");
}

void test_a6_serialize_json() {
  std::printf("\n--- A6: SerializeBatchJson ---\n");

  std::vector<QueueItem> batch;
  batch.push_back(MakeItemWithDetection(0, 42));
  batch.push_back(MakeItemWithDetection(1, 18));

  std::string json = Aggregator::SerializeBatchJson(batch);

  if (json.empty()) {
    Fail("A6_empty", "serialized JSON is empty");
    return;
  }
  if (json.front() != '[' || json.back() != ']') {
    Fail("A6_array", "JSON should be array, got: " + json.substr(0, 50));
    return;
  }
  if (json.find("\"channel_id\":0") == std::string::npos) {
    Fail("A6_ch0", "missing channel_id 0");
    return;
  }
  if (json.find("\"channel_id\":1") == std::string::npos) {
    Fail("A6_ch1", "missing channel_id 1");
    return;
  }
  if (json.find("\"frame_id\":42") == std::string::npos) {
    Fail("A6_frame42", "missing frame_id 42");
    return;
  }
  if (json.find("\"label\":\"person\"") == std::string::npos) {
    Fail("A6_label", "missing label person");
    return;
  }
  if (json.find("\"detections\":[") == std::string::npos) {
    Fail("A6_detections", "missing detections array");
    return;
  }

  Pass("A6_serialize_json");
}

void test_a7_empty_stop() {
  std::printf("\n--- A7: Empty Stop ---\n");

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 6;
  cfg.queue_capacity = 6;
  cfg.drain_timeout_ms = 200;

  Aggregator agg(cfg);
  std::string err;
  if (!agg.Start(&err)) {
    Fail("A7_start", err);
    return;
  }

  agg.Stop();

  if (agg.EnqueuedCount() != 0) {
    Fail("A7_enqueued", "expected 0");
    return;
  }
  if (agg.BatchesSentCount() != 0) {
    Fail("A7_batches", "expected 0 batches");
    return;
  }

  Pass("A7_empty_stop");
}

void test_a8_multi_producer() {
  std::printf("\n--- A8: Multi-Producer Concurrent ---\n");

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 6;
  cfg.queue_capacity = 600;
  cfg.drain_timeout_ms = 500;

  Aggregator agg(cfg);
  std::string err;
  if (!agg.Start(&err)) {
    Fail("A8_start", err);
    return;
  }

  constexpr int kThreads = 6;
  constexpr int kPerThread = 100;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&agg, t]() {
      for (int i = 0; i < kPerThread; ++i) {
        agg.TryEnqueue(MakeItem(static_cast<uint32_t>(t),
                                static_cast<uint64_t>(i)));
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  agg.Stop();

  const uint64_t total = agg.EnqueuedCount() + agg.DroppedCount();
  if (total != kThreads * kPerThread) {
    Fail("A8_total",
         "expected " + std::to_string(kThreads * kPerThread) +
         ", got enqueued=" + std::to_string(agg.EnqueuedCount()) +
         " dropped=" + std::to_string(agg.DroppedCount()));
    return;
  }

  if (agg.BatchesSentCount() < 1) {
    Fail("A8_batches", "expected at least 1 batch sent");
    return;
  }

  std::printf("    enqueued=%lu, dropped=%lu, batches=%lu\n",
              static_cast<unsigned long>(agg.EnqueuedCount()),
              static_cast<unsigned long>(agg.DroppedCount()),
              static_cast<unsigned long>(agg.BatchesSentCount()));

  Pass("A8_multi_producer");
}

// =============================================================================
// Group B: MQTT Integration Tests (requires mosquitto broker)
// =============================================================================

void test_b1_e2e_batch(const TestArgs& args) {
  std::printf("\n--- B1: E2E Batch Send ---\n");

  using namespace multi_cam_app::mqtt;
  std::string err;
  const std::string topic = "test/aggregator/b1";

  MqttSubscriber sub(args.broker_ip, args.broker_port, topic, 1);
  if (!sub.Initialize("test_agg_b1_sub", &err)) {
    Fail("B1_sub_init", err);
    return;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  MqttPublisher pub(args.broker_ip, args.broker_port, topic, 1);
  if (!pub.Initialize("test_agg_b1_pub", &err)) {
    Fail("B1_pub_init", err);
    sub.Shutdown();
    return;
  }

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 3;
  cfg.queue_capacity = 6;
  cfg.drain_timeout_ms = 500;

  Aggregator agg(cfg);
  agg.SetMqttPublisher(&pub);
  if (!agg.Start(&err)) {
    Fail("B1_agg_start", err);
    pub.Shutdown();
    sub.Shutdown();
    return;
  }

  for (uint32_t i = 0; i < 3; ++i) {
    agg.TryEnqueue(MakeItemWithDetection(i, 100 + i));
  }

  std::string recv_topic, recv_payload;
  bool received = false;
  for (int i = 0; i < 30; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (sub.PopMessage(&recv_topic, &recv_payload)) {
      received = true;
      break;
    }
  }

  agg.Stop();

  if (!received) {
    Fail("B1_receive", "no message received within 3s");
    pub.Shutdown();
    sub.Shutdown();
    return;
  }
  if (recv_payload.front() != '[' || recv_payload.back() != ']') {
    Fail("B1_format", "payload not JSON array: " + recv_payload.substr(0, 80));
    pub.Shutdown();
    sub.Shutdown();
    return;
  }
  if (recv_payload.find("\"channel_id\":0") == std::string::npos ||
      recv_payload.find("\"channel_id\":1") == std::string::npos ||
      recv_payload.find("\"channel_id\":2") == std::string::npos) {
    Fail("B1_content", "missing channel_id 0/1/2 in payload");
    pub.Shutdown();
    sub.Shutdown();
    return;
  }

  pub.Shutdown();
  sub.Shutdown();
  Pass("B1_e2e_batch");
}

void test_b2_multi_batch(const TestArgs& args) {
  std::printf("\n--- B2: Multi-Batch Send ---\n");

  using namespace multi_cam_app::mqtt;
  std::string err;
  const std::string topic = "test/aggregator/b2";

  MqttSubscriber sub(args.broker_ip, args.broker_port, topic, 1);
  if (!sub.Initialize("test_agg_b2_sub", &err)) {
    Fail("B2_sub_init", err);
    return;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  MqttPublisher pub(args.broker_ip, args.broker_port, topic, 1);
  if (!pub.Initialize("test_agg_b2_pub", &err)) {
    Fail("B2_pub_init", err);
    sub.Shutdown();
    return;
  }

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 3;
  cfg.queue_capacity = 20;
  cfg.drain_timeout_ms = 500;

  Aggregator agg(cfg);
  agg.SetMqttPublisher(&pub);
  if (!agg.Start(&err)) {
    Fail("B2_agg_start", err);
    pub.Shutdown();
    sub.Shutdown();
    return;
  }

  for (uint32_t i = 0; i < 9; ++i) {
    agg.TryEnqueue(MakeItemWithDetection(i % 3, i));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  agg.Stop();

  int msg_count = 0;
  std::string t, p;
  while (sub.PopMessage(&t, &p)) {
    ++msg_count;
  }

  if (msg_count != 3) {
    Fail("B2_batch_count",
         "expected 3 messages, got " + std::to_string(msg_count));
    pub.Shutdown();
    sub.Shutdown();
    return;
  }

  pub.Shutdown();
  sub.Shutdown();
  Pass("B2_multi_batch");
}

void test_b3_stop_drain_to_broker(const TestArgs& args) {
  std::printf("\n--- B3: Stop Drain to Broker ---\n");

  using namespace multi_cam_app::mqtt;
  std::string err;
  const std::string topic = "test/aggregator/b3";

  MqttSubscriber sub(args.broker_ip, args.broker_port, topic, 1);
  if (!sub.Initialize("test_agg_b3_sub", &err)) {
    Fail("B3_sub_init", err);
    return;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  MqttPublisher pub(args.broker_ip, args.broker_port, topic, 1);
  if (!pub.Initialize("test_agg_b3_pub", &err)) {
    Fail("B3_pub_init", err);
    sub.Shutdown();
    return;
  }

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 3;
  cfg.queue_capacity = 10;
  cfg.drain_timeout_ms = 5000;

  Aggregator agg(cfg);
  agg.SetMqttPublisher(&pub);
  if (!agg.Start(&err)) {
    Fail("B3_agg_start", err);
    pub.Shutdown();
    sub.Shutdown();
    return;
  }

  for (uint32_t i = 0; i < 4; ++i) {
    agg.TryEnqueue(MakeItemWithDetection(i, i));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  agg.Stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  int msg_count = 0;
  std::string t, p;
  while (sub.PopMessage(&t, &p)) {
    ++msg_count;
  }

  if (msg_count != 2) {
    Fail("B3_msg_count",
         "expected 2 messages (batch=3 + drain=1), got " + std::to_string(msg_count));
    pub.Shutdown();
    sub.Shutdown();
    return;
  }

  pub.Shutdown();
  sub.Shutdown();
  Pass("B3_stop_drain_to_broker");
}

void test_b4_payload_format(const TestArgs& args) {
  std::printf("\n--- B4: Payload Format Validation ---\n");

  using namespace multi_cam_app::mqtt;
  std::string err;
  const std::string topic = "test/aggregator/b4";

  MqttSubscriber sub(args.broker_ip, args.broker_port, topic, 1);
  if (!sub.Initialize("test_agg_b4_sub", &err)) {
    Fail("B4_sub_init", err);
    return;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  MqttPublisher pub(args.broker_ip, args.broker_port, topic, 1);
  if (!pub.Initialize("test_agg_b4_pub", &err)) {
    Fail("B4_pub_init", err);
    sub.Shutdown();
    return;
  }

  AggregatorConfig cfg;
  cfg.enabled = true;
  cfg.batch_size = 2;
  cfg.queue_capacity = 10;
  cfg.drain_timeout_ms = 500;

  Aggregator agg(cfg);
  agg.SetMqttPublisher(&pub);
  if (!agg.Start(&err)) {
    Fail("B4_agg_start", err);
    pub.Shutdown();
    sub.Shutdown();
    return;
  }

  agg.TryEnqueue(MakeItemWithDetection(0, 99));
  agg.TryEnqueue(MakeItemWithDetection(1, 200));

  std::string recv_topic, payload;
  bool received = false;
  for (int i = 0; i < 30; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (sub.PopMessage(&recv_topic, &payload)) {
      received = true;
      break;
    }
  }

  agg.Stop();

  if (!received) {
    Fail("B4_receive", "no message within 3s");
    pub.Shutdown();
    sub.Shutdown();
    return;
  }

  bool ok = true;
  auto check = [&](const char* field) {
    if (payload.find(field) == std::string::npos) {
      Fail("B4_field", std::string("missing field: ") + field);
      ok = false;
    }
  };

  check("\"channel_id\":");
  check("\"frame_id\":");
  check("\"timestamp_ns\":");
  check("\"detections\":[");
  check("\"class_id\":");
  check("\"track_id\":");
  check("\"label\":");
  check("\"score\":");
  check("\"bbox\":[");

  if (ok) {
    Pass("B4_payload_format");
  }

  pub.Shutdown();
  sub.Shutdown();
}

}  // namespace

int main(int argc, char** argv) {
  const TestArgs args = ParseArgs(argc, argv);

  std::printf("============================================\n");
  std::printf("  Aggregator Test (group=%s)\n", args.group.c_str());
  std::printf("============================================\n");
  std::fflush(stdout);

  if (args.group == "A" || args.group == "all") {
    std::printf("\n=== Group A: Aggregator Unit Tests ===\n");
    std::fflush(stdout);

    test_a1_enqueue_consume();
    test_a2_capacity_overflow();
    test_a3_batch_trigger();
    test_a4_timeout_drain();
    test_a5_stop_drain();
    test_a6_serialize_json();
    test_a7_empty_stop();
    test_a8_multi_producer();
  }

  if (args.group == "B" || args.group == "all") {
    std::printf("\n=== Group B: MQTT Integration Tests ===\n");
    std::printf("  Broker: %s:%d\n", args.broker_ip.c_str(), args.broker_port);
    std::fflush(stdout);

    test_b1_e2e_batch(args);
    test_b2_multi_batch(args);
    test_b3_stop_drain_to_broker(args);
    test_b4_payload_format(args);
  }

  std::printf("\n============================================\n");
  std::printf("  Results: %d passed, %d failed\n", passed, failed);
  std::printf("============================================\n");
  std::fflush(stdout);

  return (failed > 0) ? 1 : 0;
}
