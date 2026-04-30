#ifndef MULTI_CAM_APP_CONFIG_APP_CONFIG_HPP
#define MULTI_CAM_APP_CONFIG_APP_CONFIG_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace multi_cam_app::config {

struct AppConfig {
  std::string model_path {};
  std::vector<std::string> videos_path {};
  std::string labels_path {};
  std::string output_dir {};
  std::string log_dir {};

  std::string qnn_backend {"/usr/lib/libQnnHtp.so"};
  std::string qnn_system {"/usr/lib/libQnnSystem.so"};
  uint32_t qnn_backend_device_id {0U};
  std::vector<std::string> qnn_tensors {"boxes", "scores", "class_idx"};
  std::vector<std::string> qtimlvconverter_engine_order {"fcv", "c2d"};
  std::string qtiobjtracker_parameters {};

  std::string postprocess_module {"yolov8"};
  double confidence {40.0};

  uint32_t max_channels {6U};
  bool appsink_sync {false};
  bool appsink_drop {false};
  uint32_t appsink_max_buffers {8U};

  // Face module config
  bool face_enabled {false};
  int  face_channel_mask {0xFF};    // bitmask: bit N enables face for channel N; default=all
  std::string face_config_path {};
  std::string gallery_path {};
  float gallery_threshold {0.3f};
  uint32_t gallery_min_face_size {40U};  // min face bbox px for enrollment
  float gallery_min_score {0.6f};        // min SCRFD score for enrollment
  uint32_t face_interval_ms {0U};        // 0=every frame; N=min ms between face detections (e.g., 1000=1fps)

  // Memory leak testing
  uint32_t memtest_loop_count {1U};            // 1=single pass(default), 0=infinite
  uint32_t memtest_mem_sample_interval_s {5U}; // seconds between /proc/self/status samples
  std::string memtest_log_path {};             // override; defaults to log_dir/memtest.log

  // Aggregator config (concurrent queue + batch MQTT send)
  struct AggregatorConfig {
    bool enabled {false};
    uint32_t batch_size {6U};
    uint32_t queue_capacity {6U};
    uint32_t drain_timeout_ms {1000U};
  };
  AggregatorConfig aggregator {};

  // MQTT publish/subscribe config
  struct MqttConfig {
    bool enabled {false};
    std::string broker_ip {"127.0.0.1"};
    int port {1883};
    std::string pub_topic {"multi_cam_app/detections"};
    std::string sub_topic {"multi_cam_app/commands"};
    std::string client_id {"multi_cam_app"};
    int qos {1};
  };
  MqttConfig mqtt {};

  // FrameCache (DmaBufProducer + MQTT frame_done): DMA slot pool for NV12->RGB24 frame storage
  struct FrameCacheConfig {
    bool enabled {false};
    std::string socket_path {"/tmp/dmabuf_ipc.sock"};
    std::string heap_path {"/dev/dma_heap/qcom,system"};
    int slot_count {8};
    int width {1920};
    int height {1080};
    bool relay_mode {true};
    // Same key/topic as rule_process publish; must be set when frame_cache is enabled
    std::string frame_done_topic {};
  };
  FrameCacheConfig frame_cache {};

  // MsgAgg config (GStreamer-level N-channel JSON aggregation → qtimsgpub MQTT)
  struct MsgAggConfig {
    bool enabled {false};
    uint32_t timeout_ms {500U};
    std::string scene_jsonl_dir {};  // if non-empty, write scene_update JSONL per channel here
  };
  MsgAggConfig msgagg {};

  // Latency test config (Step 26 pad-probe timing)
  struct LatencyTestConfig {
    bool enabled {false};
    std::string output_dir {};
    uint32_t sample_every_n {1U};
    bool flush_per_sample {true};
  };
  LatencyTestConfig latency_test {};
};

}  // namespace multi_cam_app::config

#endif  // MULTI_CAM_APP_CONFIG_APP_CONFIG_HPP
