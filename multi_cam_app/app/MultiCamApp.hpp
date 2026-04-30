#ifndef MULTI_CAM_APP_APP_MULTI_CAM_APP_HPP
#define MULTI_CAM_APP_APP_MULTI_CAM_APP_HPP

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gst/gst.h>

#include "aggregator/Aggregator.hpp"
#include "app/AppContext.hpp"
#include "channel_worker/ChannelWorker.hpp"
#include "config/AppConfig.hpp"
#include "frame_cache/FrameCacheService.hpp"
#include "utils/FaceGallery.hpp"
#include "utils/mqtt/MqttPublisher.hpp"

namespace multi_cam_app::app {

class MultiCamApp {
public:
  explicit MultiCamApp(std::string config_path);
  ~MultiCamApp();

  MultiCamApp(const MultiCamApp&) = delete;
  MultiCamApp& operator=(const MultiCamApp&) = delete;

  int Run();

private:
  bool Initialize(std::string* out_error);
  bool StartPipeline(std::string* out_error);
  void StopPipeline();
  void ReleaseMainLoop();

  static gboolean OnBusMessage(GstBus* bus, GstMessage* message, gpointer user_data);

  // Memtest methods
  bool SeekPipelineToStart();
  void StartMemoryMonitor();
  void StopMemoryMonitor();
  static gboolean OnMemoryMonitorTick(gpointer user_data);
  void SampleMemory(const char* event_tag);
  static bool ReadProcSelfStatus(uint64_t* out_vm_rss_kb, uint64_t* out_vm_size_kb);
  void WriteMemtestSummary();

  std::string config_path_ {};
  config::AppConfig config_ {};
  AppContext app_context_ {};
  std::vector<std::unique_ptr<channel_worker::ChannelWorker>> workers_ {};
  std::unique_ptr<gallery::FaceGallery> shared_gallery_;
  std::unique_ptr<mqtt::MqttPublisher> mqtt_publisher_;
  std::unique_ptr<aggregator::Aggregator> aggregator_;
  std::unique_ptr<frame_cache::FrameCacheService> frame_cache_service_;
  GstElement* pipeline_ {nullptr};
  guint bus_watch_id_ {0U};
  GMainLoop* main_loop_ {nullptr};

  // Memtest state
  uint32_t current_iteration_ {0U};
  guint mem_monitor_timer_id_ {0U};
  FILE* memtest_log_ {nullptr};
  uint64_t memtest_start_rss_kb_ {0ULL};
  uint64_t memtest_peak_rss_kb_ {0ULL};
  uint64_t memtest_sample_count_ {0ULL};
};

}  // namespace multi_cam_app::app

#endif  // MULTI_CAM_APP_APP_MULTI_CAM_APP_HPP
