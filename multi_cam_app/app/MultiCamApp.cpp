#include "app/MultiCamApp.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <utility>

#include "config/ConfigLoader.hpp"

namespace multi_cam_app::app {

MultiCamApp::MultiCamApp(std::string config_path)
    : config_path_(std::move(config_path)) {}

MultiCamApp::~MultiCamApp() {
  StopMemoryMonitor();
  StopPipeline();
  ReleaseMainLoop();
}

int MultiCamApp::Run() {
  std::string error;
  if (!Initialize(&error)) {
    std::printf("MultiCamApp init failed: %s\n", error.c_str());
    StopMemoryMonitor();
    StopPipeline();
    ReleaseMainLoop();
    return EXIT_FAILURE;
  }

  if (!StartPipeline(&error)) {
    std::printf("MultiCamApp start failed: %s\n", error.c_str());
    StopMemoryMonitor();
    StopPipeline();
    ReleaseMainLoop();
    return EXIT_FAILURE;
  }

  std::printf("MultiCamApp running with %u channels (loop_count=%u)\n",
              static_cast<unsigned>(workers_.size()),
              config_.memtest_loop_count);
  std::fflush(stdout);
  g_main_loop_run(main_loop_);

  StopMemoryMonitor();
  StopPipeline();
  ReleaseMainLoop();

  if (app_context_.error_received.load()) {
    std::printf("MultiCamApp stopped due to pipeline error\n");
    return EXIT_FAILURE;
  }

  std::printf("MultiCamApp stopped after all channels EOS\n");
  return EXIT_SUCCESS;
}

bool MultiCamApp::Initialize(std::string* out_error) {
  gst_init(nullptr, nullptr);

  if (!config::ConfigLoader::LoadFromFile(config_path_, &config_, out_error)) {
    return false;
  }

  if (config_.videos_path.empty()) {
    if (out_error != nullptr) {
      *out_error = "videos_path is empty after config parse";
    }
    return false;
  }

  main_loop_ = g_main_loop_new(nullptr, FALSE);
  if (main_loop_ == nullptr) {
    if (out_error != nullptr) {
      *out_error = "Failed to create GMainLoop";
    }
    return false;
  }

  app_context_.main_loop = main_loop_;

  // Create single shared pipeline
  pipeline_ = gst_pipeline_new("multi-cam-pipeline");
  if (pipeline_ == nullptr) {
    if (out_error != nullptr) {
      *out_error = "Failed to create shared GstPipeline";
    }
    return false;
  }

  // Create shared gallery (once, before workers) if face is enabled
  gallery::FaceGallery* gallery_ptr = nullptr;
  if (config_.face_enabled && !config_.gallery_path.empty()) {
    shared_gallery_ = std::make_unique<gallery::FaceGallery>();
    if (shared_gallery_->Load(config_.gallery_path)) {
      std::printf("Shared gallery loaded: %zu entries\n", shared_gallery_->Size());
      std::fflush(stdout);
      gallery_ptr = shared_gallery_.get();
    } else {
      std::printf("Shared gallery load failed, auto-enrollment disabled\n");
      std::fflush(stdout);
      shared_gallery_.reset();
    }
  }

  workers_.clear();
  workers_.reserve(config_.videos_path.size());
  for (size_t i = 0U; i < config_.videos_path.size(); ++i) {
    auto worker = std::make_unique<channel_worker::ChannelWorker>(
        static_cast<uint32_t>(i), config_.videos_path[i]);
    if (!worker->Initialize(config_, &app_context_, GST_BIN(pipeline_),
                            gallery_ptr, out_error)) {
      return false;
    }
    std::printf("Channel %u output: %s\n",
                static_cast<unsigned>(i),
                worker->OutputJsonlPath().c_str());
    std::fflush(stdout);
    workers_.push_back(std::move(worker));
  }

  // Mutual exclusion: msgagg (GStreamer-level) replaces C++ aggregator
  if (config_.msgagg.enabled && config_.aggregator.enabled) {
    std::printf("WARN: msgagg and aggregator both enabled, msgagg takes priority\n");
    std::fflush(stdout);
    config_.aggregator.enabled = false;
  }

  // Initialize FrameCache (DmaBufProducer + MQTT frame_done) if enabled
  if (config_.frame_cache.enabled) {
    frame_cache_service_ = std::make_unique<frame_cache::FrameCacheService>();
    frame_cache::FrameCacheService::InitParams fcp;
    fcp.producer.socket_path = config_.frame_cache.socket_path.c_str();
    fcp.producer.heap_path   = config_.frame_cache.heap_path.c_str();
    fcp.producer.slot_count  = config_.frame_cache.slot_count;
    fcp.producer.width       = config_.frame_cache.width;
    fcp.producer.height      = config_.frame_cache.height;
    fcp.producer.relay_mode  = config_.frame_cache.relay_mode;
    fcp.broker_ip            = config_.mqtt.broker_ip;
    fcp.port                 = config_.mqtt.port;
    fcp.frame_done_topic     = config_.frame_cache.frame_done_topic;
    fcp.mqtt_client_id       = config_.mqtt.client_id;
    fcp.mqtt_qos             = config_.mqtt.qos;

    std::printf("FrameCache: waiting for consumer on %s; MQTT frame_done on '%s' ...\n",
                config_.frame_cache.socket_path.c_str(),
                config_.frame_cache.frame_done_topic.c_str());
    std::fflush(stdout);

    std::string fc_err;
    if (!frame_cache_service_->Init(fcp, &fc_err)) {
      std::fprintf(stderr, "FrameCacheService init failed: %s\n", fc_err.c_str());
      std::fflush(stderr);
      if (out_error != nullptr) {
        *out_error = fc_err;
      }
      return false;
    }
    std::printf("FrameCache: connected, %d slots (%dx%d RGB24, %.1f MB total)\n",
                config_.frame_cache.slot_count,
                config_.frame_cache.width, config_.frame_cache.height,
                config_.frame_cache.slot_count *
                  config_.frame_cache.width * config_.frame_cache.height * 3.0 / 1048576.0);
    std::fflush(stdout);

    for (auto& worker : workers_) {
      worker->SetFrameCacheProducer(frame_cache_service_->Producer());
    }

    if (gallery_ptr != nullptr && config_.msgagg.enabled) {
      for (auto& worker : workers_) {
        worker->SetFaceGallery(gallery_ptr);
      }
    }
  }

  // Initialize GStreamer-level message aggregation + MQTT publish
  if (config_.msgagg.enabled && config_.frame_cache.enabled && config_.mqtt.enabled) {
    GstElement* msgagg = gst_element_factory_make("qtimsgagg", "msgagg");
    GstElement* msgpub = gst_element_factory_make("qtimsgpub", "msgpub");
    if (msgagg == nullptr || msgpub == nullptr) {
      std::fprintf(stderr, "MsgAgg: failed to create qtimsgagg/qtimsgpub (plugin not installed?)\n");
      std::fflush(stderr);
      if (msgagg != nullptr) gst_object_unref(msgagg);
      if (msgpub != nullptr) gst_object_unref(msgpub);
    } else {
      g_object_set(G_OBJECT(msgagg),
          "timeout", config_.msgagg.timeout_ms,
          "merge-scene-update", static_cast<gboolean>(config_.frame_cache.enabled),
          nullptr);
      // json=TRUE: publish buffer as JSON text for broker; rule_process uses nlohmann::parse on payload.
      // json=FALSE caused non-JSON MQTT bodies and "payload not JSON" in rule_process.
      g_object_set(G_OBJECT(msgpub),
          "protocol", "mqtt",
          "host", config_.mqtt.broker_ip.c_str(),
          "port", config_.mqtt.port,
          "topic", config_.mqtt.pub_topic.c_str(),
          "json", TRUE,
          nullptr);

      gst_bin_add_many(GST_BIN(pipeline_), msgagg, msgpub, nullptr);
      if (!gst_element_link(msgagg, msgpub)) {
        std::fprintf(stderr, "MsgAgg: failed to link qtimsgagg -> qtimsgpub\n");
        std::fflush(stderr);
      }

      size_t linked_count = 0;
      for (size_t i = 0; i < workers_.size(); ++i) {
        std::string queue_name = "queue_agg_ch" + std::to_string(i);
        GstElement* qa = gst_bin_get_by_name(GST_BIN(pipeline_), queue_name.c_str());
        if (qa == nullptr) continue;

        GstPad* agg_data = gst_element_request_pad_simple(msgagg, "data_%u");
        GstPad* queue_src = gst_element_get_static_pad(qa, "src");
        if (agg_data != nullptr && queue_src != nullptr) {
          if (gst_pad_link(queue_src, agg_data) == GST_PAD_LINK_OK) {
            ++linked_count;
          } else {
            std::fprintf(stderr, "MsgAgg: failed to link %s -> qtimsgagg\n",
                         queue_name.c_str());
            std::fflush(stderr);
          }
        }
        if (queue_src != nullptr) gst_object_unref(queue_src);
        if (agg_data != nullptr) gst_object_unref(agg_data);
        gst_object_unref(qa);
      }

      std::printf("MsgAgg: %zu/%zu channels → qtimsgagg(timeout=%u) → "
                  "qtimsgpub(mqtt://%s:%d/%s)\n",
                  linked_count, workers_.size(),
                  config_.msgagg.timeout_ms,
                  config_.mqtt.broker_ip.c_str(), config_.mqtt.port,
                  config_.mqtt.pub_topic.c_str());
      std::fflush(stdout);
    }
  }

  // Initialize Aggregator + MQTT if enabled
  if (config_.aggregator.enabled) {
    if (config_.mqtt.enabled) {
      mqtt_publisher_ = std::make_unique<mqtt::MqttPublisher>(
          config_.mqtt.broker_ip, config_.mqtt.port,
          config_.mqtt.pub_topic, config_.mqtt.qos);
      std::string mqtt_err;
      if (!mqtt_publisher_->Initialize(config_.mqtt.client_id + "_agg", &mqtt_err)) {
        std::fprintf(stderr, "Aggregator MQTT init failed: %s\n", mqtt_err.c_str());
        std::fflush(stderr);
        mqtt_publisher_.reset();
      } else {
        std::printf("Aggregator MQTT connected to %s:%d\n",
                    config_.mqtt.broker_ip.c_str(), config_.mqtt.port);
        std::fflush(stdout);
      }
    }

    aggregator::AggregatorConfig agg_cfg;
    agg_cfg.enabled = true;
    agg_cfg.batch_size = config_.aggregator.batch_size;
    agg_cfg.queue_capacity = config_.aggregator.queue_capacity;
    agg_cfg.drain_timeout_ms = config_.aggregator.drain_timeout_ms;
    aggregator_ = std::make_unique<aggregator::Aggregator>(agg_cfg);
    if (mqtt_publisher_) {
      aggregator_->SetMqttPublisher(mqtt_publisher_.get());
    }
    std::string agg_err;
    if (!aggregator_->Start(&agg_err)) {
      std::fprintf(stderr, "Aggregator start failed: %s\n", agg_err.c_str());
      std::fflush(stderr);
      aggregator_.reset();
    }

    if (aggregator_) {
      for (auto& worker : workers_) {
        worker->SetAggregator(aggregator_.get());
      }
    }
  }

  // Single bus watch for the shared pipeline
  GstBus* bus = gst_element_get_bus(pipeline_);
  if (bus == nullptr) {
    if (out_error != nullptr) {
      *out_error = "Failed to get pipeline bus";
    }
    return false;
  }
  bus_watch_id_ = gst_bus_add_watch(bus, &MultiCamApp::OnBusMessage, this);
  gst_object_unref(bus);
  if (bus_watch_id_ == 0U) {
    if (out_error != nullptr) {
      *out_error = "Failed to add bus watch";
    }
    return false;
  }

  // Start memory monitor if looping is configured
  if (config_.memtest_loop_count != 1U) {
    StartMemoryMonitor();
  }

  return true;
}

bool MultiCamApp::StartPipeline(std::string* out_error) {
  if (pipeline_ == nullptr) {
    if (out_error != nullptr) {
      *out_error = "Pipeline is null";
    }
    return false;
  }

  const GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    if (out_error != nullptr) {
      *out_error = "Failed to set pipeline to PLAYING";
    }
    return false;
  }

  return true;
}

void MultiCamApp::StopPipeline() {
  if (bus_watch_id_ != 0U) {
    g_source_remove(bus_watch_id_);
    bus_watch_id_ = 0U;
  }

  for (auto& worker : workers_) {
    if (worker != nullptr) {
      worker->Stop();
    }
  }
  workers_.clear();

  if (aggregator_) {
    aggregator_->Stop();
    aggregator_.reset();
  }
  if (mqtt_publisher_) {
    mqtt_publisher_->Shutdown();
    mqtt_publisher_.reset();
  }

  if (pipeline_ != nullptr) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }

  if (frame_cache_service_) {
    frame_cache_service_->Shutdown();
    frame_cache_service_.reset();
  }
}

gboolean MultiCamApp::OnBusMessage(GstBus* /*bus*/, GstMessage* message, gpointer user_data) {
  auto* self = static_cast<MultiCamApp*>(user_data);
  if ((self == nullptr) || (message == nullptr)) {
    return TRUE;
  }

  switch (GST_MESSAGE_TYPE(message)) {
    case GST_MESSAGE_EOS: {
      self->current_iteration_++;
      const uint32_t loop_count = self->config_.memtest_loop_count;
      const bool should_loop =
          (loop_count == 0U) ||
          (self->current_iteration_ < loop_count);

      if (should_loop) {
        std::printf("Pipeline EOS: iteration %u/%s complete, seeking to start\n",
                    self->current_iteration_,
                    (loop_count == 0U) ? "inf" : std::to_string(loop_count).c_str());
        std::fflush(stdout);
        self->SampleMemory("iteration_boundary");
        if (!self->SeekPipelineToStart()) {
          std::printf("Pipeline seek failed, stopping\n");
          std::fflush(stdout);
          self->app_context_.RequestStop();
        }
      } else {
        std::printf("Pipeline EOS: all %u iterations finished\n",
                    self->current_iteration_);
        std::fflush(stdout);
        self->app_context_.RequestStop();
      }
      break;
    }
    case GST_MESSAGE_ERROR: {
      GError* error = nullptr;
      gchar* debug = nullptr;
      gst_message_parse_error(message, &error, &debug);
      const char* src_name =
          (GST_MESSAGE_SRC(message) != nullptr)
              ? GST_OBJECT_NAME(GST_MESSAGE_SRC(message))
              : "unknown";
      std::printf("Pipeline ERROR from [%s]: %s\n",
                  src_name,
                  (error != nullptr && error->message != nullptr) ? error->message : "unknown");
      if (debug != nullptr) {
        std::printf("  debug: %s\n", debug);
      }
      std::fflush(stdout);
      if (error != nullptr) {
        g_error_free(error);
      }
      if (debug != nullptr) {
        g_free(debug);
      }
      self->app_context_.error_received.store(true);
      self->app_context_.RequestStop();
      break;
    }
    default:
      break;
  }

  return TRUE;
}

void MultiCamApp::ReleaseMainLoop() {
  if (main_loop_ != nullptr) {
    g_main_loop_unref(main_loop_);
    main_loop_ = nullptr;
  }
  app_context_.main_loop = nullptr;
}

// ---- Memtest: seek-based video loop ----

bool MultiCamApp::SeekPipelineToStart() {
  if (pipeline_ == nullptr) {
    return false;
  }
  const gboolean result = gst_element_seek_simple(
      pipeline_,
      GST_FORMAT_TIME,
      static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT),
      0);
  return (result != FALSE);
}

// ---- Memtest: memory monitor ----

void MultiCamApp::StartMemoryMonitor() {
  std::string log_path = config_.memtest_log_path;
  if (log_path.empty()) {
    log_path = config_.log_dir + "/memtest.log";
  }

  {
    std::error_code ec;
    std::filesystem::path parent = std::filesystem::path(log_path).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent, ec);
    }
  }

  memtest_log_ = std::fopen(log_path.c_str(), "w");
  if (memtest_log_ == nullptr) {
    std::printf("MEMTEST: failed to open log file: %s\n", log_path.c_str());
    std::fflush(stdout);
    return;
  }

  std::fprintf(memtest_log_,
      "# memtest log: loop_count=%u sample_interval=%us\n"
      "# timestamp_s,event,iteration,vm_rss_kb,vm_size_kb,delta_rss_kb\n",
      config_.memtest_loop_count,
      config_.memtest_mem_sample_interval_s);
  std::fflush(memtest_log_);

  ReadProcSelfStatus(&memtest_start_rss_kb_, nullptr);
  memtest_peak_rss_kb_ = memtest_start_rss_kb_;
  SampleMemory("start");

  const guint interval_ms = config_.memtest_mem_sample_interval_s * 1000U;
  mem_monitor_timer_id_ = g_timeout_add(interval_ms, &MultiCamApp::OnMemoryMonitorTick, this);

  std::printf("MEMTEST: monitoring started (interval=%us, log=%s)\n",
              config_.memtest_mem_sample_interval_s, log_path.c_str());
  std::fflush(stdout);
}

void MultiCamApp::StopMemoryMonitor() {
  if (mem_monitor_timer_id_ != 0U) {
    g_source_remove(mem_monitor_timer_id_);
    mem_monitor_timer_id_ = 0U;
  }

  if (memtest_log_ != nullptr) {
    SampleMemory("stop");
    WriteMemtestSummary();
    std::fclose(memtest_log_);
    memtest_log_ = nullptr;
  }
}

gboolean MultiCamApp::OnMemoryMonitorTick(gpointer user_data) {
  auto* self = static_cast<MultiCamApp*>(user_data);
  if (self != nullptr) {
    self->SampleMemory("periodic");
  }
  return TRUE;
}

void MultiCamApp::SampleMemory(const char* event_tag) {
  uint64_t vm_rss_kb = 0ULL;
  uint64_t vm_size_kb = 0ULL;
  if (!ReadProcSelfStatus(&vm_rss_kb, &vm_size_kb)) {
    return;
  }

  if (vm_rss_kb > memtest_peak_rss_kb_) {
    memtest_peak_rss_kb_ = vm_rss_kb;
  }
  memtest_sample_count_++;

  const int64_t delta_rss = static_cast<int64_t>(vm_rss_kb) -
                            static_cast<int64_t>(memtest_start_rss_kb_);

  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const double timestamp_s =
      std::chrono::duration<double>(now).count();

  if (memtest_log_ != nullptr) {
    std::fprintf(memtest_log_,
        "%.3f,%s,%u,%llu,%llu,%lld\n",
        timestamp_s,
        (event_tag != nullptr) ? event_tag : "unknown",
        current_iteration_,
        static_cast<unsigned long long>(vm_rss_kb),
        static_cast<unsigned long long>(vm_size_kb),
        static_cast<long long>(delta_rss));
    std::fflush(memtest_log_);
  }

  std::printf("MEMTEST [%s] iter=%u rss=%lluKB vmsz=%lluKB delta=%+lldKB\n",
              (event_tag != nullptr) ? event_tag : "?",
              current_iteration_,
              static_cast<unsigned long long>(vm_rss_kb),
              static_cast<unsigned long long>(vm_size_kb),
              static_cast<long long>(delta_rss));
  std::fflush(stdout);
}

bool MultiCamApp::ReadProcSelfStatus(uint64_t* out_vm_rss_kb,
                                      uint64_t* out_vm_size_kb) {
  FILE* f = std::fopen("/proc/self/status", "r");
  if (f == nullptr) {
    return false;
  }

  uint64_t rss = 0ULL;
  uint64_t vmsz = 0ULL;
  char line[256];
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, "VmRSS:", 6) == 0) {
      char* p = line + 6;
      while (*p == ' ' || *p == '\t') ++p;
      rss = std::strtoull(p, nullptr, 10);
    } else if (std::strncmp(line, "VmSize:", 7) == 0) {
      char* p = line + 7;
      while (*p == ' ' || *p == '\t') ++p;
      vmsz = std::strtoull(p, nullptr, 10);
    }
  }
  std::fclose(f);

  if (out_vm_rss_kb != nullptr) *out_vm_rss_kb = rss;
  if (out_vm_size_kb != nullptr) *out_vm_size_kb = vmsz;
  return true;
}

void MultiCamApp::WriteMemtestSummary() {
  uint64_t end_rss = 0ULL;
  uint64_t end_vmsz = 0ULL;
  ReadProcSelfStatus(&end_rss, &end_vmsz);

  const int64_t delta = static_cast<int64_t>(end_rss) -
                        static_cast<int64_t>(memtest_start_rss_kb_);

  if (memtest_log_ != nullptr) {
    std::fprintf(memtest_log_,
        "# --- SUMMARY ---\n"
        "# iterations_completed: %u\n"
        "# start_rss_kb: %llu\n"
        "# end_rss_kb: %llu\n"
        "# peak_rss_kb: %llu\n"
        "# delta_rss_kb: %lld\n"
        "# samples_collected: %llu\n",
        current_iteration_,
        static_cast<unsigned long long>(memtest_start_rss_kb_),
        static_cast<unsigned long long>(end_rss),
        static_cast<unsigned long long>(memtest_peak_rss_kb_),
        static_cast<long long>(delta),
        static_cast<unsigned long long>(memtest_sample_count_));
    std::fflush(memtest_log_);
  }

  std::printf("MEMTEST SUMMARY: iterations=%u start_rss=%lluKB end_rss=%lluKB "
              "peak_rss=%lluKB delta=%+lldKB samples=%llu\n",
              current_iteration_,
              static_cast<unsigned long long>(memtest_start_rss_kb_),
              static_cast<unsigned long long>(end_rss),
              static_cast<unsigned long long>(memtest_peak_rss_kb_),
              static_cast<long long>(delta),
              static_cast<unsigned long long>(memtest_sample_count_));
  std::fflush(stdout);
}

}  // namespace multi_cam_app::app
