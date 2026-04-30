#ifndef MULTI_CAM_APP_CHANNEL_WORKER_CHANNEL_WORKER_HPP
#define MULTI_CAM_APP_CHANNEL_WORKER_CHANNEL_WORKER_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <gst/gst.h>

#include "app/AppContext.hpp"
#include "channel_worker/infer_module/DetectionInferModule.hpp"
// GstFaceDetectionMeta for reading face results from qtifaceinfer plugin
extern "C" {
#include "channel_worker/infer_module/face_infer/gst_plugin/gstfacedetectionmeta.h"
}
#include "config/AppConfig.hpp"
#include "utils/FaceGallery.hpp"
#include "utils/JsonlWriter.hpp"

namespace multi_cam_app::aggregator { class Aggregator; }
namespace dmabuf_producer { class DmaBufProducer; }

namespace multi_cam_app::channel_worker {

class ChannelWorker {
public:
  ChannelWorker(uint32_t channel_id, std::string video_path);
  ~ChannelWorker();

  ChannelWorker(const ChannelWorker&) = delete;
  ChannelWorker& operator=(const ChannelWorker&) = delete;

  bool Initialize(const config::AppConfig& config,
                  app::AppContext* app_context,
                  GstBin* pipeline_bin,
                  gallery::FaceGallery* shared_gallery,
                  std::string* out_error);

  void Stop();

  void SetAggregator(aggregator::Aggregator* agg) { aggregator_ = agg; }
  void SetFrameCacheProducer(dmabuf_producer::DmaBufProducer* p) {
    frame_cache_producer_ = p;
    if (frame_offload_element_ != nullptr && p != nullptr) {
      g_object_set(G_OBJECT(frame_offload_element_), "producer", p, nullptr);
    }
  }

  void SetFaceGallery(gallery::FaceGallery* gallery) {
    if (frame_offload_element_ != nullptr && gallery != nullptr) {
      g_object_set(G_OBJECT(frame_offload_element_), "gallery", gallery, nullptr);
    }
  }

  std::string OutputJsonlPath() const;
  uint32_t ChannelId() const {
    return channel_id_;
  }

private:
  struct LabelIndexEntry {
    uint32_t class_id {0U};
    std::string canonical_label {};
  };

  // Combined appsink callback (NV12 + ROI meta + optional FaceMeta)
  static GstFlowReturn OnNewCombinedSample(GstElement* appsink, gpointer user_data);

  bool HandleCombinedSample(GstSample* sample);

  bool LoadLabelIndex(const std::string& labels_path, std::string* out_error);
  bool ResolveDetectionClass(const std::string& raw_label,
                             uint32_t* out_class_id,
                             std::string* out_label) const;
  static std::string ReplaceChar(std::string text, char from, char to);
  static std::string TrimAscii(std::string text);
  static uint64_t SystemNowNs();
  void ResetSampleStats();
  void MaybeLogPeriodicSampleStats(uint64_t sample_total);
  void LogSampleStats(const char* stage) const;
  void LogFinalSampleStatsOnce(const char* stage);

  uint32_t channel_id_ {0U};
  std::string video_path_ {};
  app::AppContext* app_context_ {nullptr};

  infer_module::DetectionInferModule infer_module_ {};
  GstElement* appsink_ {nullptr};       // appsink_combined
  gulong appsink_signal_id_ {0UL};

  // Face processing (qtifaceinfer plugin handles FaceProcessor internally)
  bool face_enabled_ {false};

  // Face gallery (shared across all channels, owned by MultiCamApp)
  gallery::FaceGallery* gallery_ptr_ {nullptr};
  FILE* gallery_diag_log_ {nullptr};
  float gallery_threshold_ {0.3f};
  uint32_t gallery_min_face_size_ {40U};
  float gallery_min_score_ {0.6f};
  std::unordered_map<uint32_t, int> track_id_to_face_;
  std::unordered_map<uint32_t, uint64_t> track_id_last_seen_;

  aggregator::Aggregator* aggregator_ {nullptr};
  dmabuf_producer::DmaBufProducer* frame_cache_producer_ {nullptr};
  GstElement* frame_offload_element_ {nullptr};
  std::unique_ptr<output::JsonlWriter> writer_;
  std::unordered_map<std::string, LabelIndexEntry> label_index_;
  mutable std::mutex state_mutex_;
  uint64_t local_frame_counter_ {0ULL};
  uint64_t last_timestamp_ns_ {0ULL};
  bool initialized_ {false};

  std::atomic<uint64_t> appsink_sample_total_ {0ULL};
  std::atomic<uint64_t> appsink_video_meta_ok_ {0ULL};
  std::atomic<uint64_t> appsink_video_meta_fail_ {0ULL};
  std::atomic<uint64_t> appsink_write_ok_ {0ULL};
  std::atomic<uint64_t> appsink_write_fail_ {0ULL};
  std::atomic<bool> first_sample_received_logged_ {false};
  std::atomic<bool> final_sample_stats_logged_ {false};
};

}  // namespace multi_cam_app::channel_worker

#endif  // MULTI_CAM_APP_CHANNEL_WORKER_CHANNEL_WORKER_HPP
