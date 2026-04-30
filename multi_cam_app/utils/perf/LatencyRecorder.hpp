#ifndef MULTI_CAM_APP_UTILS_PERF_LATENCY_RECORDER_HPP
#define MULTI_CAM_APP_UTILS_PERF_LATENCY_RECORDER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

#include <gst/gst.h>

namespace multi_cam_app::perf {

enum class LatencyStage : uint8_t {
  kH264ParseSrc = 0U,
  kV4L2DecCapsSrc,
  kYoloQnnSink,
  kYoloQnnSrc,
  kYoloPostprocessSrc,
  kTrackerSink,
  kTrackerSrc,
  kFaceInferSink,
  kFaceInferSrc,
};

class LatencyRecorder {
public:
  struct Config {
    uint32_t channel_id {0U};
    std::string output_dir {};
    uint32_t sample_every_n {1U};
    bool flush_per_sample {true};
  };

  LatencyRecorder() = default;
  ~LatencyRecorder();

  LatencyRecorder(const LatencyRecorder&) = delete;
  LatencyRecorder& operator=(const LatencyRecorder&) = delete;

  bool Initialize(const Config& config, std::string* out_error);
  void RecordStageTimestamp(LatencyStage stage,
                            uint64_t frame_key,
                            uint64_t timestamp_ns);
  void Close();

  static bool ExtractFrameKey(const GstBuffer* buffer, uint64_t* out_frame_key);

private:
  enum class MetricKind : uint8_t {
    kFaceInferToEmbedding = 0U,
    kByteTrack,
    kFileToDec,
    kDecToYolo,
    kDecToTracker,
    kDecToFace,
    kYoloQnnInfer,
    kCount
  };

  static constexpr size_t kMetricCount =
      static_cast<size_t>(MetricKind::kCount);

  struct FrameTiming {
    uint64_t h264parse_src_ns {0ULL};
    uint64_t v4l2dec_caps_src_ns {0ULL};
    uint64_t yolo_qnn_sink_ns {0ULL};
    uint64_t yolo_qnn_src_ns {0ULL};
    uint64_t yolo_postprocess_src_ns {0ULL};
    uint64_t tracker_sink_ns {0ULL};
    uint64_t tracker_src_ns {0ULL};
    uint64_t faceinfer_sink_ns {0ULL};
    uint64_t faceinfer_src_ns {0ULL};

    bool has_h264parse_src {false};
    bool has_v4l2dec_caps_src {false};
    bool has_yolo_qnn_sink {false};
    bool has_yolo_qnn_src {false};
    bool has_yolo_postprocess_src {false};
    bool has_tracker_sink {false};
    bool has_tracker_src {false};
    bool has_faceinfer_sink {false};
    bool has_faceinfer_src {false};

    bool emitted_face_infer_to_embedding {false};
    bool emitted_bytetrack {false};
    bool emitted_file_to_dec {false};
    bool emitted_dec_to_yolo {false};
    bool emitted_dec_to_tracker {false};
    bool emitted_dec_to_face {false};
    bool emitted_yolo_qnn_infer {false};

    uint8_t emitted_count {0U};
    uint64_t last_seen_order {0ULL};
  };

  static uint64_t MonotonicNowNs();
  static uint64_t RealtimeEpochMs();
  static const char* MetricName(MetricKind kind);
  static const char* MeasurementMethod();
  bool ShouldSample(MetricKind kind);
  void ResetTimingForFrameReuse(FrameTiming* timing, uint64_t now_ns);
  void MaybeEmitMetrics(uint64_t frame_key, FrameTiming* timing);
  void EmitMetric(uint64_t frame_key,
                  MetricKind kind,
                  uint64_t start_ns,
                  uint64_t end_ns,
                  bool* emitted_flag,
                  FrameTiming* timing);
  void PruneIfNeeded();
  void RefreshResourceSamples();
  static int SampleGpuLoadPct();
  static int64_t SampleRssKb();

  std::mutex mutex_ {};
  std::unordered_map<uint64_t, FrameTiming> frame_timings_ {};
  std::array<uint64_t, kMetricCount> metric_seen_count_ {};

  FILE* file_ {nullptr};
  uint32_t channel_id_ {0U};
  uint32_t sample_every_n_ {1U};
  bool flush_per_sample_ {true};
  uint64_t event_order_ {0ULL};

  int cached_gpu_pct_ {-1};
  int64_t cached_rss_kb_ {-1};
  uint64_t last_resource_sample_ns_ {0ULL};
};

}  // namespace multi_cam_app::perf

#endif  // MULTI_CAM_APP_UTILS_PERF_LATENCY_RECORDER_HPP
