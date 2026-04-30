#include "utils/perf/LatencyRecorder.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <time.h>

namespace multi_cam_app::perf {
namespace {

constexpr uint64_t kMaxTrackedFrames = 8192ULL;
constexpr uint64_t kStaleOrderWindow = 8192ULL;
constexpr uint64_t kHardMapLimit = 20000ULL;
constexpr uint64_t kFrameReuseResetNs = 1000000000ULL;  // 1 second
constexpr uint64_t kResourceSampleIntervalNs = 200000000ULL;  // 200ms

uint64_t TimespecToNs(const timespec& ts) {
  return (static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL) +
         static_cast<uint64_t>(ts.tv_nsec);
}

clockid_t MonotonicClockId() {
#ifdef CLOCK_MONOTONIC_RAW
  return CLOCK_MONOTONIC_RAW;
#else
  return CLOCK_MONOTONIC;
#endif
}

}  // namespace

LatencyRecorder::~LatencyRecorder() {
  Close();
}

bool LatencyRecorder::Initialize(const Config& config, std::string* out_error) {
  Close();

  channel_id_ = config.channel_id;
  sample_every_n_ = (config.sample_every_n > 0U) ? config.sample_every_n : 1U;
  flush_per_sample_ = config.flush_per_sample;

  if (config.output_dir.empty()) {
    if (out_error != nullptr) {
      *out_error = "LatencyRecorder output_dir is empty";
    }
    return false;
  }

  std::error_code ec;
  std::filesystem::create_directories(config.output_dir, ec);

  const std::string path = config.output_dir + "/latency_ch" +
      std::to_string(channel_id_) + ".csv";
  file_ = std::fopen(path.c_str(), "w");
  if (file_ == nullptr) {
    if (out_error != nullptr) {
      *out_error = "Failed to open latency csv: " + path;
    }
    return false;
  }

  std::fprintf(file_,
      "ts_epoch_ms,channel_id,metric,frame_key,latency_ns,latency_ms,start_ns,end_ns,gpu_load_pct,rss_kb,measurement_method\n");
  std::fflush(file_);

  frame_timings_.clear();
  metric_seen_count_.fill(0ULL);
  event_order_ = 0ULL;
  cached_gpu_pct_ = -1;
  cached_rss_kb_ = -1;
  last_resource_sample_ns_ = 0ULL;
  return true;
}

void LatencyRecorder::RecordStageTimestamp(LatencyStage stage,
                                          uint64_t frame_key,
                                          uint64_t timestamp_ns) {
  if ((file_ == nullptr) || (timestamp_ns == 0ULL)) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto [it, inserted] = frame_timings_.try_emplace(frame_key);
  FrameTiming& timing = it->second;
  timing.last_seen_order = ++event_order_;
  if (!inserted &&
      (stage == LatencyStage::kH264ParseSrc) &&
      timing.has_h264parse_src &&
      (timestamp_ns > timing.h264parse_src_ns) &&
      ((timestamp_ns - timing.h264parse_src_ns) > kFrameReuseResetNs)) {
    ResetTimingForFrameReuse(&timing, timestamp_ns);
  }

  switch (stage) {
    case LatencyStage::kH264ParseSrc:
      timing.h264parse_src_ns = timestamp_ns;
      timing.has_h264parse_src = true;
      break;
    case LatencyStage::kV4L2DecCapsSrc:
      timing.v4l2dec_caps_src_ns = timestamp_ns;
      timing.has_v4l2dec_caps_src = true;
      break;
    case LatencyStage::kYoloQnnSink:
      timing.yolo_qnn_sink_ns = timestamp_ns;
      timing.has_yolo_qnn_sink = true;
      break;
    case LatencyStage::kYoloQnnSrc:
      timing.yolo_qnn_src_ns = timestamp_ns;
      timing.has_yolo_qnn_src = true;
      break;
    case LatencyStage::kYoloPostprocessSrc:
      timing.yolo_postprocess_src_ns = timestamp_ns;
      timing.has_yolo_postprocess_src = true;
      break;
    case LatencyStage::kTrackerSink:
      timing.tracker_sink_ns = timestamp_ns;
      timing.has_tracker_sink = true;
      break;
    case LatencyStage::kTrackerSrc:
      timing.tracker_src_ns = timestamp_ns;
      timing.has_tracker_src = true;
      break;
    case LatencyStage::kFaceInferSink:
      timing.faceinfer_sink_ns = timestamp_ns;
      timing.has_faceinfer_sink = true;
      break;
    case LatencyStage::kFaceInferSrc:
      timing.faceinfer_src_ns = timestamp_ns;
      timing.has_faceinfer_src = true;
      break;
    default:
      break;
  }

  MaybeEmitMetrics(frame_key, &timing);
  PruneIfNeeded();
}

void LatencyRecorder::Close() {
  std::lock_guard<std::mutex> lock(mutex_);
  frame_timings_.clear();
  metric_seen_count_.fill(0ULL);
  event_order_ = 0ULL;
  cached_gpu_pct_ = -1;
  cached_rss_kb_ = -1;
  last_resource_sample_ns_ = 0ULL;
  if (file_ != nullptr) {
    std::fflush(file_);
    std::fclose(file_);
    file_ = nullptr;
  }
}

bool LatencyRecorder::ExtractFrameKey(const GstBuffer* buffer,
                                      uint64_t* out_frame_key) {
  if ((buffer == nullptr) || (out_frame_key == nullptr)) {
    return false;
  }
  if (GST_BUFFER_PTS_IS_VALID(buffer)) {
    *out_frame_key = static_cast<uint64_t>(GST_BUFFER_PTS(buffer));
    return true;
  }
  if (GST_BUFFER_DTS_IS_VALID(buffer)) {
    *out_frame_key = static_cast<uint64_t>(GST_BUFFER_DTS(buffer));
    return true;
  }
  if (GST_BUFFER_OFFSET_IS_VALID(buffer)) {
    *out_frame_key = static_cast<uint64_t>(GST_BUFFER_OFFSET(buffer));
    return true;
  }
  return false;
}

uint64_t LatencyRecorder::MonotonicNowNs() {
  timespec ts {};
  if (clock_gettime(MonotonicClockId(), &ts) != 0) {
    return 0ULL;
  }
  return TimespecToNs(ts);
}

uint64_t LatencyRecorder::RealtimeEpochMs() {
  timespec ts {};
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    return 0ULL;
  }
  return TimespecToNs(ts) / 1000000ULL;
}

const char* LatencyRecorder::MetricName(MetricKind kind) {
  switch (kind) {
    case MetricKind::kFaceInferToEmbedding:
      return "face_infer_to_embedding";
    case MetricKind::kByteTrack:
      return "bytetrack";
    case MetricKind::kFileToDec:
      return "file_to_dec";
    case MetricKind::kDecToYolo:
      return "dec_to_yolo";
    case MetricKind::kDecToTracker:
      return "dec_to_tracker";
    case MetricKind::kDecToFace:
      return "dec_to_face";
    case MetricKind::kYoloQnnInfer:
      return "yolo_qnn_infer";
    default:
      return "unknown";
  }
}

const char* LatencyRecorder::MeasurementMethod() {
  return "timing_mark";
}

bool LatencyRecorder::ShouldSample(MetricKind kind) {
  const size_t idx = static_cast<size_t>(kind);
  if (idx >= metric_seen_count_.size()) {
    return false;
  }
  const uint64_t ordinal = ++metric_seen_count_[idx];
  return ((ordinal - 1ULL) % sample_every_n_) == 0ULL;
}

void LatencyRecorder::ResetTimingForFrameReuse(FrameTiming* timing,
                                               uint64_t now_ns) {
  if (timing == nullptr) {
    return;
  }
  *timing = FrameTiming {};
  timing->h264parse_src_ns = now_ns;
  timing->has_h264parse_src = true;
  timing->last_seen_order = event_order_;
}

void LatencyRecorder::MaybeEmitMetrics(uint64_t frame_key, FrameTiming* timing) {
  if ((timing == nullptr) || (file_ == nullptr)) {
    return;
  }

  if (timing->has_faceinfer_sink && timing->has_faceinfer_src) {
    EmitMetric(frame_key,
               MetricKind::kFaceInferToEmbedding,
               timing->faceinfer_sink_ns,
               timing->faceinfer_src_ns,
               &timing->emitted_face_infer_to_embedding,
               timing);
  }
  if (timing->has_tracker_sink && timing->has_tracker_src) {
    EmitMetric(frame_key,
               MetricKind::kByteTrack,
               timing->tracker_sink_ns,
               timing->tracker_src_ns,
               &timing->emitted_bytetrack,
               timing);
  }
  if (timing->has_h264parse_src && timing->has_v4l2dec_caps_src) {
    EmitMetric(frame_key,
               MetricKind::kFileToDec,
               timing->h264parse_src_ns,
               timing->v4l2dec_caps_src_ns,
               &timing->emitted_file_to_dec,
               timing);
  }
  if (timing->has_yolo_qnn_sink && timing->has_yolo_qnn_src) {
    EmitMetric(frame_key,
               MetricKind::kYoloQnnInfer,
               timing->yolo_qnn_sink_ns,
               timing->yolo_qnn_src_ns,
               &timing->emitted_yolo_qnn_infer,
               timing);
  }
  if (timing->has_v4l2dec_caps_src && timing->has_yolo_postprocess_src) {
    EmitMetric(frame_key,
               MetricKind::kDecToYolo,
               timing->v4l2dec_caps_src_ns,
               timing->yolo_postprocess_src_ns,
               &timing->emitted_dec_to_yolo,
               timing);
  }
  if (timing->has_v4l2dec_caps_src && timing->has_tracker_src) {
    EmitMetric(frame_key,
               MetricKind::kDecToTracker,
               timing->v4l2dec_caps_src_ns,
               timing->tracker_src_ns,
               &timing->emitted_dec_to_tracker,
               timing);
  }
  if (timing->has_v4l2dec_caps_src && timing->has_faceinfer_src) {
    EmitMetric(frame_key,
               MetricKind::kDecToFace,
               timing->v4l2dec_caps_src_ns,
               timing->faceinfer_src_ns,
               &timing->emitted_dec_to_face,
               timing);
  }
}

void LatencyRecorder::EmitMetric(uint64_t frame_key,
                                 MetricKind kind,
                                 uint64_t start_ns,
                                 uint64_t end_ns,
                                 bool* emitted_flag,
                                 FrameTiming* timing) {
  if ((file_ == nullptr) || (emitted_flag == nullptr) || (timing == nullptr)) {
    return;
  }
  if (*emitted_flag) {
    return;
  }
  *emitted_flag = true;
  if (timing->emitted_count < static_cast<uint8_t>(kMetricCount)) {
    ++timing->emitted_count;
  }

  if ((end_ns < start_ns) || !ShouldSample(kind)) {
    return;
  }

  const uint64_t latency_ns = end_ns - start_ns;
  const double latency_ms = static_cast<double>(latency_ns) / 1000000.0;
  const uint64_t ts_epoch_ms = RealtimeEpochMs();
  RefreshResourceSamples();
  const int gpu_pct = cached_gpu_pct_;
  const int64_t rss_kb = cached_rss_kb_;
  std::fprintf(file_,
               "%llu,%u,%s,%llu,%llu,%.3f,%llu,%llu,%d,%lld,%s\n",
               static_cast<unsigned long long>(ts_epoch_ms),
               static_cast<unsigned>(channel_id_),
               MetricName(kind),
               static_cast<unsigned long long>(frame_key),
               static_cast<unsigned long long>(latency_ns),
               latency_ms,
               static_cast<unsigned long long>(start_ns),
               static_cast<unsigned long long>(end_ns),
               gpu_pct,
               static_cast<long long>(rss_kb),
               MeasurementMethod());
  if (flush_per_sample_) {
    std::fflush(file_);
  }
}

void LatencyRecorder::PruneIfNeeded() {
  if (frame_timings_.size() <= kMaxTrackedFrames) {
    return;
  }

  for (auto it = frame_timings_.begin(); it != frame_timings_.end();) {
    const bool stale =
        (event_order_ > it->second.last_seen_order) &&
        ((event_order_ - it->second.last_seen_order) > kStaleOrderWindow);
    const bool completed = (it->second.emitted_count >= kMetricCount);
    if (stale || completed) {
      it = frame_timings_.erase(it);
    } else {
      ++it;
    }
  }

  if (frame_timings_.size() > kHardMapLimit) {
    frame_timings_.clear();
  }
}

void LatencyRecorder::RefreshResourceSamples() {
  const uint64_t now = MonotonicNowNs();
  if ((now == 0ULL) ||
      ((now - last_resource_sample_ns_) < kResourceSampleIntervalNs)) {
    return;
  }
  last_resource_sample_ns_ = now;
  cached_gpu_pct_ = SampleGpuLoadPct();
  cached_rss_kb_ = SampleRssKb();
}

int LatencyRecorder::SampleGpuLoadPct() {
  FILE* f = std::fopen("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage", "r");
  if (f == nullptr) {
    return -1;
  }
  char buf[32] {};
  if (std::fgets(buf, sizeof(buf), f) == nullptr) {
    std::fclose(f);
    return -1;
  }
  std::fclose(f);
  int val = -1;
  std::sscanf(buf, "%d", &val);
  return val;
}

int64_t LatencyRecorder::SampleRssKb() {
  FILE* f = std::fopen("/proc/self/status", "r");
  if (f == nullptr) {
    return -1;
  }
  char line[256];
  int64_t rss = -1;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, "VmRSS:", 6) == 0) {
      std::sscanf(line + 6, " %lld", reinterpret_cast<long long*>(&rss));
      break;
    }
  }
  std::fclose(f);
  return rss;
}

}  // namespace multi_cam_app::perf
