#include "channel_worker/ChannelWorker.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include "aggregator/Aggregator.hpp"
#include "channel_worker/infer_module/face_infer/scrfd_infer/FaceTypes.hpp"
#include "pipeline/PipelineUtils.hpp"

namespace multi_cam_app::channel_worker {
namespace {
constexpr uint64_t kSampleStatsLogInterval = 100ULL;

void LogChannel(uint32_t channel_id, const std::string& text) {
  std::printf("[channel %u] %s\n", channel_id, text.c_str());
  std::fflush(stdout);
}

std::string BuildChannelOutputPath(const std::string& output_dir, uint32_t channel_id) {
  if (output_dir.empty()) {
    return "channel_" + std::to_string(channel_id) + ".jsonl";
  }
  const char tail = output_dir.back();
  if ((tail == '/') || (tail == '\\')) {
    return output_dir + "channel_" + std::to_string(channel_id) + ".jsonl";
  }
  return output_dir + "/channel_" + std::to_string(channel_id) + ".jsonl";
}

void MatchFacesToPersons(
    const std::vector<face_infer::FaceResult>& faces,
    std::vector<output::DetectionRecord>& detections,
    bool has_arcface,
    uint32_t frame_width,
    uint32_t frame_height,
    gallery::FaceGallery* gallery,
    std::unordered_map<uint32_t, int>* track_cache,
    float gallery_threshold,
    uint32_t gallery_min_face_size,
    float gallery_min_score,
    uint32_t channel_id,
    uint64_t frame_number,
    FILE* diag_log) {

  const float fw = static_cast<float>(frame_width);
  const float fh = static_cast<float>(frame_height);
  if (fw < 1.0f || fh < 1.0f) return;

  uint32_t diag_no_person = 0, diag_cache_hit = 0;
  uint32_t diag_match = 0, diag_enrolled = 0, diag_quality_fail = 0;

  uint32_t person_det_count = 0;
  for (const auto& det : detections) {
    if (det.label == "person" && !det.face.has_value()) ++person_det_count;
  }

  for (const auto& face : faces) {
    const float cx = ((face.det.x1 + face.det.x2) / 2.0f) / fw;
    const float cy = ((face.det.y1 + face.det.y2) / 2.0f) / fh;

    int best_idx = -1;
    double best_score = -1.0;

    for (size_t d = 0; d < detections.size(); ++d) {
      const auto& det = detections[d];
      if (det.label != "person") continue;
      if (det.face.has_value()) continue;

      if (cx >= det.left && cx <= det.right && cy >= det.top && cy <= det.bottom) {
        double area = (det.right - det.left) * (det.bottom - det.top);
        if (area <= 0.0) continue;

        double bbox_mid_y = (det.top + det.bottom) / 2.0;
        bool in_upper_half = (cy <= bbox_mid_y);
        double score = (in_upper_half ? 1e6 : 0.0) + (1.0 / area);

        if (score > best_score) {
          best_score = score;
          best_idx = static_cast<int>(d);
        }
      }
    }

    if (best_idx >= 0) {
      output::FaceInfo fi;
      fi.confidence = face.det.score;
      fi.bbox_x1 = face.det.x1 / fw;
      fi.bbox_y1 = face.det.y1 / fh;
      fi.bbox_x2 = face.det.x2 / fw;
      fi.bbox_y2 = face.det.y2 / fh;
      for (int k = 0; k < face_infer::kFaceNumLandmarks; ++k) {
        fi.landmarks[k * 2] = face.det.landmarks[k][0] / fw;
        fi.landmarks[k * 2 + 1] = face.det.landmarks[k][1] / fh;
      }
      if (has_arcface) {
        fi.embedding.assign(face.embedding,
                            face.embedding + face_infer::kFaceEmbeddingDim);
      }
      fi.linked_track_id = detections[static_cast<size_t>(best_idx)].track_id;

      if (has_arcface && !fi.embedding.empty() && gallery != nullptr) {
        uint32_t tid = fi.linked_track_id;
        auto cache_it = track_cache->find(tid);
        if (cache_it != track_cache->end()) {
          fi.face_id = cache_it->second;
          auto result = gallery->Match(face.embedding, 0.0f);
          fi.face_similarity = result.similarity;
          ++diag_cache_hit;
          if (diag_log) {
            std::fprintf(diag_log,
                "frame=%llu track=%u event=CACHE_HIT face_id=%d sim=%.4f\n",
                static_cast<unsigned long long>(frame_number), tid,
                fi.face_id, fi.face_similarity);
          }
        } else {
          auto result = gallery->Match(face.embedding, gallery_threshold);
          if (result.face_id >= 0) {
            fi.face_id = result.face_id;
            fi.face_similarity = result.similarity;
            ++diag_match;
            if (diag_log) {
              std::fprintf(diag_log,
                  "frame=%llu track=%u event=GALLERY_MATCH face_id=%d sim=%.4f "
                  "gallery_size=%zu\n",
                  static_cast<unsigned long long>(frame_number), tid,
                  fi.face_id, fi.face_similarity, gallery->Size());
            }
          } else {
            float face_w = face.det.x2 - face.det.x1;
            float face_h = face.det.y2 - face.det.y1;
            bool quality_ok =
                (face_w >= static_cast<float>(gallery_min_face_size)) &&
                (face_h >= static_cast<float>(gallery_min_face_size)) &&
                (face.det.score >= gallery_min_score);
            if (quality_ok) {
              fi.face_id = gallery->Enroll(face.embedding);
              ++diag_enrolled;
              if (diag_log) {
                std::fprintf(diag_log,
                    "frame=%llu track=%u event=ENROLLED face_id=%d best_sim=%.4f "
                    "face=%.0fx%.0f score=%.3f gallery_size=%zu\n",
                    static_cast<unsigned long long>(frame_number), tid,
                    fi.face_id, result.similarity,
                    face_w, face_h, face.det.score, gallery->Size());
              }
            } else {
              fi.face_id = -1;
              ++diag_quality_fail;
              if (diag_log) {
                std::fprintf(diag_log,
                    "frame=%llu track=%u event=QUALITY_FAIL best_sim=%.4f "
                    "face=%.0fx%.0f score=%.3f min_size=%u min_score=%.2f\n",
                    static_cast<unsigned long long>(frame_number), tid,
                    result.similarity, face_w, face_h, face.det.score,
                    gallery_min_face_size, gallery_min_score);
              }
            }
            fi.face_similarity = 0.0f;
          }
          if (fi.face_id >= 0) {
            track_cache->emplace(tid, fi.face_id);
          }
        }
      }

      detections[static_cast<size_t>(best_idx)].face = std::move(fi);
    } else {
      ++diag_no_person;
      if (diag_log) {
        std::fprintf(diag_log,
            "frame=%llu event=NO_PERSON face_cx=%.3f face_cy=%.3f "
            "person_count=%u\n",
            static_cast<unsigned long long>(frame_number),
            cx, cy, person_det_count);
      }
    }
  }

  const uint32_t diag_total = static_cast<uint32_t>(faces.size());
  if (diag_log && diag_total > 0) {
    std::fprintf(diag_log,
        "frame=%llu SUMMARY faces=%u no_person=%u cache_hit=%u "
        "match=%u enrolled=%u quality_fail=%u\n",
        static_cast<unsigned long long>(frame_number),
        diag_total, diag_no_person, diag_cache_hit,
        diag_match, diag_enrolled, diag_quality_fail);
    std::fflush(diag_log);
  }
}

}  // namespace

ChannelWorker::ChannelWorker(uint32_t channel_id, std::string video_path)
    : channel_id_(channel_id),
      video_path_(std::move(video_path)),
      writer_(std::make_unique<output::JsonlWriter>(
          BuildChannelOutputPath("", channel_id_))) {}

ChannelWorker::~ChannelWorker() {
  Stop();
}

void ChannelWorker::ResetSampleStats() {
  appsink_sample_total_.store(0ULL, std::memory_order_relaxed);
  appsink_video_meta_ok_.store(0ULL, std::memory_order_relaxed);
  appsink_video_meta_fail_.store(0ULL, std::memory_order_relaxed);
  appsink_write_ok_.store(0ULL, std::memory_order_relaxed);
  appsink_write_fail_.store(0ULL, std::memory_order_relaxed);
  first_sample_received_logged_.store(false, std::memory_order_relaxed);
  final_sample_stats_logged_.store(false, std::memory_order_relaxed);
}

void ChannelWorker::MaybeLogPeriodicSampleStats(uint64_t sample_total) {
  if ((sample_total == 0ULL) || ((sample_total % kSampleStatsLogInterval) != 0ULL)) {
    return;
  }
  LogSampleStats("periodic");
}

void ChannelWorker::LogSampleStats(const char* stage) const {
  const std::string stage_name = (stage != nullptr) ? stage : "unknown";
  std::string msg =
      "sample_stats stage=" + stage_name +
      " total=" +
      std::to_string(appsink_sample_total_.load(std::memory_order_relaxed)) +
      " meta_ok=" +
      std::to_string(appsink_video_meta_ok_.load(std::memory_order_relaxed)) +
      " meta_fail=" +
      std::to_string(appsink_video_meta_fail_.load(std::memory_order_relaxed)) +
      " write_ok=" +
      std::to_string(appsink_write_ok_.load(std::memory_order_relaxed)) +
      " write_fail=" +
      std::to_string(appsink_write_fail_.load(std::memory_order_relaxed));
  LogChannel(channel_id_, msg);
}

void ChannelWorker::LogFinalSampleStatsOnce(const char* stage) {
  bool expected = false;
  if (!final_sample_stats_logged_.compare_exchange_strong(
          expected, true, std::memory_order_relaxed)) {
    return;
  }
  LogSampleStats(stage);
}

bool ChannelWorker::Initialize(const config::AppConfig& config,
                               app::AppContext* app_context,
                               GstBin* pipeline_bin,
                               gallery::FaceGallery* shared_gallery,
                               std::string* out_error) {
  if (initialized_) {
    return true;
  }
  if (app_context == nullptr) {
    if (out_error != nullptr) {
      *out_error = "ChannelWorker.Initialize received null app_context";
    }
    return false;
  }
  if (pipeline_bin == nullptr) {
    if (out_error != nullptr) {
      *out_error = "ChannelWorker.Initialize received null pipeline_bin";
    }
    return false;
  }

  app_context_ = app_context;
  ResetSampleStats();
  if (!config.msgagg.enabled) {
    writer_ = std::make_unique<output::JsonlWriter>(
        BuildChannelOutputPath(config.output_dir, channel_id_));
  }

  if (!LoadLabelIndex(config.labels_path, out_error)) {
    return false;
  }

  LogChannel(channel_id_, "building detection chain...");
  GstElement* combined_appsink = nullptr;
  if (!infer_module_.BuildChain(config, channel_id_, video_path_, pipeline_bin,
                                &combined_appsink, out_error)) {
    return false;
  }
  LogChannel(channel_id_, "detection chain built");

  if (config.frame_cache.enabled) {
    frame_offload_element_ = gst_bin_get_by_name(pipeline_bin,
        pipeline::MakeElementName("frame_offload", channel_id_).c_str());
  }

  appsink_ = combined_appsink;

  if (appsink_ == nullptr) {
    if (out_error != nullptr) {
      *out_error = "No appsink available after chain construction";
    }
    Stop();
    return false;
  }

  g_object_set(G_OBJECT(appsink_),
               "emit-signals", TRUE,
               "sync", config.appsink_sync,
               "drop", config.appsink_drop,
               "max-buffers", config.appsink_max_buffers,
               nullptr);

  if (!config.msgagg.enabled) {
    appsink_signal_id_ = g_signal_connect(appsink_, "new-sample",
        G_CALLBACK(OnNewCombinedSample), this);
    if (appsink_signal_id_ == 0UL) {
      if (out_error != nullptr) {
        *out_error = "Failed to connect appsink new-sample callback";
      }
      Stop();
      return false;
    }
  }

  // Face processing: set flag based on config
  // (qtifaceinfer is in pipeline; we only read its GstFaceDetectionMeta here)
  face_enabled_ = config.face_enabled &&
      (config.face_channel_mask & (1 << channel_id_));
  if (face_enabled_ && shared_gallery != nullptr) {
    gallery_ptr_ = shared_gallery;
    gallery_threshold_ = config.gallery_threshold;
    gallery_min_face_size_ = config.gallery_min_face_size;
    gallery_min_score_ = config.gallery_min_score;
    LogChannel(channel_id_, "Using shared gallery (" +
               std::to_string(gallery_ptr_->Size()) + " entries"
               ", min_face=" + std::to_string(gallery_min_face_size_) +
               "px, min_score=" + std::to_string(gallery_min_score_) + ")");

    if (!config.log_dir.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(config.log_dir, ec);
      std::string log_path = config.log_dir + "/gallery_diag_ch"
                            + std::to_string(channel_id_) + ".log";
      gallery_diag_log_ = std::fopen(log_path.c_str(), "w");
      if (gallery_diag_log_) {
        LogChannel(channel_id_, "gallery diag log: " + log_path);
      } else {
        LogChannel(channel_id_, "failed to open gallery diag log: " + log_path);
      }
    }
  }

  if ((writer_ == nullptr) || !writer_->Open(out_error)) {
    Stop();
    return false;
  }

  initialized_ = true;
  LogChannel(channel_id_,
             std::string("initialized (face=") + (face_enabled_ ? "on" : "off") + ")");
  return true;
}

void ChannelWorker::Stop() {
  LogFinalSampleStatsOnce("final");

  if (gallery_diag_log_ != nullptr) {
    std::fclose(gallery_diag_log_);
    gallery_diag_log_ = nullptr;
  }

  face_enabled_ = false;

  if ((appsink_ != nullptr) && (appsink_signal_id_ != 0UL)) {
    g_signal_handler_disconnect(appsink_, appsink_signal_id_);
    appsink_signal_id_ = 0UL;
  }
  appsink_ = nullptr;

  if (frame_offload_element_ != nullptr) {
    gst_object_unref(frame_offload_element_);
    frame_offload_element_ = nullptr;
  }

  if (writer_ != nullptr) {
    writer_->Close();
  }
  initialized_ = false;
}

std::string ChannelWorker::OutputJsonlPath() const {
  if (writer_ == nullptr) {
    return {};
  }
  return writer_->FilePath();
}

// ---- Combined appsink callback ----

GstFlowReturn ChannelWorker::OnNewCombinedSample(GstElement* appsink, gpointer user_data) {
  auto* self = static_cast<ChannelWorker*>(user_data);
  if ((self == nullptr) || (appsink == nullptr)) {
    return GST_FLOW_OK;
  }

  GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
  if (sample == nullptr) {
    return GST_FLOW_OK;
  }

  const uint64_t sample_total =
      self->appsink_sample_total_.fetch_add(1ULL, std::memory_order_relaxed) + 1ULL;
  bool expected = false;
  if (self->first_sample_received_logged_.compare_exchange_strong(
          expected, true, std::memory_order_relaxed)) {
    LogChannel(self->channel_id_,
               "first_sample_received=1 appsink_sample_total=" +
                   std::to_string(sample_total));
  }

  bool ok = self->HandleCombinedSample(sample);
  gst_sample_unref(sample);
  self->MaybeLogPeriodicSampleStats(sample_total);

  if (!ok) {
    self->LogSampleStats("new-sample-fail");
    if (self->app_context_ != nullptr) {
      self->app_context_->error_received.store(true);
      self->app_context_->RequestStop();
    }
    return GST_FLOW_ERROR;
  }
  return GST_FLOW_OK;
}

bool ChannelWorker::HandleCombinedSample(GstSample* sample) {
  if (sample == nullptr) {
    appsink_video_meta_fail_.fetch_add(1ULL, std::memory_order_relaxed);
    return false;
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (buffer == nullptr) {
    appsink_video_meta_fail_.fetch_add(1ULL, std::memory_order_relaxed);
    return false;
  }

  // Get frame dimensions from caps
  uint32_t frame_w = 0, frame_h = 0;
  GstCaps* caps = gst_sample_get_caps(sample);
  if (caps != nullptr) {
    GstVideoInfo vinfo;
    if (gst_video_info_from_caps(&vinfo, caps)) {
      frame_w = static_cast<uint32_t>(GST_VIDEO_INFO_WIDTH(&vinfo));
      frame_h = static_cast<uint32_t>(GST_VIDEO_INFO_HEIGHT(&vinfo));
    }
  }

  if (frame_w == 0 || frame_h == 0) {
    appsink_video_meta_fail_.fetch_add(1ULL, std::memory_order_relaxed);
    return false;
  }

  // 1. Read GstVideoRegionOfInterestMeta (tracker results via metamux, filtered by qtideduper)
  std::vector<output::DetectionRecord> detections;
  gpointer state = nullptr;
  GstMeta* meta;
  while ((meta = gst_buffer_iterate_meta(buffer, &state)) != nullptr) {
    if (meta->info->api != GST_VIDEO_REGION_OF_INTEREST_META_API_TYPE) {
      continue;
    }
    auto* roi = reinterpret_cast<GstVideoRegionOfInterestMeta*>(meta);
    output::DetectionRecord det;

    const char* roi_label = g_quark_to_string(roi->roi_type);
    if (roi_label == nullptr) { roi_label = ""; }

    if (!ResolveDetectionClass(std::string(roi_label), &det.class_id, &det.label)) {
      continue;  // skip unknown labels
    }

    // ROI coordinates: absolute pixels -> normalized [0,1]
    det.left   = static_cast<double>(roi->x) / frame_w;
    det.top    = static_cast<double>(roi->y) / frame_h;
    det.right  = static_cast<double>(roi->x + roi->w) / frame_w;
    det.bottom = static_cast<double>(roi->y + roi->h) / frame_h;

    // tracking-id and confidence from ROI params GstStructure
    // (metamux strips "rectangle" and "id", keeps all other fields including "tracking-id")
    for (GList* p = roi->params; p != nullptr; p = p->next) {
      auto* s = static_cast<GstStructure*>(p->data);
      guint tid = 0;
      if (gst_structure_get_uint(s, "tracking-id", &tid)) {
        det.track_id = tid;
      }
      gdouble conf = 0.0;
      if (gst_structure_get_double(s, "confidence", &conf)) {
        det.score = conf;
      }
    }

    detections.push_back(std::move(det));
  }

  appsink_video_meta_ok_.fetch_add(1ULL, std::memory_order_relaxed);

  // 2. Read GstFaceDetectionMeta (optional, only when face_enabled=true)
  if (face_enabled_) {
    std::vector<face_infer::FaceResult> faces;
    bool has_arcface = false;

    GstFaceDetectionMeta* face_meta = gst_buffer_get_face_detection_meta(buffer);
    if (face_meta != nullptr && face_meta->n_faces > 0) {
      faces.resize(face_meta->n_faces);
      for (guint i = 0; i < face_meta->n_faces; i++) {
        const GstFaceDetectionEntry& entry = face_meta->faces[i];
        faces[i].det.x1 = entry.x1;
        faces[i].det.y1 = entry.y1;
        faces[i].det.x2 = entry.x2;
        faces[i].det.y2 = entry.y2;
        faces[i].det.score = entry.score;
        std::memcpy(faces[i].det.landmarks, entry.landmarks, sizeof(entry.landmarks));
        if (entry.has_embedding) {
          std::memcpy(faces[i].embedding, entry.embedding, sizeof(entry.embedding));
        } else {
          std::memset(faces[i].embedding, 0, sizeof(faces[i].embedding));
        }
      }
      has_arcface = (face_meta->faces[0].has_embedding != FALSE);
    }

    // 3. MatchFacesToPersons
    if (!faces.empty()) {
      const uint64_t fn = appsink_sample_total_.load(std::memory_order_relaxed);
      MatchFacesToPersons(faces, detections,
                          has_arcface,
                          frame_w, frame_h,
                          gallery_ptr_, &track_id_to_face_, gallery_threshold_,
                          gallery_min_face_size_, gallery_min_score_,
                          channel_id_, fn, gallery_diag_log_);
    }

    // 4. Track cache maintenance
    {
      const uint64_t frame_num = appsink_sample_total_.load(std::memory_order_relaxed);
      for (const auto& det : detections) {
        if (track_id_to_face_.count(det.track_id)) {
          track_id_last_seen_[det.track_id] = frame_num;
        }
      }
      if ((frame_num % 100U) == 0U) {
        for (auto it = track_id_to_face_.begin(); it != track_id_to_face_.end(); ) {
          auto seen_it = track_id_last_seen_.find(it->first);
          if (seen_it == track_id_last_seen_.end() || (frame_num - seen_it->second) > 300U) {
            track_id_last_seen_.erase(it->first);
            it = track_id_to_face_.erase(it);
          } else {
            ++it;
          }
        }
      }
    }
  }

  // 5. Write JSONL (deduper already filtered — only "interesting" frames reach here)
  output::FrameRecord record;
  record.channel_id = channel_id_;
  record.detections = std::move(detections);
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    record.frame_id = local_frame_counter_++;
    uint64_t ts = SystemNowNs();
    if (ts <= last_timestamp_ns_) { ts = last_timestamp_ns_ + 1ULL; }
    last_timestamp_ns_ = ts;
    record.timestamp_ns = ts;
  }

  std::string error;
  if ((writer_ != nullptr) && writer_->Write(record, &error)) {
    appsink_write_ok_.fetch_add(1ULL, std::memory_order_relaxed);
  } else {
    appsink_write_fail_.fetch_add(1ULL, std::memory_order_relaxed);
    LogChannel(channel_id_, "jsonl write failed: " + error);
  }

  if (aggregator_ != nullptr) {
    aggregator::QueueItem item;
    item.record = record;
    aggregator_->TryEnqueue(std::move(item));
  }

  return true;
}

// ---- Label index ----

bool ChannelWorker::LoadLabelIndex(const std::string& labels_path,
                                   std::string* out_error) {
  label_index_.clear();

  std::ifstream file(labels_path);
  if (!file.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open labels file: " + labels_path;
    }
    return false;
  }

  uint32_t class_id = 0U;
  std::string line;
  while (std::getline(file, line)) {
    const std::string label = TrimAscii(line);
    if (label.empty()) {
      continue;
    }

    const auto add_alias = [&](const std::string& alias) -> bool {
      if (alias.empty()) {
        return true;
      }
      auto [it, inserted] = label_index_.emplace(
          alias, LabelIndexEntry {class_id, label});
      if (!inserted && (it->second.class_id != class_id)) {
        if (out_error != nullptr) {
          *out_error = "Duplicate label alias '" + alias +
                       "' in labels file: " + labels_path;
        }
        return false;
      }
      return true;
    };

    if (!add_alias(label)) {
      return false;
    }

    const std::string dotted = ReplaceChar(label, ' ', '.');
    if ((dotted != label) && !add_alias(dotted)) {
      return false;
    }

    ++class_id;
  }

  if (label_index_.empty()) {
    if (out_error != nullptr) {
      *out_error = "Labels file does not contain any valid label entries: " + labels_path;
    }
    return false;
  }

  return true;
}

bool ChannelWorker::ResolveDetectionClass(const std::string& raw_label,
                                          uint32_t* out_class_id,
                                          std::string* out_label) const {
  if ((out_class_id == nullptr) || (out_label == nullptr) || raw_label.empty()) {
    return false;
  }

  auto it = label_index_.find(raw_label);
  if (it == label_index_.end()) {
    const std::string spaced = ReplaceChar(raw_label, '.', ' ');
    it = label_index_.find(spaced);
    if (it == label_index_.end()) {
      return false;
    }
  }

  *out_class_id = it->second.class_id;
  *out_label = it->second.canonical_label;
  return true;
}

std::string ChannelWorker::ReplaceChar(std::string text, char from, char to) {
  std::replace(text.begin(), text.end(), from, to);
  return text;
}

std::string ChannelWorker::TrimAscii(std::string text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.pop_back();
  }

  size_t first = 0U;
  while ((first < text.size()) && std::isspace(static_cast<unsigned char>(text[first]))) {
    ++first;
  }

  return text.substr(first);
}

uint64_t ChannelWorker::SystemNowNs() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

}  // namespace multi_cam_app::channel_worker
