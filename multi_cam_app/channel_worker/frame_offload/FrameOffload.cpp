#include "FrameOffload.hpp"

#include <chrono>
#include <cstring>
#include <vector>

#include <gst/video/gstvideometa.h>

#include "channel_worker/infer_module/face_infer/gst_plugin/gstfacedetectionmeta.h"
#include "utils/JsonlWriter.hpp"

namespace multi_cam_app::frame_offload {

FrameOffload::FrameOffload(const FrameOffloadConfig& config)
    : config_(config) {
    if (!config_.jsonl_out_dir.empty()) {
        std::string path = config_.jsonl_out_dir + "/scene_update_ch"
                         + std::to_string(config_.channel_id) + ".jsonl";
        jsonl_file_.open(path, std::ios::app);
    }
}

void FrameOffload::WriteSceneJsonl(const std::string& json) {
    if (jsonl_file_.is_open()) {
        jsonl_file_ << json << "\n" << std::flush;
    }
}

std::string FrameOffload::ExtractAndSerialize(GstBuffer* buffer,
                                              uint32_t frame_w, uint32_t frame_h,
                                              int32_t image_path) {
    std::vector<output::DetectionRecord> detections;

    // 1. Iterate GstVideoRegionOfInterestMeta (tracker results via metamux, filtered by deduper)
    if (frame_w > 0 && frame_h > 0) {
        gpointer state = nullptr;
        GstMeta* meta;
        while ((meta = gst_buffer_iterate_meta(buffer, &state)) != nullptr) {
            if (meta->info->api != GST_VIDEO_REGION_OF_INTEREST_META_API_TYPE)
                continue;

            auto* roi = reinterpret_cast<GstVideoRegionOfInterestMeta*>(meta);
            output::DetectionRecord det;

            const char* roi_label = g_quark_to_string(roi->roi_type);
            if (roi_label != nullptr) {
                det.label = roi_label;
            }
            det.class_id = roi->id;

            det.left   = static_cast<double>(roi->x) / frame_w;
            det.top    = static_cast<double>(roi->y) / frame_h;
            det.right  = static_cast<double>(roi->x + roi->w) / frame_w;
            det.bottom = static_cast<double>(roi->y + roi->h) / frame_h;

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
    }

    // 2. Read GstFaceDetectionMeta (optional)
    GstFaceDetectionMeta* face_meta = gst_buffer_get_face_detection_meta(buffer);
    if (face_meta != nullptr && face_meta->n_faces > 0) {
        for (guint fi = 0; fi < face_meta->n_faces; fi++) {
            const GstFaceDetectionEntry& entry = face_meta->faces[fi];

            float face_cx = (entry.x1 + entry.x2) * 0.5f;
            float face_cy = (entry.y1 + entry.y2) * 0.5f;

            // Match face to best overlapping person detection (upper-half priority)
            int best_idx = -1;
            double best_score = -1.0;
            for (size_t di = 0; di < detections.size(); di++) {
                const auto& d = detections[di];
                double abs_left   = d.left   * frame_w;
                double abs_top    = d.top    * frame_h;
                double abs_right  = d.right  * frame_w;
                double abs_bottom = d.bottom * frame_h;
                if (face_cx >= abs_left && face_cx <= abs_right &&
                    face_cy >= abs_top  && face_cy <= abs_bottom) {
                    double mid_y = (abs_top + abs_bottom) * 0.5;
                    bool in_upper = (face_cy <= mid_y);
                    double area = (abs_right - abs_left) * (abs_bottom - abs_top);
                    if (area < 1.0) area = 1.0;
                    double sc = (in_upper ? 1e6 : 0.0) + 1.0 / area;
                    if (sc > best_score) {
                        best_score = sc;
                        best_idx = static_cast<int>(di);
                    }
                }
            }

            if (best_idx >= 0) {
                output::FaceInfo fi_out;
                fi_out.confidence = entry.score;
                fi_out.bbox_x1 = (frame_w > 0) ? entry.x1 / frame_w : 0.0f;
                fi_out.bbox_y1 = (frame_h > 0) ? entry.y1 / frame_h : 0.0f;
                fi_out.bbox_x2 = (frame_w > 0) ? entry.x2 / frame_w : 0.0f;
                fi_out.bbox_y2 = (frame_h > 0) ? entry.y2 / frame_h : 0.0f;
                for (int li = 0; li < FACE_META_NUM_LANDMARKS; li++) {
                    fi_out.landmarks[li * 2]     = (frame_w > 0) ? entry.landmarks[li][0] / frame_w : 0.0f;
                    fi_out.landmarks[li * 2 + 1] = (frame_h > 0) ? entry.landmarks[li][1] / frame_h : 0.0f;
                }
                if (entry.has_embedding) {
                    fi_out.embedding.assign(entry.embedding,
                                            entry.embedding + FACE_META_EMBEDDING_DIM);
                }
                fi_out.linked_track_id = detections[best_idx].track_id;
                fi_out.face_id = -1;
                fi_out.face_similarity = 0.0f;

                // Gallery matching (Match/Enroll/cache) when gallery is configured
                if (config_.gallery != nullptr && entry.has_embedding) {
                    uint32_t tid = static_cast<uint32_t>(fi_out.linked_track_id);
                    auto cache_it = track_id_to_face_.find(tid);
                    if (cache_it != track_id_to_face_.end()) {
                        fi_out.face_id = cache_it->second;
                        fi_out.face_similarity = 1.0f;  // cached match
                    } else {
                        gallery::MatchResult match_result = config_.gallery->Match(
                            entry.embedding, config_.gallery_threshold);
                        if (match_result.face_id >= 0) {
                            fi_out.face_id = match_result.face_id;
                            fi_out.face_similarity = match_result.similarity;
                            track_id_to_face_[tid] = match_result.face_id;
                        } else {
                            float face_w = entry.x2 - entry.x1;
                            float face_h = entry.y2 - entry.y1;
                            float face_size = (face_w < face_h) ? face_w : face_h;
                            if (face_size >= static_cast<float>(config_.gallery_min_face_size) &&
                                entry.score >= config_.gallery_min_score) {
                                int new_face_id = config_.gallery->Enroll(entry.embedding);
                                fi_out.face_id = new_face_id;
                                fi_out.face_similarity = 0.0f;
                                track_id_to_face_[tid] = new_face_id;
                            }
                        }
                    }
                }

                detections[best_idx].face = std::move(fi_out);
            }
        }
    }

    // 3. Build FrameRecord
    output::FrameRecord record;
    record.channel_id = config_.channel_id;
    record.image_path = image_path;
    record.detections = std::move(detections);
    record.frame_id = local_frame_counter_++;

    auto now = std::chrono::system_clock::now().time_since_epoch();
    uint64_t ts = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    if (ts <= last_timestamp_ns_) { ts = last_timestamp_ns_ + 1ULL; }
    last_timestamp_ns_ = ts;
    record.timestamp_ns = ts;

    frames_processed_++;

    // Track cache maintenance: update last-seen for all tracked detections
    for (const auto& det : record.detections) {
        track_id_last_seen_[det.track_id] = frames_processed_;
    }
    // Periodic eviction: every 100 frames, drop tracks not seen in 300+ frames
    if (frames_processed_ % 100 == 0) {
        auto it = track_id_last_seen_.begin();
        while (it != track_id_last_seen_.end()) {
            if (frames_processed_ - it->second > 300) {
                track_id_to_face_.erase(it->first);
                it = track_id_last_seen_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 4. Serialize
    if (config_.scene_update_mqtt) {
      return output::JsonlWriter::SerializeSceneSource(
          record, config_.meta_width, config_.meta_height);
    }
    return output::JsonlWriter::Serialize(record);
}

}  // namespace multi_cam_app::frame_offload
