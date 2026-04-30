/*
 * Forked from new_pipeline/rule_engine/src/adapters/EventAdapter.cpp
 * NvSciBuf_Pipeline: image_meta required; image_path/rgba_path skip source.
 */
#include "adapters/EventAdapter.h"
#include "infra/Logger.h"
#include <nlohmann/json.hpp>
#include <cstdio>

namespace {

std::string formatSourceId(int sourceId) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "source_%03d", sourceId);
  return std::string(buf);
}

} // namespace

namespace adapters {

std::vector<core::EventMeta>
EventAdapter::adaptSceneUpdate(const std::string &jsonString) {
  std::vector<core::EventMeta> events;
  const auto j = nlohmann::json::parse(jsonString, nullptr, false);
  if (j.is_discarded()) {
    utils::Logger::warn("EventAdapter", "JSON parse failed");
    return events;
  }

  if (j.value("type", "") != "scene_update") {
    return events;
  }

  auto sourcesIt = j.find("sources");
  if (sourcesIt == j.end() || !sourcesIt->is_array()) {
    return events;
  }

  for (const auto& source : *sourcesIt) {
    if (!source.is_object()) {
      continue;
    }

    if (source.contains("image_path") || source.contains("rgba_path")) {
      utils::Logger::warn("EventAdapter", "Skip source: image_path/rgba_path not allowed");
      continue;
    }

    auto im = source.find("image_meta");
    if (im == source.end() || !im->is_object()) {
      utils::Logger::warn("EventAdapter", "Skip source: missing image_meta");
      continue;
    }

    const auto& imo = *im;
    if (!imo.contains("slot_index") || !imo.contains("width") || !imo.contains("height")) {
      utils::Logger::warn("EventAdapter", "Skip source: incomplete image_meta");
      continue;
    }

    core::EventMeta meta;
    const int sid = source.value("source_id", 0);
    meta.source_id_num_ = sid;
    meta.source_id_ = formatSourceId(sid);
    meta.timestamp_ns_ = source.value("timestamp_ns", 0ULL);
    meta.timestamp_ = meta.timestamp_ns_ / 1000;
    meta.slot_index_ = imo["slot_index"].get<uint32_t>();
    meta.width_ = imo["width"].get<uint32_t>();
    meta.height_ = imo["height"].get<uint32_t>();
    meta.channels_ = imo.value("channels", 3u);

    auto removedIt = source.find("removed_track_ids");
    if (removedIt != source.end() && removedIt->is_array()) {
      meta.removed_track_ids_ = removedIt->get<std::vector<int>>();
    }

    auto eventsIt = source.find("events");
    if (eventsIt != source.end() && eventsIt->is_array()) {
      for (const auto& evt : *eventsIt) {
        if (!evt.is_object()) {
          continue;
        }
        core::Detection det;
        det.class_name_ = evt.value("class_name", "unknown");
        det.track_id_ = evt.value("track_id", 0);
        det.confidence_ = evt.value("confidence", 1.0f);
        auto bboxIt = evt.find("bbox");
        if (bboxIt != evt.end() && bboxIt->is_array() && bboxIt->size() == 4) {
          det.bbox_ = bboxIt->get<std::vector<float>>();
        }
        auto embIt = evt.find("embedding");
        if (embIt != evt.end() && embIt->is_array()) {
          det.embedding_ = embIt->get<std::vector<float>>();
        }
        meta.detections_.push_back(det);
      }
    }

    events.push_back(meta);
  }
  return events;
}

} // namespace adapters
