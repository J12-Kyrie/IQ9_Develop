#include "vlm/jsonl_logger.h"
#include <nlohmann/json.hpp>

namespace vlm {

JsonlLogger::JsonlLogger(const std::string& path) : path_(path) {
    file_.open(path, std::ios::app);
}

JsonlLogger::~JsonlLogger() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

bool JsonlLogger::log(const VlmLogEntry& entry) {
    nlohmann::json j = {
        {"ts", entry.timestamp_ns},
        {"src", entry.source_id},
        {"rule", entry.rule_name},
        {"dets", entry.detections.empty() ? nlohmann::json::array() : nlohmann::json::parse(entry.detections, nullptr, false)},
        {"resp", entry.vlm_response},
        {"ok", entry.vlm_success},
        {"err", entry.vlm_error},
        {"enc_ms", entry.encode_ms},
        {"http_ms", entry.http_ms},
        {"total_ms", entry.total_ms},
        {"frame", entry.frame_number},
        {"jpeg_bytes", entry.jpeg_size_bytes}
    };

    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_.is_open()) return false;
    file_ << j.dump() << "\n";
    file_.flush();
    return true;
}

} // namespace vlm
