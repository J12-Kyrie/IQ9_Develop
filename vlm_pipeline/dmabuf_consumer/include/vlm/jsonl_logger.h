#pragma once
#include <mutex>
#include <string>
#include <fstream>
#include <cstdint>

namespace vlm {

struct VlmLogEntry {
    uint64_t timestamp_ns;
    int source_id;
    std::string rule_name;
    std::string detections;
    std::string vlm_response;
    bool vlm_success;
    std::string vlm_error;
    double encode_ms;
    double http_ms;
    double total_ms;
    uint64_t frame_number;
    size_t jpeg_size_bytes;
};

class JsonlLogger {
public:
    explicit JsonlLogger(const std::string& path);
    ~JsonlLogger();

    bool log(const VlmLogEntry& entry);

private:
    std::string path_;
    std::mutex mutex_;
    std::ofstream file_;
};

} // namespace vlm
