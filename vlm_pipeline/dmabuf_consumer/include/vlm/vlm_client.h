#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vlm {

struct VlmRequest {
    std::vector<uint8_t> jpeg_data;
    std::string prompt;
    int source_id;
    std::string rule_name;
    uint64_t timestamp_ns;
    uint64_t frame_number;
    std::string detections_json;
};

struct VlmResponse {
    bool success;
    std::string content;
    std::string error;
    double latency_ms;
    double vlm_inference_ms;
    int64_t created;
};

class VlmClient {
public:
    struct Config {
        std::string server_url = "http://127.0.0.1:8000";
        std::string model = "qwen3-vl-4b";
        int timeout_seconds = 30;
        int max_retries = 1;
    };

    explicit VlmClient(const Config& cfg);
    VlmClient();
    ~VlmClient();

    VlmResponse infer(const VlmRequest& req);

private:
    Config config_;

    std::string buildRequestBody(const VlmRequest& req, const std::string& file_url);
    VlmResponse parseResponse(const std::string& body, double latency_ms);
};

} // namespace vlm
