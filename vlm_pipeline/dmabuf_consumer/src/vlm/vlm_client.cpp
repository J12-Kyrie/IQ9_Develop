#include "vlm/vlm_client.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdio>
#include <sys/stat.h>

namespace vlm {

namespace {

size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

} // namespace

VlmClient::VlmClient(const Config& cfg) : config_(cfg) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

VlmClient::VlmClient() : config_(Config{}) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

VlmClient::~VlmClient() {
    curl_global_cleanup();
}

std::string VlmClient::buildRequestBody(const VlmRequest& req, const std::string& file_url) {
    nlohmann::json image_url_obj;
    image_url_obj["url"] = file_url;
    nlohmann::json content_item_img;
    content_item_img["type"] = "image_url";
    content_item_img["image_url"] = image_url_obj;
    nlohmann::json content_item_txt;
    content_item_txt["type"] = "text";
    content_item_txt["text"] = req.prompt;
    nlohmann::json message;
    message["role"] = "user";
    message["content"] = nlohmann::json::array({content_item_img, content_item_txt});
    nlohmann::json body;
    body["model"] = config_.model;
    body["messages"] = nlohmann::json::array({message});
    return body.dump();
}

VlmResponse VlmClient::parseResponse(const std::string& body, double latency_ms) {
    VlmResponse resp;
    resp.latency_ms = latency_ms;
    resp.vlm_inference_ms = 0.0;
    resp.created = 0;
    try {
        auto j = nlohmann::json::parse(body);
        resp.content = j["choices"][0]["message"]["content"].get<std::string>();
        if (j.contains("created")) {
            resp.created = j["created"].get<int64_t>();
        }
        resp.success = true;
    } catch (const std::exception& e) {
        resp.success = false;
        resp.error = std::string("JSON parse error: ") + e.what();
    }
    return resp;
}

VlmResponse VlmClient::infer(const VlmRequest& req) {
    auto t0 = std::chrono::steady_clock::now();

    CURL* curl = curl_easy_init();
    if (!curl) {
        VlmResponse resp{};
        resp.success = false;
        resp.error = "curl_easy_init failed";
        return resp;
    }

    // Save JPEG to output directory
    static const char* kOutputDir = "/mnt/workspace/develop/vlm_pipeline/output";
    ::mkdir(kOutputDir, 0755);
    std::string tmp_path = std::string(kOutputDir) + "/vlm_frame_" + std::to_string(req.frame_number) + ".jpg";
    {
        FILE* fp = std::fopen(tmp_path.c_str(), "wb");
        if (!fp) {
            VlmResponse resp{};
            resp.success = false;
            resp.error = "failed to open temp file: " + tmp_path;
            curl_easy_cleanup(curl);
            return resp;
        }
        std::fwrite(req.jpeg_data.data(), 1, req.jpeg_data.size(), fp);
        std::fclose(fp);
    }

    std::string url = config_.server_url + "/v1/chat/completions";
    std::string file_url = "file://" + tmp_path;
    std::string request_body = buildRequestBody(req, file_url);
    std::string response_body;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(request_body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(config_.timeout_seconds));

    CURLcode res = curl_easy_perform(curl);

    auto t1 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::milli>(t1 - t0).count();

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        VlmResponse resp{};
        resp.success = false;
        resp.error = std::string("curl error: ") + curl_easy_strerror(res);
        resp.latency_ms = latency;
        return resp;
    }

    return parseResponse(response_body, latency);
}

} // namespace vlm
