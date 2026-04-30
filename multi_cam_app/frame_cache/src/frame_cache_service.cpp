#include "frame_cache/FrameCacheService.hpp"

#include <cstdio>

#include <nlohmann/json.hpp>

#include "utils/mqtt/MqttSubscriber.hpp"

namespace multi_cam_app::frame_cache {

FrameCacheService::FrameCacheService() = default;

FrameCacheService::~FrameCacheService() {
  Shutdown();
}

bool FrameCacheService::Init(const InitParams& p, std::string* out_error) {
  if (inited_) {
    if (out_error != nullptr) {
      *out_error = "FrameCacheService: already initialized";
    }
    return false;
  }

  if (p.frame_done_topic.empty()) {
    if (out_error != nullptr) {
      *out_error = "FrameCacheService: frame_done_topic is empty";
    }
    return false;
  }

  producer_ = std::make_unique<dmabuf_producer::DmaBufProducer>();
  std::string pe;
  if (!producer_->Init(p.producer, &pe)) {
    producer_.reset();
    if (out_error != nullptr) {
      *out_error = "FrameCacheService DmaBufProducer: " + pe;
    }
    return false;
  }

  mqtt_sub_ = std::make_unique<mqtt::MqttSubscriber>(
      p.broker_ip, p.port, p.frame_done_topic, p.mqtt_qos);

  mqtt_sub_->SetMessageHandler(
      [this](const std::string& topic, const std::string& payload) {
        OnFrameDonePayload(topic, payload);
      });

  std::string me;
  const std::string cid = p.mqtt_client_id + "_frame_done";
  if (!mqtt_sub_->Initialize(cid, &me)) {
    producer_->Shutdown();
    producer_.reset();
    mqtt_sub_.reset();
    if (out_error != nullptr) {
      *out_error = "FrameCacheService MQTT: " + me;
    }
    return false;
  }

  inited_ = true;
  std::printf("FrameCacheService: MQTT subscribed to '%s'\n",
              p.frame_done_topic.c_str());
  std::fflush(stdout);
  return true;
}

void FrameCacheService::OnFrameDonePayload(const std::string& /*topic*/,
                                           const std::string& payload) {
  if (producer_ == nullptr) {
    return;
  }
  nlohmann::json j = nlohmann::json::parse(payload, nullptr, false);
  if (j.is_discarded()) {
    return;
  }
  if (j.value("type", std::string()) != "frame_done") {
    return;
  }
  if (!j.contains("release_entries") || !j["release_entries"].is_array()) {
    return;
  }
  for (const auto& re : j["release_entries"]) {
    if (!re.is_object()) {
      continue;
    }
    const uint32_t slot_index = re.value("slot_index", 0u);
    producer_->ReleaseSlotDirect(slot_index);
  }
}

void FrameCacheService::Shutdown() {
  if (!inited_) {
    return;
  }
  if (mqtt_sub_ != nullptr) {
    mqtt_sub_->Shutdown();
    mqtt_sub_.reset();
  }
  if (producer_ != nullptr) {
    producer_->Shutdown();
    producer_.reset();
  }
  inited_ = false;
}

}  // namespace multi_cam_app::frame_cache
