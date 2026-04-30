#include "utils/mqtt/MqttPublisher.hpp"
#include "utils/mqtt/MqttClient.hpp"

#include <cstdio>

namespace multi_cam_app::mqtt {

MqttPublisher::MqttPublisher(const std::string& broker_ip, int port,
                              const std::string& default_topic, int qos)
    : broker_ip_(broker_ip)
    , port_(port)
    , default_topic_(default_topic)
    , default_qos_(qos) {}

MqttPublisher::~MqttPublisher() {
  Shutdown();
}

bool MqttPublisher::Initialize(const std::string& client_id,
                                std::string* out_error) {
  client_ = std::make_unique<MqttClient>(broker_ip_, port_);
  if (!client_->Connect(client_id, out_error)) {
    client_.reset();
    return false;
  }
  std::printf("[MqttPublisher] Initialized (topic='%s', qos=%d)\n",
              default_topic_.c_str(), default_qos_);
  std::fflush(stdout);
  return true;
}

void MqttPublisher::Shutdown() {
  if (client_ != nullptr) {
    client_->Disconnect();
    client_.reset();
  }
}

bool MqttPublisher::Publish(const std::string& payload,
                             std::string* out_error) {
  return Publish(default_topic_, payload, default_qos_, out_error);
}

bool MqttPublisher::Publish(const std::string& topic,
                             const std::string& payload,
                             int qos, std::string* out_error) {
  if (client_ == nullptr) {
    if (out_error != nullptr) {
      *out_error = "MqttPublisher not initialized";
    }
    return false;
  }
  return client_->Publish(topic, payload, qos, false, out_error);
}

bool MqttPublisher::IsConnected() const {
  return (client_ != nullptr) && client_->IsConnected();
}

uint64_t MqttPublisher::PublishedCount() const {
  return (client_ != nullptr) ? client_->PublishedCount() : 0ULL;
}

}  // namespace multi_cam_app::mqtt
