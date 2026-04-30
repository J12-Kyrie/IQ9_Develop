#include "utils/mqtt/MqttSubscriber.hpp"
#include "utils/mqtt/MqttClient.hpp"

#include <cstdio>

namespace multi_cam_app::mqtt {

MqttSubscriber::MqttSubscriber(const std::string& broker_ip, int port,
                                const std::string& default_topic, int qos)
    : broker_ip_(broker_ip)
    , port_(port)
    , default_topic_(default_topic)
    , default_qos_(qos) {}

MqttSubscriber::~MqttSubscriber() {
  Shutdown();
}

bool MqttSubscriber::Initialize(const std::string& client_id,
                                 std::string* out_error) {
  client_ = std::make_unique<MqttClient>(broker_ip_, port_);
  if (!client_->Connect(client_id, out_error)) {
    client_.reset();
    return false;
  }

  client_->SetMessageCallback(
      [this](const std::string& topic, const std::string& payload) {
        OnMessage(topic, payload);
      });

  if (!default_topic_.empty()) {
    if (!client_->Subscribe(default_topic_, default_qos_, out_error)) {
      client_->Disconnect();
      client_.reset();
      return false;
    }
  }

  std::printf("[MqttSubscriber] Initialized (topic='%s', qos=%d)\n",
              default_topic_.c_str(), default_qos_);
  std::fflush(stdout);
  return true;
}

void MqttSubscriber::Shutdown() {
  if (client_ != nullptr) {
    client_->Disconnect();
    client_.reset();
  }
}

bool MqttSubscriber::Subscribe(const std::string& topic, int qos,
                                std::string* out_error) {
  if (client_ == nullptr) {
    if (out_error != nullptr) {
      *out_error = "MqttSubscriber not initialized";
    }
    return false;
  }
  return client_->Subscribe(topic, qos, out_error);
}

bool MqttSubscriber::SubscribeMultiple(const std::vector<std::string>& topics,
                                        int qos, std::string* out_error) {
  for (const auto& topic : topics) {
    if (!Subscribe(topic, qos, out_error)) {
      return false;
    }
  }
  return true;
}

void MqttSubscriber::SetMessageHandler(MessageHandler handler) {
  message_handler_ = std::move(handler);
}

void MqttSubscriber::OnMessage(const std::string& topic,
                                const std::string& payload) {
  if (message_handler_) {
    try {
      message_handler_(topic, payload);
    } catch (const std::exception& e) {
      std::printf("[MqttSubscriber] Exception in handler: %s\n", e.what());
      std::fflush(stdout);
    }
  }

  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (message_queue_.size() >= kMaxQueueSize) {
    message_queue_.pop();
  }
  message_queue_.push({topic, payload});
}

bool MqttSubscriber::HasMessages() const {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return !message_queue_.empty();
}

bool MqttSubscriber::PopMessage(std::string* out_topic,
                                 std::string* out_payload) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (message_queue_.empty()) {
    return false;
  }
  auto& front = message_queue_.front();
  if (out_topic != nullptr) {
    *out_topic = std::move(front.first);
  }
  if (out_payload != nullptr) {
    *out_payload = std::move(front.second);
  }
  message_queue_.pop();
  return true;
}

size_t MqttSubscriber::QueueSize() const {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return message_queue_.size();
}

bool MqttSubscriber::IsConnected() const {
  return (client_ != nullptr) && client_->IsConnected();
}

uint64_t MqttSubscriber::ReceivedCount() const {
  return (client_ != nullptr) ? client_->ReceivedCount() : 0ULL;
}

}  // namespace multi_cam_app::mqtt
