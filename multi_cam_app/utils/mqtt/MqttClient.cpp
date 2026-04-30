#include "utils/mqtt/MqttClient.hpp"

#include <cstdio>

namespace multi_cam_app::mqtt {

MqttClient::MqttClient(const std::string& broker_ip, int port)
    : broker_url_("tcp://" + broker_ip + ":" + std::to_string(port)) {}

MqttClient::~MqttClient() {
  Disconnect();
}

bool MqttClient::Connect(const std::string& client_id,
                          std::string* out_error) {
  std::lock_guard<std::mutex> lock(connect_mutex_);

  int rc = MQTTClient_create(&client_, broker_url_.c_str(), client_id.c_str(),
                              MQTTCLIENT_PERSISTENCE_NONE, nullptr);
  if (rc != MQTTCLIENT_SUCCESS) {
    if (out_error != nullptr) {
      *out_error = std::string("MQTTClient_create failed: ") +
                   MQTTClient_strerror(rc);
    }
    return false;
  }

  MQTTClient_setCallbacks(client_, this, OnConnectionLost,
                           OnMessageArrived, nullptr);

  MQTTClient_connectOptions opts = MQTTClient_connectOptions_initializer;
  opts.keepAliveInterval = 20;
  opts.cleansession = 1;

  rc = MQTTClient_connect(client_, &opts);
  if (rc != MQTTCLIENT_SUCCESS) {
    if (out_error != nullptr) {
      *out_error = "Failed to connect to " + broker_url_ + ": " +
                   MQTTClient_strerror(rc);
    }
    MQTTClient_destroy(&client_);
    client_ = nullptr;
    return false;
  }

  connected_.store(true);
  std::printf("[MqttClient] Connected to %s\n", broker_url_.c_str());
  std::fflush(stdout);
  return true;
}

void MqttClient::Disconnect() {
  if (!connected_.load()) {
    return;
  }

  std::lock_guard<std::mutex> lock(connect_mutex_);
  if (client_ != nullptr) {
    MQTTClient_disconnect(client_, 1000);
    MQTTClient_destroy(&client_);
    client_ = nullptr;
  }
  connected_.store(false);
  std::printf("[MqttClient] Disconnected\n");
  std::fflush(stdout);
}

bool MqttClient::Subscribe(const std::string& topic, int qos,
                            std::string* out_error) {
  if (!connected_.load()) {
    if (out_error != nullptr) {
      *out_error = "Cannot subscribe: not connected";
    }
    return false;
  }

  int rc = MQTTClient_subscribe(client_, topic.c_str(), qos);
  if (rc != MQTTCLIENT_SUCCESS) {
    if (out_error != nullptr) {
      *out_error = "Subscribe failed for topic '" + topic + "': " +
                   MQTTClient_strerror(rc);
    }
    return false;
  }

  std::printf("[MqttClient] Subscribed to '%s' (qos=%d)\n",
              topic.c_str(), qos);
  std::fflush(stdout);
  return true;
}

bool MqttClient::Unsubscribe(const std::string& topic,
                              std::string* out_error) {
  if (!connected_.load()) {
    if (out_error != nullptr) {
      *out_error = "Cannot unsubscribe: not connected";
    }
    return false;
  }

  int rc = MQTTClient_unsubscribe(client_, topic.c_str());
  if (rc != MQTTCLIENT_SUCCESS) {
    if (out_error != nullptr) {
      *out_error = "Unsubscribe failed for topic '" + topic + "': " +
                   MQTTClient_strerror(rc);
    }
    return false;
  }
  return true;
}

bool MqttClient::Publish(const std::string& topic, const std::string& payload,
                          int qos, bool retained, std::string* out_error) {
  if (!connected_.load()) {
    if (out_error != nullptr) {
      *out_error = "Cannot publish: not connected";
    }
    return false;
  }

  std::lock_guard<std::mutex> lock(publish_mutex_);

  int rc = MQTTClient_publish(client_, topic.c_str(),
                               static_cast<int>(payload.size()),
                               payload.data(), qos,
                               retained ? 1 : 0, nullptr);
  if (rc != MQTTCLIENT_SUCCESS) {
    if (out_error != nullptr) {
      *out_error = "Publish failed to '" + topic + "': " +
                   MQTTClient_strerror(rc);
    }
    return false;
  }

  published_count_.fetch_add(1);
  return true;
}

void MqttClient::SetMessageCallback(MessageCallback callback) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  message_callback_ = std::move(callback);
}

int MqttClient::OnMessageArrived(void* context, char* topic_name,
                                  int topic_len,
                                  MQTTClient_message* message) {
  auto* self = static_cast<MqttClient*>(context);
  if (self != nullptr && message != nullptr) {
    std::string topic = (topic_len > 0)
        ? std::string(topic_name, static_cast<size_t>(topic_len))
        : std::string(topic_name);
    std::string payload(static_cast<const char*>(message->payload),
                        static_cast<size_t>(message->payloadlen));

    self->received_count_.fetch_add(1);

    std::lock_guard<std::mutex> lock(self->callback_mutex_);
    if (self->message_callback_) {
      try {
        self->message_callback_(topic, payload);
      } catch (const std::exception& e) {
        std::printf("[MqttClient] Exception in callback: %s\n", e.what());
        std::fflush(stdout);
      }
    }
  }

  MQTTClient_freeMessage(&message);
  MQTTClient_free(topic_name);
  return 1;
}

void MqttClient::OnConnectionLost(void* context, char* cause) {
  auto* self = static_cast<MqttClient*>(context);
  if (self != nullptr) {
    self->connected_.store(false);
    std::printf("[MqttClient] Connection lost: %s\n",
                (cause != nullptr) ? cause : "unknown");
    std::fflush(stdout);
  }
}

}  // namespace multi_cam_app::mqtt
