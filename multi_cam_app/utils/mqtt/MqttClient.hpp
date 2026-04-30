#ifndef MULTI_CAM_APP_UTILS_MQTT_MQTT_CLIENT_HPP
#define MULTI_CAM_APP_UTILS_MQTT_MQTT_CLIENT_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

#include <MQTTClient.h>

namespace multi_cam_app::mqtt {

class MqttClient {
public:
  using MessageCallback = std::function<void(const std::string& topic,
                                             const std::string& payload)>;

  MqttClient(const std::string& broker_ip, int port);
  ~MqttClient();

  MqttClient(const MqttClient&) = delete;
  MqttClient& operator=(const MqttClient&) = delete;

  bool Connect(const std::string& client_id, std::string* out_error);
  void Disconnect();
  bool IsConnected() const { return connected_.load(); }

  bool Subscribe(const std::string& topic, int qos, std::string* out_error);
  bool Unsubscribe(const std::string& topic, std::string* out_error);

  bool Publish(const std::string& topic, const std::string& payload,
               int qos, bool retained, std::string* out_error);

  void SetMessageCallback(MessageCallback callback);

  uint64_t PublishedCount() const { return published_count_.load(); }
  uint64_t ReceivedCount() const { return received_count_.load(); }

private:
  static int OnMessageArrived(void* context, char* topic_name,
                              int topic_len, MQTTClient_message* message);
  static void OnConnectionLost(void* context, char* cause);

  std::string broker_url_;
  MQTTClient client_ {nullptr};

  std::mutex connect_mutex_;
  std::mutex publish_mutex_;
  std::mutex callback_mutex_;
  MessageCallback message_callback_;

  std::atomic<bool> connected_ {false};
  std::atomic<uint64_t> published_count_ {0};
  std::atomic<uint64_t> received_count_ {0};
};

}  // namespace multi_cam_app::mqtt

#endif  // MULTI_CAM_APP_UTILS_MQTT_MQTT_CLIENT_HPP
