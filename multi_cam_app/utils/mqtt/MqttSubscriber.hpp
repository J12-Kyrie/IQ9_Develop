#ifndef MULTI_CAM_APP_UTILS_MQTT_MQTT_SUBSCRIBER_HPP
#define MULTI_CAM_APP_UTILS_MQTT_MQTT_SUBSCRIBER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace multi_cam_app::mqtt {

class MqttClient;

class MqttSubscriber {
public:
  using MessageHandler = std::function<void(const std::string& topic,
                                            const std::string& payload)>;

  MqttSubscriber(const std::string& broker_ip, int port,
                 const std::string& default_topic, int qos);
  ~MqttSubscriber();

  MqttSubscriber(const MqttSubscriber&) = delete;
  MqttSubscriber& operator=(const MqttSubscriber&) = delete;

  bool Initialize(const std::string& client_id, std::string* out_error);
  void Shutdown();

  bool Subscribe(const std::string& topic, int qos, std::string* out_error);
  bool SubscribeMultiple(const std::vector<std::string>& topics, int qos,
                         std::string* out_error);

  void SetMessageHandler(MessageHandler handler);

  bool HasMessages() const;
  bool PopMessage(std::string* out_topic, std::string* out_payload);
  size_t QueueSize() const;

  bool IsConnected() const;
  uint64_t ReceivedCount() const;

private:
  static constexpr size_t kMaxQueueSize = 1000;

  void OnMessage(const std::string& topic, const std::string& payload);

  std::string broker_ip_;
  int port_;
  std::string default_topic_;
  int default_qos_;
  std::unique_ptr<MqttClient> client_;

  MessageHandler message_handler_;

  mutable std::mutex queue_mutex_;
  std::queue<std::pair<std::string, std::string>> message_queue_;
};

}  // namespace multi_cam_app::mqtt

#endif  // MULTI_CAM_APP_UTILS_MQTT_MQTT_SUBSCRIBER_HPP
