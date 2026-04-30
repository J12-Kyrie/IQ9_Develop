#ifndef MULTI_CAM_APP_UTILS_MQTT_MQTT_PUBLISHER_HPP
#define MULTI_CAM_APP_UTILS_MQTT_MQTT_PUBLISHER_HPP

#include <cstdint>
#include <memory>
#include <string>

namespace multi_cam_app::mqtt {

class MqttClient;

class MqttPublisher {
public:
  MqttPublisher(const std::string& broker_ip, int port,
                const std::string& default_topic, int qos);
  ~MqttPublisher();

  MqttPublisher(const MqttPublisher&) = delete;
  MqttPublisher& operator=(const MqttPublisher&) = delete;

  bool Initialize(const std::string& client_id, std::string* out_error);
  void Shutdown();

  bool Publish(const std::string& payload, std::string* out_error);
  bool Publish(const std::string& topic, const std::string& payload,
               int qos, std::string* out_error);

  bool IsConnected() const;
  uint64_t PublishedCount() const;

private:
  std::string broker_ip_;
  int port_;
  std::string default_topic_;
  int default_qos_;
  std::unique_ptr<MqttClient> client_;
};

}  // namespace multi_cam_app::mqtt

#endif  // MULTI_CAM_APP_UTILS_MQTT_MQTT_PUBLISHER_HPP
