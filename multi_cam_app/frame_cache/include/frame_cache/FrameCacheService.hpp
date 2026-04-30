#ifndef MULTI_CAM_APP_FRAME_CACHE_FRAME_CACHE_SERVICE_HPP
#define MULTI_CAM_APP_FRAME_CACHE_FRAME_CACHE_SERVICE_HPP

#include <memory>
#include <string>

#include "dmabuf_producer/producer.h"

namespace multi_cam_app::mqtt {
class MqttSubscriber;
}

namespace multi_cam_app::frame_cache {

// Owns DmaBufProducer + MQTT frame_done subscription (dual path slot release with UDS).
class FrameCacheService {
public:
  struct InitParams {
    dmabuf_producer::DmaBufProducer::Config producer;
    std::string broker_ip;
    int port {1883};
    std::string frame_done_topic;
    std::string mqtt_client_id;
    int mqtt_qos {1};
  };

  FrameCacheService();
  ~FrameCacheService();

  FrameCacheService(const FrameCacheService&) = delete;
  FrameCacheService& operator=(const FrameCacheService&) = delete;

  bool Init(const InitParams& p, std::string* out_error);
  void Shutdown();

  dmabuf_producer::DmaBufProducer* Producer() { return producer_.get(); }

private:
  void OnFrameDonePayload(const std::string& topic, const std::string& payload);

  std::unique_ptr<dmabuf_producer::DmaBufProducer> producer_;
  std::unique_ptr<mqtt::MqttSubscriber> mqtt_sub_;
  bool inited_ {false};
};

}  // namespace multi_cam_app::frame_cache

#endif
