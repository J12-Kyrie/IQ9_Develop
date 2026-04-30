#include "MqttPublisher.h"

#include <memory>
#include "infra/Logger.h"

namespace mqtt {

MqttPublisher::MqttPublisher(const std::string& brokerIp, int port, const std::string& topic)
    : m_brokerIp(brokerIp), m_port(port), m_topic(topic) {
}

MqttPublisher::~MqttPublisher() {
    shutdown();
}

bool MqttPublisher::initialize() {
    utils::Logger::info("[MqttPublisher] Initializing MQTT publisher...");
    m_adapter = std::make_unique<adapters::MqttAdapter>(m_brokerIp, m_port);
    if (!m_adapter->connect("nvsci_frame_done_pub")) {
        utils::Logger::error("[MqttPublisher] Failed to connect to MQTT broker");
        return false;
    }
    utils::Logger::info("[MqttPublisher] MQTT publisher initialized successfully");
    return true;
}

void MqttPublisher::shutdown() {
    if (m_adapter) {
        m_adapter->disconnect();
        m_adapter.reset();
        utils::Logger::info("[MqttPublisher] MQTT publisher shutdown complete");
    }
}

bool MqttPublisher::publishFrame(const nlohmann::json& frame) {
    if (!m_adapter || !m_adapter->isConnected()) {
        utils::Logger::warn("[MqttPublisher] Cannot publish: not connected");
        return false;
    }
    std::string payload = frame.dump();
    if (m_adapter->publish(m_topic, payload, 1, false)) {
        m_publishedCount.fetch_add(1);
        return true;
    }
    return false;
}

} // namespace mqtt
