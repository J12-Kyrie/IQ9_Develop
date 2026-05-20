#include "MqttSubscriber.h"

#include <memory>
#include "infra/Logger.h"

namespace mqtt {

MqttSubscriber::MqttSubscriber(const std::string& brokerIp, int port, const std::string& topic)
    : m_brokerIp(brokerIp), m_port(port), m_defaultTopic(topic) {
}

MqttSubscriber::~MqttSubscriber() {
    shutdown();
}

bool MqttSubscriber::initialize() {
    utils::Logger::info("[MqttSubscriber] Initializing MQTT subscriber...");

    m_adapter = std::make_unique<adapters::MqttAdapter>(m_brokerIp, m_port);

    if (!m_adapter->connect("nvsci_rule_sub")) {
        utils::Logger::error("[MqttSubscriber] Failed to connect to MQTT broker");
        return false;
    }

    // Set up message callback
    m_adapter->setMessageCallback([this](const std::string& topic, const std::string& payload) {
        onMessage(topic, payload);
    });

    // Subscribe to default topic
    if (!m_defaultTopic.empty()) {
        subscribe(m_defaultTopic);
    }

    utils::Logger::info("[MqttSubscriber] MQTT subscriber initialized successfully");
    return true;
}

void MqttSubscriber::shutdown() {
    if (m_adapter) {
        m_adapter->disconnect();
        m_adapter.reset();
    }

    utils::Logger::info("[MqttSubscriber] MQTT subscriber shutdown complete");
}

bool MqttSubscriber::subscribe(const std::string& topic, int qos) {
    if (!m_adapter || !m_adapter->isConnected()) {
        utils::Logger::warn("[MqttSubscriber] Cannot subscribe: not connected");
        return false;
    }

    if (m_adapter->subscribe(topic, qos)) {
        utils::Logger::info("[MqttSubscriber] Subscribed to topic: " + topic);
        return true;
    }

    return false;
}

void MqttSubscriber::setMessageHandler(MessageHandler handler) {
    m_messageHandler = std::move(handler);
}

void MqttSubscriber::onMessage(const std::string& topic, const std::string& payload) {
    m_receivedCount.fetch_add(1);

    // If a handler is set, call it
    if (m_messageHandler) {
        try {
            m_messageHandler(topic, payload);
        } catch (const std::exception& e) {
            utils::Logger::error("[MqttSubscriber] Exception in message handler: " +
                               std::string(e.what()));
        }
    }
}

} // namespace mqtt
