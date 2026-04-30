#pragma once

#include "MqttAdapter.h"
#include <cstdint>
#include <string>
#include <memory>
#include <functional>
#include <atomic>

namespace mqtt {

class MqttSubscriber {
public:
    using MessageHandler = std::function<void(const std::string& topic, const std::string& payload)>;

    MqttSubscriber(const std::string& brokerIp, int port, const std::string& topic);
    ~MqttSubscriber();

    bool initialize();
    void shutdown();

    // Subscribe to topics
    bool subscribe(const std::string& topic, int qos = 1);

    // Register message handler
    void setMessageHandler(MessageHandler handler);

    // Statistics
    bool isConnected() const { return m_adapter && m_adapter->isConnected(); }
    uint64_t getReceivedCount() const { return m_receivedCount.load(); }

private:
    void onMessage(const std::string& topic, const std::string& payload);

    std::string m_brokerIp;
    int m_port;
    std::string m_defaultTopic;
    std::unique_ptr<adapters::MqttAdapter> m_adapter;

    MessageHandler m_messageHandler;

    std::atomic<uint64_t> m_receivedCount{0};
};

} // namespace mqtt
