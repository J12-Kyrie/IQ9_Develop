#pragma once

#include "MqttAdapter.h"
#include <atomic>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>

namespace mqtt {

class MqttPublisher {
public:
    MqttPublisher(const std::string& brokerIp, int port, const std::string& topic);
    ~MqttPublisher();

    bool initialize();
    void shutdown();

    bool publishFrame(const nlohmann::json& frame);

    // Statistics
    bool isConnected() const { return m_adapter && m_adapter->isConnected(); }
    uint64_t getPublishedCount() const { return m_publishedCount.load(); }

private:
    std::string m_brokerIp;
    int m_port;
    std::string m_topic;
    std::unique_ptr<adapters::MqttAdapter> m_adapter;
    std::atomic<uint64_t> m_publishedCount{0};
};

} // namespace mqtt
