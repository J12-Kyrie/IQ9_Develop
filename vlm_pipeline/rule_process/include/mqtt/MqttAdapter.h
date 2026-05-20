#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

#include <MQTTClient.h>

namespace adapters {

/**
 * @brief MQTT Adapter - C++ wrapper for Paho MQTT C library
 *
 * Responsibilities:
 * - Manage MQTT client lifecycle (connect/disconnect)
 * - Message subscription and callback dispatch
 * - Message publishing (for publisher service)
 */
class MqttAdapter {
public:
    using MessageCallback = std::function<void(const std::string& topic, const std::string& payload)>;

    /**
     * @brief Constructor
     * @param brokerIp MQTT broker IP address
     * @param port MQTT broker port
     */
    MqttAdapter(const std::string& brokerIp, int port);
    
    /**
     * @brief Destructor - automatically disconnects
     */
    ~MqttAdapter();

    // Disable copy
    MqttAdapter(const MqttAdapter&) = delete;
    MqttAdapter& operator=(const MqttAdapter&) = delete;

    /// Connect to MQTT broker
    bool connect(const std::string& clientId);
    
    /// Disconnect from MQTT broker
    void disconnect();
    
    /// Check connection status
    bool isConnected() const { return m_connected.load(); }

    /// Subscribe to MQTT topic
    bool subscribe(const std::string& topic, int qos = 1);

    /// Publish message to MQTT topic
    bool publish(const std::string& topic, const std::string& payload, int qos = 1);
    bool publish(const std::string& topic, const std::string& payload, int qos, bool retained);

    /// Set message arrival callback
    void setMessageCallback(MessageCallback callback);

private:
    // Paho MQTT callbacks (static for C interop)
    static int onMessage(void* context, char* topicName, int topicLen, MQTTClient_message* message);
    static void onConnectionLost(void* context, char* cause);

    // Internal message handler
    void onMessageReceived(const std::string& topic, const std::string& payload);

    // Member variables
    std::string m_brokerIp;
    std::string m_brokerUrl;
    int m_port;

    MessageCallback m_callback;
    MQTTClient m_client{nullptr};

    std::atomic<bool> m_connected{false};
    std::mutex m_mutex;  // Protects connect/disconnect

    // Statistics
    std::atomic<uint64_t> m_publishedMessages{0};
    std::atomic<uint64_t> m_receivedMessages{0};
};

}  // namespace adapters
