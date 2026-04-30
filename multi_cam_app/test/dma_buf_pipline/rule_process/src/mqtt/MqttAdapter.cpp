#include "MqttAdapter.h"
#include "infra/Logger.h"

namespace adapters {

MqttAdapter::MqttAdapter(const std::string& brokerIp, int port)
    : m_brokerIp(brokerIp)
    , m_port(port)
    , m_client(nullptr)
    , m_connected(false) {
    m_brokerUrl = "tcp://" + brokerIp + ":" + std::to_string(port);
}

MqttAdapter::~MqttAdapter() {
    disconnect();
}

bool MqttAdapter::connect(const std::string& clientId) {
    std::lock_guard<std::mutex> lock(m_mutex);

    int rc;
    rc = MQTTClient_create(&m_client, m_brokerUrl.c_str(), clientId.c_str(),
                          MQTTCLIENT_PERSISTENCE_NONE, nullptr);
    if (rc != MQTTCLIENT_SUCCESS) {
        utils::Logger::error("[MqttAdapter] Failed to create MQTT client: " +
                           std::string(MQTTClient_strerror(rc)));
        return false;
    }

    // Set callbacks
    MQTTClient_setCallbacks(m_client, this, onConnectionLost, onMessage, nullptr);

    // Connect options
    MQTTClient_connectOptions connOpts = MQTTClient_connectOptions_initializer;
    rc = MQTTClient_connect(m_client, &connOpts);

    if (rc != MQTTCLIENT_SUCCESS) {
        utils::Logger::error("[MqttAdapter] Failed to connect to MQTT broker: " +
                           m_brokerUrl);
        MQTTClient_destroy(&m_client);
        m_client = nullptr;
        return false;
    }

    m_connected.store(true);
    utils::Logger::info("[MqttAdapter] Connected to MQTT broker: " + m_brokerUrl);

    return true;
}

void MqttAdapter::disconnect() {
    if (!m_connected.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_client) {
        MQTTClient_disconnect(m_client, 0);
        MQTTClient_destroy(&m_client);
        m_client = nullptr;
    }

    m_connected.store(false);
    utils::Logger::info("[MqttAdapter] Disconnected from MQTT broker: " + m_brokerUrl);
}

bool MqttAdapter::subscribe(const std::string& topic, int qos) {
    if (!m_connected.load()) {
        utils::Logger::warn("[MqttAdapter] Cannot subscribe: not connected");
        return false;
    }

    int rc;
    rc = MQTTClient_subscribe(m_client, topic.c_str(), qos);

    if (rc != MQTTCLIENT_SUCCESS) {
        utils::Logger::error("[MqttAdapter] Failed to subscribe to topic: " + topic);
        return false;
    }

    utils::Logger::info("[MqttAdapter] Subscribed to topic: " + topic);
    return true;
}

bool MqttAdapter::publish(const std::string& topic, const std::string& payload, int qos) {
    return publish(topic, payload, qos, false);
}

bool MqttAdapter::publish(const std::string& topic, const std::string& payload, int qos, bool retained) {
    if (!m_connected.load()) {
        utils::Logger::warn("[MqttAdapter] Cannot publish: not connected");
        return false;
    }

    int rc;
    rc = MQTTClient_publish(m_client, topic.c_str(), static_cast<int>(payload.length()),
                           const_cast<char*>(payload.c_str()), qos, retained, nullptr);

    if (rc != MQTTCLIENT_SUCCESS) {
        utils::Logger::error("[MqttAdapter] Failed to publish message to topic: " + topic);
        return false;
    }

    m_publishedMessages.fetch_add(1);
    return true;
}

void MqttAdapter::setMessageCallback(MessageCallback callback) {
    m_callback = std::move(callback);
}

int MqttAdapter::onMessage(void* context, char* topicName, int topicLen, MQTTClient_message* message) {
    // Paho allocates both topicName and message; we must free them before returning.
    MqttAdapter* adapter = static_cast<MqttAdapter*>(context);
    if (adapter && topicName && message) {
        std::string topic = (topicLen > 0) ? std::string(topicName, topicLen)
                                           : std::string(topicName);
        std::string payload;
        if (message->payload && message->payloadlen > 0) {
            payload.assign(static_cast<const char*>(message->payload), message->payloadlen);
        }
        adapter->onMessageReceived(topic, payload);
    }

    MQTTClient_freeMessage(&message);
    MQTTClient_free(topicName);
    return 1;
}

void MqttAdapter::onConnectionLost(void* context, char* cause) {
    MqttAdapter* adapter = static_cast<MqttAdapter*>(context);
    if (adapter) {
        adapter->m_connected.store(false);
        utils::Logger::warn("[MqttAdapter] Connection lost: " + std::string(cause ? cause : "unknown"));
    }
}

void MqttAdapter::onMessageReceived(const std::string& topic, const std::string& payload) {
    m_receivedMessages.fetch_add(1);

    if (m_callback) {
        try {
            m_callback(topic, payload);
        } catch (const std::exception& e) {
            utils::Logger::error("[MqttAdapter] Exception in message callback: " + 
                               std::string(e.what()));
        }
    }
}

}  // namespace adapters
