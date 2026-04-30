/*
 * IQ9 rule_process: MQTT scene_update -> RuleEngine -> UDS relay (alerts) + MQTT frame_done (non-alerts)
 */
#include "adapters/EventAdapter.h"
#include "core/AlertManager.h"
#include "core/RuleEngine.h"
#include "core/DataTypes.h"
#include "infra/Logger.h"
#include "channel_protocol.hpp"

#ifdef ENABLE_MQTT
#include "mqtt/MqttSubscriber.h"
#include "mqtt/MqttPublisher.h"
#endif

#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr const char* kTag = "RuleProcess";

struct EngineConfig {
  core::AlertManagerParams alert_manager_params;
  std::vector<core::RuleConfig> rule_configs;
};

struct AppConfig {
  std::string mqtt_broker;
  int mqtt_port = 1883;
  std::string scene_topic = "iq9/scene_update";
  std::string frame_done_topic = "iq9/frame_done";
  std::string relay_socket_path = "/tmp/iq9_rule_relay.sock";
  std::string rules_config_path;
  std::string scene_debug_jsonl;
  bool log_scene_timing = false;
};

EngineConfig loadRulesJson(const std::string& path) {
  EngineConfig cfg_bundle;
  std::ifstream file(path.c_str());
  if (!file.is_open()) {
    utils::Logger::error(kTag, "Cannot open rules: " + path);
    return cfg_bundle;
  }
  nlohmann::json root;
  try {
    file >> root;
  } catch (const std::exception& e) {
    utils::Logger::error(kTag, std::string("rules JSON: ") + e.what());
    return cfg_bundle;
  }
  if (root.contains("alert_manager") && root["alert_manager"].is_object()) {
    const auto& am = root["alert_manager"];
    if (am.contains("cooldown_seconds") && am["cooldown_seconds"].is_number_integer()) {
      cfg_bundle.alert_manager_params.cooldown_seconds = am["cooldown_seconds"].get<int>();
    }
    if (am.contains("max_history_size") && am["max_history_size"].is_number_integer()) {
      cfg_bundle.alert_manager_params.max_history_size =
          static_cast<size_t>(am["max_history_size"].get<int>());
    }
    if (am.contains("cleanup_threshold") && am["cleanup_threshold"].is_number_integer()) {
      cfg_bundle.alert_manager_params.cleanup_threshold =
          static_cast<size_t>(am["cleanup_threshold"].get<int>());
    }
    if (am.contains("cleanup_ratio") && am["cleanup_ratio"].is_number()) {
      cfg_bundle.alert_manager_params.cleanup_ratio = am["cleanup_ratio"].get<double>();
    }
  }
  if (!root.contains("rules") || !root["rules"].is_array()) {
    utils::Logger::error(kTag, "rules: missing rules array");
    return cfg_bundle;
  }
  for (const auto& rule : root["rules"]) {
    if (!rule.contains("name") || !rule.contains("priority") || !rule.contains("target_sources") ||
        !rule.contains("target_classes")) {
      utils::Logger::warn(kTag, "skip incomplete rule");
      continue;
    }
    core::RuleConfig cfg;
    cfg.name_ = rule["name"].get<std::string>();
    cfg.priority_ = rule["priority"].get<int>();
    cfg.target_sources_ = rule["target_sources"].get<std::vector<std::string>>();
    cfg.target_classes_ = rule["target_classes"].get<std::vector<std::string>>();
    if (rule.contains("params") && rule["params"].is_object()) {
      for (auto it = rule["params"].begin(); it != rule["params"].end(); ++it) {
        if (it.value().is_number()) {
          cfg.params_[it.key()] = it.value().get<float>();
        } else if (it.value().is_boolean()) {
          cfg.params_[it.key()] = it.value().get<bool>() ? 1.0f : 0.0f;
        } else if (it.value().is_string()) {
          cfg.str_params_[it.key()] = it.value().get<std::string>();
        }
      }
    }
    cfg_bundle.rule_configs.push_back(cfg);
  }
  return cfg_bundle;
}

core::Status loadAppConfig(const std::string& path, AppConfig& out) {
  std::ifstream f(path);
  if (!f) {
    return core::Status::Error("Cannot open config: " + path);
  }
  nlohmann::json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    return core::Status::Error(std::string("config JSON: ") + e.what());
  }
  if (j.contains("mqtt")) {
    const auto& m = j["mqtt"];
    if (m.contains("broker_ip")) out.mqtt_broker = m["broker_ip"].get<std::string>();
    if (m.contains("port")) out.mqtt_port = m["port"].get<int>();
    if (m.contains("scene_update_topic")) out.scene_topic = m["scene_update_topic"].get<std::string>();
    if (m.contains("frame_done_topic")) out.frame_done_topic = m["frame_done_topic"].get<std::string>();
  }
  if (j.contains("relay_socket") && j["relay_socket"].is_string()) {
    out.relay_socket_path = j["relay_socket"].get<std::string>();
  }
  if (!j.contains("rules_config") || !j["rules_config"].is_string()) {
    return core::Status::Error("rules_config path required");
  }
  out.rules_config_path = j["rules_config"].get<std::string>();
  if (j.contains("scene_debug_jsonl") && j["scene_debug_jsonl"].is_string()) {
    out.scene_debug_jsonl = j["scene_debug_jsonl"].get<std::string>();
  }
  if (j.contains("log_scene_timing") && j["log_scene_timing"].is_boolean()) {
    out.log_scene_timing = j["log_scene_timing"].get<bool>();
  }
  return core::Status::Ok();
}

static void appendScenePayloadJsonl(std::mutex* fileMu, std::string const& path,
                                    std::string const& payload) {
  if (path.empty()) return;
  std::lock_guard<std::mutex> lk(*fileMu);
  std::ofstream out(path.c_str(), std::ios::app | std::ios::binary);
  if (!out) {
    static bool once = false;
    if (!once) {
      utils::Logger::warn(kTag, "scene_debug_jsonl: cannot open for append: " + path);
      once = true;
    }
    return;
  }
  out << payload << '\n';
  out.flush();
}

/// When several rules emit alerts for the same source in one frame, pick the one with smallest priority_.
core::Alert selectAlertForVqaMeta(const std::vector<core::Alert>& alerts) {
  return *std::min_element(alerts.begin(), alerts.end(),
      [](const core::Alert& a, const core::Alert& b) { return a.priority_ < b.priority_; });
}

/// Wall time in milliseconds with 3 fractional digits.
inline std::string fmtMs3(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
  const int64_t us =
      std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3f", static_cast<double>(us) / 1000.0);
  return buf;
}

bool processScenePayload(
    const std::string& payload,
    const AppConfig& ac,
    core::RuleEngine& ruleEngine,
    std::mutex& relayMutex,
    std::mutex& sceneJsonlMutex,
    int& relayFd,
    const std::atomic<bool>* relay_main_ready_opt,
    std::set<int>* outVqaSources,
    std::vector<core::EventMeta>* outMetas)
{
  // qtimsgpub with json=TRUE produces an envelope with UNESCAPED inner JSON:
  //   {"Topic":"...","Message":{"MessageInGstBuffer":{"contents":"<INNER>"}}}
  // The inner quotes are not escaped, making the envelope invalid JSON.
  // Extract the inner scene_update by locating its head and counting braces.
  std::string inner_payload;
  const char* head = "{\"type\":\"scene_update\"";
  size_t start = payload.find(head);
  if (start != std::string::npos) {
    int depth = 0;
    size_t end = start;
    for (size_t i = start; i < payload.size(); ++i) {
      if (payload[i] == '{') ++depth;
      else if (payload[i] == '}') {
        --depth;
        if (depth == 0) { end = i + 1; break; }
      }
    }
    if (end > start) {
      inner_payload = payload.substr(start, end - start);
    }
  }

  nlohmann::json root = nlohmann::json::parse(
      inner_payload.empty() ? payload : inner_payload, nullptr, false);
  if (root.is_discarded()) {
    std::string preview = payload.size() > 120 ? payload.substr(0, 120) + "..." : payload;
    utils::Logger::warn(kTag, std::string("payload not JSON [") + std::to_string(payload.size()) + "B]: " + preview);
    return false;
  }
  if (root.value("type", "") != "scene_update") {
    return false;
  }

  std::string scene_json = inner_payload.empty() ? payload : inner_payload;
  adapters::EventAdapter adapter;
  std::vector<core::EventMeta> metas = adapter.adaptSceneUpdate(scene_json);
  if (metas.empty()) {
    utils::Logger::warn(kTag, "no sources after adapt");
    return false;
  }

  appendScenePayloadJsonl(&sceneJsonlMutex, ac.scene_debug_jsonl, payload);

  std::set<int> vqaSources;
  for (const auto& meta : metas) {
    std::vector<core::Alert> alerts = ruleEngine.processEvent(meta);
    if (alerts.empty()) continue;
    core::Alert const chosen = selectAlertForVqaMeta(alerts);

    IQ9_Task_Packet pkt{};
    pkt.magic = IQ9_TASK_MAGIC;
    pkt.slot_index = meta.slot_index_;
    pkt.width = meta.width_;
    pkt.height = meta.height_;
    pkt.channels = meta.channels_;
    pkt.data_type = PIXEL_FMT_RGB24;
    pkt.timestamp_ns = meta.timestamp_ns_;

    std::lock_guard<std::mutex> lock(relayMutex);
    if (relayFd < 0) {
      if (relay_main_ready_opt && !relay_main_ready_opt->load()) {
        continue; // wait for main-thread connectUds; do not steal consumer's only accept() slot
      }
      std::string err;
      if (!iq9::channel::connectUds(ac.relay_socket_path.c_str(), &relayFd, &err)) {
        utils::Logger::error(kTag, "relay reconnect failed: " + err);
        continue;
      }
      utils::Logger::info(kTag, "relay (re)connected");
    }
    if (!iq9::channel::sendAll(relayFd, &pkt, sizeof(pkt))) {
      utils::Logger::error(kTag, "UDS send relay failed, closing fd for reconnect");
      close(relayFd);
      relayFd = -1;
      continue;
    }
    vqaSources.insert(meta.source_id_num_);
    utils::Logger::info(kTag, "relay sent source=" + std::to_string(meta.source_id_num_) +
                        " slot=" + std::to_string(meta.slot_index_) + " rule=" + chosen.rule_name_ +
                        " priority=" + std::to_string(chosen.priority_));
  }

  if (outVqaSources) *outVqaSources = vqaSources;
  if (outMetas) *outMetas = metas;
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << (argc > 0 ? argv[0] : "rule_process") << " <rule_process_config.json>\n";
    return 1;
  }

  AppConfig ac;
  auto st = loadAppConfig(argv[1], ac);
  if (!st.ok()) {
    utils::Logger::error(kTag, st.message_);
    return 1;
  }

  EngineConfig ec = loadRulesJson(ac.rules_config_path);
  if (ec.rule_configs.empty()) {
    utils::Logger::error(kTag, "No rules loaded");
    return 1;
  }

  auto ruleEngine = std::make_shared<core::RuleEngine>(ec.rule_configs, ec.alert_manager_params);

  int relayFd = -1;
  std::atomic<bool> relayMainReady{false};
  std::mutex relayMutex;
  std::mutex sceneJsonlMutex;

  if (!ac.scene_debug_jsonl.empty()) {
    utils::Logger::info(kTag, "scene_debug_jsonl=" + ac.scene_debug_jsonl);
  }
  if (ac.log_scene_timing) {
    utils::Logger::info(kTag, "log_scene_timing enabled");
  }

#ifdef ENABLE_MQTT
  // Initialize MQTT BEFORE connecting to relay UDS.
  // The consumer blocks on relay accept(); connecting to relay unblocks it,
  // which triggers the producer to start publishing scene_updates.
  // MQTT must be subscribed before that happens.
  mqtt::MqttPublisher publisher(ac.mqtt_broker, ac.mqtt_port, ac.frame_done_topic);
  if (!publisher.initialize()) {
    return 1;
  }

  mqtt::MqttSubscriber subscriber(ac.mqtt_broker, ac.mqtt_port, ac.scene_topic);
  subscriber.setMessageHandler([&](const std::string& /*topic*/, const std::string& payload) {
    using clock = std::chrono::steady_clock;
    clock::time_point const t0 = clock::now();

    std::set<int> vqaSources;
    std::vector<core::EventMeta> metas;
    if (!processScenePayload(payload, ac, *ruleEngine, relayMutex, sceneJsonlMutex,
                             relayFd, &relayMainReady, &vqaSources, &metas)) {
      return;
    }

    clock::time_point const t2 = clock::now();

    nlohmann::json done;
    done["type"] = "frame_done";
    nlohmann::json rel = nlohmann::json::array();
    for (const auto& meta : metas) {
      if (vqaSources.count(meta.source_id_num_)) continue;
      nlohmann::json e;
      e["slot_index"] = meta.slot_index_;
      e["source_id"] = meta.source_id_num_;
      rel.push_back(e);
    }
    done["release_entries"] = rel;
    bool const frameDoneOk = publisher.publishFrame(done);
    clock::time_point const t3 = clock::now();

    if (ac.log_scene_timing) {
      utils::Logger::info(kTag, std::string("timing total_ms=") + fmtMs3(t0, t3) +
                          " rules_relay_ms=" + fmtMs3(t0, t2) +
                          " frame_done_ms=" + fmtMs3(t2, t3) +
                          " relay_sent=" + std::to_string(vqaSources.size()) +
                          " release_n=" + std::to_string(rel.size()) +
                          " frame_done_ok=" + (frameDoneOk ? "1" : "0"));
    }

    if (!frameDoneOk) {
      utils::Logger::error(kTag, "publish frame_done failed");
    }
  });

  if (!subscriber.initialize()) {
    publisher.shutdown();
    return 1;
  }

  // Now connect to relay UDS. This unblocks the consumer's accept(),
  // which then connects to the producer and starts frame processing.
  std::string connErr;
  if (!iq9::channel::connectUds(ac.relay_socket_path.c_str(), &relayFd, &connErr)) {
    utils::Logger::error(kTag, "relay UDS: " + connErr);
    subscriber.shutdown();
    publisher.shutdown();
    return 1;
  }
  relayMainReady.store(true);
  utils::Logger::info(kTag, "relay connected: " + ac.relay_socket_path);

  utils::Logger::info(kTag, "listening MQTT, Ctrl+C to stop");
  while (true) {
    sleep(3600);
  }

#else
  {
    std::string connErr;
    if (!iq9::channel::connectUds(ac.relay_socket_path.c_str(), &relayFd, &connErr)) {
      utils::Logger::error(kTag, "relay UDS: " + connErr);
      return 1;
    }
    utils::Logger::info(kTag, "relay connected: " + ac.relay_socket_path);
  }

  utils::Logger::info(kTag, "MQTT disabled — reading scene payloads from " +
                      (ac.scene_debug_jsonl.empty() ? "stdin" : ac.scene_debug_jsonl));

  std::istream* input = &std::cin;
  std::ifstream jsonlFile;
  if (!ac.scene_debug_jsonl.empty()) {
    jsonlFile.open(ac.scene_debug_jsonl);
    if (!jsonlFile.is_open()) {
      utils::Logger::error(kTag, "Cannot open scene_debug_jsonl: " + ac.scene_debug_jsonl);
      close(relayFd);
      return 1;
    }
    input = &jsonlFile;
  }

  std::string line;
  while (std::getline(*input, line)) {
    if (line.empty()) continue;
    processScenePayload(line, ac, *ruleEngine, relayMutex, sceneJsonlMutex,
                        relayFd, nullptr, nullptr, nullptr, nullptr);
  }

  utils::Logger::info(kTag, "MQTT fallback: finished processing input");
  if (relayFd >= 0) close(relayFd);
  return 0;
#endif
}
