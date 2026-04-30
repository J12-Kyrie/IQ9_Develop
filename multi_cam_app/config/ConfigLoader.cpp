#include "config/ConfigLoader.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <json-glib/json-glib.h>

namespace multi_cam_app::config {
namespace {

constexpr uint32_t kHardChannelLimit = 6U;

bool IsAbsolutePath(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  if ((path[0] == '/') || (path[0] == '\\')) {
    return true;
  }
  if ((path.size() >= 3U) &&
      std::isalpha(static_cast<unsigned char>(path[0])) &&
      (path[1] == ':') &&
      ((path[2] == '/') || (path[2] == '\\'))) {
    return true;
  }
  return false;
}

std::string NormalizePathSeparators(std::string path) {
  std::replace(path.begin(), path.end(), '\\', '/');
  return path;
}

std::string ResolvePath(const std::string& base_dir, const std::string& value) {
  if (value.empty()) {
    return value;
  }
  if (IsAbsolutePath(value)) {
    return NormalizePathSeparators(value);
  }
  return NormalizePathSeparators(
      (std::filesystem::path(base_dir) / value).lexically_normal().string());
}

bool GetStringMember(JsonObject* object, const char* key, std::string* out_value) {
  if ((object == nullptr) || (key == nullptr) || (out_value == nullptr)) {
    return false;
  }
  if (!json_object_has_member(object, key)) {
    return false;
  }
  JsonNode* node = json_object_get_member(object, key);
  if (!JSON_NODE_HOLDS_VALUE(node) ||
      (json_node_get_value_type(node) != G_TYPE_STRING)) {
    return false;
  }
  const gchar* value = json_node_get_string(node);
  if (value == nullptr) {
    return false;
  }
  *out_value = value;
  return true;
}

bool GetStringArrayMember(JsonObject* object,
                          const char* key,
                          std::vector<std::string>* out_values) {
  if ((object == nullptr) || (key == nullptr) || (out_values == nullptr)) {
    return false;
  }
  if (!json_object_has_member(object, key)) {
    return false;
  }
  JsonNode* node = json_object_get_member(object, key);
  if (!JSON_NODE_HOLDS_ARRAY(node)) {
    return false;
  }
  JsonArray* array = json_node_get_array(node);
  if (array == nullptr) {
    return false;
  }
  const guint length = json_array_get_length(array);
  out_values->clear();
  out_values->reserve(length);
  for (guint i = 0U; i < length; ++i) {
    const gchar* value = json_array_get_string_element(array, i);
    if (value == nullptr) {
      return false;
    }
    out_values->emplace_back(value);
  }
  return true;
}

std::string ErrorWithPrefix(const std::string& prefix, const std::string& details) {
  if (details.empty()) {
    return prefix;
  }
  return prefix + ": " + details;
}

std::string ToLowerAscii(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool IsSupportedMlConverterEngineNick(const std::string& nick) {
  return (nick == "fcv") || (nick == "c2d") || (nick == "gles") || (nick == "ocv") || (nick == "none");
}

}  // namespace

bool ConfigLoader::LoadFromFile(const std::string& config_file_path,
                                AppConfig* out_config,
                                std::string* out_error) {
  if (out_config == nullptr) {
    if (out_error != nullptr) {
      *out_error = "ConfigLoader received null out_config";
    }
    return false;
  }

  JsonParser* parser = json_parser_new();
  GError* error = nullptr;

  const gboolean loaded =
      json_parser_load_from_file(parser, config_file_path.c_str(), &error);
  if (!loaded) {
    if (out_error != nullptr) {
      const std::string details =
          (error != nullptr && error->message != nullptr) ? error->message : "";
      *out_error = ErrorWithPrefix("Failed to parse config json", details);
    }
    if (error != nullptr) {
      g_error_free(error);
    }
    g_object_unref(parser);
    return false;
  }

  JsonNode* root = json_parser_get_root(parser);
  if ((root == nullptr) || !JSON_NODE_HOLDS_OBJECT(root)) {
    if (out_error != nullptr) {
      *out_error = "Config root must be a JSON object";
    }
    g_object_unref(parser);
    return false;
  }

  JsonObject* root_obj = json_node_get_object(root);
  if (root_obj == nullptr) {
    if (out_error != nullptr) {
      *out_error = "Failed to read root object from config json";
    }
    g_object_unref(parser);
    return false;
  }

  AppConfig config {};
  const std::filesystem::path cfg_path =
      std::filesystem::path(config_file_path).lexically_normal();
  const std::string cfg_dir = cfg_path.parent_path().string();

  std::string tmp;
  if (!GetStringMember(root_obj, "model_path", &tmp) || tmp.empty()) {
    if (out_error != nullptr) {
      *out_error = "config.model_path is required and must be a string";
    }
    g_object_unref(parser);
    return false;
  }
  config.model_path = ResolvePath(cfg_dir, tmp);

  std::vector<std::string> videos;
  if (!GetStringArrayMember(root_obj, "videos_path", &videos) || videos.empty()) {
    if (out_error != nullptr) {
      *out_error = "config.videos_path is required and must be a non-empty string array";
    }
    g_object_unref(parser);
    return false;
  }
  for (const auto& value : videos) {
    if (value.empty()) {
      if (out_error != nullptr) {
        *out_error = "config.videos_path contains an empty path";
      }
      g_object_unref(parser);
      return false;
    }
    config.videos_path.push_back(ResolvePath(cfg_dir, value));
  }

  if (!GetStringMember(root_obj, "labels_path", &tmp) || tmp.empty()) {
    if (out_error != nullptr) {
      *out_error = "config.labels_path is required and must be a string";
    }
    g_object_unref(parser);
    return false;
  }
  config.labels_path = ResolvePath(cfg_dir, tmp);

  if (!GetStringMember(root_obj, "output_dir", &tmp) || tmp.empty()) {
    if (out_error != nullptr) {
      *out_error = "config.output_dir is required and must be a string";
    }
    g_object_unref(parser);
    return false;
  }
  config.output_dir = ResolvePath(cfg_dir, tmp);

  // log_dir (optional, defaults to output_dir/../log)
  if (GetStringMember(root_obj, "log_dir", &tmp) && !tmp.empty()) {
    config.log_dir = ResolvePath(cfg_dir, tmp);
  } else {
    config.log_dir = config.output_dir + "/../log";
  }

  if (GetStringMember(root_obj, "qnn_backend", &tmp) && !tmp.empty()) {
    config.qnn_backend = ResolvePath(cfg_dir, tmp);
  }

  if (GetStringMember(root_obj, "qnn_system", &tmp) && !tmp.empty()) {
    config.qnn_system = ResolvePath(cfg_dir, tmp);
  }

  if (GetStringMember(root_obj, "postprocess_module", &tmp) && !tmp.empty()) {
    config.postprocess_module = tmp;
  }

  if (json_object_has_member(root_obj, "qnn_backend_device_id")) {
    config.qnn_backend_device_id =
        static_cast<uint32_t>(json_object_get_int_member(root_obj, "qnn_backend_device_id"));
  }

  if (json_object_has_member(root_obj, "confidence")) {
    JsonNode* node = json_object_get_member(root_obj, "confidence");
    if (!JSON_NODE_HOLDS_VALUE(node)) {
      if (out_error != nullptr) {
        *out_error = "config.confidence must be a numeric value";
      }
      g_object_unref(parser);
      return false;
    }
    config.confidence = json_node_get_double(node);
    if ((config.confidence < 0.0) || (config.confidence > 100.0)) {
      if (out_error != nullptr) {
        *out_error = "config.confidence must be in range [0, 100]";
      }
      g_object_unref(parser);
      return false;
    }
  }

  if (json_object_has_member(root_obj, "max_channels")) {
    const gint value = json_object_get_int_member(root_obj, "max_channels");
    if (value <= 0) {
      if (out_error != nullptr) {
        *out_error = "config.max_channels must be greater than 0";
      }
      g_object_unref(parser);
      return false;
    }
    config.max_channels = static_cast<uint32_t>(value);
  }

  if (json_object_has_member(root_obj, "appsink_sync")) {
    config.appsink_sync = json_object_get_boolean_member(root_obj, "appsink_sync");
  }

  if (json_object_has_member(root_obj, "appsink_drop")) {
    config.appsink_drop = json_object_get_boolean_member(root_obj, "appsink_drop");
  }

  if (json_object_has_member(root_obj, "appsink_max_buffers")) {
    const gint value = json_object_get_int_member(root_obj, "appsink_max_buffers");
    if (value <= 0) {
      if (out_error != nullptr) {
        *out_error = "config.appsink_max_buffers must be greater than 0";
      }
      g_object_unref(parser);
      return false;
    }
    config.appsink_max_buffers = static_cast<uint32_t>(value);
  }

  std::vector<std::string> tensors;
  if (GetStringArrayMember(root_obj, "qnn_tensors", &tensors) && !tensors.empty()) {
    config.qnn_tensors = std::move(tensors);
  }

  if (json_object_has_member(root_obj, "qtimlvconverter_engine_order")) {
    std::vector<std::string> engines;
    if (!GetStringArrayMember(root_obj, "qtimlvconverter_engine_order", &engines) ||
        engines.empty()) {
      if (out_error != nullptr) {
        *out_error =
            "config.qtimlvconverter_engine_order must be a non-empty string array";
      }
      g_object_unref(parser);
      return false;
    }

    config.qtimlvconverter_engine_order.clear();
    config.qtimlvconverter_engine_order.reserve(engines.size());
    for (const auto& engine : engines) {
      if (engine.empty()) {
        if (out_error != nullptr) {
          *out_error = "config.qtimlvconverter_engine_order contains an empty value";
        }
        g_object_unref(parser);
        return false;
      }

      const std::string normalized = ToLowerAscii(engine);
      if (!IsSupportedMlConverterEngineNick(normalized)) {
        if (out_error != nullptr) {
          *out_error = "config.qtimlvconverter_engine_order contains invalid value '" +
                       engine +
                       "'. Allowed values: fcv, c2d, gles, none";
        }
        g_object_unref(parser);
        return false;
      }

      config.qtimlvconverter_engine_order.push_back(normalized);
    }
  }

  if (json_object_has_member(root_obj, "qtiobjtracker_parameters")) {
    if (!GetStringMember(root_obj, "qtiobjtracker_parameters", &tmp)) {
      if (out_error != nullptr) {
        *out_error = "config.qtiobjtracker_parameters must be a string";
      }
      g_object_unref(parser);
      return false;
    }
    config.qtiobjtracker_parameters = tmp;
  }

  // Parse optional "face" object
  if (json_object_has_member(root_obj, "face")) {
    JsonNode* face_node = json_object_get_member(root_obj, "face");
    if (face_node != nullptr && JSON_NODE_HOLDS_OBJECT(face_node)) {
      JsonObject* face_obj = json_node_get_object(face_node);
      if (face_obj != nullptr) {
        if (json_object_has_member(face_obj, "enabled")) {
          config.face_enabled = json_object_get_boolean_member(face_obj, "enabled");
        }
        if (json_object_has_member(face_obj, "channel_mask")) {
          const gint v = json_object_get_int_member(face_obj, "channel_mask");
          config.face_channel_mask = static_cast<int>(v);
        }
        if (json_object_has_member(face_obj, "face_interval_ms")) {
          config.face_interval_ms = static_cast<uint32_t>(
              json_object_get_int_member(face_obj, "face_interval_ms"));
        }
        if (GetStringMember(face_obj, "face_config_path", &tmp) && !tmp.empty()) {
          config.face_config_path = ResolvePath(cfg_dir, tmp);
        }
        if (GetStringMember(face_obj, "gallery_path", &tmp) && !tmp.empty()) {
          config.gallery_path = ResolvePath(cfg_dir, tmp);
        }
        if (json_object_has_member(face_obj, "gallery_threshold")) {
          config.gallery_threshold = static_cast<float>(
              json_object_get_double_member(face_obj, "gallery_threshold"));
        }
        if (json_object_has_member(face_obj, "gallery_min_face_size")) {
          const gint v = json_object_get_int_member(face_obj, "gallery_min_face_size");
          if (v > 0) {
            config.gallery_min_face_size = static_cast<uint32_t>(v);
          }
        }
        if (json_object_has_member(face_obj, "gallery_min_score")) {
          config.gallery_min_score = static_cast<float>(
              json_object_get_double_member(face_obj, "gallery_min_score"));
        }
      }
    }
  }

  // Parse optional "memtest" object
  if (json_object_has_member(root_obj, "memtest")) {
    JsonNode* mt_node = json_object_get_member(root_obj, "memtest");
    if (mt_node != nullptr && JSON_NODE_HOLDS_OBJECT(mt_node)) {
      JsonObject* mt_obj = json_node_get_object(mt_node);
      if (mt_obj != nullptr) {
        if (json_object_has_member(mt_obj, "loop_count")) {
          const gint v = json_object_get_int_member(mt_obj, "loop_count");
          if (v >= 0) {
            config.memtest_loop_count = static_cast<uint32_t>(v);
          }
        }
        if (json_object_has_member(mt_obj, "mem_sample_interval_s")) {
          const gint v = json_object_get_int_member(mt_obj, "mem_sample_interval_s");
          if (v > 0) {
            config.memtest_mem_sample_interval_s = static_cast<uint32_t>(v);
          }
        }
        if (GetStringMember(mt_obj, "log_path", &tmp) && !tmp.empty()) {
          config.memtest_log_path = ResolvePath(cfg_dir, tmp);
        }
      }
    }
  }

  // Parse optional "aggregator" object
  if (json_object_has_member(root_obj, "aggregator")) {
    JsonNode* agg_node = json_object_get_member(root_obj, "aggregator");
    if (agg_node != nullptr && JSON_NODE_HOLDS_OBJECT(agg_node)) {
      JsonObject* agg_obj = json_node_get_object(agg_node);
      if (agg_obj != nullptr) {
        if (json_object_has_member(agg_obj, "enabled")) {
          config.aggregator.enabled =
              json_object_get_boolean_member(agg_obj, "enabled");
        }
        if (json_object_has_member(agg_obj, "batch_size")) {
          const gint v = json_object_get_int_member(agg_obj, "batch_size");
          if (v > 0) {
            config.aggregator.batch_size = static_cast<uint32_t>(v);
          }
        }
        if (json_object_has_member(agg_obj, "queue_capacity")) {
          const gint v = json_object_get_int_member(agg_obj, "queue_capacity");
          if (v > 0) {
            config.aggregator.queue_capacity = static_cast<uint32_t>(v);
          }
        }
        if (json_object_has_member(agg_obj, "drain_timeout_ms")) {
          const gint v = json_object_get_int_member(agg_obj, "drain_timeout_ms");
          if (v > 0) {
            config.aggregator.drain_timeout_ms = static_cast<uint32_t>(v);
          }
        }
      }
    }
  }

  // Parse optional "mqtt" object
  if (json_object_has_member(root_obj, "mqtt")) {
    JsonNode* mqtt_node = json_object_get_member(root_obj, "mqtt");
    if (mqtt_node != nullptr && JSON_NODE_HOLDS_OBJECT(mqtt_node)) {
      JsonObject* mqtt_obj = json_node_get_object(mqtt_node);
      if (mqtt_obj != nullptr) {
        if (json_object_has_member(mqtt_obj, "enabled")) {
          config.mqtt.enabled =
              json_object_get_boolean_member(mqtt_obj, "enabled");
        }
        if (GetStringMember(mqtt_obj, "broker_ip", &tmp) && !tmp.empty()) {
          config.mqtt.broker_ip = tmp;
        }
        if (json_object_has_member(mqtt_obj, "port")) {
          const gint v = json_object_get_int_member(mqtt_obj, "port");
          if (v > 0 && v <= 65535) {
            config.mqtt.port = static_cast<int>(v);
          }
        }
        if (GetStringMember(mqtt_obj, "pub_topic", &tmp) && !tmp.empty()) {
          config.mqtt.pub_topic = tmp;
        }
        if (GetStringMember(mqtt_obj, "sub_topic", &tmp) && !tmp.empty()) {
          config.mqtt.sub_topic = tmp;
        }
        if (GetStringMember(mqtt_obj, "client_id", &tmp) && !tmp.empty()) {
          config.mqtt.client_id = tmp;
        }
        if (json_object_has_member(mqtt_obj, "qos")) {
          const gint v = json_object_get_int_member(mqtt_obj, "qos");
          if (v >= 0 && v <= 2) {
            config.mqtt.qos = static_cast<int>(v);
          }
        }
      }
    }
  }

  // Parse optional "frame_cache" object
  if (json_object_has_member(root_obj, "frame_cache")) {
    JsonNode* fc_node = json_object_get_member(root_obj, "frame_cache");
    if (fc_node != nullptr && JSON_NODE_HOLDS_OBJECT(fc_node)) {
      JsonObject* fc_obj = json_node_get_object(fc_node);
      if (fc_obj != nullptr) {
        if (json_object_has_member(fc_obj, "enabled")) {
          config.frame_cache.enabled =
              json_object_get_boolean_member(fc_obj, "enabled");
        }
        if (GetStringMember(fc_obj, "socket_path", &tmp) && !tmp.empty()) {
          config.frame_cache.socket_path = tmp;
        }
        if (GetStringMember(fc_obj, "heap_path", &tmp) && !tmp.empty()) {
          config.frame_cache.heap_path = tmp;
        }
        if (json_object_has_member(fc_obj, "slot_count")) {
          const gint v = json_object_get_int_member(fc_obj, "slot_count");
          if (v > 0 && v <= 32) {
            config.frame_cache.slot_count = static_cast<int>(v);
          }
        }
        if (json_object_has_member(fc_obj, "width")) {
          const gint v = json_object_get_int_member(fc_obj, "width");
          if (v > 0) {
            config.frame_cache.width = static_cast<int>(v);
          }
        }
        if (json_object_has_member(fc_obj, "height")) {
          const gint v = json_object_get_int_member(fc_obj, "height");
          if (v > 0) {
            config.frame_cache.height = static_cast<int>(v);
          }
        }
        if (json_object_has_member(fc_obj, "relay_mode")) {
          config.frame_cache.relay_mode =
              json_object_get_boolean_member(fc_obj, "relay_mode");
        }
        if (GetStringMember(fc_obj, "frame_done_topic", &tmp) && !tmp.empty()) {
          config.frame_cache.frame_done_topic = tmp;
        }
      }
    }
  }

  if (config.frame_cache.enabled) {
    if (!config.mqtt.enabled) {
      if (out_error != nullptr) {
        *out_error = "frame_cache.enabled requires mqtt.enabled (for frame_done subscription)";
      }
      g_object_unref(parser);
      return false;
    }
    if (config.frame_cache.frame_done_topic.empty()) {
      if (out_error != nullptr) {
        *out_error = "frame_cache.enabled requires a non-empty frame_cache.frame_done_topic";
      }
      g_object_unref(parser);
      return false;
    }
  }

  // Parse optional "msgagg" object
  if (json_object_has_member(root_obj, "msgagg")) {
    JsonNode* ma_node = json_object_get_member(root_obj, "msgagg");
    if (ma_node != nullptr && JSON_NODE_HOLDS_OBJECT(ma_node)) {
      JsonObject* ma_obj = json_node_get_object(ma_node);
      if (ma_obj != nullptr) {
        if (json_object_has_member(ma_obj, "enabled")) {
          config.msgagg.enabled =
              json_object_get_boolean_member(ma_obj, "enabled");
        }
        if (json_object_has_member(ma_obj, "timeout_ms")) {
          const gint v = json_object_get_int_member(ma_obj, "timeout_ms");
          if (v > 0) {
            config.msgagg.timeout_ms = static_cast<uint32_t>(v);
          }
        }
        if (GetStringMember(ma_obj, "scene_jsonl_dir", &tmp) && !tmp.empty()) {
          config.msgagg.scene_jsonl_dir = ResolvePath(cfg_dir, tmp);
        }
      }
    }
  }

  // Parse optional "latency_test" object
  if (json_object_has_member(root_obj, "latency_test")) {
    JsonNode* lt_node = json_object_get_member(root_obj, "latency_test");
    if (lt_node != nullptr && JSON_NODE_HOLDS_OBJECT(lt_node)) {
      JsonObject* lt_obj = json_node_get_object(lt_node);
      if (lt_obj != nullptr) {
        if (json_object_has_member(lt_obj, "enabled")) {
          config.latency_test.enabled =
              json_object_get_boolean_member(lt_obj, "enabled");
        }
        if (GetStringMember(lt_obj, "output_dir", &tmp) && !tmp.empty()) {
          config.latency_test.output_dir = ResolvePath(cfg_dir, tmp);
        }
        if (json_object_has_member(lt_obj, "sample_every_n")) {
          const gint v = json_object_get_int_member(lt_obj, "sample_every_n");
          if (v > 0) {
            config.latency_test.sample_every_n = static_cast<uint32_t>(v);
          }
        }
        if (json_object_has_member(lt_obj, "flush_per_sample")) {
          config.latency_test.flush_per_sample =
              json_object_get_boolean_member(lt_obj, "flush_per_sample");
        }
      }
    }
  }

  config.max_channels = std::min(config.max_channels, kHardChannelLimit);
  if (config.max_channels == 0U) {
    config.max_channels = 1U;
  }
  if (config.videos_path.size() > config.max_channels) {
    config.videos_path.resize(config.max_channels);
  }

  *out_config = std::move(config);
  g_object_unref(parser);
  return true;
}

}  // namespace multi_cam_app::config
