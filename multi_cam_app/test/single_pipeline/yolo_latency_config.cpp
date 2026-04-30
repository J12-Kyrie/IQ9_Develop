#include "yolo_latency_config.hpp"

#include <cstdint>
#include <filesystem>

#include <glib.h>
#include <json-glib/json-glib.h>

namespace single_pipeline {
namespace {

bool GetStringMember(JsonObject* object, const char* key, std::string* out_value) {
  if ((object == nullptr) || (key == nullptr) || (out_value == nullptr)) {
    return false;
  }
  if (!json_object_has_member(object, key)) {
    return false;
  }
  JsonNode* node = json_object_get_member(object, key);
  if (!JSON_NODE_HOLDS_VALUE(node) || (json_node_get_value_type(node) != G_TYPE_STRING)) {
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

bool GetUIntMember(JsonObject* object, const char* key, uint32_t* out_value) {
  if ((object == nullptr) || (key == nullptr) || (out_value == nullptr)) {
    return false;
  }
  if (!json_object_has_member(object, key)) {
    return false;
  }
  JsonNode* node = json_object_get_member(object, key);
  const GType t = json_node_get_value_type(node);
  if (t == G_TYPE_INT64) {
    const gint64 v = json_node_get_int(node);
    if (v < 0) {
      return false;
    }
    *out_value = static_cast<uint32_t>(v);
    return true;
  }
  if (t == G_TYPE_INT) {
    const gint64 v = static_cast<gint64>(json_node_get_int(node));
    if (v < 0) {
      return false;
    }
    *out_value = static_cast<uint32_t>(v);
    return true;
  }
  if (t == G_TYPE_DOUBLE) {
    const gdouble d = json_node_get_double(node);
    if (d < 0.0) {
      return false;
    }
    *out_value = static_cast<uint32_t>(d);
    return true;
  }
  return false;
}

bool GetBoolMember(JsonObject* object, const char* key, bool* out_value) {
  if ((object == nullptr) || (key == nullptr) || (out_value == nullptr)) {
    return false;
  }
  if (!json_object_has_member(object, key)) {
    return false;
  }
  JsonNode* node = json_object_get_member(object, key);
  if (json_node_get_value_type(node) != G_TYPE_BOOLEAN) {
    return false;
  }
  *out_value = json_node_get_boolean(node) != FALSE;
  return true;
}

}  // namespace

bool LoadYoloQnnLatencyConfigFromFile(const std::string& path,
                                      YoloQnnLatencyConfig* out_config,
                                      std::string* out_error) {
  if (out_config == nullptr) {
    if (out_error != nullptr) {
      *out_error = "out_config is null";
    }
    return false;
  }

  GError* gerr = nullptr;
  {
    const std::filesystem::path fs_path(path);
    if (!std::filesystem::exists(fs_path)) {
      if (out_error != nullptr) {
        *out_error = "Config file not found: " + path;
      }
      return false;
    }
  }

  gchar* contents = nullptr;
  gsize length = 0U;
  if (!g_file_get_contents(path.c_str(), &contents, &length, &gerr)) {
    if (out_error != nullptr) {
      *out_error = std::string("Failed to read config: ") +
          ((gerr != nullptr) ? gerr->message : path);
    }
    if (gerr != nullptr) {
      g_error_free(gerr);
    }
    return false;
  }
  std::string json_text(contents, length);
  g_free(contents);

  JsonParser* parser = json_parser_new();
  if (!json_parser_load_from_data(parser, json_text.c_str(),
                                  static_cast<gssize>(json_text.size()), &gerr)) {
    if (out_error != nullptr) {
      *out_error = std::string("JSON parse error: ") +
          ((gerr != nullptr) ? gerr->message : "unknown");
    }
    if (gerr != nullptr) {
      g_error_free(gerr);
    }
    g_object_unref(parser);
    return false;
  }

  JsonNode* root = json_parser_get_root(parser);
  if (!JSON_NODE_HOLDS_OBJECT(root)) {
    if (out_error != nullptr) {
      *out_error = "Config root must be a JSON object";
    }
    g_object_unref(parser);
    return false;
  }

  JsonObject* root_obj = json_node_get_object(root);
  YoloQnnLatencyConfig cfg {};

  std::string s;
  if (!GetStringArrayMember(root_obj, "videos", &cfg.videos) || cfg.videos.empty()) {
    if (out_error != nullptr) {
      *out_error = "Missing or empty \"videos\" string array";
    }
    g_object_unref(parser);
    return false;
  }
  if (!GetStringMember(root_obj, "output_dir", &cfg.output_dir) || cfg.output_dir.empty()) {
    if (out_error != nullptr) {
      *out_error = "Missing or empty \"output_dir\"";
    }
    g_object_unref(parser);
    return false;
  }
  if (!GetStringMember(root_obj, "model_path", &cfg.model_path) || cfg.model_path.empty()) {
    if (out_error != nullptr) {
      *out_error = "Missing or empty \"model_path\"";
    }
    g_object_unref(parser);
    return false;
  }

  if (GetStringMember(root_obj, "qnn_backend", &s)) {
    cfg.qnn_backend = s;
  }
  if (GetStringMember(root_obj, "qnn_system", &s)) {
    cfg.qnn_system = s;
  }
  uint32_t u = 0U;
  if (GetUIntMember(root_obj, "qnn_backend_device_id", &u)) {
    cfg.qnn_backend_device_id = u;
  }
  if (GetStringArrayMember(root_obj, "qnn_tensors", &cfg.qnn_tensors) &&
      !cfg.qnn_tensors.empty()) {
    // use loaded
  } else {
    cfg.qnn_tensors = {"boxes", "scores", "class_idx"};
  }
  if (GetStringArrayMember(root_obj, "qtimlvconverter_engine_order",
                           &cfg.qtimlvconverter_engine_order) &&
      !cfg.qtimlvconverter_engine_order.empty()) {
    // use loaded
  } else {
    cfg.qtimlvconverter_engine_order = {"fcv", "c2d"};
  }
  if (GetUIntMember(root_obj, "sample_every_n", &u) && (u > 0U)) {
    cfg.sample_every_n = u;
  }
  bool b = false;
  if (GetBoolMember(root_obj, "flush_per_sample", &b)) {
    cfg.flush_per_sample = b;
  }
  if (GetUIntMember(root_obj, "max_channels", &u) && (u > 0U)) {
    cfg.max_channels = u;
  }

  if (cfg.videos.size() > cfg.max_channels) {
    if (out_error != nullptr) {
      *out_error = "Too many videos: " + std::to_string(cfg.videos.size()) +
          " (max_channels=" + std::to_string(cfg.max_channels) + ")";
    }
    g_object_unref(parser);
    return false;
  }

  *out_config = std::move(cfg);
  g_object_unref(parser);
  return true;
}

}  // namespace single_pipeline
