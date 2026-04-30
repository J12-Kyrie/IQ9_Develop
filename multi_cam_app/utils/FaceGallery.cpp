#include "utils/FaceGallery.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <shared_mutex>

#include <json-glib/json-glib.h>

namespace multi_cam_app::gallery {

float FaceGallery::CosineSimilarity(const float* a, const float* b, int dim) {
  float dot = 0.0f;
  for (int i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return dot;  // Both vectors are L2-normalized, so dot product == cosine similarity
}

bool FaceGallery::Load(const std::string& json_path) {
  file_path_ = json_path;
  entries_.clear();
  next_id_ = 1;
  loaded_ = false;

  // If file does not exist, create empty gallery
  if (!std::filesystem::exists(json_path)) {
    std::printf("[gallery] File not found, creating empty gallery: %s\n", json_path.c_str());
    std::fflush(stdout);
    loaded_ = true;
    return Save();
  }

  JsonParser* parser = json_parser_new();
  GError* error = nullptr;
  gboolean ok = json_parser_load_from_file(parser, json_path.c_str(), &error);
  if (!ok) {
    std::printf("[gallery] Failed to parse JSON: %s\n",
                (error && error->message) ? error->message : "unknown");
    std::fflush(stdout);
    if (error) g_error_free(error);
    g_object_unref(parser);
    return false;
  }

  JsonNode* root = json_parser_get_root(parser);
  if (!root || !JSON_NODE_HOLDS_OBJECT(root)) {
    std::printf("[gallery] Root is not a JSON object\n");
    std::fflush(stdout);
    g_object_unref(parser);
    return false;
  }

  JsonObject* root_obj = json_node_get_object(root);

  if (json_object_has_member(root_obj, "next_id")) {
    next_id_ = static_cast<int>(json_object_get_int_member(root_obj, "next_id"));
  }

  if (json_object_has_member(root_obj, "entries")) {
    JsonArray* arr = json_object_get_array_member(root_obj, "entries");
    if (arr) {
      guint len = json_array_get_length(arr);
      entries_.reserve(len);
      for (guint i = 0; i < len; ++i) {
        JsonObject* entry_obj = json_array_get_object_element(arr, i);
        if (!entry_obj) continue;

        GalleryEntry entry;
        if (json_object_has_member(entry_obj, "face_id")) {
          entry.face_id = static_cast<int>(json_object_get_int_member(entry_obj, "face_id"));
        }

        if (json_object_has_member(entry_obj, "embedding")) {
          JsonArray* emb_arr = json_object_get_array_member(entry_obj, "embedding");
          if (emb_arr) {
            guint emb_len = json_array_get_length(emb_arr);
            entry.embedding.resize(emb_len);
            for (guint j = 0; j < emb_len; ++j) {
              entry.embedding[j] = static_cast<float>(json_array_get_double_element(emb_arr, j));
            }
          }
        }

        if (!entry.embedding.empty()) {
          entries_.push_back(std::move(entry));
        }
      }
    }
  }

  g_object_unref(parser);
  loaded_ = true;

  std::printf("[gallery] Loaded %zu entries, next_id=%d from %s\n",
              entries_.size(), next_id_, json_path.c_str());
  std::fflush(stdout);
  return true;
}

bool FaceGallery::Save() const {
  if (file_path_.empty()) return false;

  // Ensure parent directory exists
  std::filesystem::path p(file_path_);
  std::error_code ec;
  if (!p.parent_path().empty()) {
    std::filesystem::create_directories(p.parent_path(), ec);
  }

  JsonBuilder* builder = json_builder_new();

  json_builder_begin_object(builder);

  json_builder_set_member_name(builder, "next_id");
  json_builder_add_int_value(builder, next_id_);

  json_builder_set_member_name(builder, "entries");
  json_builder_begin_array(builder);

  for (const auto& entry : entries_) {
    json_builder_begin_object(builder);

    json_builder_set_member_name(builder, "face_id");
    json_builder_add_int_value(builder, entry.face_id);

    json_builder_set_member_name(builder, "embedding");
    json_builder_begin_array(builder);
    for (float v : entry.embedding) {
      json_builder_add_double_value(builder, static_cast<double>(v));
    }
    json_builder_end_array(builder);

    json_builder_end_object(builder);
  }

  json_builder_end_array(builder);
  json_builder_end_object(builder);

  JsonGenerator* gen = json_generator_new();
  json_generator_set_pretty(gen, TRUE);
  json_generator_set_indent(gen, 4);
  JsonNode* root = json_builder_get_root(builder);
  json_generator_set_root(gen, root);

  GError* error = nullptr;
  gboolean ok = json_generator_to_file(gen, file_path_.c_str(), &error);

  json_node_unref(root);
  g_object_unref(gen);
  g_object_unref(builder);

  if (!ok) {
    std::printf("[gallery] Failed to save: %s\n",
                (error && error->message) ? error->message : "unknown");
    std::fflush(stdout);
    if (error) g_error_free(error);
    return false;
  }

  return true;
}

MatchResult FaceGallery::Match(const float* embedding, float threshold) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  MatchResult best;
  best.face_id = -1;
  best.similarity = -1.0f;

  if (embedding == nullptr || entries_.empty()) {
    return best;
  }

  for (const auto& entry : entries_) {
    if (static_cast<int>(entry.embedding.size()) != kEmbeddingDim) continue;

    float sim = CosineSimilarity(embedding, entry.embedding.data(), kEmbeddingDim);
    if (sim > best.similarity) {
      best.similarity = sim;
      best.face_id = entry.face_id;
    }
  }

  // Apply threshold: if best similarity < threshold, return -1
  if (best.similarity < threshold) {
    best.face_id = -1;
  }

  return best;
}

int FaceGallery::Enroll(const float* embedding) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (embedding == nullptr) return -1;

  const int new_id = next_id_++;

  GalleryEntry entry;
  entry.face_id = new_id;
  entry.embedding.assign(embedding, embedding + kEmbeddingDim);
  entries_.push_back(std::move(entry));

  std::printf("[gallery] Enrolled face_id=%d, gallery size=%zu\n",
              new_id, entries_.size());
  std::fflush(stdout);

  Save();
  return new_id;
}

}  // namespace multi_cam_app::gallery
