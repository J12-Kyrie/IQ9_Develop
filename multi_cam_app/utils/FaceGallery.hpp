#ifndef MULTI_CAM_APP_UTILS_FACE_GALLERY_HPP
#define MULTI_CAM_APP_UTILS_FACE_GALLERY_HPP

#include <shared_mutex>
#include <string>
#include <vector>

namespace multi_cam_app::gallery {

struct GalleryEntry {
  int face_id {0};
  std::vector<float> embedding;  // 512-dim, L2 normalized
};

struct MatchResult {
  int face_id {-1};         // matched ID, -1 = no match / gallery empty
  float similarity {0.0f};  // highest cosine similarity
};

class FaceGallery {
public:
  /// Load gallery from JSON file. Creates empty gallery if file does not exist.
  bool Load(const std::string& json_path);

  /// Save current gallery to JSON file.
  bool Save() const;

  /// Match embedding against all gallery entries.
  /// Returns the face_id of the entry with the highest cosine similarity.
  /// If max similarity < threshold, returns face_id = -1.
  MatchResult Match(const float* embedding, float threshold) const;

  /// Enroll a new face: face_id = next_id_++, append entry, save to disk.
  /// @return newly assigned face_id (integer).
  int Enroll(const float* embedding);

  bool Empty() const { return entries_.empty(); }
  size_t Size() const { return entries_.size(); }
  bool IsLoaded() const { return loaded_; }

private:
  static constexpr int kEmbeddingDim = 512;
  static float CosineSimilarity(const float* a, const float* b, int dim);

  mutable std::shared_mutex mutex_;  // Match=shared, Enroll=exclusive
  std::vector<GalleryEntry> entries_;
  std::string file_path_;
  int next_id_ = 1;
  bool loaded_ = false;
};

}  // namespace multi_cam_app::gallery

#endif  // MULTI_CAM_APP_UTILS_FACE_GALLERY_HPP
