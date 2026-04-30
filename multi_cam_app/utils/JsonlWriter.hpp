#ifndef MULTI_CAM_APP_OUTPUT_JSONL_WRITER_HPP
#define MULTI_CAM_APP_OUTPUT_JSONL_WRITER_HPP

#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace multi_cam_app::output {

struct FaceInfo {
  float confidence {0.0f};
  float bbox_x1 {0.0f};
  float bbox_y1 {0.0f};
  float bbox_x2 {0.0f};
  float bbox_y2 {0.0f};
  float landmarks[10] {};              // 5 landmarks, x/y interleaved
  std::vector<float> embedding {};     // 512-dim L2 normalized, empty if ArcFace disabled
  uint32_t linked_track_id {0U};
  int face_id {-1};                    // gallery identity ID (-1 = not enabled / no match)
  float face_similarity {0.0f};        // cosine similarity with best gallery entry
};

struct DetectionRecord {
  uint32_t class_id {0U};
  uint32_t track_id {0U};
  std::string label {};
  double score {0.0};
  double left {0.0};
  double top {0.0};
  double right {0.0};
  double bottom {0.0};
  std::optional<FaceInfo> face {};
};

struct FrameRecord {
  uint32_t channel_id {0U};
  int32_t image_path {-1};
  uint64_t frame_id {0ULL};
  uint64_t timestamp_ns {0ULL};
  std::vector<DetectionRecord> detections {};
};

class JsonlWriter {
public:
  explicit JsonlWriter(std::string file_path);
  ~JsonlWriter();

  JsonlWriter(const JsonlWriter&) = delete;
  JsonlWriter& operator=(const JsonlWriter&) = delete;

  bool Open(std::string* out_error);
  void Close();

  bool Write(const FrameRecord& record, std::string* out_error);

  static std::string Serialize(const FrameRecord& record);

  // One element of scene_update.sources[] (rule_process EventAdapter / producer_demo shape).
  static std::string SerializeSceneSource(const FrameRecord& record,
                                          uint32_t meta_width,
                                          uint32_t meta_height);

  const std::string& FilePath() const {
    return file_path_;
  }

private:
  static std::string EscapeJson(const std::string& text);

  std::string file_path_;
  std::ofstream file_;
  std::mutex write_mutex_;
};

}  // namespace multi_cam_app::output

#endif  // MULTI_CAM_APP_OUTPUT_JSONL_WRITER_HPP
