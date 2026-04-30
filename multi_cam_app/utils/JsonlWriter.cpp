#include "utils/JsonlWriter.hpp"

#include <filesystem>
#include <sstream>

namespace multi_cam_app::output {

JsonlWriter::JsonlWriter(std::string file_path) : file_path_(std::move(file_path)) {}

JsonlWriter::~JsonlWriter() {
  Close();
}

bool JsonlWriter::Open(std::string* out_error) {
  std::lock_guard<std::mutex> lock(write_mutex_);

  if (file_.is_open()) {
    return true;
  }

  const std::filesystem::path path(file_path_);
  const std::filesystem::path parent = path.parent_path();
  std::error_code ec;
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      if (out_error != nullptr) {
        *out_error =
            "Failed to create output directory: " + parent.string() + " (" + ec.message() + ")";
      }
      return false;
    }
  }

  file_.open(file_path_, std::ios::out | std::ios::trunc);
  if (!file_.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open jsonl file: " + file_path_;
    }
    return false;
  }

  return true;
}

void JsonlWriter::Close() {
  std::lock_guard<std::mutex> lock(write_mutex_);
  if (file_.is_open()) {
    file_.flush();
    file_.close();
  }
}

std::string JsonlWriter::Serialize(const FrameRecord& record) {
  std::ostringstream oss;
  oss << "{\"channel_id\":" << record.channel_id
      << ",\"image_path\":" << record.image_path
      << ",\"frame_id\":" << record.frame_id
      << ",\"timestamp_ns\":" << record.timestamp_ns
      << ",\"detections\":[";

  for (size_t i = 0U; i < record.detections.size(); ++i) {
    const auto& det = record.detections[i];
    if (i != 0U) {
      oss << ",";
    }
    oss << "{\"class_id\":" << det.class_id
        << ",\"track_id\":" << det.track_id
        << ",\"label\":\"" << EscapeJson(det.label) << "\""
        << ",\"score\":" << det.score
        << ",\"bbox\":["
        << det.left << "," << det.top << "," << det.right << "," << det.bottom
        << "]";

    if (det.face.has_value()) {
      const auto& f = det.face.value();
      oss << ",\"face\":{\"confidence\":" << f.confidence
          << ",\"bbox\":[" << f.bbox_x1 << "," << f.bbox_y1
          << "," << f.bbox_x2 << "," << f.bbox_y2 << "]"
          << ",\"landmarks\":[";
      for (int li = 0; li < 10; ++li) {
        if (li != 0) oss << ",";
        oss << f.landmarks[li];
      }
      oss << "],\"embedding\":[";
      for (size_t ei = 0; ei < f.embedding.size(); ++ei) {
        if (ei != 0) oss << ",";
        oss << f.embedding[ei];
      }
      oss << "],\"linked_track_id\":" << f.linked_track_id;
      if (f.face_id >= 0) {
        oss << ",\"face_id\":" << f.face_id
            << ",\"face_similarity\":" << f.face_similarity;
      }
      oss << "}";
    }

    oss << "}";
  }

  oss << "]}";
  return oss.str();
}

std::string JsonlWriter::SerializeSceneSource(const FrameRecord& record,
                                              uint32_t meta_width,
                                              uint32_t meta_height) {
  std::ostringstream oss;
  oss << "{\"source_id\":" << record.channel_id
      << ",\"frame_id\":" << record.frame_id
      << ",\"timestamp_ns\":" << record.timestamp_ns
      << ",\"events\":[";
  for (size_t i = 0U; i < record.detections.size(); ++i) {
    const auto& det = record.detections[i];
    if (i != 0U) {
      oss << ",";
    }
    oss << "{\"class_name\":\"" << EscapeJson(det.label) << "\""
        << ",\"track_id\":" << det.track_id
        << ",\"confidence\":" << static_cast<float>(det.score)
        << ",\"bbox\":[" << det.left << "," << det.top << ","
        << det.right << "," << det.bottom << "]";
    if (det.face.has_value()) {
      const auto& f = det.face.value();
      oss << ",\"embedding\":[";
      for (size_t ei = 0; ei < f.embedding.size(); ++ei) {
        if (ei != 0) {
          oss << ",";
        }
        oss << f.embedding[ei];
      }
      oss << "]";
    }
    oss << "}";
  }
  oss << "],\"removed_track_ids\":[]";
  if (record.image_path >= 0) {
    oss << ",\"image_meta\":{"
        << "\"slot_index\":" << static_cast<uint32_t>(record.image_path)
        << ",\"width\":" << meta_width
        << ",\"height\":" << meta_height
        << ",\"channels\":3"
        << "}";
  }
  oss << "}";
  return oss.str();
}

bool JsonlWriter::Write(const FrameRecord& record, std::string* out_error) {
  std::lock_guard<std::mutex> lock(write_mutex_);

  if (!file_.is_open()) {
    if (out_error != nullptr) {
      *out_error = "JsonlWriter is not open: " + file_path_;
    }
    return false;
  }

  file_ << Serialize(record) << "\n";
  if (!file_) {
    if (out_error != nullptr) {
      *out_error = "Failed to write jsonl record to file: " + file_path_;
    }
    return false;
  }

  file_.flush();
  return true;
}

std::string JsonlWriter::EscapeJson(const std::string& text) {
  std::ostringstream oss;
  for (char ch : text) {
    switch (ch) {
      case '\"':
        oss << "\\\"";
        break;
      case '\\':
        oss << "\\\\";
        break;
      case '\b':
        oss << "\\b";
        break;
      case '\f':
        oss << "\\f";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << ch;
        break;
    }
  }
  return oss.str();
}

}  // namespace multi_cam_app::output
