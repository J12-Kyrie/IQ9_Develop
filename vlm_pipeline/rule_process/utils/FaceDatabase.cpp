#include "FaceDatabase.h"
#include "infra/Logger.h"
#include "nlohmann/json.hpp"
#include <fstream>

namespace utils {

FaceDatabase::FaceDatabase(size_t embeddingSize) : embedding_size_(embeddingSize) {}

FaceDatabase::~FaceDatabase() = default;

core::Status FaceDatabase::loadFromFile(const std::string& path) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    return loadPlainTextFile(path);
}

core::Status FaceDatabase::loadPlainTextFile(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            return core::Status::Error("Cannot open face database file: " + path);
        }

        nlohmann::json j;
        file >> j;

        if (!j.contains("faces") || !j["faces"].is_object()) {
            return core::Status::Error("Invalid face database format: missing faces object");
        }

        db_.clear();

        for (auto it = j["faces"].begin(); it != j["faces"].end(); ++it) {
            const std::string& faceId = it.key();

            if (!it.value().is_array()) {
                Logger::warn("FaceDatabase", "Skipping non-array embedding for face: " + faceId);
                continue;
            }

            std::vector<float> embedding = it.value().get<std::vector<float>>();

            if (embedding.size() != embedding_size_) {
                Logger::warn("FaceDatabase", "Skipping face " + faceId +
                           " with invalid embedding size: " + std::to_string(embedding.size()) +
                           " (expected: " + std::to_string(embedding_size_) + ")");
                continue;
            }

            db_[faceId] = std::move(embedding);
        }

        Logger::info("FaceDatabase", "Loaded " + std::to_string(db_.size()) + " faces from " + path);
        return core::Status::Ok();

    } catch (const std::exception& e) {
        return core::Status::Error("Error parsing face database: " + std::string(e.what()));
    }
}

std::pair<std::string, float> FaceDatabase::compare(const std::vector<float>& input_embedding) const {
    std::lock_guard<std::mutex> lock(db_mutex_);

    std::string best_match = "unknown";
    float max_score = -1.0f;

    if (input_embedding.empty()) {
        return {best_match, 0.0f};
    }

    if (input_embedding.size() != embedding_size_) {
        Logger::warn("FaceDatabase", "Input embedding size mismatch: " +
                    std::to_string(input_embedding.size()) +
                    " (expected: " + std::to_string(embedding_size_) + ")");
        return {best_match, 0.0f};
    }

    for (const auto& entry : db_) {
        float score = utils::cosine_similarity(input_embedding, entry.second);
        if (score > max_score) {
            max_score = score;
            best_match = entry.first;
        }
    }

    return {best_match, max_score};
}

size_t FaceDatabase::size() const {
    std::lock_guard<std::mutex> lock(db_mutex_);
    return db_.size();
}

} // namespace utils
