#ifndef FACE_DATABASE_H
#define FACE_DATABASE_H

#include "MathUtils.h"
#include "core/DataTypes.h"
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <mutex>

namespace utils {

class FaceDatabase {
public:
    explicit FaceDatabase(size_t embeddingSize = 128);
    ~FaceDatabase();

    core::Status loadFromFile(const std::string& path);

    // Returns {UserId, SimilarityScore}
    std::pair<std::string, float> compare(const std::vector<float>& input_embedding) const;

    size_t size() const;

private:
    core::Status loadPlainTextFile(const std::string& path);

    std::map<std::string, std::vector<float>> db_;
    mutable std::mutex db_mutex_;
    size_t embedding_size_;
};

} // namespace utils

#endif // FACE_DATABASE_H
