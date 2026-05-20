#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <array>
#include <algorithm>
#include <cmath>
#include <vector>

namespace utils {

struct BBox {
  float x1, y1, x2, y2;
  float area() const {
    return std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  }
};

inline BBox toBBox(const std::vector<float> &vec) {
  if (vec.size() < 4)
    return {0, 0, 0, 0};
  return {vec[0], vec[1], vec[2], vec[3]};
}

inline float calculate_iou_bbox(const BBox &a, const BBox &b) {
  float interX1 = std::max(a.x1, b.x1);
  float interY1 = std::max(a.y1, b.y1);
  float interX2 = std::min(a.x2, b.x2);
  float interY2 = std::min(a.y2, b.y2);

  float interArea =
      std::max(0.0f, interX2 - interX1) * std::max(0.0f, interY2 - interY1);
  float unionArea = a.area() + b.area() - interArea;

  if (unionArea <= 0.0f)
    return 0.0f;
  return interArea / unionArea;
}

inline float calculate_iou(const std::vector<float> &boxA,
                           const std::vector<float> &boxB) {
  return calculate_iou_bbox(toBBox(boxA), toBBox(boxB));
}

inline std::vector<float> calculate_danger_zone(const std::vector<float> &car_bbox,
                                                float h_expand = 0.15f,
                                                float v_expand = 0.25f) {
  BBox b = toBBox(car_bbox);
  float width = std::max(0.0f, b.x2 - b.x1);
  float height = std::max(0.0f, b.y2 - b.y1);

  float pad_x = width * h_expand;
  float pad_y = height * v_expand;

  std::vector<float> out;
  out.reserve(4);
  out.push_back(b.x1 - pad_x);
  out.push_back(b.y1);
  out.push_back(b.x2 + pad_x);
  out.push_back(b.y2 + pad_y);
  return out;
}

// Vector math for Embeddings
inline float dot_product(const std::vector<float> &a,
                         const std::vector<float> &b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

inline float magnitude(const std::vector<float> &a) {
  return std::sqrt(dot_product(a, a));
}

inline float cosine_similarity(const std::vector<float> &a,
                               const std::vector<float> &b) {
  float magA = magnitude(a);
  float magB = magnitude(b);
  if (magA == 0 || magB == 0)
    return 0.0f;
  return dot_product(a, b) / (magA * magB);
}

} // namespace utils

#endif // MATH_UTILS_H
