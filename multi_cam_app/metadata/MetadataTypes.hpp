#ifndef MULTI_CAM_APP_METADATA_METADATA_TYPES_HPP
#define MULTI_CAM_APP_METADATA_METADATA_TYPES_HPP

#include <cstdint>

namespace multi_cam_app::metadata {

struct BoundingBoxNorm {
  float x {0.0F};
  float y {0.0F};
  float width {0.0F};
  float height {0.0F};
};

}  // namespace multi_cam_app::metadata

#endif  // MULTI_CAM_APP_METADATA_METADATA_TYPES_HPP
