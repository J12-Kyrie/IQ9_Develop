#pragma once

#include "core/DataTypes.h"
#include <string>
#include <vector>

namespace adapters {

class EventAdapter {
public:
  // Convert JSON to EventMeta struct.
  // Returns empty vector if validation/file check fails.
  // scene_update bbox JSON is normalized [x, y, w, h]; converted to [x1, y1, x2, y2] for rules/IoU.
  std::vector<core::EventMeta> adaptSceneUpdate(const std::string &jsonPayload);

private:
};

} // namespace adapters
