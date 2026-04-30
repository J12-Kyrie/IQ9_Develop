#ifndef MULTI_CAM_APP_AGGREGATOR_QUEUE_ITEM_HPP
#define MULTI_CAM_APP_AGGREGATOR_QUEUE_ITEM_HPP

#include "utils/JsonlWriter.hpp"

namespace multi_cam_app::aggregator {

struct QueueItem {
  output::FrameRecord record;
};

}  // namespace multi_cam_app::aggregator

#endif  // MULTI_CAM_APP_AGGREGATOR_QUEUE_ITEM_HPP
