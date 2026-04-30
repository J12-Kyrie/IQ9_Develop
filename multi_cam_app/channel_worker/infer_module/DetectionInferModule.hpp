#ifndef MULTI_CAM_APP_CHANNEL_WORKER_INFER_MODULE_DETECTION_INFER_MODULE_HPP
#define MULTI_CAM_APP_CHANNEL_WORKER_INFER_MODULE_DETECTION_INFER_MODULE_HPP

#include <string>

#include <gst/gst.h>

#include "config/AppConfig.hpp"

namespace multi_cam_app::channel_worker::infer_module {

class DetectionInferModule {
public:
  // Builds a unified detection+tracker+metamux+deduper element chain.
  // Both face_enabled paths output a single appsink_combined (NV12 + ROI meta).
  bool BuildChain(const config::AppConfig& config,
                  uint32_t channel_id,
                  const std::string& video_path,
                  GstBin* pipeline_bin,
                  GstElement** out_appsink,
                  std::string* out_error) const;

private:
  static void OnDemuxPadAdded(GstElement* element, GstPad* pad, gpointer user_data);

  static bool SetQnnTensors(GstElement* qtimlqnn,
                            const std::vector<std::string>& tensors,
                            std::string* out_error);
};

}  // namespace multi_cam_app::channel_worker::infer_module

#endif  // MULTI_CAM_APP_CHANNEL_WORKER_INFER_MODULE_DETECTION_INFER_MODULE_HPP
