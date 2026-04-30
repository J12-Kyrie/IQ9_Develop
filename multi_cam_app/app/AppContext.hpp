#ifndef MULTI_CAM_APP_APP_APP_CONTEXT_HPP
#define MULTI_CAM_APP_APP_APP_CONTEXT_HPP

#include <atomic>
#include <gst/gst.h>

namespace multi_cam_app::app {

struct AppContext {
  std::atomic<bool> stop_requested {false};
  std::atomic<bool> error_received {false};
  GMainLoop* main_loop {nullptr};

  void RequestStop() {
    stop_requested.store(true);
    if (main_loop != nullptr) {
      g_main_loop_quit(main_loop);
    }
  }
};

}  // namespace multi_cam_app::app

#endif  // MULTI_CAM_APP_APP_APP_CONTEXT_HPP
