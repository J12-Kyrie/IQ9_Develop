#ifndef MULTI_CAM_APP_CONFIG_CONFIG_LOADER_HPP
#define MULTI_CAM_APP_CONFIG_CONFIG_LOADER_HPP

#include <string>

#include "config/AppConfig.hpp"

namespace multi_cam_app::config {

class ConfigLoader {
public:
  static bool LoadFromFile(const std::string& config_file_path,
                           AppConfig* out_config,
                           std::string* out_error);
};

}  // namespace multi_cam_app::config

#endif  // MULTI_CAM_APP_CONFIG_CONFIG_LOADER_HPP
