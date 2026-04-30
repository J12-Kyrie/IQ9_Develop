#include <cstdlib>
#include <cstdio>
#include <string>

#include "app/MultiCamApp.hpp"

namespace {

std::string ParseConfigPath(int argc, const char** argv) {
  std::string config_path = "config/config.json";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--config") || (arg == "-c")) {
      if ((i + 1) < argc) {
        config_path = argv[++i];
      }
    }
  }

  return config_path;
}

}  // namespace

int main(int argc, const char** argv) {
  const std::string config_path = ParseConfigPath(argc, argv);
  std::printf("Using config: %s\n", config_path.c_str());

  multi_cam_app::app::MultiCamApp app(config_path);
  return app.Run();
}
