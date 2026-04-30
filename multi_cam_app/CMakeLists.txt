cmake_minimum_required(VERSION 3.16)
project(multi_cam_app LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- AddressSanitizer option for memory testing ---
option(ENABLE_ASAN "Enable AddressSanitizer (-fsanitize=address)" OFF)
if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer ENABLED")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
endif()

if(NOT DEFINED GST_VERSION_REQUIRED)
  set(GST_VERSION_REQUIRED "1.16")
endif()

if(NOT DEFINED GST_PLUGINS_QTI_OSS_INSTALL_BINDIR)
  set(GST_PLUGINS_QTI_OSS_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}")
endif()

if(NOT DEFINED GST_PLUGINS_QTI_OSS_INSTALL_CONFIG)
  set(GST_PLUGINS_QTI_OSS_INSTALL_CONFIG "${CMAKE_INSTALL_SYSCONFDIR}/configs")
endif()

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# Paho MQTT C library (optional - for MQTT publish/subscribe)
find_library(PAHO_MQTT3C_LIB paho-mqtt3c)
if(PAHO_MQTT3C_LIB)
  message(STATUS "Found paho-mqtt3c: ${PAHO_MQTT3C_LIB}")
else()
  message(WARNING "paho-mqtt3c not found - install with: apt install libpaho-mqtt-dev")
endif()

pkg_check_modules(GST REQUIRED gstreamer-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_JSON REQUIRED json-glib-1.0)
pkg_check_modules(GST_QCOM_VIDEO gstreamer-qcom-oss-video-1.0>=1.0.0)
pkg_check_modules(GST_ALLOC REQUIRED gstreamer-allocators-1.0>=${GST_VERSION_REQUIRED})

set(APP_TARGET multi-cam-app)
set(APP_SOURCES
  main.cpp
  app/MultiCamApp.cpp
  config/ConfigLoader.cpp
  channel_worker/ChannelWorker.cpp
  channel_worker/infer_module/DetectionInferModule.cpp
  pipeline/PipelineUtils.cpp
  utils/JsonlWriter.cpp
  utils/FaceGallery.cpp
  utils/mqtt/MqttClient.cpp
  utils/mqtt/MqttPublisher.cpp
  utils/mqtt/MqttSubscriber.cpp
  aggregator/Aggregator.cpp
  # FrameCache (DmaBufProducer): DMA slot pool for NV12->RGB24 frame storage
  frame_cache/src/producer.cpp
  frame_cache/src/slot_pool.cpp
  frame_cache/src/nv12_to_rgb.cpp
  frame_cache/src/ipc_sender.cpp
  frame_cache/src/frame_cache_service.cpp
  # GstFaceDetectionMeta: needed by ChannelWorker to read face results from qtifaceinfer plugin
  channel_worker/infer_module/face_infer/gst_plugin/gstfacedetectionmeta.c
)

add_executable(${APP_TARGET} ${APP_SOURCES})

target_include_directories(${APP_TARGET} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/aggregator/concurrentqueue
  ${CMAKE_CURRENT_SOURCE_DIR}/frame_cache/include
  ${CMAKE_CURRENT_SOURCE_DIR}/frame_cache/src
  ${GST_INCLUDE_DIRS}
  ${GST_APP_INCLUDE_DIRS}
  ${GST_VIDEO_INCLUDE_DIRS}
  ${GST_JSON_INCLUDE_DIRS}
  ${GST_QCOM_VIDEO_INCLUDE_DIRS}
  ${GST_ALLOC_INCLUDE_DIRS}
)

target_compile_options(${APP_TARGET} PRIVATE
  -Wall
  -Wextra
  -Werror
  -Wno-unused-parameter
)

target_link_libraries(${APP_TARGET} PRIVATE
  ${GST_LIBRARIES}
  ${GST_APP_LIBRARIES}
  ${GST_VIDEO_LIBRARIES}
  ${GST_JSON_LIBRARIES}
  ${GST_QCOM_VIDEO_LIBRARIES}
  ${GST_ALLOC_LIBRARIES}
  Threads::Threads
  ${PAHO_MQTT3C_LIB}
  nlohmann_json::nlohmann_json
)

if(GST_QCOM_VIDEO_FOUND)
  target_compile_definitions(${APP_TARGET} PRIVATE HAVE_GST_QCOM_VIDEO=1)
endif()

install(TARGETS ${APP_TARGET}
  RUNTIME DESTINATION ${GST_PLUGINS_QTI_OSS_INSTALL_BINDIR}
)

install(FILES config/config.json
  DESTINATION ${GST_PLUGINS_QTI_OSS_INSTALL_CONFIG}
)
