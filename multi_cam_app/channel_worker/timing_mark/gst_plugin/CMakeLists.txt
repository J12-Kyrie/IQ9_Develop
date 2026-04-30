cmake_minimum_required(VERSION 3.16)
project(gst_plugin_timingmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED APP_DIR)
  message(FATAL_ERROR "APP_DIR not defined. Pass -DAPP_DIR=<path to multi_cam_app>")
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.16)
pkg_check_modules(GST_BASE REQUIRED gstreamer-base-1.0>=1.16)

set(TIMING_MARK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(PLUGIN_TARGET gstqtitimingmark)

add_library(${PLUGIN_TARGET} SHARED
  gsttimingmark.cpp
  ${APP_DIR}/utils/perf/LatencyRecorder.cpp
  ${APP_DIR}/utils/perf/TimingMarkRuntime.cpp
)

target_include_directories(${PLUGIN_TARGET} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${APP_DIR}
  ${GST_INCLUDE_DIRS}
  ${GST_BASE_INCLUDE_DIRS}
)

target_compile_definitions(${PLUGIN_TARGET} PRIVATE
  PACKAGE="qtitimingmark"
)

target_compile_options(${PLUGIN_TARGET} PRIVATE
  -Wall -Wextra -Werror
  -Wno-unused-parameter
)

target_link_libraries(${PLUGIN_TARGET} PRIVATE
  ${GST_LIBRARIES}
  ${GST_BASE_LIBRARIES}
)
