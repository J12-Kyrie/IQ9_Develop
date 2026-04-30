cmake_minimum_required(VERSION 3.16)
project(gst_plugin_frameoffload LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED APP_DIR)
  message(FATAL_ERROR "APP_DIR not defined. Pass -DAPP_DIR=<path to multi_cam_app>")
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(GST       REQUIRED gstreamer-1.0>=1.16)
pkg_check_modules(GST_BASE  REQUIRED gstreamer-base-1.0>=1.16)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0>=1.16)
pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0>=1.0)

set(FRAME_OFFLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(FACE_META_DIR "${APP_DIR}/channel_worker/infer_module/face_infer/gst_plugin")
set(FRAME_CACHE_DIR "${APP_DIR}/frame_cache")

set(PLUGIN_TARGET gstqtiframeoffload)

add_library(${PLUGIN_TARGET} SHARED
    gstframeoffload.cpp
    ${FRAME_OFFLOAD_DIR}/FrameOffload.cpp
    ${APP_DIR}/utils/JsonlWriter.cpp
    ${FACE_META_DIR}/gstfacedetectionmeta.c
    ${FRAME_CACHE_DIR}/src/producer.cpp
    ${FRAME_CACHE_DIR}/src/slot_pool.cpp
    ${APP_DIR}/utils/FaceGallery.cpp
    ${FRAME_CACHE_DIR}/src/nv12_to_rgb.cpp
    ${FRAME_CACHE_DIR}/src/ipc_sender.cpp
)

target_include_directories(${PLUGIN_TARGET} PRIVATE
    ${JSON_GLIB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FRAME_OFFLOAD_DIR}
    ${APP_DIR}
    ${APP_DIR}/frame_cache/include
    ${APP_DIR}/frame_cache/src
    ${GST_INCLUDE_DIRS}
    ${GST_BASE_INCLUDE_DIRS}
    ${GST_VIDEO_INCLUDE_DIRS}
)

target_compile_definitions(${PLUGIN_TARGET} PRIVATE
    PACKAGE="qtiframeoffload"
)

target_compile_options(${PLUGIN_TARGET} PRIVATE
    -Wall -Wextra -Werror
    -Wno-unused-parameter
)

target_link_libraries(${PLUGIN_TARGET} PRIVATE
    ${JSON_GLIB_LIBRARIES}
    ${GST_LIBRARIES}
    ${GST_BASE_LIBRARIES}
    ${GST_VIDEO_LIBRARIES}
)
