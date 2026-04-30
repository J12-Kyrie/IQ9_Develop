cmake_minimum_required(VERSION 3.16)
project(gst_plugin_faceinfer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.16)
pkg_check_modules(GST_BASE REQUIRED gstreamer-base-1.0>=1.16)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0>=1.16)
pkg_check_modules(GST_ALLOC REQUIRED gstreamer-allocators-1.0>=1.16)
pkg_check_modules(GST_JSON REQUIRED json-glib-1.0)

# Face infer source directory (parent of gst_plugin/)
set(FACE_INFER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
# multi_cam_app root (for includes)
set(APP_DIR "${FACE_INFER_DIR}/../../..")

set(PLUGIN_TARGET gstqtifaceinfer)

add_library(${PLUGIN_TARGET} SHARED
  # Plugin files
  gstfacedetectionmeta.c
  gstfaceinfer.cpp

  # FaceProcessor + all submodules (compiled into the plugin .so)
  ${FACE_INFER_DIR}/FaceProcessor.cpp
  ${FACE_INFER_DIR}/face_preprocess/FacePreprocess.cpp
  ${FACE_INFER_DIR}/scrfd_infer/ScrfdDecode.cpp
  ${FACE_INFER_DIR}/scrfd_infer/QnnInferencer.cpp
  ${FACE_INFER_DIR}/arcface_infer/ArcFaceInfer.cpp
  ${FACE_INFER_DIR}/arcface_infer/QnnInferencer.cpp
  ${FACE_INFER_DIR}/mem_management/opencl_loader.cpp
  ${FACE_INFER_DIR}/mem_management/dma_buffer.cpp
)

target_include_directories(${PLUGIN_TARGET} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${FACE_INFER_DIR}
  ${APP_DIR}
  ${GST_INCLUDE_DIRS}
  ${GST_BASE_INCLUDE_DIRS}
  ${GST_VIDEO_INCLUDE_DIRS}
  ${GST_ALLOC_INCLUDE_DIRS}
  ${GST_JSON_INCLUDE_DIRS}
)

target_compile_definitions(${PLUGIN_TARGET} PRIVATE
  PACKAGE="qtifaceinfer"
)

target_compile_options(${PLUGIN_TARGET} PRIVATE
  -Wall -Wextra -Werror
  -Wno-unused-parameter
)

target_link_libraries(${PLUGIN_TARGET} PRIVATE
  ${GST_LIBRARIES}
  ${GST_BASE_LIBRARIES}
  ${GST_VIDEO_LIBRARIES}
  ${GST_ALLOC_LIBRARIES}
  ${GST_JSON_LIBRARIES}
  dl
)
