cmake_minimum_required(VERSION 3.16)
project(gst_plugin_deduper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GST       REQUIRED gstreamer-1.0>=1.16)
pkg_check_modules(GST_BASE  REQUIRED gstreamer-base-1.0>=1.16)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0>=1.16)

# Deduper core logic directory (parent of gst_plugin/)
set(DEDUPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(PLUGIN_TARGET gstqtideduper)

add_library(${PLUGIN_TARGET} SHARED
    gstdeduper.cpp
    ${DEDUPER_DIR}/Deduper.cpp
)

target_include_directories(${PLUGIN_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DEDUPER_DIR}
    ${GST_INCLUDE_DIRS}
    ${GST_BASE_INCLUDE_DIRS}
    ${GST_VIDEO_INCLUDE_DIRS}
)

target_compile_definitions(${PLUGIN_TARGET} PRIVATE
    PACKAGE="qtideduper"
)

target_compile_options(${PLUGIN_TARGET} PRIVATE
    -Wall -Wextra -Werror
    -Wno-unused-parameter
)

target_link_libraries(${PLUGIN_TARGET} PRIVATE
    ${GST_LIBRARIES}
    ${GST_BASE_LIBRARIES}
    ${GST_VIDEO_LIBRARIES}
)
