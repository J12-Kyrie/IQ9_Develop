cmake_minimum_required(VERSION 3.16)
project(test_deduper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)

# --- Deduper core (parent directory) ---
set(DEDUPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(APP_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

# ======================================
# Group A: Pure C++ unit test (no GStreamer)
# ======================================
add_executable(test_deduper_unit
    test_deduper_unit.cpp
    ${DEDUPER_DIR}/Deduper.cpp
)

target_include_directories(test_deduper_unit PRIVATE
    ${DEDUPER_DIR}
)

target_compile_options(test_deduper_unit PRIVATE
    -Wall -Wextra -Werror -Wno-unused-parameter
)

# ======================================
# Group B: GStreamer pipeline integration test
# ======================================
pkg_check_modules(GST       REQUIRED gstreamer-1.0>=1.16)
pkg_check_modules(GST_APP   REQUIRED gstreamer-app-1.0>=1.16)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0>=1.16)

add_executable(test_deduper_pipeline
    test_deduper_pipeline.cpp
    ${APP_DIR}/pipeline/PipelineUtils.cpp
)

target_include_directories(test_deduper_pipeline PRIVATE
    ${APP_DIR}
    ${DEDUPER_DIR}
    ${GST_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
    ${GST_VIDEO_INCLUDE_DIRS}
)

target_compile_options(test_deduper_pipeline PRIVATE
    -Wall -Wextra -Werror -Wno-unused-parameter
)

target_link_libraries(test_deduper_pipeline PRIVATE
    ${GST_LIBRARIES}
    ${GST_APP_LIBRARIES}
    ${GST_VIDEO_LIBRARIES}
)
