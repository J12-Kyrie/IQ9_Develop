cmake_minimum_required(VERSION 3.16)
project(single_yolo_qnn_latency LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

if(NOT DEFINED GST_VERSION_REQUIRED)
  set(GST_VERSION_REQUIRED "1.16")
endif()

pkg_check_modules(GST REQUIRED gstreamer-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_JSON REQUIRED json-glib-1.0)

add_executable(single-yolo-qnn-latency
  main.cpp
  pipeline_local.cpp
  yolo_latency_config.cpp
  yolo_chain.cpp
)

target_include_directories(single-yolo-qnn-latency PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GST_INCLUDE_DIRS}
  ${GST_JSON_INCLUDE_DIRS}
)

target_compile_options(single-yolo-qnn-latency PRIVATE
  -Wall
  -Wextra
  -Werror
  -Wno-unused-parameter
)

target_link_libraries(single-yolo-qnn-latency PRIVATE
  ${GST_LIBRARIES}
  ${GST_JSON_LIBRARIES}
  Threads::Threads
)

target_compile_definitions(single-yolo-qnn-latency PRIVATE
  G_LOG_DOMAIN="single-yolo-qnn-latency"
)
