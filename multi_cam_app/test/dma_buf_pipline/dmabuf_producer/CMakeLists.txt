cmake_minimum_required(VERSION 3.16)
project(dmabuf_producer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
)

# --- Static library (no GStreamer link) ---
add_library(dmabuf_producer_lib STATIC
    src/slot_pool.cpp
    src/ipc_sender.cpp
    src/nv12_to_rgb.cpp
    src/producer.cpp
)

target_include_directories(dmabuf_producer_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(dmabuf_producer_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dmabuf_producer_lib PRIVATE pthread)

# --- nlohmann_json (header-only, for demo) ---
find_package(nlohmann_json QUIET)

# --- Paho MQTT C (synchronous client, for demo) ---
find_library(PAHO_MQTT3C_LIB paho-mqtt3c)

# --- Demo executable (links GStreamer + MQTT + nlohmann_json) ---
add_executable(producer_demo demo/producer_demo.cpp)

target_include_directories(producer_demo PRIVATE
    ${GST_INCLUDE_DIRS}
)

target_link_libraries(producer_demo PRIVATE
    dmabuf_producer_lib
    ${GST_LIBRARIES}
    pthread
    ${PAHO_MQTT3C_LIB}
)

if(nlohmann_json_FOUND)
    target_link_libraries(producer_demo PRIVATE nlohmann_json::nlohmann_json)
endif()

target_compile_options(producer_demo PRIVATE ${GST_CFLAGS_OTHER})

set_target_properties(producer_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# --- NV12->RGB unit test (no GStreamer dependency) ---
add_executable(test_nv12_to_rgb test/test_nv12_to_rgb.cpp src/nv12_to_rgb.cpp)

target_include_directories(test_nv12_to_rgb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(test_nv12_to_rgb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

enable_testing()
add_test(NAME nv12_to_rgb COMMAND test_nv12_to_rgb)
