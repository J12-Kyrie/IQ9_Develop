cmake_minimum_required(VERSION 3.10)
project(VEGCombinedRunner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# QNN SDK: default to system install (/usr) on IQ9
if(NOT QNN_SDK_ROOT OR QNN_SDK_ROOT STREQUAL "")
    set(QNN_SDK_ROOT /usr)
endif()

message(STATUS "QNN_SDK_ROOT: ${QNN_SDK_ROOT}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QNN_SDK_ROOT}/include
    ${QNN_SDK_ROOT}/include/QNN
)

# Find system OpenCL
find_package(OpenCL REQUIRED)

# Common sources
set(COMMON_SOURCES
    qwen_vl_preprocessor.cpp
    qwen_vl_opencl_accelerator.cpp
)

# Runner
add_executable(veg_combined_runner veg_combined_runner.cpp ${COMMON_SOURCES})
target_link_libraries(veg_combined_runner PRIVATE OpenCL::OpenCL ${CMAKE_DL_LIBS} pthread)
set_target_properties(veg_combined_runner PROPERTIES OUTPUT_NAME "veg-combined-runner")

# Server
add_executable(veg_server_combined veg_server_combined.cpp ${COMMON_SOURCES})
target_link_libraries(veg_server_combined PRIVATE OpenCL::OpenCL ${CMAKE_DL_LIBS} pthread)
set_target_properties(veg_server_combined PROPERTIES OUTPUT_NAME "veg-server-combined")

# Client (no QNN/OpenCL dependency)
add_executable(veg_client veg_client.cpp)
target_link_libraries(veg_client PRIVATE ${CMAKE_DL_LIBS} pthread)
set_target_properties(veg_client PROPERTIES OUTPUT_NAME "veg-client")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
