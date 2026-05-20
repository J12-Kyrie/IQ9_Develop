cmake_minimum_required(VERSION 3.10)
project(VEGCombinedRunner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT QNN_SDK_ROOT OR QNN_SDK_ROOT STREQUAL "")
    message(FATAL_ERROR "QNN_SDK_ROOT is not set. Please set it via:\n"
                        "  cmake -DQNN_SDK_ROOT=/path/to/qnn/sdk ..")
endif()

message(STATUS "QNN_SDK_ROOT: ${QNN_SDK_ROOT}")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QNN_SDK_ROOT}/include
    ${QNN_SDK_ROOT}/include/QNN
)

# Source files (only runner — no server/client for single-image verification)
set(RUNNER_SOURCES
    veg_combined_runner.cpp
    qwen_vl_preprocessor.cpp
    qwen_vl_opencl_accelerator.cpp
)

add_executable(veg_combined_runner ${RUNNER_SOURCES})

# Link libraries for OE-Linux (not Android)
# Use system OpenCL from sysroot, dlopen for QNN, pthread for threading
target_link_libraries(veg_combined_runner
    PRIVATE
    OpenCL
    ${CMAKE_DL_LIBS}
    pthread
)

set_target_properties(veg_combined_runner PROPERTIES
    OUTPUT_NAME "veg-combined-runner"
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
