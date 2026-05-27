cmake_minimum_required(VERSION 3.14)
project(opencl_kernel_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

# Kernel.cl path — configurable, no hardcoding
set(KERNEL_CL_DIR "${CMAKE_SOURCE_DIR}/../solution" CACHE PATH "Directory containing kernel.cl")
set(KERNEL_TOML "${CMAKE_SOURCE_DIR}/../kernel.toml" CACHE FILEPATH "Path to kernel.toml")

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark OpenCL::OpenCL)
target_compile_options(benchmark PRIVATE -O2 -Wall -Wextra)

target_compile_definitions(benchmark PRIVATE
    KERNEL_PATH="${KERNEL_CL_DIR}/kernel.cl"
    TOML_PATH="${KERNEL_TOML}"
)

# Link per-branch CPU baseline if present
if(EXISTS "${CMAKE_SOURCE_DIR}/../solution/cpu_baseline.cpp")
    target_sources(benchmark PRIVATE "${CMAKE_SOURCE_DIR}/../solution/cpu_baseline.cpp")
    message(STATUS "Linked cpu_baseline.cpp from solution/")
endif()
