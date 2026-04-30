cmake_minimum_required(VERSION 3.16)
project(iq9_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(validate_ppm validate_ppm.cpp)

set_target_properties(validate_ppm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
