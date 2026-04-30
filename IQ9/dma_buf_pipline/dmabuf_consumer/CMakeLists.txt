cmake_minimum_required(VERSION 3.16)
project(dmabuf_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static library from src/*.cpp
add_library(dmabuf_consumer_lib STATIC
    src/ipc_receiver.cpp
    src/consumer.cpp
)

target_include_directories(dmabuf_consumer_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../dmabuf_producer/include
)

target_include_directories(dmabuf_consumer_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dmabuf_consumer_lib PUBLIC pthread)

# Demo executable
add_executable(consumer_demo demo/consumer_demo.cpp)

target_link_libraries(consumer_demo PRIVATE dmabuf_consumer_lib)

set_target_properties(consumer_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
