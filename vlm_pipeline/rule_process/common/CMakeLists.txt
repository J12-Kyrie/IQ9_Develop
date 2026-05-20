cmake_minimum_required(VERSION 3.16)
project(nvsci_pipeline_common)

add_library(nvsci_pipeline_common INTERFACE)

target_include_directories(nvsci_pipeline_common INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
