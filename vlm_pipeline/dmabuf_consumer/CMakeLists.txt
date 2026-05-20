cmake_minimum_required(VERSION 3.16)
project(dmabuf_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_VLM "Enable VLM inference support" ON)

# ---- VLM dependency discovery ----
if(ENABLE_VLM)
    # Find libjpeg-turbo via pkg-config, then manual fallback
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TURBOJPEG QUIET libturbojpeg)
    endif()
    if(NOT TURBOJPEG_FOUND)
        find_path(TURBOJPEG_INCLUDE_DIR turbojpeg.h
            PATHS /usr/include /usr/local/include /opt/libjpeg-turbo/include)
        find_library(TURBOJPEG_LIBRARY NAMES turbojpeg
            PATHS /usr/lib /usr/local/lib /opt/libjpeg-turbo/lib)
        if(TURBOJPEG_INCLUDE_DIR AND TURBOJPEG_LIBRARY)
            set(TURBOJPEG_FOUND TRUE)
            set(TURBOJPEG_INCLUDE_DIRS ${TURBOJPEG_INCLUDE_DIR})
            set(TURBOJPEG_LIBRARIES ${TURBOJPEG_LIBRARY})
        endif()
    endif()

    # Find libcurl
    find_package(CURL QUIET)
    if(NOT CURL_FOUND)
        if(PkgConfig_FOUND)
            pkg_check_modules(CURL QUIET libcurl)
        endif()
    endif()

    # Find nlohmann/json (header-only)
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS /usr/include /usr/local/include
              ${CMAKE_CURRENT_SOURCE_DIR}/../rule_process/include)
    if(NLOHMANN_JSON_INCLUDE_DIR)
        set(NLOHMANN_JSON_FOUND TRUE)
    endif()

    if(NOT TURBOJPEG_FOUND)
        message(FATAL_ERROR "libjpeg-turbo not found. Install: apt install libjpeg-turbo8-dev")
    endif()
    if(NOT CURL_FOUND)
        message(FATAL_ERROR "libcurl not found. Install: apt install libcurl4-openssl-dev")
    endif()
    if(NOT NLOHMANN_JSON_FOUND)
        message(FATAL_ERROR "nlohmann/json not found. Install: apt install nlohmann-json3-dev")
    endif()
endif()

# Static library from src/*.cpp
add_library(dmabuf_consumer_lib STATIC
    src/ipc_receiver.cpp
    src/consumer.cpp
)

target_include_directories(dmabuf_consumer_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(dmabuf_consumer_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dmabuf_consumer_lib PUBLIC pthread)

# VLM sources
set(VLM_SOURCES "")
if(ENABLE_VLM)
    list(APPEND VLM_SOURCES
        src/vlm/jpeg_encoder.cpp
        src/vlm/vlm_client.cpp
        src/vlm/jsonl_logger.cpp
        src/vlm/thread_pool.cpp
    )
endif()

# Demo executable
add_executable(consumer_demo demo/consumer_demo.cpp ${VLM_SOURCES})

target_link_libraries(consumer_demo PRIVATE dmabuf_consumer_lib)

if(ENABLE_VLM)
    target_include_directories(consumer_demo PRIVATE
        ${TURBOJPEG_INCLUDE_DIRS}
        ${CURL_INCLUDE_DIRS}
        ${NLOHMANN_JSON_INCLUDE_DIR}
    )
    target_link_libraries(consumer_demo PRIVATE
        ${TURBOJPEG_LIBRARIES}
        ${CURL_LIBRARIES}
    )
    target_compile_definitions(consumer_demo PRIVATE ENABLE_VLM)
endif()

set_target_properties(consumer_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
