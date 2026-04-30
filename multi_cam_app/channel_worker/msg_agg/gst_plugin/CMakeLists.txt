cmake_minimum_required(VERSION 3.16)
project(gst_plugin_msgagg LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GST      REQUIRED gstreamer-1.0>=1.16)
pkg_check_modules(GST_BASE REQUIRED gstreamer-base-1.0>=1.16)
pkg_check_modules(JSON     REQUIRED json-glib-1.0)

set(PLUGIN_TARGET gstqtimsgagg)

add_library(${PLUGIN_TARGET} SHARED
    gstmsgagg.cpp
)

target_include_directories(${PLUGIN_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GST_INCLUDE_DIRS}
    ${GST_BASE_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIRS}
)

target_compile_definitions(${PLUGIN_TARGET} PRIVATE
    PACKAGE="qtimsgagg"
)

target_compile_options(${PLUGIN_TARGET} PRIVATE
    -Wall -Wextra -Werror
    -Wno-unused-parameter
)

target_link_libraries(${PLUGIN_TARGET} PRIVATE
    ${GST_LIBRARIES}
    ${GST_BASE_LIBRARIES}
    ${JSON_LIBRARIES}
)
