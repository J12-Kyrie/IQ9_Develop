cmake_minimum_required(VERSION 3.16)
project(rule_process LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(ENABLE_MQTT "Enable MQTT support" ON)

find_package(PahoMQTT QUIET)
if(NOT PahoMQTT_FOUND AND ENABLE_MQTT)
  message(WARNING "PahoMQTT not found — disabling MQTT support (ENABLE_MQTT=OFF). "
                  "Install libpaho-mqtt-dev to enable MQTT.")
  set(ENABLE_MQTT OFF)
endif()

set(RULE_PROCESS_SOURCES
  src/main.cpp
  src/adapters/event_adapter.cpp
  src/core/RuleEngine.cpp
  src/core/Dispatcher.cpp
  src/core/Registry.cpp
  src/rules/DoorCollisionRule.cpp
  src/rules/LeftItemRule.cpp
  src/rules/OpenTrunkFaceRule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/FaceDatabase.cpp
)

if(ENABLE_MQTT)
  list(APPEND RULE_PROCESS_SOURCES
    src/mqtt/MqttAdapter.cpp
    src/mqtt/MqttSubscriber.cpp
    src/mqtt/MqttPublisher.cpp
  )
endif()

add_executable(rule_process ${RULE_PROCESS_SOURCES})

target_include_directories(rule_process PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/mqtt
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/../dmabuf_producer/include
)

if(ENABLE_MQTT)
  target_include_directories(rule_process PRIVATE ${PAHO_MQTT_INCLUDE_DIRS})
  target_compile_definitions(rule_process PRIVATE ENABLE_MQTT)
endif()

target_link_libraries(rule_process PRIVATE
  pthread
)

if(ENABLE_MQTT)
  target_link_libraries(rule_process PRIVATE ${PAHO_MQTT_LIBRARIES})
endif()

set_target_properties(rule_process PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
