#!/bin/bash
# Build standalone Aggregator test binary for QCS9075 board validation.
# Usage: bash aggregator/test/build_test_aggregator.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$(cd "$AGG_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== Build: test_aggregator ==="
echo "  APP_DIR: $APP_DIR"
echo ""

# --- Step 1: Find libpaho-mqtt3c.so ---
echo "--- Searching for Paho MQTT C library ---"
PAHO_LIB=""

for path in /usr/lib/libpaho-mqtt3c.so \
            /usr/lib/aarch64-linux-gnu/libpaho-mqtt3c.so \
            /usr/local/lib/libpaho-mqtt3c.so \
            /usr/lib64/libpaho-mqtt3c.so; do
  if [ -f "$path" ]; then
    PAHO_LIB="$path"
    echo "  [fixed-path] Found: $PAHO_LIB"
    break
  fi
done

if [ -z "$PAHO_LIB" ]; then
  echo "  [fixed-path] Not found in well-known paths"
  echo "  [ldconfig] Searching ldconfig cache..."
  LDCONFIG_HIT=$(ldconfig -p 2>/dev/null | grep 'libpaho-mqtt3c\.so' | head -1 | sed 's/.*=> //' || true)
  if [ -n "$LDCONFIG_HIT" ] && [ -f "$LDCONFIG_HIT" ]; then
    PAHO_LIB="$LDCONFIG_HIT"
    echo "  [ldconfig] Found: $PAHO_LIB"
  else
    echo "  [ldconfig] Not found"
  fi
fi

if [ -z "$PAHO_LIB" ]; then
  echo "  [find] Searching /usr /opt /lib ..."
  FIND_HIT=$(find /usr /opt /lib -name 'libpaho-mqtt3c.so*' -type f 2>/dev/null | head -1 || true)
  if [ -n "$FIND_HIT" ]; then
    PAHO_LIB="$FIND_HIT"
    echo "  [find] Found: $PAHO_LIB"
  else
    echo "  [find] Not found"
  fi
fi

if [ -z "$PAHO_LIB" ]; then
  echo ""
  echo "ERROR: libpaho-mqtt3c.so not found."
  echo "  Install: sudo apt install libpaho-mqtt-dev"
  exit 1
fi

# --- Step 2: Find MQTTClient.h ---
echo ""
echo "--- Searching for MQTTClient.h ---"
PAHO_INC=""

for inc in /usr/include /usr/local/include; do
  if [ -f "$inc/MQTTClient.h" ]; then
    PAHO_INC="$inc"
    echo "  [fixed-path] Found: $PAHO_INC/MQTTClient.h"
    break
  fi
done

if [ -z "$PAHO_INC" ]; then
  echo "  [fixed-path] Not found"
  echo "  [find] Searching /usr /opt ..."
  FIND_INC=$(find /usr /opt -name 'MQTTClient.h' -type f 2>/dev/null | head -1 || true)
  if [ -n "$FIND_INC" ]; then
    PAHO_INC="$(dirname "$FIND_INC")"
    echo "  [find] Found: $FIND_INC"
  else
    echo "  [find] Not found"
    echo "ERROR: MQTTClient.h not found. Install libpaho-mqtt-dev."
    exit 1
  fi
fi

echo ""
echo "  Library: $PAHO_LIB"
echo "  Header:  $PAHO_INC/MQTTClient.h"
echo ""

# --- Step 3: Compile ---
echo "--- Compiling test_aggregator ---"
mkdir -p "$BUILD_DIR"

g++ -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter -O2 \
  -I"$APP_DIR" \
  -I"$APP_DIR/aggregator/concurrentqueue" \
  -I"$PAHO_INC" \
  -o "$BUILD_DIR/test_aggregator" \
  "$SCRIPT_DIR/test_aggregator.cpp" \
  "$AGG_DIR/Aggregator.cpp" \
  "$APP_DIR/utils/JsonlWriter.cpp" \
  "$APP_DIR/utils/mqtt/MqttClient.cpp" \
  "$APP_DIR/utils/mqtt/MqttPublisher.cpp" \
  "$APP_DIR/utils/mqtt/MqttSubscriber.cpp" \
  "$PAHO_LIB" \
  -lpthread

echo ""
echo "=== Build succeeded ==="
echo "  Binary: $BUILD_DIR/test_aggregator"
echo "  Run:    bash aggregator/test/run_test_aggregator.sh"
