#!/bin/bash
# Run Aggregator test binary.
# Group A (pure C++): always runs
# Group B (MQTT integration): requires mosquitto broker, skipped if unavailable
#
# Usage:
#   bash aggregator/test/run_test_aggregator.sh
#   bash aggregator/test/run_test_aggregator.sh --broker-ip 192.168.1.100
#   bash aggregator/test/run_test_aggregator.sh --group A
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$SCRIPT_DIR/build/test_aggregator"
BROKER_IP="127.0.0.1"
BROKER_PORT="1883"
GROUP="all"
STARTED_MOSQUITTO=0

while [ $# -gt 0 ]; do
  case "$1" in
    --broker-ip)   BROKER_IP="$2"; shift 2 ;;
    --broker-port) BROKER_PORT="$2"; shift 2 ;;
    --group)       GROUP="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [ ! -f "$BINARY" ]; then
  echo "ERROR: $BINARY not found."
  echo "  Run: bash aggregator/test/build_test_aggregator.sh"
  exit 1
fi

echo "============================================"
echo "  Aggregator Test Runner"
echo "  Group: $GROUP"
echo "============================================"
echo ""

# --- Group A: always run ---
if [ "$GROUP" = "A" ] || [ "$GROUP" = "all" ]; then
  echo "--- Running Group A (unit tests, no broker needed) ---"
  "$BINARY" --group A
  A_RC=$?
  echo ""
else
  A_RC=0
fi

# --- Group B: check/start mosquitto ---
if [ "$GROUP" = "B" ] || [ "$GROUP" = "all" ]; then
  echo "--- Preparing Group B (MQTT integration) ---"
  echo "  Broker: $BROKER_IP:$BROKER_PORT"

  BROKER_OK=0

  if [ "$BROKER_IP" = "127.0.0.1" ] || [ "$BROKER_IP" = "localhost" ]; then
    if pgrep -x mosquitto >/dev/null 2>&1; then
      echo "  mosquitto already running (PID $(pgrep -x mosquitto))"
      BROKER_OK=1
    else
      echo "  mosquitto not running."
      if command -v mosquitto &>/dev/null; then
        echo "  Starting mosquitto on port $BROKER_PORT..."
        mosquitto -p "$BROKER_PORT" -d 2>/dev/null || true
        sleep 1
        if pgrep -x mosquitto >/dev/null 2>&1; then
          echo "  mosquitto started (PID $(pgrep -x mosquitto))"
          STARTED_MOSQUITTO=1
          BROKER_OK=1
        else
          echo "  WARNING: Failed to start mosquitto"
        fi
      else
        echo "  WARNING: mosquitto not installed (apt install mosquitto)"
      fi
    fi
  else
    BROKER_OK=1
  fi

  echo ""

  if [ "$BROKER_OK" = "1" ]; then
    "$BINARY" --group B --broker-ip "$BROKER_IP" --broker-port "$BROKER_PORT"
    B_RC=$?
  else
    echo "  SKIPPING Group B: mosquitto broker not available"
    B_RC=0
  fi

  # Cleanup
  if [ "$STARTED_MOSQUITTO" = "1" ]; then
    echo ""
    echo "Stopping mosquitto (we started it)..."
    killall mosquitto 2>/dev/null || true
  fi
else
  B_RC=0
fi

# --- Summary ---
echo ""
echo "============================================"
echo "  Final Results"
echo "============================================"
if [ "$GROUP" = "A" ] || [ "$GROUP" = "all" ]; then
  echo "  Group A (unit):        $([ $A_RC -eq 0 ] && echo 'PASS' || echo 'FAIL')"
fi
if [ "$GROUP" = "B" ] || [ "$GROUP" = "all" ]; then
  if [ "$BROKER_OK" = "1" ] 2>/dev/null; then
    echo "  Group B (integration): $([ $B_RC -eq 0 ] && echo 'PASS' || echo 'FAIL')"
  else
    echo "  Group B (integration): SKIPPED"
  fi
fi
echo "============================================"

exit $(( A_RC + B_RC ))
