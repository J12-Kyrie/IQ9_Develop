#!/bin/bash
# E2E Pipeline Test: Producer → Consumer → Rule Process
# All 3 processes run in nohup background; logs written to /tmp/.
# Usage: bash scripts/run_e2e_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DMA_DIR="$APP_DIR/test/dma_buf_pipline"

PRODUCER="$APP_DIR/build/app/multi-cam-app"
CONSUMER="$DMA_DIR/dmabuf_consumer/build_local/bin/consumer_demo"
RULE="$DMA_DIR/rule_process/build_local/bin/rule_process"
PRODUCER_CONFIG="$APP_DIR/config/config.json"
RULE_CONFIG="$DMA_DIR/rule_process/config/rule_process_config.json"

INIT_SOCK="/tmp/dmabuf_ipc.sock"
RELAY_SOCK="/tmp/iq9_rule_relay.sock"
PRODUCER_LOG="/tmp/producer.log"
CONSUMER_LOG="/tmp/consumer.log"
RULE_LOG="/tmp/rule_process.log"

# ---- cleanup ----
echo "=== Cleanup ==="
pkill -9 -f multi-cam-app 2>/dev/null || true
pkill -9 -f consumer_demo 2>/dev/null || true
pkill -9 -f rule_process 2>/dev/null || true
sleep 1
rm -f "$INIT_SOCK" "$RELAY_SOCK"
mosquitto_pub -t 'iq9/scene_update' -n -r -h 127.0.0.1 2>/dev/null || true

# ---- validate ----
ok=1
for f in "$PRODUCER" "$CONSUMER" "$RULE" "$PRODUCER_CONFIG" "$RULE_CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f"
    ok=0
  fi
done
[[ "$ok" -eq 0 ]] && exit 1
echo "All binaries/configs found"

# ---- start ----
echo ""
echo "=== Step 1/3: Producer ==="
nohup "$PRODUCER" -c "$PRODUCER_CONFIG" > "$PRODUCER_LOG" 2>&1 &
PRODUCER_PID=$!
echo "  PID=$PRODUCER_PID  log=$PRODUCER_LOG"
sleep 2

echo "=== Step 2/3: Consumer ==="
cd "$(dirname "$CONSUMER")"
nohup "$CONSUMER" "$INIT_SOCK" "$RELAY_SOCK" 100 ./ppm_output/ 100 0 > "$CONSUMER_LOG" 2>&1 &
CONSUMER_PID=$!
echo "  PID=$CONSUMER_PID  log=$CONSUMER_LOG"
sleep 2

echo "=== Step 3/3: Rule Process (15s timeout) ==="
cd "$(dirname "$RULE")"
timeout 15 "$RULE" "$RULE_CONFIG" > "$RULE_LOG" 2>&1 || true
echo "  Rule process finished"

sleep 2

# ---- results ----
echo ""
echo "============================================"
echo "  Producer:  tail -5  $PRODUCER_LOG"
echo "  Consumer:  cat     $CONSUMER_LOG"
echo "  Rule:      grep -E 'WARN|ERROR|relay sent' $RULE_LOG"
echo "============================================"
echo ""

echo "--- Producer (last 5 lines) ---"
tail -5 "$PRODUCER_LOG"
echo ""
echo "--- Consumer ---"
cat "$CONSUMER_LOG"
echo ""
echo "--- Rule Process (WARN/ERROR/relay) ---"
grep -E 'WARN|ERROR|relay sent' "$RULE_LOG" || echo "(clean — no warnings)"
echo ""

# ---- cleanup ----
pkill -9 -f multi-cam-app 2>/dev/null || true
pkill -9 -f consumer_demo 2>/dev/null || true
echo "Done."
