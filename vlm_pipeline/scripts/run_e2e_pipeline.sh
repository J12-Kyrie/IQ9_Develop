#!/bin/bash
# E2E Pipeline Test: multi-cam-app -> rule_process -> consumer_demo (VLM)
# All processes run in nohup background; logs written to /tmp/.
# Usage: bash scripts/run_e2e_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# multi-cam-app paths
MULTI_CAM_APP="/mnt/workspace/develop/multi_cam_app"
PRODUCER="$MULTI_CAM_APP/build/app/multi-cam-app"
PRODUCER_CONFIG="$MULTI_CAM_APP/config/config.json"

# VLM pipeline paths (local to vlm_pipeline)
CONSUMER="$APP_DIR/dmabuf_consumer/build_local/bin/consumer_demo"
RULE="$APP_DIR/rule_process/build_local/bin/rule_process"
RULE_CONFIG="$APP_DIR/rule_process/config/rule_process_config.json"

INIT_SOCK="/tmp/dmabuf_ipc.sock"
RELAY_SOCK="/tmp/iq9_rule_relay.sock"
PRODUCER_LOG="/tmp/producer.log"
CONSUMER_LOG="/tmp/consumer.log"
RULE_LOG="/tmp/rule_process.log"

# VLM configuration
VLM_ENABLED=true
VLM_PORT=8000
VLM_JSONL="/tmp/vlm_responses.jsonl"
CONSUMER_VLM_ARGS="--vlm-enabled --vlm-url http://127.0.0.1:$VLM_PORT --vlm-jsonl $VLM_JSONL --vlm-threads 2 --vlm-drop-policy drop-newest"

# ---- cleanup ----
cleanup() {
    echo ""
    echo "=== Cleanup ==="
    # Send SIGTERM to consumer first for graceful pool drain
    if [ -n "${CONSUMER_PID:-}" ]; then
        kill -TERM "$CONSUMER_PID" 2>/dev/null || true
        wait "$CONSUMER_PID" 2>/dev/null || true
        echo "[cleanup] consumer stopped (SIGTERM)"
    fi
    pkill -9 -f multi-cam-app 2>/dev/null || true
    pkill -9 -f consumer_demo 2>/dev/null || true
    pkill -9 -f rule_process 2>/dev/null || true
    sleep 2
    rm -f "$INIT_SOCK" "$RELAY_SOCK"
    mosquitto_pub -t 'iq9/scene_update' -n -r -h 127.0.0.1 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Cleanup ==="
pkill -9 -f multi-cam-app 2>/dev/null || true
pkill -9 -f consumer_demo 2>/dev/null || true
pkill -9 -f rule_process 2>/dev/null || true
sleep 2
rm -f "$INIT_SOCK" "$RELAY_SOCK"
mosquitto_pub -t 'iq9/scene_update' -n -r -h 127.0.0.1 2>/dev/null || true

# ---- install custom GStreamer plugins ----
echo "=== Installing GStreamer plugins ==="
GST_DIR=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
PLUGIN_DIR="$MULTI_CAM_APP/build/plugins"
for p in msgagg faceinfer frameoffload deduper timingmark; do
  src="$PLUGIN_DIR/$p/libgstqti${p}.so"
  if [[ -f "$src" ]]; then
    sudo cp "$src" "$GST_DIR/"
  fi
done
rm -f ~/.cache/gstreamer-1.0/registry.*.bin
echo "Plugins installed, registry flushed"

# ---- validate ----
ok=1
for f in "$PRODUCER" "$PRODUCER_CONFIG" "$CONSUMER" "$RULE" "$RULE_CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f"
    ok=0
  fi
done
[[ "$ok" -eq 0 ]] && exit 1
echo "All binaries/configs found"

# ---- start ----
echo ""
echo "=== Step 1/4: Producer (multi-cam-app) ==="
nohup "$PRODUCER" -c "$PRODUCER_CONFIG" > "$PRODUCER_LOG" 2>&1 &
PRODUCER_PID=$!
echo "  PID=$PRODUCER_PID  log=$PRODUCER_LOG"
sleep 2

echo "=== Step 2/4: VLM Server ==="
if $VLM_ENABLED; then
    if curl -s --max-time 2 http://127.0.0.1:$VLM_PORT/v1/models > /dev/null 2>&1; then
        echo "  VLM server already running on port $VLM_PORT"
    else
        echo "  [WARN] VLM server not detected on port $VLM_PORT"
        echo "  Please start it manually: cd /mnt/workspace/bundle_iq9 && bash start.sh"
        echo "  Continuing anyway (VLM calls will fail until server is up)"
    fi
else
    echo "  [skip] VLM disabled"
fi

echo "=== Step 3/4: Consumer ==="
cd "$(dirname "$CONSUMER")"
if $VLM_ENABLED; then
    nohup "$CONSUMER" "$INIT_SOCK" "$RELAY_SOCK" 100 ./ppm_output/ 100 0 \
        $CONSUMER_VLM_ARGS > "$CONSUMER_LOG" 2>&1 &
else
    nohup "$CONSUMER" "$INIT_SOCK" "$RELAY_SOCK" 100 ./ppm_output/ 100 0 > "$CONSUMER_LOG" 2>&1 &
fi
CONSUMER_PID=$!
echo "  PID=$CONSUMER_PID  log=$CONSUMER_LOG"
sleep 2

echo "=== Step 4/4: Rule Process (60s timeout) ==="
cd "$(dirname "$RULE")"
timeout 60 "$RULE" "$RULE_CONFIG" > "$RULE_LOG" 2>&1 || true
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
tail -5 "$PRODUCER_LOG" 2>/dev/null || echo "(no producer log)"
echo ""
echo "--- Consumer ---"
cat "$CONSUMER_LOG" 2>/dev/null || echo "(no consumer log)"
echo ""
echo "--- Rule Process (rule triggers) ---"
grep -oP 'rule=[^ ]+' "$RULE_LOG" 2>/dev/null | sort | uniq -c | sort -rn || echo "(no rules triggered)"
echo ""
echo "--- Rule Process (WARN/ERROR/relay) ---"
grep -E 'WARN|ERROR|relay sent' "$RULE_LOG" 2>/dev/null || echo "(clean -- no warnings)"
echo ""

if $VLM_ENABLED; then
    echo "--- VLM Responses ---"
    if [[ -f "$VLM_JSONL" ]]; then
        echo "Total responses: $(wc -l < "$VLM_JSONL")"
        head -3 "$VLM_JSONL" | python3 -m json.tool 2>/dev/null || head -3 "$VLM_JSONL"
    else
        echo "(no VLM responses logged)"
    fi
    echo ""
fi

echo "Done."
