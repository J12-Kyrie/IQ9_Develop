#!/bin/bash
set -e

TEST_VIDEO=${TEST_VIDEO:-/mnt/workspace/gst-plugins-qti-oss-imsdk.lnx.2.0.0.r2-rel/multi_cam_app/data/media/camera_id_0_10fps.mp4}
JSONL_PATH=${JSONL_PATH:-/mnt/workspace/gst-plugins-qti-oss-imsdk.lnx.2.0.0.r2-rel/multi_cam_app/output/channel_0.jsonl}
PRODUCER=/mnt/workspace/develop/IQ9/dmabuf_producer/build/bin/producer_demo
CONSUMER=/mnt/workspace/develop/IQ9/dmabuf_consumer/build/bin/consumer_demo
RULE_PROCESS=/mnt/workspace/develop/IQ9/rule_process/build/bin/nvsci_rule_process
RULE_CONFIG=/mnt/workspace/develop/IQ9/rule_process/config/rule_process_config.json
NV12_TEST=/mnt/workspace/develop/IQ9/dmabuf_producer/build/bin/test_nv12_to_rgb
PPM_VALIDATOR=/mnt/workspace/develop/IQ9/test/build/bin/validate_ppm
INIT_SOCKET=/tmp/dmabuf_ipc.sock
RELAY_SOCKET=/tmp/iq9_rule_relay.sock
MQTT_BROKER=${MQTT_BROKER:-tcp://localhost:1883}
OUTPUT_DIR=/tmp/ppm_output
NUM_FRAMES=${NUM_FRAMES:-300}

cleanup() {
    echo ""
    echo "=== CLEANUP ==="
    [ -n "$RULE_PID" ] && kill "$RULE_PID" 2>/dev/null && echo "[cleanup] killed rule_process ($RULE_PID)"
    [ -n "$CONSUMER_PID" ] && kill "$CONSUMER_PID" 2>/dev/null && echo "[cleanup] killed consumer ($CONSUMER_PID)"
    [ -n "$PRODUCER_PID" ] && kill "$PRODUCER_PID" 2>/dev/null && echo "[cleanup] killed producer ($PRODUCER_PID)"
    sleep 1
    rm -f "$INIT_SOCKET" "$RELAY_SOCKET"
}
trap cleanup EXIT

rm -rf "$OUTPUT_DIR" "$INIT_SOCKET" "$RELAY_SOCKET"
mkdir -p "$OUTPUT_DIR"

echo "=== NV12->RGB UNIT TEST ==="
if [ -x "$NV12_TEST" ]; then
    $NV12_TEST
    NV12_EXIT=$?
    if [ $NV12_EXIT -ne 0 ]; then
        echo "=== NV12 UNIT TEST: FAIL (exit=$NV12_EXIT) ==="
        exit 1
    fi
    echo "=== NV12 UNIT TEST: PASS ==="
else
    echo "[warn] test_nv12_to_rgb not found at $NV12_TEST, skipping"
fi

echo ""
echo "=== VERIFYING MOSQUITTO ==="
if ! command -v mosquitto_pub >/dev/null 2>&1; then
    echo "[error] mosquitto_pub not found, install mosquitto-clients"
    exit 1
fi
if ! mosquitto_pub -t iq9/test -m "ping" 2>/dev/null; then
    echo "[warn] mosquitto broker not responding, attempting to start..."
    mosquitto -d -c /etc/mosquitto/mosquitto.conf 2>/dev/null || mosquitto -d 2>/dev/null || true
    sleep 1
    if ! mosquitto_pub -t iq9/test -m "ping" 2>/dev/null; then
        echo "[error] mosquitto broker still not responding"
        exit 1
    fi
fi
echo "[ok] mosquitto broker responding"

echo ""
echo "=== DETECTION-DRIVEN 3-PROCESS STRESS TEST ==="
echo "[info] video=$TEST_VIDEO"
echo "[info] jsonl=$JSONL_PATH"
echo "[info] mqtt=$MQTT_BROKER"
echo "[info] num_frames=$NUM_FRAMES"

# 1. Producer first: binds init UDS, blocks on AcceptClient()
echo ""
echo "[stress] starting producer (waiting for consumer)..."
$PRODUCER "$TEST_VIDEO" "$JSONL_PATH" "$INIT_SOCKET" "$MQTT_BROKER" "iq9/scene_update" "iq9/frame_done" "$NUM_FRAMES" \
    > /tmp/producer.log 2>&1 &
PRODUCER_PID=$!
sleep 2

# 2. Consumer second: creates relay UDS and blocks on accept() for rule_process
#    (does NOT connect to producer yet — waits for rule_process first)
echo "[stress] starting consumer (waiting for rule_process on relay)..."
$CONSUMER "$INIT_SOCKET" "$RELAY_SOCKET" "$NUM_FRAMES" "$OUTPUT_DIR" 10 0 \
    > /tmp/consumer.log 2>&1 &
CONSUMER_PID=$!
sleep 2

# 3. rule_process third: subscribes MQTT first, then connects to relay UDS
#    (relay connect unblocks consumer, which then connects to producer → pipeline starts)
echo "[stress] starting rule_process..."
$RULE_PROCESS "$RULE_CONFIG" \
    > /tmp/rule_process.log 2>&1 &
RULE_PID=$!
sleep 2

echo "[stress] all processes started (producer=$PRODUCER_PID, consumer=$CONSUMER_PID, rule=$RULE_PID)"

# Mid-run snapshot
sleep 5
echo ""
echo "=== MID-RUN SNAPSHOT (t=5s) ==="
for NAME_PID in "producer:$PRODUCER_PID" "consumer:$CONSUMER_PID" "rule_process:$RULE_PID"; do
    NAME=${NAME_PID%%:*}
    PID=${NAME_PID##*:}
    if [ -d /proc/$PID ]; then
        FD_COUNT=$(ls /proc/$PID/fd 2>/dev/null | wc -l)
        RSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo "N/A")
        echo "[$NAME] fd_count=$FD_COUNT  RSS=${RSS}kB"
    else
        echo "[$NAME] (exited)"
    fi
done

# Wait for producer to finish (it drives the pipeline)
echo ""
echo "[stress] waiting for completion (timeout 180s)..."
timeout 180 tail --pid=$PRODUCER_PID -f /dev/null 2>/dev/null || true
sleep 2
timeout 30 tail --pid=$CONSUMER_PID -f /dev/null 2>/dev/null || true

# Kill rule_process (it runs forever in MQTT loop)
kill "$RULE_PID" 2>/dev/null || true
wait "$RULE_PID" 2>/dev/null || true

PROD_EXIT=0
CONS_EXIT=0
wait $PRODUCER_PID 2>/dev/null || PROD_EXIT=$?
wait $CONSUMER_PID 2>/dev/null || CONS_EXIT=$?

echo ""
echo "=== EXIT STATUS ==="
echo "[producer] exit_code=$PROD_EXIT"
echo "[consumer] exit_code=$CONS_EXIT"

echo ""
echo "=== PRODUCER LOG (last 30 lines) ==="
tail -30 /tmp/producer.log 2>/dev/null || echo "(empty)"

echo ""
echo "=== CONSUMER LOG (last 30 lines) ==="
tail -30 /tmp/consumer.log 2>/dev/null || echo "(empty)"

echo ""
echo "=== RULE_PROCESS LOG (last 30 lines) ==="
tail -30 /tmp/rule_process.log 2>/dev/null || echo "(empty)"

echo ""
echo "=== FRAME COUNT VERIFICATION ==="
PROD_SUBMITTED=$(grep -oP 'submitted=\K[0-9]+' /tmp/producer.log 2>/dev/null | tail -1)
PROD_SUBMITTED=${PROD_SUBMITTED:-0}
PROD_SKIPPED=$(grep -oP 'skipped=\K[0-9]+' /tmp/producer.log 2>/dev/null | tail -1)
PROD_SKIPPED=${PROD_SKIPPED:-0}
MQTT_RELEASED=$(grep -oP 'mqtt_released=\K[0-9]+' /tmp/producer.log 2>/dev/null | tail -1)
MQTT_RELEASED=${MQTT_RELEASED:-0}
UDS_RELEASED=$(grep -oP 'uds_released=\K[0-9]+' /tmp/producer.log 2>/dev/null | tail -1)
UDS_RELEASED=${UDS_RELEASED:-0}
CONS_FRAMES=$(grep -c '\[consumer\] frame' /tmp/consumer.log 2>/dev/null || true)
CONS_FRAMES=${CONS_FRAMES:-0}
RELAY_SENT=$(grep -c 'relay sent' /tmp/rule_process.log 2>/dev/null || true)
RELAY_SENT=${RELAY_SENT:-0}

echo "[producer] submitted=$PROD_SUBMITTED skipped=$PROD_SKIPPED mqtt_released=$MQTT_RELEASED uds_released=$UDS_RELEASED"
echo "[consumer] frames_received=$CONS_FRAMES"
echo "[rule_process] relay_sent=$RELAY_SENT"

echo ""
echo "=== IOCTL SYNC LATENCY ==="
PROD_SYNC_LINE=$(grep 'ioctl_sync_write_us' /tmp/producer.log 2>/dev/null | tail -1 || true)
CONS_SYNC_LINE=$(grep 'ioctl_sync_read_us' /tmp/consumer.log 2>/dev/null | tail -1 || true)
if [ -n "$PROD_SYNC_LINE" ]; then
    echo "$PROD_SYNC_LINE"
else
    echo "[warn] producer sync latency summary not found"
fi
if [ -n "$CONS_SYNC_LINE" ]; then
    echo "$CONS_SYNC_LINE"
else
    echo "[warn] consumer sync latency summary not found"
fi

echo ""
echo "=== SLOT OFFSET FORMULA CHECK ==="
OFFSET_OK=$(grep -c 'offset_check=ok' /tmp/consumer.log 2>/dev/null || true)
OFFSET_BAD=$(grep -c 'offset_check=mismatch' /tmp/consumer.log 2>/dev/null || true)
echo "[consumer] offset_check_ok=$OFFSET_OK offset_check_mismatch=$OFFSET_BAD"
if [ "$OFFSET_BAD" -gt 0 ]; then
    echo "[FAIL] slot_id*slot_stride formula mismatch detected"
    PASS=false
elif [ "$OFFSET_OK" -gt 0 ]; then
    echo "[ok] slot_id*slot_stride mapping verified in consumer logs"
else
    echo "[warn] no offset_check lines found in consumer logs"
fi

echo ""
echo "=== SELECTIVE SUBMIT VERIFICATION ==="
PASS=true

if [ "$PROD_SUBMITTED" -gt 0 ]; then
    echo "[ok] producer submitted frames ($PROD_SUBMITTED)"
else
    echo "[FAIL] producer submitted 0 frames"
    PASS=false
fi

if [ "$PROD_SKIPPED" -gt 0 ]; then
    echo "[ok] producer skipped non-detection frames ($PROD_SKIPPED)"
else
    echo "[warn] producer skipped 0 frames (all frames had detections?)"
fi

if [ "$CONS_FRAMES" -gt 0 ]; then
    echo "[ok] consumer received alert frames ($CONS_FRAMES)"
else
    echo "[FAIL] consumer received 0 frames (no alerts triggered)"
    PASS=false
fi

if [ "$CONS_FRAMES" -lt "$PROD_SUBMITTED" ] 2>/dev/null; then
    echo "[ok] filtering active: consumer ($CONS_FRAMES) < producer ($PROD_SUBMITTED)"
else
    echo "[warn] no filtering detected: consumer=$CONS_FRAMES producer=$PROD_SUBMITTED"
fi

echo ""
echo "=== SLOT ACCOUNTING ==="
ACCOUNTED=$((MQTT_RELEASED + CONS_FRAMES))
echo "[info] mqtt_released=$MQTT_RELEASED + consumer_alerts=$CONS_FRAMES = accounted=$ACCOUNTED vs submitted=$PROD_SUBMITTED"
if [ "$ACCOUNTED" -eq "$PROD_SUBMITTED" ] 2>/dev/null; then
    echo "[ok] all slots accounted for (no leaks)"
else
    echo "[warn] slot mismatch: accounted=$ACCOUNTED != submitted=$PROD_SUBMITTED"
    PASS=false
fi
echo "[info] uds_released=$UDS_RELEASED (GLib timer caught $UDS_RELEASED of $CONS_FRAMES alert releases; rest reclaimed inside SubmitFrame)"

echo ""
echo "=== PPM OUTPUT ==="
PPM_COUNT=$(ls "$OUTPUT_DIR"/*.ppm 2>/dev/null | wc -l)
echo "ppm_files=$PPM_COUNT"
ls -la "$OUTPUT_DIR/" 2>/dev/null | head -10

echo ""
echo "=== PPM VALIDATION ==="
PPM_VALID=0
if [ -x "$PPM_VALIDATOR" ] && [ "$PPM_COUNT" -gt 0 ]; then
    $PPM_VALIDATOR "$OUTPUT_DIR"
    PPM_VALID=$?
    if [ $PPM_VALID -eq 0 ]; then
        echo "=== PPM VALIDATION: PASS ==="
    else
        echo "=== PPM VALIDATION: FAIL ==="
        PASS=false
    fi
else
    if [ ! -x "$PPM_VALIDATOR" ]; then
        echo "[warn] validate_ppm not found at $PPM_VALIDATOR, skipping"
    else
        echo "[warn] no PPM files to validate"
    fi
fi

echo ""
echo "=== LEAK CHECK (post-exit) ==="
if [ -e "$INIT_SOCKET" ]; then
    echo "[warn] init socket still exists: $INIT_SOCKET"
else
    echo "[ok] init socket cleaned up"
fi
if [ -e "$RELAY_SOCKET" ]; then
    echo "[warn] relay socket still exists: $RELAY_SOCKET"
else
    echo "[ok] relay socket cleaned up"
fi

echo ""
if [ "$PROD_EXIT" -eq 0 ] && [ "$CONS_EXIT" -eq 0 ] && [ "$PROD_SUBMITTED" -gt 0 ] && [ "$CONS_FRAMES" -gt 0 ] && [ "$PASS" = true ]; then
    echo "=== RESULT: PASS (submitted=$PROD_SUBMITTED alerts=$CONS_FRAMES skipped=$PROD_SKIPPED) ==="
else
    echo "=== RESULT: FAIL ==="
    echo "  producer_exit=$PROD_EXIT consumer_exit=$CONS_EXIT submitted=$PROD_SUBMITTED alerts=$CONS_FRAMES pass=$PASS"
fi
