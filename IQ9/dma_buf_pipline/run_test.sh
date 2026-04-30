#!/bin/bash
set -e

TEST_VIDEO=/mnt/workspace/gst-plugins-qti-oss-imsdk.lnx.2.0.0.r2-rel/multi_cam_app/data/media/camera_id_0.mp4
PRODUCER=/mnt/workspace/develop/IQ9/dmabuf_producer/build/bin/producer_demo
CONSUMER=/mnt/workspace/develop/IQ9/dmabuf_consumer/build/bin/consumer_demo
SOCKET=/tmp/dmabuf_ipc.sock
OUTPUT_DIR=/tmp/ppm_output
NUM_FRAMES=${1:-10}

rm -rf "$OUTPUT_DIR" "$SOCKET"
mkdir -p "$OUTPUT_DIR"

echo "[test] starting producer (waiting for consumer)..."
$PRODUCER "$TEST_VIDEO" "$SOCKET" "$NUM_FRAMES" > /tmp/producer.log 2>&1 &
PRODUCER_PID=$!
sleep 2

echo "[test] starting consumer..."
$CONSUMER "$SOCKET" "$OUTPUT_DIR" "$NUM_FRAMES" > /tmp/consumer.log 2>&1 &
CONSUMER_PID=$!

echo "[test] waiting for completion (timeout 60s)..."
timeout 60 tail --pid=$PRODUCER_PID -f /dev/null 2>/dev/null || true
timeout 10 tail --pid=$CONSUMER_PID -f /dev/null 2>/dev/null || true

echo ""
echo "=== PRODUCER LOG ==="
cat /tmp/producer.log || true
echo ""
echo "=== CONSUMER LOG ==="
cat /tmp/consumer.log || true
echo ""
echo "=== PPM OUTPUT ==="
ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "(no files)"
echo ""
echo "[test] done"
