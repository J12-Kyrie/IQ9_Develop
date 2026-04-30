#!/bin/bash
# Run multi_cam_app with face + deduper integration and verify JSONL output.
# Usage: bash scripts/run_face_integration.sh [config_path]
#
# Default config: config/config.json (face.enabled=true)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$APP_DIR/build"
BINARY="$BUILD_DIR/app/multi-cam-app"
CONFIG="${1:-$APP_DIR/config/config.json}"
OUTPUT_DIR="$APP_DIR/output"

if [ ! -f "$BINARY" ]; then
  echo "ERROR: Binary not found: $BINARY"
  echo "  Run: bash scripts/build.sh"
  exit 1
fi

# 本地编出的插件（与 build.sh 的 build/plugins/* 一致）；若已 sudo 装到系统目录可省略
P="$APP_DIR/build/plugins"
FACEINFER_LIB="${P}/gstfaceinfer"
DEDUPER_LIB="${P}/deduper"
export GST_PLUGIN_PATH="${FACEINFER_LIB}:${DEDUPER_LIB}:${P}/frameoffload:${P}/msgagg:${P}/timingmark:${GST_PLUGIN_PATH:-}"

echo "=== Running multi-cam-app (face + deduper integration) ==="
echo "  Binary: $BINARY"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo "  GST_PLUGIN_PATH: $GST_PLUGIN_PATH"
echo ""

# Quick plugin check
PLUGIN_OK=1
if [ -f "$FACEINFER_LIB/libgstqtifaceinfer.so" ]; then
    echo "  qtifaceinfer .so: OK"
else
    echo "  WARNING: libgstqtifaceinfer.so not found in $FACEINFER_LIB"
    PLUGIN_OK=0
fi
if [ -f "$DEDUPER_LIB/libgstqtideduper.so" ]; then
    echo "  qtideduper .so: OK"
else
    echo "  WARNING: libgstqtideduper.so not found in $DEDUPER_LIB"
    PLUGIN_OK=0
fi

# Also check system path (may have been deployed there)
if gst-inspect-1.0 qtimetamux > /dev/null 2>&1; then
    echo "  qtimetamux: OK (gst-inspect)"
else
    echo "  WARNING: qtimetamux not found via gst-inspect"
    PLUGIN_OK=0
fi
echo ""

# Clean previous output
rm -f "$OUTPUT_DIR"/channel_*.jsonl 2>/dev/null || true

# Run with deduper + metamux debug output
cd "$APP_DIR"
GST_DEBUG=qtideduper:4,qtimetamux:3,qtiframeoffload:4 "$BINARY" -c "$CONFIG" 2>&1 | tee "$APP_DIR/run_face_integration.log"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Run completed (exit code 0) ==="
else
    echo "=== Run failed (exit code $EXIT_CODE) ==="
fi

# --- Verify JSONL output ---
echo ""
echo "=== JSONL output verification ==="
FACE_FOUND=0
for f in "$OUTPUT_DIR"/channel_*.jsonl; do
    if [ -f "$f" ]; then
        LINES=$(wc -l < "$f")
        FACE_LINES="$(grep -c '"face"' "$f" 2>/dev/null)" || FACE_LINES=0
        TRACK_LINES="$(grep -c '"track_id"' "$f" 2>/dev/null)" || TRACK_LINES=0
        echo "  $(basename "$f"): $LINES lines, $FACE_LINES with face, $TRACK_LINES with track_id"
        if [ "$FACE_LINES" -gt 0 ]; then
            FACE_FOUND=1
        fi
    fi
done

if [ "$FACE_FOUND" -eq 1 ]; then
    echo "=== PASS: Face data found in JSONL output ==="
else
    echo "=== WARN: No face data found in JSONL output ==="
    echo "  Check face.enabled in config and FaceMeta passthrough through metamux"
fi

# --- Deduper stats from log ---
echo ""
echo "=== Deduper stats ==="
DEDUPER_LINES=$(grep -c "qtideduper" "$APP_DIR/run_face_integration.log" 2>/dev/null) || DEDUPER_LINES=0
echo "  Deduper log lines: $DEDUPER_LINES"
if [ "$DEDUPER_LINES" -gt 0 ]; then
    echo "  Sample deduper output:"
    grep "qtideduper" "$APP_DIR/run_face_integration.log" | tail -5
fi

# --- FrameOffload stats from log ---
echo ""
echo "=== FrameOffload stats ==="
FO_LINES=$(grep -c "qtiframeoffload\|FrameOffload\|frames_processed" "$APP_DIR/run_face_integration.log" 2>/dev/null) || FO_LINES=0
echo "  FrameOffload log lines: $FO_LINES"
if [ "$FO_LINES" -gt 0 ]; then
    echo "  FrameOffload output:"
    grep "qtiframeoffload\|FrameOffload\|frames_processed" "$APP_DIR/run_face_integration.log"
fi

exit $EXIT_CODE
