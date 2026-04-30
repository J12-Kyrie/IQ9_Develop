#!/usr/bin/env bash
# Build and run deduper tests (Group A + optional Group B)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# deduper/test/ → deduper/ → channel_worker/ → multi_cam_app/
APP_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEST_DIR="$SCRIPT_DIR"
BUILD_DIR="${APP_DIR}/build_test_deduper"

# 在 cd 之前解析路径（支持相对路径）
VIDEO="${TEST_VIDEO:-${TEST_DIR}/data/test_video.mp4}"
MODEL="${TEST_MODEL:-${APP_DIR}/data/models/yolov11/yolov11_det.bin}"
LABELS="${TEST_LABELS:-${APP_DIR}/data/models/yolov11/labels.txt}"
# 相对路径 → 绝对路径（cd 后仍有效）
[[ "$VIDEO"  != /* ]] && VIDEO="$(pwd)/$VIDEO"
[[ "$MODEL"  != /* ]] && MODEL="$(pwd)/$MODEL"
[[ "$LABELS" != /* ]] && LABELS="$(pwd)/$LABELS"

echo "========================================"
echo "=== Build Deduper Tests ==="
echo "========================================"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$TEST_DIR" -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc) VERBOSE=1 2>&1

echo ""
echo "========================================"
echo "=== Run Group A: Unit Tests ==="
echo "========================================"
./test_deduper_unit
UNIT_RC=$?

echo ""
echo "========================================"
echo "=== Group B: Pipeline Test ==="
echo "========================================"

if [ ! -f "$VIDEO" ]; then
    echo "WARNING: test video not found: $VIDEO"
    echo "  Run 'bash channel_worker/deduper/test/prepare_test_data.sh' first to create test video"
    echo "  Or set TEST_VIDEO environment variable"
    echo "  Skipping Group B"
    echo ""
    echo "========================================"
    echo "=== Group A result: $([ $UNIT_RC -eq 0 ] && echo 'PASS' || echo 'FAIL') ==="
    echo "========================================"
    exit $UNIT_RC
fi

if [ ! -f "$MODEL" ]; then
    echo "WARNING: model not found: $MODEL"
    echo "  Set TEST_MODEL environment variable"
    echo "  Skipping Group B"
    exit $UNIT_RC
fi

echo "Video:  $VIDEO"
echo "Model:  $MODEL"
echo "Labels: $LABELS"
echo ""

GST_DEBUG=qtideduper:6 ./test_deduper_pipeline "$VIDEO" "$MODEL" "$LABELS"
PIPE_RC=$?

echo ""
echo "========================================"
echo "=== Results ==="
echo "========================================"
echo "Group A (unit):     $([ $UNIT_RC -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Group B (pipeline): $([ $PIPE_RC -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "========================================"

exit $(( UNIT_RC + PIPE_RC ))
