#!/bin/bash
# run_gallery_test.sh — 一键运行 Gallery 一致性测试
#
# 流程: 编译 → enroll (生成 gallery) → verify (对比验证) → 输出报告
#
# 用法:
#   bash run_gallery_test.sh [face_config.json] [pic_dir] [--matrix]
#
# 默认:
#   face_config.json = face_infer/config/face_config_board.json
#   pic_dir          = multi_cam_app/data/pic4test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FACE_INFER_DIR="$SCRIPT_DIR/.."
MULTI_CAM_DIR="$FACE_INFER_DIR/../../../.."

# ---- 参数解析 ----
FACE_CONFIG="${1:-$FACE_INFER_DIR/config/face_config_board.json}"
PIC_DIR="${2:-$MULTI_CAM_DIR/data/pic4test}"
GALLERY_JSON="$SCRIPT_DIR/gallery_test_output.json"
BINARY="$SCRIPT_DIR/test_gallery_consistency"

# 检查是否有 --matrix 参数
MATRIX_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--matrix" ]; then
        MATRIX_FLAG="--matrix"
    fi
done

echo "============================================="
echo "  Gallery Consistency Test — Full Run"
echo "============================================="
echo ""
echo "  Config:    $FACE_CONFIG"
echo "  Pic dir:   $PIC_DIR"
echo "  Gallery:   $GALLERY_JSON"
echo "  Matrix:    ${MATRIX_FLAG:-disabled}"
echo ""

# ---- Step 1: 编译 ----
echo "=== Step 1: Build ==="
bash "$SCRIPT_DIR/build_test_gallery_consistency.sh"
echo ""

# ---- Step 2: Enroll ----
echo "=== Step 2: Enroll (generate gallery) ==="
"$BINARY" "$FACE_CONFIG" --enroll "$PIC_DIR" "$GALLERY_JSON"
ENROLL_RET=$?
if [ $ENROLL_RET -ne 0 ]; then
    echo ""
    echo "FATAL: Enroll phase failed (exit code $ENROLL_RET)"
    exit $ENROLL_RET
fi
echo ""

# ---- Step 3: Verify ----
echo "=== Step 3: Verify (match against gallery) ==="
"$BINARY" "$FACE_CONFIG" --verify "$PIC_DIR" "$GALLERY_JSON" $MATRIX_FLAG
VERIFY_RET=$?
echo ""

# ---- 最终状态 ----
if [ $VERIFY_RET -eq 0 ]; then
    echo "============================================="
    echo "  RESULT: ALL TESTS PASSED"
    echo "============================================="
else
    echo "============================================="
    echo "  RESULT: SOME TESTS FAILED (exit code $VERIFY_RET)"
    echo "============================================="
fi

exit $VERIFY_RET
