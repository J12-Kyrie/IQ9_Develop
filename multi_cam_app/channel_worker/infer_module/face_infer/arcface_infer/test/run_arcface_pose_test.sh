#!/bin/bash
# run_arcface_pose_test.sh — 一键运行 SCRFD + ArcFace 姿态相似度测试
#
# 流程: 编译 → enroll baseline → verify all images → 输出报告 + 交叉矩阵
#
# 所有路径已硬编码, 直接运行:
#   bash run_arcface_pose_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$SCRIPT_DIR/test_arcface_pose"
GALLERY_JSON="$SCRIPT_DIR/gallery_arcface_pose.json"

# ---- 硬编码路径 ----
WORKSPACE="/mnt/workspace/gst-plugins-qti-oss-imsdk.lnx.2.0.0.r2-rel"
SCRFD_MODEL="$WORKSPACE/multi_cam_app/data/models/face/det_2.5g_qcs9075.bin"
ARCFACE_MODEL="$WORKSPACE/multi_cam_app/data/models/face/w600k_r50_qcs9075.bin"
BASELINE_PATH="$WORKSPACE/multi_cam_app/data/pic4test/base1/base1.png"
PIC_DIR="$WORKSPACE/multi_cam_app/data/pic4test/validate1"

# ---- 验证路径 ----
echo "============================================="
echo "  ArcFace Pose Similarity Test — Full Run"
echo "  (SCRFD aligned pipeline)"
echo "============================================="
echo ""
echo "  SCRFD model:  $SCRFD_MODEL"
echo "  ArcFace model: $ARCFACE_MODEL"
echo "  Baseline:     $BASELINE_PATH"
echo "  Pic dir:      $PIC_DIR"
echo "  Gallery:      $GALLERY_JSON"
echo ""

CHECKS_OK=true
for f in "$SCRFD_MODEL" "$ARCFACE_MODEL" "$BASELINE_PATH"; do
    if [ ! -f "$f" ]; then
        echo "FATAL: File not found: $f"
        CHECKS_OK=false
    fi
done
if [ ! -d "$PIC_DIR" ]; then
    echo "FATAL: Directory not found: $PIC_DIR"
    CHECKS_OK=false
fi
if [ "$CHECKS_OK" = false ]; then
    exit 1
fi

# ---- Step 1: Build (if needed) ----
if [ ! -f "$BINARY" ]; then
    echo "=== Step 1: Build ==="
    bash "$SCRIPT_DIR/build_test_arcface_pose.sh"
    echo ""
else
    echo "=== Step 1: Build (skipped, binary exists) ==="
    echo ""
fi

# ---- Step 2: Enroll baseline ----
echo "=== Step 2: Enroll baseline ==="
"$BINARY" "$SCRFD_MODEL" "$ARCFACE_MODEL" --enroll "$BASELINE_PATH" "$GALLERY_JSON"
ENROLL_RET=$?
if [ $ENROLL_RET -ne 0 ]; then
    echo ""
    echo "FATAL: Enroll phase failed (exit code $ENROLL_RET)"
    exit $ENROLL_RET
fi
echo ""

# ---- Step 3: Verify all images against baseline ----
echo "=== Step 3: Verify all images (with matrix) ==="
"$BINARY" "$SCRFD_MODEL" "$ARCFACE_MODEL" --verify "$PIC_DIR" "$GALLERY_JSON" --matrix
VERIFY_RET=$?
echo ""

# ---- Final status ----
echo "============================================="
if [ $VERIFY_RET -eq 0 ]; then
    echo "  POSE TEST COMPLETE — ALL OK"
else
    echo "  POSE TEST COMPLETE — SOME FAILURES (exit code $VERIFY_RET)"
fi
echo ""
echo "  Baseline: $(basename "$BASELINE_PATH")"
echo "  Gallery:  $GALLERY_JSON"
echo "  Tip: BEST_SIM column = cosine similarity to baseline"
echo "  Tip: Save log: bash $0 2>&1 | tee arcface_pose_test.log"
echo "============================================="

exit $VERIFY_RET
