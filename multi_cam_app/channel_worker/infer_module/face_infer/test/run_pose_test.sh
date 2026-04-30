#!/bin/bash
# run_pose_test.sh — 同一人不同姿态余弦相似度测试
#
# 流程: 选一张 baseline 图片 enroll → 全部图片 verify → 输出相似度报告 + 交叉矩阵
#
# 用法:
#   bash run_pose_test.sh <face_config.json> <baseline_image_path> [pic_dir]
#
# 示例:
#   bash run_pose_test.sh ../config/face_config_board.json /path/to/pic4test/base1/base1.png /path/to/pic4test
#   bash run_pose_test.sh ../config/face_config_board.json ./000001.jpg
#
# 参数:
#   face_config.json      — FaceProcessor 配置文件
#   baseline_image_path   — 基准图片的完整路径 (支持 .jpg/.jpeg/.png)
#   pic_dir               — 待验证图片目录 (默认: multi_cam_app/data/pic4test)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FACE_INFER_DIR="$SCRIPT_DIR/.."
MULTI_CAM_DIR="$FACE_INFER_DIR/../../../.."

BINARY="$SCRIPT_DIR/test_gallery_consistency"
GALLERY_JSON="$SCRIPT_DIR/gallery_pose_baseline.json"

# ---- 参数解析 ----
if [ $# -lt 2 ]; then
    echo "Usage: bash $0 <face_config.json> <baseline_image_path> [pic_dir]"
    echo ""
    echo "  face_config.json      FaceProcessor config file"
    echo "  baseline_image_path   Full path to baseline image (.jpg/.jpeg/.png)"
    echo "  pic_dir               Image directory (default: multi_cam_app/data/pic4test)"
    exit 1
fi

FACE_CONFIG="$1"
BASELINE_PATH="$2"
PIC_DIR="${3:-$MULTI_CAM_DIR/data/pic4test}"

# ---- 验证参数 ----
if [ ! -f "$FACE_CONFIG" ]; then
    echo "FATAL: Config file not found: $FACE_CONFIG"
    exit 1
fi

if [ ! -f "$BASELINE_PATH" ]; then
    echo "FATAL: Baseline image not found: $BASELINE_PATH"
    exit 1
fi

if [ ! -d "$PIC_DIR" ]; then
    echo "FATAL: Pic dir not found: $PIC_DIR"
    exit 1
fi

BASELINE_FNAME=$(basename "$BASELINE_PATH")

# ---- 临时 baseline 目录 ----
BASELINE_DIR="$SCRIPT_DIR/_pose_baseline_tmp"
trap 'rm -rf "$BASELINE_DIR"' EXIT

echo "============================================="
echo "  Pose Similarity Test"
echo "============================================="
echo ""
echo "  Config:     $FACE_CONFIG"
echo "  Baseline:   $BASELINE_PATH"
echo "  Pic dir:    $PIC_DIR"
echo "  Gallery:    $GALLERY_JSON"
echo ""

# ---- Step 1: 编译 (如果需要) ----
if [ ! -f "$BINARY" ]; then
    echo "=== Step 1: Build ==="
    bash "$SCRIPT_DIR/build_test_gallery_consistency.sh"
    echo ""
else
    echo "=== Step 1: Build (skipped, binary exists) ==="
    echo ""
fi

# ---- Step 2: 准备 baseline 目录 ----
echo "=== Step 2: Prepare baseline ==="
rm -rf "$BASELINE_DIR"
mkdir -p "$BASELINE_DIR"

# scan_jpg() 只扫描 .jpg/.jpeg, 如果 baseline 是 .png 则重命名为 .jpg
# OpenCV imread 按文件内容识别格式, 重命名不影响读取
EXT="${BASELINE_FNAME##*.}"
EXT_LOWER=$(echo "$EXT" | tr '[:upper:]' '[:lower:]')
if [ "$EXT_LOWER" = "png" ] || [ "$EXT_LOWER" = "bmp" ]; then
    COPY_NAME="${BASELINE_FNAME%.*}.jpg"
    cp "$BASELINE_PATH" "$BASELINE_DIR/$COPY_NAME"
    echo "  Copied $BASELINE_FNAME -> $BASELINE_DIR/$COPY_NAME (renamed for scan_jpg)"
else
    COPY_NAME="$BASELINE_FNAME"
    cp "$BASELINE_PATH" "$BASELINE_DIR/$COPY_NAME"
    echo "  Copied $BASELINE_FNAME -> $BASELINE_DIR/"
fi
echo ""

# ---- Step 3: Enroll baseline ----
echo "=== Step 3: Enroll baseline (1 image) ==="
"$BINARY" "$FACE_CONFIG" --enroll "$BASELINE_DIR" "$GALLERY_JSON"
ENROLL_RET=$?
if [ $ENROLL_RET -ne 0 ]; then
    echo ""
    echo "FATAL: Enroll phase failed (exit code $ENROLL_RET)"
    exit $ENROLL_RET
fi
echo ""

# ---- Step 4: Verify all images against baseline ----
echo "=== Step 4: Verify all images against baseline (with matrix) ==="
"$BINARY" "$FACE_CONFIG" --verify "$PIC_DIR" "$GALLERY_JSON" --matrix
VERIFY_RET=$?
echo ""

# ---- 最终状态 ----
echo "============================================="
if [ $VERIFY_RET -eq 0 ]; then
    echo "  POSE TEST COMPLETE"
else
    echo "  POSE TEST COMPLETE (some checks failed, exit code $VERIFY_RET)"
fi
echo ""
echo "  Baseline: $BASELINE_FNAME"
echo "  Gallery:  $GALLERY_JSON"
echo "  Tip: BEST_SIM column = cosine similarity to baseline"
echo "  Tip: Use log redirect: bash $0 ... 2>&1 | tee pose_test.log"
echo "============================================="

exit $VERIFY_RET
