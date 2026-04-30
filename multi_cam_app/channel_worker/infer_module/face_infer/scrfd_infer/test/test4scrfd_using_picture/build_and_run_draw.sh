#!/bin/bash
# build_and_run_draw.sh — SCRFD 可视化测试: 编译 + 运行
#
# 完整管线: JPEG→NV12→GPU(OpenCL)→QNN(HTP)→ScrfdDecode→绘图→JPEG
#
# 用法:
#   bash build_and_run_draw.sh [model_path] [conf_thresh] [nms_thresh]
#
# 输出:
#   <image_dir>/output/*.jpg (标注后的图片)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRFD_INFER_DIR="$SCRIPT_DIR/../.."
FACE_INFER_DIR="$SCRFD_INFER_DIR/.."
FACE_PRE_DIR="$FACE_INFER_DIR/face_preprocess"
MEM_DIR="$FACE_INFER_DIR/mem_management"
KERNEL_DIR="$FACE_PRE_DIR/kernels"
MULTI_CAM_DIR="$FACE_INFER_DIR/../../.."
MODEL="${1:-$MULTI_CAM_DIR/data/models/face/det_10g_qcs9075.bin}"
CONF="${2:-0.5}"
NMS="${3:-0.4}"
OUT="$SCRIPT_DIR/test_scrfd_draw"

echo "=========================================="
echo " SCRFD Draw Test — Visual Bbox Overlay"
echo "=========================================="
echo ""

echo "=== 前置检查 ==="
[ -f "$MODEL" ]                 || { echo "FATAL: model $MODEL not found"; exit 1; }
[ -f /usr/lib/libQnnHtp.so ]    || { echo "FATAL: libQnnHtp.so not found"; exit 1; }
[ -f /usr/lib/libQnnSystem.so ] || { echo "FATAL: libQnnSystem.so not found"; exit 1; }
[ -d "$KERNEL_DIR" ]            || { echo "FATAL: kernel dir $KERNEL_DIR not found"; exit 1; }

IMG_COUNT=$(ls "$SCRIPT_DIR"/*.jpg 2>/dev/null | wc -l || echo 0)
echo "  Model:      $MODEL"
echo "  Kernel dir: $KERNEL_DIR"
echo "  Images:     $IMG_COUNT .jpg file(s) in $SCRIPT_DIR"
echo "  Conf:       $CONF"
echo "  NMS:        $NMS"
echo ""

if [ "$IMG_COUNT" -eq 0 ]; then
    echo "FATAL: No .jpg files found in $SCRIPT_DIR"
    exit 1
fi

# OpenCV flags
OPENCV_CFLAGS=$(pkg-config --cflags opencv4)
OPENCV_LIBS="-lopencv_imgcodecs -lopencv_imgproc -lopencv_core"

echo "=== 编译 test_scrfd_draw ==="
echo "  源文件:"
echo "    $SCRIPT_DIR/test_scrfd_draw.cpp"
echo "    $SCRFD_INFER_DIR/QnnInferencer.cpp"
echo "    $SCRFD_INFER_DIR/ScrfdDecode.cpp"
echo "    $FACE_PRE_DIR/FacePreprocess.cpp"
echo "    $MEM_DIR/opencl_loader.cpp"
echo "    $MEM_DIR/dma_buffer.cpp"
echo "  OpenCV: $OPENCV_CFLAGS $OPENCV_LIBS"
echo ""

g++ -std=c++17 -O2 -Wall -Wextra \
    $OPENCV_CFLAGS \
    -I/usr/include \
    "$SCRIPT_DIR/test_scrfd_draw.cpp" \
    "$SCRFD_INFER_DIR/QnnInferencer.cpp" \
    "$SCRFD_INFER_DIR/ScrfdDecode.cpp" \
    "$FACE_PRE_DIR/FacePreprocess.cpp" \
    "$MEM_DIR/opencl_loader.cpp" \
    "$MEM_DIR/dma_buffer.cpp" \
    $OPENCV_LIBS \
    -ldl -lm \
    -o "$OUT"

echo "[OK] 编译完成: $OUT"
echo ""

echo "=== 运行推理 + 绘图 ==="
echo ""
"$OUT" "$SCRIPT_DIR" "$KERNEL_DIR" "$MODEL" "$CONF" "$NMS"

echo ""
echo "=========================================="
echo " 输出结果"
echo "=========================================="
OUT_DIR="$SCRIPT_DIR/output"
if [ -d "$OUT_DIR" ]; then
    OUT_COUNT=$(ls "$OUT_DIR"/*.jpg 2>/dev/null | wc -l || echo 0)
    echo "  输出目录:   $OUT_DIR"
    echo "  标注图片数: $OUT_COUNT"
    echo ""
    echo "--- 输出文件列表 ---"
    ls -lh "$OUT_DIR"/*.jpg 2>/dev/null || echo "  (no files)"
else
    echo "  WARNING: output directory $OUT_DIR not found"
fi
