#!/bin/bash
# build_and_run.sh — SCRFD 准确性测试: 编译 + 运行
#
# 完整管线: JPEG→NV12→GPU(OpenCL)→QNN(HTP)→ScrfdDecode→CSV
#
# 用法:
#   bash build_and_run.sh [model_path] [conf_thresh] [nms_thresh]
#
# 输出:
#   scrfd_results.csv (与图片同目录)
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
OUT="$SCRIPT_DIR/test_scrfd_accuracy"

echo "=========================================="
echo " SCRFD Accuracy Test — Full Pipeline"
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

echo "=== 编译 test_scrfd_accuracy ==="
echo "  源文件:"
echo "    $SCRIPT_DIR/test_scrfd_accuracy.cpp"
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
    "$SCRIPT_DIR/test_scrfd_accuracy.cpp" \
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

echo "=== 运行推理 ==="
echo ""
"$OUT" "$SCRIPT_DIR" "$KERNEL_DIR" "$MODEL" "$CONF" "$NMS"

echo ""
echo "=========================================="
echo " 结果文件"
echo "=========================================="
RESULT="$SCRIPT_DIR/scrfd_results.csv"
if [ -f "$RESULT" ]; then
    LINES=$(wc -l < "$RESULT")
    FACES=$((LINES - 1))
    echo "  CSV:    $RESULT"
    echo "  行数:   $LINES (含表头)"
    echo "  人脸数: $FACES"
    echo ""
    echo "--- 前 10 条检测结果 ---"
    head -11 "$RESULT"
else
    echo "  WARNING: $RESULT not found"
fi
