#!/bin/bash
# build_test_arcface.sh — 在 QCS9075 板上编译和运行 ArcFace 推理端到端测试
# 编译链接: ArcFaceInfer + QnnInferencer(arcface本地副本) + FacePreprocess + mem_management
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCFACE_DIR="$SCRIPT_DIR/.."
FACE_INFER_DIR="$ARCFACE_DIR/.."
FACE_PRE_DIR="$FACE_INFER_DIR/face_preprocess"
MEM_DIR="$FACE_INFER_DIR/mem_management"
KERNEL_DIR="$FACE_PRE_DIR/kernels"
MULTI_CAM_DIR="$FACE_INFER_DIR/../../.."
MODEL="$MULTI_CAM_DIR/data/models/face/w600k_r50_qcs9075.bin"
OUT="$SCRIPT_DIR/test_arcface_inference"

echo "=== 前置检查 ==="
[ -f "$MODEL" ]                 || { echo "FATAL: model not found: $MODEL"; exit 1; }
[ -f /usr/lib/libQnnHtp.so ]    || { echo "FATAL: libQnnHtp.so not found"; exit 1; }
[ -f /usr/lib/libQnnSystem.so ] || { echo "FATAL: libQnnSystem.so not found"; exit 1; }
[ -d "$KERNEL_DIR" ]            || { echo "FATAL: kernel dir not found: $KERNEL_DIR"; exit 1; }

echo "  Model:      $MODEL"
echo "  Kernel dir: $KERNEL_DIR"
echo ""

echo "=== 编译 test_arcface_inference ==="
echo "  源文件:"
echo "    $SCRIPT_DIR/test_arcface_inference.cpp"
echo "    $ARCFACE_DIR/ArcFaceInfer.cpp"
echo "    $ARCFACE_DIR/QnnInferencer.cpp"
echo "    $FACE_PRE_DIR/FacePreprocess.cpp"
echo "    $MEM_DIR/opencl_loader.cpp"
echo "    $MEM_DIR/dma_buffer.cpp"

g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include \
    "$SCRIPT_DIR/test_arcface_inference.cpp" \
    "$ARCFACE_DIR/ArcFaceInfer.cpp" \
    "$ARCFACE_DIR/QnnInferencer.cpp" \
    "$FACE_PRE_DIR/FacePreprocess.cpp" \
    "$MEM_DIR/opencl_loader.cpp" \
    "$MEM_DIR/dma_buffer.cpp" \
    -ldl -lm \
    -o "$OUT"
echo "[OK] 编译完成: $OUT"
echo ""

echo "=== 运行测试 ==="
"$OUT" "$KERNEL_DIR" "$MODEL"

echo ""
echo "=== 内存泄漏检查 ==="
echo "运行前 MemAvailable:"
grep MemAvailable /proc/meminfo
echo "运行后 MemAvailable (应无持续下降):"
"$OUT" "$KERNEL_DIR" "$MODEL" > /dev/null 2>&1
grep MemAvailable /proc/meminfo
