#!/bin/bash
# build_test_affine.sh — 在 QCS9075 板上编译和运行 Umeyama 逆仿射测试
# 编译链接: inverse_affine.hpp (header-only) + FacePreprocess + mem_management
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREPROCESS_DIR="$SCRIPT_DIR/.."
FACE_INFER_DIR="$PREPROCESS_DIR/.."
MEM_DIR="$FACE_INFER_DIR/mem_management"
KERNEL_DIR="$PREPROCESS_DIR/kernels"
OUT="$SCRIPT_DIR/test_inverse_affine"

echo "=== 前置检查 ==="
[ -d "$KERNEL_DIR" ] || { echo "FATAL: kernel dir $KERNEL_DIR not found"; exit 1; }

echo "  Kernel dir: $KERNEL_DIR"
echo ""

echo "=== 编译 test_inverse_affine ==="
echo "  源文件:"
echo "    $SCRIPT_DIR/test_inverse_affine.cpp"
echo "    $PREPROCESS_DIR/FacePreprocess.cpp"
echo "    $MEM_DIR/opencl_loader.cpp"
echo "    $MEM_DIR/dma_buffer.cpp"

g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include \
    "$SCRIPT_DIR/test_inverse_affine.cpp" \
    "$PREPROCESS_DIR/FacePreprocess.cpp" \
    "$MEM_DIR/opencl_loader.cpp" \
    "$MEM_DIR/dma_buffer.cpp" \
    -ldl -lm \
    -o "$OUT"
echo "[OK] 编译完成: $OUT"
echo ""

echo "=== 运行测试 ==="
"$OUT" "$KERNEL_DIR"
