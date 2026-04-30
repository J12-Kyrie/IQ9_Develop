#!/bin/bash
# build_test.sh — 在 QCS9075 板上编译和运行 FacePreprocess 独立测试
# 编译时链接 Step 1 源文件 (opencl_loader.cpp, dma_buffer.cpp)
# 以及 FacePreprocess.cpp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/../kernels"
MEM_DIR="$SCRIPT_DIR/../../mem_management"
PREPROCESS_DIR="$SCRIPT_DIR/.."
OUT="$SCRIPT_DIR/test_gpu_preprocess"

echo "=== 编译 test_gpu_preprocess ==="
echo "  源文件:"
echo "    $SCRIPT_DIR/test_gpu_preprocess.cpp"
echo "    $PREPROCESS_DIR/FacePreprocess.cpp"
echo "    $MEM_DIR/opencl_loader.cpp"
echo "    $MEM_DIR/dma_buffer.cpp"

g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include \
    "$SCRIPT_DIR/test_gpu_preprocess.cpp" \
    "$PREPROCESS_DIR/FacePreprocess.cpp" \
    "$MEM_DIR/opencl_loader.cpp" \
    "$MEM_DIR/dma_buffer.cpp" \
    -ldl \
    -lm \
    -o "$OUT"
echo "[OK] 编译完成: $OUT"
echo ""

echo "=== 运行测试 ==="
echo "Kernel 目录: $KERNEL_DIR"
"$OUT" "$KERNEL_DIR"

echo ""
echo "=== 内存泄漏检查 ==="
echo "运行前 MemAvailable:"
grep MemAvailable /proc/meminfo
echo "运行后 MemAvailable (应无持续下降):"
"$OUT" "$KERNEL_DIR" > /dev/null 2>&1
grep MemAvailable /proc/meminfo
