#!/bin/bash
# build_test.sh — QCS9075 板上编译 DmaBuffer 测试程序
#
# 用法: bash build_test.sh
# 前提: 在 mem_management/test/ 目录下执行, 或从项目根目录执行
#
# 编译产出: ./test_dma_buffer
# 运行: ./test_dma_buffer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/.."
OUT="$SCRIPT_DIR/test_dma_buffer"

echo "=== Building DmaBuffer test ==="
echo "  Source dir: $SRC_DIR"
echo "  Output:     $OUT"

g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include \
    "$SCRIPT_DIR/test_dma_buffer.cpp" \
    "$SRC_DIR/dma_buffer.cpp" \
    "$SRC_DIR/opencl_loader.cpp" \
    -ldl \
    -o "$OUT"

echo "=== Build OK ==="
echo ""
echo "Run:  $OUT"
echo ""

# 可选: 自动运行
if [[ "${1:-}" == "--run" ]]; then
    echo "=== Running test ==="
    "$OUT"
fi
