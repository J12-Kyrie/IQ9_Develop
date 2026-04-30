#!/bin/bash
# build_test_face_processor.sh — 在 QCS9075 板上编译和运行 FaceProcessor 端到端测试
# 编译链接: FaceProcessor + 全部子模块 (FacePreprocess + SCRFD QNN + ScrfdDecode
#           + ArcFace QNN + mem_management)
#
# 配置从 face_config.json 读取 (模型路径、OpenCL kernel 路径、检测参数)
# 用法: bash build_test_face_processor.sh [face_config.json]
#   默认使用 face_infer/config/face_config.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FACE_INFER_DIR="$SCRIPT_DIR/.."
FACE_PRE_DIR="$FACE_INFER_DIR/face_preprocess"
SCRFD_DIR="$FACE_INFER_DIR/scrfd_infer"
ARCFACE_DIR="$FACE_INFER_DIR/arcface_infer"
MEM_DIR="$FACE_INFER_DIR/mem_management"
OUT="$SCRIPT_DIR/test_face_processor"

# 配置文件: 优先使用命令行参数, 否则默认 face_infer/config/face_config.json
FACE_CONFIG="${1:-$FACE_INFER_DIR/config/face_config.json}"

echo "============================================="
echo " FaceProcessor 端到端测试 — 编译 & 运行"
echo "============================================="
echo ""

# ---- 前置检查 ----
echo "=== 前置检查 ==="
CHECKS_OK=true

check_file() {
    if [ -f "$1" ]; then
        echo "  [OK] $2: $1"
    else
        echo "  [FAIL] $2 not found: $1"
        CHECKS_OK=false
    fi
}

check_file "$FACE_CONFIG"             "face_config.json"
check_file /usr/lib/libQnnHtp.so      "QNN HTP backend"
check_file /usr/lib/libQnnSystem.so   "QNN System library"

# 检查源文件
SRC_FILES=(
    "$SCRIPT_DIR/test_face_processor.cpp"
    "$FACE_INFER_DIR/FaceProcessor.cpp"
    "$FACE_PRE_DIR/FacePreprocess.cpp"
    "$SCRFD_DIR/QnnInferencer.cpp"
    "$SCRFD_DIR/ScrfdDecode.cpp"
    "$ARCFACE_DIR/ArcFaceInfer.cpp"
    "$ARCFACE_DIR/QnnInferencer.cpp"
    "$MEM_DIR/opencl_loader.cpp"
    "$MEM_DIR/dma_buffer.cpp"
)
for src in "${SRC_FILES[@]}"; do
    check_file "$src" "$(basename "$src")"
done

if [ "$CHECKS_OK" = false ]; then
    echo ""
    echo "FATAL: 前置检查失败, 请确认文件路径"
    exit 1
fi
echo ""

# ---- 编译 ----
echo "=== 编译 test_face_processor ==="
echo "  编译单元: ${#SRC_FILES[@]} 个源文件"
echo ""

g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include \
    -I"$FACE_INFER_DIR" \
    "${SRC_FILES[@]}" \
    -ldl -lm \
    -o "$OUT"

echo "[OK] 编译完成: $OUT"
echo "  Binary size: $(du -h "$OUT" | cut -f1)"
echo ""

# ---- 运行 ----
echo "=== 运行测试 ==="
echo "  Config file: $FACE_CONFIG"
echo ""

"$OUT" "$FACE_CONFIG"

echo ""
echo "=== 内存泄漏检查 (二次运行) ==="
MEM_BEFORE=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
echo "  MemAvailable before: ${MEM_BEFORE} kB"

"$OUT" "$FACE_CONFIG" > /dev/null 2>&1

MEM_AFTER=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
echo "  MemAvailable after:  ${MEM_AFTER} kB"
DIFF=$((MEM_BEFORE - MEM_AFTER))
echo "  Delta: ${DIFF} kB (should be < 1000 kB)"
if [ "$DIFF" -gt 5000 ]; then
    echo "  [WARN] Possible memory leak (>5MB delta)"
else
    echo "  [OK] No significant memory leak"
fi
