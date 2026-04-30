#!/bin/bash
# build_test_gallery_consistency.sh — 编译 Gallery 一致性测试
#
# 链接: FaceProcessor + 全部子模块 + OpenCV (imread)
#
# 用法: bash build_test_gallery_consistency.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FACE_INFER_DIR="$SCRIPT_DIR/.."
FACE_PRE_DIR="$FACE_INFER_DIR/face_preprocess"
SCRFD_DIR="$FACE_INFER_DIR/scrfd_infer"
ARCFACE_DIR="$FACE_INFER_DIR/arcface_infer"
MEM_DIR="$FACE_INFER_DIR/mem_management"
OUT="$SCRIPT_DIR/test_gallery_consistency"

echo "============================================="
echo " Gallery Consistency Test — Build"
echo "============================================="
echo ""

# ---- 前置检查 ----
echo "=== Pre-flight checks ==="
CHECKS_OK=true

check_file() {
    if [ -f "$1" ]; then
        echo "  [OK] $2: $1"
    else
        echo "  [FAIL] $2 not found: $1"
        CHECKS_OK=false
    fi
}

SRC_FILES=(
    "$SCRIPT_DIR/test_gallery_consistency.cpp"
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

# 检查 OpenCV
if pkg-config --exists opencv4 2>/dev/null; then
    echo "  [OK] opencv4 (pkg-config)"
elif pkg-config --exists opencv 2>/dev/null; then
    echo "  [OK] opencv (pkg-config, legacy)"
else
    echo "  [WARN] opencv4 not found via pkg-config, will try default paths"
fi

if [ "$CHECKS_OK" = false ]; then
    echo ""
    echo "FATAL: Pre-flight checks failed"
    exit 1
fi
echo ""

# ---- OpenCV flags ----
if pkg-config --exists opencv4 2>/dev/null; then
    OPENCV_CFLAGS=$(pkg-config --cflags opencv4)
    OPENCV_LIBS=$(pkg-config --libs opencv4)
elif pkg-config --exists opencv 2>/dev/null; then
    OPENCV_CFLAGS=$(pkg-config --cflags opencv)
    OPENCV_LIBS=$(pkg-config --libs opencv)
else
    OPENCV_CFLAGS=""
    OPENCV_LIBS="-lopencv_core -lopencv_imgcodecs -lopencv_imgproc"
fi

echo "=== Build ==="
echo "  Sources: ${#SRC_FILES[@]} files"
echo "  OpenCV CFLAGS: $OPENCV_CFLAGS"
echo "  OpenCV LIBS:   $OPENCV_LIBS"
echo ""

g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include \
    -I"$FACE_INFER_DIR" \
    $OPENCV_CFLAGS \
    "${SRC_FILES[@]}" \
    -ldl -lm \
    $OPENCV_LIBS \
    -o "$OUT"

echo "[OK] Build complete: $OUT"
echo "  Binary size: $(du -h "$OUT" | cut -f1)"
echo ""
echo "Usage:"
echo "  $OUT <face_config.json> --enroll <pic_dir> <gallery.json>"
echo "  $OUT <face_config.json> --verify <pic_dir> <gallery.json> [--matrix]"
