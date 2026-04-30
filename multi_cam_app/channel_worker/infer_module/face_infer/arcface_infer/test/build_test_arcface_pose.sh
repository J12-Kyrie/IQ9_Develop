#!/bin/bash
# build_test_arcface_pose.sh — 编译 SCRFD + ArcFace 姿态相似度测试
#
# 链接: SCRFD QnnInferencer + ScrfdDecode + ArcFaceInfer + arcface QnnInferencer + OpenCV
# 不需要: FacePreprocess, DmaBuffer, OpenCL, mem_management
#
# 用法: bash build_test_arcface_pose.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCFACE_DIR="$SCRIPT_DIR/.."
FACE_INFER_DIR="$ARCFACE_DIR/.."
SCRFD_DIR="$FACE_INFER_DIR/scrfd_infer"
OUT="$SCRIPT_DIR/test_arcface_pose"

echo "============================================="
echo " ArcFace Pose Test (SCRFD aligned) — Build"
echo "============================================="
echo ""

# ---- Pre-flight checks ----
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
    "$SCRIPT_DIR/test_arcface_pose.cpp"
    "$ARCFACE_DIR/ArcFaceInfer.cpp"
    "$ARCFACE_DIR/QnnInferencer.cpp"
    "$SCRFD_DIR/QnnInferencer.cpp"
    "$SCRFD_DIR/ScrfdDecode.cpp"
)

for src in "${SRC_FILES[@]}"; do
    check_file "$src" "$(basename "$src")"
done

# Header-only dependency
check_file "$FACE_INFER_DIR/face_preprocess/inverse_affine.hpp" "inverse_affine.hpp"

# QNN runtime
check_file /usr/lib/libQnnHtp.so "libQnnHtp.so"
check_file /usr/lib/libQnnSystem.so "libQnnSystem.so"

# OpenCV
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
    -I"$ARCFACE_DIR" \
    -I"$FACE_INFER_DIR" \
    -I"$SCRFD_DIR" \
    $OPENCV_CFLAGS \
    "${SRC_FILES[@]}" \
    -ldl -lm \
    $OPENCV_LIBS \
    -o "$OUT"

echo "[OK] Build complete: $OUT"
echo "  Binary size: $(du -h "$OUT" | cut -f1)"
echo ""
echo "Usage:"
echo "  $OUT <scrfd_model> <arcface_model> --enroll <image_path> <gallery.json>"
echo "  $OUT <scrfd_model> <arcface_model> --verify <pic_dir> <gallery.json> [--matrix]"
