#!/usr/bin/env bash
# Convert test PNG images to H.264 MP4 video for pipeline testing
#
# 用户将连续帧 PNG 图片放入:
#   multi_cam_app/channel_worker/deduper/test/data/frame_00001.png
#   multi_cam_app/channel_worker/deduper/test/data/frame_00002.png
#   ...
#
# 脚本输出:
#   multi_cam_app/channel_worker/deduper/test/data/test_video.mp4
#
# 要求:
#   - PNG 图片命名: frame_%05d.png (从 00001 开始)
#   - 所有图片尺寸一致
#   - 板上需安装 ffmpeg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT="${DATA_DIR}/test_video.mp4"

echo "========================================"
echo "=== Prepare Test Data ==="
echo "========================================"

# 检查图片
COUNT=$(ls -1 "${DATA_DIR}"/frame_*.png 2>/dev/null | wc -l)
if [ "$COUNT" -eq 0 ]; then
    echo "ERROR: No frame_*.png found in ${DATA_DIR}/"
    echo ""
    echo "Please place consecutive PNG frames as:"
    echo "  ${DATA_DIR}/frame_00001.png"
    echo "  ${DATA_DIR}/frame_00002.png"
    echo "  ..."
    echo ""
    echo "Tip: Extract frames from existing video:"
    echo "  ffmpeg -i input.mp4 -vframes 100 ${DATA_DIR}/frame_%05d.png"
    exit 1
fi

echo "Found ${COUNT} PNG frames in ${DATA_DIR}/"

# 检测第一张图片尺寸
FIRST=$(ls -1 "${DATA_DIR}"/frame_*.png | head -1)
echo "First frame: $(basename "$FIRST")"

# ffmpeg 编码: High profile + yuv420p (QCS9075 v4l2h264dec 要求)
echo "Encoding to H.264 (High profile, yuv420p, 30fps)..."
ffmpeg -y -framerate 30 \
    -i "${DATA_DIR}/frame_%05d.png" \
    -c:v libx264 -profile:v high -pix_fmt yuv420p \
    -an "$OUTPUT" 2>&1

echo ""
echo "Output: $OUTPUT"
ls -lh "$OUTPUT"
echo ""
echo "========================================"
echo "=== Done ==="
echo "========================================"
echo ""
echo "Run pipeline test:"
echo "  bash channel_worker/deduper/test/build_test_deduper.sh"
