#!/bin/bash
# Extract all frames from camera_id_1_10fps.mp4 into test/data/frames/
#
# Usage (on remote device):
#   cd <project_root>/multi_cam_app/channel_worker/deduper/test
#   bash extract_frames.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO="${SCRIPT_DIR}/../../../data/media/camera_id_1_10fps.mp4"
OUT_DIR="${SCRIPT_DIR}/data/frames"

if [ ! -f "$VIDEO" ]; then
    echo "ERROR: video not found: $VIDEO"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Extracting frames from: $VIDEO"
echo "Output directory:       $OUT_DIR"

ffmpeg -i "$VIDEO" -vsync 0 "${OUT_DIR}/frame_%05d.png"

FRAME_COUNT=$(ls -1 "${OUT_DIR}"/frame_*.png 2>/dev/null | wc -l)
echo "Done. Extracted ${FRAME_COUNT} frames to ${OUT_DIR}"
