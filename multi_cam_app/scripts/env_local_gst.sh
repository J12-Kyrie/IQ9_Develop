# Source this file in your shell to use locally built GStreamer plugins without sudo copy:
#   source "$(dirname "$0")/env_local_gst.sh"   # or: source scripts/env_local_gst.sh
# Expects: bash scripts/build.sh — 插件在 build/plugins/*（与 scripts/build.sh 一致）
_multi_cam_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export GST_PLUGIN_PATH="\
${_multi_cam_dir}/build/plugins/gstfaceinfer:\
${_multi_cam_dir}/build/plugins/deduper:\
${_multi_cam_dir}/build/plugins/frameoffload:\
${_multi_cam_dir}/build/plugins/msgagg:\
${_multi_cam_dir}/build/plugins/timingmark:\
${GST_PLUGIN_PATH:-}"
