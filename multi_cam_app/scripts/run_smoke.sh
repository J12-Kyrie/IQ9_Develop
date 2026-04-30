#!/bin/bash
# 烟雾测试：使用本地/系统插件路径运行 multi-cam-app（不替代板端长稳压测）。
# 用法:
#   bash scripts/run_smoke.sh [path/to/config.json]
# 环境: 可先 source scripts/env_local_gst.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$APP_DIR/build"
BINARY="$BUILD_DIR/app/multi-cam-app"
CONFIG="${1:-$APP_DIR/config/config.json}"
LOG_FILE="$APP_DIR/run_smoke.log"

# 若未设置 GST_PLUGIN_PATH，则尝试与 env_local_gst 一致
if [[ -z "${GST_PLUGIN_PATH:-}" && -d "$APP_DIR/build/plugins/frameoffload" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env_local_gst.sh"
fi

if [[ ! -f "$BINARY" ]]; then
  echo "ERROR: 未找到 $BINARY ，请先: bash $SCRIPT_DIR/build.sh"
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: 配置不存在: $CONFIG"
  exit 1
fi

echo "=== run_smoke: multi-cam-app ==="
echo "  binary:  $BINARY"
echo "  config:  $CONFIG"
echo "  log:     $LOG_FILE"
echo "  GST_PLUGIN_PATH: ${GST_PLUGIN_PATH:-<system default>}"
echo ""

cd "$APP_DIR"
# 可按需打开调试级别
GST_DEBUG="${GST_DEBUG:-qtimsgagg:2,qtiframeoffload:2}" \
  "$BINARY" -c "$CONFIG" 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"