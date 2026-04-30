#!/bin/bash
# 在 IQ9 树内编译与 multi_cam 联调用的组件:
#   - dma_buf_pipline/dmabuf_consumer  -> bin/consumer_demo
#   - dma_buf_pipline/rule_process   -> bin/nvsci_rule_process
#
# 假设本脚本位于:  <IQ9>/multi_cam_app/scripts/
# 用法:
#   bash scripts/build_iq9_dma_pipeline.sh
#   BUILD_DIR=~/build-iq9 bash scripts/build_iq9_dma_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IQ9_ROOT="$(cd "$APP_DIR/.." && pwd)"
DEMO_ROOT="$IQ9_ROOT/dma_buf_pipline"
OUT="${BUILD_DIR:-$DEMO_ROOT/build_iq9_pipeline}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
nproc_v="${NPROC:-$(nproc 2>/dev/null || echo 4)}"
cmake_args=( -DCMAKE_BUILD_TYPE="$BUILD_TYPE" )

echo "=== IQ9 sidecars (consumer + rule_process) ==="
echo "  IQ9_ROOT:    $IQ9_ROOT"
echo "  Out dir:     $OUT"
echo ""

if [[ ! -d "$DEMO_ROOT/dmabuf_consumer" || ! -d "$DEMO_ROOT/rule_process" ]]; then
  echo "ERROR: 未找到 $DEMO_ROOT 下的 dmabuf_consumer 或 rule_process"
  exit 1
fi

echo "--- dmabuf_consumer (consumer_demo) ---"
mkdir -p "$OUT/consumer"
cmake -S "$DEMO_ROOT/dmabuf_consumer" -B "$OUT/consumer" "${cmake_args[@]}"
cmake --build "$OUT/consumer" -j"$nproc_v"
echo "  -> $OUT/consumer/bin/consumer_demo"
echo ""

echo "--- rule_process (nvsci_rule_process, MQTT) ---"
mkdir -p "$OUT/rule_process"
# ENABLE_MQTT 需要 Paho; 若未找到 FindPahoMQTT，会关掉 MQTT
cmake -S "$DEMO_ROOT/rule_process" -B "$OUT/rule_process" "${cmake_args[@]}"
cmake --build "$OUT/rule_process" -j"$nproc_v"
echo "  -> $OUT/rule_process/bin/nvsci_rule_process"
echo ""

echo "=== done ==="
echo "  联调启动顺序与配置见: multi_cam_app/scripts/JOINT_DEBUG.md"
