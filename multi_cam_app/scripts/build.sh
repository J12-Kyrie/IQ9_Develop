#!/bin/bash
# 重构后 multi_cam_app 统一编译脚本：GStreamer 插件 + multi-cam-app 主程序。
# 依赖：CMake、GStreamer 开发包、json-glib（主程序与 qtimsgagg）、Paho C（MQTT/FrameCache）、libpaho-mqtt-dev。
# nlohmann_json：若系统未装，CMake 会 FetchContent 拉取，需能访问 GitHub 或提前安装 nlohmann-json3-dev。
#
# 用法:
#   bash scripts/build.sh                    # Release，尽量 sudo 安装插件到 GStreamer 目录
#   INSTALL_GST_PLUGINS=0 bash scripts/build.sh   # 只编译到 build_* 目录，不 sudo 拷系统目录
#   bash scripts/build.sh --debug            # CMAKE_BUILD_TYPE=Debug
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$APP_DIR/build}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
if [[ "${1:-}" == "--debug" ]]; then
  BUILD_TYPE=Debug
fi

# aarch64 与 x86_64 常见插件目录
MACHINE="$(uname -m 2>/dev/null || echo unknown)"
if [[ -d /usr/lib/${MACHINE}-linux-gnu/gstreamer-1.0 ]]; then
  GST_PLUGIN_DIR="/usr/lib/${MACHINE}-linux-gnu/gstreamer-1.0"
else
  GST_PLUGIN_DIR="/usr/lib/gstreamer-1.0"
fi
INSTALL_GST="${INSTALL_GST_PLUGINS:-1}"

echo "=== multi_cam_app build (refactored) ==="
echo "  Source:   $APP_DIR"
echo "  Build:    $BUILD_DIR  ($BUILD_TYPE)"
echo "  Install:  GST plugins -> $GST_PLUGIN_DIR  (INSTALL_GST_PLUGINS=$INSTALL_GST)"
echo ""

echo "--- dependencies ---"
for pc in gstreamer-allocators-1.0 gstreamer-app-1.0 gstreamer-video-1.0 gstreamer-base-1.0 json-glib-1.0; do
  if pkg-config --exists "$pc" 2>/dev/null; then
    echo "  $pc: OK"
  else
    echo "  ERROR: $pc not found (install dev packages)"
    exit 1
  fi
done

if [[ -f "$APP_DIR/aggregator/concurrentqueue/blockingconcurrentqueue.h" ]]; then
  echo "  concurrentqueue: OK"
else
  echo "  ERROR: aggregator/concurrentqueue/ missing"
  exit 1
fi

PAHO_FOUND=0
for path in /usr/lib/libpaho-mqtt3c.so "/usr/lib/${MACHINE}-linux-gnu/libpaho-mqtt3c.so" /usr/local/lib/libpaho-mqtt3c.so; do
  if [[ -f "$path" ]]; then
    echo "  paho-mqtt3c: OK ($path)"
    PAHO_FOUND=1
    break
  fi
done
if [[ "$PAHO_FOUND" -eq 0 ]]; then
  echo "  WARN: libpaho-mqtt3c not found — link may fail for MQTT / FrameCacheService"
fi

if pkg-config --exists nlohmann_json 2>/dev/null; then
  echo "  nlohmann_json (pkg-config): OK"
else
  echo "  NOTE: nlohmann_json not in pkg-config; CMake may FetchContent (needs git+network) or install nlohmann-json3-dev"
fi
echo ""

nproc_v="${NPROC:-$(nproc 2>/dev/null || echo 4)}"
cmake_cfg=( -DCMAKE_BUILD_TYPE="$BUILD_TYPE" )

# --- plugins：全部在 $BUILD_DIR/plugins/<子目录> 下 out-of-tree 构建 ---
# 避免在仓库根目录使用 build_deduper_plugin 等旧目录：拷贝工程时常带有旧绝对路径的 CMakeCache。
PLUGINS_ROOT="$BUILD_DIR/plugins"
FACEINFER_SRC="$APP_DIR/channel_worker/infer_module/face_infer/gst_plugin"
FACEINFER_BUILD="$PLUGINS_ROOT/gstfaceinfer"
echo "--- GstFaceInfer ---"
if [[ -d "$FACEINFER_SRC/lib" ]]; then
  echo "  NOTE: 可删旧 in-tree 缓存: rm -rf \"$FACEINFER_SRC/lib\""
fi
mkdir -p "$FACEINFER_BUILD"
cmake -S "$FACEINFER_SRC" -B "$FACEINFER_BUILD" "${cmake_cfg[@]}"
cmake --build "$FACEINFER_BUILD" -j"$nproc_v"

DEDUPER_SRC="$APP_DIR/channel_worker/deduper/gst_plugin"
DEDUPER_BUILD="$PLUGINS_ROOT/deduper"
echo "--- qtideduper ---"
if [[ -d "$APP_DIR/build_deduper_plugin" ]]; then
  echo "  NOTE: 根目录下旧构建目录可删（含旧 CMake 缓存）: rm -rf \"$APP_DIR/build_deduper_plugin\""
fi
mkdir -p "$DEDUPER_BUILD"
cmake -S "$DEDUPER_SRC" -B "$DEDUPER_BUILD" "${cmake_cfg[@]}"
cmake --build "$DEDUPER_BUILD" -j"$nproc_v"

FO_SRC="$APP_DIR/channel_worker/frame_offload/gst_plugin"
FO_BUILD="$PLUGINS_ROOT/frameoffload"
echo "--- qtiframeoffload ---"
mkdir -p "$FO_BUILD"
cmake -S "$FO_SRC" -B "$FO_BUILD" "${cmake_cfg[@]}" -DAPP_DIR="$APP_DIR"
cmake --build "$FO_BUILD" -j"$nproc_v"

MA_SRC="$APP_DIR/channel_worker/msg_agg/gst_plugin"
MA_BUILD="$PLUGINS_ROOT/msgagg"
echo "--- qtimsgagg (json-glib) ---"
mkdir -p "$MA_BUILD"
cmake -S "$MA_SRC" -B "$MA_BUILD" "${cmake_cfg[@]}"
cmake --build "$MA_BUILD" -j"$nproc_v"

TM_SRC="$APP_DIR/channel_worker/timing_mark/gst_plugin"
TM_BUILD="$PLUGINS_ROOT/timingmark"
echo "--- qtitimingmark ---"
mkdir -p "$TM_BUILD"
cmake -S "$TM_SRC" -B "$TM_BUILD" "${cmake_cfg[@]}" -DAPP_DIR="$APP_DIR"
cmake --build "$TM_BUILD" -j"$nproc_v"

install_plugin() {
  local so="$1"
  local name="$2"
  if [[ ! -f "$so" ]]; then
    echo "ERROR: missing $so"
    exit 1
  fi
  if [[ "$INSTALL_GST" == "1" ]]; then
    if [[ -w "$GST_PLUGIN_DIR" ]] 2>/dev/null; then
      cp -f "$so" "$GST_PLUGIN_DIR/"
    else
      sudo install -m0644 "$so" "$GST_PLUGIN_DIR/"
    fi
    rm -f "$HOME/.cache/gstreamer-1.0/registry."*.bin 2>/dev/null || true
    echo "  installed $name -> $GST_PLUGIN_DIR"
  else
    echo "  (skip system install) $name"
  fi
}

echo "--- install selected plugins to GStreamer (optional) ---"
if [[ "$INSTALL_GST" == "1" ]]; then
  install_plugin "$FO_BUILD/libgstqtiframeoffload.so" qtiframeoffload
  install_plugin "$MA_BUILD/libgstqtimsgagg.so" qtimsgagg
  install_plugin "$TM_BUILD/libgstqtitimingmark.so" qtitimingmark
else
  echo "  INSTALL_GST_PLUGINS=0, .so under $PLUGINS_ROOT/* only"
fi
echo ""

echo "--- multi-cam-app ---"
# 使用 $BUILD_DIR/app 而非 $BUILD_DIR 作为顶层工程构建目录，避免拷贝工程时随 copy 带来的
# build/CMakeCache.txt（仍指向 gst-plugins-qti-oss-imsdk.../multi_cam_app）与当前路径冲突。
MAIN_APP_BUILD="$BUILD_DIR/app"
mkdir -p "$MAIN_APP_BUILD"
cmake -S "$APP_DIR" -B "$MAIN_APP_BUILD" "${cmake_cfg[@]}"
cmake --build "$MAIN_APP_BUILD" -j"$nproc_v"

echo ""
echo "=== build OK ==="
echo "  Binary:  $MAIN_APP_BUILD/multi-cam-app"
echo "  插件 .so:  $PLUGINS_ROOT/{gstfaceinfer,deduper,frameoffload,msgagg,timingmark}/libgst*.so"
echo "  Quick:  source $SCRIPT_DIR/env_local_gst.sh && gst-inspect-1.0 qtimsgagg | head -3"
echo "  Run:     bash $SCRIPT_DIR/run_smoke.sh [config.json]"
echo ""

# aggregator test: optional
AGG_TEST="$APP_DIR/aggregator/test/build_test_aggregator.sh"
if [[ -f "$AGG_TEST" ]]; then
  echo "--- optional: aggregator test ---"
  bash "$AGG_TEST" || true
else
  echo "  (skip) aggregator/test not present: $AGG_TEST"
fi
