#!/bin/bash
# 已迁移到 build.sh。保留本文件以便旧命令仍可用。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/build.sh" "$@"
