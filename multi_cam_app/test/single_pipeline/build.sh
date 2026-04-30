#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build build -j"$(nproc 2>/dev/null || echo 4)"
echo "Built: $(pwd)/build/single-yolo-qnn-latency"
