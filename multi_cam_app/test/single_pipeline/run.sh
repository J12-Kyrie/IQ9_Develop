#!/usr/bin/env bash
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin="${dir}/build/single-yolo-qnn-latency"
cfg="${1:-"$dir/yolo_qnn_latency_config.json"}"

if [[ ! -f "$bin" ]]; then
  echo "Binary not found: $bin" >&2
  echo "Run: bash $dir/build.sh" >&2
  exit 1
fi
if [[ ! -f "$cfg" ]]; then
  echo "Config not found: $cfg" >&2
  exit 1
fi

# Optional: ensure custom qti plugin path (e.g. qtitimingmark) is visible on the target.
# export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0:...
# If preroll/PLAYING still fails, capture element errors:
#   GST_DEBUG=2 "$bin" "$cfg"   or   GST_DEBUG=qtimlqnn:5,qtimlvconverter:5
# First HTP/preroll can take many minutes: default wait is 8 min (see main.cpp);
#   export SINGLE_PIPELINE_STATE_TIMEOUT_SEC=1200

echo "== gst-inspect (target sanity; optional) =="
if command -v gst-inspect-1.0 &>/dev/null; then
  gst-inspect-1.0 qtitimingmark &>/dev/null && echo "qtitimingmark: OK" || echo "qtitimingmark: MISSING" >&2
  gst-inspect-1.0 qtimlqnn &>/dev/null && echo "qtimlqnn: OK" || echo "qtimlqnn: MISSING" >&2
else
  echo "gst-inspect-1.0 not in PATH, skip"
fi

echo "== run =="
"$bin" "$cfg"
rc=$?

# Parse output_dir from JSON (portable: python3) for yolo_qnn_infer summary
out_dir=""
if command -v python3 &>/dev/null; then
  out_dir="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$cfg" 2>/dev/null || true)"
fi
if [[ -n "$out_dir" && -d "$out_dir" ]]; then
  echo "== yolo_qnn_infer summary (mean latency_ms) =="
  shopt -s nullglob
  for f in "$out_dir"/latency_ch*.csv; do
    if command -v awk &>/dev/null; then
      awk -F',' 'NR>1 && $3=="yolo_qnn_infer" {s+=$6; n++} END {if(n) printf "%s: count=%d mean_ms=%.6f\n", FILENAME, n, s/n; else print FILENAME, ": no yolo_qnn_infer rows"}' "$f"
    else
      echo "awk not in PATH, skip summary for $f"
    fi
  done
  shopt -u nullglob
fi

exit "$rc"
