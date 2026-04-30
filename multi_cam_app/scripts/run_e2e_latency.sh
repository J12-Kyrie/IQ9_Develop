#!/bin/bash
# E2E Pipeline Latency Test: Producer → Consumer → Rule Process
# WITH 7 pipeline + 3 E2E latency metrics + Qualcomm device telemetry
#
# Extends run_e2e_pipeline.sh with:
#   - qtitimingmark-based pipeline latency (7 metrics from 9 probe points)
#   - Cross-process E2E latency (3 metrics from log correlation)
#   - Qualcomm resource telemetry (GPU, CDSP/HTP, NSP thermal, DDR, CMA)
#   - CPU affinity + GPU frequency lock for consistent measurements
#   - Multi-iteration support with device cooldown
#   - Statistical post-processing (trimmed percentiles, correlation)
#
# Usage:
#   bash scripts/run_e2e_latency.sh [options]
#   --config <path>        Producer config (default: config/config.json)
#   --channels <int>       Number of channels (default: from config)
#   --timeout <int>        Rule process timeout seconds (default: 15)
#   --loop-count <int>     Full E2E runs (default: 1)
#   --sample-sec <float>   Telemetry sample interval (default: 0.5)
#   --no-gpu-lock          Skip GPU frequency lock
#   --no-affinity          Skip CPU affinity pinning
#   --trim-frames <int>    Frames to trim from head/tail (default: 20)
#   -h|--help              Show help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DMA_DIR="$APP_DIR/test/dma_buf_pipline"

# ── defaults ──
PRODUCER="$APP_DIR/build/app/multi-cam-app"
CONSUMER="$DMA_DIR/dmabuf_consumer/build_local/bin/consumer_demo"
RULE="$DMA_DIR/rule_process/build_local/bin/nvsci_rule_process"
BASE_CONFIG="$APP_DIR/config/config.json"
RULE_CONFIG="$DMA_DIR/rule_process/config/rule_process_config.json"
TIMINGMARK_LIB="$APP_DIR/build/plugins/timingmark/libgstqtitimingmark.so"

INIT_SOCK="/tmp/dmabuf_ipc.sock"
RELAY_SOCK="/tmp/iq9_rule_relay.sock"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$APP_DIR/log/e2e_latency_${RUN_TAG}"

TIMEOUT_SEC=15
NUM_ITERS=1
SAMPLE_SEC=0.5
TRIM_FRAMES=20
LOCK_GPU=true
USE_AFFINITY=true
CHANNELS_OVERRIDE=""

PRODUCER_LOG="$RUN_DIR/producer.log"
CONSUMER_LOG="$RUN_DIR/consumer.log"
RULE_LOG="$RUN_DIR/rule_process.log"
EFFECTIVE_CONFIG="$RUN_DIR/config_effective.json"
ENV_LOG="$RUN_DIR/env.txt"
TELEMETRY_CSV="$RUN_DIR/telemetry.csv"
LATENCY_SUMMARY="$RUN_DIR/latency_summary.csv"
E2E_SUMMARY="$RUN_DIR/e2e_summary.csv"
ITER_LOG="$RUN_DIR/iterations.csv"
APP_PIDFILE="$RUN_DIR/.app_pid"

SAMPLER_PID=""
GPU_MIN_FREQ_ORIG=""

# ── helpers ──
usage() {
  sed -n '2,20p' "$0"
  exit 0
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing $1"; exit 1; }
}

cleanup() {
  if [[ -n "${SAMPLER_PID:-}" ]] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  if [[ -n "${GPU_MIN_FREQ_ORIG:-}" ]] && [[ -w /sys/class/kgsl/kgsl-3d0/devfreq/min_freq ]]; then
    echo "$GPU_MIN_FREQ_ORIG" > /sys/class/kgsl/kgsl-3d0/devfreq/min_freq 2>/dev/null || true
  fi
  rm -f "$APP_PIDFILE" 2>/dev/null || true
}
trap cleanup EXIT

# ── Qualcomm device helpers ──
sample_gpu_load() {
  local v
  v="$(cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage 2>/dev/null || true)"
  v="${v//%/}"; v="${v// /}"; echo "${v:--1}"
}

sample_gpu_freq() {
  cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null || echo 0
}

sample_cdsp_fd() {
  local pid="$1" n=0 f tgt
  [[ -n "$pid" && -r "/proc/$pid/fd" ]] || { echo -1; return; }
  for f in "/proc/$pid/fd"/*; do
    [[ -e "$f" ]] || continue
    tgt="$(readlink "$f" 2>/dev/null || true)"
    [[ "$tgt" == /dev/fastrpc-cdsp* ]] && ((n++)) || true
  done
  echo "$n"
}

sample_nsp_thermal_max() {
  local z max=0 v
  for z in 10 22 23 24 25 26; do
    v="$(cat /sys/class/thermal/thermal_zone${z}/temp 2>/dev/null || echo 0)"
    (( v > max )) && max=$v
  done
  echo "$max"
}

sample_cpu_thermal_max() {
  local z max=0 v
  for z in 1 2 17 18 19 20 21; do
    v="$(cat /sys/class/thermal/thermal_zone${z}/temp 2>/dev/null || echo 0)"
    (( v > max )) && max=$v
  done
  echo "$max"
}

sample_ddr_thermal() {
  cat /sys/class/thermal/thermal_zone11/temp 2>/dev/null || echo 0
}

sample_ddr_freq() {
  cat /sys/class/devfreq/kgsl-busmon/cur_freq 2>/dev/null || echo 0
}

sample_cma_kb() {
  local total free
  total="$(awk '/CmaTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  free="$(awk '/CmaFree:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  echo "$total,$free"
}

sample_rss_kb() {
  local pid="$1"
  awk '/VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0
}

sample_rproc_state() {
  local idx="$1"
  cat "/sys/class/remoteproc/remoteproc$idx/state" 2>/dev/null || echo "unknown"
}

# ── background telemetry sampler ──
telemetry_loop() {
  local pidfile="$1" out="$2" interval="$3"

  # CSV header
  {
    printf '%s' "ts_epoch_ms,pid,gpu_load_pct,gpu_freq_hz,cdsp_fd_count,"
    printf '%s' "nsp_thermal_max_mC,cpu_thermal_max_mC,ddr_thermal_mC,ddr_freq_hz,"
    printf '%s' "cma_total_kb,cma_free_kb,rss_kb,rproc4_state,mem_used_pct"
    printf '\n'
  } > "$out"

  local mem_total_kb
  mem_total_kb="$(awk '/MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 1)"

  while true; do
    local pid=""
    [[ -f "$pidfile" ]] && pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      sleep "$interval"; continue
    fi

    local ts rss avail used_pct cma gpu_load gpu_freq cdsp_fd nsp_temp cpu_temp ddr_temp ddr_freq rproc4
    ts="$(date +%s%3N)"
    rss="$(sample_rss_kb "$pid")"
    avail="$(awk '/MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    used_pct="$(awk -v t="$mem_total_kb" -v a="$avail" 'BEGIN{if(t>0)printf "%.1f",(t-a)*100/t; else print 0}')"
    cma="$(sample_cma_kb)"
    gpu_load="$(sample_gpu_load)"
    gpu_freq="$(sample_gpu_freq)"
    cdsp_fd="$(sample_cdsp_fd "$pid")"
    nsp_temp="$(sample_nsp_thermal_max)"
    cpu_temp="$(sample_cpu_thermal_max)"
    ddr_temp="$(sample_ddr_thermal)"
    ddr_freq="$(sample_ddr_freq)"
    rproc4="$(sample_rproc_state 4)"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$ts" "$pid" "$gpu_load" "$gpu_freq" "$cdsp_fd" \
      "$nsp_temp" "$cpu_temp" "$ddr_temp" "$ddr_freq" \
      "${cma%,*}" "${cma#*,}" "$rss" "$rproc4" "$used_pct" \
      >> "$out"
    sleep "$interval"
  done
}

# ── argument parsing ──
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        BASE_CONFIG="$2"; shift 2 ;;
    --channels)      CHANNELS_OVERRIDE="$2"; shift 2 ;;
    --timeout)       TIMEOUT_SEC="$2"; shift 2 ;;
    --loop-count)    NUM_ITERS="$2"; shift 2 ;;
    --sample-sec)    SAMPLE_SEC="$2"; shift 2 ;;
    --trim-frames)   TRIM_FRAMES="$2"; shift 2 ;;
    --no-gpu-lock)   LOCK_GPU=false; shift ;;
    --no-affinity)   USE_AFFINITY=false; shift ;;
    -h|--help)       usage ;;
    *) echo "Unknown: $1"; usage ;;
  esac
done

# ── pre-flight: tool checks ──
require_cmd python3
require_cmd awk
require_cmd gst-inspect-1.0
require_cmd taskset
if ! gst-inspect-1.0 qtitimingmark >/dev/null 2>&1; then
  echo "ERROR: qtitimingmark plugin not found in GStreamer path"
  echo "  Ensure GST_PLUGIN_PATH includes: $TIMINGMARK_LIB"
  exit 1
fi
for b in "$PRODUCER" "$CONSUMER" "$RULE" "$BASE_CONFIG" "$RULE_CONFIG"; do
  [[ -f "$b" ]] || { echo "ERROR: missing $b"; exit 1; }
done

mkdir -p "$RUN_DIR"

# ── pre-flight: device availability ──
echo "=== Pre-flight: device availability ==="
pf_ok=true; pf_wait=0; pf_max=60
while (( pf_wait < pf_max )); do
  pf_busy=false
  fuser -s /dev/video32 2>/dev/null && pf_busy=true
  for dev in /dev/fastrpc-cdsp*; do
    [[ -e "$dev" ]] || continue
    fuser -s "$dev" 2>/dev/null && { pf_busy=true; break; }
  done
  if ! $pf_busy; then break; fi
  (( pf_wait % 10 == 0 )) && echo "  waiting... (${pf_wait}s)"
  sleep 1; ((pf_wait++))
done
if (( pf_wait > 0 )); then
  echo "  Devices freed after ${pf_wait}s + 3s pad"
  sleep 3
else
  echo "  Devices idle, ready"
fi

# ── thermal guard ──
nsp_temp="$(sample_nsp_thermal_max)"
cpu_temp="$(sample_cpu_thermal_max)"
echo "  NSP temp: $((nsp_temp/1000))°C  CPU temp: $((cpu_temp/1000))°C"
if (( nsp_temp > 85000 || cpu_temp > 90000 )); then
  echo "WARNING: high temperature, results may be affected by throttling"
fi

# ── pre-flight: GPU lock ──
if $LOCK_GPU && [[ -w /sys/class/kgsl/kgsl-3d0/devfreq/min_freq ]]; then
  GPU_MIN_FREQ_ORIG="$(cat /sys/class/kgsl/kgsl-3d0/devfreq/min_freq)"
  echo 800000000 > /sys/class/kgsl/kgsl-3d0/devfreq/min_freq 2>/dev/null || true
  echo "  GPU min freq locked to 800MHz (was $GPU_MIN_FREQ_ORIG)"
fi

# ── config generation ──
python3 - "$BASE_CONFIG" "$EFFECTIVE_CONFIG" "$RUN_DIR" "$CHANNELS_OVERRIDE" <<'PY'
import json, sys

base_cfg, out_cfg, run_dir, ch_override = sys.argv[1:]

with open(base_cfg, encoding="utf-8") as f:
    cfg = json.load(f)

if ch_override and int(ch_override) > 0:
    cfg["max_channels"] = int(ch_override)

# Keep MQTT/FrameCache/msgagg enabled (E2E needs them)
# Add latency_test section alongside them
cfg["latency_test"] = {
    "enabled": True,
    "output_dir": run_dir,
    "sample_every_n": 1,
    "flush_per_sample": True,
}

# Ensure face is enabled for face_infer metrics
cfg.setdefault("face", {})["enabled"] = True

# log_dir to run_dir so gallery diag logs don't pollute cwd
cfg["log_dir"] = run_dir
cfg["output_dir"] = run_dir

with open(out_cfg, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print(f"Effective config: {out_cfg}")
PY

# ── env log ──
{
  echo "date=$(date -Iseconds)"
  echo "producer=$PRODUCER"
  echo "consumer=$CONSUMER"
  echo "rule=$RULE"
  echo "base_config=$BASE_CONFIG"
  echo "effective_config=$EFFECTIVE_CONFIG"
  echo "rule_config=$RULE_CONFIG"
  echo "run_dir=$RUN_DIR"
  echo "num_iters=$NUM_ITERS"
  echo "timeout_sec=$TIMEOUT_SEC"
  echo "gpu_lock=$LOCK_GPU"
  echo "cpu_affinity=$USE_AFFINITY"
  echo "measurement_method=timing_mark+epoch_ms_log_injection"
  echo "metrics=pipeline(7):file_to_dec,dec_to_yolo,yolo_qnn_infer,dec_to_tracker,bytetrack,dec_to_face,face_infer_to_embedding;e2e(3):producer_to_rule(slot_matched),rule_to_consumer(slot_matched),total_pipeline(first_frame)"
} > "$ENV_LOG"

# ── GST plugin path ──
export GST_PLUGIN_PATH="$(dirname "$TIMINGMARK_LIB"):${GST_PLUGIN_PATH:-}"

# ── start telemetry sampler ──
telemetry_loop "$APP_PIDFILE" "$TELEMETRY_CSV" "$SAMPLE_SEC" &
SAMPLER_PID=$!

# ── header ──
echo "============================================================"
echo "  E2E Latency Test: Producer → Consumer → Rule Process"
echo "  Run dir : $RUN_DIR"
echo "  Config  : $EFFECTIVE_CONFIG"
echo "  Timeout : ${TIMEOUT_SEC}s  Iters: $NUM_ITERS"
echo "  GPU lock: $LOCK_GPU  Affinity: $USE_AFFINITY"
echo "============================================================"

# ── iteration header ──
echo "iter,ts_start_ms,ts_end_ms,duration_sec,exit_code_producer,exit_code_consumer,exit_code_rule" > "$ITER_LOG"
OVERALL_RC=0

for (( ITER=1; ITER<=NUM_ITERS; ITER++ )); do
  echo ""
  echo "─── Iteration $ITER / $NUM_ITERS ───"
  ITER_START="$(date +%s%3N)"

  # ── cleanup before each iteration ──
  # Kill any leftover processes from prior crashed runs, then clear IPC artifacts
  pkill -9 -f multi-cam-app 2>/dev/null || true
  pkill -9 -f consumer_demo 2>/dev/null || true
  pkill -9 -f rule_process 2>/dev/null || true
  pkill -9 -f nvsci_rule_process 2>/dev/null || true
  sleep 1
  rm -f "$INIT_SOCK" "$RELAY_SOCK"
  rm -rf "$RUN_DIR/ppm" 2>/dev/null || true
  mkdir -p "$RUN_DIR/ppm"
  # Clear MQTT retained messages so stale scene_updates don't trigger spurious rules
  mosquitto_pub -t 'iq9/scene_update' -n -r -h 127.0.0.1 2>/dev/null || true
  mosquitto_pub -t 'iq9/frame_done'  -n -r -h 127.0.0.1 2>/dev/null || true

  # ── Step 1: Producer (with optional CPU affinity) ──
  # Pipe through timestamp injector: every stdout/stderr line gets epoch_ms prefix
  echo "  [1/3] Starting Producer..."
  PRODUCER_CMD=("$PRODUCER" "-c" "$EFFECTIVE_CONFIG")
  if $USE_AFFINITY; then
    { taskset 0x0F "${PRODUCER_CMD[@]}" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s%3N)" "$line"; done > "$PRODUCER_LOG"; } &
  else
    { "${PRODUCER_CMD[@]}" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s%3N)" "$line"; done > "$PRODUCER_LOG"; } &
  fi
  PRODUCER_PID=$!
  echo "$PRODUCER_PID" > "$APP_PIDFILE"
  echo "    PID=$PRODUCER_PID"
  sleep 2

  # ── Step 2: Consumer (line-buffered: stdbuf -oL prevents timestamp injection skew) ──
  echo "  [2/3] Starting Consumer..."
  CONSUMER_CMD=("$CONSUMER" "$INIT_SOCK" "$RELAY_SOCK" 100 "$RUN_DIR/ppm/" 100 0)
  if $USE_AFFINITY; then
    { taskset 0xF0 stdbuf -oL "${CONSUMER_CMD[@]}" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s%3N)" "$line"; done > "$CONSUMER_LOG"; } &
  else
    { stdbuf -oL "${CONSUMER_CMD[@]}" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s%3N)" "$line"; done > "$CONSUMER_LOG"; } &
  fi
  CONSUMER_PID=$!
  echo "    PID=$CONSUMER_PID"
  sleep 2

  # ── Step 3: Rule Process (timestamp-injected stdout via pipe) ──
  echo "  [3/3] Starting Rule Process (${TIMEOUT_SEC}s timeout)..."
  cd "$(dirname "$RULE")"
  set +e
  { timeout "$TIMEOUT_SEC" "$RULE" "$RULE_CONFIG" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s%3N)" "$line"; done > "$RULE_LOG"; }
  RULE_RC=$?
  set -e
  echo "    exit=$RULE_RC"

  # Rule process timed out or finished — gracefully stop producer + consumer
  # Signal them so they flush output and exit (instead of blocking on wait)
  for sig in TERM TERM TERM KILL; do
    kill -0 "$PRODUCER_PID" 2>/dev/null || break
    kill -0 "$CONSUMER_PID" 2>/dev/null || break
    kill -"$sig" "$PRODUCER_PID" 2>/dev/null || true
    kill -"$sig" "$CONSUMER_PID" 2>/dev/null || true
    sleep 1
  done

  set +e
  wait "$PRODUCER_PID" 2>/dev/null; PRODUCER_RC=$?
  wait "$CONSUMER_PID" 2>/dev/null; CONSUMER_RC=$?
  set -e

  rm -f "$APP_PIDFILE" 2>/dev/null || true

  ITER_END="$(date +%s%3N)"
  ITER_DUR="$(awk -v s="$ITER_START" -v e="$ITER_END" 'BEGIN{printf "%.1f",(e-s)/1000}')"
  echo "$ITER,$ITER_START,$ITER_END,$ITER_DUR,$PRODUCER_RC,$CONSUMER_RC,$RULE_RC" >> "$ITER_LOG"
  echo "  Duration: ${ITER_DUR}s  RC: producer=$PRODUCER_RC consumer=$CONSUMER_RC rule=$RULE_RC"

  # merge per-channel latency CSVs across iters
  for csv_file in "$RUN_DIR"/latency_ch*.csv; do
    [[ -f "$csv_file" ]] || continue
    combined="$RUN_DIR/latency_combined_$(basename "$csv_file")"
    if [[ ! -f "$combined" ]]; then
      echo "iter,$(head -1 "$csv_file")" > "$combined"
    fi
    tail -n +2 "$csv_file" | sed "s/^/${ITER},/" >> "$combined"
  done

  [[ $PRODUCER_RC -ne 0 ]] && OVERALL_RC=$PRODUCER_RC

  # cooldown between iterations
  if (( ITER < NUM_ITERS )); then
    wait_ok=0; max_wait=20; waited=0
    while (( waited < max_wait )); do
      busy=false
      fuser -s /dev/video32 2>/dev/null && busy=true
      for dev in /dev/fastrpc-cdsp*; do
        [[ -e "$dev" ]] || continue
        fuser -s "$dev" 2>/dev/null && { busy=true; break; }
      done
      if ! $busy; then wait_ok=1; break; fi
      sleep 1; ((waited++))
    done
    if (( wait_ok )); then
      sleep 3
      echo "  cooldown: ${waited}s + 3s pad"
    fi
  fi
done

# ── stop sampler ──
if [[ -n "${SAMPLER_PID:-}" ]] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
  kill "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true
fi
SAMPLER_PID=""

# ── restore GPU ──
if $LOCK_GPU && [[ -n "${GPU_MIN_FREQ_ORIG:-}" ]] && [[ -w /sys/class/kgsl/kgsl-3d0/devfreq/min_freq ]]; then
  echo "$GPU_MIN_FREQ_ORIG" > /sys/class/kgsl/kgsl-3d0/devfreq/min_freq 2>/dev/null || true
  echo "GPU min freq restored to $GPU_MIN_FREQ_ORIG"
fi

# ── pipeline latency summary (7 metrics) ──
echo ""
echo "=== Pipeline Latency Summary (7 metrics) ==="
python3 - "$RUN_DIR" "$LATENCY_SUMMARY" "$TRIM_FRAMES" <<'PY'
import csv, glob, os, statistics, sys
from collections import defaultdict

run_dir, out_csv, trim_n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows_all = []

def load_combined(path):
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_all.append(row)

combined = sorted(glob.glob(os.path.join(run_dir, "latency_combined_latency_ch*.csv")))
if combined:
    for p in combined:
        load_combined(p)
else:
    for p in sorted(glob.glob(os.path.join(run_dir, "latency_ch*.csv"))):
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["iter"] = "1"
                rows_all.append(row)

if not rows_all:
    print("WARNING: no latency samples found", file=sys.stderr)
    sys.exit(0)

groups = defaultdict(list)
for r in rows_all:
    key = (r.get("channel_id", "0"), r.get("iter", "1"))
    groups[key].append(r)

kept = []
trimmed = 0
for (ch, it), grp in groups.items():
    fkeys = sorted(set(int(r.get("frame_key", 0)) for r in grp))
    if len(fkeys) > trim_n * 2:
        drop = set(fkeys[:trim_n]) | set(fkeys[-trim_n:])
        before = len(grp)
        grp = [r for r in grp if int(r.get("frame_key", 0)) not in drop]
        trimmed += before - len(grp)
    kept.extend(grp)

print(f"Trimmed {trimmed} rows (first/last {trim_n} frames per ch/iter)")

lat_vals = defaultdict(list)
for r in kept:
    metric = r.get("metric", "").strip()
    if not metric:
        continue
    try:
        lat = float(r.get("latency_ms", "nan"))
    except ValueError:
        continue
    if lat < 0:
        continue
    lat_vals[metric].append(lat)

def pct(data, q):
    s = sorted(data)
    idx = max(0, min(len(s)-1, int(round(q/100.0*(len(s)-1)))))
    return s[idx]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric","count","min_ms","p50_ms","p90_ms","p95_ms","p99_ms","max_ms","avg_ms"])
    for metric in sorted(lat_vals):
        arr = lat_vals[metric]
        w.writerow([metric, len(arr),
            f"{min(arr):.3f}", f"{pct(arr,50):.3f}", f"{pct(arr,90):.3f}",
            f"{pct(arr,95):.3f}", f"{pct(arr,99):.3f}", f"{max(arr):.3f}",
            f"{statistics.fmean(arr):.3f}"])

print(f"Summary: {len(lat_vals)} metrics -> {out_csv}")
for metric in sorted(lat_vals):
    arr = lat_vals[metric]
    print(f"  {metric}: count={len(arr)} avg={statistics.fmean(arr):.3f}ms "
          f"p50={pct(arr,50):.3f}ms p99={pct(arr,99):.3f}ms")
PY

# ── E2E cross-process latency (3 metrics) ──
# All 3 process logs now have epoch_ms timestamps injected at line start
# Format: "1777438883698 [original log line]"
# Latency CSV also has ts_epoch_ms column (CLOCK_REALTIME)
# Cross-correlation key: slot number (appears in rule+consumer logs)
echo ""
echo "=== E2E Cross-Process Latency (3 metrics) ==="
python3 - "$RUN_DIR" "$E2E_SUMMARY" <<'PY'
import csv, glob, os, re, statistics, sys
from collections import defaultdict

run_dir, out_csv = sys.argv[1], sys.argv[2]

# ── Parse rule_process log (timestamp-injected) ──
# Format: "1777438883698 [INFO][RuleProcess] relay sent source=0 slot=1 rule=..."
rule_events = []  # [(epoch_ms, source, slot)]
rule_log = os.path.join(run_dir, "rule_process.log")
if os.path.exists(rule_log):
    with open(rule_log, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r'^(\d{13}) (.*)$', line)
            if not m:
                continue
            ts_ms = int(m.group(1))
            body = m.group(2)
            if 'relay sent' in body:
                sm = re.search(r'source=(\d+)\s+slot=(\d+)', body)
                if sm:
                    rule_events.append((ts_ms, int(sm.group(1)), int(sm.group(2))))

# ── Parse consumer log (stdbuf -oL ensures line-buffered, injected ts is accurate) ──
# Format: "1777438883698 [consumer] frame=0 slot=0 ... ts=1777438883698392155 ..."
# injected_ts_ms = consumer-side frame processing time (line-buffered, ~1ms accuracy)
# internal ts=<ns>  = producer-side frame submission time (CLOCK_REALTIME, for cross-check)
consumer_events = []  # [(inj_epoch_ms, slot, prod_ts_ns)]
consumer_log = os.path.join(run_dir, "consumer.log")
if os.path.exists(consumer_log):
    with open(consumer_log, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r'^(\d{13}) (.*)$', line)
            if not m:
                continue
            ts_ms = int(m.group(1))
            body = m.group(2)
            if 'frame=' in body and 'slot=' in body:
                sm = re.search(r'slot=(\d+).*ts=(\d{10,})', body)
                if sm:
                    consumer_events.append((ts_ms, int(sm.group(1)), int(sm.group(2))))

# ── Parse producer pipeline completion from latency CSVs ──
# Each latency row has ts_epoch_ms (when metric was emitted ≈ pipeline stage completion)
# Use the LATEST metric per frame_key as the "frame ready for downstream" timestamp
prod_frames = []  # [(ts_epoch_ms, channel_id)]
latency_files = sorted(glob.glob(os.path.join(run_dir, "latency_ch*.csv")))
if latency_files:
    for lf in latency_files:
        with open(lf, encoding="utf-8") as f:
            # Only take 1 row per frame_key: the one with largest end_ns (last stage done)
            frame_last = {}  # frame_key -> (ts_epoch_ms, end_ns)
            for row in csv.DictReader(f):
                fk = row.get("frame_key", "")
                ts = int(row.get("ts_epoch_ms", "0") or "0")
                en = int(row.get("end_ns", "0") or "0")
                if fk and ts > 0:
                    prev = frame_last.get(fk)
                    if prev is None or en > prev[1]:
                        frame_last[fk] = (ts, en)
            for fk, (ts, _) in frame_last.items():
                prod_frames.append((ts, int(row.get("channel_id", "0") or "0")))

# ── Correlate rule→consumer by slot number ──
# For each rule relay event, find the nearest consumer frame for the same slot
e2e_rule_to_consumer = []
slot_to_rule = defaultdict(list)
for ts, src, slot in rule_events:
    slot_to_rule[slot].append(ts)
slot_to_consumer = defaultdict(list)
for ts, slot, ts_ns in consumer_events:
    slot_to_consumer[slot].append(ts)

for slot, rtimes in slot_to_rule.items():
    ctimes = slot_to_consumer.get(slot, [])
    if not ctimes:
        continue
    # Match each rule relay to closest consumer frame after it
    ci = 0
    for rt in sorted(rtimes):
        while ci < len(ctimes) and ctimes[ci] < rt:
            ci += 1
        if ci < len(ctimes):
            delta_ms = ctimes[ci] - rt
            if 0 <= delta_ms < 5000:  # reasonable E2E range: 0-5s
                e2e_rule_to_consumer.append(delta_ms)

# ── First-frame E2E (single-shot, reliable) ──
first_prod_ms = min(ts for ts, _ in prod_frames) if prod_frames else 0
first_rule_ms = min(ts for ts, _, _ in rule_events) if rule_events else 0
first_cons_ms = min(ts for ts, _, _ in consumer_events) if consumer_events else 0

# ── Write summary ──
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["e2e_metric", "count", "min_ms", "avg_ms", "max_ms", "notes"])
    if first_prod_ms and first_rule_ms:
        delta = first_rule_ms - first_prod_ms
        w.writerow(["e2e_producer_to_rule", 1, delta, delta, delta,
                     f"1st producer frame→1st rule relay"])
    if e2e_rule_to_consumer:
        w.writerow(["e2e_rule_to_consumer", len(e2e_rule_to_consumer),
                     f"{min(e2e_rule_to_consumer):.1f}",
                     f"{statistics.fmean(e2e_rule_to_consumer):.1f}",
                     f"{max(e2e_rule_to_consumer):.1f}",
                     "Rule relay→Consumer frame (slot-matched)"])
    if first_prod_ms and first_cons_ms:
        delta = first_cons_ms - first_prod_ms
        w.writerow(["e2e_total_pipeline", 1, delta, delta, delta,
                     f"1st producer frame→1st consumer PPM"])

print(f"  Rule relay events:       {len(rule_events)}")
print(f"  Consumer frame events:   {len(consumer_events)}")
print(f"  Producer frame events:   {len(prod_frames)}")
if first_prod_ms and first_rule_ms:
    print(f"  e2e_producer_to_rule:   {first_rule_ms - first_prod_ms}ms (1st frame)")
if e2e_rule_to_consumer:
    print(f"  e2e_rule_to_consumer:   avg={statistics.fmean(e2e_rule_to_consumer):.1f}ms "
          f"(n={len(e2e_rule_to_consumer)} slot-matched pairs)")
if first_prod_ms and first_cons_ms:
    print(f"  e2e_total_pipeline:     {first_cons_ms - first_prod_ms}ms (1st frame)")
print(f"  → {out_csv}")
PY

# ── final summary ──
echo ""
echo "============================================================"
echo "  E2E Latency Test Complete"
echo "  Exit code: $OVERALL_RC"
echo "  Artifacts: $RUN_DIR/"
echo "    config_effective.json         — merged config with latency_test"
echo "    latency_summary.csv           — 7 pipeline metrics (trimmed)"
echo "    e2e_summary.csv               — 3 cross-process metrics"
echo "    telemetry.csv                 — Qualcomm resource telemetry"
echo "    iterations.csv                — per-iteration timing"
echo "    producer.log / consumer.log / rule_process.log"
echo "    latency_ch*.csv               — per-channel raw timing data"
echo "============================================================"

# ── quick telemetry summary ──
echo ""
echo "=== Telemetry Quick Summary ==="
python3 - "$TELEMETRY_CSV" <<'PY'
import csv, sys

with open(sys.argv[1], encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("  No telemetry data")
    sys.exit(0)

def safe_float(r, k):
    try: return float(r.get(k, 0) or 0)
    except: return 0.0

gpu_loads = [safe_float(r, 'gpu_load_pct') for r in rows if safe_float(r, 'gpu_load_pct') >= 0]
nsp_temps = [safe_float(r, 'nsp_thermal_max_mC')/1000 for r in rows if safe_float(r, 'nsp_thermal_max_mC') > 0]
cdsp_fds = [safe_float(r, 'cdsp_fd_count') for r in rows if safe_float(r, 'cdsp_fd_count') >= 0]
rss_vals = [safe_float(r, 'rss_kb')/1024 for r in rows if safe_float(r, 'rss_kb') > 0]

import statistics
print(f"  GPU load:    avg={statistics.fmean(gpu_loads):.1f}% max={max(gpu_loads):.0f}%") if gpu_loads else None
print(f"  NSP temp:    avg={statistics.fmean(nsp_temps):.1f}°C max={max(nsp_temps):.0f}°C") if nsp_temps else None
print(f"  CDSP fd:     avg={statistics.fmean(cdsp_fds):.1f} max={max(cdsp_fds):.0f}") if cdsp_fds else None
print(f"  RSS:         avg={statistics.fmean(rss_vals):.0f}MB max={max(rss_vals):.0f}MB") if rss_vals else None
print(f"  Samples:     {len(rows)}")
PY

exit "$OVERALL_RC"
