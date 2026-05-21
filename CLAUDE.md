# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Model

**Mac is a lightweight dev shell.** All source code, builds, and execution happen on the IQ9 Device. Claude Code on Mac edits files via SSH, runs builds/tests remotely, and manages git on the device.

**Device**: `ubuntu@192.168.100.102` (password: `Hello123`)
**Workspace**: `/mnt/workspace/develop`
**SSH**: `sshpass -p "Hello123" ssh -o StrictHostKeyChecking=no ubuntu@192.168.100.102`

Mac local directory (`/Users/kyrie/Downloads/visteon_develop/IQ9/`) contains only:
- `CLAUDE.md` (this file)
- `.claude/` (Claude Code config)
- `.omc/` (OMC state)

## Project Overview

IQ9 is a multi-camera object detection, tracking, and VLM (Vision-Language Model) inference platform built for **Qualcomm QCS9075** (aarch64 Linux) using the **Qualcomm Intelligent Multimedia SDK (IM SDK)**. It processes multiple H.264 video streams through GStreamer pipelines with YOLOv11 inference via QNN on the HTP, ByteTrack tracking, face detection/gallery matching, and optional VLM augmentation.

## Repository Structure

```
/mnt/workspace/develop/
├── multi_cam_app/     # Core detection pipeline (YOLO + ByteTrack + Face + DMA-BUF)
├── vlm_pipeline/      # VLM-augmented pipeline (extends multi_cam_app with VLM inference)
├── qwen3vit/          # Standalone VIT deployment (QNN, OpenCL acceleration)
├── qwen3vl/           # Full Qwen3-VL pipeline (Genie SDK, dual-CDSP, speculative decoding)
└── .gitignore
```

### Git Repository

| Setting | Value |
|---------|-------|
| Remote | `git@github.com:J12-Kyrie/IQ9_Develop.git` |
| Branch | `main` |
| SSH Key | `~/.ssh/id_ed25519` |

`.gitignore` excludes: `build/`, `*.bin`, `*.onnx`, `*.tflite`, `*.dlc`, `*.qnn`, `*.serialized`, `*.mp4`, `*.ppm`, `log/`, `build_local/`, `*.jsonl`, `output/`.

---

## multi_cam_app/ — Core Detection Pipeline

Multi-camera YOLOv11 detection + ByteTrack tracking + face recognition + DMA-BUF frame sharing.

### Data Flow (per channel)

```
filesrc -> qtdemux -> h264parse -> v4l2h264dec (NV12 dmabuf)
  -> qtimlvconverter -> qtimlqnn (YOLOv11) -> qtimlpostprocess (yolov8 module)
  -> qtiobjtracker (ByteTrack) -> text/x-raw capsfilter
  -> metamux -> qtideduper (whitelist + IoU dedup) -> tee_deduper
       |-> queue_combined -> appsink -> ChannelWorker (JSONL, only !msgagg)
       |-> queue_fo -> qtiframeoffload -> queue_agg -> qtimsgagg(timeout=200ms) -> qtimsgpub(json=TRUE) -> MQTT
           |-> WriteSceneJsonl: scene_update_ch{N}.jsonl (when msgagg.enabled && scene_jsonl_dir set)
           └─ Face gallery: Match/Enroll via qtifaceinfer(face_interval_ms=1000, 1fps gate)
```

### Key Components

| Component | Path | Purpose |
|-----------|------|---------|
| `MultiCamApp` | `app/` | Top-level orchestrator. GstPipeline, GMainLoop, bus watch, lifecycle |
| `ChannelWorker` | `channel_worker/` | Per-stream: appsink, bbox metadata parsing, label resolution, JSONL output |
| `DetectionInferModule` | `channel_worker/infer_module/` | GStreamer ML chain: demux, decode, QNN inference, postprocess, ByteTrack |
| `FrameOffload` | `channel_worker/frame_offload/` | `qtiframeoffload` plugin: DMA-BUF submit, scene JSON, face gallery for msgagg |
| `gstmsgagg` | `channel_worker/msg_agg/` | Custom plugin: merges per-channel JSON into `scene_update` envelope (200ms timeout) |
| `qtifaceinfer` | `channel_worker/infer_module/face_infer/gst_plugin/` | Face detection plugin: SCRFD + ArcFace (512-dim), interval gating |
| `Deduper` | `channel_worker/deduper/` | `qtideduper` plugin: class whitelist + IoU dedup |
| `ConfigLoader` | `config/` | JSON config parser (json-glib). Limits: max_channels=6, slot_count<=32 |
| `JsonlWriter` | `utils/` | Thread-safe JSONL writer (per-frame + MQTT scene_update) |
| `FaceGallery` | `utils/` | Cosine-similarity face gallery (match/enroll/eviction) |
| `MqttPublisher` | `utils/mqtt/` | Paho MQTT publisher for `iq9/scene_update` |
| `SlotPool` | `frame_cache/src/` | DMA-BUF slot pool (1-32 slots, shared across channels) |
| `Aggregator` | `aggregator/` | Batch aggregator with lock-free concurrent queue |

### Config (`config/config.json`)

| Field | Default | Notes |
|-------|---------|-------|
| `model_path` | — | YOLOv11 QNN model path |
| `videos_path` | — | String array, up to 6 video files |
| `max_channels` | 6 | 1-6 |
| `slot_count` | 8 | DMA-BUF pool size (1-32) |
| `face.enabled` | false | Global face toggle |
| `face.channel_mask` | 0xFF | Bitmask: bit N = face on channel N |
| `face.face_interval_ms` | 0 | 0=all frames; 1000=1fps |
| `msgagg.enabled` | — | Must be true for MQTT scene_update |
| `msgagg.timeout_ms` | 200 | Aggregation window |
| `msgagg.scene_jsonl_dir` | "" | If set, writes scene_update_ch{N}.jsonl |
| `frame_cache.enabled` | — | Must be true for DMA-BUF + MQTT path |

### Namespace

All code under `multi_cam_app::` with sub-namespaces: `app`, `channel_worker`, `channel_worker::infer_module`, `config`, `pipeline`, `output`, `metadata`.

### Build (on device)

```bash
cd /mnt/workspace/develop/multi_cam_app/build
make -j$(nproc)
# Binary: build/multi-cam-app (NOT build/app/)

# Build custom GStreamer plugins
for plugin in msgagg faceinfer frameoffload deduper timingmark; do
  cd /mnt/workspace/develop/multi_cam_app/build/plugins/$plugin && make -j$(nproc)
done

# Install plugins
GST_DIR=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
sudo cp build/plugins/{msgagg,faceinfer,frameoffload,deduper,timingmark}/libgstqti*.so "$GST_DIR/"
rm -f ~/.cache/gstreamer-1.0/registry.*.bin

# Copy binary for E2E script
cp build/multi-cam-app build/app/multi-cam-app
```

**faceinfer warning**: Do NOT `make clean` in `build/plugins/faceinfer/`. Clean rebuild changes .so linkage and breaks ArcFace embedding consistency.

### Test

```bash
cd /mnt/workspace/develop/multi_cam_app
bash scripts/run_e2e_pipeline.sh    # Full E2E: cleanup + plugins + 3 processes + 60s
bash scripts/run_smoke.sh           # Quick smoke test
bash scripts/run_face_integration.sh  # Face pipeline test
```

**Pre-test cleanup**: `pkill -9 -f multi-cam-app; pkill -9 -f consumer_demo; pkill -9 -f rule_process; rm -f /tmp/dmabuf_ipc.sock /tmp/iq9_rule_relay.sock`

---

## vlm_pipeline/ — VLM-Augmented Pipeline

Extends `multi_cam_app` with Vision-Language Model inference on rule-triggered frames. Has its own `CLAUDE.md` with full details.

### Architecture

```
multi_cam_app (Producer) → DMA-BUF → MQTT scene_update
    ↓
rule_process (Rule Engine) → UDS relay with metadata header
    ↓
consumer_demo (Consumer + VLM)
    MapFrame → NV12→RGB → resize 448x448 → JPEG → HTTP POST
    ↓
VLM Server (genie-t2t-run, port 8000, OpenAI-compatible /v1/chat/completions)
    ↓
consumer_demo: log response to /tmp/vlm_responses.jsonl
```

### Build

```bash
# Consumer with VLM
cmake -S dmabuf_consumer -B dmabuf_consumer/build_local -DCMAKE_BUILD_TYPE=Release -DENABLE_VLM=ON
cmake --build dmabuf_consumer/build_local -j

# Rule process
cmake -S rule_process -B rule_process/build_local -DCMAKE_BUILD_TYPE=Release
cmake --build rule_process/build_local -j
```

### Run

```bash
# Terminal 1: VLM server
cd /mnt/workspace/bundle_iq9 && bash start.sh

# Terminal 2: Pipeline
bash scripts/run_e2e_pipeline.sh
```

---

## qwen3vl/ — Qwen3-VL Full Pipeline

Complete Qwen3 Vision-Language model deployment using Qualcomm Genie SDK on QCS9075 HTP.

### Key Files

| File | Purpose |
|------|---------|
| `bin/genie-t2t-run` | Qualcomm Genie text-to-text inference binary |
| `env.sh` | Environment setup (LD_LIBRARY_PATH, ADSP/CDSP paths) |
| `qwen3vl.json` | Genie inference config (SSD-Q1 speculative decoding, HTP backend) |
| `tokenizer.json` | Qwen3 tokenizer (151,936 vocab) |
| `models/` | Model shards (3-part split, AR128+AR256 configs) |
| `tools/qwen3vl_profile_runner.py` | Profiling automation |

### Run

```bash
cd /mnt/workspace/develop/qwen3vl
source env.sh
bash run_qwen3vl.sh
```

### Specs

- **Backend**: QnnHtp, mmap, poll mode, 3 threads, KV-dim=128
- **Decoding**: SSD-Q1 speculative decoding, context=2048, max 100 tokens, greedy (temp=0.8, top-k=1)
- **DSP**: Dual CDSP support
- **Embedding**: INT8 quantized LUT, 2560 dimensions

---

## qwen3vit/ — Standalone VIT Deployment

Standalone Vision Transformer deployment with OpenCL GPU acceleration. Produces `veg-combined-runner` binary.

### Build

```bash
cd /mnt/workspace/develop/qwen3vit/build
cmake ../src/qwen3vit_deploy && make -j$(nproc)
```

### Key Source

- `src/qwen3vit_deploy/veg_combined_runner.cpp` — Main runner (VIT+embedding+generation)
- `src/qwen3vit_deploy/qwen_vl_preprocessor.cpp` — Image preprocessing
- `src/qwen3vit_deploy/qwen_vl_opencl_accelerator.cpp` — OpenCL GPU acceleration
- `qnn_sdk/include/` — QNN SDK headers (Genie, HTP, GPU, CPU)

---

## Dependencies

- **GStreamer 1.x** with `gst-plugins-base` (appsink)
- **json-glib** for config parsing
- **nlohmann/json** for rule_process MQTT payload parsing
- **Qualcomm IM SDK GStreamer plugins**: `qtimlvconverter`, `qtimlqnn`, `qtimlpostprocess`, `qtiobjtracker`, `qtimsgpub`, `qtideduper`
- **Qualcomm V4L2 decoder**: `v4l2h264dec` with dmabuf I/O mode
- **QNN runtime**: `libQnnHtp.so`, `libQnnSystem.so`
- **Custom GStreamer plugins**: `qtiframeoffload`, `qtimsgagg`, `qtifaceinfer`, `qtideduper`, `qtitimingmark`
- **Paho MQTT C/C++**: `libpaho-mqtt3c`, `libpaho-mqttpp3`
- **mosquitto**: MQTT broker on device (127.0.0.1:1883)
- **VLM deps**: libjpeg-turbo, libcurl, nlohmann-json3
- **Qualcomm Genie SDK**: `genie-t2t-run` binary

## DMA-BUF Pipeline

```
MultiCamApp (Producer)
  qtiframeoffload → SubmitFrame(NV12) → DMA-BUF slot pool (8-32 slots)
  qtimsgagg → qtimsgpub(json=TRUE) → MQTT iq9/scene_update

rule_process
  MQTT subscribe → RuleEngine → UDS relay → consumer_demo
  MQTT publish iq9/frame_done → FrameCacheService → ReleaseSlotDirect

consumer_demo
  MapFrame (mmap dma-buf by slot_index) → WritePpm/VLM → ReleaseFrame
```

**Critical**: qtimsgpub `json=TRUE` wraps output with UNESCAPED inner JSON. rule_process extracts via string search for `{"type":"scene_update"` and brace-counting.

**Slot exhaustion cascade**: When slots fill, SubmitFrame fails → frame_offload drops frame → no msgagg data → rule_process idle → no frame_done → deadlock. Mitigate with `slot_count` increase.

## Coding Conventions

- C++17 with RAII wrappers around GLib/GStreamer C APIs where practical
- Error reporting via `std::string* out_error` output parameters (not exceptions)
- GStreamer element names include channel ID suffix: `{element}_ch{N}`
- All `#include` paths relative to project root (e.g., `"config/AppConfig.hpp"`)
- Header guards: `MULTI_CAM_APP_{PATH}_HPP` pattern
- Atomic counters with `std::memory_order_relaxed` for per-channel statistics
- macOS clang diagnostics are expected — code targets aarch64 Linux, ignore them

## Post-reboot

Custom GStreamer plugins revert to system defaults on device reboot. The E2E script handles reinstall automatically. For manual runs:

```bash
cd /mnt/workspace/develop/multi_cam_app/build/plugins
sudo cp {msgagg,faceinfer,frameoffload,deduper,timingmark}/libgstqti*.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
rm -f ~/.cache/gstreamer-1.0/registry.*.bin
```
