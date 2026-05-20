# CLAUDE.md

VLM-augmented DMA-BUF pipeline for IQ9. Extends the standard `multi_cam_app` pipeline with VLM (Vision-Language Model) inference on rule-triggered frames.

## Pipeline Architecture

```
multi_cam_app (Producer, GStreamer YOLO+ByteTrack)
  qtiframeoffload → SubmitFrame(NV12) → DMA-BUF slot pool
  qtimsgagg → qtimsgpub → MQTT scene_update
      ↓
rule_process (Rule Engine)
  MQTT subscribe → RuleEngine → UDS relay
      ↓ [uint32_t meta_len][JSON metadata][IQ9_Task_Packet]
consumer_demo (Consumer + VLM)
  MapFrame → RGB24 → resize 448x448 → JPEG → HTTP POST
      ↓
VLM HTTP Server (genie-t2t-run on port 8000)
  OpenAI-compatible /v1/chat/completions
      ↓
consumer_demo: log response to JSONL
```

## Layout

| Path | Role |
|------|------|
| `dmabuf_consumer/` | `consumer_demo` with VLM integration (`-DENABLE_VLM=ON`) |
| `rule_process/` | `rule_process`, MQTT + rules + relay (sends metadata header) |
| `config/` | `rule_process_config.json`, `vlm_config.json` |
| `scripts/` | `run_e2e_pipeline.sh` |

**Producer** is `multi-cam-app` at `/mnt/workspace/develop/multi_cam_app/build/app/multi-cam-app` (not built here).

## VLM Integration

### Build

```bash
# Consumer with VLM
cmake -S dmabuf_consumer -B dmabuf_consumer/build_local -DCMAKE_BUILD_TYPE=Release -DENABLE_VLM=ON
cmake --build dmabuf_consumer/build_local -j

# Rule process
cmake -S rule_process -B rule_process/build_local -DCMAKE_BUILD_TYPE=Release
cmake --build rule_process/build_local -j
```

### VLM Dependencies (device)

```bash
apt install libjpeg-turbo8-dev libturbojpeg0-dev libcurl4-openssl-dev nlohmann-json3-dev
```

### VLM Headers

| File | Function |
|------|----------|
| `dmabuf_consumer/include/vlm/jpeg_encoder.h` | libjpeg-turbo wrapper + 448x448 resize |
| `dmabuf_consumer/include/vlm/vlm_client.h` | libcurl per-call CURL* + OpenAI JSON |
| `dmabuf_consumer/include/vlm/thread_pool.h` | Bounded thread pool: submit/drain/evict_oldest |
| `dmabuf_consumer/include/vlm/jsonl_logger.h` | Thread-safe JSONL writer |

### Protocol Extension

`rule_process` sends metadata header before each `IQ9_Task_Packet`:
```
[uint32_t meta_len][JSON: {"rule", "source_id", "detections"}][IQ9_Task_Packet]
```
`consumer_demo` uses `MSG_PEEK` dispatch for backward compatibility (old rule_process without metadata still works).

## Running

```bash
# Terminal 1: Start VLM server
cd /mnt/workspace/bundle_iq9 && bash start.sh

# Terminal 2: Run pipeline
bash scripts/run_e2e_pipeline.sh
```

Results: `/tmp/vlm_responses.jsonl`

## IPC Sockets

| Socket | Purpose |
|--------|---------|
| `/tmp/dmabuf_ipc.sock` | Producer → Consumer (DMA-BUF FD via SCM_RIGHTS) |
| `/tmp/iq9_rule_relay.sock` | Rule Process → Consumer (IQ9_Task_Packet + metadata) |
