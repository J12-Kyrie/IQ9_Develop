# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IQ9 is a multi-camera object detection and tracking application built for **Qualcomm development boards** (QCS9075 / SA8775P / similar) using the **Qualcomm Intelligent Multimedia SDK (IM SDK)**. It processes multiple H.264 video streams in parallel through GStreamer pipelines that perform YOLOv11 inference via QNN (Qualcomm Neural Network) on the HTP (Hexagon Tensor Processor), then tracks objects with ByteTrack, and writes per-frame detection results as JSONL output.

## Architecture

### Data Flow (per channel)

```
filesrc -> qtdemux -> h264parse -> v4l2h264dec (NV12 dmabuf)
  -> qtimlvconverter -> qtimlqnn (YOLOv11) -> qtimlpostprocess (yolov8 module)
  -> qtiobjtracker (ByteTrack) -> text/x-raw capsfilter -> appsink
  -> ChannelWorker::HandleSample -> JsonlWriter
```

All channels share a single `GstPipeline` with one bus watch. The `GMainLoop` runs in the main thread; `appsink` `new-sample` callbacks fire from GStreamer streaming threads.

### Key Components

- **`MultiCamApp`** (`app/`) -- Top-level orchestrator. Owns the shared GstPipeline, GMainLoop, and all ChannelWorkers. Handles bus messages (EOS/ERROR) and lifecycle.
- **`ChannelWorker`** (`channel_worker/`) -- Per-video-stream worker. Connects to an appsink, parses GStreamer serialized metadata (bounding boxes with tracking IDs), resolves labels via a label index, and writes `FrameRecord` to JSONL.
- **`DetectionInferModule`** (`channel_worker/infer_module/`) -- Builds the GStreamer element chain for one channel: demux, decode, ML convert, QNN inference, postprocess, tracker. Handles demux pad-added linking and non-video pad fakesink routing.
- **`ConfigLoader`** (`config/`) -- Parses JSON config using `json-glib`. Resolves relative paths against the config file directory. Validates and clamps channels to a hard limit of 6.
- **`AppConfig`** (`config/`) -- All tuneable parameters: model path, video paths, QNN backend/system libs, tensor names, converter engine order, tracker params, postprocess module, confidence threshold, appsink settings.
- **`JsonlWriter`** (`utils/`) -- Thread-safe JSONL file writer. One line per frame with `channel_id`, `frame_id`, `timestamp_ns`, and detection array (class_id, track_id, label, score, bbox).
- **`PipelineUtils`** (`pipeline/`) -- GStreamer helper functions: element creation with channel-scoped naming (`{base}_ch{id}`), batch add/link, and enum property setting by nick with error reporting.

### Namespace Structure

All code lives under `multi_cam_app::` with sub-namespaces: `app`, `channel_worker`, `channel_worker::infer_module`, `config`, `pipeline`, `output`, `metadata`.

## Configuration

JSON config files live in `multi_cam_app/config/`. Three variants exist:

- **`config.json`** -- Standard 3-channel detection config
- **`config_no_arc.json`** -- Single-channel with face detection enabled (SCRFD model, no ArcFace)
- **`config_face_probe.json`** -- 3-channel with face detection + recognition pipeline (DLC models via SNPE)

### Required config fields
`model_path`, `videos_path` (string array), `labels_path`, `output_dir`

### Key optional fields
`qnn_backend` (default: `/usr/lib/libQnnHtp.so`), `qnn_system`, `qnn_tensors` (default: `["boxes","scores","class_idx"]`), `qtimlvconverter_engine_order` (default: `["fcv","c2d"]`; `gles`/`none` are skipped in headless mode), `postprocess_module` (default: `yolov8`), `confidence` (0-100, default: 40), `max_channels` (hard cap: 6), `qtiobjtracker_parameters` (GStreamer structure string for ByteTrack tuning).

## Dependencies

- **GStreamer 1.x** with `gst-plugins-base` (appsink)
- **json-glib** for config parsing
- **Qualcomm IM SDK GStreamer plugins**: `qtimlvconverter`, `qtimlqnn`, `qtimlpostprocess`, `qtiobjtracker`
- **Qualcomm V4L2 decoder**: `v4l2h264dec` with dmabuf I/O mode
- **QNN runtime**: `libQnnHtp.so`, `libQnnSystem.so`

## Reference Documentation

IMSDK reference manuals are in `docs/imsdk-reference/` (01-08). The Qualcomm CV GStreamer skill definition is at `docs/qualcomm-cv-gstreamer-cpp/SKILL.md` -- follow its verification workflow when adding or modifying GStreamer pipeline elements:
- Verify each plugin property against `gst-inspect-1.0` or the IMSDK reference files before use.
- Never invent API names or plugin properties; check headers and docs first.
- Keep buffer/handle ownership explicit (GStreamer ref-counting rules apply).

## Coding Conventions

- C++17 with RAII wrappers around GLib/GStreamer C APIs where practical
- Error reporting via `std::string* out_error` output parameters (not exceptions)
- GStreamer element names include channel ID suffix: `{element}_ch{N}`
- All `#include` paths are relative to `multi_cam_app/` (e.g., `"config/AppConfig.hpp"`)
- Header guards follow `MULTI_CAM_APP_{PATH}_HPP` pattern
- Atomic counters with `std::memory_order_relaxed` for per-channel sample statistics
- `std::mutex` for state requiring multi-field consistency (frame counter + timestamp)
- Converter engine selection skips `gles`/`none` in headless mode and falls back through configured order
