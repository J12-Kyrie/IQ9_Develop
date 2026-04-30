下面是 **`frame_cache` 目录**、**IQ9 `dmabuf_*`** 与 **`multi_cam_app` 运行时行为** 的关系说明。

---

## 1. `frame_cache/` 与 IQ9 `dmabuf_producer` 的关系

- `multi_cam_app/frame_cache/` 里是 **内嵌的一份 `dmabuf_producer` 源码**（`include/dmabuf_producer/*`、`src/{producer,slot_pool,ipc_sender,nv12_to_rgb}.cpp`），与 `/mnt/workspace/develop/IQ9/dmabuf_producer` **同一套设计**：`dma_heap` 分配槽位、`SCM_RIGHTS` 握手传 fd、每帧 `DMA_BUF_IOCTL_SYNC`、可选 UDS 上的 `IQ9_Task_Packet` / `IQ9_Release_Packet`。
- **`dmabuf_consumer` 不在 `multi_cam_app` 主程序里链接**；它扮演的是 **独立进程**：连 `socket_path`，在握手阶段收 fd、`mmap` 读 RGB，再按需 `ReleaseFrame`。  
  在工程里的关系可以概括为：**`multi_cam_app` = 生产者侧集成；`IQ9/dmabuf_consumer` = 可对接的消费者参考实现**。

---

## 2. “Frame Cache”在配置里指什么

`AppConfig` 里 **FrameCache** = 应用内 **`FrameCacheService`**：内部持有一个共享的 **`DmaBufProducer`**（DMA 槽池 + NV12→RGB24），并在 **与 `rule_process` 相同的 MQTT broker/port 上订阅 `frame_cache.frame_done_topic`**（与 `rule_process` 的 `frame_done_topic` **同键、同 topic 字符串**），解析 `type: "frame_done"` 的 payload，对 `release_entries[].slot_index` 调用 **`ReleaseSlotDirect`**，与 **`SubmitFrame` 内触发的 UDS `PollReleases`** 形成 **双路还槽**（不再依赖单独的定时 `PollReleases` 线程）。

启用 `frame_cache` 时 **必须** `mqtt.enabled: true` 且配置非空的 `frame_done_topic`，且 **MQTT 必须能连上并成功订阅**；否则 **`MultiCamApp::Initialize` 整体失败**。

JSON 里 `frame_cache.enabled` 为真时，才会走下面整条链路。

---

## 3. 在 `multi_cam_app` 里如何启动、如何接到 GStreamer

### 3.1 应用层：`FrameCacheService`（Producer + MQTT `frame_done`）+ 阻塞握手

`MultiCamApp::Initialize` 在 **所有 `ChannelWorker` 建链完成之后**，若开启 `frame_cache`：

1. `std::make_unique<FrameCacheService>()`，用 `DmaBufProducer::Config` + `mqtt` 的 broker/port 与 **`frame_cache.frame_done_topic`** 填 `InitParams`。
2. **`DmaBufProducer::Init()`** 会 `listen` → `accept` → `SendInit`（含所有槽位的 fd），日志里会有 “waiting for consumer on …”；**外部 consumer 必须先连上**，否则初始化失败。
3. **MQTT**：对 `frame_done_topic` 订阅成功（与 `rule_process` 发布的 **`frame_done`** 消息对接）；若连接/订阅失败，**整程序 Init 失败**。
4. 成功后对每个 worker 调用 `SetFrameCacheProducer(frame_cache_service_->Producer())`。

**`FrameCacheService` 生命周期**应覆盖 **整条 pipeline 使用 `Producer` 的周期**：在 `StopPipeline`、pipeline 与 worker 停干净 **之后** 再 `Shutdown` / 析构（当前 `MultiCamApp::StopPipeline` 顺序：**先停 worker / NULL pipeline，再** `frame_cache_service_->Shutdown()`）。

多路 **scene_update** 到 MQTT：在 **`msgagg` + `frame_cache`** 时，各路的 `qtiframeoffload` 使用 **`scene-update-mqtt` + `meta-width`/`meta-height`** 输出与 **`EventAdapter::adaptSceneUpdate`** 兼容的 **单路 `sources[]` 元素**；全局 **`qtimsgagg` 的 `merge-scene-update=true`** 将多路片段合并为 **一条** `type: "scene_update"` 的 JSON（与 `producer_demo` / `EventAdapter` 一致），避免原先 `{` + 多段 + `}` 的非法拼 JSON 形态。

### 3.2 Worker：把指针塞进 `qtiframeoffload`

`ChannelWorker` 在 `frame_cache.enabled` 时从 pipeline 里按名字取出 `frame_offload` 元素；`SetFrameCacheProducer` 里对其执行：

`g_object_set(..., "producer", p, nullptr)`。

```46:51:/mnt/workspace/develop/multi_cam_app/channel_worker/ChannelWorker.hpp
  void SetFrameCacheProducer(dmabuf_producer::DmaBufProducer* p) {
    frame_cache_producer_ = p;
    if (frame_offload_element_ != nullptr && p != nullptr) {
      g_object_set(G_OBJECT(frame_offload_element_), "producer", p, nullptr);
    }
  }
```

初始化顺序是：**先 `ChannelWorker::Initialize`（创建并引用 `frame_offload`）→ 再 `FrameCache` `Init` → 再 `SetFrameCacheProducer`**，因此 `frame_offload_element_` 已非空，`producer` 属性能正确设上。

### 3.3 管线拓扑：`tee` 分流 —— 一路进 Frame Cache，一路进 appsink

`DetectionInferModule::BuildChain` 在 `frame_cache.enabled` 时，在 **deduper 之后** 插入 `tee_deduper`：

- **分支 A**：`queue_fo` → **`qtiframeoffload`** → `fakesink_fo`（未开 `msgagg`）或 `queue_agg`（开 `msgagg` 时接到全局 `qtimsgagg`）。
- **分支 B**：`queue_combined` → **`appsink_combined`**（原有 C++ 回调、JSONL、人脸逻辑等）。

两路来自 **同一 tee**，是 **同一帧的两份 buffer**，互不抢同一条链路上的 caps。

### 3.4 `qtiframeoffload`：NV12 进 Cache，buffer 变成 JSON

核心在 `gst_frame_offload_chain`：

1. `gst_buffer_map` 读 NV12，用 `GstVideoMeta` 取 `stride/offset`。
2. 若 `producer != nullptr`：调用 **`DmaBufProducer::SubmitFrame`**（内部 NV12→RGB 写入 dma 槽、`PollReleases` 回收槽位；若 `relay_mode == false` 还会 **`SendFrame` 发 `IQ9_Task_Packet`**）。
3. **`FrameOffload::ExtractAndSerialize`** 从 buffer 上扫 ROI / 人脸 meta，组 **`FrameRecord`**，其中 **`image_path` 传入的是 Submit 成功时的 `slot_index`**（失败则为 -1）。
4. 把 **同一块 `GstBuffer` 的 memory 换成 UTF-8 JSON**（`text/x-raw`），再 `gst_pad_push` 下游。

因此：**“Frame Cache”在 GStreamer 里的落点 = 在 dedupe 后的同步点，把当前帧写入共享 DMA 槽，并把“检测结果 + 槽索引”打成 JSON**；视频像素本身在这条支路上不再往下传（`fakesink` 吃掉或进 msgagg）。

---

## 4. 与 `dmabuf_consumer` 的衔接方式（含 `relay_mode`）

| 模式 | 行为 |
|------|------|
| **`relay_mode: false`** | 每帧 `SubmitFrame` 会通过 UDS 发 **`IQ9_Task_Packet`**，外部 **`DmaBufConsumer::RecvFrame`** 可与之一一对应，读对应槽的 RGB，再 **`ReleaseFrame`** 把槽还回 producer。 |
| **`relay_mode: true`（配置默认）** | `SubmitFrame` **不发** 每帧 task 包，仅写 DMA 槽；握手时的 fd 仍会发。外部 consumer **不能只靠 UDS 收帧号**，需要 **别路信令**（例如 **`msgagg`/`qtimsgpub` 发出的 JSON 里 `image_path` = slot_index**）才能和槽内容对齐。 |

这与 IQ9 里对 **`relay_mode`** 的设计一致：**控制面可脱离 producer 自带 socket**，适合 MQTT/中继进程转发 meta。

---

## 5. 多通道共池的含义（架构上要注意的一点）

**所有 channel 的 `qtiframeoffload` 共用一个 `DmaBufProducer`**，槽位在进程内全局竞争：

- 并发路数 × 每路帧率 高时，容易出现 **`SubmitFrame` → “no free slots”**（除非外部 consumer 足够快、持续 `Release`，或增大 `slot_count`）。
- **`PollReleases` 只在 `SubmitFrame` 前顺带调用**，没有在独立线程里专门 drain；消费者若不及时还槽，会反压整条 offload 支路（`queue_fo` 为 leaky=2 可丢帧缓解，但槽仍可能被占满）。

---

## 6. 小结

- **`frame_cache/`**：把 **IQ9 同源 `dmabuf_producer`** 编进 `multi_cam_app`，避免单独装库。  
- **Frame Cache 在应用中的职责**：在 **deduper 之后** 用 **`qtiframeoffload`** 把 **NV12 写入 dma_heap 槽（RGB24）**，并把 **检测/人脸 meta + `slot_index`（`image_path`）** 序列化成 JSON；可选 **UDS + `dmabuf_consumer`** 做零拷贝取帧，或通过 **`relay_mode` + MQTT/msgagg** 把索引交给另一进程。  
- **与纯 GStreamer 主链的关系**：**appsink 支路仍是 NV12 + meta**，Frame Cache 支路是 **并行副本**，专供 **跨进程 DMA 共享** 与 **轻量 JSON 侧车**，而不是替换 appsink 的图像路径。