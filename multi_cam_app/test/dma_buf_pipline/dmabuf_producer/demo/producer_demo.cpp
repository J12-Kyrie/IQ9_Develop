#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <nlohmann/json.hpp>
#include <MQTTClient.h>

#include "dmabuf_producer/producer.h"

static inline uint64_t AlignUpU64(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Detection data structures
// ---------------------------------------------------------------------------
struct DetectionRecord {
    std::string class_name;
    int track_id;
    float confidence;
    float bbox[4]; // [x1, y1, x2, y2] normalized
};

struct DetectionEntry {
    uint64_t timestamp_ns;
    int frame_id;
    std::vector<DetectionRecord> detections;
};

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
struct AppContext {
    dmabuf_producer::DmaBufProducer producer;
    GMainLoop* loop;
    GstElement* pipeline;
    GstElement* h264parse;  // target for demux pad-added link

    int width;
    int height;
    int max_frames;
    std::atomic<int> frame_count;
    bool caps_resolved;

    std::unordered_map<int, DetectionEntry> jsonl_index;
    MQTTClient mqtt_client;
    std::string scene_topic;
    std::string frame_done_topic;
    std::atomic<int> decoded_index;
    std::atomic<int> frames_skipped;
    std::atomic<int> frames_submitted;
    std::atomic<int> frame_done_released;
    std::atomic<int> uds_released;
};

// ---------------------------------------------------------------------------
// JSONL loader
// ---------------------------------------------------------------------------
static bool LoadJsonlIndex(const char* path,
                           std::unordered_map<int, DetectionEntry>* out,
                           std::string* out_error) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        if (out_error) {
            *out_error = std::string("failed to open JSONL file: ") + path;
        }
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) {
            continue;
        }

        nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded()) {
            continue;
        }

        DetectionEntry entry;
        entry.timestamp_ns = j.value("timestamp_ns", uint64_t(0));
        entry.frame_id = j.value("frame_id", 0);

        if (j.contains("detections") && j["detections"].is_array()) {
            for (const auto& det : j["detections"]) {
                DetectionRecord rec;
                rec.class_name = det.value("label", std::string());
                rec.track_id = det.value("track_id", 0);
                rec.confidence = det.value("score", 0.0f);

                rec.bbox[0] = 0.0f;
                rec.bbox[1] = 0.0f;
                rec.bbox[2] = 0.0f;
                rec.bbox[3] = 0.0f;
                if (det.contains("bbox") && det["bbox"].is_array() &&
                    det["bbox"].size() >= 4) {
                    for (int i = 0; i < 4; ++i) {
                        rec.bbox[i] = det["bbox"][i].get<float>();
                    }
                }

                entry.detections.push_back(rec);
            }
        }

        (*out)[entry.frame_id] = entry;
    }

    return true;
}

// ---------------------------------------------------------------------------
// MQTT frame_done callback
// ---------------------------------------------------------------------------
static int OnMqttMessage(void* context, char* topicName, int topicLen,
                         MQTTClient_message* message) {
    (void)topicLen;
    AppContext* ctx = static_cast<AppContext*>(context);

    if (ctx != NULL && message != NULL && message->payload != NULL) {
        std::string payload(static_cast<const char*>(message->payload),
                            static_cast<size_t>(message->payloadlen));
        nlohmann::json j = nlohmann::json::parse(payload, nullptr, false);

        if (!j.is_discarded() && j.value("type", std::string()) == "frame_done") {
            if (j.contains("release_entries") && j["release_entries"].is_array()) {
                for (const auto& re : j["release_entries"]) {
                    uint32_t slot_index = re.value("slot_index", uint32_t(0));
                    ctx->producer.ReleaseSlotDirect(slot_index);
                    ctx->frame_done_released.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    MQTTClient_freeMessage(&message);
    MQTTClient_free(topicName);
    return 1;
}

// ---------------------------------------------------------------------------
// Periodic UDS release poll (complements MQTT frame_done path)
// ---------------------------------------------------------------------------
static gboolean OnPollReleases(gpointer user_data) {
    AppContext* ctx = static_cast<AppContext*>(user_data);
    int released = ctx->producer.PollReleases();
    if (released > 0) {
        ctx->uds_released.fetch_add(released, std::memory_order_relaxed);
        fprintf(stderr, "[producer] UDS poll: released %d slot(s)\n", released);
    }
    return TRUE;
}

// ---------------------------------------------------------------------------
// Publish scene_update over MQTT
// ---------------------------------------------------------------------------
static bool PublishSceneUpdate(AppContext* ctx, uint32_t slot_index,
                               uint64_t timestamp_ns,
                               const DetectionEntry& entry) {
    nlohmann::json events = nlohmann::json::array();
    for (size_t i = 0; i < entry.detections.size(); ++i) {
        const DetectionRecord& d = entry.detections[i];
        nlohmann::json ev;
        ev["class_name"] = d.class_name;
        ev["track_id"] = d.track_id;
        ev["confidence"] = d.confidence;
        ev["bbox"] = { d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3] };
        events.push_back(ev);
    }

    nlohmann::json source;
    source["source_id"] = 0;
    source["timestamp_ns"] = timestamp_ns;

    nlohmann::json image_meta;
    image_meta["slot_index"] = slot_index;
    image_meta["width"] = ctx->width;
    image_meta["height"] = ctx->height;
    image_meta["channels"] = 3;
    source["image_meta"] = image_meta;

    source["events"] = events;
    source["removed_track_ids"] = nlohmann::json::array();

    nlohmann::json msg;
    msg["type"] = "scene_update";
    msg["sources"] = nlohmann::json::array({ source });

    std::string payload = msg.dump();

    MQTTClient_message pubmsg = MQTTClient_message_initializer;
    pubmsg.payload = const_cast<char*>(payload.c_str());
    pubmsg.payloadlen = static_cast<int>(payload.size());
    pubmsg.qos = 0;
    pubmsg.retained = 0;

    MQTTClient_deliveryToken token;
    int rc = MQTTClient_publishMessage(ctx->mqtt_client,
                                       ctx->scene_topic.c_str(),
                                       &pubmsg, &token);
    return rc == MQTTCLIENT_SUCCESS;
}

// ---------------------------------------------------------------------------
// Demux pad-added handler (reuses IQ9 DetectionInferModule pattern)
// ---------------------------------------------------------------------------
static gboolean IsVideoH264Pad(GstPad* pad) {
    const gchar* pad_name = GST_OBJECT_NAME(pad);
    if (pad_name == NULL || !g_str_has_prefix(pad_name, "video")) {
        return FALSE;
    }

    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (caps == NULL) {
        caps = gst_pad_query_caps(pad, NULL);
    }
    if (caps == NULL || gst_caps_get_size(caps) == 0) {
        if (caps != NULL) {
            gst_caps_unref(caps);
        }
        return TRUE;  // assume video if no caps yet
    }

    gboolean is_h264 = FALSE;
    guint n = gst_caps_get_size(caps);
    for (guint i = 0; i < n; ++i) {
        const GstStructure* s = gst_caps_get_structure(caps, i);
        if (s != NULL && g_strcmp0(gst_structure_get_name(s), "video/x-h264") == 0) {
            is_h264 = TRUE;
            break;
        }
    }
    gst_caps_unref(caps);
    return is_h264;
}

static void OnDemuxPadAdded(GstElement* element, GstPad* pad, gpointer user_data) {
    if (IsVideoH264Pad(pad)) {
        GstElement* parser = static_cast<GstElement*>(user_data);
        if (parser == NULL) {
            return;
        }

        GstPad* sinkpad = gst_element_get_static_pad(parser, "sink");
        if (sinkpad == NULL) {
            return;
        }

        if (gst_pad_is_linked(sinkpad)) {
            gst_object_unref(sinkpad);
            return;
        }

        gst_pad_link(pad, sinkpad);
        gst_object_unref(sinkpad);
    } else {
        // Route non-video pads to fakesink to prevent not-linked errors
        GstElement* parent = GST_ELEMENT(gst_element_get_parent(element));
        if (parent != NULL) {
            GstElement* fakesink = gst_element_factory_make("fakesink", NULL);
            if (fakesink != NULL) {
                g_object_set(G_OBJECT(fakesink), "sync", FALSE, NULL);
                gst_bin_add(GST_BIN(parent), fakesink);
                GstPad* sink_pad = gst_element_get_static_pad(fakesink, "sink");
                if (sink_pad != NULL) {
                    gst_pad_link(pad, sink_pad);
                    gst_object_unref(sink_pad);
                }
                gst_element_sync_state_with_parent(fakesink);
            }
            gst_object_unref(parent);
        }
    }
}

// ---------------------------------------------------------------------------
// Resolve width/height from appsink negotiated caps
// ---------------------------------------------------------------------------
static bool ResolveCaps(GstElement* appsink, int* out_width, int* out_height) {
    GstPad* pad = gst_element_get_static_pad(appsink, "sink");
    if (pad == NULL) {
        return false;
    }

    GstCaps* caps = gst_pad_get_current_caps(pad);
    gst_object_unref(pad);
    if (caps == NULL) {
        return false;
    }

    const GstStructure* s = gst_caps_get_structure(caps, 0);
    if (s == NULL) {
        gst_caps_unref(caps);
        return false;
    }

    gint w = 0, h = 0;
    gboolean ok = gst_structure_get_int(s, "width", &w) &&
                  gst_structure_get_int(s, "height", &h);
    gst_caps_unref(caps);

    if (!ok || w <= 0 || h <= 0) {
        return false;
    }

    *out_width = w;
    *out_height = h;
    return true;
}

// ---------------------------------------------------------------------------
// Appsink new-sample callback
// ---------------------------------------------------------------------------
static GstFlowReturn OnNewSample(GstElement* appsink, gpointer user_data) {
    AppContext* ctx = static_cast<AppContext*>(user_data);
    if (ctx == NULL || appsink == NULL) {
        return GST_FLOW_ERROR;
    }

    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    if (sample == NULL) {
        return GST_FLOW_ERROR;
    }

    // Resolve caps on first frame
    if (!ctx->caps_resolved) {
        if (!ResolveCaps(appsink, &ctx->width, &ctx->height)) {
            fprintf(stderr, "[producer] failed to resolve caps\n");
            gst_sample_unref(sample);
            return GST_FLOW_ERROR;
        }
        fprintf(stderr, "[producer] resolved caps: %dx%d\n",
                ctx->width, ctx->height);
        ctx->caps_resolved = true;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (buffer == NULL) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    uint64_t timestamp_ns = GST_BUFFER_PTS(buffer);
    int frame_idx = ctx->decoded_index.fetch_add(1, std::memory_order_relaxed);

    // Selective submit: match by sequential frame_id (JSONL uses wall-clock
    // timestamp_ns, not GST_BUFFER_PTS, so we match by frame position)
    auto it = ctx->jsonl_index.find(frame_idx);
    if (it == ctx->jsonl_index.end()) {
        ctx->frames_skipped.fetch_add(1, std::memory_order_relaxed);
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    const DetectionEntry& entry = it->second;

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        fprintf(stderr, "[producer] gst_buffer_map failed\n");
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    // Extract stride and UV offset from GstVideoMeta (NOT from caps)
    GstVideoMeta* vmeta = gst_buffer_get_video_meta(buffer);
    int y_stride = vmeta ? static_cast<int>(vmeta->stride[0]) : ctx->width;
    int uv_offset = vmeta ? static_cast<int>(vmeta->offset[1])
                          : (y_stride * ((ctx->height + 31) & ~31));

    uint32_t slot_index = 0;
    std::string error;
    bool ok = ctx->producer.SubmitFrame(
        map.data, ctx->width, ctx->height, y_stride, uv_offset,
        timestamp_ns, &error, &slot_index);

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    if (ok) {
        ctx->frames_submitted.fetch_add(1, std::memory_order_relaxed);
        int frame_num = ctx->frame_count.fetch_add(1, std::memory_order_relaxed) + 1;
        uint64_t slot_stride = AlignUpU64(
            static_cast<uint64_t>(ctx->width) *
            static_cast<uint64_t>(ctx->height) * 3ULL, 4096ULL);
        uint64_t slot_offset = static_cast<uint64_t>(slot_index) * slot_stride;
        fprintf(stderr, "[producer] frame=%d slot=%u stride=%lu offset=%lu %dx%d ts=%lu\n",
                frame_num, slot_index,
                static_cast<unsigned long>(slot_stride),
                static_cast<unsigned long>(slot_offset),
                ctx->width, ctx->height,
                static_cast<unsigned long>(timestamp_ns));

        usleep(30000);

        if (!PublishSceneUpdate(ctx, slot_index, timestamp_ns, entry)) {
            fprintf(stderr, "[producer] MQTT publish failed, releasing slot %u\n",
                    slot_index);
            ctx->producer.ReleaseSlotDirect(slot_index);
        }

        // Check frame limit
        if (ctx->max_frames > 0 && frame_num >= ctx->max_frames) {
            fprintf(stderr, "[producer] frame limit reached (%d), quitting\n",
                    ctx->max_frames);
            g_main_loop_quit(ctx->loop);
            return GST_FLOW_EOS;
        }
    } else {
        fprintf(stderr, "[producer] SubmitFrame failed: %s\n", error.c_str());
    }

    return GST_FLOW_OK;
}

// ---------------------------------------------------------------------------
// Bus watch callback
// ---------------------------------------------------------------------------
static gboolean OnBusMessage(GstBus* bus, GstMessage* msg, gpointer user_data) {
    AppContext* ctx = static_cast<AppContext*>(user_data);
    (void)bus;

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        fprintf(stderr, "[producer] EOS received\n");
        g_main_loop_quit(ctx->loop);
        break;

    case GST_MESSAGE_ERROR: {
        GError* err = NULL;
        gchar* debug = NULL;
        gst_message_parse_error(msg, &err, &debug);
        fprintf(stderr, "[producer] ERROR: %s\n", err ? err->message : "unknown");
        if (debug != NULL) {
            fprintf(stderr, "[producer] DEBUG: %s\n", debug);
        }
        if (err != NULL) g_error_free(err);
        if (debug != NULL) g_free(debug);
        g_main_loop_quit(ctx->loop);
        break;
    }

    default:
        break;
    }

    return TRUE;
}

// ---------------------------------------------------------------------------
// SetEnumPropertyByNick (local helper, matches PipelineUtils pattern)
// ---------------------------------------------------------------------------
static bool SetEnumPropertyByNick(GstElement* element, const char* prop_name,
                                  const char* nick) {
    GObjectClass* klass = G_OBJECT_GET_CLASS(element);
    GParamSpec* pspec = g_object_class_find_property(klass, prop_name);
    if (pspec == NULL || !G_IS_PARAM_SPEC_ENUM(pspec)) {
        fprintf(stderr, "[producer] property '%s' not found or not enum\n", prop_name);
        return false;
    }

    GEnumClass* enum_class = G_ENUM_CLASS(g_type_class_ref(pspec->value_type));
    if (enum_class == NULL) {
        return false;
    }

    GEnumValue* val = g_enum_get_value_by_nick(enum_class, nick);
    if (val == NULL) {
        fprintf(stderr, "[producer] enum nick '%s' not found for property '%s'\n",
                nick, prop_name);
        g_type_class_unref(enum_class);
        return false;
    }

    g_object_set(G_OBJECT(element), prop_name, val->value, NULL);
    g_type_class_unref(enum_class);
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <video.mp4> <channel_0.jsonl> [socket_path] "
            "[mqtt_broker] [scene_topic] [frame_done_topic] [num_frames]\n",
            argv[0]);
        return 1;
    }

    const char* video_path = argv[1];
    const char* jsonl_path = argv[2];
    const char* socket_path = (argc > 3) ? argv[3] : "/tmp/dmabuf_ipc.sock";
    const char* mqtt_broker = (argc > 4) ? argv[4] : "tcp://localhost:1883";
    const char* scene_topic = (argc > 5) ? argv[5] : "iq9/scene_update";
    const char* frame_done_topic = (argc > 6) ? argv[6] : "iq9/frame_done";
    int num_frames = (argc > 7) ? atoi(argv[7]) : 100;

    gst_init(&argc, &argv);

    AppContext ctx;
    ctx.loop = NULL;
    ctx.pipeline = NULL;
    ctx.h264parse = NULL;
    ctx.width = 0;
    ctx.height = 0;
    ctx.max_frames = num_frames;
    ctx.frame_count.store(0, std::memory_order_relaxed);
    ctx.caps_resolved = false;
    ctx.mqtt_client = NULL;
    ctx.scene_topic = scene_topic;
    ctx.frame_done_topic = frame_done_topic;
    ctx.decoded_index.store(0, std::memory_order_relaxed);
    ctx.frames_skipped.store(0, std::memory_order_relaxed);
    ctx.frames_submitted.store(0, std::memory_order_relaxed);
    ctx.frame_done_released.store(0, std::memory_order_relaxed);
    ctx.uds_released.store(0, std::memory_order_relaxed);

    // --- Load JSONL index ---
    std::string error;
    if (!LoadJsonlIndex(jsonl_path, &ctx.jsonl_index, &error)) {
        fprintf(stderr, "[producer] %s\n", error.c_str());
        return 1;
    }
    fprintf(stderr, "[producer] loaded %zu detection entries from %s\n",
            ctx.jsonl_index.size(), jsonl_path);

    // --- Init producer (blocks until consumer connects) ---
    fprintf(stderr, "[producer] waiting for consumer to connect on '%s'...\n",
            socket_path);

    dmabuf_producer::DmaBufProducer::Config prod_cfg;
    memset(&prod_cfg, 0, sizeof(prod_cfg));
    prod_cfg.socket_path = socket_path;
    prod_cfg.heap_path = "/dev/dma_heap/qcom,system";
    prod_cfg.slot_count = 6;
    prod_cfg.relay_mode = true;
    // We don't know width/height yet; use 1920x1080 as initial allocation.
    // The CMA slot must be large enough for the largest frame.
    prod_cfg.width = 1920;
    prod_cfg.height = 1080;

    if (!ctx.producer.Init(prod_cfg, &error)) {
        fprintf(stderr, "[producer] Init failed: %s\n", error.c_str());
        return 1;
    }
    fprintf(stderr, "[producer] consumer connected, starting pipeline\n");

    // --- MQTT setup ---
    int mqtt_rc = MQTTClient_create(&ctx.mqtt_client, mqtt_broker,
                                    "iq9_producer",
                                    MQTTCLIENT_PERSISTENCE_NONE, NULL);
    if (mqtt_rc != MQTTCLIENT_SUCCESS) {
        fprintf(stderr, "[producer] MQTTClient_create failed: %d\n", mqtt_rc);
        ctx.producer.Shutdown();
        return 1;
    }

    MQTTClient_setCallbacks(ctx.mqtt_client, &ctx, NULL, OnMqttMessage, NULL);

    MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
    conn_opts.keepAliveInterval = 20;
    conn_opts.cleansession = 1;

    mqtt_rc = MQTTClient_connect(ctx.mqtt_client, &conn_opts);
    if (mqtt_rc != MQTTCLIENT_SUCCESS) {
        fprintf(stderr, "[producer] MQTT connect failed: %d\n", mqtt_rc);
        MQTTClient_destroy(&ctx.mqtt_client);
        ctx.producer.Shutdown();
        return 1;
    }
    fprintf(stderr, "[producer] MQTT connected to %s\n", mqtt_broker);

    mqtt_rc = MQTTClient_subscribe(ctx.mqtt_client, frame_done_topic, 1);
    if (mqtt_rc != MQTTCLIENT_SUCCESS) {
        fprintf(stderr, "[producer] MQTT subscribe to '%s' failed: %d\n",
                frame_done_topic, mqtt_rc);
        MQTTClient_disconnect(ctx.mqtt_client, 1000);
        MQTTClient_destroy(&ctx.mqtt_client);
        ctx.producer.Shutdown();
        return 1;
    }
    fprintf(stderr, "[producer] subscribed to '%s'\n", frame_done_topic);

    // --- Build GStreamer pipeline ---
    ctx.pipeline = gst_pipeline_new("producer_pipeline");
    GstBin* bin = GST_BIN(ctx.pipeline);

    GstElement* filesrc = gst_element_factory_make("filesrc", "filesrc");
    GstElement* qtdemux = gst_element_factory_make("qtdemux", "qtdemux");
    GstElement* h264parse = gst_element_factory_make("h264parse", "h264parse");
    GstElement* decoder = gst_element_factory_make("v4l2h264dec", "v4l2h264dec");
    GstElement* capsfilter = gst_element_factory_make("capsfilter", "nv12_caps");
    GstElement* appsink = gst_element_factory_make("appsink", "appsink");

    if (!filesrc || !qtdemux || !h264parse || !decoder || !capsfilter || !appsink) {
        fprintf(stderr, "[producer] failed to create GStreamer elements\n");
        MQTTClient_disconnect(ctx.mqtt_client, 1000);
        MQTTClient_destroy(&ctx.mqtt_client);
        ctx.producer.Shutdown();
        return 1;
    }

    gst_bin_add_many(bin, filesrc, qtdemux, h264parse, decoder,
                     capsfilter, appsink, NULL);

    // filesrc -> qtdemux (demux pad-added links to h264parse dynamically)
    if (!gst_element_link(filesrc, qtdemux)) {
        fprintf(stderr, "[producer] failed to link filesrc -> qtdemux\n");
        MQTTClient_disconnect(ctx.mqtt_client, 1000);
        MQTTClient_destroy(&ctx.mqtt_client);
        ctx.producer.Shutdown();
        gst_object_unref(ctx.pipeline);
        return 1;
    }

    ctx.h264parse = h264parse;
    g_signal_connect(qtdemux, "pad-added", G_CALLBACK(OnDemuxPadAdded), h264parse);

    // h264parse -> decoder -> capsfilter -> appsink
    if (!gst_element_link_many(h264parse, decoder, capsfilter, appsink, NULL)) {
        fprintf(stderr, "[producer] failed to link h264parse -> decoder -> capsfilter -> appsink\n");
        MQTTClient_disconnect(ctx.mqtt_client, 1000);
        MQTTClient_destroy(&ctx.mqtt_client);
        ctx.producer.Shutdown();
        gst_object_unref(ctx.pipeline);
        return 1;
    }

    // --- Configure elements ---
    g_object_set(G_OBJECT(filesrc), "location", video_path, NULL);

    // v4l2h264dec: capture-io-mode=dmabuf, output-io-mode=dmabuf
    if (!SetEnumPropertyByNick(decoder, "capture-io-mode", "dmabuf")) {
        fprintf(stderr, "[producer] warning: could not set capture-io-mode=dmabuf\n");
    }
    if (!SetEnumPropertyByNick(decoder, "output-io-mode", "dmabuf")) {
        fprintf(stderr, "[producer] warning: could not set output-io-mode=dmabuf\n");
    }

    // NV12 capsfilter
    GstCaps* nv12_caps = gst_caps_new_simple("video/x-raw",
                                              "format", G_TYPE_STRING, "NV12",
                                              NULL);
    g_object_set(G_OBJECT(capsfilter), "caps", nv12_caps, NULL);
    gst_caps_unref(nv12_caps);

    // Appsink: emit-signals=TRUE, sync=FALSE, drop=FALSE
    g_object_set(G_OBJECT(appsink),
                 "emit-signals", TRUE,
                 "sync", FALSE,
                 "drop", FALSE,
                 NULL);

    g_signal_connect(appsink, "new-sample", G_CALLBACK(OnNewSample), &ctx);

    // --- Bus watch ---
    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(ctx.pipeline));
    gst_bus_add_watch(bus, OnBusMessage, &ctx);
    gst_object_unref(bus);

    // --- Start pipeline ---
    ctx.loop = g_main_loop_new(NULL, FALSE);
    g_timeout_add(50, OnPollReleases, &ctx);

    GstStateChangeReturn ret = gst_element_set_state(ctx.pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "[producer] failed to set pipeline to PLAYING\n");
        MQTTClient_disconnect(ctx.mqtt_client, 1000);
        MQTTClient_destroy(&ctx.mqtt_client);
        ctx.producer.Shutdown();
        gst_object_unref(ctx.pipeline);
        g_main_loop_unref(ctx.loop);
        return 1;
    }

    fprintf(stderr, "[producer] pipeline started, processing up to %d frames\n",
            num_frames);

    g_main_loop_run(ctx.loop);

    int outstanding = ctx.frames_submitted.load(std::memory_order_relaxed)
                    - ctx.frame_done_released.load(std::memory_order_relaxed)
                    - ctx.uds_released.load(std::memory_order_relaxed);
    for (int i = 0; i < 10 && outstanding > 0; ++i) {
        usleep(50000);
        int n = ctx.producer.PollReleases();
        if (n > 0) {
            ctx.uds_released.fetch_add(n, std::memory_order_relaxed);
            outstanding -= n;
            fprintf(stderr, "[producer] UDS drain: released %d slot(s)\n", n);
        }
    }

    // --- Cleanup ---
    fprintf(stderr, "[producer] shutting down - submitted=%d skipped=%d mqtt_released=%d uds_released=%d\n",
            ctx.frames_submitted.load(std::memory_order_relaxed),
            ctx.frames_skipped.load(std::memory_order_relaxed),
            ctx.frame_done_released.load(std::memory_order_relaxed),
            ctx.uds_released.load(std::memory_order_relaxed));

    gst_element_set_state(ctx.pipeline, GST_STATE_NULL);
    gst_object_unref(ctx.pipeline);
    g_main_loop_unref(ctx.loop);

    MQTTClient_disconnect(ctx.mqtt_client, 1000);
    MQTTClient_destroy(&ctx.mqtt_client);

    ctx.producer.Shutdown();

    return 0;
}
