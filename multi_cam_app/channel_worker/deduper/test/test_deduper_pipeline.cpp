/// @file test_deduper_pipeline.cpp
/// @brief Group B: GStreamer pipeline integration test for qtideduper
///
/// Pipeline:
///   filesrc → qtdemux → h264parse → v4l2h264dec(dmabuf) → NV12 capsfilter
///     → tee(allow-not-linked=true)
///       ├→ queue → qtimlvconverter → qtimlqnn(YOLO) → qtimlpostprocess
///       │    → qtiobjtracker → queue → capsfilter(utf8) → metamux.data_0
///       └→ queue → metamux.sink
///                       ↓
///                  metamux.src → qtideduper → appsink
///
/// Usage:
///   GST_DEBUG=qtideduper:6 ./test_deduper_pipeline
///       <video_file> <model_path> <labels_path> [qnn_backend] [qnn_system]
///
/// Example:
///   GST_DEBUG=qtideduper:6 ./test_deduper_pipeline
///       ../channel_worker/deduper/test/data/test_video.mp4
///       ../data/models/yolov11/yolov11_det.bin
///       ../data/models/yolov11/labels.txt

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/gstvideometa.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "pipeline/PipelineUtils.hpp"

using multi_cam_app::pipeline::MakeElement;
using multi_cam_app::pipeline::SetEnumPropertyByNick;

/* --------------- counters + timing --------------- */
static std::atomic<uint64_t> g_total_in{0};
static std::atomic<uint64_t> g_passed{0};

/* Per-buffer timing: measured via sink→src pad probe pair.
 * Only captures PASSED buffers (dropped buffers don't reach src pad).
 * Plugin's own GST_DEBUG logs show timing for ALL buffers (passed+dropped). */
static std::atomic<int64_t>  g_probe_t_start{0};
static std::atomic<uint64_t> g_timing_count{0};
static std::atomic<uint64_t> g_timing_total_us{0};
static std::atomic<uint64_t> g_timing_min_us{UINT64_MAX};
static std::atomic<uint64_t> g_timing_max_us{0};

/* --------------- pad probe: count frames entering deduper + record start time --------------- */
static GstPadProbeReturn
deduper_sink_probe(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer /*user_data*/)
{
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) {
        g_total_in++;
        g_probe_t_start.store(g_get_monotonic_time(), std::memory_order_relaxed);
    }
    return GST_PAD_PROBE_OK;
}

/* --------------- pad probe: measure elapsed time for passed buffers --------------- */
static GstPadProbeReturn
deduper_src_probe(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer /*user_data*/)
{
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) {
        const int64_t t0 = g_probe_t_start.load(std::memory_order_relaxed);
        if (t0 > 0) {
            const uint64_t elapsed = (uint64_t)(g_get_monotonic_time() - t0);
            g_timing_count++;
            g_timing_total_us += elapsed;
            uint64_t cur_min = g_timing_min_us.load(std::memory_order_relaxed);
            while (elapsed < cur_min &&
                   !g_timing_min_us.compare_exchange_weak(cur_min, elapsed)) {}
            uint64_t cur_max = g_timing_max_us.load(std::memory_order_relaxed);
            while (elapsed > cur_max &&
                   !g_timing_max_us.compare_exchange_weak(cur_max, elapsed)) {}
        }
    }
    return GST_PAD_PROBE_OK;
}

/* --------------- appsink callback: log passed frames --------------- */
static GstFlowReturn
on_new_sample(GstAppSink* /*appsink*/, gpointer user_data)
{
    GstAppSink* sink = GST_APP_SINK(user_data);
    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_OK;

    const uint64_t frame_num = ++g_passed;
    GstBuffer* buffer = gst_sample_get_buffer(sample);

    /* --- 读取 ROI meta --- */
    int roi_count = 0;
    gpointer state = nullptr;
    GstMeta* meta;
    while ((meta = gst_buffer_iterate_meta(buffer, &state)) != nullptr) {
        if (meta->info->api != GST_VIDEO_REGION_OF_INTEREST_META_API_TYPE)
            continue;

        auto* roi = reinterpret_cast<GstVideoRegionOfInterestMeta*>(meta);
        const gchar* label = g_quark_to_string(roi->roi_type);

        guint tid = 0;
        for (GList* p = roi->params; p; p = p->next) {
            GstStructure* s = (GstStructure*) p->data;
            gst_structure_get_uint(s, "tracking-id", &tid);
        }

        fprintf(stderr, "  [PASSED frame %lu] ROI: %s#%u @ (%u,%u,%ux%u)\n",
            (unsigned long) frame_num,
            label ? label : "?", tid,
            roi->x, roi->y, roi->w, roi->h);
        roi_count++;
    }

    fprintf(stderr, "[PASSED frame %lu] total_in=%lu, roi_count=%d\n",
        (unsigned long) frame_num,
        (unsigned long) g_total_in.load(),
        roi_count);

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

/* --------------- demux pad-added (video → parse, non-video → fakesink) --------------- */
static void
on_demux_pad_added(GstElement* demux, GstPad* new_pad, gpointer data)
{
    GstElement* parse = (GstElement*) data;
    GstCaps* caps = gst_pad_get_current_caps(new_pad);
    if (!caps) caps = gst_pad_query_caps(new_pad, nullptr);

    const gchar* name = gst_structure_get_name(gst_caps_get_structure(caps, 0));
    if (g_str_has_prefix(name, "video/")) {
        GstPad* sink = gst_element_get_static_pad(parse, "sink");
        if (!gst_pad_is_linked(sink)) {
            gst_pad_link(new_pad, sink);
        }
        gst_object_unref(sink);
    } else {
        GstElement* pipe = GST_ELEMENT(gst_element_get_parent(demux));
        GstElement* fakesink = gst_element_factory_make("fakesink", nullptr);
        gst_bin_add(GST_BIN(pipe), fakesink);
        gst_element_sync_state_with_parent(fakesink);
        GstPad* fs_sink = gst_element_get_static_pad(fakesink, "sink");
        gst_pad_link(new_pad, fs_sink);
        gst_object_unref(fs_sink);
        gst_object_unref(pipe);
    }
    gst_caps_unref(caps);
}

/* --------------- helper: create + check element --------------- */
static GstElement*
make(const char* factory, const char* name)
{
    std::string err;
    GstElement* e = MakeElement(factory, name, &err);
    if (!e) {
        fprintf(stderr, "FATAL: cannot create '%s' (%s): %s\n", factory, name, err.c_str());
        std::exit(1);
    }
    return e;
}

/* --------------- main --------------- */
int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <video_file> <model_path> <labels_path>"
            " [qnn_backend] [qnn_system]\n", argv[0]);
        return 1;
    }

    const char* video_file  = argv[1];
    const char* model_path  = argv[2];
    const char* labels_path = argv[3];
    const char* qnn_backend = argc > 4 ? argv[4] : "/usr/lib/libQnnHtp.so";
    const char* qnn_system  = argc > 5 ? argv[5] : "/usr/lib/libQnnSystem.so";

    gst_init(&argc, &argv);

    fprintf(stderr, "========================================\n");
    fprintf(stderr, "=== Deduper Pipeline Test (Group B) ===\n");
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "video:  %s\n", video_file);
    fprintf(stderr, "model:  %s\n", model_path);
    fprintf(stderr, "labels: %s\n", labels_path);
    fprintf(stderr, "========================================\n\n");

    GstElement* pipeline = gst_pipeline_new("test-deduper");

    /* ---- Source chain ---- */
    auto* filesrc    = make("filesrc",      "filesrc");
    auto* demux      = make("qtdemux",      "demux");
    auto* parse      = make("h264parse",    "parse");
    auto* decode     = make("v4l2h264dec",  "decode");
    auto* capsf_nv12 = make("capsfilter",   "capsf_nv12");

    g_object_set(filesrc, "location", video_file, nullptr);

    std::string err;
    SetEnumPropertyByNick(decode, "capture-io-mode", "dmabuf", &err);
    SetEnumPropertyByNick(decode, "output-io-mode",  "dmabuf", &err);

    GstCaps* nv12_caps = gst_caps_from_string("video/x-raw, format=NV12");
    g_object_set(capsf_nv12, "caps", nv12_caps, nullptr);
    gst_caps_unref(nv12_caps);

    /* ---- Tee ---- */
    auto* tee = make("tee", "tee_decode");
    g_object_set(tee, "allow-not-linked", TRUE, nullptr);

    /* ---- YOLO + Tracker branch ---- */
    auto* queue_yolo  = make("queue",              "queue_yolo");
    auto* converter   = make("qtimlvconverter",    "converter");
    auto* qnn         = make("qtimlqnn",           "qnn_yolo");
    auto* postprocess = make("qtimlpostprocess",   "postprocess");
    auto* tracker     = make("qtiobjtracker",       "tracker");
    auto* queue_meta  = make("queue",              "queue_meta");
    auto* capsf_text  = make("capsfilter",         "capsf_text");

    /* converter engine: try fcv (verified working on QCS9075) */
    SetEnumPropertyByNick(converter, "engine", "fcv", &err);

    g_object_set(qnn,
        "model",   model_path,
        "backend", qnn_backend,
        "system",  qnn_system, nullptr);

    SetEnumPropertyByNick(postprocess, "module", "yolov8", &err);
    g_object_set(postprocess, "labels", labels_path, nullptr);
    g_object_set(postprocess, "settings", "{\"confidence\":40.0}", nullptr);

    SetEnumPropertyByNick(tracker, "algo", "bytetrack", &err);
    g_object_set(tracker, "parameters",
        "params, frame-rate=(int)<30>, track-buffer=(int)<30>, "
        "wh-smooth-factor=(double)<0.9>, track-thresh=(double)<0.4>, "
        "high-thresh=(double)<0.45>", nullptr);

    GstCaps* text_caps = gst_caps_from_string("text/x-raw, format=utf8");
    g_object_set(capsf_text, "caps", text_caps, nullptr);
    gst_caps_unref(text_caps);

    /* ---- Video branch ---- */
    auto* queue_video = make("queue", "queue_video");

    /* ---- Metamux ---- */
    auto* metamux = make("qtimetamux", "metamux");

    /* ---- Deduper ---- */
    auto* deduper = make("qtideduper", "deduper");
    g_object_set(deduper, "iou-threshold", 0.75f, nullptr);

    /* ---- Appsink ---- */
    auto* appsink = make("appsink", "appsink_test");
    g_object_set(appsink,
        "emit-signals", TRUE,
        "sync",         FALSE,
        "drop",         FALSE,
        "max-buffers",  (guint) 4, nullptr);

    /* ---- Add all to pipeline ---- */
    gst_bin_add_many(GST_BIN(pipeline),
        filesrc, demux, parse, decode, capsf_nv12, tee,
        queue_yolo, converter, qnn, postprocess, tracker, queue_meta, capsf_text,
        queue_video,
        metamux, deduper, appsink,
        nullptr);

    /* ---- Link source chain (filesrc → demux は pad-added で接続) ---- */
    gst_element_link(filesrc, demux);
    gst_element_link_many(parse, decode, capsf_nv12, tee, nullptr);

    /* ---- Link YOLO branch ---- */
    gst_element_link_many(queue_yolo, converter, qnn, postprocess,
                          tracker, queue_meta, capsf_text, nullptr);

    /* tee → queue_yolo */
    GstPad* tee_src0 = gst_element_request_pad_simple(tee, "src_%u");
    GstPad* qy_sink  = gst_element_get_static_pad(queue_yolo, "sink");
    gst_pad_link(tee_src0, qy_sink);
    gst_object_unref(tee_src0);
    gst_object_unref(qy_sink);

    /* tee → queue_video */
    GstPad* tee_src1 = gst_element_request_pad_simple(tee, "src_%u");
    GstPad* qv_sink  = gst_element_get_static_pad(queue_video, "sink");
    gst_pad_link(tee_src1, qv_sink);
    gst_object_unref(tee_src1);
    gst_object_unref(qv_sink);

    /* metamux: capsf_text → data_0, queue_video → sink */
    GstPad* mm_data0 = gst_element_request_pad_simple(metamux, "data_%u");
    GstPad* ct_src   = gst_element_get_static_pad(capsf_text, "src");
    gst_pad_link(ct_src, mm_data0);
    gst_object_unref(ct_src);
    gst_object_unref(mm_data0);

    GstPad* mm_sink = gst_element_get_static_pad(metamux, "sink");
    GstPad* qv_src  = gst_element_get_static_pad(queue_video, "src");
    gst_pad_link(qv_src, mm_sink);
    gst_object_unref(qv_src);
    gst_object_unref(mm_sink);

    /* metamux → deduper → appsink */
    gst_element_link_many(metamux, deduper, appsink, nullptr);

    /* ---- Callbacks ---- */
    g_signal_connect(demux, "pad-added", G_CALLBACK(on_demux_pad_added), parse);

    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = on_new_sample;
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink), &callbacks,
                               (gpointer) appsink, nullptr);

    /* ---- Pad probes on deduper sink+src (count + timing) ---- */
    GstPad* deduper_sink = gst_element_get_static_pad(deduper, "sink");
    gst_pad_add_probe(deduper_sink, GST_PAD_PROBE_TYPE_BUFFER,
                      deduper_sink_probe, nullptr, nullptr);
    gst_object_unref(deduper_sink);

    GstPad* deduper_src = gst_element_get_static_pad(deduper, "src");
    gst_pad_add_probe(deduper_src, GST_PAD_PROBE_TYPE_BUFFER,
                      deduper_src_probe, nullptr, nullptr);
    gst_object_unref(deduper_src);

    /* ---- Run ---- */
    fprintf(stderr, "Setting pipeline to PLAYING...\n");
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "FATAL: pipeline failed to start\n");
        gst_object_unref(pipeline);
        return 1;
    }

    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* msg = gst_bus_timed_pop_filtered(bus,
        GST_CLOCK_TIME_NONE,
        (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));

    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* gerr = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(msg, &gerr, &debug);
        fprintf(stderr, "ERROR: %s\n", gerr->message);
        if (debug) fprintf(stderr, "  debug: %s\n", debug);
        g_error_free(gerr);
        g_free(debug);
    }

    gst_message_unref(msg);
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    /* ---- Summary ---- */
    const uint64_t total   = g_total_in.load();
    const uint64_t passed  = g_passed.load();
    const uint64_t dropped = total >= passed ? total - passed : 0;

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "=== DEDUPER PIPELINE TEST SUMMARY ===\n");
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "Total frames into deduper:  %lu\n", (unsigned long) total);
    fprintf(stderr, "Passed frames (to appsink): %lu\n", (unsigned long) passed);
    fprintf(stderr, "Dropped frames (by deduper):%lu\n", (unsigned long) dropped);
    fprintf(stderr, "Drop rate:                  %.1f%%\n",
        total > 0 ? 100.0 * (double) dropped / (double) total : 0.0);
    fprintf(stderr, "----------------------------------------\n");
    const uint64_t tc = g_timing_count.load();
    if (tc > 0) {
        const double avg = (double) g_timing_total_us.load() / (double) tc;
        fprintf(stderr, "Passed-buffer timing (pad probe):\n");
        fprintf(stderr, "  count=%lu  avg=%.1fus  min=%luus  max=%luus\n",
            (unsigned long) tc, avg,
            (unsigned long) g_timing_min_us.load(),
            (unsigned long) g_timing_max_us.load());
    } else {
        fprintf(stderr, "Passed-buffer timing: no data (0 buffers reached src pad)\n");
    }
    fprintf(stderr, "========================================\n");

    /* 基本检查: 至少有一些帧通过（视频有目标时） */
    if (total > 0 && passed == 0) {
        fprintf(stderr, "WARNING: zero frames passed — check video content or plugin deployment\n");
    }
    if (total > 0 && dropped == 0) {
        fprintf(stderr, "WARNING: zero frames dropped — deduper may not be filtering\n");
    }

    return 0;
}
