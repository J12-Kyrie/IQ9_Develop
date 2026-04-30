#include "yolo_chain.hpp"
#include "yolo_latency_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

#include <glib.h>
#include <gst/gst.h>

namespace {

GMainLoop* g_main_loop = nullptr;

// HTP + first-time QNN / decode preroll can take minutes; 30s often returns
// get_state=ASYNC with no ERROR on the bus, which looks like an empty error dump.
// Override with env: SINGLE_PIPELINE_STATE_TIMEOUT_SEC (default 480 = 8 minutes).
static GstClockTime PrerollWaitMax() {
  const char* e = g_getenv("SINGLE_PIPELINE_STATE_TIMEOUT_SEC");
  if (e == nullptr || e[0] == '\0') {
    return 8 * 60 * GST_SECOND;
  }
  char* end = nullptr;
  const gulong sec = g_ascii_strtoull(e, &end, 10);
  if (sec < 1UL || sec > 3600UL) {
    return 8 * 60 * GST_SECOND;
  }
  return static_cast<GstClockTime>(sec) * GST_SECOND;
}

void PrintPipelineState(const char* label, GstElement* pipeline) {
  if (pipeline == nullptr) {
    return;
  }
  GstState cur = GST_STATE_NULL;
  GstState pen = GST_STATE_NULL;
  (void)gst_element_get_state(pipeline, &cur, &pen, 0);
  const char* const pen_s =
      (pen == GST_STATE_VOID_PENDING) ? "void-pending" : gst_element_state_get_name(pen);
  std::fprintf(stderr, "%s: current=%s pending=%s (enum %d, %d)\n", label,
               gst_element_state_get_name(cur), pen_s, static_cast<int>(cur),
               static_cast<int>(pen));
}

void LogOneBusMessage(GstMessage* m) {
  if (m == nullptr) {
    return;
  }
  if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_ERROR) {
    GError* errv = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_error(m, &errv, &dbg);
    const char* en =
        (GST_MESSAGE_SRC(m) != nullptr) ? GST_OBJECT_NAME(GST_MESSAGE_SRC(m)) : "unknown";
    std::fprintf(
        stderr, "[bus] ERROR from [%s]: %s\n", en,
        (errv != nullptr && errv->message != nullptr) ? errv->message : "?");
    if (dbg != nullptr) {
      std::fprintf(stderr, "  debug: %s\n", dbg);
    }
    if (errv != nullptr) {
      g_error_free(errv);
    }
    g_free(dbg);
  } else if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_WARNING) {
    GError* errv = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_warning(m, &errv, &dbg);
    const char* en =
        (GST_MESSAGE_SRC(m) != nullptr) ? GST_OBJECT_NAME(GST_MESSAGE_SRC(m)) : "unknown";
    std::fprintf(
        stderr, "[bus] WARNING from [%s]: %s\n", en,
        (errv != nullptr && errv->message != nullptr) ? errv->message : "?");
    if (dbg != nullptr) {
      std::fprintf(stderr, "  debug: %s\n", dbg);
    }
    if (errv != nullptr) {
      g_error_free(errv);
    }
    g_free(dbg);
  } else if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_STATE_CHANGED) {
    GstState oldst = GST_STATE_NULL;
    GstState newst = GST_STATE_NULL;
    // GStreamer 1.16+ (three-parameter form; also valid on 1.20+ toolchains)
    gst_message_parse_state_changed(m, &oldst, &newst);
    const char* en =
        (GST_MESSAGE_SRC(m) != nullptr) ? GST_OBJECT_NAME(GST_MESSAGE_SRC(m)) : "?";
    std::fprintf(stderr, "[state] %s: %s -> %s\n", en, gst_element_state_get_name(oldst),
                 gst_element_state_get_name(newst));
  } else {
    const char* en = (GST_MESSAGE_SRC(m) != nullptr) ? GST_OBJECT_NAME(GST_MESSAGE_SRC(m)) : "?";
    std::fprintf(stderr, "[bus] (type %d) from [%s]\n", static_cast<int>(GST_MESSAGE_TYPE(m)), en);
  }
}

// Blocking pops first (catch slow errors), then non-blocking drain.
void DumpBusMessages(GstElement* pipeline) {
  GstBus* bus = gst_element_get_bus(pipeline);
  if (bus == nullptr) {
    return;
  }
  for (int k = 0; k < 8; k++) {
    GstMessage* m = gst_bus_timed_pop(bus, 2 * GST_SECOND);
    if (m == nullptr) {
      break;
    }
    LogOneBusMessage(m);
    gst_message_unref(m);
  }
  for (;;) {
    GstMessage* m = gst_bus_timed_pop(bus, 0);
    if (m == nullptr) {
      break;
    }
    LogOneBusMessage(m);
    gst_message_unref(m);
  }
  gst_object_unref(bus);
  std::fflush(stderr);
}

void DumpPipelineStateAndBus(GstElement* pipeline) {
  PrintPipelineState("pipeline", pipeline);
  g_usleep(200 * 1000);
  std::fprintf(stderr, "Draining GStreamer bus (state changes / errors)...\n");
  DumpBusMessages(pipeline);
  std::fprintf(
      stderr,
      "If preroll was still in progress, increase wait: "
      "SINGLE_PIPELINE_STATE_TIMEOUT_SEC=600 ; try 1 video in JSON to isolate load.\n");
  std::fflush(stderr);
}

bool SetPipelineStateAndSync(GstElement* pipeline, GstState target) {
  const GstStateChangeReturn ret = gst_element_set_state(pipeline, target);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    return false;
  }
  if (ret == GST_STATE_CHANGE_SUCCESS) {
    return true;
  }
  if (ret == GST_STATE_CHANGE_NO_PREROLL) {
    return true;
  }
  // ASYNC: preroll / QNN init — wait for completion (not a short 30s slice that ends in ASYNC).
  const GstClockTime maxwait = PrerollWaitMax();
  GstStateChangeReturn wait = gst_element_get_state(pipeline, nullptr, nullptr, maxwait);
  if (wait == GST_STATE_CHANGE_FAILURE) {
    return false;
  }
  if (wait == GST_STATE_CHANGE_SUCCESS) {
    return true;
  }
  if (wait == GST_STATE_CHANGE_NO_PREROLL) {
    return true;
  }
  if (wait == GST_STATE_CHANGE_ASYNC) {
    std::fprintf(
        stderr,
        "get_state still ASYNC after %llu s; preroll or QNN init may be slow or stuck.\n",
        static_cast<unsigned long long>(PrerollWaitMax() / GST_SECOND));
    return false;
  }
  return false;
}

gboolean OnBusMessage(GstBus* /*bus*/, GstMessage* message, gpointer /*user_data*/) {
  if (message == nullptr) {
    return TRUE;
  }
  switch (GST_MESSAGE_TYPE(message)) {
  case GST_MESSAGE_EOS: {
    std::printf("single-yolo-qnn-latency: received EOS, stopping.\n");
    std::fflush(stdout);
    if (g_main_loop != nullptr) {
      g_main_loop_quit(g_main_loop);
    }
    break;
  }
  case GST_MESSAGE_ERROR: {
    GError* err = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_error(message, &err, &dbg);
    const char* name =
        (GST_MESSAGE_SRC(message) != nullptr) ? GST_OBJECT_NAME(GST_MESSAGE_SRC(message)) : "unknown";
    std::fprintf(stderr, "Pipeline ERROR from [%s]: %s\n", name,
                 (err != nullptr && err->message != nullptr) ? err->message : "unknown");
    if (dbg != nullptr) {
      std::fprintf(stderr, "  debug: %s\n", dbg);
    }
    if (g_main_loop != nullptr) {
      g_main_loop_quit(g_main_loop);
    }
    if (err != nullptr) {
      g_error_free(err);
    }
    g_free(dbg);
    break;
  }
  default:
    break;
  }
  return TRUE;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <yolo_qnn_latency_config.json>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string config_path = argv[1];
  single_pipeline::YoloQnnLatencyConfig config {};
  std::string load_error;
  if (!single_pipeline::LoadYoloQnnLatencyConfigFromFile(config_path, &config, &load_error)) {
    std::fprintf(stderr, "Config load failed: %s\n", load_error.c_str());
    return EXIT_FAILURE;
  }

  gst_init(&argc, &argv);

  GstElement* pipeline = gst_pipeline_new("yolo-qnn-latency-pipeline");
  if (pipeline == nullptr) {
    std::fprintf(stderr, "Failed to create pipeline\n");
    return EXIT_FAILURE;
  }

  for (size_t i = 0U; i < config.videos.size(); ++i) {
    std::string chain_error;
    if (!single_pipeline::AddYoloQnnLatencyChain(
            config, static_cast<uint32_t>(i), config.videos[i], GST_BIN(pipeline), &chain_error)) {
      std::fprintf(stderr, "AddYoloQnnLatencyChain ch%zu failed: %s\n", i, chain_error.c_str());
      gst_object_unref(pipeline);
      return EXIT_FAILURE;
    }
  }

  g_main_loop = g_main_loop_new(nullptr, FALSE);
  if (g_main_loop == nullptr) {
    std::fprintf(stderr, "Failed to create GMainLoop\n");
    gst_object_unref(pipeline);
    return EXIT_FAILURE;
  }

  GstBus* bus = gst_element_get_bus(pipeline);
  if (bus == nullptr) {
    std::fprintf(stderr, "Failed to get bus\n");
    g_main_loop_unref(g_main_loop);
    g_main_loop = nullptr;
    gst_object_unref(pipeline);
    return EXIT_FAILURE;
  }
  const guint watch_id = gst_bus_add_watch(bus, OnBusMessage, nullptr);
  gst_object_unref(bus);
  if (watch_id == 0U) {
    std::fprintf(stderr, "Failed to add bus watch\n");
    g_main_loop_unref(g_main_loop);
    g_main_loop = nullptr;
    gst_object_unref(pipeline);
    return EXIT_FAILURE;
  }

  std::printf("output_dir: %s\n", config.output_dir.c_str());
  for (size_t i = 0U; i < config.videos.size(); ++i) {
    std::printf("  latency_ch%zu.csv (channel_id=%zu)\n", i, i);
  }
  std::printf("Running %zu parallel chain(s)...\n", config.videos.size());
  std::fflush(stdout);

  if (!SetPipelineStateAndSync(GST_ELEMENT(pipeline), GST_STATE_PAUSED)) {
    std::fprintf(stderr, "Failed to set pipeline to PAUSED (preroll). Diagnostics:\n");
    DumpPipelineStateAndBus(GST_ELEMENT(pipeline));
    g_source_remove(watch_id);
    g_main_loop_unref(g_main_loop);
    g_main_loop = nullptr;
    gst_object_unref(pipeline);
    return EXIT_FAILURE;
  }
  if (!SetPipelineStateAndSync(GST_ELEMENT(pipeline), GST_STATE_PLAYING)) {
    std::fprintf(stderr, "Failed to set pipeline to PLAYING. Diagnostics:\n");
    DumpPipelineStateAndBus(GST_ELEMENT(pipeline));
    g_source_remove(watch_id);
    g_main_loop_unref(g_main_loop);
    g_main_loop = nullptr;
    gst_object_unref(pipeline);
    return EXIT_FAILURE;
  }

  g_main_loop_run(g_main_loop);

  gst_element_set_state(pipeline, GST_STATE_NULL);
  if (watch_id != 0U) {
    g_source_remove(watch_id);
  }
  g_main_loop_unref(g_main_loop);
  g_main_loop = nullptr;
  gst_object_unref(pipeline);

  std::printf("Done.\n");
  return EXIT_SUCCESS;
}
