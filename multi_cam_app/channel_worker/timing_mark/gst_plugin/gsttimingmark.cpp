#include "gsttimingmark.h"

#include <new>
#include <time.h>

#include <string>

#include "utils/perf/LatencyRecorder.hpp"
#include "utils/perf/TimingMarkRuntime.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_timing_mark_debug);
#define GST_CAT_DEFAULT gst_timing_mark_debug

enum {
  PROP_0,
  PROP_CHANNEL_ID,
  PROP_STAGE_ID,
  PROP_OUTPUT_DIR,
  PROP_SAMPLE_EVERY_N,
  PROP_FLUSH_PER_SAMPLE,
};

struct _GstTimingMarkPrivate {
  guint channel_id {0U};
  guint stage_id {0U};
  std::string output_dir {};
  guint sample_every_n {1U};
  gboolean flush_per_sample {TRUE};
  gboolean registered {FALSE};
};

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS_ANY);

#define gst_timing_mark_parent_class parent_class
G_DEFINE_TYPE_WITH_PRIVATE(GstTimingMark, gst_timing_mark, GST_TYPE_BASE_TRANSFORM)

namespace {

uint64_t MonotonicNowNs() {
  timespec ts {};
#ifdef CLOCK_MONOTONIC_RAW
  constexpr clockid_t kClockId = CLOCK_MONOTONIC_RAW;
#else
  constexpr clockid_t kClockId = CLOCK_MONOTONIC;
#endif
  if (clock_gettime(kClockId, &ts) != 0) {
    return 0ULL;
  }
  return (static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL) +
         static_cast<uint64_t>(ts.tv_nsec);
}

bool IsValidStageId(guint stage_id) {
  return stage_id <= static_cast<guint>(
      multi_cam_app::perf::LatencyStage::kFaceInferSrc);
}

}  // namespace

static void gst_timing_mark_set_property(GObject* object,
                                         guint prop_id,
                                         const GValue* value,
                                         GParamSpec* pspec);
static void gst_timing_mark_get_property(GObject* object,
                                         guint prop_id,
                                         GValue* value,
                                         GParamSpec* pspec);
static void gst_timing_mark_finalize(GObject* object);
static gboolean gst_timing_mark_start(GstBaseTransform* base);
static gboolean gst_timing_mark_stop(GstBaseTransform* base);
static GstFlowReturn gst_timing_mark_transform_ip(GstBaseTransform* base,
                                                  GstBuffer* buffer);

static void gst_timing_mark_class_init(GstTimingMarkClass* klass) {
  GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass* transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  gobject_class->set_property = gst_timing_mark_set_property;
  gobject_class->get_property = gst_timing_mark_get_property;
  gobject_class->finalize = gst_timing_mark_finalize;

  g_object_class_install_property(
      gobject_class,
      PROP_CHANNEL_ID,
      g_param_spec_uint("channel-id",
                        "Channel ID",
                        "Logical channel id for latency aggregation",
                        0U,
                        G_MAXUINT,
                        0U,
                        static_cast<GParamFlags>(G_PARAM_READWRITE |
                                                 G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class,
      PROP_STAGE_ID,
      g_param_spec_uint("stage-id",
                        "Stage ID",
                        "LatencyStage enum value to submit for this marker",
                        0U,
                        G_MAXUINT,
                        0U,
                        static_cast<GParamFlags>(G_PARAM_READWRITE |
                                                 G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class,
      PROP_OUTPUT_DIR,
      g_param_spec_string("output-dir",
                          "Output Dir",
                          "Directory where latency_ch*.csv is written",
                          "",
                          static_cast<GParamFlags>(G_PARAM_READWRITE |
                                                   G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class,
      PROP_SAMPLE_EVERY_N,
      g_param_spec_uint("sample-every-n",
                        "Sample Every N",
                        "Sample every Nth metric per channel recorder",
                        1U,
                        G_MAXUINT,
                        1U,
                        static_cast<GParamFlags>(G_PARAM_READWRITE |
                                                 G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class,
      PROP_FLUSH_PER_SAMPLE,
      g_param_spec_boolean("flush-per-sample",
                           "Flush Per Sample",
                           "Flush CSV output after every emitted sample",
                           TRUE,
                           static_cast<GParamFlags>(G_PARAM_READWRITE |
                                                    G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata(
      element_class,
      "QTI Timing Mark",
      "Filter/Analyzer/Generic",
      "Low-overhead latency timing marker",
      "multi_cam_app");
  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);

  transform_class->start = GST_DEBUG_FUNCPTR(gst_timing_mark_start);
  transform_class->stop = GST_DEBUG_FUNCPTR(gst_timing_mark_stop);
  transform_class->transform_ip =
      GST_DEBUG_FUNCPTR(gst_timing_mark_transform_ip);
  transform_class->transform_ip_on_passthrough = TRUE;
  transform_class->passthrough_on_same_caps = TRUE;

  GST_DEBUG_CATEGORY_INIT(
      gst_timing_mark_debug, "qtitimingmark", 0, "QTI Timing Mark");
}

static void gst_timing_mark_init(GstTimingMark* self) {
  self->priv = static_cast<GstTimingMarkPrivate*>(
      gst_timing_mark_get_instance_private(self));
  new (self->priv) GstTimingMarkPrivate();

  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), TRUE);
}

static void gst_timing_mark_finalize(GObject* object) {
  GstTimingMark* self = GST_TIMING_MARK(object);
  if (self->priv != nullptr) {
    self->priv->~GstTimingMarkPrivate();
  }
  G_OBJECT_CLASS(parent_class)->finalize(object);
}

static void gst_timing_mark_set_property(GObject* object,
                                         guint prop_id,
                                         const GValue* value,
                                         GParamSpec* pspec) {
  GstTimingMark* self = GST_TIMING_MARK(object);
  GstTimingMarkPrivate* priv = self->priv;

  switch (prop_id) {
    case PROP_CHANNEL_ID:
      priv->channel_id = g_value_get_uint(value);
      break;
    case PROP_STAGE_ID:
      priv->stage_id = g_value_get_uint(value);
      break;
    case PROP_OUTPUT_DIR:
      priv->output_dir =
          (g_value_get_string(value) != nullptr) ? g_value_get_string(value) : "";
      break;
    case PROP_SAMPLE_EVERY_N:
      priv->sample_every_n = g_value_get_uint(value);
      break;
    case PROP_FLUSH_PER_SAMPLE:
      priv->flush_per_sample = g_value_get_boolean(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void gst_timing_mark_get_property(GObject* object,
                                         guint prop_id,
                                         GValue* value,
                                         GParamSpec* pspec) {
  GstTimingMark* self = GST_TIMING_MARK(object);
  const GstTimingMarkPrivate* priv = self->priv;

  switch (prop_id) {
    case PROP_CHANNEL_ID:
      g_value_set_uint(value, priv->channel_id);
      break;
    case PROP_STAGE_ID:
      g_value_set_uint(value, priv->stage_id);
      break;
    case PROP_OUTPUT_DIR:
      g_value_set_string(value, priv->output_dir.c_str());
      break;
    case PROP_SAMPLE_EVERY_N:
      g_value_set_uint(value, priv->sample_every_n);
      break;
    case PROP_FLUSH_PER_SAMPLE:
      g_value_set_boolean(value, priv->flush_per_sample);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static gboolean gst_timing_mark_start(GstBaseTransform* base) {
  GstTimingMark* self = GST_TIMING_MARK(base);
  GstTimingMarkPrivate* priv = self->priv;

  priv->registered = FALSE;

  if (!IsValidStageId(priv->stage_id)) {
    GST_ERROR_OBJECT(self, "invalid stage-id=%u", priv->stage_id);
    return FALSE;
  }
  if (priv->output_dir.empty()) {
    GST_ERROR_OBJECT(self, "output-dir is empty");
    return FALSE;
  }

  multi_cam_app::perf::LatencyRecorder::Config config;
  config.channel_id = priv->channel_id;
  config.output_dir = priv->output_dir;
  config.sample_every_n = (priv->sample_every_n > 0U) ? priv->sample_every_n : 1U;
  config.flush_per_sample = (priv->flush_per_sample != FALSE);

  std::string error;
  if (!multi_cam_app::perf::TimingMarkRuntime::Instance().RegisterChannel(
          config, &error)) {
    GST_ERROR_OBJECT(self,
                     "TimingMarkRuntime register failed for channel=%u: %s",
                     priv->channel_id,
                     error.c_str());
    return FALSE;
  }

  priv->registered = TRUE;
  return TRUE;
}

static gboolean gst_timing_mark_stop(GstBaseTransform* base) {
  GstTimingMark* self = GST_TIMING_MARK(base);
  GstTimingMarkPrivate* priv = self->priv;

  if (priv->registered) {
    multi_cam_app::perf::TimingMarkRuntime::Instance().UnregisterChannel(
        priv->channel_id);
    priv->registered = FALSE;
  }
  return TRUE;
}

static GstFlowReturn gst_timing_mark_transform_ip(GstBaseTransform* base,
                                                  GstBuffer* buffer) {
  GstTimingMark* self = GST_TIMING_MARK(base);
  GstTimingMarkPrivate* priv = self->priv;

  if (!priv->registered || (buffer == nullptr)) {
    return GST_FLOW_OK;
  }

  uint64_t frame_key = 0ULL;
  if (!multi_cam_app::perf::LatencyRecorder::ExtractFrameKey(buffer, &frame_key)) {
    return GST_FLOW_OK;
  }

  multi_cam_app::perf::TimingMarkRuntime::Instance().Submit(
      priv->channel_id,
      static_cast<multi_cam_app::perf::LatencyStage>(priv->stage_id),
      frame_key,
      MonotonicNowNs());
  return GST_FLOW_OK;
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(
      plugin, "qtitimingmark", GST_RANK_NONE, GST_TYPE_TIMING_MARK);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  qtitimingmark,
                  "QTI Timing Mark",
                  plugin_init,
                  "1.0",
                  "Proprietary",
                  "multi_cam_app",
                  "https://www.qualcomm.com")
