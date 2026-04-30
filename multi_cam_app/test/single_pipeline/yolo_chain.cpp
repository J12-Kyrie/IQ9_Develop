#include "yolo_chain.hpp"

#include "pipeline_local.hpp"

#include <sstream>
#include <string>
#include <vector>

#include <gst/gst.h>

namespace single_pipeline {
namespace {

// Must match multi_cam_app::perf::LatencyStage: kYoloQnnSink=2, kYoloQnnSrc=3
constexpr guint kStageYoloQnnSink = 2U;
constexpr guint kStageYoloQnnSrc = 3U;

std::string JoinStrings(const std::vector<std::string>& values, const char* delimiter) {
  std::ostringstream oss;
  for (size_t i = 0U; i < values.size(); ++i) {
    if (i != 0U) {
      oss << delimiter;
    }
    oss << values[i];
  }
  return oss.str();
}

std::vector<std::string> GetEnumPropertyNicks(GstElement* element, const char* property_name) {
  std::vector<std::string> result;
  if ((element == nullptr) || (property_name == nullptr)) {
    return result;
  }
  GObjectClass* klass = G_OBJECT_GET_CLASS(element);
  GParamSpec* param = g_object_class_find_property(klass, property_name);
  if ((param == nullptr) || !G_IS_PARAM_SPEC_ENUM(param)) {
    return result;
  }
  GEnumClass* enum_class = G_ENUM_CLASS(g_type_class_ref(param->value_type));
  if (enum_class == nullptr) {
    return result;
  }
  result.reserve(static_cast<size_t>(enum_class->n_values));
  for (guint i = 0U; i < enum_class->n_values; ++i) {
    const GEnumValue* value = &(enum_class->values[i]);
    if ((value != nullptr) && (value->value_nick != nullptr) && (value->value_nick[0] != '\0')) {
      result.emplace_back(value->value_nick);
    }
  }
  g_type_class_unref(enum_class);
  return result;
}

bool SetMlConverterEngineWithFallback(
    GstElement* qtimlvconverter,
    const std::vector<std::string>& configured_engine_order,
    std::string* out_error) {
  if (qtimlvconverter == nullptr) {
    if (out_error != nullptr) {
      *out_error = "SetMlConverterEngineWithFallback received null qtimlvconverter";
    }
    return false;
  }

  std::vector<std::string> candidate_order;
  std::vector<std::string> skipped_by_policy;
  candidate_order.reserve(configured_engine_order.size());
  for (const auto& nick : configured_engine_order) {
    if (nick.empty()) {
      continue;
    }
    if ((nick == "gles") || (nick == "none")) {
      skipped_by_policy.push_back(nick);
      continue;
    }
    candidate_order.push_back(nick);
  }
  if (candidate_order.empty()) {
    candidate_order = {"fcv", "ocv"};
  }

  std::string set_error;
  std::vector<std::string> attempted;
  for (const auto& nick : candidate_order) {
    attempted.push_back(nick);
    if (SetEnumPropertyByNick(qtimlvconverter, "engine", nick, &set_error)) {
      return true;
    }
  }
  const std::vector<std::string> available_nicks = GetEnumPropertyNicks(qtimlvconverter, "engine");
  if (out_error != nullptr) {
    std::ostringstream oss;
    oss << "Failed to configure qtimlvconverter.engine. Attempted=["
        << JoinStrings(attempted, ",") << "]";
    if (!skipped_by_policy.empty()) {
      oss << ", skipped_by_policy=[" << JoinStrings(skipped_by_policy, ",") << "]";
    }
    if (!available_nicks.empty()) {
      oss << ", available=[" << JoinStrings(available_nicks, ",") << "]";
    }
    if (!set_error.empty()) {
      oss << ", last_error=" << set_error;
    }
    *out_error = oss.str();
  }
  return false;
}

bool SetQnnTensors(GstElement* qtimlqnn, const std::vector<std::string>& tensors) {
  if (qtimlqnn == nullptr) {
    return false;
  }
  if (tensors.empty()) {
    return true;
  }
  GValue array = G_VALUE_INIT;
  g_value_init(&array, GST_TYPE_ARRAY);
  for (const auto& tensor_name : tensors) {
    if (tensor_name.empty()) {
      continue;
    }
    GValue item = G_VALUE_INIT;
    g_value_init(&item, G_TYPE_STRING);
    g_value_set_string(&item, tensor_name.c_str());
    gst_value_array_append_value(&array, &item);
    g_value_unset(&item);
  }
  g_object_set_property(G_OBJECT(qtimlqnn), "tensors", &array);
  g_value_unset(&array);
  return true;
}

bool IsVideoH264Pad(GstPad* pad) {
  if (pad == nullptr) {
    return false;
  }
  const gchar* pad_name = GST_OBJECT_NAME(pad);
  if ((pad_name == nullptr) || !g_str_has_prefix(pad_name, "video")) {
    return false;
  }
  GstCaps* caps = gst_pad_get_current_caps(pad);
  if (caps == nullptr) {
    caps = gst_pad_query_caps(pad, nullptr);
  }
  if ((caps == nullptr) || (gst_caps_get_size(caps) <= 0U)) {
    if (caps != nullptr) {
      gst_caps_unref(caps);
    }
    return true;
  }
  bool is_video_h264 = false;
  const guint n_structs = gst_caps_get_size(caps);
  for (guint i = 0U; i < n_structs; ++i) {
    const GstStructure* structure = gst_caps_get_structure(caps, i);
    const gchar* media_type = (structure != nullptr) ? gst_structure_get_name(structure) : nullptr;
    if ((media_type != nullptr) && (g_strcmp0(media_type, "video/x-h264") == 0)) {
      is_video_h264 = true;
      break;
    }
  }
  gst_caps_unref(caps);
  return is_video_h264;
}

void OnDemuxPadAdded(GstElement* element, GstPad* pad, gpointer user_data) {
  if (IsVideoH264Pad(pad)) {
    auto* h264parse = static_cast<GstElement*>(user_data);
    if (h264parse == nullptr) {
      return;
    }
    GstPad* sinkpad = gst_element_get_static_pad(h264parse, "sink");
    if (sinkpad == nullptr) {
      return;
    }
    if (gst_pad_is_linked(sinkpad)) {
      gst_object_unref(sinkpad);
      return;
    }
    (void)gst_pad_link(pad, sinkpad);
    gst_object_unref(sinkpad);
  } else {
    GstElement* parent = GST_ELEMENT(gst_element_get_parent(element));
    if (parent != nullptr) {
      GstElement* fakesink = gst_element_factory_make("fakesink", nullptr);
      if (fakesink != nullptr) {
        g_object_set(G_OBJECT(fakesink), "sync", FALSE, nullptr);
        gst_bin_add(GST_BIN(parent), fakesink);
        GstPad* sink_pad = gst_element_get_static_pad(fakesink, "sink");
        if (sink_pad != nullptr) {
          (void)gst_pad_link(pad, sink_pad);
          gst_object_unref(sink_pad);
        }
        gst_element_sync_state_with_parent(fakesink);
      }
      gst_object_unref(parent);
    }
  }
}

}  // namespace

bool AddYoloQnnLatencyChain(const YoloQnnLatencyConfig& config,
                            uint32_t channel_id,
                            const std::string& video_path,
                            GstBin* pipeline_bin,
                            std::string* out_error) {
  if (pipeline_bin == nullptr) {
    if (out_error != nullptr) {
      *out_error = "pipeline_bin is null";
    }
    return false;
  }

  std::string error;
  GstElement* filesrc = MakeElement("filesrc", MakeElementName("filesrc", channel_id), &error);
  GstElement* qtdemux = MakeElement("qtdemux", MakeElementName("qtdemux", channel_id), &error);
  GstElement* h264parse = MakeElement("h264parse", MakeElementName("h264parse", channel_id), &error);
  GstElement* v4l2h264dec = MakeElement("v4l2h264dec", MakeElementName("v4l2h264dec", channel_id), &error);
  GstElement* v4l2h264dec_caps = MakeElement("capsfilter", MakeElementName("v4l2h264dec_caps", channel_id), &error);
  GstElement* queue = MakeElement("queue", MakeElementName("queue_yolo", channel_id), &error);
  GstElement* qtimlvconverter = MakeElement("qtimlvconverter", MakeElementName("qtimlvconverter", channel_id), &error);
  GstElement* timing_in =
      MakeElement("qtitimingmark", MakeElementName("timing_yolo_qnn_in", channel_id), &error);
  GstElement* qtimlqnn = MakeElement("qtimlqnn", MakeElementName("qtimlqnn", channel_id), &error);
  GstElement* timing_out =
      MakeElement("qtitimingmark", MakeElementName("timing_yolo_qnn_out", channel_id), &error);
  GstElement* queue_out = MakeElement("queue", MakeElementName("queue_after_qnn", channel_id), &error);
  GstElement* fakesink = MakeElement("fakesink", MakeElementName("fakesink_yolo", channel_id), &error);

  const std::vector<GstElement*> all = {filesrc,      qtdemux,  h264parse,         v4l2h264dec,
                                        v4l2h264dec_caps,  queue,    qtimlvconverter,  timing_in,
                                        qtimlqnn,  timing_out,  queue_out,  fakesink};

  for (GstElement* el : all) {
    if (el == nullptr) {
      if (out_error != nullptr) {
        *out_error = "Failed to create element: " + error;
      }
      return false;
    }
  }

  g_object_set(G_OBJECT(timing_in),
               "channel-id", channel_id,
               "stage-id", kStageYoloQnnSink,
               "output-dir", config.output_dir.c_str(),
               "sample-every-n", config.sample_every_n,
               "flush-per-sample", config.flush_per_sample,
               nullptr);
  g_object_set(G_OBJECT(timing_out),
               "channel-id", channel_id,
               "stage-id", kStageYoloQnnSrc,
               "output-dir", config.output_dir.c_str(),
               "sample-every-n", config.sample_every_n,
               "flush-per-sample", config.flush_per_sample,
               nullptr);

  if (!AddElements(pipeline_bin, all, &error)) {
    if (out_error != nullptr) {
      *out_error = error;
    }
    return false;
  }

  g_object_set(G_OBJECT(filesrc), "location", video_path.c_str(), nullptr);

  if (!SetEnumPropertyByNick(v4l2h264dec, "capture-io-mode", "dmabuf", &error)) {
    if (out_error != nullptr) {
      *out_error = error;
    }
    return false;
  }
  if (!SetEnumPropertyByNick(v4l2h264dec, "output-io-mode", "dmabuf", &error)) {
    if (out_error != nullptr) {
      *out_error = error;
    }
    return false;
  }

  GstCaps* decode_caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", nullptr);
  if (decode_caps == nullptr) {
    if (out_error != nullptr) {
      *out_error = "Failed to create decode caps";
    }
    return false;
  }
  g_object_set(G_OBJECT(v4l2h264dec_caps), "caps", decode_caps, nullptr);
  gst_caps_unref(decode_caps);

  g_object_set(G_OBJECT(queue),
               "max-size-buffers", 2U,
               "leaky", 2,
               "max-size-bytes", 0U,
               "max-size-time", static_cast<guint64>(0),
               nullptr);

  if (!SetMlConverterEngineWithFallback(qtimlvconverter, config.qtimlvconverter_engine_order, &error)) {
    if (out_error != nullptr) {
      *out_error = error;
    }
    return false;
  }

  g_object_set(G_OBJECT(qtimlqnn),
               "model", config.model_path.c_str(),
               "backend", config.qnn_backend.c_str(),
               "system", config.qnn_system.c_str(),
               "backend-device-id", config.qnn_backend_device_id,
               nullptr);
  if (!SetQnnTensors(qtimlqnn, config.qnn_tensors)) {
    if (out_error != nullptr) {
      *out_error = "SetQnnTensors failed";
    }
    return false;
  }

  g_object_set(G_OBJECT(queue_out),
               "max-size-buffers", 2U,
               "leaky", 2,
               "max-size-bytes", 0U,
               "max-size-time", static_cast<guint64>(0),
               nullptr);
  g_object_set(G_OBJECT(fakesink), "sync", FALSE, "async", FALSE, "qos", FALSE, nullptr);

  if (!gst_element_link(filesrc, qtdemux)) {
    if (out_error != nullptr) {
      *out_error = "Failed to link filesrc->qtdemux";
    }
    return false;
  }

  g_signal_connect(qtdemux, "pad-added", G_CALLBACK(OnDemuxPadAdded), h264parse);

  std::vector<GstElement*> decode_chain = {h264parse,  v4l2h264dec,     v4l2h264dec_caps,          queue,
                                            qtimlvconverter, timing_in,  qtimlqnn,  timing_out,  queue_out,  fakesink};
  if (!LinkElements(decode_chain, &error)) {
    if (out_error != nullptr) {
      *out_error = error;
    }
    return false;
  }

  return true;
}

}  // namespace single_pipeline
