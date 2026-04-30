#include "channel_worker/infer_module/DetectionInferModule.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "pipeline/PipelineUtils.hpp"
#include "utils/perf/LatencyRecorder.hpp"

namespace multi_cam_app::channel_worker::infer_module {
namespace {

std::string BuildPostprocessSettings(double confidence) {
  return std::string("{\"confidence\":") + std::to_string(confidence) + "}";
}

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

std::vector<std::string> GetEnumPropertyNicks(GstElement* element,
                                              const char* property_name) {
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
    if ((value != nullptr) && (value->value_nick != nullptr) &&
        (value->value_nick[0] != '\0')) {
      result.emplace_back(value->value_nick);
    }
  }

  g_type_class_unref(enum_class);
  return result;
}

void AppendIfNotNull(std::vector<GstElement*>* elements, GstElement* element) {
  if ((elements != nullptr) && (element != nullptr)) {
    elements->push_back(element);
  }
}

GstElement* CreateTimingMark(const config::AppConfig& config,
                             uint32_t channel_id,
                             const char* element_suffix,
                             perf::LatencyStage stage,
                             std::string* out_error) {
  if (!config.latency_test.enabled) {
    return nullptr;
  }

  GstElement* marker = pipeline::MakeElement(
      "qtitimingmark",
      pipeline::MakeElementName(element_suffix, channel_id),
      out_error);
  if (marker == nullptr) {
    return nullptr;
  }

  const std::string output_dir = config.latency_test.output_dir.empty()
      ? config.log_dir
      : config.latency_test.output_dir;
  g_object_set(G_OBJECT(marker),
               "channel-id", channel_id,
               "stage-id", static_cast<guint>(stage),
               "output-dir", output_dir.c_str(),
               "sample-every-n", config.latency_test.sample_every_n,
               "flush-per-sample", config.latency_test.flush_per_sample,
               nullptr);
  return marker;
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
  attempted.reserve(candidate_order.size());

  for (const auto& nick : candidate_order) {
    attempted.push_back(nick);
    if (pipeline::SetEnumPropertyByNick(
            qtimlvconverter, "engine", nick, &set_error)) {
      return true;
    }
  }

  const std::vector<std::string> available_nicks =
      GetEnumPropertyNicks(qtimlvconverter, "engine");

  if (out_error != nullptr) {
    std::ostringstream oss;
    oss << "Failed to configure qtimlvconverter.engine for headless mode. "
        << "Attempted=[" << JoinStrings(attempted, ",") << "]";
    if (!skipped_by_policy.empty()) {
      oss << ", skipped_by_policy=[" << JoinStrings(skipped_by_policy, ",")
          << "]";
    }
    if (!available_nicks.empty()) {
      oss << ", available=[" << JoinStrings(available_nicks, ",") << "]";
    }
    if (!set_error.empty()) {
      oss << ", last_error=" << set_error;
    }
    oss << ". Headless policy forbids fallback to gles.";
    *out_error = oss.str();
  }

  return false;
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
  if ((caps == nullptr) || (gst_caps_get_size(caps) <= 0)) {
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

}  // namespace

bool DetectionInferModule::BuildChain(const config::AppConfig& config,
                                      uint32_t channel_id,
                                      const std::string& video_path,
                                      GstBin* pipeline_bin,
                                      GstElement** out_appsink,
                                      std::string* out_error) const {
  if ((pipeline_bin == nullptr) || (out_appsink == nullptr)) {
    if (out_error != nullptr) {
      *out_error = "BuildChain: pipeline_bin or out_appsink is null";
    }
    return false;
  }

  *out_appsink = nullptr;
  const bool face_enabled = config.face_enabled &&
      (config.face_channel_mask & (1 << channel_id));

  std::string error;

  // --- Base elements (always) ---
  GstElement* filesrc =
      pipeline::MakeElement("filesrc", pipeline::MakeElementName("filesrc", channel_id), &error);
  GstElement* qtdemux =
      pipeline::MakeElement("qtdemux", pipeline::MakeElementName("qtdemux", channel_id), &error);
  GstElement* h264parse =
      pipeline::MakeElement("h264parse", pipeline::MakeElementName("h264parse", channel_id), &error);
  GstElement* timing_parse_out =
      CreateTimingMark(config, channel_id, "timing_parse_out",
                       perf::LatencyStage::kH264ParseSrc, &error);
  GstElement* v4l2h264dec =
      pipeline::MakeElement("v4l2h264dec", pipeline::MakeElementName("v4l2h264dec", channel_id), &error);
  GstElement* v4l2h264dec_caps =
      pipeline::MakeElement("capsfilter", pipeline::MakeElementName("v4l2h264dec_caps", channel_id), &error);
  GstElement* timing_dec_out =
      CreateTimingMark(config, channel_id, "timing_dec_out",
                       perf::LatencyStage::kV4L2DecCapsSrc, &error);

  // --- Tee (always needed: YOLO branch + video branch) ---
  GstElement* tee_decode =
      pipeline::MakeElement("tee", pipeline::MakeElementName("tee_decode", channel_id), &error);

  // --- YOLO branch ---
  GstElement* queue_decode =
      pipeline::MakeElement("queue", pipeline::MakeElementName("queue_decode", channel_id), &error);
  GstElement* qtimlvconverter =
      pipeline::MakeElement("qtimlvconverter", pipeline::MakeElementName("qtimlvconverter", channel_id), &error);
  GstElement* timing_yolo_qnn_in =
      CreateTimingMark(config, channel_id, "timing_yolo_qnn_in",
                       perf::LatencyStage::kYoloQnnSink, &error);
  GstElement* qtimlqnn =
      pipeline::MakeElement("qtimlqnn", pipeline::MakeElementName("qtimlqnn", channel_id), &error);
  GstElement* timing_yolo_qnn_out =
      CreateTimingMark(config, channel_id, "timing_yolo_qnn_out",
                       perf::LatencyStage::kYoloQnnSrc, &error);
  GstElement* qtimlpostprocess =
      pipeline::MakeElement("qtimlpostprocess", pipeline::MakeElementName("qtimlpostprocess", channel_id), &error);
  GstElement* timing_yolo_post_out =
      CreateTimingMark(config, channel_id, "timing_yolo_post_out",
                       perf::LatencyStage::kYoloPostprocessSrc, &error);
  GstElement* queue_tracker =
      pipeline::MakeElement("queue", pipeline::MakeElementName("queue_tracker", channel_id), &error);
  GstElement* timing_tracker_in =
      CreateTimingMark(config, channel_id, "timing_tracker_in",
                       perf::LatencyStage::kTrackerSink, &error);
  GstElement* qtiobjtracker =
      pipeline::MakeElement("qtiobjtracker", pipeline::MakeElementName("qtiobjtracker", channel_id), &error);
  GstElement* timing_tracker_out =
      CreateTimingMark(config, channel_id, "timing_tracker_out",
                       perf::LatencyStage::kTrackerSrc, &error);
  GstElement* queue_meta =
      pipeline::MakeElement("queue", pipeline::MakeElementName("queue_meta", channel_id), &error);
  GstElement* text_capsfilter =
      pipeline::MakeElement("capsfilter", pipeline::MakeElementName("text_caps", channel_id), &error);

  // --- Unified output: metamux + queue_deduper + qtideduper + appsink_combined ---
  GstElement* metamux =
      pipeline::MakeElement("qtimetamux", pipeline::MakeElementName("metamux", channel_id), &error);
  GstElement* queue_deduper =
      pipeline::MakeElement("queue", pipeline::MakeElementName("queue_deduper", channel_id), &error);
  GstElement* deduper =
      pipeline::MakeElement("qtideduper", pipeline::MakeElementName("deduper", channel_id), &error);
  GstElement* appsink_combined =
      pipeline::MakeElement("appsink", pipeline::MakeElementName("appsink_combined", channel_id), &error);

  // --- frame_offload branch (conditional on frame_cache) ---
  GstElement* tee_deduper = nullptr;
  GstElement* queue_fo = nullptr;
  GstElement* frame_offload = nullptr;
  GstElement* fakesink_fo = nullptr;
  GstElement* queue_agg = nullptr;
  GstElement* queue_combined = nullptr;

  if (config.frame_cache.enabled) {
    tee_deduper = pipeline::MakeElement("tee",
        pipeline::MakeElementName("tee_deduper", channel_id), &error);
    queue_fo = pipeline::MakeElement("queue",
        pipeline::MakeElementName("queue_fo", channel_id), &error);
    frame_offload = pipeline::MakeElement("qtiframeoffload",
        pipeline::MakeElementName("frame_offload", channel_id), &error);
    if (config.msgagg.enabled) {
      queue_agg = pipeline::MakeElement("queue",
          pipeline::MakeElementName("queue_agg", channel_id), &error);
    } else {
      fakesink_fo = pipeline::MakeElement("fakesink",
          pipeline::MakeElementName("fakesink_fo", channel_id), &error);
    }
    // Only create Path B (appsink JSONL) when msgagg is NOT handling output
    if (!config.msgagg.enabled) {
      queue_combined = pipeline::MakeElement("queue",
          pipeline::MakeElementName("queue_combined", channel_id), &error);
    }
  }

  // --- Video branch (face-dependent) ---
  GstElement* queue_face = nullptr;
  GstElement* timing_face_in = nullptr;
  GstElement* faceinfer = nullptr;
  GstElement* timing_face_out = nullptr;
  GstElement* queue_video = nullptr;

  if (face_enabled) {
    queue_face =
        pipeline::MakeElement("queue", pipeline::MakeElementName("queue_face", channel_id), &error);
    timing_face_in =
        CreateTimingMark(config, channel_id, "timing_face_in",
                         perf::LatencyStage::kFaceInferSink, &error);
    faceinfer =
        pipeline::MakeElement("qtifaceinfer", pipeline::MakeElementName("faceinfer", channel_id), &error);
    timing_face_out =
        CreateTimingMark(config, channel_id, "timing_face_out",
                         perf::LatencyStage::kFaceInferSrc, &error);
  } else {
    queue_video =
        pipeline::MakeElement("queue", pipeline::MakeElementName("queue_video", channel_id), &error);
  }

  // --- Null check all elements ---
  std::vector<GstElement*> all_elements = {
      filesrc, qtdemux, h264parse, v4l2h264dec, v4l2h264dec_caps,
      tee_decode, queue_decode, qtimlvconverter, qtimlqnn, qtimlpostprocess,
      queue_tracker, qtiobjtracker, queue_meta, text_capsfilter,
      metamux, queue_deduper, deduper, appsink_combined};
  AppendIfNotNull(&all_elements, timing_parse_out);
  AppendIfNotNull(&all_elements, timing_dec_out);
  AppendIfNotNull(&all_elements, timing_yolo_qnn_in);
  AppendIfNotNull(&all_elements, timing_yolo_qnn_out);
  AppendIfNotNull(&all_elements, timing_yolo_post_out);
  AppendIfNotNull(&all_elements, timing_tracker_in);
  AppendIfNotNull(&all_elements, timing_tracker_out);

  if (face_enabled) {
    all_elements.push_back(queue_face);
    AppendIfNotNull(&all_elements, timing_face_in);
    all_elements.push_back(faceinfer);
    AppendIfNotNull(&all_elements, timing_face_out);
  } else {
    all_elements.push_back(queue_video);
  }

  if (config.frame_cache.enabled) {
    all_elements.push_back(tee_deduper);
    all_elements.push_back(queue_fo);
    all_elements.push_back(frame_offload);
    all_elements.push_back(config.msgagg.enabled ? queue_agg : fakesink_fo);
    if (queue_combined != nullptr) {
      all_elements.push_back(queue_combined);
    }
  }

  if (config.latency_test.enabled) {
    const bool missing_timing_mark =
        (timing_parse_out == nullptr) ||
        (timing_dec_out == nullptr) ||
        (timing_yolo_qnn_in == nullptr) ||
        (timing_yolo_qnn_out == nullptr) ||
        (timing_yolo_post_out == nullptr) ||
        (timing_tracker_in == nullptr) ||
        (timing_tracker_out == nullptr) ||
        (face_enabled &&
         ((timing_face_in == nullptr) || (timing_face_out == nullptr)));
    if (missing_timing_mark) {
      if (out_error != nullptr) {
        *out_error = "Failed to create required timing mark element: " + error;
      }
      return false;
    }
  }

  for (GstElement* element : all_elements) {
    if (element == nullptr) {
      if (out_error != nullptr) {
        *out_error = "Failed to create required element: " + error;
      }
      return false;
    }
  }

  // --- Add elements to shared pipeline bin ---
  std::vector<GstElement*> base_elements = {
      filesrc, qtdemux, h264parse, v4l2h264dec, v4l2h264dec_caps,
      tee_decode, queue_decode, qtimlvconverter, qtimlqnn, qtimlpostprocess,
      queue_tracker, qtiobjtracker, queue_meta, text_capsfilter,
      metamux, queue_deduper, deduper, appsink_combined};
  AppendIfNotNull(&base_elements, timing_parse_out);
  AppendIfNotNull(&base_elements, timing_dec_out);
  AppendIfNotNull(&base_elements, timing_yolo_qnn_in);
  AppendIfNotNull(&base_elements, timing_yolo_qnn_out);
  AppendIfNotNull(&base_elements, timing_yolo_post_out);
  AppendIfNotNull(&base_elements, timing_tracker_in);
  AppendIfNotNull(&base_elements, timing_tracker_out);
  if (!pipeline::AddElements(pipeline_bin, base_elements, &error)) {
    if (out_error != nullptr) { *out_error = error; }
    return false;
  }

  if (face_enabled) {
    std::vector<GstElement*> face_elements = {queue_face, faceinfer};
    if (timing_face_in != nullptr) {
      face_elements.insert(face_elements.begin() + 1, timing_face_in);
    }
    if (timing_face_out != nullptr) {
      face_elements.push_back(timing_face_out);
    }
    if (!pipeline::AddElements(pipeline_bin, face_elements, &error)) {
      if (out_error != nullptr) { *out_error = error; }
      return false;
    }
  } else {
    if (!pipeline::AddElements(pipeline_bin, {queue_video}, &error)) {
      if (out_error != nullptr) { *out_error = error; }
      return false;
    }
  }

  if (config.frame_cache.enabled) {
    GstElement* fo_tail = config.msgagg.enabled ? queue_agg : fakesink_fo;
    std::vector<GstElement*> fo_elements = {tee_deduper, queue_fo, frame_offload, fo_tail};
    if (queue_combined != nullptr) {
      fo_elements.push_back(queue_combined);
    }
    if (!pipeline::AddElements(pipeline_bin, fo_elements, &error)) {
      if (out_error != nullptr) { *out_error = error; }
      return false;
    }
  }

  // --- Link filesrc → qtdemux ---
  if (!gst_element_link(filesrc, qtdemux)) {
    if (out_error != nullptr) { *out_error = "Failed to link filesrc->qtdemux"; }
    return false;
  }
  g_signal_connect(qtdemux, "pad-added", G_CALLBACK(OnDemuxPadAdded), h264parse);

  // --- Link h264parse → v4l2h264dec → caps → tee_decode ---
  std::vector<GstElement*> decode_chain = {
      h264parse,
      v4l2h264dec,
      v4l2h264dec_caps,
      tee_decode};
  if (timing_parse_out != nullptr) {
    decode_chain.insert(decode_chain.begin() + 1, timing_parse_out);
  }
  if (timing_dec_out != nullptr) {
    decode_chain.insert(decode_chain.end() - 1, timing_dec_out);
  }
  if (!pipeline::LinkElements(decode_chain, &error)) {
    if (out_error != nullptr) { *out_error = error; }
    return false;
  }

  // --- YOLO branch: queue_decode → ... → text_capsfilter ---
  std::vector<GstElement*> yolo_chain = {
      queue_decode,
      qtimlvconverter,
      qtimlqnn,
      qtimlpostprocess,
      queue_tracker,
      qtiobjtracker,
      queue_meta,
      text_capsfilter};
  if (timing_yolo_qnn_in != nullptr) {
    yolo_chain.insert(yolo_chain.begin() + 2, timing_yolo_qnn_in);
  }
  if (timing_yolo_qnn_out != nullptr) {
    yolo_chain.insert(yolo_chain.begin() + 4, timing_yolo_qnn_out);
  }
  if (timing_yolo_post_out != nullptr) {
    yolo_chain.insert(yolo_chain.begin() + 6, timing_yolo_post_out);
  }
  if (timing_tracker_in != nullptr) {
    yolo_chain.insert(yolo_chain.end() - 3, timing_tracker_in);
  }
  if (timing_tracker_out != nullptr) {
    yolo_chain.insert(yolo_chain.end() - 2, timing_tracker_out);
  }
  if (!pipeline::LinkElements(yolo_chain, &error)) {
    if (out_error != nullptr) { *out_error = error; }
    return false;
  }

  // --- Link tee → queue_decode (YOLO branch, request pad) ---
  {
    GstPad* tee_src0 = gst_element_request_pad_simple(tee_decode, "src_%u");
    GstPad* yolo_sink = gst_element_get_static_pad(queue_decode, "sink");
    if ((tee_src0 == nullptr) || (yolo_sink == nullptr) ||
        (gst_pad_link(tee_src0, yolo_sink) != GST_PAD_LINK_OK)) {
      if (out_error != nullptr) { *out_error = "Failed to link tee_decode -> queue_decode"; }
      if (tee_src0 != nullptr) gst_object_unref(tee_src0);
      if (yolo_sink != nullptr) gst_object_unref(yolo_sink);
      return false;
    }
    gst_object_unref(tee_src0);
    gst_object_unref(yolo_sink);
  }

  // --- Link tee → video branch head (request pad) ---
  GstElement* video_branch_head = face_enabled ? queue_face : queue_video;
  {
    GstPad* tee_src1 = gst_element_request_pad_simple(tee_decode, "src_%u");
    GstPad* video_sink = gst_element_get_static_pad(video_branch_head, "sink");
    if ((tee_src1 == nullptr) || (video_sink == nullptr) ||
        (gst_pad_link(tee_src1, video_sink) != GST_PAD_LINK_OK)) {
      if (out_error != nullptr) { *out_error = "Failed to link tee_decode -> video branch"; }
      if (tee_src1 != nullptr) gst_object_unref(tee_src1);
      if (video_sink != nullptr) gst_object_unref(video_sink);
      return false;
    }
    gst_object_unref(tee_src1);
    gst_object_unref(video_sink);
  }

  // --- Link video branch → metamux(sink) ---
  if (face_enabled) {
    std::vector<GstElement*> face_chain = {queue_face, faceinfer, metamux};
    if (timing_face_in != nullptr) {
      face_chain.insert(face_chain.begin() + 1, timing_face_in);
    }
    if (timing_face_out != nullptr) {
      face_chain.insert(face_chain.end() - 1, timing_face_out);
    }
    if (!pipeline::LinkElements(face_chain, &error)) {
      if (out_error != nullptr) { *out_error = "Failed to link queue_face -> faceinfer -> metamux"; }
      return false;
    }
  } else {
    if (!gst_element_link(queue_video, metamux)) {
      if (out_error != nullptr) { *out_error = "Failed to link queue_video -> metamux"; }
      return false;
    }
  }

  // --- Link text_capsfilter → metamux(data_0) via request pad ---
  {
    GstPad* data_pad = gst_element_request_pad_simple(metamux, "data_%u");
    GstPad* text_src = gst_element_get_static_pad(text_capsfilter, "src");
    if ((data_pad == nullptr) || (text_src == nullptr) ||
        (gst_pad_link(text_src, data_pad) != GST_PAD_LINK_OK)) {
      if (out_error != nullptr) { *out_error = "Failed to link text_capsfilter -> metamux data pad"; }
      if (data_pad != nullptr) gst_object_unref(data_pad);
      if (text_src != nullptr) gst_object_unref(text_src);
      return false;
    }
    gst_object_unref(text_src);
    gst_object_unref(data_pad);
  }

  // --- Link metamux → queue_deduper → deduper → [tee_deduper or appsink_combined] ---
  if (config.frame_cache.enabled) {
    // metamux → queue_deduper → deduper → tee_deduper
    if (!gst_element_link_many(metamux, queue_deduper, deduper, tee_deduper, nullptr)) {
      if (out_error != nullptr) { *out_error = "Failed to link metamux->...->tee_deduper"; }
      return false;
    }

    // tee_deduper → queue_fo (request pad)
    {
      GstPad* tee_src0 = gst_element_request_pad_simple(tee_deduper, "src_%u");
      GstPad* fo_sink = gst_element_get_static_pad(queue_fo, "sink");
      if ((tee_src0 == nullptr) || (fo_sink == nullptr) ||
          (gst_pad_link(tee_src0, fo_sink) != GST_PAD_LINK_OK)) {
        if (out_error != nullptr) { *out_error = "Failed to link tee_deduper -> queue_fo"; }
        if (tee_src0 != nullptr) gst_object_unref(tee_src0);
        if (fo_sink != nullptr) gst_object_unref(fo_sink);
        return false;
      }
      gst_object_unref(tee_src0);
      gst_object_unref(fo_sink);
    }

    // queue_fo → frame_offload → [queue_agg or fakesink_fo]
    {
      GstElement* fo_tail = config.msgagg.enabled ? queue_agg : fakesink_fo;
      if (!gst_element_link_many(queue_fo, frame_offload, fo_tail, nullptr)) {
        if (out_error != nullptr) {
          *out_error = config.msgagg.enabled
              ? "Failed to link queue_fo -> frame_offload -> queue_agg"
              : "Failed to link queue_fo -> frame_offload -> fakesink_fo";
        }
        return false;
      }
    }

    if (queue_combined != nullptr) {
      // tee_deduper → queue_combined → appsink_combined (request pad)
      {
        GstPad* tee_src1 = gst_element_request_pad_simple(tee_deduper, "src_%u");
        GstPad* comb_sink = gst_element_get_static_pad(queue_combined, "sink");
        if ((tee_src1 == nullptr) || (comb_sink == nullptr) ||
            (gst_pad_link(tee_src1, comb_sink) != GST_PAD_LINK_OK)) {
          if (out_error != nullptr) { *out_error = "Failed to link tee_deduper -> queue_combined"; }
          if (tee_src1 != nullptr) gst_object_unref(tee_src1);
          if (comb_sink != nullptr) gst_object_unref(comb_sink);
          return false;
        }
        gst_object_unref(tee_src1);
        gst_object_unref(comb_sink);
      }
      if (!gst_element_link(queue_combined, appsink_combined)) {
        if (out_error != nullptr) { *out_error = "Failed to link queue_combined -> appsink_combined"; }
        return false;
      }
    }

    // Configure tee_deduper + queues + frame_offload + fakesink
    g_object_set(G_OBJECT(tee_deduper), "allow-not-linked", TRUE, nullptr);
    g_object_set(G_OBJECT(queue_fo),
                 "max-size-buffers", 2U, "leaky", 2,
                 "max-size-bytes", 0U,
                 "max-size-time", static_cast<guint64>(0), nullptr);
    if (queue_combined != nullptr) {
      g_object_set(G_OBJECT(queue_combined),
                   "max-size-buffers", 2U, "leaky", 2,
                   "max-size-bytes", 0U,
                   "max-size-time", static_cast<guint64>(0), nullptr);
    }
    g_object_set(G_OBJECT(frame_offload), "channel-id", channel_id, nullptr);
    if (config.frame_cache.enabled && config.msgagg.enabled) {
      g_object_set(G_OBJECT(frame_offload),
          "scene-update-mqtt", TRUE,
          "meta-width", static_cast<guint>(config.frame_cache.width),
          "meta-height", static_cast<guint>(config.frame_cache.height),
          nullptr);
      if (!config.msgagg.scene_jsonl_dir.empty()) {
        g_object_set(G_OBJECT(frame_offload),
            "jsonl-out-dir", config.msgagg.scene_jsonl_dir.c_str(),
            nullptr);
      }
      if (face_enabled) {
        g_object_set(G_OBJECT(frame_offload),
            "gallery-threshold", static_cast<gfloat>(config.gallery_threshold),
            "gallery-min-face-size", config.gallery_min_face_size,
            "gallery-min-score", static_cast<gfloat>(config.gallery_min_score),
            nullptr);
        // gallery pointer itself is set later by MultiCamApp via ChannelWorker::SetFaceGallery
      }
    }
    if (config.msgagg.enabled) {
      g_object_set(G_OBJECT(queue_agg),
                   "max-size-buffers", 3U, "leaky", 2,
                   "max-size-bytes", 0U,
                   "max-size-time", static_cast<guint64>(0), nullptr);
    } else {
      g_object_set(G_OBJECT(fakesink_fo), "async", FALSE, "sync", FALSE, nullptr);
    }
  } else {
    if (!gst_element_link_many(metamux, queue_deduper, deduper, appsink_combined, nullptr)) {
      if (out_error != nullptr) { *out_error = "Failed to link metamux -> queue_deduper -> deduper -> appsink_combined"; }
      return false;
    }
  }

  // --- Configure queue_deduper (async boundary before deduper) ---
  g_object_set(G_OBJECT(queue_deduper),
               "max-size-buffers", 2U,
               "leaky", 2,
               "max-size-bytes", 0U,
               "max-size-time", static_cast<guint64>(0),
               nullptr);

  // --- Configure tee ---
  g_object_set(G_OBJECT(tee_decode), "allow-not-linked", TRUE, nullptr);

  // --- Configure video branch ---
  if (face_enabled) {
    g_object_set(G_OBJECT(queue_face),
                 "leaky", 2,
                 "max-size-buffers", 2U,
                 "max-size-bytes", 0U,
                 "max-size-time", static_cast<guint64>(0),
                 nullptr);
    g_object_set(G_OBJECT(faceinfer),
                 "config-path", config.face_config_path.c_str(),
                 "face-interval-ms", config.face_interval_ms,
                 nullptr);
  }

  // --- Configure YOLO branch queues (leaky to prevent tee backpressure during HTP warmup) ---
  g_object_set(G_OBJECT(queue_decode),
               "max-size-buffers", 2U,
               "leaky", 2,
               "max-size-bytes", 0U,
               "max-size-time", static_cast<guint64>(0),
               nullptr);

  // --- Configure common base elements ---
  g_object_set(G_OBJECT(filesrc), "location", video_path.c_str(), nullptr);

  if (!pipeline::SetEnumPropertyByNick(v4l2h264dec, "capture-io-mode", "dmabuf", &error)) {
    if (out_error != nullptr) { *out_error = error; }
    return false;
  }
  if (!pipeline::SetEnumPropertyByNick(v4l2h264dec, "output-io-mode", "dmabuf", &error)) {
    if (out_error != nullptr) { *out_error = error; }
    return false;
  }

  GstCaps* decode_caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", nullptr);
  if (decode_caps == nullptr) {
    if (out_error != nullptr) { *out_error = "Failed to create decode caps video/x-raw,format=NV12"; }
    return false;
  }
  g_object_set(G_OBJECT(v4l2h264dec_caps), "caps", decode_caps, nullptr);
  gst_caps_unref(decode_caps);

  if (!SetMlConverterEngineWithFallback(
          qtimlvconverter, config.qtimlvconverter_engine_order, &error)) {
    if (out_error != nullptr) { *out_error = error; }
    return false;
  }

  g_object_set(G_OBJECT(qtimlqnn),
               "model", config.model_path.c_str(),
               "backend", config.qnn_backend.c_str(),
               "system", config.qnn_system.c_str(),
               "backend-device-id", config.qnn_backend_device_id,
               nullptr);

  if (!SetQnnTensors(qtimlqnn, config.qnn_tensors, &error)) {
    if (out_error != nullptr) { *out_error = error; }
    return false;
  }

  if (!pipeline::SetEnumPropertyByNick(
          qtimlpostprocess, "module", config.postprocess_module, &error)) {
    if (out_error != nullptr) { *out_error = error; }
    return false;
  }

  if (!pipeline::SetEnumPropertyByNick(qtiobjtracker, "algo", "bytetrack", &error)) {
    if (out_error != nullptr) { *out_error = "Failed to configure qtiobjtracker.algo: " + error; }
    return false;
  }

  if (!config.qtiobjtracker_parameters.empty()) {
    g_object_set(G_OBJECT(qtiobjtracker),
                 "parameters", config.qtiobjtracker_parameters.c_str(),
                 nullptr);
  }

  const std::string postprocess_settings = BuildPostprocessSettings(config.confidence);
  g_object_set(G_OBJECT(qtimlpostprocess),
               "labels", config.labels_path.c_str(),
               "settings", postprocess_settings.c_str(),
               nullptr);

  // --- Configure text capsfilter (tracker text → metamux data pad) ---
  GstCaps* text_caps = gst_caps_new_simple("text/x-raw", "format", G_TYPE_STRING, "utf8", nullptr);
  if (text_caps == nullptr) {
    if (out_error != nullptr) { *out_error = "Failed to create text/x-raw,format=utf8 caps"; }
    return false;
  }
  g_object_set(G_OBJECT(text_capsfilter), "caps", text_caps, nullptr);
  gst_caps_unref(text_caps);

  // --- Configure appsink_combined (NV12 + ROI meta + optional FaceMeta) ---
  GstCaps* combined_caps = gst_caps_new_simple("video/x-raw",
      "format", G_TYPE_STRING, "NV12", nullptr);
  if (combined_caps == nullptr) {
    if (out_error != nullptr) { *out_error = "Failed to create combined caps"; }
    return false;
  }
  g_object_set(G_OBJECT(appsink_combined), "caps", combined_caps, nullptr);
  gst_caps_unref(combined_caps);
  // Note: emit-signals, sync, drop, max-buffers are set by ChannelWorker::Initialize

  *out_appsink = appsink_combined;
  return true;
}

void DetectionInferModule::OnDemuxPadAdded(GstElement* element,
                                           GstPad* pad,
                                           gpointer user_data) {
  if (IsVideoH264Pad(pad)) {
    auto* parser = static_cast<GstElement*>(user_data);
    if (parser == nullptr) {
      return;
    }

    GstPad* sinkpad = gst_element_get_static_pad(parser, "sink");
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
    // Link non-video pads (audio, subtitle, etc.) to fakesink to prevent
    // not-linked errors that would kill the pipeline.
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

bool DetectionInferModule::SetQnnTensors(GstElement* qtimlqnn,
                                         const std::vector<std::string>& tensors,
                                         std::string* out_error) {
  if (qtimlqnn == nullptr) {
    if (out_error != nullptr) {
      *out_error = "SetQnnTensors received null qtimlqnn";
    }
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

}  // namespace multi_cam_app::channel_worker::infer_module
