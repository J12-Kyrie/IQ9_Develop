#include "gstframeoffload.h"
#include "../FrameOffload.hpp"

#include <gst/video/video.h>
#include <gst/video/gstvideometa.h>

#include "dmabuf_producer/producer.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

GST_DEBUG_CATEGORY_STATIC(gst_frame_offload_debug);
#define GST_CAT_DEFAULT gst_frame_offload_debug

enum {
    PROP_0,
    PROP_CHANNEL_ID,
    PROP_PRODUCER,
    PROP_SCENE_UPDATE_MQTT,
    PROP_META_WIDTH,
    PROP_META_HEIGHT,
    PROP_GALLERY,
    PROP_GALLERY_THRESHOLD,
    PROP_GALLERY_MIN_FACE_SIZE,
    PROP_GALLERY_MIN_SCORE,
    PROP_JSONL_OUT_DIR,
};

struct _GstFrameOffloadPrivate {
    std::unique_ptr<multi_cam_app::frame_offload::FrameOffload> offload;
    dmabuf_producer::DmaBufProducer* producer;
    guint channel_id;
    gint frame_w;
    gint frame_h;
    gboolean scene_update_mqtt;
    guint meta_width;
    guint meta_height;
    multi_cam_app::gallery::FaceGallery* gallery;
    gfloat gallery_threshold;
    guint gallery_min_face_size;
    gfloat gallery_min_score;
    gchar* jsonl_out_dir;
};

/* Pad templates */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw, format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("text/x-raw, format=(string)utf8"));

G_DEFINE_TYPE_WITH_PRIVATE(GstFrameOffload, gst_frame_offload, GST_TYPE_ELEMENT)

/* Forward declarations */
static void gst_frame_offload_set_property(GObject* object, guint prop_id,
    const GValue* value, GParamSpec* pspec);
static void gst_frame_offload_get_property(GObject* object, guint prop_id,
    GValue* value, GParamSpec* pspec);
static void gst_frame_offload_finalize(GObject* object);
static GstFlowReturn gst_frame_offload_chain(GstPad* pad, GstObject* parent,
    GstBuffer* buf);
static gboolean gst_frame_offload_sink_event(GstPad* pad, GstObject* parent,
    GstEvent* event);
static GstStateChangeReturn gst_frame_offload_change_state(GstElement* element,
    GstStateChange transition);

static void
gst_frame_offload_class_init(GstFrameOffloadClass* klass)
{
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);

    gobject_class->set_property = gst_frame_offload_set_property;
    gobject_class->get_property = gst_frame_offload_get_property;
    gobject_class->finalize     = gst_frame_offload_finalize;

    element_class->change_state = gst_frame_offload_change_state;

    g_object_class_install_property(gobject_class, PROP_CHANNEL_ID,
        g_param_spec_uint("channel-id", "Channel ID",
            "Channel identifier written into JSON output",
            0, G_MAXUINT, 0,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_PRODUCER,
        g_param_spec_pointer("producer", "DmaBufProducer",
            "Pointer to DmaBufProducer instance (NULL = no frame caching)",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_SCENE_UPDATE_MQTT,
        g_param_spec_boolean("scene-update-mqtt", "Scene update MQTT",
            "Emit scene_update per-source JSON (for qtimsgagg merge)",
            FALSE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_META_WIDTH,
        g_param_spec_uint("meta-width", "Image meta width",
            "Width for image_meta in scene_update (RGB slot dimensions)",
            0, G_MAXUINT, 0,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_META_HEIGHT,
        g_param_spec_uint("meta-height", "Image meta height",
            "Height for image_meta in scene_update",
            0, G_MAXUINT, 0,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_GALLERY,
        g_param_spec_pointer("gallery", "FaceGallery pointer",
            "Pointer to FaceGallery instance for face matching (NULL = disabled)",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_GALLERY_THRESHOLD,
        g_param_spec_float("gallery-threshold", "Gallery threshold",
            "Minimum cosine similarity for gallery face match",
            0.0f, 1.0f, 0.3f,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_GALLERY_MIN_FACE_SIZE,
        g_param_spec_uint("gallery-min-face-size", "Gallery min face size",
            "Minimum face bbox size in pixels for gallery enrollment",
            0, G_MAXUINT, 40,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_GALLERY_MIN_SCORE,
        g_param_spec_float("gallery-min-score", "Gallery min score",
            "Minimum SCRFD detection score for gallery enrollment",
            0.0f, 1.0f, 0.6f,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_JSONL_OUT_DIR,
        g_param_spec_string("jsonl-out-dir", "JSONL output directory",
            "If non-empty, write scene_update JSONL per channel to this directory",
            nullptr,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_static_metadata(element_class,
        "QTI Frame Offload",
        "Filter/Metadata",
        "Extract metadata from NV12 frames, cache to DMA slots, output JSON",
        "multi_cam_app");

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    GST_DEBUG_CATEGORY_INIT(gst_frame_offload_debug, "qtiframeoffload", 0,
        "QTI Frame Offload");
}

static void
gst_frame_offload_init(GstFrameOffload* self)
{
    self->priv = (GstFrameOffloadPrivate*)
        gst_frame_offload_get_instance_private(self);
    new (self->priv) GstFrameOffloadPrivate();
    self->priv->producer = nullptr;
    self->priv->channel_id = 0;
    self->priv->frame_w = 0;
    self->priv->frame_h = 0;
    self->priv->scene_update_mqtt = FALSE;
    self->priv->meta_width = 0;
    self->priv->meta_height = 0;
    self->priv->gallery = nullptr;
    self->priv->gallery_threshold = 0.3f;
    self->priv->gallery_min_face_size = 40;
    self->priv->gallery_min_score = 0.6f;
    self->priv->jsonl_out_dir = nullptr;

    /* Create sink pad */
    self->sinkpad = gst_pad_new_from_static_template(&sink_template, "sink");
    gst_pad_set_chain_function(self->sinkpad,
        GST_DEBUG_FUNCPTR(gst_frame_offload_chain));
    gst_pad_set_event_function(self->sinkpad,
        GST_DEBUG_FUNCPTR(gst_frame_offload_sink_event));
    GST_PAD_SET_PROXY_ALLOCATION(self->sinkpad);
    gst_element_add_pad(GST_ELEMENT(self), self->sinkpad);

    /* Create src pad */
    self->srcpad = gst_pad_new_from_static_template(&src_template, "src");
    gst_element_add_pad(GST_ELEMENT(self), self->srcpad);
}

static void
gst_frame_offload_finalize(GObject* object)
{
    GstFrameOffload* self = GST_FRAME_OFFLOAD(object);
    if (self->priv) {
        g_free(self->priv->jsonl_out_dir);
        self->priv->~GstFrameOffloadPrivate();
    }
    G_OBJECT_CLASS(gst_frame_offload_parent_class)->finalize(object);
}

static void
gst_frame_offload_set_property(GObject* object, guint prop_id,
    const GValue* value, GParamSpec* pspec)
{
    GstFrameOffload* self = GST_FRAME_OFFLOAD(object);
    switch (prop_id) {
        case PROP_CHANNEL_ID:
            self->priv->channel_id = g_value_get_uint(value);
            break;
        case PROP_PRODUCER:
            self->priv->producer =
                static_cast<dmabuf_producer::DmaBufProducer*>(g_value_get_pointer(value));
            break;
        case PROP_SCENE_UPDATE_MQTT:
            self->priv->scene_update_mqtt = g_value_get_boolean(value);
            break;
        case PROP_META_WIDTH:
            self->priv->meta_width = g_value_get_uint(value);
            break;
        case PROP_META_HEIGHT:
            self->priv->meta_height = g_value_get_uint(value);
            break;
        case PROP_GALLERY:
            self->priv->gallery =
                static_cast<multi_cam_app::gallery::FaceGallery*>(g_value_get_pointer(value));
            break;
        case PROP_GALLERY_THRESHOLD:
            self->priv->gallery_threshold = g_value_get_float(value);
            break;
        case PROP_GALLERY_MIN_FACE_SIZE:
            self->priv->gallery_min_face_size = g_value_get_uint(value);
            break;
        case PROP_GALLERY_MIN_SCORE:
            self->priv->gallery_min_score = g_value_get_float(value);
            break;
        case PROP_JSONL_OUT_DIR:
            g_free(self->priv->jsonl_out_dir);
            self->priv->jsonl_out_dir = g_value_dup_string(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_frame_offload_get_property(GObject* object, guint prop_id,
    GValue* value, GParamSpec* pspec)
{
    GstFrameOffload* self = GST_FRAME_OFFLOAD(object);
    switch (prop_id) {
        case PROP_CHANNEL_ID:
            g_value_set_uint(value, self->priv->channel_id);
            break;
        case PROP_PRODUCER:
            g_value_set_pointer(value, self->priv->producer);
            break;
        case PROP_SCENE_UPDATE_MQTT:
            g_value_set_boolean(value, self->priv->scene_update_mqtt);
            break;
        case PROP_META_WIDTH:
            g_value_set_uint(value, self->priv->meta_width);
            break;
        case PROP_META_HEIGHT:
            g_value_set_uint(value, self->priv->meta_height);
            break;
        case PROP_GALLERY:
            g_value_set_pointer(value, self->priv->gallery);
            break;
        case PROP_GALLERY_THRESHOLD:
            g_value_set_float(value, self->priv->gallery_threshold);
            break;
        case PROP_GALLERY_MIN_FACE_SIZE:
            g_value_set_uint(value, self->priv->gallery_min_face_size);
            break;
        case PROP_GALLERY_MIN_SCORE:
            g_value_set_float(value, self->priv->gallery_min_score);
            break;
        case PROP_JSONL_OUT_DIR:
            g_value_set_string(value, self->priv->jsonl_out_dir);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static gboolean
gst_frame_offload_sink_event(GstPad* pad, GstObject* parent, GstEvent* event)
{
    GstFrameOffload* self = GST_FRAME_OFFLOAD(parent);

    if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) {
        GstCaps* caps;
        gst_event_parse_caps(event, &caps);
        GstVideoInfo vinfo;
        if (gst_video_info_from_caps(&vinfo, caps)) {
            self->priv->frame_w = GST_VIDEO_INFO_WIDTH(&vinfo);
            self->priv->frame_h = GST_VIDEO_INFO_HEIGHT(&vinfo);
            GST_INFO_OBJECT(self, "sink caps: %dx%d NV12",
                self->priv->frame_w, self->priv->frame_h);
        }
        gst_event_unref(event);

        /* Push text caps on src pad (NV12 -> JSON text transformation) */
        GstCaps* src_caps = gst_caps_new_simple("text/x-raw",
            "format", G_TYPE_STRING, "utf8", NULL);
        gst_pad_push_event(self->srcpad, gst_event_new_caps(src_caps));
        gst_caps_unref(src_caps);
        return TRUE;
    }

    return gst_pad_event_default(pad, parent, event);
}

static GstFlowReturn
gst_frame_offload_chain(GstPad* pad, GstObject* parent, GstBuffer* buf)
{
    (void)pad;
    GstFrameOffload* self = GST_FRAME_OFFLOAD(parent);
    GstFrameOffloadPrivate* priv = self->priv;

    if (!priv->offload) {
        GST_ERROR_OBJECT(self, "FrameOffload not initialized");
        gst_buffer_unref(buf);
        return GST_FLOW_ERROR;
    }

    const uint32_t fw = static_cast<uint32_t>(priv->frame_w);
    const uint32_t fh = static_cast<uint32_t>(priv->frame_h);

    // 1. Map NV12 data for SubmitFrame
    int32_t image_path = -1;
    if (priv->producer != nullptr && fw > 0 && fh > 0) {
        GstMapInfo in_map;
        if (gst_buffer_map(buf, &in_map, GST_MAP_READ)) {
            GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
            int y_stride = vmeta ? static_cast<int>(vmeta->stride[0])
                                 : static_cast<int>(fw);
            int uv_offset = vmeta ? static_cast<int>(vmeta->offset[1])
                                  : (y_stride * static_cast<int>(fh));

            uint32_t slot_index = 0;
            std::string err;
            if (priv->producer->SubmitFrame(
                    in_map.data, static_cast<int>(fw), static_cast<int>(fh),
                    y_stride, uv_offset, GST_BUFFER_PTS(buf),
                    &err, &slot_index)) {
                image_path = static_cast<int32_t>(slot_index);
            } else {
                GST_WARNING_OBJECT(self, "SubmitFrame failed: %s", err.c_str());
            }
            gst_buffer_unmap(buf, &in_map);
        }
    }

    // 2. Drop frame when SubmitFrame failed — pushing JSON without image_meta
    //    downstream causes EventAdapter to reject the source, blocking slot release.
    if (image_path < 0) {
        GST_INFO_OBJECT(self, "Dropping frame: SubmitFrame failed (image_path=%d)", image_path);
        gst_buffer_unref(buf);
        return GST_FLOW_OK;
    }

    // 3. Extract meta + serialize to JSON
    std::string json = priv->offload->ExtractAndSerialize(buf, fw, fh, image_path);

    // Write to local JSONL (same JSON that goes downstream to msgagg -> MQTT)
    if (priv->jsonl_out_dir != nullptr && !json.empty()) {
        priv->offload->WriteSceneJsonl(json);
    }

    // Write to local JSONL (same JSON that goes downstream to msgagg -> MQTT)
    if (priv->jsonl_out_dir != nullptr && !json.empty()) {
        priv->offload->WriteSceneJsonl(json);
    }

    // 4. Replace buffer content: NV12 -> JSON text (reuse same GstBuffer)
    buf = gst_buffer_make_writable(buf);
    gst_buffer_remove_all_memory(buf);

    GstMemory* mem = gst_allocator_alloc(NULL, json.size(), NULL);
    GstMapInfo out_map;
    if (gst_memory_map(mem, &out_map, GST_MAP_WRITE)) {
        memcpy(out_map.data, json.data(), json.size());
        gst_memory_unmap(mem, &out_map);
    }
    gst_buffer_append_memory(buf, mem);

    // 4. Push downstream
    return gst_pad_push(self->srcpad, buf);
}

static GstStateChangeReturn
gst_frame_offload_change_state(GstElement* element, GstStateChange transition)
{
    GstFrameOffload* self = GST_FRAME_OFFLOAD(element);

    switch (transition) {
        case GST_STATE_CHANGE_NULL_TO_READY:
            break;
        case GST_STATE_CHANGE_READY_TO_PAUSED: {
            multi_cam_app::frame_offload::FrameOffloadConfig cfg;
            cfg.channel_id = self->priv->channel_id;
            cfg.scene_update_mqtt =
                (self->priv->scene_update_mqtt != FALSE);
            cfg.meta_width = self->priv->meta_width;
            cfg.meta_height = self->priv->meta_height;
            cfg.gallery = self->priv->gallery;
            cfg.gallery_threshold = self->priv->gallery_threshold;
            cfg.gallery_min_face_size = self->priv->gallery_min_face_size;
            cfg.gallery_min_score = self->priv->gallery_min_score;
            if (self->priv->jsonl_out_dir != nullptr) {
                cfg.jsonl_out_dir = self->priv->jsonl_out_dir;
            }
            self->priv->offload =
                std::make_unique<multi_cam_app::frame_offload::FrameOffload>(cfg);
            GST_INFO_OBJECT(self, "FrameOffload created: channel_id=%u, producer=%p",
                self->priv->channel_id, (void*)self->priv->producer);
            break;
        }
        default:
            break;
    }

    GstStateChangeReturn ret =
        GST_ELEMENT_CLASS(gst_frame_offload_parent_class)->change_state(element, transition);

    switch (transition) {
        case GST_STATE_CHANGE_PAUSED_TO_READY:
            if (self->priv->offload) {
                GST_INFO_OBJECT(self, "stopping: frames_processed=%lu",
                    (unsigned long)self->priv->offload->FramesProcessed());
                self->priv->offload.reset();
            }
            self->priv->frame_w = 0;
            self->priv->frame_h = 0;
            break;
        default:
            break;
    }

    return ret;
}

/* Plugin registration */
static gboolean
plugin_init(GstPlugin* plugin)
{
    return gst_element_register(plugin, "qtiframeoffload",
        GST_RANK_NONE, GST_TYPE_FRAME_OFFLOAD);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    qtiframeoffload,
    "QTI Frame Offload (NV12 -> DMA cache + JSON metadata)",
    plugin_init,
    "1.0",
    "Proprietary",
    "multi_cam_app",
    "https://www.qualcomm.com"
)
