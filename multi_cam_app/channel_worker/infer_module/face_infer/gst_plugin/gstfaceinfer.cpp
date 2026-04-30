/// @file gstfaceinfer.cpp
/// @brief GstFaceInfer plugin implementation
///
/// GstBaseTransform in-place subclass:
///   - set_caps: parse NV12 caps, init FaceProcessor (once)
///   - transform_ip: DMABuf fd -> cl_mem -> FaceProcessor -> GstFaceDetectionMeta
///   - stop: destroy FaceProcessor

#include "gstfaceinfer.h"
#include "gstfacedetectionmeta.h"

#include "../FaceProcessor.hpp"
#include "../config/FaceConfigLoader.hpp"
#include "../mem_management/opencl_loader.hpp"
#include "../mem_management/mem_types.hpp"

#include <gst/video/video.h>
#include <gst/allocators/gstfdmemory.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

GST_DEBUG_CATEGORY_STATIC (gst_face_infer_debug);
#define GST_CAT_DEFAULT gst_face_infer_debug

/* Properties */
enum {
    PROP_0,
    PROP_CONFIG_PATH,
    PROP_FACE_INTERVAL_MS,
};

struct _GstFaceInferPrivate {
    std::string config_path;
    std::unique_ptr<face_infer::FaceProcessor> processor;
    GstVideoInfo vinfo;
    gboolean processor_initialized;
    guint64 last_face_ms{0};
    guint   face_interval_ms{0};  // 0=every frame
};

/* Pad templates: NV12 in, NV12 out (passthrough format) */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format=(string)NV12"));

#define gst_face_infer_parent_class parent_class
G_DEFINE_TYPE_WITH_PRIVATE (GstFaceInfer, gst_face_infer, GST_TYPE_BASE_TRANSFORM)

/* Forward declarations */
static void gst_face_infer_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec);
static void gst_face_infer_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec);
static void gst_face_infer_finalize (GObject *object);
static gboolean gst_face_infer_set_caps (GstBaseTransform *base,
    GstCaps *incaps, GstCaps *outcaps);
static GstFlowReturn gst_face_infer_transform_ip (GstBaseTransform *base,
    GstBuffer *buffer);
static gboolean gst_face_infer_stop (GstBaseTransform *base);

static void
gst_face_infer_class_init (GstFaceInferClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
    GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS (klass);

    gobject_class->set_property = gst_face_infer_set_property;
    gobject_class->get_property = gst_face_infer_get_property;
    gobject_class->finalize = gst_face_infer_finalize;

    g_object_class_install_property (gobject_class, PROP_CONFIG_PATH,
        g_param_spec_string ("config-path", "Config Path",
            "Path to face_config.json",
            "", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_FACE_INTERVAL_MS,
        g_param_spec_uint ("face-interval-ms", "Face Interval (ms)",
            "Minimum interval between face detections in ms (0=every frame)",
            0U, G_MAXUINT, 0U,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_static_metadata (element_class,
        "QTI Face Inference",
        "Filter/Analyzer/Video",
        "SCRFD face detection + ArcFace recognition (OpenCL + QNN HTP)",
        "multi_cam_app");

    gst_element_class_add_static_pad_template (element_class, &sink_template);
    gst_element_class_add_static_pad_template (element_class, &src_template);

    transform_class->set_caps = GST_DEBUG_FUNCPTR (gst_face_infer_set_caps);
    transform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_face_infer_transform_ip);
    transform_class->stop = GST_DEBUG_FUNCPTR (gst_face_infer_stop);

    /* In-place mode: framework calls gst_buffer_make_writable() so we can
     * attach GstMeta. Video data is zero-copy (DMABuf refcount+1). */
    transform_class->passthrough_on_same_caps = FALSE;

    GST_DEBUG_CATEGORY_INIT (gst_face_infer_debug, "qtifaceinfer", 0,
        "QTI Face Inference");
}

static void
gst_face_infer_init (GstFaceInfer *self)
{
    self->priv = (GstFaceInferPrivate *)
        gst_face_infer_get_instance_private (self);
    new (self->priv) GstFaceInferPrivate ();
    self->priv->processor_initialized = FALSE;

    gst_base_transform_set_in_place (GST_BASE_TRANSFORM (self), TRUE);
}

static void
gst_face_infer_finalize (GObject *object)
{
    GstFaceInfer *self = GST_FACE_INFER (object);
    if (self->priv) {
        self->priv->~GstFaceInferPrivate ();
    }
    G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_face_infer_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
    GstFaceInfer *self = GST_FACE_INFER (object);
    switch (prop_id) {
        case PROP_CONFIG_PATH:
            self->priv->config_path = g_value_get_string (value);
            break;
        case PROP_FACE_INTERVAL_MS:
            self->priv->face_interval_ms = g_value_get_uint (value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

static void
gst_face_infer_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
    GstFaceInfer *self = GST_FACE_INFER (object);
    switch (prop_id) {
        case PROP_CONFIG_PATH:
            g_value_set_string (value, self->priv->config_path.c_str ());
            break;
        case PROP_FACE_INTERVAL_MS:
            g_value_set_uint (value, self->priv->face_interval_ms);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

static gboolean
gst_face_infer_set_caps (GstBaseTransform *base,
    GstCaps *incaps, GstCaps *outcaps)
{
    (void) outcaps;
    GstFaceInfer *self = GST_FACE_INFER (base);

    if (!gst_video_info_from_caps (&self->priv->vinfo, incaps)) {
        GST_ERROR_OBJECT (self, "Failed to parse video info from caps");
        return FALSE;
    }

    GST_INFO_OBJECT (self, "set_caps: %dx%d NV12",
        GST_VIDEO_INFO_WIDTH (&self->priv->vinfo),
        GST_VIDEO_INFO_HEIGHT (&self->priv->vinfo));

    /* Initialize FaceProcessor once (survives caps renegotiation if same format) */
    if (!self->priv->processor_initialized) {
        if (self->priv->config_path.empty ()) {
            GST_ERROR_OBJECT (self, "config-path property not set");
            return FALSE;
        }

        face_infer::FaceProcessorConfig cfg;
        if (!face_infer::LoadFaceConfig (self->priv->config_path, cfg)) {
            GST_ERROR_OBJECT (self, "Failed to load face config: %s",
                self->priv->config_path.c_str ());
            return FALSE;
        }

        self->priv->processor = std::make_unique<face_infer::FaceProcessor> ();
        if (!self->priv->processor->Init (cfg)) {
            GST_ERROR_OBJECT (self, "FaceProcessor::Init failed");
            self->priv->processor.reset ();
            return FALSE;
        }

        self->priv->processor_initialized = TRUE;
        GST_INFO_OBJECT (self, "FaceProcessor initialized (ArcFace=%s)",
            self->priv->processor->HasArcFace () ? "on" : "off");
    }

    return TRUE;
}

static GstFlowReturn
gst_face_infer_transform_ip (GstBaseTransform *base, GstBuffer *buffer)
{
    GstFaceInfer *self = GST_FACE_INFER (base);
    GstFaceInferPrivate *priv = self->priv;

    if (priv->face_interval_ms > 0) {
        guint64 now_ms = g_get_monotonic_time() / 1000;
        if ((now_ms - priv->last_face_ms) < priv->face_interval_ms) {
            GST_LOG_OBJECT (self, "face-interval gate skip (last_ms=%"
                G_GUINT64_FORMAT " now_ms=%" G_GUINT64_FORMAT " interval=%u)",
                priv->last_face_ms, now_ms, priv->face_interval_ms);
            return GST_FLOW_OK;
        }
        priv->last_face_ms = now_ms;
    }

    if (!priv->processor_initialized || !priv->processor) {
        return GST_FLOW_OK;
    }

    /* Map video frame to extract DMABuf fd and plane info */
    GstVideoFrame vframe;
    if (!gst_video_frame_map (&vframe, &priv->vinfo, buffer, GST_MAP_READ)) {
        GST_WARNING_OBJECT (self, "Failed to map video frame");
        return GST_FLOW_OK;
    }

    GstMemory *mem = gst_buffer_peek_memory (buffer, 0);
    if (mem == NULL || !gst_is_fd_memory (mem)) {
        GST_WARNING_OBJECT (self, "Buffer is not FD-backed, skipping");
        gst_video_frame_unmap (&vframe);
        return GST_FLOW_OK;
    }

    int fd = gst_fd_memory_get_fd (mem);

    face_infer::FramePlaneInfo plane {};
    plane.fd        = fd;
    plane.frame_len = static_cast<uint32_t> (gst_buffer_get_size (buffer));
    plane.y_offset  = static_cast<uint32_t> (GST_VIDEO_FRAME_PLANE_OFFSET (&vframe, 0));
    plane.uv_offset = static_cast<uint32_t> (GST_VIDEO_FRAME_PLANE_OFFSET (&vframe, 1));
    plane.y_stride  = static_cast<uint32_t> (GST_VIDEO_FRAME_PLANE_STRIDE (&vframe, 0));
    plane.uv_stride = static_cast<uint32_t> (GST_VIDEO_FRAME_PLANE_STRIDE (&vframe, 1));
    plane.width     = static_cast<uint32_t> (GST_VIDEO_FRAME_WIDTH (&vframe));
    plane.height    = static_cast<uint32_t> (GST_VIDEO_FRAME_HEIGHT (&vframe));

    /* Per-frame cl_mem import (same pattern as overlay plugin) */
    auto ocl = face_infer::OpenClLoader::Get ();
    cl_context ctx = priv->processor->GetContext ();
    cl_mem nv12_cl = nullptr;

    if (ocl && ctx) {
        void *vaddr = GST_VIDEO_FRAME_PLANE_DATA (&vframe, 0);
        cl_mem_ion_host_ptr ionmem {};
        ionmem.ext_host_ptr.allocation_type = CL_MEM_DMABUF_HOST_PTR_QCOM;
        ionmem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
        ionmem.ion_hostptr = vaddr;
        ionmem.ion_filedesc = fd;

        cl_int rc;
        cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR
                           | CL_MEM_EXT_HOST_PTR_QCOM;
        nv12_cl = ocl->CreateBuffer (ctx, flags, plane.frame_len, &ionmem, &rc);
        if (rc != CL_SUCCESS) {
            GST_WARNING_OBJECT (self, "clCreateBuffer(DMABuf) failed rc=%d", rc);
            nv12_cl = nullptr;
        }
    }

    /* Run face detection + recognition */
    std::vector<face_infer::FaceResult> faces;
    if (nv12_cl != nullptr) {
        faces = priv->processor->ProcessFrame (nv12_cl, plane);
        if (ocl) {
            ocl->ReleaseMemObject (nv12_cl);
        }
    }

    gst_video_frame_unmap (&vframe);

    /* Attach results as GstFaceDetectionMeta */
    if (!faces.empty ()) {
        GstFaceDetectionMeta *meta = gst_buffer_add_face_detection_meta (buffer);
        if (meta != NULL) {
            guint n = static_cast<guint> (faces.size ());
            if (n > FACE_META_MAX_FACES) n = FACE_META_MAX_FACES;
            meta->n_faces = n;

            gboolean has_arc = priv->processor->HasArcFace () ? TRUE : FALSE;
            for (guint i = 0; i < n; i++) {
                GstFaceDetectionEntry *e = &meta->faces[i];
                const face_infer::FaceResult &fr = faces[i];
                e->x1 = fr.det.x1;
                e->y1 = fr.det.y1;
                e->x2 = fr.det.x2;
                e->y2 = fr.det.y2;
                e->score = fr.det.score;
                memcpy (e->landmarks, fr.det.landmarks, sizeof (e->landmarks));
                memcpy (e->embedding, fr.embedding, sizeof (e->embedding));
                e->has_embedding = has_arc;
            }
        }
    }

    return GST_FLOW_OK;
}

static gboolean
gst_face_infer_stop (GstBaseTransform *base)
{
    GstFaceInfer *self = GST_FACE_INFER (base);
    if (self->priv->processor) {
        self->priv->processor->Destroy ();
        self->priv->processor.reset ();
    }
    self->priv->processor_initialized = FALSE;
    GST_INFO_OBJECT (self, "stopped");
    return TRUE;
}

/* Plugin registration */
static gboolean
plugin_init (GstPlugin *plugin)
{
    return gst_element_register (plugin, "qtifaceinfer",
        GST_RANK_NONE, GST_TYPE_FACE_INFER);
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    qtifaceinfer,
    "QTI Face Detection + Recognition (SCRFD + ArcFace)",
    plugin_init,
    "1.0",
    "Proprietary",
    "multi_cam_app",
    "https://www.qualcomm.com"
)
