/// @file gstdeduper.cpp
/// @brief GstDeduper plugin implementation
///
/// GstBaseTransform in-place subclass:
///   - set_caps:     parse NV12 caps (get frame dimensions for ROI coord normalization)
///   - transform_ip: read ROI meta -> whitelist filter -> IoU dedup -> pass/drop
///   - stop:         reset Deduper state

#include "gstdeduper.h"
#include "../Deduper.hpp"

#include <gst/video/video.h>
#include <gst/video/gstvideometa.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

GST_DEBUG_CATEGORY_STATIC (gst_deduper_debug);
#define GST_CAT_DEFAULT gst_deduper_debug

/* Properties */
enum {
    PROP_0,
    PROP_IOU_THRESHOLD,
};

struct _GstDeduperPrivate {
    std::unique_ptr<deduper::Deduper> deduper;
    GstVideoInfo vinfo;
    float iou_threshold;
    uint64_t frames_total;
    uint64_t frames_passed;
    uint64_t frames_dropped;
    /* Per-buffer timing (microseconds) */
    uint64_t time_total_us;
    uint64_t time_min_us;
    uint64_t time_max_us;
};

/* Pad templates: NV12 in, NV12 out (passthrough format) */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format=(string)NV12"));

#define gst_deduper_parent_class parent_class
G_DEFINE_TYPE_WITH_PRIVATE (GstDeduper, gst_deduper, GST_TYPE_BASE_TRANSFORM)

/* Forward declarations */
static void gst_deduper_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec);
static void gst_deduper_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec);
static void gst_deduper_finalize (GObject *object);
static gboolean gst_deduper_set_caps (GstBaseTransform *base,
    GstCaps *incaps, GstCaps *outcaps);
static GstFlowReturn gst_deduper_transform_ip (GstBaseTransform *base,
    GstBuffer *buffer);
static gboolean gst_deduper_stop (GstBaseTransform *base);

static void
gst_deduper_class_init (GstDeduperClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
    GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS (klass);

    gobject_class->set_property = gst_deduper_set_property;
    gobject_class->get_property = gst_deduper_get_property;
    gobject_class->finalize     = gst_deduper_finalize;

    g_object_class_install_property (gobject_class, PROP_IOU_THRESHOLD,
        g_param_spec_float ("iou-threshold", "IoU Threshold",
            "IoU threshold for frame deduplication (lower = more frames pass)",
            0.0f, 1.0f, 0.75f,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_static_metadata (element_class,
        "QTI Detection Deduper",
        "Filter/Video",
        "Class whitelist filtering + IoU-based frame deduplication",
        "multi_cam_app");

    gst_element_class_add_static_pad_template (element_class, &sink_template);
    gst_element_class_add_static_pad_template (element_class, &src_template);

    transform_class->set_caps     = GST_DEBUG_FUNCPTR (gst_deduper_set_caps);
    transform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_deduper_transform_ip);
    transform_class->stop         = GST_DEBUG_FUNCPTR (gst_deduper_stop);

    /* In-place mode: buffer is made writable so we can remove ROI meta.
     * passthrough_on_same_caps = FALSE ensures writable even when caps match. */
    transform_class->passthrough_on_same_caps = FALSE;

    GST_DEBUG_CATEGORY_INIT (gst_deduper_debug, "qtideduper", 0,
        "QTI Detection Deduper");
}

static void
gst_deduper_init (GstDeduper *self)
{
    self->priv = (GstDeduperPrivate *)
        gst_deduper_get_instance_private (self);
    new (self->priv) GstDeduperPrivate ();
    self->priv->iou_threshold = 0.75f;
    self->priv->frames_total = 0;
    self->priv->frames_passed = 0;
    self->priv->frames_dropped = 0;
    self->priv->time_total_us = 0;
    self->priv->time_min_us = UINT64_MAX;
    self->priv->time_max_us = 0;

    gst_base_transform_set_in_place (GST_BASE_TRANSFORM (self), TRUE);
}

static void
gst_deduper_finalize (GObject *object)
{
    GstDeduper *self = GST_DEDUPER (object);
    if (self->priv) {
        self->priv->~GstDeduperPrivate ();
    }
    G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_deduper_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
    GstDeduper *self = GST_DEDUPER (object);
    switch (prop_id) {
        case PROP_IOU_THRESHOLD:
            self->priv->iou_threshold = g_value_get_float (value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

static void
gst_deduper_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
    GstDeduper *self = GST_DEDUPER (object);
    switch (prop_id) {
        case PROP_IOU_THRESHOLD:
            g_value_set_float (value, self->priv->iou_threshold);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

static gboolean
gst_deduper_set_caps (GstBaseTransform *base, GstCaps *incaps, GstCaps *outcaps)
{
    (void) outcaps;
    GstDeduper *self = GST_DEDUPER (base);

    if (!gst_video_info_from_caps (&self->priv->vinfo, incaps)) {
        GST_ERROR_OBJECT (self, "Failed to parse video info from caps");
        return FALSE;
    }

    /* 初始化 Deduper */
    deduper::DeduperConfig cfg;
    cfg.iou_threshold = self->priv->iou_threshold;
    self->priv->deduper = std::make_unique<deduper::Deduper> (cfg);

    GST_INFO_OBJECT (self, "set_caps: %dx%d NV12, iou_threshold=%.2f",
        GST_VIDEO_INFO_WIDTH (&self->priv->vinfo),
        GST_VIDEO_INFO_HEIGHT (&self->priv->vinfo),
        self->priv->iou_threshold);

    return TRUE;
}

/*
 * transform_ip: 核心处理函数（三阶段）
 *
 * 步骤:
 *   1. 遍历 buffer 上所有 GstVideoRegionOfInterestMeta
 *   2. 白名单过滤：非白名单类别的 ROI meta 标记为待移除
 *   3. 收集白名单 ROI -> 构造 TrackEntry 列表
 *   4. 移除非白名单 ROI meta（迭代结束后执行，避免迭代器失效）
 *  4a. 空帧检查：entries 为空 → 直接丢弃 buffer + 清空去重状态
 *   5. 调用 Deduper::IsInteresting() 判断帧是否有变化
 *   6. 无变化 -> 返回 GST_BASE_TRANSFORM_FLOW_DROPPED
 *      有变化 -> 返回 GST_FLOW_OK（buffer 携带过滤后的 ROI meta 推送到下游）
 *
 * GstFaceDetectionMeta: 完全不处理，透传到下游。
 */
static GstFlowReturn
gst_deduper_transform_ip (GstBaseTransform *base, GstBuffer *buffer)
{
    GstDeduper *self = GST_DEDUPER (base);
    GstDeduperPrivate *priv = self->priv;

    if (!priv->deduper) {
        return GST_FLOW_OK;
    }

    const guint frame_w = (guint) GST_VIDEO_INFO_WIDTH (&priv->vinfo);
    const guint frame_h = (guint) GST_VIDEO_INFO_HEIGHT (&priv->vinfo);
    if (frame_w == 0 || frame_h == 0) {
        return GST_FLOW_OK;
    }

    priv->frames_total++;

    const gint64 t_start = g_get_monotonic_time ();

    /* --- Step 1-3: 遍历 ROI meta，收集白名单条目 + 标记非白名单条目 --- */
    std::vector<deduper::TrackEntry> entries;
    std::vector<GstMeta*> to_remove;

    gpointer state = NULL;
    GstMeta *meta;
    while ((meta = gst_buffer_iterate_meta (buffer, &state)) != NULL) {
        if (meta->info->api != GST_VIDEO_REGION_OF_INTEREST_META_API_TYPE)
            continue;

        GstVideoRegionOfInterestMeta *roi =
            (GstVideoRegionOfInterestMeta *) meta;

        const gchar *label_str = g_quark_to_string (roi->roi_type);
        std::string label = (label_str != NULL) ? label_str : "";

        if (!deduper::Deduper::IsAllowed (label)) {
            to_remove.push_back (meta);
            GST_DEBUG_OBJECT (self,
                "frame %lu: FILTERED-OUT ROI: %s#%u @ (%u,%u,%ux%u) — not in whitelist",
                priv->frames_total, label.c_str(), roi->id,
                roi->x, roi->y, roi->w, roi->h);
            continue;
        }

        /* 构造 TrackEntry（归一化 ROI 绝对像素坐标到 [0,1]） */
        deduper::TrackEntry entry;
        entry.label  = label;
        entry.left   = static_cast<double>(roi->x)           / frame_w;
        entry.top    = static_cast<double>(roi->y)           / frame_h;
        entry.right  = static_cast<double>(roi->x + roi->w)  / frame_w;
        entry.bottom = static_cast<double>(roi->y + roi->h)  / frame_h;

        /* 从 ROI params 读 tracking-id
         * metamux 将 tracker 字段存入 GstStructure "ObjectDetection"。
         * roi->id 可能为 0（metamux 读 "id" 字段，tracker 输出 "tracking-id"，不匹配）。
         * 必须从 params 中读取。 */
        for (GList *p = roi->params; p != NULL; p = p->next) {
            GstStructure *s = (GstStructure *) p->data;
            guint tid = 0;
            if (gst_structure_get_uint (s, "tracking-id", &tid)) {
                entry.track_id = tid;
            }
        }

        entries.push_back (std::move (entry));
    }

    /* --- Step 4: 移除非白名单 ROI meta（必须在迭代完成后） ---
     * gst_buffer_iterate_meta 的 state 在迭代完成后已无效，
     * 但收集的 GstMeta* 指针仍然有效（链表节点未被修改）。
     * gst_buffer_remove_meta 移除一个节点不影响其他节点的指针。 */
    for (GstMeta *m : to_remove) {
        gst_buffer_remove_meta (buffer, m);
    }

    if (!to_remove.empty()) {
        GST_DEBUG_OBJECT (self, "frame %lu: removed %zu non-whitelist ROI meta",
            priv->frames_total, to_remove.size());
    }

    /* --- Step 4a: 白名单过滤后无剩余检测 → 直接丢弃 ---
     * 如果 buffer 上所有 detection 都被白名单过滤掉了（或原本就没有），
     * 没有有效数据需要处理，直接丢弃整个 buffer 并释放资源。
     * 同时清空去重状态，下次有目标时视为全新出现。 */
    if (entries.empty()) {
        priv->frames_dropped++;
        priv->deduper->Reset();
        {
            const uint64_t elapsed = (uint64_t)(g_get_monotonic_time () - t_start);
            priv->time_total_us += elapsed;
            if (elapsed < priv->time_min_us) priv->time_min_us = elapsed;
            if (elapsed > priv->time_max_us) priv->time_max_us = elapsed;
        }
        GST_LOG_OBJECT (self,
            "frame %lu: no whitelisted detections (removed=%zu), dropped",
            priv->frames_total, to_remove.size());
        return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

    /* --- Step 5: IoU 帧级去重 --- */
    const gboolean interesting = priv->deduper->IsInteresting (entries);

    /* Record per-buffer timing */
    const uint64_t elapsed_us = (uint64_t)(g_get_monotonic_time () - t_start);
    priv->time_total_us += elapsed_us;
    if (elapsed_us < priv->time_min_us) priv->time_min_us = elapsed_us;
    if (elapsed_us > priv->time_max_us) priv->time_max_us = elapsed_us;

    if (!interesting) {
        priv->frames_dropped++;

        for (const auto& e : entries) {
            GST_DEBUG_OBJECT (self,
                "frame %lu: DEDUP-DROP ROI: %s#%u @ (%.4f,%.4f,%.4f,%.4f) — IoU unchanged",
                priv->frames_total, e.label.c_str(), e.track_id,
                e.left, e.top, e.right, e.bottom);
        }

        if ((priv->frames_total % 200) == 0) {
            const double avg = priv->frames_total > 0
                ? (double) priv->time_total_us / priv->frames_total : 0.0;
            GST_INFO_OBJECT (self,
                "dedup stats: total=%lu passed=%lu dropped=%lu (%.1f%% drop)"
                " | time: avg=%.1fus min=%luus max=%luus",
                priv->frames_total, priv->frames_passed, priv->frames_dropped,
                100.0 * priv->frames_dropped / priv->frames_total,
                avg,
                (unsigned long) priv->time_min_us,
                (unsigned long) priv->time_max_us);
        }

        return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

    /* --- Step 6: 有变化，通过 --- */
    priv->frames_passed++;

    if ((priv->frames_total % 200) == 0) {
        const double avg = priv->frames_total > 0
            ? (double) priv->time_total_us / priv->frames_total : 0.0;
        GST_INFO_OBJECT (self,
            "dedup stats: total=%lu passed=%lu dropped=%lu (%.1f%% drop)"
            " | time: avg=%.1fus min=%luus max=%luus",
            priv->frames_total, priv->frames_passed, priv->frames_dropped,
            100.0 * priv->frames_dropped / priv->frames_total,
            avg,
            (unsigned long) priv->time_min_us,
            (unsigned long) priv->time_max_us);
    }

    return GST_FLOW_OK;
}

static gboolean
gst_deduper_stop (GstBaseTransform *base)
{
    GstDeduper *self = GST_DEDUPER (base);

    {
        const uint64_t total = self->priv->frames_total;
        const double avg = total > 0
            ? (double) self->priv->time_total_us / total : 0.0;
        const uint64_t min_us = (self->priv->time_min_us == UINT64_MAX)
            ? 0 : self->priv->time_min_us;
        GST_INFO_OBJECT (self,
            "stopping: total=%lu passed=%lu dropped=%lu"
            " | time: avg=%.1fus min=%luus max=%luus total=%lums",
            total, self->priv->frames_passed, self->priv->frames_dropped,
            avg, (unsigned long) min_us, (unsigned long) self->priv->time_max_us,
            (unsigned long)(self->priv->time_total_us / 1000));
    }

    if (self->priv->deduper) {
        self->priv->deduper->Reset ();
        self->priv->deduper.reset ();
    }
    self->priv->frames_total = 0;
    self->priv->frames_passed = 0;
    self->priv->frames_dropped = 0;
    self->priv->time_total_us = 0;
    self->priv->time_min_us = UINT64_MAX;
    self->priv->time_max_us = 0;
    return TRUE;
}

/* Plugin registration */
static gboolean
plugin_init (GstPlugin *plugin)
{
    return gst_element_register (plugin, "qtideduper",
        GST_RANK_NONE, GST_TYPE_DEDUPER);
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    qtideduper,
    "QTI Detection Deduper (class whitelist + IoU frame dedup)",
    plugin_init,
    "1.0",
    "Proprietary",
    "multi_cam_app",
    "https://www.qualcomm.com"
)
