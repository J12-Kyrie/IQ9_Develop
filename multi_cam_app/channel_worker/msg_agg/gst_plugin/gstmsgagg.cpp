#include "gstmsgagg.h"

#include <cstdio>
#include <cstring>  // strlen, memcpy

#include <json-glib/json-glib.h>

GST_DEBUG_CATEGORY_STATIC(gst_msg_agg_debug);
#define GST_CAT_DEFAULT gst_msg_agg_debug

#define DEFAULT_TIMEOUT_MS 500

enum {
    PROP_0,
    PROP_TIMEOUT,
    PROP_MERGE_SCENE_UPDATE,
};

/* ========================================================================
 * GstMsgAggDataPad — custom pad GType (borrowing metamux pattern)
 * ======================================================================== */

G_DEFINE_TYPE(GstMsgAggDataPad, gst_msg_agg_data_pad, GST_TYPE_PAD);

static void
gst_msg_agg_data_pad_finalize(GObject *object)
{
    GstMsgAggDataPad *dpad = GST_MSG_AGG_DATA_PAD(object);

    if (dpad->queue) {
        gchar *item;
        while ((item = (gchar *)g_queue_pop_head(dpad->queue)) != NULL)
            g_free(item);
        g_queue_free(dpad->queue);
        dpad->queue = NULL;
    }

    G_OBJECT_CLASS(gst_msg_agg_data_pad_parent_class)->finalize(object);
}

static void
gst_msg_agg_data_pad_class_init(GstMsgAggDataPadClass *klass)
{
    GObjectClass *gobject = (GObjectClass *)klass;
    gobject->finalize = GST_DEBUG_FUNCPTR(gst_msg_agg_data_pad_finalize);
}

static void
gst_msg_agg_data_pad_init(GstMsgAggDataPad *pad)
{
    pad->queue = g_queue_new();
    pad->eos = FALSE;
}

/* ========================================================================
 * GstMsgAgg — main element
 * ======================================================================== */

G_DEFINE_TYPE(GstMsgAgg, gst_msg_agg, GST_TYPE_ELEMENT);

/* Pad templates */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("data_%u",
    GST_PAD_SINK, GST_PAD_REQUEST,
    GST_STATIC_CAPS("text/x-raw, format=(string)utf8"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("text/x-raw, format=(string)utf8"));

/* Forward declarations */
static void gst_msg_agg_finalize(GObject *object);
static void gst_msg_agg_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec);
static void gst_msg_agg_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec);
static GstStateChangeReturn gst_msg_agg_change_state(GstElement *element,
    GstStateChange transition);
static GstPad *gst_msg_agg_request_pad(GstElement *element,
    GstPadTemplate *templ, const gchar *reqname, const GstCaps *caps);
static void gst_msg_agg_release_pad(GstElement *element, GstPad *pad);

static GstFlowReturn gst_msg_agg_data_chain(GstPad *pad, GstObject *parent,
    GstBuffer *buffer);
static gboolean gst_msg_agg_data_event(GstPad *pad, GstObject *parent,
    GstEvent *event);
static void gst_msg_agg_worker_task(gpointer userdata);

/* ---- helpers ---- */

static gchar *
gst_msg_agg_build_scene_merge(GstMsgAgg *agg) {
    JsonArray *sources = json_array_new();
    guint64 max_fid = 0U;

    for (GList *l = agg->datapads; l; l = l->next) {
        GstMsgAggDataPad *dpad = GST_MSG_AGG_DATA_PAD(l->data);
        gchar *json = (gchar *)g_queue_pop_head(dpad->queue);
        if (json == NULL) {
            continue;
        }
        JsonParser *parser = json_parser_new();
        GError *jerr = NULL;
        if (!json_parser_load_from_data(parser, json, -1, &jerr)) {
            GST_DEBUG_OBJECT(agg, "merge: parse fail: %s",
                (jerr != NULL && jerr->message) ? jerr->message : "?");
            g_clear_error(&jerr);
            g_object_unref(parser);
            g_free(json);
            continue;
        }
        JsonNode *root = json_parser_get_root(parser);
        if (root != NULL && json_node_get_node_type(root) == JSON_NODE_OBJECT) {
            JsonObject *o = json_node_get_object(root);
            if (json_object_has_member(o, "frame_id")) {
                gint64 fi = json_object_get_int_member(o, "frame_id");
                if (fi > (gint64)max_fid) {
                    max_fid = (guint64)fi;
                }
            }
            json_array_add_element(sources, json_node_copy(root));
        }
        g_object_unref(parser);
        g_free(json);
    }

    JsonObject *msg = json_object_new();
    json_object_set_string_member(msg, "type", "scene_update");
    json_object_set_array_member(msg, "sources", sources);

    JsonNode *n = json_node_new(JSON_NODE_OBJECT);
    json_node_take_object(n, msg);
    gchar *out = json_to_string(n, FALSE);
    json_node_free(n);
    return out;
}

static gboolean
all_pads_have_data(GstMsgAgg *agg)
{
    if (agg->datapads == NULL)
        return FALSE;

    for (GList *l = agg->datapads; l; l = l->next) {
        GstMsgAggDataPad *dpad = GST_MSG_AGG_DATA_PAD(l->data);
        if (!dpad->eos && g_queue_is_empty(dpad->queue))
            return FALSE;
    }
    return TRUE;
}

static gboolean
all_pads_eos(GstMsgAgg *agg)
{
    if (agg->datapads == NULL)
        return FALSE;

    for (GList *l = agg->datapads; l; l = l->next) {
        GstMsgAggDataPad *dpad = GST_MSG_AGG_DATA_PAD(l->data);
        if (!dpad->eos)
            return FALSE;
    }
    return TRUE;
}

static gboolean
any_pad_has_data(GstMsgAgg *agg)
{
    for (GList *l = agg->datapads; l; l = l->next) {
        GstMsgAggDataPad *dpad = GST_MSG_AGG_DATA_PAD(l->data);
        if (!g_queue_is_empty(dpad->queue))
            return TRUE;
    }
    return FALSE;
}

/* ---- class_init / init ---- */

static void
gst_msg_agg_class_init(GstMsgAggClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

    gobject_class->finalize = gst_msg_agg_finalize;
    gobject_class->set_property = gst_msg_agg_set_property;
    gobject_class->get_property = gst_msg_agg_get_property;

    g_object_class_install_property(gobject_class, PROP_TIMEOUT,
        g_param_spec_uint("timeout", "Timeout (ms)",
            "Wait timeout in milliseconds before using {} placeholder for missing pads",
            0, G_MAXUINT, DEFAULT_TIMEOUT_MS,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_MERGE_SCENE_UPDATE,
        g_param_spec_boolean("merge-scene-update", "Merge scene_update",
            "When TRUE, combine per-pad JSON objects into one scene_update payload",
            FALSE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    element_class->change_state = gst_msg_agg_change_state;
    element_class->request_new_pad = gst_msg_agg_request_pad;
    element_class->release_pad = gst_msg_agg_release_pad;

    gst_element_class_set_static_metadata(element_class,
        "QTI Message Aggregator",
        "Generic",
        "Aggregate N text/x-raw JSON streams into a single combined output",
        "multi_cam_app");

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    GST_DEBUG_CATEGORY_INIT(gst_msg_agg_debug, "qtimsgagg", 0,
        "QTI Message Aggregator");
}

static void
gst_msg_agg_init(GstMsgAgg *agg)
{
    g_mutex_init(&agg->lock);
    g_cond_init(&agg->wakeup);

    agg->datapads = NULL;
    agg->nextidx = 0;
    agg->active = FALSE;
    agg->started = FALSE;
    agg->timeout_ms = DEFAULT_TIMEOUT_MS;
    agg->merge_scene_update = FALSE;
    agg->batches_pushed = 0;
    agg->timeout_count = 0;
    agg->placeholder_total = 0;
    agg->placeholder_eos = 0;
    agg->placeholder_timeout = 0;

    /* Create src pad (ALWAYS) */
    agg->srcpad = gst_pad_new_from_static_template(&src_template, "src");
    gst_element_add_pad(GST_ELEMENT(agg), agg->srcpad);

    /* Create worker task */
    g_rec_mutex_init(&agg->worklock);
    agg->worker = gst_task_new(gst_msg_agg_worker_task, agg, NULL);
    gst_task_set_lock(agg->worker, &agg->worklock);
}

static void
gst_msg_agg_finalize(GObject *object)
{
    GstMsgAgg *agg = GST_MSG_AGG(object);

    if (agg->worker) {
        gst_task_join(agg->worker);
        gst_object_unref(agg->worker);
        agg->worker = NULL;
    }

    g_rec_mutex_clear(&agg->worklock);

    g_list_free(agg->datapads);
    agg->datapads = NULL;

    g_cond_clear(&agg->wakeup);
    g_mutex_clear(&agg->lock);

    GST_INFO_OBJECT(agg, "finalize: teardown complete");

    G_OBJECT_CLASS(gst_msg_agg_parent_class)->finalize(object);
}

/* ---- properties ---- */

static void
gst_msg_agg_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
    GstMsgAgg *agg = GST_MSG_AGG(object);
    switch (prop_id) {
    case PROP_TIMEOUT:
        agg->timeout_ms = g_value_get_uint(value);
        break;
    case PROP_MERGE_SCENE_UPDATE:
        agg->merge_scene_update = g_value_get_boolean(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_msg_agg_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
    GstMsgAgg *agg = GST_MSG_AGG(object);
    switch (prop_id) {
    case PROP_TIMEOUT:
        g_value_set_uint(value, agg->timeout_ms);
        break;
    case PROP_MERGE_SCENE_UPDATE:
        g_value_set_boolean(value, agg->merge_scene_update);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/* ---- request / release pad ---- */

static GstPad *
gst_msg_agg_request_pad(GstElement *element, GstPadTemplate *templ,
    const gchar *reqname, const GstCaps *caps)
{
    (void)caps;
    GstMsgAgg *agg = GST_MSG_AGG(element);
    guint index = 0;
    guint nextindex = 0;

    g_mutex_lock(&agg->lock);

    if (reqname && sscanf(reqname, "data_%u", &index) == 1) {
        nextindex = (index >= agg->nextidx) ? index + 1 : agg->nextidx;
    } else {
        index = agg->nextidx;
        nextindex = index + 1;
    }

    gchar *name = g_strdup_printf("data_%u", index);

    GstPad *pad = (GstPad *)g_object_new(GST_TYPE_MSG_AGG_DATA_PAD,
        "name", name, "direction", templ->direction,
        "template", templ, NULL);
    g_free(name);

    if (pad == NULL) {
        GST_ERROR_OBJECT(agg, "Failed to create data pad");
        g_mutex_unlock(&agg->lock);
        return NULL;
    }

    gst_pad_set_chain_function(pad,
        GST_DEBUG_FUNCPTR(gst_msg_agg_data_chain));
    gst_pad_set_event_function(pad,
        GST_DEBUG_FUNCPTR(gst_msg_agg_data_event));

    if (!gst_element_add_pad(element, pad)) {
        GST_ERROR_OBJECT(agg, "Failed to add data pad");
        gst_object_unref(pad);
        g_mutex_unlock(&agg->lock);
        return NULL;
    }

    agg->datapads = g_list_append(agg->datapads, pad);
    agg->nextidx = nextindex;

    g_mutex_unlock(&agg->lock);

    GST_DEBUG_OBJECT(agg, "Created pad: %s (total=%u)",
        GST_PAD_NAME(pad), g_list_length(agg->datapads));
    return pad;
}

static void
gst_msg_agg_release_pad(GstElement *element, GstPad *pad)
{
    GstMsgAgg *agg = GST_MSG_AGG(element);

    GST_DEBUG_OBJECT(agg, "Releasing pad: %s", GST_PAD_NAME(pad));

    g_mutex_lock(&agg->lock);
    agg->datapads = g_list_remove(agg->datapads, pad);
    g_mutex_unlock(&agg->lock);

    gst_element_remove_pad(element, pad);
}

/* ---- data pad chain / event ---- */

static GstFlowReturn
gst_msg_agg_data_chain(GstPad *pad, GstObject *parent, GstBuffer *buffer)
{
    GstMsgAgg *agg = GST_MSG_AGG(parent);
    GstMsgAggDataPad *dpad = GST_MSG_AGG_DATA_PAD(pad);

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        GST_WARNING_OBJECT(pad, "Failed to map buffer");
        gst_buffer_unref(buffer);
        return GST_FLOW_ERROR;
    }

    gchar *json = g_strndup((const gchar *)map.data, map.size);
    gst_buffer_unmap(buffer, &map);
    gst_buffer_unref(buffer);

    g_mutex_lock(&agg->lock);
    g_queue_push_tail(dpad->queue, json);
    g_cond_signal(&agg->wakeup);
    g_mutex_unlock(&agg->lock);

    GST_LOG_OBJECT(pad, "enqueued %u bytes (depth=%u)",
        (guint)map.size, g_queue_get_length(dpad->queue));

    return GST_FLOW_OK;
}

static gboolean
gst_msg_agg_data_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
    GstMsgAgg *agg = GST_MSG_AGG(parent);

    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_STREAM_START: {
        gst_event_unref(event);
        return TRUE;
    }
    case GST_EVENT_EOS: {
        GstMsgAggDataPad *dpad = GST_MSG_AGG_DATA_PAD(pad);
        GST_DEBUG_OBJECT(pad, "received EOS");

        g_mutex_lock(&agg->lock);
        dpad->eos = TRUE;
        g_cond_signal(&agg->wakeup);
        g_mutex_unlock(&agg->lock);

        gst_event_unref(event);
        return TRUE;
    }
    case GST_EVENT_CAPS: {
        gst_event_unref(event);
        return TRUE;
    }
    case GST_EVENT_SEGMENT: {
        gst_event_unref(event);
        return TRUE;
    }
    default:
        return gst_pad_event_default(pad, parent, event);
    }
}

/* ---- worker task (two-phase timeout) ---- */

static void
gst_msg_agg_worker_task(gpointer userdata)
{
    GstMsgAgg *agg = GST_MSG_AGG(userdata);

    g_mutex_lock(&agg->lock);

    /* Phase 1: wait indefinitely until at least one pad has data (prevent spin) */
    while (agg->active && !any_pad_has_data(agg) && !all_pads_eos(agg))
        g_cond_wait(&agg->wakeup, &agg->lock);

    if (!agg->active) {
        g_mutex_unlock(&agg->lock);
        return;
    }

    if (all_pads_eos(agg) && !any_pad_has_data(agg)) {
        g_mutex_unlock(&agg->lock);

        if (!agg->started) {
            gchar *stream_id = g_strdup_printf("qtimsgagg-%s",
                GST_ELEMENT_NAME(agg));
            gst_pad_push_event(agg->srcpad,
                gst_event_new_stream_start(stream_id));
            g_free(stream_id);

            GstCaps *caps = gst_caps_new_simple("text/x-raw",
                "format", G_TYPE_STRING, "utf8", NULL);
            gst_pad_push_event(agg->srcpad, gst_event_new_caps(caps));
            gst_caps_unref(caps);

            GstSegment seg;
            gst_segment_init(&seg, GST_FORMAT_TIME);
            gst_pad_push_event(agg->srcpad, gst_event_new_segment(&seg));

            agg->started = TRUE;
        }

        GST_DEBUG_OBJECT(agg, "all pads EOS, pushing EOS downstream "
            "(batches=%" G_GUINT64_FORMAT " timeouts=%" G_GUINT64_FORMAT ")",
            agg->batches_pushed, agg->timeout_count);
        gst_pad_push_event(agg->srcpad, gst_event_new_eos());
        gst_task_pause(agg->worker);
        return;
    }

    /* Phase 2: wait for remaining pads with timeout */
    if (!all_pads_have_data(agg)) {
        gint64 deadline = g_get_monotonic_time()
                        + (gint64)agg->timeout_ms * G_TIME_SPAN_MILLISECOND;
        while (agg->active && !all_pads_have_data(agg) && !all_pads_eos(agg)) {
            if (!g_cond_wait_until(&agg->wakeup, &agg->lock, deadline))
                break;
        }
        if (!agg->active) {
            g_mutex_unlock(&agg->lock);
            return;
        }
        if (!all_pads_have_data(agg))
            agg->timeout_count++;
    }

    const gboolean use_merge = agg->merge_scene_update;
    gchar *merged = NULL;
    GString *batch = NULL;

    /* Phase 3: pop from each pad; missing pads get {} placeholder (non-merge) */
    if (use_merge) {
        merged = gst_msg_agg_build_scene_merge(agg);
        if (merged == NULL) {
            merged = g_strdup("{\"type\":\"scene_update\",\"sources\":[]}");
        }
    } else {
        batch = g_string_new("{");
        for (GList *l = agg->datapads; l; l = l->next) {
            GstMsgAggDataPad *dpad = GST_MSG_AGG_DATA_PAD(l->data);
            gchar *json = (gchar *)g_queue_pop_head(dpad->queue);
            if (batch->len > 1)
                g_string_append_c(batch, ',');
            if (json) {
                g_string_append(batch, json);
                g_free(json);
            } else {
                agg->placeholder_total++;
                if (dpad->eos)
                    agg->placeholder_eos++;
                else
                    agg->placeholder_timeout++;
                g_string_append(batch, "{}");
            }
        }
        g_string_append_c(batch, '}');
    }
    agg->batches_pushed++;

    g_mutex_unlock(&agg->lock);

    gsize len = 0;
    gchar *str = NULL;
    GstBuffer *buf = NULL;

    if (use_merge) {
        len = strlen(merged);
        buf = gst_buffer_new_and_alloc(len);
        GstMapInfo map;
        if (buf != NULL && gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
            memcpy(map.data, merged, len);
            gst_buffer_unmap(buf, &map);
        } else {
            if (buf != NULL) {
                gst_buffer_unref(buf);
                buf = NULL;
            }
        }
        g_free(merged);
        merged = NULL;
    } else {
        len = batch->len;
        str = g_string_free(batch, FALSE);
        buf = gst_buffer_new_wrapped(str, len);
    }

    if (buf == NULL) {
        g_free(merged);
        if (str != NULL) {
            g_free(str);
        }
        return;
    }

    if (!agg->started) {
        gchar *stream_id = g_strdup_printf("qtimsgagg-%s",
            GST_ELEMENT_NAME(agg));
        gst_pad_push_event(agg->srcpad,
            gst_event_new_stream_start(stream_id));
        g_free(stream_id);

        GstCaps *caps = gst_caps_new_simple("text/x-raw",
            "format", G_TYPE_STRING, "utf8", NULL);
        gst_pad_push_event(agg->srcpad, gst_event_new_caps(caps));
        gst_caps_unref(caps);

        GstSegment seg;
        gst_segment_init(&seg, GST_FORMAT_TIME);
        gst_pad_push_event(agg->srcpad, gst_event_new_segment(&seg));

        agg->started = TRUE;
        GST_DEBUG_OBJECT(agg, "sent stream-start + caps + segment on srcpad");
    }

    GST_LOG_OBJECT(agg, "pushing batch #%" G_GUINT64_FORMAT " (%"
        G_GSIZE_FORMAT " bytes)", agg->batches_pushed, len);

    GstFlowReturn ret = gst_pad_push(agg->srcpad, buf);
    if (ret != GST_FLOW_OK && ret != GST_FLOW_FLUSHING) {
        GST_WARNING_OBJECT(agg, "pad push returned %s",
            gst_flow_get_name(ret));
    }
}

/* ---- state change ---- */

static GstStateChangeReturn
gst_msg_agg_change_state(GstElement *element, GstStateChange transition)
{
    GstMsgAgg *agg = GST_MSG_AGG(element);

    /* --- pre-transition --- */
    switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
        GST_DEBUG_OBJECT(agg, "READY->PAUSED: pads=%u",
            g_list_length(agg->datapads));
        break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
        GST_INFO_OBJECT(agg, "PAUSED->READY: stopping worker");
        g_mutex_lock(&agg->lock);
        agg->active = FALSE;
        g_cond_signal(&agg->wakeup);
        g_mutex_unlock(&agg->lock);
        gst_task_pause(agg->worker);
        break;
    default:
        break;
    }

    GstStateChangeReturn ret =
        GST_ELEMENT_CLASS(gst_msg_agg_parent_class)->change_state(element,
            transition);

    if (ret == GST_STATE_CHANGE_FAILURE)
        return ret;

    /* --- post-transition --- */
    switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
        GST_DEBUG_OBJECT(agg, "starting worker task");
        g_mutex_lock(&agg->lock);
        agg->active = TRUE;
        g_mutex_unlock(&agg->lock);
        gst_task_start(agg->worker);
        break;
    case GST_STATE_CHANGE_PAUSED_TO_READY: {
        guint64 final_batches = 0;
        guint64 final_timeouts = 0;
        guint64 final_placeholders = 0;
        guint64 final_placeholder_eos = 0;
        guint64 final_placeholder_timeout = 0;
        gst_task_join(agg->worker);
        g_mutex_lock(&agg->lock);
        final_batches = agg->batches_pushed;
        final_timeouts = agg->timeout_count;
        final_placeholders = agg->placeholder_total;
        final_placeholder_eos = agg->placeholder_eos;
        final_placeholder_timeout = agg->placeholder_timeout;
        for (GList *l = agg->datapads; l; l = l->next) {
            GstMsgAggDataPad *dpad = GST_MSG_AGG_DATA_PAD(l->data);
            gchar *item;
            while ((item = (gchar *)g_queue_pop_head(dpad->queue)) != NULL)
                g_free(item);
            dpad->eos = FALSE;
        }
        agg->batches_pushed = 0;
        agg->timeout_count = 0;
        agg->placeholder_total = 0;
        agg->placeholder_eos = 0;
        agg->placeholder_timeout = 0;
        agg->started = FALSE;
        g_mutex_unlock(&agg->lock);
        GST_INFO_OBJECT(agg, "final run stats: batches_pushed=%"
            G_GUINT64_FORMAT " timeout_count=%" G_GUINT64_FORMAT
            " placeholder_total=%" G_GUINT64_FORMAT
            " placeholder_eos=%" G_GUINT64_FORMAT
            " placeholder_timeout=%" G_GUINT64_FORMAT,
            final_batches, final_timeouts, final_placeholders,
            final_placeholder_eos, final_placeholder_timeout);
        break;
    }
    default:
        break;
    }

    return ret;
}

/* ---- plugin registration ---- */

static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "qtimsgagg",
        GST_RANK_NONE, GST_TYPE_MSG_AGG);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    qtimsgagg,
    "QTI Message Aggregator (N text streams -> 1 combined JSON)",
    plugin_init,
    "1.0",
    "Proprietary",
    "multi_cam_app",
    "https://www.qualcomm.com"
)
