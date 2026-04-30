#ifndef GST_MSG_AGG_H
#define GST_MSG_AGG_H

#include <gst/gst.h>

G_BEGIN_DECLS

/* --- GstMsgAggDataPad (custom pad GType, simplified from GstMetaMuxDataPad) --- */

#define GST_TYPE_MSG_AGG_DATA_PAD            (gst_msg_agg_data_pad_get_type())
#define GST_MSG_AGG_DATA_PAD(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_MSG_AGG_DATA_PAD, GstMsgAggDataPad))
#define GST_MSG_AGG_DATA_PAD_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_MSG_AGG_DATA_PAD, GstMsgAggDataPadClass))
#define GST_IS_MSG_AGG_DATA_PAD(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_MSG_AGG_DATA_PAD))
#define GST_IS_MSG_AGG_DATA_PAD_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_MSG_AGG_DATA_PAD))

typedef struct _GstMsgAggDataPad GstMsgAggDataPad;
typedef struct _GstMsgAggDataPadClass GstMsgAggDataPadClass;

struct _GstMsgAggDataPad {
    GstPad    parent;
    GQueue   *queue;
    gboolean  eos;
};

struct _GstMsgAggDataPadClass {
    GstPadClass parent;
};

GType gst_msg_agg_data_pad_get_type(void);

/* --- GstMsgAgg (main element) --- */

#define GST_TYPE_MSG_AGG            (gst_msg_agg_get_type())
#define GST_MSG_AGG(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_MSG_AGG, GstMsgAgg))
#define GST_MSG_AGG_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_MSG_AGG, GstMsgAggClass))
#define GST_IS_MSG_AGG(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_MSG_AGG))
#define GST_IS_MSG_AGG_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_MSG_AGG))

typedef struct _GstMsgAgg GstMsgAgg;
typedef struct _GstMsgAggClass GstMsgAggClass;

struct _GstMsgAgg {
    GstElement   parent;

    GMutex       lock;
    GCond        wakeup;

    GList       *datapads;
    guint        nextidx;
    GstPad      *srcpad;

    GstTask     *worker;
    GRecMutex    worklock;
    gboolean     active;
    gboolean     started;

    guint        timeout_ms;
    gboolean     merge_scene_update;
    guint64      batches_pushed;
    guint64      timeout_count;
    guint64      placeholder_total;
    guint64      placeholder_eos;
    guint64      placeholder_timeout;
};

struct _GstMsgAggClass {
    GstElementClass parent_class;
};

GType gst_msg_agg_get_type(void);

G_END_DECLS

#endif /* GST_MSG_AGG_H */
