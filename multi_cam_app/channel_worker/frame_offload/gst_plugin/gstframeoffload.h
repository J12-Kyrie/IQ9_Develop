#ifndef GST_FRAME_OFFLOAD_H
#define GST_FRAME_OFFLOAD_H

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_FRAME_OFFLOAD            (gst_frame_offload_get_type())
#define GST_FRAME_OFFLOAD(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_FRAME_OFFLOAD, GstFrameOffload))
#define GST_FRAME_OFFLOAD_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_FRAME_OFFLOAD, GstFrameOffloadClass))
#define GST_IS_FRAME_OFFLOAD(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_FRAME_OFFLOAD))
#define GST_IS_FRAME_OFFLOAD_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_FRAME_OFFLOAD))

typedef struct _GstFrameOffload GstFrameOffload;
typedef struct _GstFrameOffloadClass GstFrameOffloadClass;
typedef struct _GstFrameOffloadPrivate GstFrameOffloadPrivate;

struct _GstFrameOffload {
    GstElement parent;
    GstPad *sinkpad;
    GstPad *srcpad;
    GstFrameOffloadPrivate *priv;
};

struct _GstFrameOffloadClass {
    GstElementClass parent_class;
};

GType gst_frame_offload_get_type (void);

G_END_DECLS

#endif /* GST_FRAME_OFFLOAD_H */
