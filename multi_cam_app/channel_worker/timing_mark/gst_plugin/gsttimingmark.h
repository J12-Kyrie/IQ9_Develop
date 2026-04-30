#ifndef GST_TIMING_MARK_H
#define GST_TIMING_MARK_H

#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_TIMING_MARK            (gst_timing_mark_get_type())
#define GST_TIMING_MARK(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_TIMING_MARK, GstTimingMark))
#define GST_TIMING_MARK_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_TIMING_MARK, GstTimingMarkClass))
#define GST_IS_TIMING_MARK(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_TIMING_MARK))
#define GST_IS_TIMING_MARK_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_TIMING_MARK))

typedef struct _GstTimingMark GstTimingMark;
typedef struct _GstTimingMarkClass GstTimingMarkClass;
typedef struct _GstTimingMarkPrivate GstTimingMarkPrivate;

struct _GstTimingMark {
  GstBaseTransform parent;
  GstTimingMarkPrivate* priv;
};

struct _GstTimingMarkClass {
  GstBaseTransformClass parent_class;
};

GType gst_timing_mark_get_type(void);

G_END_DECLS

#endif  // GST_TIMING_MARK_H
