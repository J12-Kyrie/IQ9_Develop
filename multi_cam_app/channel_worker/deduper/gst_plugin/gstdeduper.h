/// @file gstdeduper.h
/// @brief GstDeduper: GstBaseTransform in-place plugin for detection deduplication
///
/// Pipeline usage:
///   ... ! video/x-raw,format=NV12 ! qtideduper iou-threshold=0.75 ! ...
///
/// Input:  NV12 frames + GstVideoRegionOfInterestMeta (tracker results)
///                      + GstFaceDetectionMeta (optional, passthrough)
/// Output: filtered NV12 frames (uninteresting frames dropped)

#ifndef GST_DEDUPER_H
#define GST_DEDUPER_H

#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_DEDUPER            (gst_deduper_get_type())
#define GST_DEDUPER(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_DEDUPER, GstDeduper))
#define GST_DEDUPER_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_DEDUPER, GstDeduperClass))
#define GST_IS_DEDUPER(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_DEDUPER))
#define GST_IS_DEDUPER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_DEDUPER))

typedef struct _GstDeduper GstDeduper;
typedef struct _GstDeduperClass GstDeduperClass;
typedef struct _GstDeduperPrivate GstDeduperPrivate;

struct _GstDeduper {
    GstBaseTransform parent;
    GstDeduperPrivate *priv;
};

struct _GstDeduperClass {
    GstBaseTransformClass parent_class;
};

GType gst_deduper_get_type (void);

G_END_DECLS

#endif /* GST_DEDUPER_H */
