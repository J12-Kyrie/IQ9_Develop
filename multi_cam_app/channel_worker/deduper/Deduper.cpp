#include "Deduper.hpp"
#include <algorithm>

namespace deduper {

Deduper::Deduper(const DeduperConfig& config)
    : config_(config) {}

void Deduper::Reset() {
    last_frame_objects_.clear();
}

bool Deduper::IsAllowed(const std::string& label) {
    return kAllowedClasses.count(label) > 0;
}

Deduper::TrackKey Deduper::MakeKey(const TrackEntry& entry) {
    const std::string& name = entry.label.empty() ? "unknown" : entry.label;
    return name + "#" + std::to_string(entry.track_id);
}

double Deduper::CalculateIoU(const BoxSnapshot& a, const BoxSnapshot& b) {
    // Orin 原版使用 {x, y, width, height}，这里适配为 {left, top, right, bottom}
    const double x1 = std::max(a.left,  b.left);
    const double y1 = std::max(a.top,   b.top);
    const double x2 = std::min(a.right, b.right);
    const double y2 = std::min(a.bottom, b.bottom);

    const double w = std::max(0.0, x2 - x1);
    const double h = std::max(0.0, y2 - y1);
    const double overlap = w * h;

    const double area_a = (a.right - a.left) * (a.bottom - a.top);
    const double area_b = (b.right - b.left) * (b.bottom - b.top);
    const double union_area = area_a + area_b - overlap;

    if (union_area <= 0.0) return 0.0;
    return overlap / union_area;
}

bool Deduper::IsInteresting(const std::vector<TrackEntry>& entries) {
    std::map<TrackKey, BoxSnapshot> current_map;
    bool has_interesting = false;

    for (const auto& entry : entries) {
        if (entry.track_id == 0U) continue;  // 无 tracking-id，跳过

        const TrackKey key = MakeKey(entry);
        BoxSnapshot snap;
        snap.left   = entry.left;
        snap.top    = entry.top;
        snap.right  = entry.right;
        snap.bottom = entry.bottom;
        current_map[key] = snap;

        // 规则 1: 新目标（上一帧没有此 key）
        const auto it = last_frame_objects_.find(key);
        if (it == last_frame_objects_.end()) {
            has_interesting = true;
        } else {
            // 规则 2: 目标显著移动（IoU < threshold）
            const double iou = CalculateIoU(snap, it->second);
            if (iou < static_cast<double>(config_.iou_threshold)) {
                has_interesting = true;
            }
        }
    }

    // 规则 3: 目标消失（上一帧有但当前帧没有的 key）
    if (!has_interesting) {
        for (const auto& [key, box] : last_frame_objects_) {
            (void)box;
            if (current_map.find(key) == current_map.end()) {
                has_interesting = true;
                break;
            }
        }
    }

    // 仅在 interesting 时更新状态（与 Orin 原版一致）
    if (has_interesting) {
        last_frame_objects_ = std::move(current_map);
    }

    return has_interesting;
}

}  // namespace deduper
