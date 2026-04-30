#ifndef MULTI_CAM_APP_DEDUPER_HPP
#define MULTI_CAM_APP_DEDUPER_HPP

#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace deduper {

/// 白名单：仅保留这些 COCO 类别的检测结果
/// 参考: multi_cam_app/data/models/yolov11/labels.txt
static const std::unordered_set<std::string> kAllowedClasses = {
    "person", "bicycle", "car", "backpack", "umbrella", "handbag"
};

/// 轻量级检测条目，用于去重计算
/// 由 GStreamer 插件从 ROI meta 构造，或由应用层从 DetectionRecord 构造
struct TrackEntry {
    std::string label;
    uint32_t    track_id {0U};
    double      left   {0.0};   // 归一化 [0,1]
    double      top    {0.0};
    double      right  {0.0};
    double      bottom {0.0};
};

struct DeduperConfig {
    float iou_threshold {0.75f};
};

class Deduper {
public:
    explicit Deduper(const DeduperConfig& config = {});
    ~Deduper() = default;

    /// 判断当前帧是否"有变化"（应输出到 JSONL / 推送到下游）
    ///
    /// 三条规则（复刻 Orin DwDeduper）:
    ///   1. 新目标出现（track_key 在上一帧不存在）-> interesting
    ///   2. 目标显著移动（同 track_key 的 IoU < threshold）-> interesting
    ///   3. 目标消失（上一帧的 track_key 在当前帧不存在）-> interesting
    ///
    /// @param entries 当前帧的白名单过滤后的检测条目（调用方需先过滤）
    /// @return true = 有变化（通过），false = 无变化（丢弃）
    bool IsInteresting(const std::vector<TrackEntry>& entries);

    /// 重置内部状态。在 EOS / pipeline stop / seek 时调用。
    void Reset();

    /// 检查标签是否在白名单中
    static bool IsAllowed(const std::string& label);

private:
    using TrackKey = std::string;  // "label#track_id"

    struct BoxSnapshot {
        double left   {0.0};
        double top    {0.0};
        double right  {0.0};
        double bottom {0.0};
    };

    static TrackKey MakeKey(const TrackEntry& entry);
    static double CalculateIoU(const BoxSnapshot& a, const BoxSnapshot& b);

    DeduperConfig config_;
    std::map<TrackKey, BoxSnapshot> last_frame_objects_;
};

}  // namespace deduper

#endif  // MULTI_CAM_APP_DEDUPER_HPP
