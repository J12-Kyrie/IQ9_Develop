#ifndef MULTI_CAM_APP_UTILS_PERF_TIMING_MARK_RUNTIME_HPP
#define MULTI_CAM_APP_UTILS_PERF_TIMING_MARK_RUNTIME_HPP

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "utils/perf/LatencyRecorder.hpp"

namespace multi_cam_app::perf {

class TimingMarkRuntime {
public:
  static TimingMarkRuntime& Instance();

  bool RegisterChannel(const LatencyRecorder::Config& config,
                       std::string* out_error);
  void UnregisterChannel(uint32_t channel_id);
  void Submit(uint32_t channel_id,
              LatencyStage stage,
              uint64_t frame_key,
              uint64_t timestamp_ns);
  void Flush();

private:
  struct StageEvent {
    uint32_t channel_id {0U};
    LatencyStage stage {LatencyStage::kH264ParseSrc};
    uint64_t frame_key {0ULL};
    uint64_t timestamp_ns {0ULL};
  };

  struct ChannelState {
    LatencyRecorder::Config config {};
    std::shared_ptr<LatencyRecorder> recorder {};
    uint32_t ref_count {0U};
  };

  TimingMarkRuntime() = default;
  ~TimingMarkRuntime();

  TimingMarkRuntime(const TimingMarkRuntime&) = delete;
  TimingMarkRuntime& operator=(const TimingMarkRuntime&) = delete;

  void EnsureWorkerLocked();
  void WaitForChannelIdleLocked(std::unique_lock<std::mutex>* lock,
                                uint32_t channel_id);
  void WorkerLoop();

  std::mutex mutex_ {};
  std::condition_variable cv_ {};
  std::condition_variable idle_cv_ {};
  std::unordered_map<uint32_t, ChannelState> channels_ {};
  std::unordered_map<uint32_t, uint64_t> in_flight_counts_ {};
  std::deque<StageEvent> events_ {};
  std::thread worker_ {};
  bool stop_requested_ {false};
  uint32_t active_workers_ {0U};
};

}  // namespace multi_cam_app::perf

#endif  // MULTI_CAM_APP_UTILS_PERF_TIMING_MARK_RUNTIME_HPP
