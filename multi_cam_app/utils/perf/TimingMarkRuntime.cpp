#include "utils/perf/TimingMarkRuntime.hpp"

#include <utility>

namespace multi_cam_app::perf {

TimingMarkRuntime& TimingMarkRuntime::Instance() {
  static TimingMarkRuntime runtime;
  return runtime;
}

TimingMarkRuntime::~TimingMarkRuntime() {
  Flush();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_requested_ = true;
  }
  cv_.notify_all();

  if (worker_.joinable()) {
    worker_.join();
  }
}

bool TimingMarkRuntime::RegisterChannel(const LatencyRecorder::Config& config,
                                        std::string* out_error) {
  if (config.output_dir.empty()) {
    if (out_error != nullptr) {
      *out_error = "TimingMarkRuntime output_dir is empty";
    }
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  EnsureWorkerLocked();

  auto it = channels_.find(config.channel_id);
  if (it != channels_.end()) {
    ++it->second.ref_count;
    return true;
  }

  auto recorder = std::make_shared<LatencyRecorder>();
  std::string rec_error;
  if (!recorder->Initialize(config, &rec_error)) {
    if (out_error != nullptr) {
      *out_error = rec_error;
    }
    return false;
  }

  ChannelState state {};
  state.config = config;
  state.recorder = std::move(recorder);
  state.ref_count = 1U;
  channels_.emplace(config.channel_id, std::move(state));
  return true;
}

void TimingMarkRuntime::UnregisterChannel(uint32_t channel_id) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = channels_.find(channel_id);
    if (it == channels_.end()) {
      return;
    }
    if (it->second.ref_count > 1U) {
      --it->second.ref_count;
      return;
    }
  }

  {
    std::unique_lock<std::mutex> lock(mutex_);
    WaitForChannelIdleLocked(&lock, channel_id);
  }

  std::shared_ptr<LatencyRecorder> recorder;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = channels_.find(channel_id);
    if (it == channels_.end()) {
      return;
    }
    recorder = std::move(it->second.recorder);
    channels_.erase(it);
  }

  if (recorder) {
    recorder->Close();
  }
}

void TimingMarkRuntime::Submit(uint32_t channel_id,
                               LatencyStage stage,
                               uint64_t frame_key,
                               uint64_t timestamp_ns) {
  if (timestamp_ns == 0ULL) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (channels_.find(channel_id) == channels_.end()) {
    return;
  }

  ++in_flight_counts_[channel_id];
  events_.push_back(StageEvent{channel_id, stage, frame_key, timestamp_ns});
  cv_.notify_one();
}

void TimingMarkRuntime::Flush() {
  std::unique_lock<std::mutex> lock(mutex_);
  idle_cv_.wait(lock, [this]() {
    return events_.empty() && (active_workers_ == 0U);
  });
}

void TimingMarkRuntime::EnsureWorkerLocked() {
  if (worker_.joinable()) {
    return;
  }
  stop_requested_ = false;
  worker_ = std::thread(&TimingMarkRuntime::WorkerLoop, this);
}

void TimingMarkRuntime::WaitForChannelIdleLocked(std::unique_lock<std::mutex>* lock,
                                                 uint32_t channel_id) {
  if (lock == nullptr) {
    return;
  }
  idle_cv_.wait(*lock, [this, channel_id]() {
    const auto it = in_flight_counts_.find(channel_id);
    return (it == in_flight_counts_.end()) || (it->second == 0ULL);
  });
}

void TimingMarkRuntime::WorkerLoop() {
  while (true) {
    StageEvent event {};
    std::shared_ptr<LatencyRecorder> recorder;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() {
        return stop_requested_ || !events_.empty();
      });

      if (stop_requested_ && events_.empty()) {
        return;
      }

      event = events_.front();
      events_.pop_front();

      auto it = channels_.find(event.channel_id);
      if (it != channels_.end()) {
        recorder = it->second.recorder;
      }
      ++active_workers_;
    }

    if (recorder) {
      recorder->RecordStageTimestamp(
          event.stage, event.frame_key, event.timestamp_ns);
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      bool notify_idle = false;
      auto in_flight_it = in_flight_counts_.find(event.channel_id);
      if (in_flight_it != in_flight_counts_.end()) {
        if (in_flight_it->second > 0ULL) {
          --in_flight_it->second;
        }
        if (in_flight_it->second == 0ULL) {
          in_flight_counts_.erase(in_flight_it);
          notify_idle = true;
        }
      }
      if (active_workers_ > 0U) {
        --active_workers_;
      }
      if (events_.empty() && (active_workers_ == 0U)) {
        notify_idle = true;
      }
      if (notify_idle) {
        idle_cv_.notify_all();
      }
    }
  }
}

}  // namespace multi_cam_app::perf
