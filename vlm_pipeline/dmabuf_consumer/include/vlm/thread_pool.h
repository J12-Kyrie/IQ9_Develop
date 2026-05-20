#pragma once
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace vlm {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads, size_t max_queue_size = 16);
    ~ThreadPool();

    bool submit(std::function<void()> task);
    void drain();
    bool evict_oldest();

    size_t pending() const;
    size_t completed() const;

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> queue_;
    size_t max_queue_size_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable drain_cv_;
    bool stop_ = false;
    size_t completed_ = 0;
    size_t active_ = 0;
};

} // namespace vlm
