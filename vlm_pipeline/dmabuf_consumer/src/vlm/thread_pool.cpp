#include "vlm/thread_pool.h"

namespace vlm {

ThreadPool::ThreadPool(size_t num_threads, size_t max_queue_size)
    : max_queue_size_(max_queue_size) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                    if (stop_ && queue_.empty()) return;
                    task = std::move(queue_.front());
                    queue_.pop();
                    ++active_;
                }
                task();
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    --active_;
                    ++completed_;
                }
                drain_cv_.notify_all();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

bool ThreadPool::submit(std::function<void()> task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_) return false;
    if (queue_.size() >= max_queue_size_) return false;
    queue_.push(std::move(task));
    cv_.notify_one();
    return true;
}

void ThreadPool::drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    drain_cv_.wait(lock, [this] {
        return queue_.empty() && active_ == 0;
    });
}

bool ThreadPool::evict_oldest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    queue_.pop();
    return true;
}

size_t ThreadPool::pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

size_t ThreadPool::completed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
}

} // namespace vlm
