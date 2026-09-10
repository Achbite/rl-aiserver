#pragma once
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

class PeriodicMetricFlush {
public:
    ~PeriodicMetricFlush() { Stop(); }
    void Start(std::chrono::milliseconds period, std::function<bool()> flush) {
        worker_ = std::thread([this, period, flush = std::move(flush)] {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!changed_.wait_for(lock, period, [this] { return stopped_; })) {
                lock.unlock();
                const bool accepted = flush();
                lock.lock();
                if (!accepted) break;
            }
        });
    }
    void Stop() {
        { std::lock_guard<std::mutex> lock(mutex_); stopped_ = true; }
        changed_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread worker_;
    bool stopped_ = false;
};
