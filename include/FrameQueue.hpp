#pragma once
#include "Frame.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

class FrameQueue {
    std::queue<Frame>       frames_;
    std::mutex              mtx_;
    std::condition_variable cv_;
    bool                    done_ = false;
public:
    void push(Frame&& f) {
        {
            std::lock_guard<std::mutex> g(mtx_);
            frames_.push(std::move(f));
        }
        cv_.notify_one();
    }

    std::optional<Frame> pop() {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [this] { return !frames_.empty() || done_; });
        if (frames_.empty() && done_) return std::nullopt;
        Frame f = std::move(frames_.front());
        frames_.pop();
        return f;
    }

    void set_done() {
        {
            std::lock_guard<std::mutex> g(mtx_);
            done_ = true;
        }
        cv_.notify_all();
    }
};
