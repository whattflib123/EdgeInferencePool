#pragma once
#include <memory>

class Frame {
    std::unique_ptr<char[]> buf_;
    int id_;
    int size_;
public:
    Frame(int id, int size)
        : buf_(std::make_unique<char[]>(size)), id_(id), size_(size) {}

    Frame(const Frame&)            = delete;
    Frame& operator=(const Frame&) = delete;
    Frame(Frame&&)                 = default;
    Frame& operator=(Frame&&)      = default;

    int   get_id() const { return id_; }
    char* data()   const { return buf_.get(); }
    int   bytes()  const { return size_; }
};
