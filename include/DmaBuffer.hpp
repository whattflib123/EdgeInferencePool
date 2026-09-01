#pragma once
#include <cstdlib>
#include <cstddef>
#include <iostream>

class DmaBuffer {
    void*  ptr_;
    size_t size_;
public:
    explicit DmaBuffer(size_t n)
        : ptr_(malloc(n)), size_(n) {
        std::cout << "[DmaBuffer] alloc " << n << " bytes\n";
    }

    ~DmaBuffer() {
        std::cout << "[DmaBuffer] free\n";
        free(ptr_);
    }

    DmaBuffer(const DmaBuffer&)            = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;

    DmaBuffer(DmaBuffer&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_  = nullptr;
        o.size_ = 0;
        std::cout << "[DmaBuffer] move\n";
    }

    DmaBuffer& operator=(DmaBuffer&& o) noexcept {
        if (this != &o) {
            free(ptr_);
            ptr_  = o.ptr_;  size_  = o.size_;
            o.ptr_ = nullptr; o.size_ = 0;
        }
        return *this;
    }

    void*  data()  const { return ptr_; }
    size_t bytes() const { return size_; }
};
