#pragma once
#include <span>
#include <vector>

struct Detection {
    float x1, y1, x2, y2, confidence;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual std::vector<Detection> run(std::span<const float> input) = 0;
};
