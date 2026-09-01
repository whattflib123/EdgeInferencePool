#pragma once
#include "InferenceBackend.hpp"
#include <memory>

namespace xir { class Subgraph; }
namespace vart { class Runner; }

class DpuBackend : public InferenceBackend {
    std::unique_ptr<vart::Runner> runner_;
public:
    explicit DpuBackend(const xir::Subgraph* subgraph);
    std::vector<Detection> run(std::span<const float> input) override;
};
