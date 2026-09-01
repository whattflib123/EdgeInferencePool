#pragma once
#include "InferenceBackend.hpp"
#include <memory>
#include <vart/runner_ext.hpp>
#include <xir/attrs/attrs.hpp>

namespace xir { class Subgraph; }

class DpuBackend : public InferenceBackend {
    std::unique_ptr<xir::Attrs>      attrs_;
    std::unique_ptr<vart::RunnerExt> runner_;
public:
    explicit DpuBackend(const xir::Subgraph* subgraph);
    std::vector<Detection> run(std::span<const float> input) override;
};
