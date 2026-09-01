#include "DpuBackend.hpp"
#include <vart/runner.hpp>
#include <xir/graph/subgraph.hpp>
#include <iostream>

DpuBackend::DpuBackend(const xir::Subgraph* subgraph) {
    runner_ = vart::Runner::create_runner(subgraph, "run");
    std::cout << "[DpuBackend] runner created\n";
}

std::vector<Detection> DpuBackend::run(std::span<const float> input) {
    auto inputs  = runner_->get_inputs();
    auto outputs = runner_->get_outputs();

    // TODO: copy `input` data into inputs[0]'s tensor buffer
    //   hint: inputs[0]->data() returns {void*, size_t}
    //   use memcpy(ptr, input.data(), input.size_bytes())

    auto job = runner_->execute_async(inputs, outputs);
    runner_->wait(job.first, -1);

    // TODO: read outputs[0]'s tensor buffer → parse into Detection objects
    //   hint: outputs[0]->data() returns {void*, size_t}
    //   shape depends on your model's output layer

    return {};
}
