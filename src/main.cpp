#include <iostream>
#include <thread>
#include <string>
#include <xir/graph/graph.hpp>
#include "DpuBackend.hpp"
#include "FrameQueue.hpp"
#include "Frame.hpp"

static const xir::Subgraph* find_dpu_subgraph(xir::Graph* graph) {
    auto* root = graph->get_root_subgraph();
    for (auto* child : root->children_topological_sort()) {
        if (child->has_attr("device") &&
            child->get_attr<std::string>("device") == "DPU") {
            return child;
        }
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.xmodel>\n";
        return 1;
    }

    auto graph      = xir::Graph::deserialize(argv[1]);
    auto* subgraph  = find_dpu_subgraph(graph.get());
    if (!subgraph) {
        std::cerr << "no DPU subgraph in model\n";
        return 1;
    }

    DpuBackend backend(subgraph);
    FrameQueue  queue;

    std::thread producer([&] {
        for (int i = 0; i < 3; ++i) {
            queue.push(Frame(i, 1228800));
            std::cout << "pushed frame " << i << '\n';
        }
        queue.set_done();
    });

    std::thread consumer([&] {
        while (auto f = queue.pop()) {
            auto detections = backend.run(std::span<const float>(
                reinterpret_cast<const float*>(f->data()),
                f->bytes() / sizeof(float)
            ));
            std::cout << "frame " << f->get_id()
                      << ": " << detections.size() << " detections\n";
        }
    });

    producer.join();
    consumer.join();
    return 0;
}
