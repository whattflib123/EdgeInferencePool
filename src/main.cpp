#include <iostream>
#include <thread>
#include <string>
#include <xir/graph/graph.hpp>
#include "DpuBackend.hpp"
#include "FrameQueue.hpp"
#include "Frame.hpp"
#include <opencv2/opencv.hpp>
#include <cstring>




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
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.xmodel> <image.jpg>\n";
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
        cv::Mat img = cv::imread(argv[2]);          // BGR，保持不轉
        cv::resize(img, img, cv::Size(224, 224));
        img.convertTo(img, CV_32F);                 // uint8 → float
        cv::Mat mean(224, 224, CV_32FC3, cv::Scalar(104.0f, 117.0f, 123.0f));
        img = (img - mean) * (1.0f / 255.0f);      // Caffe BGR mean subtract

        size_t nbytes = 224 * 224 * 3 * sizeof(float);
        Frame f(0, nbytes);
        std::memcpy(f.data(), img.data, nbytes);
        std::cout << "pushed frame 0 (" << argv[2] << ")\n";
        queue.push(std::move(f));
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
            for (const auto& d : detections) {
                std::cout << "  class=" << d.class_id << " conf=" << d.confidence << "\n";
            }

        }
    });

    producer.join();
    consumer.join();
    return 0;
}
