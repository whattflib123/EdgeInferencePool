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
        std::cerr << "usage: " << argv[0] << " <model.xmodel> <video.mp4|0>\n";
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
        cv::VideoCapture cap;
        std::string src(argv[2]);
        if (src == "0") cap.open(0);
        else            cap.open(src);
        if (!cap.isOpened()) {
            std::cerr << "failed to open: " << argv[2] << "\n";
            queue.set_done();
            return;
        }
        const cv::Mat mean_mat(224, 224, CV_32FC3, cv::Scalar(104.0f, 117.0f, 123.0f));
        const size_t nbytes = 224 * 224 * 3 * sizeof(float);
        int frame_id = 0;
        cv::Mat img;
        while (cap.read(img)) {
            cv::resize(img, img, cv::Size(224, 224));
            img.convertTo(img, CV_32F);
            img = (img - mean_mat) * (1.0f / 255.0f);
            Frame f(frame_id++, nbytes);
            std::memcpy(f.data(), img.data, nbytes);
            queue.push(std::move(f));
        }
        queue.set_done();
    });

    std::thread consumer([&] {
        while (auto f = queue.pop()) {
            auto detections = backend.run(std::span<const float>(
                reinterpret_cast<const float*>(f->data()),
                f->bytes() / sizeof(float)
            ));
            if (!detections.empty()) {
                std::cout << "frame " << f->get_id()
                          << " top1: class=" << detections[0].class_id
                          << " conf=" << detections[0].confidence << "\n";
            }

        }
    });

    producer.join();
    consumer.join();
    return 0;
}
