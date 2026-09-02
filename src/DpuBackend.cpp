#include "DpuBackend.hpp"
#include <vart/runner_ext.hpp>
#include <xir/graph/subgraph.hpp>
#include <xir/attrs/attrs.hpp>
#include <iostream>
#include <numeric>    // std::iota
#include <algorithm>  // std::partial_sort



DpuBackend::DpuBackend(const xir::Subgraph* subgraph) {
    attrs_  = xir::Attrs::create();
    runner_ = vart::RunnerExt::create_runner(subgraph, attrs_.get());
    std::cout << "[DpuBackend] runner created\n";

    for (auto* tb : runner_->get_inputs()) {
        auto shape = tb->get_tensor()->get_shape();
        std::cout << "[IN ] " << tb->get_tensor()->get_name() << " shape:";
        for (int d : shape) std::cout << " " << d;
        std::cout << "\n";
    }
    for (auto* tb : runner_->get_outputs()) {
        auto shape = tb->get_tensor()->get_shape();
        std::cout << "[OUT] " << tb->get_tensor()->get_name() << " shape:";
        for (int d : shape) std::cout << " " << d;
        std::cout << "\n";
    }
}

std::vector<Detection> DpuBackend::run(std::span<const float> input) {
    auto inputs  = runner_->get_inputs();
    auto outputs = runner_->get_outputs();

    // TODO: copy `input` data into inputs[0]'s tensor buffer
    //   hint: inputs[0]->data() returns {void*, size_t}
    //   use memcpy(ptr, input.data(), input.size_bytes())

    // input[i] (float) × 128 → clamp [-128, 127] → int8_t → 寫進 ptr[i]
    // Step1: 取得 ptr
    auto [ptr, size] = inputs[0]->data(std::vector<int>{0, 0, 0, 0});
    int8_t* dst = reinterpret_cast<int8_t*>(ptr);

    // step2: 
    int fp = inputs[0]->get_tensor()->template get_attr<int32_t>("fix_point");
    std::cout << "[DEBUG] input fix_point=" << fp << " scale=" << (1 << fp) << "\n";
    float scale = static_cast<float>(1 << fp);  // = 128.0f

    // Step 3：逐元素 float → int8
    for (size_t i = 0; i < input.size(); ++i) {
        float val = input[i] * scale;           // float × 128
        // clamp 到 int8 範圍
        if (val >  127.0f) val =  127.0f;
        if (val < -128.0f) val = -128.0f;
        dst[i] = static_cast<int8_t>(val);      // 寫進 tensor buffer
    }


    auto job = runner_->execute_async(inputs, outputs);
    runner_->wait(job.first, -1);

    // TODO: read outputs[0]'s tensor buffer → parse into Detection objects
    //   hint: outputs[0]->data() returns {void*, size_t}
    //   shape depends on your model's output layer

    //Step 1：取 output ptr

    auto [out_ptr, out_size] = outputs[0]->data(std::vector<int>{0, 0});
    int8_t* src = reinterpret_cast<int8_t*>(out_ptr);

    //Step 2：取 output fix_point

    int ofp = outputs[0]->get_tensor()->template get_attr<int32_t>("fix_point");
    float oscale = static_cast<float>(1 << ofp);

    //Step 3：找 top-5
    std::vector<int> idx(1000);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+5, idx.end(), [&](int a, int b){ return src[a] > src[b]; });



    // ---- 包成 Detection ----
    std::vector<Detection> results;
    for (int i = 0; i < 5; ++i) {
        Detection d;
        d.class_id   = idx[i];
        d.confidence = static_cast<float>(src[idx[i]]) / oscale;
        d.x1 = d.y1 = d.x2 = d.y2 = 0.0f;
        results.push_back(d);
    }
    return results;


}
