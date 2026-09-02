# EdgeInferencePool

## 這個 repo 是什麼

Felix 的 M1 DpuBackend 專案。把 VART API 接進 `InferenceBackend` 抽象介面，
跑在 KV260（AMD Kria Vision AI Starter Kit）+ Vitis-AI 環境上。

前身是 `/home/felix/Desktop/my_stuff/C` 的 stage3_integration/phase10_mini_pipeline，
那裡的 DmaBuffer / Frame / FrameQueue 已搬過來。

## 使用者背景

- Felix，SRAM AMR perception engineer，求職定位：**HW-aware ML Systems Engineer**（主軸 deployment/accelerator/FPGA，副軸 compiler/MLIR）
- C++ Stage 0~13 已完成（pointer、RAII、Rule of Five、span、vector、多執行緒、多型、template）
- 這個 repo 是把觀念搬進真實硬體專案的實戰期
- 求職敘事核心：「能指著真實程式碼解釋 RAII / move / span / virtual 怎麼串在一起」
- MLIR/LLVM 深投入**刻意排在 M1 完成後**才評估，不搶現在進度

## 環境

- **開發機**：Linux（WSL2 Ubuntu 24.04，g++ 13）
- **目標板**：KV260，IP `10.42.0.199`，SSH `root@10.42.0.199`（無密碼）
- **板子規格**：ARM64 Cortex-A53，g++ 11.2.0，cmake 3.21.1
- **VART headers**：`/usr/include/vart/`（`runner.hpp`, `runner_ext.hpp`, `tensor_buffer.hpp` 等）
- **XIR headers**：`/usr/include/xir/`
- **工作流程**：WSL2 編輯 → `rsync -av ./ root@10.42.0.199:~/EdgeInferencePool/` → KV260 上 cmake/make

### 快速 rsync + build

```bash
# WSL2 上
rsync -av /home/felix/Desktop/my_stuff/EdgeInferencePool/ root@10.42.0.199:~/EdgeInferencePool/

# KV260 上（SSH 進去）
cd ~/EdgeInferencePool/build && make 2>&1
```

## 目前狀態（2026-09-02）

**編譯：✅ KV260 上 `make` 通過**

### 已完成
- `InferenceBackend` pure virtual 介面（`Detection` struct + `run(span<const float>)`）
- `DpuBackend : InferenceBackend`，持有 `vart::RunnerExt` + `xir::Attrs`
- `DpuBackend::DpuBackend(subgraph)` — runner 建好（`RunnerExt::create_runner`）
- `main.cpp` — xmodel 載入、DPU subgraph 搜尋、producer-consumer pipeline

### 卡住點（TODO）

`src/DpuBackend.cpp` 裡兩個 TODO：

```cpp
// TODO 1: copy input span into inputs[0] tensor buffer
// hint: auto [ptr, sz] = inputs[0]->data({0,0,0,0});
//        memcpy(ptr, input.data(), input.size_bytes());

// TODO 2: parse outputs[0] tensor buffer → vector<Detection>
// hint: outputs[0]->data({0,0,0,0}) 取 ptr，shape 依模型輸出層
```

**需要 `.xmodel` 才能繼續**——input/output tensor shape 要對上模型。

### 下一步
1. KV260 上 `find / -name "*.xmodel"` 找現成模型
2. 確認 tensor shape（ctor 印 `get_inputs()[0]->get_tensor()->get_shape()`）
3. 填 TODO 1（memcpy）
4. 填 TODO 2（output parse → Detection）
5. 中期：考慮把碩論 ResNet-SpatialMixConv 量化出 xmodel，一次解決「有沒有 xmodel」跟「量化實驗數據」兩件事

## 關鍵 VART API 筆記

```cpp
// RunnerExt（非 Runner）才有 get_inputs / get_outputs
#include <vart/runner_ext.hpp>
#include <xir/attrs/attrs.hpp>

auto attrs  = xir::Attrs::create();
auto runner = vart::RunnerExt::create_runner(subgraph, attrs.get());

auto inputs  = runner->get_inputs();   // vector<TensorBuffer*>
auto outputs = runner->get_outputs();  // vector<TensorBuffer*>

// inputs[0]->data() returns std::pair<void*, size_t>
auto [ptr, size] = inputs[0]->data(std::vector<int>{0, 0, 0, 0});

auto job = runner->execute_async(inputs, outputs);
runner->wait(job.first, -1);
```

## 專案路線圖

| 里程碑 | 內容 | 狀態 |
|---|---|---|
| M1 DpuBackend | VART API 接進 InferenceBackend | 🔄 進行中（TODO 未填）|
| M2 | MobileNetV2 + Vitis-AI 量化，最小物件偵測 | 未開始 |
| M3 | 機械手臂模擬 pipeline（UR5e + MoveIt2 + Gazebo） | 未開始 |

MLIR/LLVM/自訂 NPU dialect **不列入此路線圖**，排在 M1 完成、求職面試有回饋後再評估是否開新 Mx。

## 並行支線（不佔 M1 進度，不需 KV260）

- **ONNX Graph Analyzer**：解析 `.onnx`，印出每層 shape、FLOPs、參數量
- **手刻 INT8 量化**：FP32 → INT8 → FP32，拿 ResNet-SpatialMixConv 跑出 FP32/PTQ/QAT 三組 accuracy/model size/latency 對照表
- 目的：支撐 HW-aware ML Systems Engineer 定位裡 quantization/graph 能力主張，可在等 KV260 編譯空檔穿插做

## 架構說明

```
InferenceBackend          ← pure virtual（Stage 12 vtable 觀念）
    └── DpuBackend        ← 包住 vart::RunnerExt（VART DPU runtime）

Frame                     ← move-only，unique_ptr<char[]> 管 pixel data
DmaBuffer                 ← RAII malloc（未來可換成真正 DMA alloc）
FrameQueue                ← mutex + condition_variable producer-consumer
```

## 協作規則

1. 新功能先開 feature branch，從 main 分出
2. 功能完成後自動 commit + push + PR
3. 改完後 rsync + KV260 make 驗證才算完成
4. DpuBackend 的 inference 邏輯由 Felix 自己寫，給骨架不給完整解答
5. commit 格式：`feat/fix/refactor/docs/chore(<scope>): <name>`
