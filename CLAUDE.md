# EdgeInferencePool

## 這個 repo 是什麼

Felix 的 M1 DpuBackend 專案。把 VART API 接進 `InferenceBackend` 抽象介面，
跑在 KV260（AMD Kria Vision AI Starter Kit）+ Vitis-AI 環境上。

前身是 `/home/felix/Desktop/my_stuff/C` 的 stage3_integration/phase10_mini_pipeline，
那裡的 DmaBuffer / Frame / FrameQueue 已搬過來。

## 使用者背景

- Felix，SRAM AMR perception engineer，目標求職/轉職
- C++ Stage 0~13 已完成（pointer、RAII、Rule of Five、span、vector、多執行緒、多型、template）
- 這個 repo 是把觀念搬進真實硬體專案的實戰期
- 求職敘事核心：「能指著真實程式碼解釋 RAII / move / span / virtual 怎麼串在一起」

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

## 目前狀態（2026-09-01）

**編譯：✅ KV260 上 `make` 通過**

### 已完成
- `InferenceBackend` pure virtual 介面（`Detection` struct + `run(span<const float>)`）
- `DpuBackend : InferenceBackend`，持有 `vart::RunnerExt` + `xir::Attrs`
- `DpuBackend::DpuBackend(subgraph)` — runner 建好（`RunnerExt::create_runner`）
- `main.cpp` — xmodel 載入、DPU subgraph 搜尋、producer-consumer pipeline

### 卡住點（TODO）

`src/DpuBackend.cpp` 裡兩個 TODO：

```cpp
// TODO: copy input data into inputs[0]'s tensor buffer
//   hint: inputs[0]->data() returns {void*, size_t}

// TODO: parse outputs[0] → Detection objects
//   shape depends on model output layer
```

**需要 `.xmodel` 才能繼續**——input/output tensor shape 要對上模型。

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
