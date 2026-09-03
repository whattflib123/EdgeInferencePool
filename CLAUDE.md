# EdgeInferencePool

## 這個 repo 是什麼

Felix 的 M1 DpuBackend 專案。把 VART API 接進 `InferenceBackend` 抽象介面，
跑在 KV260（AMD Kria Vision AI Starter Kit）+ Vitis-AI 環境上。

前身是 `/home/felix/Desktop/my_stuff/C` 的 stage3_integration/phase10_mini_pipeline，
那裡的 DmaBuffer / Frame / FrameQueue 已搬過來。

## 使用者背景

- Felix，SRAM AMR perception engineer，求職定位：**HW-aware ML Systems Engineer**
- 優先鎖定 deployment/runtime 職缺（NeuroPilot SDK、Vitis-AI/ROCm runtime 這類），非 compiler 或 RTL 路線
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

## 目前狀態（2026-09-02）

**M1：✅ 完成並驗證**

### 已完成
- `InferenceBackend` pure virtual 介面（`Detection` struct + `run(span<const float>)`）
- `DpuBackend : InferenceBackend`，持有 `vart::RunnerExt` + `xir::Attrs`
- INT8 量化輸入轉換（fix_point scale，逐元素 float → int8 clamp）
- 輸出解析（top-5 argmax via `std::partial_sort + std::iota`）
- OpenCV 前處理（Caffe BGR mean subtract：`(pixel - [104,117,123]) / 255`）
- `main.cpp`：`cv::imread` 單張圖推論，producer-consumer pipeline，KV260 驗證通過
- 驗證結果：`assets/dog.jpg`（金毛獵犬）→ top-1 class=337（golden retriever）✅

### 下一步
任務 1（手刻 INT8 量化）→ 任務 2（M2 VideoCapture + 現成模型量化）

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

## 專案路線圖（職涯缺口補強版）

目標職缺：AMD Taiwan AI R&D、TSMC AAID 等應用/部署路線。

| 順序 | 任務 | 對應缺口 | 狀態 |
|---|---|---|---|
| 2a | **VideoCapture 串流 pipeline** | VART API 實戰深化；`cv::VideoCapture` 取代 `cv::imread`，producer-consumer 改成 loop | 未開始 |
| 2b | **M2：現成模型量化部署**（torchvision ResNet50 或 MobileNetV2） | 量化工作流實戰：`vai_q_pytorch` PTQ calibration → `vai_c_xir` 編譯 → xmodel → KV260；每次因硬體限制被迫調整（channel 對齊、不支援 op 替換）**當下記錄**成清單；若行為與論文 ZCU102 經驗不符（工具版本差異、參數預設值改變），順手記一筆 | 未開始 |
| 3 | **刻意製造子圖切分場景** | 為 Profiler 製造可觀測案例；保留一個 DPU 不支援的 op 或跳過 op fusion，確保 xmodel 存在 DPU/CPU 交界 | 未開始 |
| 4 | **Vitis AI Profiler 實測** | Profiler 實戰：從報表指出子圖切分點的延遲量並解釋原因；**敘事框架**：用 compiler 語彙（partitioning、fusion boundary、BYOC）重描一次，讓同一份分析同時回答「懂 deployment」跟「對 compiler 那層有結構性理解」；發現併入任務 2b 的清單 → 合成「軟硬體協同設計筆記」 | 未開始 |

執行順序：2a/2b 平行開始 → 3 → 4。任務 3、4 不等 2a 全完，有可跑的 xmodel 就可插入。

**旁支（不擋主線，找空檔做）**
- 論文量化決策整理：scale factor 怎麼算、PTQ vs QAT 取捨理由、ZCU102 量化前後精度掉多少。回想＋寫下來即可，分鐘等級。
- **TVM 概覽（M2 完成後才開始，現在不要碰）**：目的是面試被問「你知道 SDK 底下 compiler 在幹嘛嗎」時能答出結構性理解。只挑 TVM（不碰 MLIR/XLA/IREE）。只需搞懂三件事：(1) IRModule 裡 Relax function（圖層）vs TIR PrimFunc（算子層）；(2) BYOC 概念——對應 vai_c 的 DPU/CPU 子圖切分；(3) Operator fusion 基本概念。看得懂官方 BYOC tutorial 即可，不用碰 relax.build() 原始碼或自己寫 pass。

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
6. Felix 同時與 claude.ai（網頁版 Claude）協作——一個階段結束後，詢問是否產出進度總覽（可貼給網頁版 Claude 同步 context）
