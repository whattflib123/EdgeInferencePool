# DPU 子圖切分 + Profiler 分析筆記

硬體：KV260（DPUCZDX8G_ISA1_B4096）  
模型：ResNet50（torchvision PTQ → vai_c_xir）  
工具：xdputil、vaitrace、xir Python API  
日期：2026-09-08

---

## 1. 子圖切分結構

用 xir Python API 查詢 xmodel：

```python
import xir
g = xir.Graph.deserialize("resnet50.xmodel")
for sg in g.get_root_subgraph().toposort_child_subgraph():
    dev = sg.get_attr("device") if sg.has_attr("device") else "CPU"
    print(sg.get_name(), "|", dev, "| ops:", sg.get_op_num())
```

結果：

| 子圖 | 裝置 | Op 數 | 內容 |
|---|---|---|---|
| `subgraph_ResNet__input_0` | USER | 1 | 輸入節點 |
| `subgraph_ResNet__AdaptiveAvgPool2d_avgpool__8286_i0` | DPU | 186 | Conv→BN→ReLU→...→AvgPool（主體） |
| `subgraph_ResNet__ResNet_Linear_fc__8290_fix_` | CPU | 1 | Linear FC（1000 類）|

**FC 被切到 CPU 的原因**：DPUCZDX8G 指令集不支援 GEMM（General Matrix Multiply）。`nn.Linear` 在 vai_c_xir 編譯時自動 fallback 到 CPU subgraph，形成 BYOC（Bring Your Own Codegen）切分點。

---

## 2. 延遲量測

### 工具 1：xdputil benchmark（純 DPU 吞吐）

```bash
xdputil benchmark resnet50.xmodel 1
```

結果：**85.2 FPS**（純 DPU 子圖，不含 CPU FC）

### 工具 2：vaitrace（端對端 pipeline）

```bash
vaitrace -o trace_output ./edge_inference resnet50.xmodel video.avi
```

結果：**36.67 FPS**（含 CPU FC + 切換 overhead）

### 工具 3：vart_trace.csv 解析

vaitrace 輸出的 `vart_trace.csv` 包含每幀 DPU start/end 事件，可算出精確 DPU latency：

| 項目 | 數值 |
|---|---|
| DPU 子圖平均延遲 | **12.14 ms** |
| DPU 最小延遲 | 11.98 ms |
| DPU 最大延遲 | 12.33 ms |
| 端對端延遲 | **27.3 ms** |
| CPU FC + 切換 overhead | **15.13 ms（55%）** |

---

## 3. 切分點代價分析

CPU FC 只有 1 個 op，卻造成 **15ms（55% 總延遲）**的 overhead。

代價來源（fusion boundary cost）：

1. **Memory copy**：DPU 輸出張量從 DPU 內部 SRAM → CPU DRAM，ResNet50 AvgPool 輸出 shape 1×2048，需要搬移資料
2. **Layout 轉換**：DPU 內部使用 NHWC（channel-last）格式，CPU 端需轉回 NCHW 或 flat vector
3. **CPU 執行 Linear**：相對快，但在 ARM Cortex-A53 上仍需數 ms

這就是為什麼「子圖切分」是部署優化的關鍵：**一個不支援的 op 讓整體速度從 85 FPS 掉到 36 FPS（下降 57%）**。

---

## 4. Compiler 語彙對應（面試框架）

| 部署觀察 | Compiler 語彙 |
|---|---|
| FC 不進 DPU | **Partitioning**：compiler 把計算圖切成可在不同後端執行的子圖 |
| DPU/CPU 交界 | **Fusion boundary**：無法跨越的切分點，兩側各自獨立優化 |
| KV260 DPU 不支援 GEMM | **BYOC（Bring Your Own Codegen）**：DPU 作為一個 accelerator backend，只接手它支援的 op，其餘留給 CPU codegen |
| 切換 overhead | **Memory traffic at partition boundary**：crossing the boundary requires data layout transform + DMA transfer |

---

## 5. 如果要消除這個切分點

方案一：**把 Linear 換成等效 Conv1x1**  
`nn.Linear(2048, 1000)` 可以用 `nn.Conv2d(2048, 1000, 1)` 代替，DPU 支援 Conv，這樣整個網路可以留在 DPU。

方案二：**接受切分，優化 memory transfer**  
如果切分不可避免，可以確保 DPU 輸出張量直接 map 到 CPU 可見記憶體（zero-copy），減少搬移成本。

方案三：**換 DPU 架構**  
DPUCVDX8G（Versal 系列）支援更多 op，可能能消化 Linear。

---

## 6. Calibration 與精度觀察

| Calibration 資料 | dog.jpg top-1 | 說明 |
|---|---|---|
| 隨機 tensor（100 次）| class=922（book jacket）❌ | activation 統計跑偏 |
| AMR 走廊幀 + dog.jpg（119 張）| class=249（Malamute）⚠️ | 有改善，但非 ImageNet 分佈 |
| ImageNet val set（建議）| class=207（golden retriever）✅ | 與訓練分佈一致 |

**結論**：Calibration dataset 要與模型訓練分佈一致，工業場景幀對 ImageNet 模型效果有限。
