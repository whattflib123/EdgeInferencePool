# TVM 概覽筆記（面試用）

目標：面試被問「你知道 SDK 底下 compiler 在幹嘛嗎」，能用結構性語言回答。  
範圍：TVM 三件事 — IRModule 層次、BYOC、Operator Fusion。  
日期：2026-09-08

---

## 1. IRModule 層次：Relax vs TIR

TVM 的中間表示（IR）分兩層，分別對應不同抽象：

| 層次 | 名稱 | 抽象等級 | 說明 |
|---|---|---|---|
| 圖層 | **Relax function** | 高層，tensor op DAG | `conv2d → relu → add`，看不見 loop |
| 算子層 | **TIR PrimFunc** | 低層，imperative | 顯式 loop + buffer access，最終 lower 到 C/CUDA/HLS |

```
IRModule
├── @main (Relax function)          ← 整張計算圖，high-level
│   └── calls R.nn.conv2d(...)
└── @conv2d_impl (TIR PrimFunc)     ← 單一算子實作，low-level
    └── for i, j, k: buffer[i,j] += ...
```

**對應 Vitis-AI**：vai_c_xir 的輸入是 xmodel（等價 Relax graph），  
DPU 子圖內部每個 Conv 的執行邏輯由 DPU 指令集決定（等價 TIR 那層，只是封裝在 DPU binary 裡）。

---

## 2. BYOC（Bring Your Own Codegen）

**核心概念**：圖層（Relax）切分 → 不支援的 op 留給 CPU codegen，支援的 op 交給 accelerator codegen。

### BYOC 流程（TVM 視角）

```
原始 Relay/Relax graph
    ↓ 1. Annotation（標記哪些 op 目標後端支援）
    ↓ 2. Partitioning（切出 subgraph，包成 Function with attr "Compiler"="your_accel"）
    ↓ 3. Codegen dispatch（你的 codegen 接手 annotated subgraph，產出 binary 或 stub）
    ↓ 4. Runtime（執行時，annotated region 呼叫你的 runtime module）
```

### 對應 KV260 實驗

你在 KV260 上跑 ResNet50 就是一個完整的 BYOC 實例：

| TVM BYOC 概念 | vai_c_xir 對應 | KV260 實際結果 |
|---|---|---|
| Annotation | vai_c_xir 標記哪些 op DPU 支援 | Conv/BN/ReLU/Pool → DPU |
| Partitioning | 產生 3 個子圖 | USER / DPU(186 ops) / CPU(1 op) |
| Codegen | DPU binary + CPU fallback | DPU subgraph 編成 DPU 指令；FC 留 CPU |
| Partition boundary cost | Memory copy + layout transform | **15ms overhead（55% 總延遲）** |

**為什麼 FC 被切到 CPU**：  
`nn.Linear` 底層是 GEMM，DPUCZDX8G 指令集不支援 GEMM。  
BYOC 機制讓它自動 fallback — 這是設計，不是 bug。

---

## 3. Operator Fusion

**定義**：把相鄰、相容的 op 合併成一個 kernel，避免中間 tensor 寫回 DRAM。

### ResNet50 典型 fusion pattern

```
Conv2d → BatchNorm → ReLU   →  fused kernel (stay in SRAM/L1)
                              不需要 Conv 輸出寫 DRAM → BN 讀 DRAM → ...
```

TVM 的 FuseOps pass 依賴關係：
- Element-wise op（ReLU、Add）可以 fuse 到前一個 injective/reduction op 後面
- Reduction op（AvgPool）可以 fuse element-wise，但不能跨 reduction 再 fuse

### Fusion boundary = 切分點的代價來源

KV260 量測：DPU 內部 186 個 op 全部 fused，不需要寫 DRAM。  
但 DPU→CPU 的 partition boundary **強制打破 fusion**：

```
[DPU] AvgPool → 輸出 NHWC tensor → write to DRAM
                                              ↓  layout transform (NHWC→flat)
[CPU] Linear FC                              ↓  read from DRAM
```

這就是 15ms 的真正來源：不是 Linear 算很慢，是 **跨 boundary 的 memory traffic**。

---

## 4. 面試框架：三個概念一起用

> 「在 KV260 上部署 ResNet50，vai_c_xir 做了 BYOC 式的子圖切分。  
> Partitioning 把 DPU 支援的 186 個 conv/bn/relu/pool 留在 DPU，  
> 把不支援 GEMM 的 FC 切給 CPU codegen。  
> DPU 子圖內部全部 operator fusion，中間 tensor 不碰 DRAM，  
> 但 fusion boundary 強制打破，產生 memory copy + layout transform，  
> 實測 overhead 15ms，佔端對端延遲 55%。  
> 如果把 Linear 換成 Conv1×1，DPU 支援 Conv，就能消除這個 partition boundary。」

---

## 5. 延伸：若面試問更深

| 問題 | 回答方向 |
|---|---|
| TVM vs Vitis-AI 差異？ | TVM 是通用 compiler（可換後端），vai_c_xir 是 AMD DPU 專用；BYOC 概念共通 |
| Relay vs Relax？ | Relay 舊版 IR（基於 ANF），Relax 新版（支援 dynamic shape、first-class dataflow）；概念相同 |
| TIR schedule 是什麼？ | 對 PrimFunc 的 loop transformation（tiling, vectorize, unroll）；相當於手動寫 DPU-friendly kernel |
| MLIR 關係？ | TVM TIR 可以 lower 到 MLIR；MLIR 是更底層的 infrastructure，不同層次 |
