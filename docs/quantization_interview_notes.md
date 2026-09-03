# 量化部署面試素材

來源：MobileFaceNet × KV260 INT8 量化實驗（2026-04）  
硬體：DPUCZDX8G_ISA1_B4096，工具：Vitis-AI 2.5 / pytorch_nndct

---

## 1. Scale factor 怎麼算，以及我踩過的坑

**標準做法（per-tensor，對稱量化）**

```
scale = 2^fix_point
INT8 = clamp(float × scale, -128, 127)
```

KV260 VART 的 `fix_point` attr 直接給出指數，`scale = 1 << fix_point`。

**我踩過的坑：`1/127.5` vs `1/128.0`**

MobileFaceNet 的前處理要把像素從 [0,255] 壓到 [-1,1]：

```python
# 錯的（論文初版）
img = (img / 127.5) - 1.0      # scale = 1/127.5

# 對的
img = (img - 128.0) / 128.0    # scale = 1/128.0，對齊 INT8 的 [-128,127] 範圍
```

`1/127.5` 讓輸入分佈略微偏移，造成 calibration 統計與推論時不對齊。修正後 embedding 相似度分佈才回到合理範圍。

**per-channel vs per-tensor 在這次有沒有用到**

pytorch_nndct 的 PTQ 預設 per-tensor。MobileFaceNet 的 GDConv（Group Depthwise Conv）對 per-tensor 量化特別敏感——每個 channel 的數值分佈差異大，per-tensor 的單一 scale 覆蓋不了所有 channel，造成 embedding space 塌陷。改成 per-channel 是理論上的解法，但 DPUCZDX8G 的 DPU IP 只支援 per-tensor weight quantization，所以這個方向在這顆硬體上走不通。

---

## 2. PTQ calibration dataset 怎麼選，為什麼

**v1 用通用人臉圖片（錯的）**

結果：FAR=11.33%，Impostor 最高相似度 0.96（float32 只有 0.66）。

原因：calibration 圖片與訓練分佈不一致，nndct 計算出的 activation 統計跑偏，scale factor 對真實推論輸入沒代表性。

**v2/v3 改用 CASIA-FaceV5 對齊臉（對的）**

- 與模型訓練分佈一致（同類型人臉資料集）
- 做過 ArcFace 112×112 標準對齊，確保前處理與推論時完全相同

**結論（面試可直接說）**

> Calibration dataset 要跟推論時的輸入分佈越接近越好。通用圖片讓 activation 統計失真，scale factor 算錯，INT8 輸出跟 float32 偏差就大。我在這次實驗中 v1 到 v2 的改善完全來自換 calibration 資料，不是改模型。

---

## 3. 量化前後精度掉多少，怎麼判斷可接受

| | PC Float32 | KV260 INT8 最佳（v3） |
|---|---|---|
| FAR（thr=0.60） | 2.00% | **25.33%** |
| TAR（thr=0.60） | 98.00% | 99.00% |
| Impostor 最高相似度 | 0.66 | **0.96** |

精度差距：FAR 從 2% 跳到 25%，**不可接受**。

**根本原因（非 calibration 問題）**

MobileFaceNet 的 GDConv 層在 INT8 下 embedding 空間嚴重塌陷——不同人的 embedding 距離過近，單靠換 calibration data 無法根本解決。這是模型架構對量化不友善的問題。

**怎麼判斷可不可接受（面試可直接說）**

> 我用 FAR/TAR 在不同閾值下的 sweep 來評估。Float32 在 thr=0.70 可以做到 FAR=0%、TAR=99%，是上限。INT8 的 Impostor 最高相似度從 0.66 跳到 0.96，代表 embedding space 結構性損壞，不是調閾值能救回來的。這種情況下 PTQ 不夠，要上 QAT 或換量化友善的模型架構。

---

## 4. PTQ vs QAT 取捨

| | PTQ | QAT |
|---|---|---|
| 需要重新訓練 | 否 | 是（fine-tune） |
| 需要 calibration data | 是（少量） | 是（訓練集） |
| 效果上限 | 模型對量化敏感時有硬上限 | 通常比 PTQ 好 5~15% FAR |
| 這次選 PTQ 的原因 | 沒有訓練環境，快速驗證部署可行性 | — |

**這次結論**：GDConv 對 INT8 精度損失太大，PTQ 做到極限（v3）後 FAR 仍 25%，建議 QAT 或換架構。

---

## 5. 最關鍵的一個 bug：ReLU6 被偷換成 ReLU

pytorch_nndct 預設把模型裡的 ReLU6 替換成 ReLU（為了量化友善），但 MobileFaceNet 是用 ReLU6 訓練的，替換後造成 train/inference mismatch。

**v2 → v3 的關鍵 fix：**
```python
from pytorch_nndct.apis import NndctOption
NndctOption.nndct_relu6_replace.value = ''  # 阻止 ReLU6 被替換
```

效果：FAR 從 ~10% 降到 4.67%（thr=0.90）。

**面試版本（30 秒能說完）**

> nndct 量化工具預設會把 ReLU6 換成 ReLU，理由是 ReLU6 的截斷值 6 在 INT8 下可能造成 scale 浪費。但如果模型訓練時用的是 ReLU6，這個替換就會讓 inference 行為跟訓練不一致。我在實驗裡發現 Impostor 相似度異常飆高（0.96），追根到這個 mismatch，加一行設定把替換關掉後精度明顯改善。
