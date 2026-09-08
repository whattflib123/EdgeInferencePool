# EdgeInferencePool

ResNet50 inference pipeline on AMD KV260 — VART API → INT8 PTQ → DPU deployment → profiling.

Built to demonstrate HW-aware ML systems engineering: from quantization workflow through runtime profiling, including subgraph partitioning analysis with compiler vocabulary.

---

## Hardware / Software

| Component | Detail |
|---|---|
| Board | AMD Kria KV260, ARM Cortex-A53 |
| DPU | DPUCZDX8G_ISA1_B4096 |
| Runtime | Vitis-AI 2.5, VART API |
| Quantization | pytorch_nndct PTQ + vai_c_xir (Docker: vitis-ai-cpu 3.5.0) |
| Host | WSL2 Ubuntu 24.04, g++ 13 / cross-build for ARM64 g++ 11 |

---

## Architecture

```
InferenceBackend          pure virtual interface
    └── DpuBackend        wraps vart::RunnerExt + xir::Attrs
                          INT8 input scale via fix_point attr
                          top-K output via std::partial_sort + std::iota

Frame                     move-only, unique_ptr<char[]> pixel data
FrameQueue                mutex + condition_variable producer-consumer
main.cpp                  VideoCapture producer → DpuBackend consumer
                          supports caffe / pytorch preprocessing selector
```

---

## Results

### Throughput (ResNet50 on KV260)

| Measurement | Tool | Result |
|---|---|---|
| Pure DPU subgraph | `xdputil benchmark` | **85.2 FPS** |
| End-to-end pipeline | `vaitrace` | **36.67 FPS** |
| DPU avg latency | vart_trace.csv | **12.14 ms** |
| CPU FC + boundary overhead | derived | **15.13 ms (55%)** |

### Subgraph Partitioning

```
subgraph_ResNet__input_0                  USER  |  1 op
subgraph_ResNet__AdaptiveAvgPool2d_...    DPU   | 186 ops  (Conv→BN→ReLU→Pool)
subgraph_ResNet__ResNet_Linear_fc_...     CPU   |   1 op   (Linear FC, GEMM unsupported)
```

FC falls to CPU because DPUCZDX8G_ISA1_B4096 does not support GEMM — a BYOC partition boundary.  
The boundary costs 15ms (55% of total latency) due to memory copy + NHWC→NCHW layout transform.

---

## Quantization Workflow

```bash
# 1. PTQ calibration + export (inside Docker)
docker run --rm -v "$(pwd)/quantize":/workspace -w /workspace \
  xilinx/vitis-ai-cpu:latest \
  bash -c "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python quantize_resnet50.py && bash compile.sh"

# 2. rsync to board
rsync -av ./ root@10.42.0.199:~/EdgeInferencePool/

# 3. Build on board
cd ~/EdgeInferencePool/build && make

# 4. Run
./edge_inference quantize/compiled_model/resnet50.xmodel assets/dog.avi pytorch
```

Calibration uses 119 real images (mixed domain — see quantization notes for accuracy implications).

---

## Build

```bash
# On KV260
mkdir build && cd build
cmake .. && make
```

Requires: OpenCV 4, xir, vart (installed on KV260 via Vitis-AI 2.5).

---

## Docs

| File | Contents |
|---|---|
| [`docs/quantization_interview_notes.md`](docs/quantization_interview_notes.md) | PTQ scale factor math, calibration dataset effects, PTQ vs QAT tradeoffs, ReLU6 replacement bug |
| [`docs/subgraph_profiling_notes.md`](docs/subgraph_profiling_notes.md) | xdputil / vaitrace methodology, latency breakdown, BYOC partition boundary cost analysis |
| [`docs/tvm_overview_notes.md`](docs/tvm_overview_notes.md) | TVM IRModule (Relax vs TIR), BYOC concept mapped to Vitis-AI, operator fusion and boundary cost |
