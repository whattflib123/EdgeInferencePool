#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON=/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python3

docker run --rm \
  -v "${SCRIPT_DIR}":/workspace \
  -w /workspace \
  xilinx/vitis-ai-cpu:latest \
  bash -c "${PYTHON} quantize_resnet50.py && bash compile.sh"
