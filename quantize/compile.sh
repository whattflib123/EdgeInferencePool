#!/bin/bash
set -e

PT_ENV=/opt/vitis_ai/conda/envs/vitis-ai-pytorch
TF_ENV=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow
ARCH=${PT_ENV}/lib/python3.7/site-packages/vaic/arch/DPUCZDX8G/KV260/arch.json
INPUT_XMODEL=quantize_result/ResNet_int.xmodel
OUTPUT_DIR=compiled_model

export PATH=${TF_ENV}/bin:${PATH}
export PYTHONPATH=${PT_ENV}/lib/python3.7/site-packages/vaic:${PYTHONPATH}

mkdir -p ${OUTPUT_DIR}

${PT_ENV}/bin/vai_c_xir \
  --xmodel ${INPUT_XMODEL} \
  --arch ${ARCH} \
  --net_name resnet50 \
  --output_dir ${OUTPUT_DIR}

echo "Compiled xmodel: ${OUTPUT_DIR}/resnet50.xmodel"
