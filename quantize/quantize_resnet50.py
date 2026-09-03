import torch
import torchvision.models as models
from pytorch_nndct.apis import torch_quantizer

OUTPUT_DIR = "quantize_result"
CALIB_ITERS = 100

model = models.resnet50(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# Phase 1: calibration
print("[1/2] Calibration...")
quantizer = torch_quantizer("calib", model, (dummy_input,), output_dir=OUTPUT_DIR)
quant_model = quantizer.quant_model

with torch.no_grad():
    for i in range(CALIB_ITERS):
        quant_model(torch.randn(1, 3, 224, 224))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{CALIB_ITERS}")

quantizer.export_quant_config()
print(f"[1/2] Done. Config saved to {OUTPUT_DIR}/")

# Phase 2: export xmodel
print("[2/2] Exporting xmodel...")
quantizer = torch_quantizer("test", model, (dummy_input,), output_dir=OUTPUT_DIR)
quant_model = quantizer.quant_model

with torch.no_grad():
    quant_model(dummy_input)

quantizer.export_xmodel(output_dir=OUTPUT_DIR, deploy_check=False)
print(f"[2/2] Done. xmodel saved to {OUTPUT_DIR}/")
