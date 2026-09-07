import os
import torch
import torchvision.models as models
import torchvision.transforms as T
from pytorch_nndct.apis import torch_quantizer
from PIL import Image

OUTPUT_DIR = "quantize_result"
CALIB_DIR  = "calib_images"

model = models.resnet50(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imgs = sorted([f for f in os.listdir(CALIB_DIR) if f.lower().endswith(('.jpg', '.png'))])
print(f"[calib] {len(imgs)} images found in {CALIB_DIR}/")

# Phase 1: calibration
print("[1/2] Calibration...")
quantizer = torch_quantizer("calib", model, (dummy_input,), output_dir=OUTPUT_DIR)
quant_model = quantizer.quant_model

with torch.no_grad():
    for i, fname in enumerate(imgs):
        img = Image.open(os.path.join(CALIB_DIR, fname)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        quant_model(tensor)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(imgs)}")

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
