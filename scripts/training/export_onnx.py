# scripts/training/export_onnx.py
import sys
import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models.ctr_gcn import CTRGCN

def main():
    model_path = PROJECT_ROOT / "weights"/ "classification"/ "ctrgcn_best.pt"
    onnx_path = PROJECT_ROOT / "weights"/ "classification"/ "ctrgcn_best.onnx"
    
    if not model_path.exists():
        print(f"Không tìm thấy model checkpoint tại: {model_path}")
        return

    device = torch.device('cpu')
    model = CTRGCN(num_classes=3).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
    model.eval()

    dummy_input = torch.randn(1, 3, 30, 17, device=device)
    print("⏳ Đang xuất mô hình ra ONNX...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"Xuất ONNX thành công tại: {onnx_path}")
    except Exception as e:
        print(f"Thử chế độ fallback export: {e}")
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
        print(f"Xuất ONNX fallback thành công tại: {onnx_path}")

if __name__ == "__main__":
    main()
