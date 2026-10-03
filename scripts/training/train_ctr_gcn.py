# scripts/training/train_ctr_gcn.py
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models.ctr_gcn import CTRGCN
from core.training.evaluation import Evaluator
from config.settings import settings

def main():
    print("Bắt đầu quá trình Transfer Learning cho CTR-GCN...")
    
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(f"Thiết bị huấn luyện: {device}")

    data_dir = PROJECT_ROOT / "data"/ "processed"/ "gcn"
    x_path = data_dir / "gcn_train_x.npy"
    y_path = data_dir / "gcn_train_y.npy"

    if not x_path.exists() or not y_path.exists():
        print(f"Không tìm thấy data tại {data_dir}. Vui lòng chạy export_gcn_dataset.py trước.")
        return

    # Load Data
    X = np.load(str(x_path))
    Y = np.load(str(y_path))
    print(f"Dữ liệu đã tải: X={X.shape}, Y={Y.shape}")

    # Chia tập Train/Test (80-20) để có tập Test cố định phục vụ đánh giá (Evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).to(device), torch.tensor(y_test).to(device)), batch_size=32, shuffle=False)

    # Khởi tạo mô hình
    model = CTRGCN(num_classes=3).to(device)
    
    # Load Pretrained Weights (Giả sử bạn đã chạy file tải weights về thư mục này)
    weights_path = PROJECT_ROOT / "weights"/ "classification"/ "ctrgcn_ntu_120.pt"
    if weights_path.exists():
        model.load_pretrained_weights(str(weights_path), device=str(device))
    else:
        print(f"CẢNH BÁO: Không tìm thấy pretrained weights tại {weights_path}. Sẽ train từ đầu (Từ chối Transfer Learning).")

    # Chỉ định Optimizer, Scheduler & Loss Function
    epochs = 40
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_model_path = PROJECT_ROOT / "weights" / "classification" / "ctrgcn_best.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Đánh giá (Evaluation) trên tập Test sau mỗi Epoch
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                out = model(batch_x)
                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
        
        # Dùng bộ Evaluator chuẩn hóa
        eval_results = Evaluator.evaluate(all_trues, all_preds)
        val_f1 = eval_results['macro_f1']
        
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {eval_results['accuracy']:.4f} | Val Macro-F1: {val_f1:.4f}")
        
        # Lưu mô hình tốt nhất
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), str(best_model_path))
            print(f"Đã lưu mô hình tốt nhất đạt Macro-F1: {best_f1:.4f}")

    print("\nHOÀN TẤT HUẤN LUYỆN! BÁO CÁO KẾT QUẢ TỐT NHẤT:")
    # Tải lại model tốt nhất và in báo cáo chuẩn
    model.load_state_dict(torch.load(str(best_model_path), weights_only=False))
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            out = model(batch_x)
            all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            all_trues.extend(batch_y.cpu().numpy())
    
    target_names = [c.capitalize() for c in settings.classes if c != "loitering"]
    final_eval = Evaluator.evaluate(all_trues, all_preds, target_names=target_names)
    Evaluator.print_report(final_eval)

    # Xuất ra định dạng ONNX
    print("\nĐang xuất mô hình ra định dạng ONNX...")
    onnx_path = best_model_path.with_suffix('.onnx')
    dummy_input = torch.randn(1, 3, 30, 17).to(device)
    try:
        torch.onnx.export(
            model, dummy_input, str(onnx_path), 
            export_params=True, opset_version=18, 
            input_names=['input'], output_names=['output']
        )
        print(f"Đã xuất ONNX thành công tại: {onnx_path}")
    except Exception as e:
        print(f"Lỗi xuất ONNX trực tiếp: {e}. Vui lòng chạy export_onnx.py sau khi cài onnxscript.")

if __name__ == "__main__":
    main()
