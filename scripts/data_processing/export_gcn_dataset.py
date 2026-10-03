# scripts/data_processing/export_gcn_dataset.py
import os
import sys
from pathlib import Path
import numpy as np
import random
from typing import List

# Set up dynamic path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

def augment_skeleton(keypoints: np.ndarray) -> List[np.ndarray]:
    """
    Kỹ thuật Data Augmentation cho khung xương
    Input: keypoints shape [T, 17, 3] (Frames, Joints, [x, y, conf])
    Output: List các biến thể của sequence này
    """
    T, V, C = keypoints.shape
    variations = [keypoints] # Luôn giữ bản gốc
    
    # 1. Thêm nhiễu ngẫu nhiên (Gaussian Noise) cho tọa độ X, Y
    noise = np.random.normal(0, 0.01, size=(T, V, 2))
    noisy_kp = keypoints.copy()
    noisy_kp[:, :, :2] += noise
    variations.append(noisy_kp)
    
    # 2. Lật gương ngang (Horizontal Flip)
    flip_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    flipped_kp = keypoints.copy()
    # Lấy tọa độ trung bình X làm tâm lật để không bị lệch khỏi màn hình
    center_x = np.mean(flipped_kp[:, :, 0])
    flipped_kp[:, :, 0] = center_x - (flipped_kp[:, :, 0] - center_x)
    for left_idx, right_idx in flip_pairs:
        temp = flipped_kp[:, left_idx, :].copy()
        flipped_kp[:, left_idx, :] = flipped_kp[:, right_idx, :]
        flipped_kp[:, right_idx, :] = temp
    variations.append(flipped_kp)
    
    # 3. Cắt xén thời gian ngẫu nhiên (Temporal Dropping)
    drop_indices = random.sample(range(1, T), max(1, int(T * 0.1)))
    dropped_kp = keypoints.copy()
    for idx in drop_indices:
        dropped_kp[idx] = dropped_kp[idx - 1]
    variations.append(dropped_kp)

    return variations

def normalize_skeleton(seq: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa tọa độ khung xương:
    Lấy tâm cơ thể ở frame đầu tiên làm gốc tọa độ (0, 0).
    Nhờ vậy, toàn bộ chuyển động rơi ngã, đấm đá giữa các frame được BẢO TOÀN NGUYÊN VẸN!
    """
    res = seq.copy()
    xy = res[:, :, :2] # [T, 17, 2]
    # Lấy tâm cơ thể ở frame đầu tiên (t=0) làm gốc tọa độ
    center = np.mean(xy[0], axis=0, keepdims=True) # [1, 2]
    xy_centered = xy - center # [T, 17, 2]
    
    # Scale theo kích thước cơ thể ở frame đầu tiên
    scale = np.max(np.linalg.norm(xy[0] - center, axis=1))
    if scale > 1e-4:
        xy_centered = xy_centered / scale
        
    res[:, :, :2] = xy_centered
    return res

def build_gcn_dataset(data_dir: Path):
    """
    Đọc dữ liệu LSTM cũ (.npy) và chuyển sang format CTR-GCN: [N, C, T, V]
    """
    classes = ["normal", "fighting", "falling"]
    gcn_data = []
    labels = []
    total_original = 0
    total_augmented = 0
    
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            continue
            
        npy_files = list(cls_dir.glob("*.npy"))
        for npy_file in npy_files:
            try:
                seq = np.load(str(npy_file))
                
                # Cắt/Padding thành đúng 30 frame
                if seq.shape[0] < 30:
                    pad_length = 30 - seq.shape[0]
                    padding = np.repeat(seq[-1:], pad_length, axis=0)
                    seq = np.concatenate([seq, padding], axis=0)
                else:
                    seq = seq[:30, :, :] 
                
                total_original += 1
                
                # Chuẩn hóa tọa độ không gian (tâm cơ thể và scale)
                seq = normalize_skeleton(seq)
                
                # Augment dữ liệu x4
                augmented_seqs = augment_skeleton(seq)
                
                for aug_seq in augmented_seqs:
                    # Đổi chiều [30(T), 17(V), 3(C)] -> [3(C), 30(T), 17(V)]
                    gcn_format = np.transpose(aug_seq, (2, 0, 1))
                    gcn_data.append(gcn_format)
                    labels.append(cls_idx) 
                    total_augmented += 1
                    
            except Exception as e:
                print(f"Lỗi đọc file {npy_file}: {e}")
                
    if not gcn_data:
        print("Không tìm thấy dữ liệu gốc để chuyển đổi!")
        return None, None
        
    final_x = np.array(gcn_data, dtype=np.float32)
    final_y = np.array(labels, dtype=np.int64)
    
    print(f"Đã chuẩn bị dữ liệu GCN: X={final_x.shape}, Y={final_y.shape}")
    print(f"Dữ liệu gốc: {total_original} sequences")
    print(f"Sau khi Augment (x4): {total_augmented} sequences")
    return final_x, final_y

if __name__ == "__main__":
    # Đã đổi thành đường dẫn động (Dynamic Paths) theo cấu hình của dự án
    data_in_dir = settings.data_dir / "processed"/ "sequences"
    data_out_dir = settings.data_dir / "processed"/ "gcn"
    data_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⏳ Đang đọc dữ liệu từ: {data_in_dir}")
    print("⏳ Đang xử lý và chuyển đổi dữ liệu sang CTR-GCN Format...")
    
    x, y = build_gcn_dataset(data_in_dir)
    
    if x is not None:
        out_x_path = data_out_dir / 'gcn_train_x.npy'
        out_y_path = data_out_dir / 'gcn_train_y.npy'
        np.save(str(out_x_path), x)
        np.save(str(out_y_path), y)
        print(f"Hoàn thành! File được lưu tại: {data_out_dir}")
