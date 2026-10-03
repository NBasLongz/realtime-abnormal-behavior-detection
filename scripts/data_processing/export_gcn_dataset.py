# scripts/data_processing/export_gcn_dataset.py
import os
import sys
import re
import random
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict

# Set up dynamic path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

def get_group_id(filename: str) -> str:
    """
    Trích xuất ID kịch bản / nhóm để phân chia tập không bị rò rỉ dữ liệu (Zero Data Leakage).
    - Các video quay cùng 1 cú ngã đa góc máy (cam1 -> cam8) có chung 'chuteXX'.
    - Các video normal có chung 'normal_XXX'.
    """
    chute_match = re.search(r"chute\d+", filename, re.IGNORECASE)
    if chute_match:
        return chute_match.group(0).lower()
    
    normal_match = re.search(r"normal_\d+", filename, re.IGNORECASE)
    if normal_match:
        return normal_match.group(0).lower()
        
    return filename.split("_id")[0]

def augment_skeleton(keypoints: np.ndarray) -> List[np.ndarray]:
    """
    Kỹ thuật Data Augmentation cho khung xương (CHỈ ÁP DỤNG TRÊN TẬP HUẤN LUYỆN)
    Input: keypoints shape [T, 17, 3] (Frames, Joints, [x, y, conf])
    Output: List các biến thể
    """
    T, V, C = keypoints.shape
    variations = [keypoints] # Luôn giữ bản gốc
    
    # 1. Thêm nhiễu ngẫu nhiên (Gaussian Noise) cho tọa độ X, Y
    noise = np.random.normal(0, 0.01, size=(T, V, 2))
    noisy_kp = keypoints.copy()
    noisy_kp[:, :, :2] += noise
    variations.append(noisy_kp)
    
    # 2. Lật gương ngang (Horizontal Flip) quanh tâm cơ thể
    flip_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    flipped_kp = keypoints.copy()
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
    Bảo tồn trọn vẹn gia tốc rơi và hướng chuyển động!
    """
    res = seq.copy()
    xy = res[:, :, :2]
    center = np.mean(xy[0], axis=0, keepdims=True)
    xy_centered = xy - center
    
    scale = np.max(np.linalg.norm(xy[0] - center, axis=1))
    if scale > 1e-4:
        xy_centered = xy_centered / scale
        
    res[:, :, :2] = xy_centered
    return res

def process_sequence(seq: np.ndarray) -> np.ndarray:
    """Cắt hoặc padding sequence về đúng 30 frames."""
    if seq.shape[0] < 30:
        pad_length = 30 - seq.shape[0]
        padding = np.repeat(seq[-1:], pad_length, axis=0)
        seq = np.concatenate([seq, padding], axis=0)
    else:
        seq = seq[:30, :, :]
    return seq

def build_gcn_dataset_zero_leakage(data_dir: Path, test_ratio: float = 0.2):
    """
    Đọc dữ liệu và chia tập Train/Test theo GROUP / CHUTE.
    ĐẢM BẢO TUYỆT ĐỐI KHÔNG CÓ DATA LEAKAGE:
    - Không bao giờ có 2 góc máy của cùng 1 cú ngã nằm ở 2 tập khác nhau.
    - Tập Test là người hoàn toàn mới, phòng mới, và KHÔNG AUGMENTATION.
    """
    classes = [c for c in settings.classes if c != "loitering"]
    
    train_x, train_y = [], []
    test_x, test_y = [], []
    
    total_train_orig = 0
    total_test_orig = 0
    
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("CHUẨN BỊ DATASET CTR-GCN (ZERO DATA LEAKAGE SPLIT)")
    print("=" * 60)

    for cls_idx, cls_name in enumerate(classes):
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            print(f"[Cảnh báo] Thư mục {cls_dir} chưa tồn tại!")
            continue

        npy_files = list(cls_dir.glob("*.npy"))
        if not npy_files:
            print(f"[Cảnh báo] Không có file .npy nào trong {cls_dir}!")
            continue

        # Gom nhóm theo kịch bản (chute hoặc video gốc)
        group_to_files = {}
        for f in npy_files:
            gid = get_group_id(f.name)
            if gid not in group_to_files:
                group_to_files[gid] = []
            group_to_files[gid].append(f)

        groups = sorted(list(group_to_files.keys()))
        random.shuffle(groups)

        n_test = max(1, int(len(groups) * test_ratio))
        test_groups = set(groups[:n_test])
        train_groups = set(groups[n_test:])

        print(f"\n[{cls_name.upper()}] Tổng {len(groups)} kịch bản / {len(npy_files)} chuỗi:")
        print(f"  -> Train: {len(train_groups)} nhóm | Test: {len(test_groups)} nhóm (Độc lập 100%)")

        # Xử lý tập TRAIN: Chuẩn hóa + Augmentation x4
        for gid in train_groups:
            for f in group_to_files[gid]:
                try:
                    seq = process_sequence(np.load(str(f)))
                    seq = normalize_skeleton(seq)
                    total_train_orig += 1

                    # Augmentation x4 cho tập train
                    for aug_seq in augment_skeleton(seq):
                        # [30(T), 17(V), 3(C)] -> [3(C), 30(T), 17(V)]
                        gcn_format = np.transpose(aug_seq, (2, 0, 1))
                        train_x.append(gcn_format)
                        train_y.append(cls_idx)
                except Exception as e:
                    print(f"Lỗi đọc {f.name}: {e}")

        # Xử lý tập TEST: Chuẩn hóa, TUYỆT ĐỐI KHÔNG AUGMENTATION (Giữ nguyên tính trung thực)
        for gid in test_groups:
            for f in group_to_files[gid]:
                try:
                    seq = process_sequence(np.load(str(f)))
                    seq = normalize_skeleton(seq)
                    total_test_orig += 1

                    gcn_format = np.transpose(seq, (2, 0, 1))
                    test_x.append(gcn_format)
                    test_y.append(cls_idx)
                except Exception as e:
                    print(f"Lỗi đọc {f.name}: {e}")

    X_train = np.array(train_x, dtype=np.float32)
    Y_train = np.array(train_y, dtype=np.int64)
    X_test = np.array(test_x, dtype=np.float32)
    Y_test = np.array(test_y, dtype=np.int64)

    print("\n" + "=" * 60)
    print(f"TỔNG KẾT TẬP DỮ LIỆU:")
    print(f"  - Tập Train: {X_train.shape} (Gốc {total_train_orig} seqs -> Augment x4 thành {len(X_train)})")
    print(f"  - Tập Test:  {X_test.shape} (Gốc {total_test_orig} seqs thực tế, không rò rỉ)")
    print("=" * 60)

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    data_in_dir = settings.data_dir / "processed" / "sequences"
    data_out_dir = settings.data_dir / "processed" / "gcn"
    data_out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, Y_tr, X_te, Y_te = build_gcn_dataset_zero_leakage(data_in_dir)

    if len(X_tr) > 0 and len(X_te) > 0:
        np.save(str(data_out_dir / 'gcn_train_x.npy'), X_tr)
        np.save(str(data_out_dir / 'gcn_train_y.npy'), Y_tr)
        np.save(str(data_out_dir / 'gcn_test_x.npy'), X_te)
        np.save(str(data_out_dir / 'gcn_test_y.npy'), Y_te)
        print(f"\nĐã lưu thành công các file GCN tại: {data_out_dir}")
