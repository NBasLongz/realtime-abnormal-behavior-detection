# scripts/data_processing/export_gcn_dataset.py
import os
import numpy as np
import random
from typing import List

def augment_skeleton(keypoints: np.ndarray) -> List[np.ndarray]:
    """
    Kỹ thuật Data Augmentation cho khung xương (Bí kíp 1).
    Input: keypoints shape [T, 17, 3] (Frames, Joints, [x, y, conf])
    Output: List các biến thể của sequence này
    """
    T, V, C = keypoints.shape
    variations = [keypoints] # Luôn giữ bản gốc
    
    # 1. Thêm nhiễu ngẫu nhiên (Gaussian Noise) cho tọa độ X, Y
    noise = np.random.normal(0, 0.02, size=(T, V, 2))
    noisy_kp = keypoints.copy()
    noisy_kp[:, :, :2] += noise
    variations.append(noisy_kp)
    
    # 2. Lật gương ngang (Horizontal Flip)
    # Cần map các khớp trái sang phải tương ứng với định dạng COCO 17
    # Mắt trái(1) <-> Mắt phải(2), Vai trái(5) <-> Vai phải(6), v.v.
    flip_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    flipped_kp = keypoints.copy()
    # Đảo X (Giả sử X đã chuẩn hóa 0-1)
    flipped_kp[:, :, 0] = 1.0 - flipped_kp[:, :, 0]
    for left_idx, right_idx in flip_pairs:
        # Swap các cặp trái phải
        temp = flipped_kp[:, left_idx, :].copy()
        flipped_kp[:, left_idx, :] = flipped_kp[:, right_idx, :]
        flipped_kp[:, right_idx, :] = temp
    variations.append(flipped_kp)
    
    # 3. Cắt xén thời gian ngẫu nhiên (Temporal Dropping)
    # Xóa ngẫu nhiên 10% số frame để AI không học thuộc lòng
    drop_indices = random.sample(range(T), max(1, int(T * 0.1)))
    dropped_kp = keypoints.copy()
    for idx in drop_indices:
        if idx > 0:
            dropped_kp[idx] = dropped_kp[idx - 1] # Copy frame trước đó đè lên
    variations.append(dropped_kp)

    return variations

def build_gcn_dataset(raw_dataset_list):
    """
    Chuyển đổi dữ liệu từ List sang chuẩn Tensor của CTR-GCN
    Format đích: [N, C, T, V]
    - N: Số lượng mẫu (Samples)
    - C: 3 kênh (X, Y, Conf)
    - T: 30 Frames
    - V: 17 Khớp (Joints)
    """
    gcn_data = []
    labels = []
    
    for item in raw_dataset_list:
        # Giả sử item["sequence"] có shape [30, 34] (đã bị duỗi thẳng)
        # 1. Reshape lại thành [30, 17, 2] (hoặc 3 nếu có conf)
        # Bạn sẽ cần sửa lại đoạn này khớp với cấu trúc numpy cũ của bạn
        seq = np.array(item["sequence"]).reshape((30, 17, -1)) 
        label = item["label"]
        
        # 2. Augment (Nhân bản dữ liệu x4 lần)
        augmented_seqs = augment_skeleton(seq)
        
        for aug_seq in augmented_seqs:
            # 3. Đảo trục từ [T, V, C] sang [C, T, V] theo chuẩn PyTorch / GCN
            # Tức là từ (30, 17, 3) thành (3, 30, 17)
            gcn_format = np.transpose(aug_seq, (2, 0, 1))
            gcn_data.append(gcn_format)
            labels.append(label)
            
    final_x = np.array(gcn_data) # [N, 3, 30, 17]
    final_y = np.array(labels)
    
    print(f"✅ Đã chuẩn bị dữ liệu GCN: X={final_x.shape}, Y={final_y.shape}")
    return final_x, final_y

if __name__ == "__main__":
    print("Script chuyển đổi và Augment dữ liệu sang chuẩn CTR-GCN đã sẵn sàng!")
    # Lưu dưới dạng .npy để đẩy lên Google Colab
    # np.save('gcn_data_x.npy', final_x)
    # np.save('gcn_data_y.npy', final_y)
