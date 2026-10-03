# core/models/ctr_gcn.py
import torch
import torch.nn as nn

class CTRGCN(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution (CTR-GCN)
    Kiến trúc hỗ trợ Transfer Learning từ tập dữ liệu NTU-RGB+D.
    """
    def __init__(self, in_channels=3, num_classes=3, num_joints=17, num_frames=30):
        super(CTRGCN, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_joints = num_joints
        
        # 1. Feature Extraction (Cấu trúc cơ bản)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # GCN block placeholder (Sẽ dùng thư viện gốc khi train trên Colab)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        
        # 2. Global Average Pooling (Cuộn lại để phân loại)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Fully Connected Layer (Tầng phân loại cuối cùng - Phục vụ Transfer Learning)
        # Sẽ bị đè nếu nạp pre-trained weights NTU-RGB+D (120 classes)
        self.fc = nn.Linear(64, num_classes)

    def load_pretrained_weights(self, weight_path: str, device: str = 'cpu'):
        """
        Hàm học chuyển giao (Transfer Learning).
        Tải cục tạ khổng lồ của NTU-RGB+D nhưng BỎ QUA lớp FC cuối (120 classes)
        để giữ nguyên lớp FC mới (3 classes) của bài toán hiện tại.
        """
        print(f"Đang tải pretrained weights từ: {weight_path}")
        try:
            state_dict = torch.load(weight_path, map_location=device)
            # Nếu file weights có bọc trong key 'state_dict'(chuẩn của mmcv/mmaction)
            if 'state_dict'in state_dict:
                state_dict = state_dict['state_dict']

            # Lọc bỏ lớp classification cuối (vì nó là 120 class, không khớp 3 class)
            filtered_dict = {k: v for k, v in state_dict.items() if 'fc'not in k}
            
            # Nạp vào model hiện tại
            missing_keys, unexpected_keys = self.load_state_dict(filtered_dict, strict=False)
            print("Đã load thành công kiến thức cơ thể người (Transfer Learning)!")
            print(f"Các lớp được giữ lại để học mới: {missing_keys}")
            
        except Exception as e:
            print(f"Lỗi khi load weights: {e}")

    def forward(self, x):
        """
        Quá trình Forward Pass (Chạy inference)
        x shape: [Batch, Channels, Frames, Joints] -> [N, 3, 30, 17]
        """
        N, C, T, V = x.size()
        
        # Chạy qua GCN / Feature Extraction
        x = self.conv1(x) 
        
        # Gom Pooling
        x = self.pool(x)
        x = x.view(N, -1)
        
        # Phân loại ra Nhãn
        out = self.fc(x)
        return out
