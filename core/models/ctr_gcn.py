# core/models/ctr_gcn.py
import torch
import torch.nn as nn

class CTRGCN(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution (CTR-GCN)
    Đây là bộ khung model để bạn chuẩn bị chạy trên Google Colab.
    Kiến trúc yêu cầu Tensor đầu vào dạng: [N, C, T, V]
    - N: Batch size
    - C: 3 kênh (X, Y, Conf)
    - T: 30 Frames
    - V: 17 Khớp (Joints theo chuẩn COCO)
    """
    def __init__(self, in_channels=3, num_classes=3, num_joints=17, num_frames=30):
        super(CTRGCN, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_joints = num_joints
        
        # 1. Feature Extraction (Cấu trúc cơ bản)
        # Thực tế CTR-GCN dùng 10 blocks (layer), ở đây tôi set up block đầu vào
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # Khai báo các GCN block ở đây (Khi đưa lên Colab, bạn sẽ import thư viện CTR-GCN gốc)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        
        # 2. Global Average Pooling (Cuộn lại để phân loại)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Fully Connected Layer (Phân loại hành vi)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Quá trình Forward Pass (Chạy inference)
        x shape: [Batch, Channels, Frames, Joints] -> [N, 3, 30, 17]
        """
        N, C, T, V = x.size()
        
        # Chạy qua GCN / Feature Extraction
        # Dữ liệu đi qua các block Không gian (Spatial) và Thời gian (Temporal)
        x = self.conv1(x) # Ví dụ placeholder
        
        # Gom Pooling
        x = self.pool(x)
        x = x.view(N, -1)
        
        # Phân loại ra Nhãn
        out = self.fc(x)
        return out
