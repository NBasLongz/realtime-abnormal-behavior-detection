# core/models/ctr_gcn.py
import torch
import torch.nn as nn
import numpy as np

def get_coco_graph():
    """
    Xay dung do thi khung xuong COCO 17 khop.
    Ket noi cac khop xuong theo giai phau co the nguoi.
    """
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Dau va mat
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Vai va tay
        (5, 11), (6, 12), (11, 12),               # Than va hong
        (11, 13), (13, 15), (12, 14), (14, 16)   # Chan
    ]
    A = np.eye(17, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
        
    # Chuan hoa bac (Degree normalization): D^(-1/2) * A * D^(-1/2)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=D > 0)
    D_mat = np.diag(D_inv_sqrt)
    norm_A = D_mat @ A @ D_mat
    return torch.tensor(norm_A, dtype=torch.float32)

class GraphConvBlock(nn.Module):
    """
    Khoi tich chap khong gian - thoi gian (Spatio-Temporal GCN Block).
    Tich hop Topology Refinement theo nguyen ly CTR-GCN.
    """
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.register_buffer('A', A)
        # Ma tran hoc loc tinh chinh do thi (Topology Refinement Matrix)
        self.M = nn.Parameter(torch.zeros(17, 17))
        
        # Spatial Graph Conv
        self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_s = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Temporal Conv (Quet qua 9 frame thoi gian de bat van toc/gia toc)
        self.conv_t = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1))
        self.bn_t = nn.BatchNorm2d(out_channels)
        
        # Duong tat Residual
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        
        # 1. Tich chap tren do thi khong gian khop xuong: A_eff = A + M
        A_eff = self.A + self.M
        x_s = torch.einsum('nctv,vw->nctw', (x, A_eff))
        x_s = self.relu(self.bn_s(self.conv_s(x_s)))
        
        # 2. Tich chap theo chieu thoi gian
        x_t = self.bn_t(self.conv_t(x_s))
        
        # 3. Ket hop Residual
        return self.relu(x_t + res)

class CTRGCN(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution (CTR-GCN)
    Kien truc Graph Convolution chuyen dung cho Nhan dien hanh vi khong gian - thoi gian.
    """
    def __init__(self, in_channels=3, num_classes=3, num_joints=17, num_frames=30):
        super(CTRGCN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        A = get_coco_graph()
        
        # Chuan hoa phan phoi dau vao
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # Cac tang Graph Convolution hoc dac trung tu nhe den sau
        self.block1 = GraphConvBlock(in_channels, 64, A, stride=1)
        self.block2 = GraphConvBlock(64, 128, A, stride=2)
        self.block3 = GraphConvBlock(128, 256, A, stride=2)
        
        # Global Average Pooling thu gon khong gian va thoi gian
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout tranh overfitting
        self.drop = nn.Dropout(0.3)
        
        # Tang phan loai cuoi cung (Classifier Head)
        self.fc = nn.Linear(256, num_classes)

    def load_pretrained_weights(self, weight_path: str, device: str = 'cpu'):
        print(f"Dang tai pretrained weights tu: {weight_path}")
        try:
            state_dict = torch.load(weight_path, map_location=device, weights_only=False)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            filtered_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            self.load_state_dict(filtered_dict, strict=False)
            print("Da load thanh cong pretrained weights!")
        except Exception as e:
            print(f"Loi khi load weights: {e}")

    def forward(self, x):
        # x shape: [N, C, T, V]
        N, C, T, V = x.size()
        
        # Chuan hoa input qua BatchNorm1d
        x_bn = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        
        # Chay qua mang Graph Convolution
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Thu gon dac trung
        x = self.pool(x)
        x = x.view(N, -1)
        x = self.drop(x)
        
        # Du doan nhan hanh vi
        out = self.fc(x)
        return out
