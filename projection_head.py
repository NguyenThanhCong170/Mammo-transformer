import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        """
        input_dim: Kích thước vector đầu ra của SwinV2 (VD: 768 cho SwinV2-T, 1024 cho SwinV2-B)
        hidden_dim: Kích thước lớp ẩn
        out_dim: Kích thước vector cuối cùng dùng để tính loss
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

