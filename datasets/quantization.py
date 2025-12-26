import numpy as np
from abc import ABC, abstractmethod
import torch
import MinkowskiEngine as ME


class Quantizer(ABC):
    @abstractmethod
    def __call__(self, pc):
        pass


class BEVQuantizer(Quantizer):
    def __init__(self,
                 coords_range=[-10., -10, -4, 10, 10, 8],
                 div_n=[256, 256, 32]):
        """
        BEV 量化器（专用于 MinkLocBev）
        Args:
            coords_range: 点云裁剪范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            div_n: 网格划分数 [nx, ny, nz]
        """
        super().__init__()
        self.coords_range = torch.tensor(coords_range, dtype=torch.float)
        self.div_n = torch.tensor(div_n, dtype=torch.int32)
        self.steps = (self.coords_range[3:] - self.coords_range[:3]) / self.div_n

        print(f"BEVQuantizer Initialized:")
        print(f"  Range: {self.coords_range.tolist()}")
        print(f"  Grid: {self.div_n.tolist()}")
        print(f"  Steps: {self.steps.tolist()}")

    def __call__(self, pc):
        """
        Args:
            pc: (N, 3) Tensor, Cartesian coordinates (X, Y, Z)
        Returns:
            unique_xy: (M, 2) Tensor, 2D坐标
            features: (M, 32) Tensor, Z轴Occupancy特征
        """
        device = pc.device
        coords_range = self.coords_range.to(device)
        steps = self.steps.to(device)
        div_n = self.div_n.to(device)

        # 过滤范围外的点
        mask = (pc[:, 0] >= coords_range[0]) & (pc[:, 0] < coords_range[3]) & \
               (pc[:, 1] >= coords_range[1]) & (pc[:, 1] < coords_range[4]) & \
               (pc[:, 2] >= coords_range[2]) & (pc[:, 2] < coords_range[5])
        pc = pc[mask]

        if pc.shape[0] == 0:
            return torch.zeros((0, 2), dtype=torch.int32, device=device), \
                   torch.zeros((0, div_n[2]), dtype=torch.float32, device=device)

        # 计算网格索引
        indices = ((pc - coords_range[:3]) / steps).long()
        indices = torch.clamp(indices, min=torch.zeros(3, dtype=torch.long, device=device),
                              max=(div_n - 1))

        xy_indices = indices[:, :2]
        z_indices = indices[:, 2]

        # XY平面去重
        unique_xy, inverse_indices = torch.unique(xy_indices, dim=0, return_inverse=True)

        # 构建特征
        num_unique = unique_xy.shape[0]
        num_channels = div_n[2]
        features = torch.zeros((num_unique, num_channels), dtype=torch.float32, device=device)
        features.index_put_((inverse_indices, z_indices), torch.tensor(1.0, device=device))

        return unique_xy.int(), features