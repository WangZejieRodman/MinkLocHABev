import torch
import torch.nn as nn
import MinkowskiEngine as ME


class VerticalContextModule(nn.Module):
    """
    创新点 2: 垂直相关性建模 (Vertical Correlation Modeling)
    通过 1D 卷积在 Channel (Z-axis) 维度上进行上下文融合。
    """

    def __init__(self, channels=32, kernel_size=3):
        super(VerticalContextModule, self).__init__()
        # 保持通道数不变，padding保持尺寸不变
        # 这里的 "channels" 实际上是 Conv1d 的 "length"，而 input_channels 是 1
        # 我们把 (N, 32) 看作 N 个长度为 32 的序列
        self.conv1d = nn.Conv1d(in_channels=1,
                                out_channels=1,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                bias=False)
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: ME.SparseTensor):
        # x.F shape: (N, 32)
        # 变换为 (N, 1, 32) 以适配 Conv1d
        feats = x.F.unsqueeze(1)

        # 1D 卷积处理 Z 轴序列
        feats = self.conv1d(feats)
        feats = self.bn(feats)
        feats = self.relu(feats)

        # 变回 (N, 32)
        feats = feats.squeeze(1)

        # 返回新的 SparseTensor (坐标不变，特征更新)
        return ME.SparseTensor(
            feats,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )


class GlobalLayerAttention(nn.Module):
    """
    创新点 1: 动态全局层注意力 (Input-Dependent Global Layer Attention)
    类似于 SE-Block，但作用于 BEV 的 Z-layers (Channels)。
    学习每一帧中哪些高度层是重要的（如侧壁），哪些是噪声（如顶底板）。
    """

    def __init__(self, channels=32, reduction=4):
        super(GlobalLayerAttention, self).__init__()
        self.avg_pool = ME.MinkowskiGlobalAvgPooling()

        # MLP 生成权重
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x: ME.SparseTensor):
        # 1. 全局池化 -> (BatchSize, 32)
        y = self.avg_pool(x)

        # 2. 计算权重 -> (BatchSize, 32)
        w = self.fc(y.F)

        # 3. 构造权重 SparseTensor
        # 注意：这里需要配合 ME 的广播机制，w 需要是全局特征
        w_sparse = ME.SparseTensor(
            w,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager
        )

        # 4. 广播乘法：x * w
        out = self.broadcast_mul(x, w_sparse)
        return out
