import torch
import torch.nn as nn
import MinkowskiEngine as ME

# ==========================================
# 导入新模块
# ==========================================
from models.layers.bev_enhancers import VerticalContextModule, GlobalLayerAttention


class MinkBottleneck(nn.Module):
    """
    稀疏 2D Bottleneck 模块 (保持原样)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dimension=2):
        super(MinkBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, mid_channels, kernel_size=1, dimension=dimension, bias=False),
            ME.MinkowskiBatchNorm(mid_channels),
            ME.MinkowskiReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(mid_channels, mid_channels, kernel_size=kernel_size, dimension=dimension,
                                    bias=False),
            ME.MinkowskiBatchNorm(mid_channels),
            ME.MinkowskiReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(mid_channels, out_channels, kernel_size=1, dimension=dimension, bias=False),
            ME.MinkowskiBatchNorm(out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, dimension=dimension, bias=False),
                ME.MinkowskiBatchNorm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return self.relu(out)


class MinkBEVBackbone(nn.Module):
    """
    MinkowskiEngine 版 BEVNet Backbone (修改版)
    集成了:
    1. Vertical Context Module
    2. Global Layer Attention
    3. Sparse Local Attention
    """

    def __init__(self, in_channels=32, out_channels=256, dimension=2):
        super(MinkBEVBackbone, self).__init__()

        # ==========================================
        # 创新点模块初始化
        # ==========================================
        # 1. 垂直混合 (针对 32 通道输入)
        self.vertical_mixer = VerticalContextModule(channels=in_channels)

        # 2. 层注意力 (针对 32 通道输入)
        self.layer_attention = GlobalLayerAttention(channels=in_channels)

        # ==========================================
        # 原始 Backbone 结构 (保持骨架)
        # ==========================================
        # 即使传入 out_channels，我们也需要定义中间层的通道增长策略
        c1 = in_channels * 2  # 64
        c2 = in_channels * 4  # 128
        c3 = in_channels * 8  # 256
        c4 = out_channels

        # Block 1 (Downsample 2x): 32 -> 64
        self.block1 = nn.Sequential(
            MinkBottleneck(in_channels, c1, kernel_size=11, dimension=dimension),
            ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=dimension),
            MinkBottleneck(c1, c1, kernel_size=11, dimension=dimension)
        )

        # Block 2 (Downsample 4x): 64 -> 128
        self.block2 = nn.Sequential(
            MinkBottleneck(c1, c2, kernel_size=7, dimension=dimension),
            ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=dimension),
            MinkBottleneck(c2, c2, kernel_size=7, dimension=dimension)
        )

        # Block 3 (Downsample 8x): 128 -> 256
        self.block3 = nn.Sequential(
            MinkBottleneck(c2, c3, kernel_size=5, dimension=dimension),
            ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=dimension),
            MinkBottleneck(c3, c3, kernel_size=5, dimension=dimension)
        )

        # Block 4 (Final Features): 256 -> out_channels
        self.block4 = ME.MinkowskiConvolution(c3, c4, kernel_size=3, dimension=dimension, bias=True)

    def forward(self, x):
        # ==========================================
        # 阶段 1: 输入增强 (Innovation 1 & 2)
        # ==========================================
        # 原始输入 x: (Batch, 32, H, W) 的稀疏表示

        # 1.1 恢复垂直相关性 (让 Layer i 看到 Layer i-1/i+1)
        x = self.vertical_mixer(x)

        # 1.2 动态层加权 (抑制顶底板噪声层)
        x = self.layer_attention(x)

        # ==========================================
        # 阶段 2: 浅层特征提取 (Raw Feature Extraction)
        # ==========================================
        x = self.block1(x)  # -> 64 channels, /2
        x = self.block2(x)  # -> 128 channels, /4
        x = self.block3(x)  # -> 256 channels, /8

        # ==========================================
        # 阶段 3: 输出对齐
        # ==========================================
        x = self.block4(x)

        return x
