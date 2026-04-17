import torch
import torch.nn as nn
import torch.nn.functional as F
import cbam


class ChannelAttention(nn.Module):
    """1D通道注意力模块（适配时间序列）"""

    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 1D自适应平均池化（输出通道数不变，时间步=1）
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 1D自适应最大池化

        # 共享MLP
        self.fc = nn.Sequential(
            nn.Conv1d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, channels, time]（3维输入）
        avg_out = self.fc(self.avg_pool(x))  # [batch, channels, 1]
        max_out = self.fc(self.max_pool(x))  # [batch, channels, 1]
        out = avg_out + max_out
        return self.sigmoid(out)  # [batch, channels, 1]


class SpatialAttention(nn.Module):
    """1D空间注意力模块（适配时间序列，关注重要时间步）"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # 1D卷积（替代原2D卷积，捕捉时间维度的依赖）
        self.conv1 = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, channels, time]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道维度平均 → [batch, 1, time]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道维度最大 → [batch, 1, time]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [batch, 2, time]（2个输入通道）
        x_out = self.conv1(x_cat)  # [batch, 1, time]
        return self.sigmoid(x_out)  # [batch, 1, time]


class CBAM(nn.Module):
    """1D-CBAM注意力模块（通道注意力+空间注意力，适配时间序列）"""

    def __init__(self, gate_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialAttention(kernel_size)

    def forward(self, x):
        # x: [batch, channels, time]（3维输入，来自并行卷积分支）
        channel_att = self.ChannelGate(x)  # [batch, channels, 1]
        x_out = x * channel_att  # 通道注意力加权（广播到时间维度）

        spatial_att = self.SpatialGate(x_out)  # [batch, 1, time]
        x_out = x_out * spatial_att  # 空间（时间）注意力加权

        return x_out  # [batch, channels, time]（输出维度不变，与输入兼容）
