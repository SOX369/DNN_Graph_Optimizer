import torch
import torch.nn as nn


class DynamicConv2d(nn.Module):
    """
    可变异的卷积层：支持通过 groups 参数进行算子分裂(分组卷积)以降低参数量和显存
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, fused_relu=False):
        super(DynamicConv2d, self).__init__()

        # [新增] 鲁棒性检查：如果输入/输出通道数无法被 groups 整除，强制重置 groups=1
        # 核心修复：解决 ResNet 第一层输入通道为3时，若 groups 为2/4/8 导致的崩溃问题
        if in_channels % groups != 0 or out_channels % groups != 0:
            # 在实际工程中这里可以打印一条 warning，但为了不刷屏日志，我们选择静默修正
            groups = 1

        # 注意：DynamicConv2d 默认 bias=False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.fused_relu = fused_relu  # 标记是否融合了ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.fused_relu:
            x = self.relu(x)
        return x