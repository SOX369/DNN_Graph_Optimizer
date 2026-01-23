import torch
import torch.nn as nn
from .layers import DynamicConv2d


class DynamicInvertedResidual(nn.Module):
    """
    MobileNetV2 的 Inverted Residual Block
    结构: Expand (1x1) -> Depthwise (3x3) -> Project (1x1)
    """

    def __init__(self, in_planes, out_planes, stride, expand_ratio, cfg_list=None):
        super(DynamicInvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers = []
        cfg_idx = 0

        # 1. Expansion Layer (1x1) - 仅当 expand_ratio != 1 时存在
        if expand_ratio != 1:
            layers.append(DynamicConv2d(in_planes, hidden_dim, kernel_size=1, stride=1, padding=0,
                                        groups=cfg_list[cfg_idx]['groups'] if cfg_list else 1,
                                        fused_relu=cfg_list[cfg_idx]['fused'] if cfg_list else True))
            cfg_idx += 1

        # 2. Depthwise Convolution (3x3)
        # 默认 MobileNetV2 的 groups = hidden_dim (即 Depthwise)
        # 优化器可能会尝试改变这个 groups
        default_dw_groups = hidden_dim
        layers.append(DynamicConv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                                    groups=cfg_list[cfg_idx]['groups'] if cfg_list else default_dw_groups,
                                    fused_relu=cfg_list[cfg_idx]['fused'] if cfg_list else True))
        cfg_idx += 1

        # 3. Projection Layer (1x1)
        # 线性激活 (无 ReLU)
        layers.append(DynamicConv2d(hidden_dim, out_planes, kernel_size=1, stride=1, padding=0,
                                    groups=cfg_list[cfg_idx]['groups'] if cfg_list else 1,
                                    fused_relu=cfg_list[cfg_idx]['fused'] if cfg_list else False))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_Cifar(nn.Module):
    """
    MobileNetV2 (适配 CIFAR-10)
    """

    def __init__(self, graph_config, num_classes=10):
        super(MobileNetV2_Cifar, self).__init__()

        self.cfg_iter = iter(graph_config)

        # Initial Conv
        first_cfg = next(self.cfg_iter, {'groups': 1, 'fused': False})
        self.conv1 = DynamicConv2d(3, 32, kernel_size=3, stride=1, padding=1,
                                   groups=first_cfg.get('groups', 1),
                                   fused_relu=first_cfg.get('fused', False))

        # MobileNetV2 配置表: [t, c, n, s]
        # t: expansion, c: output channels, n: repeated times, s: stride
        self.interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 构建 Inverted Residual Blocks
        input_channel = 32
        features = []
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1

                # 计算该 Block 包含多少个卷积层
                num_layers = 2 if t == 1 else 3
                block_cfgs = []
                try:
                    for _ in range(num_layers):
                        block_cfgs.append(next(self.cfg_iter))
                except StopIteration:
                    block_cfgs = [{'groups': 1, 'fused': False}] * num_layers

                features.append(
                    DynamicInvertedResidual(input_channel, output_channel, stride, expand_ratio=t, cfg_list=block_cfgs))
                input_channel = output_channel

        self.features = nn.Sequential(*features)

        # Last Conv 1x1
        last_cfg = next(self.cfg_iter, {'groups': 1, 'fused': False})
        self.conv_last = DynamicConv2d(input_channel, 1280, kernel_size=1, stride=1, padding=0,
                                       groups=last_cfg.get('groups', 1),
                                       fused_relu=last_cfg.get('fused', True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.conv_last(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out