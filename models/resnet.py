import torch
import torch.nn as nn
from .layers import DynamicConv2d


class DynamicBottleneck(nn.Module):
    """
    支持动态配置的 Bottleneck 结构
    Expansion = 4
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,
                 cfg_list=None,  # 传入针对该Block内部3个卷积层的配置列表
                 is_resnext=False, base_width=64, cardinality=32):
        super(DynamicBottleneck, self).__init__()

        # 如果没有传入配置，使用默认配置

        # 1x1 conv
        # [Fix History]: 移除了 bias=False，因为 DynamicConv2d 不接受该参数
        self.conv1 = DynamicConv2d(in_planes, planes, kernel_size=1,
                                   groups=cfg_list[0]['groups'] if cfg_list else 1,
                                   fused_relu=cfg_list[0]['fused'] if cfg_list else False)

        # 3x3 conv
        groups = cfg_list[1]['groups'] if cfg_list else (cardinality if is_resnext else 1)
        self.conv2 = DynamicConv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
                                   groups=groups,
                                   fused_relu=cfg_list[1]['fused'] if cfg_list else False)

        # 1x1 expansion
        self.conv3 = DynamicConv2d(planes, self.expansion * planes, kernel_size=1,
                                   groups=cfg_list[2]['groups'] if cfg_list else 1,
                                   fused_relu=cfg_list[2]['fused'] if cfg_list else False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # [Critical Fix]: 将 In-place 加法 (+=) 改为标准加法
        # 原因：若 conv3 融合了 ReLU (inplace=True)，+= 会修改 ReLU 的输出，
        # 导致 PyTorch 反向传播时报错 "modified by an inplace operation"。
        out = out + self.shortcut(x)

        out = self.final_relu(out)
        return out


class ResNet_Cifar(nn.Module):
    """
    通用的 ResNet/ResNeXt 框架，通过 graph_config 控制具体算子
    """

    def __init__(self, graph_config, depth=50, num_classes=10, is_resnext=False):
        super(ResNet_Cifar, self).__init__()
        self.in_planes = 64
        self.is_resnext = is_resnext

        # 解析 graph_config
        self.cfg_iter = iter(graph_config)

        # CIFAR-10 初始层
        # 使用 .get() 防止 KeyError
        first_cfg = next(self.cfg_iter, {'groups': 1, 'fused': False})
        self.conv1 = DynamicConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                   groups=first_cfg.get('groups', 1),
                                   fused_relu=first_cfg.get('fused', False))

        # ResNet-50 结构: [3, 4, 6, 3]
        if depth == 50:
            num_blocks = [3, 4, 6, 3]
        else:
            num_blocks = [3, 4, 6, 3]  # 默认 50

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * DynamicBottleneck.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # 提取配置
            block_cfgs = []
            try:
                block_cfgs.append(next(self.cfg_iter))
                block_cfgs.append(next(self.cfg_iter))
                block_cfgs.append(next(self.cfg_iter))
            except StopIteration:
                block_cfgs = [{'groups': 1, 'fused': False}] * 3

            layers.append(DynamicBottleneck(self.in_planes, planes, stride,
                                            cfg_list=block_cfgs, is_resnext=self.is_resnext))
            self.in_planes = planes * DynamicBottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out