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
        # cfg_list[0]: 1x1 conv, cfg_list[1]: 3x3 conv, cfg_list[2]: 1x1 expansion

        # 1x1 conv
        self.conv1 = DynamicConv2d(in_planes, planes, kernel_size=1, bias=False,
                                   groups=cfg_list[0]['groups'] if cfg_list else 1,
                                   fused_relu=cfg_list[0]['fused'] if cfg_list else False)

        # 3x3 conv (核心优化点: ResNeXt在这里默认使用分组卷积)
        groups = cfg_list[1]['groups'] if cfg_list else (cardinality if is_resnext else 1)
        # ResNeXt width调整逻辑略杂，这里简化：由外部cfg控制groups即可模拟ResNeXt特性
        self.conv2 = DynamicConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                   groups=groups,
                                   fused_relu=cfg_list[1]['fused'] if cfg_list else False)

        # 1x1 expansion
        self.conv3 = DynamicConv2d(planes, self.expansion * planes, kernel_size=1, bias=False,
                                   groups=cfg_list[2]['groups'] if cfg_list else 1,
                                   fused_relu=cfg_list[2]['fused'] if cfg_list else False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # 如果 conv3 没有融合ReLU，或者是最后相加后的ReLU，这里单独处理
        # 简化处理：C-DGOSA论文中融合通常指 Conv-BN-ReLU。
        # 这里为了保持残差结构完整性，最后的ReLU通常不融合在conv3里，而是在add之后
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
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

        # 解析 graph_config。因为它是一个扁平列表，我们需要按层取出
        self.cfg_iter = iter(graph_config)

        # CIFAR-10 初始层：使用 3x3 conv, stride=1, 无池化 (适配 32x32 输入)
        # 这一层通常不参与搜索，或者作为配置的第一项
        first_cfg = next(self.cfg_iter, {'groups': 1, 'fused': False})
        self.conv1 = DynamicConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                   groups=first_cfg['groups'], fused_relu=first_cfg['fused'])

        # ResNet-50 结构: [3, 4, 6, 3] 个 Bottleneck
        # 每个 Bottleneck 消耗 3 个配置项 (conv1, conv2, conv3)
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
            # 从全局配置中提取当前 Block 所需的 3 个卷积层配置
            block_cfgs = []
            try:
                block_cfgs.append(next(self.cfg_iter))
                block_cfgs.append(next(self.cfg_iter))
                block_cfgs.append(next(self.cfg_iter))
            except StopIteration:
                # 容错：如果配置不够，用默认值
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