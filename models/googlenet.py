import torch
import torch.nn as nn
from .layers import DynamicConv2d


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, cfg_iter):
        super(Inception, self).__init__()

        # 1x1 conv branch
        c1 = next(cfg_iter, {'groups': 1, 'fused': False})
        self.b1 = DynamicConv2d(in_planes, n1x1, kernel_size=1, groups=c1['groups'], fused_relu=c1['fused'])

        # 1x1 -> 3x3 conv branch
        c2_1 = next(cfg_iter, {'groups': 1, 'fused': False})
        c2_2 = next(cfg_iter, {'groups': 1, 'fused': False})
        self.b2 = nn.Sequential(
            DynamicConv2d(in_planes, n3x3red, kernel_size=1, groups=c2_1['groups'], fused_relu=c2_1['fused']),
            DynamicConv2d(n3x3red, n3x3, kernel_size=3, padding=1, groups=c2_2['groups'], fused_relu=c2_2['fused'])
        )

        # 1x1 -> 5x5 conv branch
        c3_1 = next(cfg_iter, {'groups': 1, 'fused': False})
        c3_2 = next(cfg_iter, {'groups': 1, 'fused': False})
        self.b3 = nn.Sequential(
            DynamicConv2d(in_planes, n5x5red, kernel_size=1, groups=c3_1['groups'], fused_relu=c3_1['fused']),
            DynamicConv2d(n5x5red, n5x5, kernel_size=3, padding=1, groups=c3_2['groups'], fused_relu=c3_2['fused'])
            # CIFAR一般用3x3代替5x5，这里为了结构一致用3x3 padding=1
        )

        # 3x3 pool -> 1x1 conv branch
        c4 = next(cfg_iter, {'groups': 1, 'fused': False})
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            DynamicConv2d(in_planes, pool_planes, kernel_size=1, groups=c4['groups'], fused_relu=c4['fused'])
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet_Cifar(nn.Module):
    def __init__(self, graph_config, num_classes=10):
        super(GoogLeNet_Cifar, self).__init__()
        self.cfg_iter = iter(graph_config)

        c0 = next(self.cfg_iter, {'groups': 1, 'fused': False})
        self.pre_layers = DynamicConv2d(3, 192, kernel_size=3, padding=1, groups=c0['groups'], fused_relu=c0['fused'])

        # Inception 模块参数 (in, 1x1, 3x3red, 3x3, 5x5red, 5x5, pool)
        # 每个Inception消耗 6 个配置项
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32, self.cfg_iter)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, self.cfg_iter)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64, self.cfg_iter)
        self.b4 = Inception(516, 160, 112, 224, 24, 64, 64, self.cfg_iter)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64, self.cfg_iter)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64, self.cfg_iter)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, self.cfg_iter)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, self.cfg_iter)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, self.cfg_iter)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out