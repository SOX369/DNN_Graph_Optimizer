import torch
import torch.nn as nn
from .layers import DynamicConv2d


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, cfg_iter):
        super(Inception, self).__init__()

        # 辅助函数：安全地从迭代器获取下一个配置字典
        def get_next_cfg():
            return next(cfg_iter, {'groups': 1, 'fused': False})

        # 1x1 conv branch
        c1 = get_next_cfg()
        self.b1 = DynamicConv2d(in_planes, n1x1, kernel_size=1,
                                groups=c1.get('groups', 1), fused_relu=c1.get('fused', False))

        # 1x1 -> 3x3 conv branch
        c2_1 = get_next_cfg()
        c2_2 = get_next_cfg()
        self.b2 = nn.Sequential(
            DynamicConv2d(in_planes, n3x3red, kernel_size=1,
                          groups=c2_1.get('groups', 1), fused_relu=c2_1.get('fused', False)),
            DynamicConv2d(n3x3red, n3x3, kernel_size=3, padding=1,
                          groups=c2_2.get('groups', 1), fused_relu=c2_2.get('fused', False))
        )

        # 1x1 -> 5x5 conv branch
        # CIFAR 版通常用 3x3 padding=1 代替 5x5 以减少计算量，同时保持特征图大小一致
        c3_1 = get_next_cfg()
        c3_2 = get_next_cfg()
        self.b3 = nn.Sequential(
            DynamicConv2d(in_planes, n5x5red, kernel_size=1,
                          groups=c3_1.get('groups', 1), fused_relu=c3_1.get('fused', False)),
            DynamicConv2d(n5x5red, n5x5, kernel_size=3, padding=1,
                          groups=c3_2.get('groups', 1), fused_relu=c3_2.get('fused', False))
        )

        # 3x3 pool -> 1x1 conv branch
        c4 = get_next_cfg()
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            DynamicConv2d(in_planes, pool_planes, kernel_size=1,
                          groups=c4.get('groups', 1), fused_relu=c4.get('fused', False))
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

        # 初始层
        c0 = next(self.cfg_iter, {'groups': 1, 'fused': False})
        self.pre_layers = DynamicConv2d(3, 192, kernel_size=3, padding=1,
                                        groups=c0.get('groups', 1), fused_relu=c0.get('fused', False))

        # Inception 模块参数 (in, 1x1, 3x3red, 3x3, 5x5red, 5x5, pool)
        # 维度检查:
        # a3 out: 64+128+32+32 = 256
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32, self.cfg_iter)
        # b3 out: 128+192+96+64 = 480
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, self.cfg_iter)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # a4 out: 192+208+48+64 = 512
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64, self.cfg_iter)

        # [Fix]: 修正这里的输入通道数，从 516 改为 512，与 a4 的输出匹配
        # b4 out: 160+224+64+64 = 512
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64, self.cfg_iter)

        # c4 out: 128+256+64+64 = 512
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64, self.cfg_iter)
        # d4 out: 112+288+64+64 = 528
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64, self.cfg_iter)
        # e4 out: 256+320+128+128 = 832
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, self.cfg_iter)

        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # a5 out: 256+320+128+128 = 832
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, self.cfg_iter)
        # b5 out: 384+384+128+128 = 1024
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
        out = self.maxpool2(out)  # 修正：使用 maxpool2，避免引用混乱
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

