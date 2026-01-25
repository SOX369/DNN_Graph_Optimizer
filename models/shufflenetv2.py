import torch
import torch.nn as nn
from .layers import DynamicConv2d


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # Reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # Transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # Flatten
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, cfg_iter):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        # ShuffleNet V2 output channels are split equally
        branch_features = oup // 2
        # [Constraint]: If stride=1, inp must equal oup for split/concat to work
        assert (self.stride != 1) or (inp == branch_features << 1), \
            f"Stride=1 requires inp({inp}) == oup({oup})"

        # Helper to get config
        def get_next_cfg():
            # 这里虽然加了默认值，但如果 cfg_iter 吐出的是 'M' 字符串，还是会报错
            # 所以关键在外部过滤
            return next(cfg_iter, {'groups': 1, 'fused': False})

        if self.stride > 1:
            # [Branch 1] (Left): 3x3 DW -> 1x1 PW
            # 3x3 DW Conv
            c1 = get_next_cfg()
            self.branch1_conv1 = DynamicConv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1,
                                               groups=c1.get('groups', 1), fused_relu=False)
            self.branch1_bn1 = nn.BatchNorm2d(inp)

            # 1x1 PW Conv
            c2 = get_next_cfg()
            self.branch1_conv2 = DynamicConv2d(inp, branch_features, kernel_size=1, stride=1, padding=0,
                                               groups=c2.get('groups', 1), fused_relu=c2.get('fused', False))
            self.branch1_bn2 = nn.BatchNorm2d(branch_features)

            # [Branch 2] (Right): 1x1 PW -> 3x3 DW -> 1x1 PW
            c3 = get_next_cfg()
            self.branch2_conv1 = DynamicConv2d(inp, branch_features, kernel_size=1, stride=1, padding=0,
                                               groups=c3.get('groups', 1), fused_relu=c3.get('fused', False))
            self.branch2_bn1 = nn.BatchNorm2d(branch_features)

            c4 = get_next_cfg()
            self.branch2_conv2 = DynamicConv2d(branch_features, branch_features, kernel_size=3, stride=self.stride,
                                               padding=1,
                                               groups=c4.get('groups', 1), fused_relu=False)
            self.branch2_bn2 = nn.BatchNorm2d(branch_features)

            c5 = get_next_cfg()
            self.branch2_conv3 = DynamicConv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0,
                                               groups=c5.get('groups', 1), fused_relu=c5.get('fused', False))
            self.branch2_bn3 = nn.BatchNorm2d(branch_features)

        else:
            # Stride = 1: Input split (half identity, half processed)
            # Only define Branch 2 (Right)
            c3 = get_next_cfg()
            self.branch2_conv1 = DynamicConv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0,
                                               groups=c3.get('groups', 1), fused_relu=c3.get('fused', False))
            self.branch2_bn1 = nn.BatchNorm2d(branch_features)

            c4 = get_next_cfg()
            self.branch2_conv2 = DynamicConv2d(branch_features, branch_features, kernel_size=3, stride=self.stride,
                                               padding=1,
                                               groups=c4.get('groups', 1), fused_relu=False)
            self.branch2_bn2 = nn.BatchNorm2d(branch_features)

            c5 = get_next_cfg()
            self.branch2_conv3 = DynamicConv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0,
                                               groups=c5.get('groups', 1), fused_relu=c5.get('fused', False))
            self.branch2_bn3 = nn.BatchNorm2d(branch_features)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            # x1 is Identity
            out2 = self.branch2_conv1(x2)
            out2 = self.branch2_bn1(out2)
            out2 = self.branch2_conv2(out2)  # DW
            out2 = self.branch2_bn2(out2)
            out2 = self.branch2_conv3(out2)
            out2 = torch.nn.functional.relu(self.branch2_bn3(out2))

            out = torch.cat((x1, out2), dim=1)
        else:
            # Branch 1
            out1 = self.branch1_conv1(x)  # DW
            out1 = self.branch1_bn1(out1)
            out1 = self.branch1_conv2(out1)
            out1 = torch.nn.functional.relu(self.branch1_bn2(out1))

            # Branch 2
            out2 = self.branch2_conv1(x)
            out2 = torch.nn.functional.relu(self.branch2_bn1(out2))
            out2 = self.branch2_conv2(out2)  # DW
            out2 = self.branch2_bn2(out2)
            out2 = self.branch2_conv3(out2)
            out2 = torch.nn.functional.relu(self.branch2_bn3(out2))

            out = torch.cat((out1, out2), dim=1)

        return channel_shuffle(out, 2)


class ShuffleNetV2_Cifar(nn.Module):
    def __init__(self, graph_config, num_classes=10):
        super(ShuffleNetV2_Cifar, self).__init__()

        # [关键修复]: 过滤掉 graph_config 中的 'M' 字符串
        # 这样模型构建时只看到字典配置，而 cost_model 依然可以看到 'M' 用于计算
        self.cfg_iter = iter([x for x in graph_config if isinstance(x, dict)])

        # Scale 0.5 Configuration (Lightweight)
        stages_repeats = [4, 8, 4]
        # [Topology Fix]: Stage 1 output 24 -> 24 (Stride 1) matches input
        stages_out_channels = [24, 24, 48, 96, 1024]

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 integers')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 integers')

        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]

        # Initial Conv
        c0 = next(self.cfg_iter, {'groups': 1, 'fused': False})
        self.conv1 = nn.Sequential(
            DynamicConv2d(input_channels, output_channels, 3, 1, 1,
                          groups=c0.get('groups', 1), fused_relu=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.features = []
        for i in range(len(stages_repeats)):
            stride = 2 if i > 0 else 1
            output_channels = self._stage_out_channels[i + 1]

            seq = [InvertedResidual(input_channels, output_channels, stride, self.cfg_iter)]
            input_channels = output_channels

            for _ in range(stages_repeats[i] - 1):
                seq.append(InvertedResidual(input_channels, output_channels, 1, self.cfg_iter))

            self.features.append(nn.Sequential(*seq))

        self.features = nn.Sequential(*self.features)

        # Final Conv
        output_channels = self._stage_out_channels[-1]
        c_last = next(self.cfg_iter, {'groups': 1, 'fused': False})
        self.conv_last = nn.Sequential(
            DynamicConv2d(input_channels, output_channels, 1, 1, 0,
                          groups=c_last.get('groups', 1), fused_relu=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.conv_last(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out