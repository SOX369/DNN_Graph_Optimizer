import torch
import torch.nn as nn
from .layers import DynamicConv2d


class VGG_Cifar(nn.Module):
    """
    手动定义的VGG风格网络，接收 graph_config 来构建具体结构
    graph_config 是优化器搜索出来的结果，包含每一层的分裂/融合状态
    """

    def __init__(self, graph_config, num_classes=10):
        super(VGG_Cifar, self).__init__()
        self.features = self._make_layers(graph_config)

        # 动态计算分类器的输入维度
        # 原始代码写死了 512，但优化后的模型最后一层通道数可能变了（比如你的日志里变成了 696）
        # CIFAR-10 图片大小 32x32，经过5个MaxPool后变成 1x1，所以输入维度就是最后一层卷积的输出通道数
        last_channel = 512  # 默认值
        # 倒序遍历配置，找到最后一个卷积层的配置
        for layer_cfg in reversed(graph_config):
            if isinstance(layer_cfg, dict):
                last_channel = layer_cfg['out']
                break

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(last_channel, 512),  # 使用动态获取的 last_channel
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # 展平操作
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        # cfg 格式:List[Dict] -> [{'type': 'conv', 'out': 64, 'groups': 1, 'fused': True}, 'M', ...]
        for layer_cfg in cfg:
            if layer_cfg == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = layer_cfg['out']
                groups = layer_cfg['groups']  # 算子分裂参数
                fused = layer_cfg['fused']  # 算子融合参数

                # 确保 groups 能被 in_channels 和 out_channels 整除
                # 如果搜索算法产生的配置不合法，这里做一个强制修正，防止报错
                if in_channels % groups != 0 or out_channels % groups != 0:
                    groups = 1

                conv2d = DynamicConv2d(in_channels, out_channels, kernel_size=3, padding=1,
                                       groups=groups, fused_relu=fused)
                layers.append(conv2d)

                # 如果没有融合ReLU，则单独添加ReLU层
                if not fused:
                    layers.append(nn.ReLU(inplace=True))

                in_channels = out_channels
        return nn.Sequential(*layers)