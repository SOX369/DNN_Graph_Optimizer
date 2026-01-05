from models.vgg import VGG_Cifar
from models.resnet import ResNet_Cifar
from models.googlenet import GoogLeNet_Cifar


def generate_default_config(model_name):
    """
    根据模型名称生成默认的 graph_config (全为标准卷积)
    """
    base_item = {'groups': 1, 'fused': False, 'out': 0}  # out在这里主要占位，ResNet等在代码里写死了通道数

    if model_name == 'vgg16':
        # VGG16 (Cifar版) 约13个卷积层
        return [
            {'type': 'conv', 'out': 64, 'groups': 1, 'fused': False}, 'M',
            {'type': 'conv', 'out': 128, 'groups': 1, 'fused': False}, 'M',
            {'type': 'conv', 'out': 256, 'groups': 1, 'fused': False},
            {'type': 'conv', 'out': 256, 'groups': 1, 'fused': False}, 'M',
            {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False},
            {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False}, 'M',
            {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False},
            {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False}, 'M',
        ]

    elif model_name == 'resnet50' or model_name == 'resnext50':
        # 1 (pre) + 3*3 (layer1) + 4*3 (layer2) + 6*3 (layer3) + 3*3 (layer4) = 1 + 9 + 12 + 18 + 9 = 49 个卷积配置
        # out 字段在 ResNet 中由结构决定，这里只存 groups 和 fused
        count = 1 + (3 + 4 + 6 + 3) * 3

        # 对于 ResNeXt, 默认 groups 可以设为 32，但为了作为 Baseline 对比，初始设为 1，让优化器去搜索
        return [{'groups': 1, 'fused': False, 'type': 'conv'} for _ in range(count)]

    elif model_name == 'googlenet':
        # 1 (pre) + 9 * 6 (Inception modules) = 55 个卷积配置
        count = 1 + 9 * 6
        return [{'groups': 1, 'fused': False, 'type': 'conv'} for _ in range(count)]

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_model(model_name, graph_config):
    if model_name == 'vgg16':
        return VGG_Cifar(graph_config)
    elif model_name == 'resnet50':
        return ResNet_Cifar(graph_config, depth=50, is_resnext=False)
    elif model_name == 'resnext50':
        return ResNet_Cifar(graph_config, depth=50, is_resnext=True)
    elif model_name == 'googlenet':
        return GoogLeNet_Cifar(graph_config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")