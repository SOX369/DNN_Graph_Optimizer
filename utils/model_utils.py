from models.vgg import VGG_Cifar
from models.resnet import ResNet_Cifar
from models.googlenet import GoogLeNet_Cifar


def generate_default_config(model_name):
    """
    生成默认的图结构配置 (Baseline)。
    [修复]: 显式包含 'out' (输出通道数) 和 'k' (卷积核大小) 字段，确保 CostModel 能正确计算 Params 和 FLOPs。
    """

    if model_name == 'vgg16':
        # VGG 全是 3x3 卷积
        return [
            {'type': 'conv', 'out': 64, 'groups': 1, 'fused': False, 'k': 3}, 'M',
            {'type': 'conv', 'out': 128, 'groups': 1, 'fused': False, 'k': 3}, 'M',
            {'type': 'conv', 'out': 256, 'groups': 1, 'fused': False, 'k': 3},
            {'type': 'conv', 'out': 256, 'groups': 1, 'fused': False, 'k': 3}, 'M',
            {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False, 'k': 3},
            {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False, 'k': 3}, 'M',
            {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False, 'k': 3},
            {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False, 'k': 3}, 'M',
        ]

    elif model_name in ['resnet50', 'resnext50']:
        # ResNet-50 结构: [3, 4, 6, 3] 个 Bottleneck
        # 每个 Bottleneck 包含: 1x1, 3x3, 1x1 (expansion=4)

        cfg = []
        # Pre-layer (conv1) - 3x3
        cfg.append({'type': 'conv', 'out': 64, 'groups': 1, 'fused': False, 'k': 3})

        # Stages: (num_blocks, base_planes)
        # Expansion is fixed to 4 in Bottleneck
        stages = [
            (3, 64),  # Layer 1
            (4, 128),  # Layer 2
            (6, 256),  # Layer 3
            (3, 512)  # Layer 4
        ]

        for num_blocks, planes in stages:
            for _ in range(num_blocks):
                # Bottleneck internal layers
                # Conv1: 1x1, reduces/keeps dimensions
                cfg.append({'type': 'conv', 'out': planes, 'groups': 1, 'fused': False, 'k': 1})
                # Conv2: 3x3, processes features
                cfg.append({'type': 'conv', 'out': planes, 'groups': 1, 'fused': False, 'k': 3})
                # Conv3: 1x1, expands dimensions (*4)
                cfg.append({'type': 'conv', 'out': planes * 4, 'groups': 1, 'fused': False, 'k': 1})

        return cfg

    elif model_name == 'googlenet':
        # GoogLeNet (Inception) 结构
        # 必须显式定义每个 Inception 模块内部 6 个卷积层的输出通道
        # 参数顺序对应 models/googlenet.py 中的初始化顺序

        cfg = []
        # Pre-layers - 3x3
        cfg.append({'type': 'conv', 'out': 192, 'groups': 1, 'fused': False, 'k': 3})

        # Inception Configs: (n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes)
        inception_params = [
            (64, 96, 128, 16, 32, 32),  # a3
            (128, 128, 192, 32, 96, 64),  # b3
            (192, 96, 208, 16, 48, 64),  # a4
            (160, 112, 224, 24, 64, 64),  # b4
            (128, 128, 256, 24, 64, 64),  # c4
            (112, 144, 288, 32, 64, 64),  # d4
            (256, 160, 320, 32, 128, 128),  # e4
            (256, 160, 320, 32, 128, 128),  # a5
            (384, 192, 384, 48, 128, 128)  # b5
        ]

        for params in inception_params:
            n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes = params
            # 依次添加模块内的卷积配置，并指定卷积核大小
            cfg.append({'type': 'conv', 'out': n1x1, 'groups': 1, 'fused': False, 'k': 1})  # b1 (1x1)
            cfg.append({'type': 'conv', 'out': n3x3red, 'groups': 1, 'fused': False, 'k': 1})  # b2_1 (3x3 reduce -> 1x1)
            cfg.append({'type': 'conv', 'out': n3x3, 'groups': 1, 'fused': False, 'k': 3})  # b2_2 (3x3)
            cfg.append({'type': 'conv', 'out': n5x5red, 'groups': 1, 'fused': False, 'k': 1})  # b3_1 (5x5 reduce -> 1x1)
            # Cifar版 GoogLeNet 通常用 3x3 代替 5x5 以保持一致性
            cfg.append({'type': 'conv', 'out': n5x5, 'groups': 1, 'fused': False, 'k': 3})  # b3_2 (5x5 -> 3x3)
            cfg.append({'type': 'conv', 'out': pool_planes, 'groups': 1, 'fused': False, 'k': 1})  # b4 (pool proj -> 1x1)

        return cfg

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