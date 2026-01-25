from models.vgg import VGG_Cifar
from models.resnet import ResNet_Cifar
from models.googlenet import GoogLeNet_Cifar
from models.mobilenetv2 import MobileNetV2_Cifar
from models.shufflenetv2 import ShuffleNetV2_Cifar


def generate_default_config(model_name):
    """
    生成默认的图结构配置 (Baseline)。
    """
    if model_name == 'vgg16':
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

    elif model_name == 'resnet18':
        cfg = []
        cfg.append({'type': 'conv', 'out': 64, 'groups': 1, 'fused': False, 'k': 3})
        stages = [(2, 64), (2, 128), (2, 256), (2, 512)]
        for num_blocks, planes in stages:
            for _ in range(num_blocks):
                cfg.append({'type': 'conv', 'out': planes, 'groups': 1, 'fused': True, 'k': 3})
                cfg.append({'type': 'conv', 'out': planes, 'groups': 1, 'fused': False, 'k': 3})
        return cfg

    elif model_name in ['resnet50', 'resnext50']:
        cfg = []
        initial_groups = 1
        cfg.append({'type': 'conv', 'out': 64, 'groups': 1, 'fused': False, 'k': 3})
        stages = [(3, 64), (4, 128), (6, 256), (3, 512)]
        for num_blocks, planes in stages:
            for _ in range(num_blocks):
                cfg.append({'type': 'conv', 'out': planes, 'groups': 1, 'fused': False, 'k': 1})
                cfg.append({'type': 'conv', 'out': planes, 'groups': initial_groups, 'fused': False, 'k': 3})
                cfg.append({'type': 'conv', 'out': planes * 4, 'groups': 1, 'fused': False, 'k': 1})
        return cfg

    elif model_name == 'mobilenetv2':
        cfg = []
        cfg.append({'type': 'conv', 'out': 32, 'groups': 1, 'fused': False, 'k': 3})
        settings = [
            [1, 16, 1, 1], [6, 24, 2, 1], [6, 32, 3, 2], [6, 64, 4, 2],
            [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
        ]
        input_channel = 32
        for t, c, n, s in settings:
            output_channel = c
            for i in range(n):
                hidden_dim = int(round(input_channel * t))
                if t != 1:
                    cfg.append({'type': 'conv', 'out': hidden_dim, 'groups': 1, 'fused': True, 'k': 1})
                cfg.append({'type': 'conv', 'out': hidden_dim, 'groups': 1, 'fused': True, 'k': 3})
                cfg.append({'type': 'conv', 'out': output_channel, 'groups': 1, 'fused': False, 'k': 1})
                input_channel = output_channel
        cfg.append({'type': 'conv', 'out': 1280, 'groups': 1, 'fused': True, 'k': 1})
        return cfg

    elif model_name == 'googlenet':
        cfg = []
        cfg.append({'type': 'conv', 'out': 192, 'groups': 1, 'fused': False, 'k': 3})
        inception_params = [
            (64, 96, 128, 16, 32, 32), (128, 128, 192, 32, 96, 64),
            (192, 96, 208, 16, 48, 64), (160, 112, 224, 24, 64, 64),
            (128, 128, 256, 24, 64, 64), (112, 144, 288, 32, 64, 64),
            (256, 160, 320, 32, 128, 128), (256, 160, 320, 32, 128, 128),
            (384, 192, 384, 48, 128, 128)
        ]
        for params in inception_params:
            n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes = params
            cfg.append({'type': 'conv', 'out': n1x1, 'groups': 1, 'fused': False, 'k': 1})
            cfg.append({'type': 'conv', 'out': n3x3red, 'groups': 1, 'fused': False, 'k': 1})
            cfg.append({'type': 'conv', 'out': n3x3, 'groups': 1, 'fused': False, 'k': 3})
            cfg.append({'type': 'conv', 'out': n5x5red, 'groups': 1, 'fused': False, 'k': 1})
            cfg.append({'type': 'conv', 'out': n5x5, 'groups': 1, 'fused': False, 'k': 3})
            cfg.append({'type': 'conv', 'out': pool_planes, 'groups': 1, 'fused': False, 'k': 1})
        return cfg

    elif model_name == 'shufflenetv2':
        cfg = []
        # Config: CIFAR adapted [24, 24, 48, 96, 1024]
        # Match the stages in shufflenetv2.py
        stages_repeats = [4, 8, 4]
        stages_out_channels = [24, 24, 48, 96, 1024]

        # Initial Conv
        cfg.append({'type': 'conv', 'in': 3, 'out': 24, 'groups': 1, 'fused': False, 'k': 3})

        input_c = 24

        for i in range(3):
            stride = 2 if i > 0 else 1
            output_c = stages_out_channels[i + 1]

            if stride == 2:
                # Stride 2: Output doubles
                # Branch 1 (Left)
                cfg.append({'type': 'conv', 'in': input_c, 'out': input_c, 'groups': input_c, 'fused': False, 'k': 3})
                cfg.append({'type': 'conv', 'in': input_c, 'out': output_c // 2, 'groups': 1, 'fused': True, 'k': 1})
                # Branch 2 (Right)
                cfg.append({'type': 'conv', 'in': input_c, 'out': output_c // 2, 'groups': 1, 'fused': True, 'k': 1})
                cfg.append({'type': 'conv', 'in': output_c // 2, 'out': output_c // 2, 'groups': output_c // 2, 'fused': False, 'k': 3})
                cfg.append({'type': 'conv', 'in': output_c // 2, 'out': output_c // 2, 'groups': 1, 'fused': True, 'k': 1})
            else:
                # Stride 1: Channels must match (24 -> 24)
                branch_c = output_c // 2
                # Branch 2 Only
                cfg.append({'type': 'conv', 'in': branch_c, 'out': branch_c, 'groups': 1, 'fused': True, 'k': 1})
                cfg.append({'type': 'conv', 'in': branch_c, 'out': branch_c, 'groups': branch_c, 'fused': False, 'k': 3})
                cfg.append({'type': 'conv', 'in': branch_c, 'out': branch_c, 'groups': 1, 'fused': True, 'k': 1})

            input_c = output_c

            for _ in range(stages_repeats[i] - 1):
                branch_c = input_c // 2
                cfg.append({'type': 'conv', 'in': branch_c, 'out': branch_c, 'groups': 1, 'fused': True, 'k': 1})
                cfg.append({'type': 'conv', 'in': branch_c, 'out': branch_c, 'groups': branch_c, 'fused': False, 'k': 3})
                cfg.append({'type': 'conv', 'in': branch_c, 'out': branch_c, 'groups': 1, 'fused': True, 'k': 1})

            if stride == 2:
                cfg.append('M')

        # Final Conv
        cfg.append({'type': 'conv', 'in': 96, 'out': 1024, 'groups': 1, 'fused': True, 'k': 1})

        return cfg

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_model(model_name, graph_config):
    if model_name == 'vgg16':
        return VGG_Cifar(graph_config)
    elif model_name == 'resnet18':
        return ResNet_Cifar(graph_config, depth=18, is_resnext=False)
    elif model_name == 'resnet50':
        return ResNet_Cifar(graph_config, depth=50, is_resnext=False)
    elif model_name == 'resnext50':
        return ResNet_Cifar(graph_config, depth=50, is_resnext=True)
    elif model_name == 'mobilenetv2':
        return MobileNetV2_Cifar(graph_config)
    elif model_name == 'googlenet':
        return GoogLeNet_Cifar(graph_config)
    elif model_name == 'shufflenetv2':
        return ShuffleNetV2_Cifar(graph_config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")