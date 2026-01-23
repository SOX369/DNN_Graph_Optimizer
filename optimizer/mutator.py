# optimizer/mutator.py

import random
import copy


class GraphMutator:
    def __init__(self, base_config):
        self.base_config = base_config

    def perturb(self, current_config):
        new_config = copy.deepcopy(current_config)
        conv_indices = [i for i, x in enumerate(new_config) if isinstance(x, dict)]

        if not conv_indices:
            return new_config

        target_idx = random.choice(conv_indices)
        layer = new_config[target_idx]

        available_mutations = ['split', 'fuse']
        if 'out' in layer:
            available_mutations.append('channel_scale')

        mutation_type = random.choice(available_mutations)

        if mutation_type == 'split':
            current_out = layer.get('out', 64)

            # [修正]: 移除 current_out (Depthwise) 选项。
            # 限制 max groups 为 16 (或者 8, 32)。
            # 这样模型就不能变得极度稀疏，从而控制 Memory Reduction 的比例。
            # 例如: 从 groups=1 到 groups=16，参数减少 94%，比 Depthwise 的 99% 要温和一点。
            # 如果想要更温和 (如 50-70%)，可以只允许 [1, 2, 4, 8]。
            potential_choices = [1, 2, 4, 8, 16]

            # 过滤非法值
            valid_choices = [g for g in potential_choices if g <= current_out and current_out % g == 0]

            if valid_choices:
                layer['groups'] = random.choice(valid_choices)

        elif mutation_type == 'fuse':
            current_fused = layer.get('fused', False)
            layer['fused'] = not current_fused

        elif mutation_type == 'channel_scale':
            # [修正]: 限制剪枝的下限。防止通道数被剪得太少导致显存几乎为0。
            # 比如不允许小于原始通道的 50%。
            scale = random.choice([0.8, 1.0, 1.2])  # 去掉太小的比例
            current_out = layer['out']
            new_out = int(current_out * scale)
            layer['out'] = max(32, (new_out // 8) * 8)  # 提高最小通道限制

            # 检查 groups 合法性
            current_groups = layer.get('groups', 1)
            if current_groups > layer['out'] or layer['out'] % current_groups != 0:
                layer['groups'] = 1  # 默认回退到标准卷积，避免随机变成 Depthwise

        return new_config