import random
import copy

class GraphMutator:
    """
    图变换算子集合。对应论文中的 Omega 空间。
    包含：Split (分裂), Fusion (融合), Resizing (通道调整)
    """

    def __init__(self, base_config):
        self.base_config = base_config

    def perturb(self, current_config):
        """
        随机选择一种策略对当前图结构进行变异
        """
        new_config = copy.deepcopy(current_config)

        # 随机选择一个卷积层进行变异
        conv_indices = [i for i, x in enumerate(new_config) if x != 'M']
        if not conv_indices:
            return new_config

        target_idx = random.choice(conv_indices)
        layer = new_config[target_idx]

        mutation_type = random.choice(['split', 'fuse', 'channel_scale'])

        if mutation_type == 'split':
            # 算子分裂：增加分组卷积的组数 (1 -> 2 -> 4 ...)
            # 创新点：通过分裂降低参数量，适应显存限制
            current_groups = layer['groups']
            choices = [1, 2, 4, 8]
            # 确保 in_channels 和 out_channels 能被 groups 整除 (这里简化处理，假设都可以)
            layer['groups'] = random.choice(choices)

        elif mutation_type == 'fuse':
            # 算子融合：切换 Conv+ReLU 的融合状态
            # 创新点：减少内核启动开销
            layer['fused'] = not layer['fused']

        elif mutation_type == 'channel_scale':
            # 通道剪枝/扩容：微调通道数
            scale = random.choice([0.8, 1.0, 1.2])
            layer['out'] = int(layer['out'] * scale)
            # 保证至少有16个通道且为8的倍数以便于硬件对齐
            layer['out'] = max(16, (layer['out'] // 8) * 8)

        return new_config