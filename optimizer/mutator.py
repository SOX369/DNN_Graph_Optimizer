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

        # 1. 筛选出所有的卷积层配置项
        # 增加 isinstance(x, dict) 判断，防止处理 'M' 字符串或其他非字典配置
        conv_indices = [i for i, x in enumerate(new_config) if isinstance(x, dict)]

        if not conv_indices:
            return new_config

        # 随机选择一个层
        target_idx = random.choice(conv_indices)
        layer = new_config[target_idx]

        # 2. 动态构建可用的变异类型列表
        # 基础变异：分裂、融合
        available_mutations = ['split', 'fuse']

        # 只有当层配置中明确包含 'out' 字段时，才允许通道剪枝/扩容变异
        # 这样可以兼容 ResNet/GoogLeNet (它们结构固定，配置文件中可能没有 'out')，避免 KeyError
        if 'out' in layer:
            available_mutations.append('channel_scale')

        # 3. 随机选择变异类型
        mutation_type = random.choice(available_mutations)

        if mutation_type == 'split':
            # 算子分裂：增加/减少分组卷积的组数
            # 使用 .get() 提供默认值 1，增强鲁棒性
            current_groups = layer.get('groups', 1)
            choices = [1, 2, 4, 8]
            layer['groups'] = random.choice(choices)

        elif mutation_type == 'fuse':
            # 算子融合：切换融合状态
            # 使用 .get() 提供默认值 False
            current_fused = layer.get('fused', False)
            layer['fused'] = not current_fused

        elif mutation_type == 'channel_scale':
            # 通道剪枝/扩容：微调通道数
            # 只有在 available_mutations 包含此项时才会进入这里，所以 layer['out'] 是安全的
            scale = random.choice([0.8, 1.0, 1.2])
            current_out = layer['out']
            new_out = int(current_out * scale)
            # 保证至少有16个通道且为8的倍数以便于硬件对齐
            layer['out'] = max(16, (new_out // 8) * 8)

        return new_config