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
        conv_indices = [i for i, x in enumerate(new_config) if isinstance(x, dict)]

        if not conv_indices:
            return new_config

        # 随机选择一个层
        target_idx = random.choice(conv_indices)
        layer = new_config[target_idx]

        # 2. 动态构建可用的变异类型列表
        available_mutations = ['split', 'fuse']
        if 'out' in layer:
            available_mutations.append('channel_scale')

        # 3. 随机选择变异类型
        mutation_type = random.choice(available_mutations)

        if mutation_type == 'split':
            # [修正]: 扩展 Groups 的搜索空间，以支持 Depthwise 和 ResNeXt
            current_out = layer.get('out', 64)

            # 基础候选: 1 (标准卷积), 2, 4, 8, 32 (ResNeXt常用)
            # 特别加入: current_out (即 Depthwise Convolution)
            potential_choices = [1, 2, 4, 8, 16, 32, current_out]

            # 过滤非法值: groups 必须能整除 out_channels
            valid_choices = [g for g in potential_choices if g <= current_out and current_out % g == 0]

            if valid_choices:
                layer['groups'] = random.choice(valid_choices)

        elif mutation_type == 'fuse':
            # 算子融合：切换融合状态
            current_fused = layer.get('fused', False)
            layer['fused'] = not current_fused

        elif mutation_type == 'channel_scale':
            # 通道剪枝/扩容：微调通道数
            scale = random.choice([0.8, 1.0, 1.2])
            current_out = layer['out']
            new_out = int(current_out * scale)
            # [优化]: 保证至少16通道，且对齐到 8 (对硬件更友好)
            layer['out'] = max(16, (new_out // 8) * 8)

            # [新增]: 通道改变后，需检查当前的 groups 是否依然合法
            # 如果 groups > new_out 或 不能整除，重置为 1 或 new_out (Depthwise)
            current_groups = layer.get('groups', 1)
            if current_groups > layer['out'] or layer['out'] % current_groups != 0:
                # 50%概率重置为1，50%概率尝试 Depthwise
                layer['groups'] = layer['out'] if random.random() > 0.5 else 1

        return new_config