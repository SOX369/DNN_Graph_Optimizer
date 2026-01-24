import sys
import os

# 将项目根目录添加到路径，确保能导入配置文件
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DNN_Graph_Optimizer.config import HARDWARE_CONSTRAINTS


class HardwareAwareCostModel:
    """
    基于硬件特性的代价评估模型。
    核心目标函数: Energy(G) = Effective_Latency(G) + lambda * Penalty(Memory)
    """

    def __init__(self):
        self.max_flops = HARDWARE_CONSTRAINTS['MAX_FLOPs']
        self.max_params = HARDWARE_CONSTRAINTS['MAX_PARAMS']
        self.penalty_coef = HARDWARE_CONSTRAINTS['PENALTY_COEF']

    def evaluate(self, graph_config):
        """
        估算计算图的 Latency (基于有效 FLOPs) 和 Params (显存占用)。
        引入了硬件效率系数 alpha，以修正不同类型卷积算子(如 Depthwise)在实际硬件上的性能差异。
        """
        total_flops = 0
        total_params = 0
        total_effective_flops = 0  # 考虑硬件效率后的等效 FLOPs

        in_c = 3
        size = 32  # CIFAR-10 输入图片大小

        for layer in graph_config:
            # 1. 处理池化层标记 (VGG风格) 或 ShuffleNet 的降采样标记
            if layer == 'M':
                size //= 2
                continue

            # 2. 跳过非法配置
            if not isinstance(layer, dict):
                continue

            # 3. 解析层配置
            # out_c: 输出通道数
            out_c = layer.get('out', in_c)

            # [关键修改]: 优先读取显式的 'in' 属性 (用于 ShuffleNet/DenseNet)
            # 如果配置中没有 'in'，则默认使用上一层的输出 (in_c)
            actual_in_c = layer.get('in', in_c)

            groups = layer.get('groups', 1)
            # 获取卷积核大小 k，默认为 3
            k = layer.get('k', 3)

            # 4. 基础计算 (理论数值)
            # FLOPs = H * W * Cin * Cout * K^2 / groups
            # 注意这里使用 actual_in_c 进行计算
            current_flops = (size * size * actual_in_c * out_c * k * k) / groups

            # Params = Cin * Cout * K^2 / groups
            current_params = (actual_in_c * out_c * k * k) / groups

            # 5. 计算硬件效率系数 (Efficiency Factor, alpha)
            alpha = 1.0
            if groups == actual_in_c and actual_in_c > 1:
                # 深度卷积 (Depthwise): 给予 2.0 倍延迟惩罚
                alpha = 2.0
            elif groups > 1:
                # 普通分组卷积 (Group Conv): 给予 1.5 倍延迟惩罚 (ResNeXt 经验值修正)
                alpha = 1.5

            effective_flops = current_flops * alpha

            # 6. 累加统计量
            total_flops += current_flops
            total_effective_flops += effective_flops
            total_params += current_params

            # 更新下一层的默认输入通道数
            # 对于 ShuffleNet 这种复杂拓扑，这个 in_c 更新可能被下一次循环的 'in' 覆盖，这没问题。
            in_c = out_c

        # 7. 计算总能量值 (Energy)
        norm_latency = total_effective_flops / 1e8

        # 显存惩罚项
        mem_violation = max(0, total_params - self.max_params)
        penalty = self.penalty_coef * (mem_violation / 1e6)

        energy = norm_latency + penalty

        return energy, total_flops, total_params