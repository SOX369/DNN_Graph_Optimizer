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
            # 1. 处理池化层标记 (VGG风格)
            if layer == 'M':
                size //= 2
                continue

            # 2. 跳过非法配置
            if not isinstance(layer, dict):
                continue

            # 3. 解析层配置
            out_c = layer.get('out', in_c)
            groups = layer.get('groups', 1)

            # 获取卷积核大小 k，默认为 3。
            # 这一步对于正确评估 ResNet/MobileNet 的 1x1 卷积至关重要。
            k = layer.get('k', 3)

            # 4. 基础计算 (理论数值)
            # FLOPs = H * W * Cin * Cout * K^2 / groups
            current_flops = (size * size * in_c * out_c * k * k) / groups
            # Params = Cin * Cout * K^2 / groups
            current_params = (in_c * out_c * k * k) / groups

            # 5. 计算硬件效率系数 (Efficiency Factor, alpha)
            # 虽然 Depthwise 卷积理论 FLOPs 很低，但受限于显存带宽，实际推理速度并没有线性提升。
            # 这里通过启发式系数 alpha 对其 Effective FLOPs 进行加权。
            alpha = 1.0
            if groups == in_c and in_c > 1:
                # 深度卷积 (Depthwise): 算术强度低，受限于带宽，给予 2.0 倍延迟惩罚
                alpha = 2.0
            elif groups > 1:
                # 普通分组卷积 (Group Conv): 相比标准卷积，内存访问效率略低
                alpha = 1.2

            effective_flops = current_flops * alpha

            # 6. 累加统计量
            total_flops += current_flops
            total_effective_flops += effective_flops
            total_params += current_params

            # 更新下一层的输入通道数
            in_c = out_c

        # 7. 计算总能量值 (Energy)
        # 使用 Effective FLOPs 来更准确地模拟真实 Latency
        norm_latency = total_effective_flops / 1e8

        # 显存惩罚项 (Barrier Function)
        # 如果 Params 超过 max_params，加入线性惩罚
        mem_violation = max(0, total_params - self.max_params)
        penalty = self.penalty_coef * (mem_violation / 1e6)

        energy = norm_latency + penalty

        # 返回 energy 用于优化搜索，返回 原始flops 和 params 用于日志记录
        return energy, total_flops, total_params