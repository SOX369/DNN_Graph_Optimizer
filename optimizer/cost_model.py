import sys
import os

# 将 train_on_CIFAR10 目录添加到路径
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DNN_Graph_Optimizer.config import HARDWARE_CONSTRAINTS

class HardwareAwareCostModel:
    """
    基于硬件特性的代价评估模型。
    对应论文公式: E(G) = Latency(G) + lambda * Penalty(Memory)
    """

    def __init__(self):
        self.max_flops = HARDWARE_CONSTRAINTS['MAX_FLOPs']
        self.max_params = HARDWARE_CONSTRAINTS['MAX_PARAMS']
        self.penalty_coef = HARDWARE_CONSTRAINTS['PENALTY_COEF']

    def evaluate(self, graph_config):
        """
        估算计算图的 FLOPs (代表时延) 和 Params (代表显存占用)
        """
        total_flops = 0
        total_params = 0
        in_c = 3
        size = 32  # CIFAR图片大小

        for layer in graph_config:
            if layer == 'M':
                size //= 2
                continue

            out_c = layer['out']
            groups = layer['groups']
            k = 3

            # FLOPs 计算公式 (简化版)
            # FLOPs = H * W * Cin * Cout * K^2 / groups
            flops = (size * size * in_c * out_c * k * k) / groups

            # 参数量计算
            # Params = (Cin * Cout * K^2) / groups
            params = (in_c * out_c * k * k) / groups

            total_flops += flops
            total_params += params
            in_c = out_c

        # 计算能量值 (Energy)
        # 归一化 Latency
        norm_latency = total_flops / 1e8

        # 显存惩罚项 (Barrier Function)
        mem_violation = max(0, total_params - self.max_params)
        penalty = self.penalty_coef * (mem_violation / 1e6)

        energy = norm_latency + penalty

        return energy, total_flops, total_params

