import sys
import os

# 将项目根目录添加到路径，确保能导入配置文件
# 假设当前文件位于 DNN_Graph_Optimizer/optimizer/cost_model.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
        估算计算图的 FLOPs (代表时延) 和 Params (代表显存占用)。
        支持 VGG (含 'M' 标记) 和 ResNet/GoogLeNet (纯配置列表) 的混合格式。
        """
        total_flops = 0
        total_params = 0
        in_c = 3
        size = 32  # CIFAR-10 输入图片大小

        for layer in graph_config:
            # 1. 处理池化层标记 (VGG风格)
            if layer == 'M':
                size //= 2
                continue

            # 2. 鲁棒性检查：如果遇到非字典项且不是'M'，跳过，防止报错
            if not isinstance(layer, dict):
                continue

            # 3. 获取层配置，使用 get 提供默认值防止 KeyError
            # 注意：对于 ResNet 等复杂网络，这里做了一个简化假设：
            # 假设 graph_config 是按拓扑顺序排列的，且包含 out 信息。
            # 如果配置中缺少 out，默认保持通道数不变
            out_c = layer.get('out', in_c)
            groups = layer.get('groups', 1)

            # [修复]: 动态获取卷积核大小 k。如果配置中没有 k，则默认为 3 (兼容旧 VGG 配置)
            # 解决了 ResNet 1x1 卷积被误算为 3x3 导致显存虚高 9 倍的问题
            k = layer.get('k', 3)

            # FLOPs 计算公式 (简化版，忽略 stride 带来的 feature map 尺寸变化细节，主要关注相对值)
            # FLOPs = H * W * Cin * Cout * K^2 / groups
            flops = (size * size * in_c * out_c * k * k) / groups

            # 参数量计算
            # Params = (Cin * Cout * K^2) / groups
            params = (in_c * out_c * k * k) / groups

            total_flops += flops
            total_params += params

            # 更新下一层的输入通道数
            in_c = out_c

        # 计算能量值 (Energy)
        # 归一化 Latency (除以 1e8 使得数值在合理范围)
        norm_latency = total_flops / 1e8

        # 显存惩罚项 (Barrier Function)
        # 如果 Params 超过 max_params，加入惩罚
        mem_violation = max(0, total_params - self.max_params)
        penalty = self.penalty_coef * (mem_violation / 1e6)

        energy = norm_latency + penalty

        return energy, total_flops, total_params