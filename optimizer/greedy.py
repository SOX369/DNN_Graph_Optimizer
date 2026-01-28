import copy
from .mutator import GraphMutator
from .cost_model import HardwareAwareCostModel


class GreedyGraphOptimizer:
    """
    贪心搜索优化器 (Greedy Search)
    策略：每一步只接受能降低 Energy 的变异。如果所有邻域变异都不能降低 Energy，则停止（陷入局部最优）。
    """

    def __init__(self, base_config, max_iter=200):
        self.base_config = base_config
        self.max_iter = max_iter
        self.mutator = GraphMutator(base_config)
        self.cost_model = HardwareAwareCostModel()

    def search(self):
        current_config = copy.deepcopy(self.base_config)
        current_energy, _, _ = self.cost_model.evaluate(current_config)

        best_config = current_config
        best_energy = current_energy

        print(f"[Greedy] Start Energy: {current_energy:.4f}")

        for i in range(self.max_iter):
            # 尝试生成一个变异
            # 贪心策略变体：
            # 1. 简单贪心：随机变异一次，好就接受，不好就拒绝（类似无温度的 SA）。
            # 2. 深度贪心：遍历所有可能的变异，选最好的（计算量大）。
            # 这里采用“随机尝试+即时接受”的策略，模拟爬山法 (Hill Climbing)。

            new_config = self.mutator.perturb(current_config)
            new_energy, flops, params = self.cost_model.evaluate(new_config)

            # 贪心核心逻辑：只接受更优解
            if new_energy < current_energy:
                print(f"[Greedy Iter {i}] Accept Improvement! Energy: {current_energy:.4f} -> {new_energy:.4f}")
                current_energy = new_energy
                current_config = new_config

                # 更新全局最优
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_config = copy.deepcopy(new_config)
            else:
                # 拒绝差解
                pass

        print(f"[Greedy] Finished. Best Energy: {best_energy:.4f}")
        return best_config, best_energy