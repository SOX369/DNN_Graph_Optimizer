import math
import random
from .cost_model import HardwareAwareCostModel
from .mutator import GraphMutator

import sys
import os

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DNN_Graph_Optimizer.config import SA_CONFIG


class SAGraphOptimizer:
    """
    模拟退火优化器。
    实现 Metropolis 准则进行全局寻优。
    """

    def __init__(self, initial_graph_config):
        self.curr_config = initial_graph_config
        self.best_config = initial_graph_config

        self.cost_model = HardwareAwareCostModel()
        self.mutator = GraphMutator(initial_graph_config)

        self.curr_energy, _, _ = self.cost_model.evaluate(initial_graph_config)
        self.best_energy = self.curr_energy

        self.temp = SA_CONFIG['INIT_TEMP']
        self.alpha = SA_CONFIG['ALPHA']
        self.min_temp = SA_CONFIG['MIN_TEMP']

        # 新增：用于记录搜索轨迹
        self.energy_history = []

    def search(self):
        print(f"Start Searching... Initial Energy: {self.curr_energy:.4f}")
        step = 0

        # 记录初始状态
        self.energy_history.append(self.curr_energy)

        while self.temp > self.min_temp:
            for _ in range(SA_CONFIG['ITER_PER_TEMP']):
                # 1. 变异生成新解
                new_config = self.mutator.perturb(self.curr_config)

                # 2. 评估新解能量
                new_energy, flops, params = self.cost_model.evaluate(new_config)

                # 3. 计算能量差
                delta_e = new_energy - self.curr_energy

                # 4. Metropolis 准则
                # 如果更好 (delta_e < 0)，直接接受
                # 如果更差，以概率 exp(-delta_e / T) 接受
                accept = False
                if delta_e < 0:
                    accept = True
                else:
                    prob = math.exp(-delta_e / self.temp)
                    if random.random() < prob:
                        accept = True

                if accept:
                    self.curr_config = new_config
                    self.curr_energy = new_energy

                    # 更新全局最优
                    if self.curr_energy < self.best_energy:
                        self.best_config = new_config
                        self.best_energy = self.curr_energy
                        print(
                            f"[Iter {step}] New Best Found! Energy: {self.best_energy:.4f} | FLOPs: {flops / 1e6:.2f}M | Params: {params / 1e6:.2f}M")

                # 新增：记录每一步的能量值（无论是否接受，记录当前的能量状态）
                self.energy_history.append(self.curr_energy)

                step += 1

            # 5. 降温
            self.temp *= self.alpha

        # 修改返回值：同时返回配置和历史数据
        return self.best_config, self.energy_history
