import torch
import sys
import os

# 将 DNN_Graph_Optimizer 目录添加到 sys.path
dnn_optimizer_dir = os.path.dirname(os.path.abspath(__file__))  # DNN_Graph_Optimizer 目录
if dnn_optimizer_dir not in sys.path:
    sys.path.insert(0, dnn_optimizer_dir)


from models.vgg import VGG_Cifar
from optimizer.simulated_annealing import SAGraphOptimizer
from utils.data_loader import get_cifar10_loaders
from train_eval import train_model, evaluate_performance
from optimizer.cost_model import HardwareAwareCostModel


def main():
    # 1. 定义初始的计算图结构 (以 VGG-11 为例)
    # 这就是我们要优化的"图结构配置"
    # groups=1 表示标准卷积，fused=False 表示未融合
    initial_graph_config = [
        {'type': 'conv', 'out': 64, 'groups': 1, 'fused': False}, 'M',
        {'type': 'conv', 'out': 128, 'groups': 1, 'fused': False}, 'M',
        {'type': 'conv', 'out': 256, 'groups': 1, 'fused': False},
        {'type': 'conv', 'out': 256, 'groups': 1, 'fused': False}, 'M',
        {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False},
        {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False}, 'M',
        {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False},
        {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False}, 'M',
    ]

    print("=== Phase 1: Baseline Evaluation ===")
    baseline_model = VGG_Cifar(initial_graph_config)
    cost_model = HardwareAwareCostModel()
    base_energy, base_flops, base_params = cost_model.evaluate(initial_graph_config)
    print(f"Baseline Theoretical Metrics -> FLOPs: {base_flops / 1e6:.2f}M, Params: {base_params / 1e6:.2f}M")

    # 2. 运行模拟退火搜索 (论文的核心创新部分)
    print("\n=== Phase 2: Running Simulated Annealing Graph Optimization ===")
    optimizer = SAGraphOptimizer(initial_graph_config)
    optimized_config = optimizer.search()

    print("\nOptimization Finished!")
    print("Optimized Graph Structure:", optimized_config)

    opt_energy, opt_flops, opt_params = cost_model.evaluate(optimized_config)
    print(f"Optimized Metrics -> FLOPs: {opt_flops / 1e6:.2f}M, Params: {opt_params / 1e6:.2f}M")
    print(
        f"Reduction -> FLOPs: {(1 - opt_flops / base_flops) * 100:.2f}%, Params: {(1 - opt_params / base_params) * 100:.2f}%")

    # 3. 实例化优化后的模型并在真实数据上验证 (验证部分)
    print("\n=== Phase 3: Training & Inference Verification ===")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    print("\nTraining Optimized Model...")
    optimized_model = VGG_Cifar(optimized_config)
    train_model(optimized_model, train_loader, epochs=10)  # 演示仅训练10轮

    print("\nEvaluating Optimized Model...")
    acc, lat = evaluate_performance(optimized_model, test_loader)
    print(f"Optimized Model Results -> Accuracy: {acc:.2f}%, Inference Latency (Batch): {lat:.2f}ms")


if __name__ == "__main__":
    main()