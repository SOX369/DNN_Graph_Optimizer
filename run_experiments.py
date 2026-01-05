import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
from config import HARDWARE_CONSTRAINTS
from models.vgg import VGG_Cifar
from optimizer.simulated_annealing import SAGraphOptimizer
from optimizer.cost_model import HardwareAwareCostModel
from utils.data_loader import get_cifar10_loaders
from train_eval import train_model, evaluate_performance

# --- 定义初始图结构 (Baseline) ---
INITIAL_GRAPH_CONFIG = [
    {'type': 'conv', 'out': 64, 'groups': 1, 'fused': False}, 'M',
    {'type': 'conv', 'out': 128, 'groups': 1, 'fused': False}, 'M',
    {'type': 'conv', 'out': 256, 'groups': 1, 'fused': False},
    {'type': 'conv', 'out': 256, 'groups': 1, 'fused': False}, 'M',
    {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False},
    {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False}, 'M',
    {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False},
    {'type': 'conv', 'out': 512, 'groups': 1, 'fused': False}, 'M',
]


def run_experiment_1_ablation():
    """
    实验 1: 对比实验 (Ablation Study)
    对比 Baseline 模型与 Optimized 模型的 FLOPs, Params, Accuracy, Latency
    """
    print("\n" + "=" * 40)
    print(">> Running Experiment 1: Ablation Study (Baseline vs. Ours)")
    print("=" * 40)

    # 1. 评估 Baseline
    print("[1/4] Evaluating Baseline Model...")
    baseline_model = VGG_Cifar(INITIAL_GRAPH_CONFIG)
    cost_model = HardwareAwareCostModel()
    _, base_flops, base_params = cost_model.evaluate(INITIAL_GRAPH_CONFIG)

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    # 为了实验速度，Baseline 只训练 1 个 epoch (论文里建议写 5-10 epoch)
    train_model(baseline_model, train_loader, epochs=10)
    base_acc, base_lat = evaluate_performance(baseline_model, test_loader)

    # 2. 搜索最优结构
    print("[2/4] Searching for Optimized Architecture...")
    optimizer = SAGraphOptimizer(INITIAL_GRAPH_CONFIG)
    opt_config, _ = optimizer.search()
    _, opt_flops, opt_params = cost_model.evaluate(opt_config)

    # 3. 评估 Optimized
    print("[3/4] Evaluating Optimized Model...")
    optimized_model = VGG_Cifar(opt_config)
    # 优化后的模型通常需要多一点 epoch 来恢复精度，这里设为 3
    train_model(optimized_model, train_loader, epochs=10)
    opt_acc, opt_lat = evaluate_performance(optimized_model, test_loader)

    # 4. 打印对比表
    data = {
        "Metric": ["FLOPs (M)", "Params (M)", "Accuracy (%)", "Latency (ms)"],
        "Baseline": [base_flops / 1e6, base_params / 1e6, base_acc, base_lat],
        "Ours (Optimized)": [opt_flops / 1e6, opt_params / 1e6, opt_acc, opt_lat],
        "Improvement": [
            f"-{(1 - opt_flops / base_flops) * 100:.2f}%",
            f"-{(1 - opt_params / base_params) * 100:.2f}%",
            f"{opt_acc - base_acc:.2f}%",
            f"-{(1 - opt_lat / base_lat) * 100:.2f}%"
        ]
    }
    df = pd.DataFrame(data)
    print("\n>>> Ablation Study Result:")
    print(df.to_string(index=False))

    # 简单绘图对比
    df.plot(x="Metric", y=["Baseline", "Ours (Optimized)"], kind="bar", figsize=(10, 6))
    plt.title("Baseline vs Optimized Model Performance")
    plt.ylabel("Value")
    plt.xticks(rotation=0)
    plt.savefig("exp1_ablation_comparison.png")
    print("Result saved to exp1_ablation_comparison.png")


def run_experiment_2_sensitivity():
    """
    实验 2: 敏感度分析 (Sensitivity Analysis)
    调节惩罚系数 lambda，观察模型压缩率的变化
    """
    print("\n" + "=" * 40)
    print(">> Running Experiment 2: Sensitivity Analysis (Penalty Coefficient)")
    print("=" * 40)

    # 定义不同的 lambda 值
    lambdas = [0.0, 1.0, 10.0, 50.0]
    results = []

    base_cost_model = HardwareAwareCostModel()
    _, base_flops, base_params = base_cost_model.evaluate(INITIAL_GRAPH_CONFIG)

    for lam in lambdas:
        print(f"Testing with Penalty Coefficient lambda = {lam}...")

        # 临时修改全局配置中的参数
        # 注意：这里需要重新实例化 CostModel 才能生效，
        # 因为我们的 SAGraphOptimizer 每次都会新建 CostModel
        original_coef = HARDWARE_CONSTRAINTS['PENALTY_COEF']
        HARDWARE_CONSTRAINTS['PENALTY_COEF'] = lam

        optimizer = SAGraphOptimizer(INITIAL_GRAPH_CONFIG)
        opt_config, _ = optimizer.search()

        # 恢复配置，以免影响后续
        HARDWARE_CONSTRAINTS['PENALTY_COEF'] = original_coef

        # 评估结果
        temp_cost_model = HardwareAwareCostModel()  # 这里的系数不重要，只用来算FLOPs
        _, opt_flops, opt_params = temp_cost_model.evaluate(opt_config)

        results.append({
            "Lambda": lam,
            "FLOPs (M)": opt_flops / 1e6,
            "Params (M)": opt_params / 1e6,
            "Compression Ratio (%)": (1 - opt_params / base_params) * 100
        })

    df = pd.DataFrame(results)
    print("\n>>> Sensitivity Analysis Result:")
    print(df.to_string(index=False))

    # 绘图：Lambda vs Compression Ratio
    plt.figure()
    plt.plot(df["Lambda"], df["Compression Ratio (%)"], marker='o', linestyle='-', color='r')
    plt.xlabel("Penalty Coefficient (Lambda)")
    plt.ylabel("Parameter Compression Ratio (%)")
    plt.title("Impact of Penalty Coefficient on Model Compression")
    plt.grid(True)
    plt.savefig("exp2_sensitivity_lambda.png")
    print("Result saved to exp2_sensitivity_lambda.png")


def run_experiment_3_trajectory():
    """
    实验 3: 搜索轨迹可视化 (Search Trajectory)
    绘制能量下降曲线，证明算法收敛
    """
    print("\n" + "=" * 40)
    print(">> Running Experiment 3: Search Trajectory Visualization")
    print("=" * 40)

    # 使用默认配置运行一次搜索
    optimizer = SAGraphOptimizer(INITIAL_GRAPH_CONFIG)
    _, history = optimizer.search()

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='System Energy', color='blue', alpha=0.6)

    # 绘制平滑曲线（移动平均）
    if len(history) > 10:
        window_size = 10
        moving_avg = np.convolve(history, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(history)), moving_avg, label='Moving Average', color='red', linewidth=2)

    plt.xlabel("Search Iterations")
    plt.ylabel("Energy Value (Cost)")
    plt.title("Evolutionary Search Trajectory (Simulated Annealing)")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp3_search_trajectory.png")
    print("Result saved to exp3_search_trajectory.png")


if __name__ == "__main__":
    # 依次运行三个实验
    # 提示：为了节省时间，你可以注释掉某个函数单独跑

    # 1. 运行搜索轨迹 (最快)
    run_experiment_3_trajectory()

    # 2. 运行敏感度分析 (较快，只做搜索不训练)
    run_experiment_2_sensitivity()

    # 3. 运行对比实验 (最慢，因为要真实训练模型)
    run_experiment_1_ablation()

    print("\nAll experiments finished!")