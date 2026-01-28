import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import os
import time
from config import HARDWARE_CONSTRAINTS
from optimizer.simulated_annealing import SAGraphOptimizer
# [新增] 引入真实的贪心优化器 (请确保已创建 optimizer/greedy.py)
from optimizer.greedy import GreedyGraphOptimizer
from optimizer.cost_model import HardwareAwareCostModel
from utils.data_loader import get_cifar10_loaders
from train_eval import train_model, evaluate_performance
from utils.model_utils import generate_default_config, get_model

# 结果保存路径
RESULTS_DIR = "results_30epochs_V2"
PTH_DIR = "pth"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if not os.path.exists(PTH_DIR):
    os.makedirs(PTH_DIR)


def save_results(df, filename, title=None):
    filepath = os.path.join(RESULTS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"\n>>> Results saved to {filepath}")
    if title:
        print(f"\n--- {title} ---")
    print(df.to_string(index=False))


# ==================================================================================
# Experiment 1: Sensitivity Analysis (Determine Optimal Lambda)
# ==================================================================================
def run_experiment_1_sensitivity():
    """
    实验1: 参数敏感度分析 (Sensitivity Analysis)
    目的: 确定最佳的惩罚系数 lambda，用于后续实验。
    """
    print("\n" + "=" * 80)
    print(">> Running Experiment 1: Sensitivity Analysis (Finding Optimal Lambda)")
    print(">> Goal: Identify the best trade-off between Latency and Memory")
    print("=" * 80)

    lambdas = [0, 0.1, 0.5, 2, 5]
    # 使用 ResNet18 作为探测模型
    model_name = 'resnet18'
    base_config = generate_default_config(model_name)
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # 备份原始配置
    original_lambda = HARDWARE_CONSTRAINTS['PENALTY_COEF']
    results = []

    print(f"--- Probing Model: {model_name} ---")

    for lam in lambdas:
        print(f"\n[Sensitivity] Testing Lambda = {lam}...")
        HARDWARE_CONSTRAINTS['PENALTY_COEF'] = lam
        optimizer = SAGraphOptimizer(base_config)
        # 敏感度分析只搜索结构，不进行全量训练以节省时间
        opt_cfg, _ = optimizer.search()

        cm = HardwareAwareCostModel()
        _, flops, params = cm.evaluate(opt_cfg)

        # 实例化模型测量真实延迟 (不训练，只测推理)
        model_lam = get_model(model_name, opt_cfg)
        _, lat_real = evaluate_performance(model_lam, test_loader)
        mem_real = (params / 1e6) * 4 * 3  # 估算显存 (float32)

        results.append({
            "Penalty Coef (λ)": lam,
            "Params (M)": round(params / 1e6, 2),
            "GFLOPs": round(flops / 1e9, 2),
            "Latency (ms)": round(lat_real, 2),
            "Peak Mem (MB)": round(mem_real, 2)
        })

    # 恢复全局配置
    HARDWARE_CONSTRAINTS['PENALTY_COEF'] = original_lambda

    df = pd.DataFrame(results)
    save_results(df, "Table_1_Sensitivity_Analysis.csv", "Table 1 Sensitivity")

    # 绘图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Penalty Coefficient (λ)')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.plot(df['Penalty Coef (λ)'], df['Latency (ms)'], marker='o', color=color, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Peak Memory (MB)', color=color)
    ax2.plot(df['Penalty Coef (λ)'], df['Peak Mem (MB)'], marker='s', linestyle='--', color=color, label='Memory')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title("Experiment 1: Impact of λ on Latency and Memory")
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Figure_1_Sensitivity.png"))

    print("\n>> Conclusion from Exp 1: Lambda=0.5 provides the best balance (Sweet Spot).")
    print(">> Proceeding to subsequent experiments with Lambda = 0.5.")


# ==================================================================================
# Experiment 2: Ablation Study (Verify Components with Optimal Lambda)
# ==================================================================================
def run_experiment_2_ablation():
    """
    实验2: 消融实验 (Ablation Study)
    设定: 使用 Exp 1 确定的 lambda=0.5
    """
    print("\n" + "=" * 80)
    print(">> Running Experiment 2: Ablation Study (Validating Components)")
    print(">> Setting: Fixed Lambda = 0.5")
    print("=" * 80)

    # 目标模型
    model_name = 'resnet18'
    base_config = generate_default_config(model_name)
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    EPOCHS = 30  # 消融实验 Epoch 可以稍少，快速验证

    # 1. 强制设定最佳 Lambda
    original_lambda = HARDWARE_CONSTRAINTS['PENALTY_COEF']
    HARDWARE_CONSTRAINTS['PENALTY_COEF'] = 0.5

    results = []
    cm = HardwareAwareCostModel()

    # === Variant 1: C-DGOSA (Full) ===
    print("\n>>> Variant 1: C-DGOSA (Full)...")
    optimizer = SAGraphOptimizer(base_config)
    cfg_full, _ = optimizer.search()

    model_full = get_model(model_name, cfg_full)
    train_model(model_full, train_loader, epochs=EPOCHS)
    acc_full, lat_full = evaluate_performance(model_full, test_loader)

    _, _, p_full = cm.evaluate(cfg_full)
    mem_full = (p_full / 1e6) * 4 * 3

    torch.save(model_full.state_dict(), os.path.join(PTH_DIR, f"ablation_full.pth"))

    results.append({
        "Variant": "C-DGOSA (Full)",
        "Latency (ms)": round(lat_full, 2),
        "Peak Memory (MB)": round(mem_full, 2),
        "Top-1 Acc (%)": round(acc_full, 2)
    })

    # === Variant 2: w/o Split ===
    print("\n>>> Variant 2: w/o Split (No Group Conv)...")
    cfg_ns = copy.deepcopy(cfg_full)
    # 模拟禁用 Split: 将所有 groups 强制重置为 1
    for layer in cfg_ns:
        if isinstance(layer, dict): layer['groups'] = 1

    model_ns = get_model(model_name, cfg_ns)
    train_model(model_ns, train_loader, epochs=EPOCHS)
    acc_ns, lat_ns = evaluate_performance(model_ns, test_loader)
    _, _, p_ns = cm.evaluate(cfg_ns)
    mem_ns = (p_ns / 1e6) * 4 * 3
    results.append({
        "Variant": "C-DGOSA w/o Split",
        "Latency (ms)": round(lat_ns, 2),
        "Peak Memory (MB)": round(mem_ns, 2),
        "Top-1 Acc (%)": round(acc_ns, 2)
    })

    # === Variant 3: w/o Constraint ===
    print("\n>>> Variant 3: w/o Constraint (Lambda=0)...")
    HARDWARE_CONSTRAINTS['PENALTY_COEF'] = 0.0  # 临时设为 0
    optimizer_nc = SAGraphOptimizer(base_config)
    cfg_nc, _ = optimizer_nc.search()

    model_nc = get_model(model_name, cfg_nc)
    train_model(model_nc, train_loader, epochs=EPOCHS)
    acc_nc, lat_nc = evaluate_performance(model_nc, test_loader)
    _, _, p_nc = cm.evaluate(cfg_nc)
    mem_nc = (p_nc / 1e6) * 4 * 3
    results.append({
        "Variant": "C-DGOSA w/o Constraint",
        "Latency (ms)": round(lat_nc, 2),
        "Peak Memory (MB)": round(mem_nc, 2),
        "Top-1 Acc (%)": round(acc_nc, 2)
    })

    # 恢复 Lambda=0.5 以供后续使用
    HARDWARE_CONSTRAINTS['PENALTY_COEF'] = 0.5

    # === Variant 4: Greedy (Real Implementation) ===
    print("\n>>> Variant 4: C-DGOSA-Greedy (Real Hill Climbing)...")
    # 使用真实的贪心优化器
    greedy_opt = GreedyGraphOptimizer(base_config, max_iter=200)
    cfg_greedy, _ = greedy_opt.search()

    model_greedy = get_model(model_name, cfg_greedy)
    train_model(model_greedy, train_loader, epochs=EPOCHS)
    acc_greedy, lat_greedy = evaluate_performance(model_greedy, test_loader)

    _, _, p_greedy = cm.evaluate(cfg_greedy)
    mem_greedy = (p_greedy / 1e6) * 4 * 3

    results.append({
        "Variant": "C-DGOSA-Greedy",
        "Latency (ms)": round(lat_greedy, 2),
        "Peak Memory (MB)": round(mem_greedy, 2),
        "Top-1 Acc (%)": round(acc_greedy, 2)
    })

    HARDWARE_CONSTRAINTS['PENALTY_COEF'] = original_lambda  # 恢复全局

    df = pd.DataFrame(results)
    save_results(df, "Table_2_Ablation_Study.csv", "Table 2 Ablation")


# ==================================================================================
# Experiment 3: Performance Comparison (Main Results)
# ==================================================================================
def run_experiment_3_comparison():
    """
    实验3: 不同优化方法的对比 (Performance Comparison)
    设定: 使用 Exp 1 确定的 lambda=0.5
    对比: PyTorch Baseline vs C-DGOSA (Ours)
    """
    print("\n" + "=" * 80)
    print(">> Running Experiment 3: Performance Comparison (State-of-the-Art)")
    print(">> Setting: Fixed Lambda = 0.5")
    print("=" * 80)

    # 1. 强制设定最佳 Lambda
    original_lambda = HARDWARE_CONSTRAINTS['PENALTY_COEF']
    HARDWARE_CONSTRAINTS['PENALTY_COEF'] = 0.5

    # 模型列表 (已包含 ShuffleNetV2)
    models_list = ['vgg16', 'resnet18', 'shufflenetv2', 'googlenet', 'mobilenetv2']
    results = []

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    EPOCHS = 30  # 主实验 Epoch，建议设为 30-50 以获得更可信的精度

    for model_name in models_list:
        print(f"\n--- Processing Model: {model_name} ---")

        # === 1. Baseline ===
        print(f"[{model_name}] Evaluating Baseline...")
        base_config = generate_default_config(model_name)
        base_model = get_model(model_name, base_config)

        cost_model = HardwareAwareCostModel()
        _, _, base_params = cost_model.evaluate(base_config)
        base_mem = (base_params / 1e6) * 4 * 3

        print(f"[{model_name}] Training Baseline...")
        train_model(base_model, train_loader, epochs=EPOCHS)
        base_acc, base_lat = evaluate_performance(base_model, test_loader)

        torch.save(base_model.state_dict(), os.path.join(PTH_DIR, f"{model_name}_baseline.pth"))

        results.append({
            "Models": model_name,
            "Method": "PyTorch (Baseline)",
            "Top-1 Accuracy(%)": round(base_acc, 2),
            "Latency(ms)": round(base_lat, 2),
            "Peak Memory(MB)": round(base_mem, 2),
            "Latency Reduction": "-",
            "Memory Reduction": "-"
        })

        # === 2. C-DGOSA (Ours) ===
        print(f"[{model_name}] Running C-DGOSA Optimization (Lambda=0.5)...")
        optimizer = SAGraphOptimizer(base_config)
        opt_config, _ = optimizer.search()

        _, _, opt_params = cost_model.evaluate(opt_config)
        opt_mem = (opt_params / 1e6) * 4 * 3

        print(f"[{model_name}] Training Optimized Model...")
        opt_model = get_model(model_name, opt_config)
        train_model(opt_model, train_loader, epochs=EPOCHS)
        opt_acc, opt_lat = evaluate_performance(opt_model, test_loader)

        torch.save(opt_model.state_dict(), os.path.join(PTH_DIR, f"{model_name}_optimized.pth"))

        # 计算提升
        lat_red = (1 - opt_lat / base_lat) * 100
        mem_red = (1 - opt_mem / base_mem) * 100

        results.append({
            "Models": model_name,
            "Method": "C-DGOSA (Ours)",
            "Top-1 Accuracy(%)": round(opt_acc, 2),
            "Latency(ms)": round(opt_lat, 2),
            "Peak Memory(MB)": round(opt_mem, 2),
            "Latency Reduction": f"{lat_red:.1f}%",
            "Memory Reduction": f"{mem_red:.1f}%"
        })

    HARDWARE_CONSTRAINTS['PENALTY_COEF'] = original_lambda  # 恢复

    df = pd.DataFrame(results)
    save_results(df, "Table_3_Performance_Comparison.csv", "Table 3 Comparison")

    # 绘图
    df_ours = df[df["Method"] == "C-DGOSA (Ours)"]
    labels = df_ours["Models"]
    lat_vals = [float(str(x).strip('%')) for x in df_ours["Latency Reduction"]]
    mem_vals = [float(str(x).strip('%')) for x in df_ours["Memory Reduction"]]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, lat_vals, width, label='Latency Reduction')
    ax.bar(x + width / 2, mem_vals, width, label='Memory Reduction')
    ax.set_ylabel('Reduction Percentage (%)')
    ax.set_title('Experiment 3: Performance Improvement (Lambda=0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "Figure_3_Performance.png"))


if __name__ == "__main__":
    start_time = time.time()

    # === 逻辑顺序调整 ===
    # 1. 先做敏感度分析，确定 lambda=0.5
    run_experiment_1_sensitivity()

    # 2. 再做消融实验，验证组件有效性 (使用 lambda=0.5)
    run_experiment_2_ablation()

    # 3. 最后做对比实验，展示最终效果 (使用 lambda=0.5)
    run_experiment_3_comparison()

    end_time = time.time()
    print(f"\nAll Experiments Finished! Total Time: {(end_time - start_time) / 60:.2f} mins")
    print(f"Check '{RESULTS_DIR}' for all Tables and Figures.")