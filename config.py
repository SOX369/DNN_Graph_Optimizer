import torch

# 硬件约束模拟 (单位: M = Million)
# 假设目标是一个边缘设备，显存很小，算力有限
HARDWARE_CONSTRAINTS = {
    "MAX_FLOPs": 500 * 1e6,   # 最大允许计算量
    "MAX_PARAMS": 15 * 1e6,   # 最大允许参数量 (模拟显存限制)
    "PENALTY_COEF": 10.0      # 超过约束时的惩罚系数 lambda
}

# 模拟退火超参数
SA_CONFIG = {
    "INIT_TEMP": 100.0,       # 初始温度
    "ALPHA": 0.95,            # 降温系数
    "MIN_TEMP": 0.1,          # 终止温度
    "ITER_PER_TEMP": 5        # 每个温度下的迭代次数
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

