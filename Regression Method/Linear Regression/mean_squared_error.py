import numpy as np
from sklearn.metrics import mean_squared_error

y_true = [100, 150, 200]
y_pred = [110, 140, 250]

# 计算 MSE
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse}")

# --- 拆解它的底层计算过程 ---
# 1. 先算出每个预测的误差：
#    (100 - 110) = -10
#    (150 - 140) = 10
#    (200 - 250) = -50

# 2. 把误差统统“平方”（消除负号，同时放大“大错误”）：
#    (-10)² = 100
#    (10)²  = 100
#    (-50)² = 2500   <-- 看，50的误差被放大了成倍！

# 3. 最后求这些平方值的平均数：
#    (100 + 100 + 2500) / 3 = 2700 / 3 = 900.0


"""
通过这行代码的对比，你能非常直观地看到：虽然平均的绝对误差（MAE）
只有 23.33，但因为第三个预测错得比较离谱（差了 50），导致平方后的 MSE 飙升到了 900。
这就是 MSE “严惩大错”的直观体现。
"""


def calculate_cost(y_actual, y_predicted):
    """
    计算课程图片中展示的代价函数（误差）。

    参数:
        y_actual (array-like): 实际的观测值 (y)。
        y_predicted (array-like): 模型的预测值 (y_hat)。

    返回:
        float: 计算得出的误差值。
    """
    # 将输入转换为 numpy 数组，以确保可以进行高效的向量化计算
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # 获取样本总数 'm'
    m = len(y_actual)

    # 计算平方差：(y - y_hat)^2
    squared_errors = (y_actual - y_predicted) ** 2

    # 应用公式的剩余部分：(1 / 2m) * 累加(平方差)
    error = (1 / (2 * m)) * np.sum(squared_errors)

    return error


# --- 示例用法 ---
# 假设以下是您的实际销售额和模型预测的销售额
actual_sales = [40, 60, 70, 85]
predicted_sales = [38, 55, 75, 80]

current_error = calculate_cost(actual_sales, predicted_sales)
print(f"当前模型的误差为: {current_error}")
