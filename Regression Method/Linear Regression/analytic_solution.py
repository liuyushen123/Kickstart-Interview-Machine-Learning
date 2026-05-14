import numpy as np

# 我们的极简数据集
X = np.array([1, 2, 3])
Y = np.array([2, 4, 6])
m = len(X)  # 样本数量

print("目标：找到完美的权重 w，期望值是 2.0\n")
print("-" * 40)

# ==========================================
# 方法 1：正规方程 (Normal Equation) - "一步到位"
# ==========================================
# 公式: w = (X^T * X)^(-1) * X^T * Y
# 注意：为了进行矩阵运算，我们需要将 X 和 Y 转换为列向量
X_mat = X.reshape(-1, 1)
Y_mat = Y.reshape(-1, 1)

# 直接套用公式进行矩阵计算
w_normal = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y_mat)

print(f"【正规方程计算结果】\n一步算出权重 w = {w_normal[0][0]:.4f}")


print("\n" + "-" * 40)


# ==========================================
# 方法 2：梯度下降 (Gradient Descent) - "摸着石头过河"
# ==========================================
w_gd = 0.0  # 第1步：随便猜一个初始权重 (比如 0)
alpha = 0.1  # 学习率 (每次下山的步子有多大)
epochs = 20  # 我们打算走多少步 (迭代次数)

print("【梯度下降计算过程】")
for i in range(epochs):
    # 根据当前的权重进行预测
    Y_pred = w_gd * X

    # 计算误差 (预测值 - 实际值)
    error = Y_pred - Y

    # 第2步：计算误差曲线在当前位置的斜率 (梯度)
    # 这里的数学求导结果是：(1/m) * sum(误差 * X)
    gradient = (1 / m) * np.sum(error * X)

    # 第3步：更新权重，朝着梯度的反方向(下坡)走一小步
    w_gd = w_gd - alpha * gradient

    # 每走5步打印一下进度，看看它是如何逼近目标 2.0 的
    if (i + 1) % 5 == 0:
        print(
            f"走了 {i + 1} 步后，当前的权重 w = {w_gd:.4f} (当前梯度: {gradient:.4f})"
        )

print(f"\n最终梯度下降算出的权重 w = {w_gd:.4f}")
