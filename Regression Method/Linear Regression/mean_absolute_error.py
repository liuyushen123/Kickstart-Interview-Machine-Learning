from sklearn.metrics import mean_absolute_error

y_true = [100, 150, 200]
y_pred = [110, 140, 250]  # 误差分别是 10, 10, 50

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae}")
# 计算过程: (10 + 10 + 50) / 3 = 23.33
