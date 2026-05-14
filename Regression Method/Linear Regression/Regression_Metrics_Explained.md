# 📊 Regression Error Metrics: MAE vs. MSE

When training a model to predict continuous values (like Titanic ticket prices or passenger ages), we need a way to measure how "wrong" our predictions are. Here are the two most common ways to calculate that error.

---

## 1. Mean Absolute Error (MAE)
**The "Intuitive" Metric**

MAE is the average of the absolute differences between the actual values and the predicted values.

### The Formula
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Key Characteristics:
* **Unit Match:** If you are predicting price in Dollars, the MAE is in Dollars.
* **Robustness:** It treats all errors linearly. A mistake of 20 is exactly twice as bad as a mistake of 10.
* **Best Use Case:** When you want a metric that is easy to explain to non-technical stakeholders or when your data has outliers that you don't want to over-influence the model.

---

## 2. Mean Squared Error (MSE)
**The "Penalty" Metric**

MSE is the average of the squared differences between the actual and predicted values.

### The Formula
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Key Characteristics:
* **Magnifies Large Errors:** Because the errors are squared, a mistake of 10 results in a penalty of 100, but a mistake of 20 results in a penalty of **400**.
* **Mathematical Convenience:** Squaring makes the function smooth and differentiable, which is very helpful for optimization algorithms (like Gradient Descent).
* **Best Use Case:** When you want to strictly avoid large errors. If being off by a lot is "expensive" or dangerous, MSE will force the model to prioritize reducing those big gaps.

---

## Summary Comparison

| Feature | MAE | MSE |
| :--- | :--- | :--- |
| **Full Name** | Mean Absolute Error | Mean Squared Error |
| **Logic** | Average of absolute distance | Average of squared distance |
| **Units** | Same as data (e.g., £) | Squared (e.g., £²) |
| **Outlier Sensitivity** | Low (Robust) | High (Sensitive) |
| **Goal** | Tell me the "average" miss | Penalize the "big" misses |

---

### 💡 Pro-Tip: Root Mean Squared Error (RMSE)
If you like the "heavy penalty" logic of **MSE** but hate that the units are squared, you take the square root of the result to get **RMSE**. 

$$\text{RMSE} = \sqrt{\text{MSE}}$$

This is often the "Goldilocks" metric for many data scientists: it punishes big errors but is still readable in the original units!