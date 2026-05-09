import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 忽略警告以保持输出整洁
warnings.filterwarnings("ignore")

# ==========================================
# 模块 1: 统计学基础与离散程度 (Statistical Concepts)
# ==========================================
print("=== 模块 1: 离差之和为零实验 ===")
"""
学习笔记：
- 均值 (Mean): 容易受极端值 (Outliers) 影响。
- 中位数 (Median): 适合偏态分布 (Skewed data)。
- 众数 (Mode): 出现频率最高的值。
- 方差逻辑: 所有观测值减去均值的偏差之和永远为0。因此我们需要将其平方以消除负号。
"""
data1 = np.array([random.randint(1, 1000) for _ in range(1, 1000)])
data2 = data1 - data1.mean()
print(f"证明实验：(x - mean) 的总和为: {round(np.sum(data2), 5)}\n")


# ==========================================
# 模块 2: 数据加载与初步探索 (EDA)
# ==========================================
print("=== 模块 2: 数据加载与探索 ===")
titanic = sns.load_dataset("titanic")

print("\n--- 原始数据集信息 ---")
print(titanic.info())

print("\n--- 数值型数据描述性统计 ---")
print(titanic.describe())


# ==========================================
# 模块 3: 数据清洗 (缺失值与重复值处理)
# ==========================================
print("\n=== 模块 3: 数据清洗 ===")

# 3.1 删除缺失值过多的无用列
if "deck" in titanic.columns:
    titanic = titanic.drop(columns="deck")

# 3.2 缺失值填充 (Imputation)
# - 年龄(age): 使用中位数填充 (不受极端值影响)
# - 登船港口(embarked): 使用众数填充 (分类变量最常见类别)
titanic["age"] = titanic["age"].fillna(titanic["age"].median())
titanic["embarked"] = titanic["embarked"].fillna(titanic["embarked"].mode()[0])

# 对于极少数缺失的行，直接删除
titanic = titanic.dropna(subset=["embark_town"])

# 3.3 处理重复值
duplicates = titanic.duplicated().sum()
if duplicates > 0:
    print(f"发现 {duplicates} 行重复数据，正在删除...")
    titanic = titanic.drop_duplicates()


# ==========================================
# 模块 4: 异常值处理 (Outlier Treatment)
# ==========================================
print("\n=== 模块 4: 异常值截断 (Clipping) ===")
# 修复了原代码的Bug：分别计算 Fare 和 Age 的 IQR 边界，并使用更高效的 .clip() 方法


def calculate_bounds(series):
    """计算基于 IQR 的上下界"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


# 处理票价 (Fare)
lower_fare, upper_fare = calculate_bounds(titanic["fare"])
fare_outliers = titanic[(titanic["fare"] < lower_fare) | (titanic["fare"] > upper_fare)]
print(f"Fare 异常值数量: {len(fare_outliers)}")
titanic["fare"] = titanic["fare"].clip(lower=lower_fare, upper=upper_fare)

# 处理年龄 (Age)
lower_age, upper_age = calculate_bounds(titanic["age"])
titanic["age"] = titanic["age"].clip(lower=lower_age, upper=upper_age)


# ==========================================
# 模块 5: 分类变量类型转换
# ==========================================
category_cols = ["sex", "embarked", "class", "who", "embark_town", "alive"]
for col in category_cols:
    if col in titanic.columns:
        titanic[col] = titanic[col].astype("category")


# ==========================================
# 模块 6: 特征工程与数据编码 (Feature Encoding) -> 【新增部分】
# ==========================================
print("\n=== 模块 6: 分类特征编码 ===")
# 机器学习模型通常只能理解数字，我们需要将文本类别转换为数字。

df_encoded = titanic.copy()  # 创建副本以防污染原数据

# 6.1 Binary Encoding (二值编码)
# 适用于只有两个类别的特征，直接映射为 0 和 1
print("执行 Binary Encoding ('sex' 转换为 0/1)...")
df_encoded["sex_binary"] = df_encoded["sex"].map({"male": 1, "female": 0})

# 6.2 Label Encoding (标签编码)
# 适用于有顺序关系的类别 (Ordinal Data)。比如 Pclass: First > Second > Third
print("执行 Label Encoding ('class' 转换为 0, 1, 2)...")
df_encoded["class_label"] = df_encoded["class"].cat.codes

# 6.3 Dummy Encoding / One-Hot Encoding (哑变量编码/独热编码)
# 适用于没有大小顺序的类别 (Nominal Data)。防止模型误以为类别之间有数值大小关系。
# drop_first=True 可以防止多重共线性 (Multicollinearity)
print("执行 Dummy Encoding ('embarked' 转换为独立列)...")
df_encoded = pd.get_dummies(
    df_encoded, columns=["embarked"], drop_first=True, dtype=int
)

# 6.4 Count / Frequency Encoding (频数编码)
# 将类别替换为它在数据集中出现的次数。适用于类别种类非常多 (High Cardinality) 的情况。
print("执行 Count Encoding ('embark_town' 转换为出现频率)...")
town_counts = df_encoded["embark_town"].value_counts()
df_encoded["embark_town_count"] = df_encoded["embark_town"].map(town_counts)

print("\n编码后的数据预览 (部分列):")
print(
    df_encoded[
        [
            "sex",
            "sex_binary",
            "class",
            "class_label",
            "embarked_Q",
            "embarked_S",
            "embark_town",
            "embark_town_count",
        ]
    ].head(3)
)


# ==========================================
# 模块 7: 总结与可视化呈现
# ==========================================
def summarize_df(df):
    """创建一个美化的 DataFrame 总结表"""
    summary = pd.DataFrame(
        {
            "数据类型 (dtype)": df.dtypes,
            "缺失值数量 (NaN)": df.isnull().sum(),
            "唯一值数量 (Unique)": df.nunique(),
            "示例数据 (First Row)": df.iloc[0],
        }
    )
    print("\n" + "=" * 60)
    print("🚀 TITANIC 数据集清洗后最终状态统计")
    print("=" * 60)
    print(summary)
    print("=" * 60)
    print(f"当前样本总数: {len(df)}")
    print("=" * 60)


summarize_df(titanic)

# --- 绘图区 ---
sns.set_style("darkgrid")

# 1. 幸存者性别分布
plt.figure(figsize=(8, 4))
ax = sns.countplot(x="sex", data=titanic, palette="Set2")
plt.title("Gender Distribution of Titanic Passengers", fontsize=14)
plt.xlabel("Sex", fontsize=12)
plt.ylabel("Count", fontsize=12)

# 给柱状图加上数字标签
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.show()

# 2. 票价与年龄的关系图 (受过异常值处理后)
plt.figure(figsize=(8, 4))
sns.scatterplot(x="fare", y="age", hue="survived", data=titanic, alpha=0.7)
plt.title("Fare vs Age (Outliers Capped)", fontsize=14)
# plt.show()


male_survivors = titanic[(titanic["sex"] == "male") & (titanic["survived"] == 1)]

print(male_survivors["survived"].count())


def survival_stats(data):
    """
    Calculates survival percentages for males and females.

    Args:
      data: Pandas DataFrame containing Titanic data.

    Returns:
      None. Prints the calculated statistics.
    """

    # Calculate survival percentages
    male_survived = data[(data["survived"] == 1) & (data["sex"] == "male")][
        "survived"
    ].count()
    female_survived = data[(data["survived"] == 1) & (data["sex"] == "female")][
        "survived"
    ].count()
    percent_male_survived = (
        male_survived / data[data["sex"] == "male"]["survived"].count()
    )
    percent_female_survived = (
        female_survived / data[data["sex"] == "female"]["survived"].count()
    )

    print("Percentage of males who survived: {:.2f}%".format(percent_male_survived))
    print("Percentage of females who survived: {:.2f}%".format(percent_female_survived))


titanic_data = sns.load_dataset("titanic")
survival_stats(titanic_data)


def most_common_embarkation(data):
    """
    Finds the embarkation port with the highest number of passengers.
    Args:
        data: pandas DataFrame containing the Titanic dataset.
    Returns:
        Most common embarkation port ('C','Q','S').
    """
    # write your code here

    print(data["embarked"])


common_port = most_common_embarkation(titanic_data)
print("Most common embarkation port:", common_port)
