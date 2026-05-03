import random

import numpy as np
import seaborn as sns

# ==========================================
# PART 1: STATISTICAL CONCEPTS & DISPERSION
# ==========================================
"""
--- Central Tendency ---
* Mean: Doesn't always work because it might be skewed by outliers.
* Median: Good for skewed data, but its problem is it relies on one (or two) middle observations.
* Mode: The most frequently occurring value.
* Others: Harmonic mean and graphical mean.

--- Study of Dispersion ---
* Range: The highest value - smallest value.
* Inter Quartile Range (IQR): Q3 - Q1 (It tells you the middle 50% range).
* Variance Logic: The sum of all deviations (x - mean) will exactly equal 0. 
  Because of this, we need to square everything to remove negative signs, average them (Variance), 
  and then take the square root to get it back to the original unit (Standard Deviation).
"""

# Experiment: Proving the sum of deviations from the mean is zero
print("--- PART 1: NUMPY VARIANCE EXPERIMENT ---")
data1 = np.array([random.randint(1, 1000) for _ in range(1, 1000)])
data2 = data1 - data1.mean()

# We round it to 5 decimal places to hide the floating-point anomaly
print(f"Sum of (x - mean): {round(np.sum(data2), 5)}\n")


# ==========================================
# PART 2: TITANIC EXPLORATORY DATA ANALYSIS
# ==========================================
# Loading titanic data from seaborn (returns a Pandas DataFrame)
titanic = sns.load_dataset("titanic")

print("--- PART 2: TITANIC DATASET INFO ---")
print(titanic.info())

print("\n--- Missing Values Count ---")
print(titanic.isnull().sum())

print("\n--- Numerical Data Summary ---")
print(titanic.describe())

"""
TITANIC DATASET - DESCRIPTIVE STATISTICS SUMMARY (N=891)
--------------------------------------------------------
Target Variable: 'survived' (Mean: 0.38 -> 38.4% Survival Rate)

FEATURE ANALYSIS:
1. Pclass (Passenger Class):
   - Distribution: Mean 2.31, Median 3.0.
   - Insight: Heavily skewed towards 3rd class. Significant feature for survival prediction.
   
2. Age (Continuous):
   - Count: 714 (177 Missing values / ~20% NaN).
   - Range: 0.42 to 80.0 years.
   - Distribution: Average age ~30, but the median (28) is lower, indicating a slight right-skew.

3. Family Features (SibSp & Parch):
   - SibSp: # of siblings/spouses. 75th percentile is 1.0, Max is 8.
   - Parch: # of parents/children. 75th percentile is 0.0, Max is 6.
   - Insight: Most passengers traveled alone (Median = 0 for both).

4. Fare (Ticket Price):
   - Statistics: Mean 32.20, Median 14.45, Max 512.33.
   - Insight: Extreme variance (Std Dev: 49.69). The high Max suggests outliers/premium suites 
              that significantly pull the mean away from the median.

DATA PREPROCESSING NOTES:
- Handle missing values in 'age'.
- Consider One-Hot Encoding for 'pclass'.
- Address 'fare' outliers for linear models.
"""

print("\n--- Categorical Data Summary ---")
# Optional: converting 'sex' to category so it appears in this describe block
titanic["sex"] = titanic["sex"].astype("category")
print(titanic.describe(include=["category"]))
