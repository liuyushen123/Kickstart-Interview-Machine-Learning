# Data Analysis Module

Welcome to the Data Analysis section of the Kickstart Interview project. This module focuses on statistical foundations, exploratory data analysis (EDA), and data visualization using Python.

## 📂 Directory Structure

```text
data_analysis/
├── data/                   # Datasets (Ignored in version control)
│   ├── raw/                # Original, immutable datasets
│   ├── processed/          # Cleaned datasets ready for modeling
│   └── metadata/           # Data dictionaries and schema descriptions
├── materials/              # Reference materials, PDFs, and slides
├── notebooks/              # Jupyter notebooks for interactive EDA
│   └── 01_titanic_eda.ipynb
└── scripts/                # Executable Python scripts
    └── descriptive_stats.py
```

## 🚀 Getting Started

### 1. Environment Setup
Ensure you are running the project within your virtual environment (`.venv`). From the root directory, install the necessary dependencies:

```bash
pip install -r ../requirements.txt
```

### 2. Running Scripts
To run the statistical tests and terminal-based Titanic summary:

```bash
python scripts/descriptive_stats.py
```

### 3. Running Interactive Notebooks
For visual and interactive data analysis, open the notebooks directly in VS Code (requires the Jupyter extension) or launch them via the terminal:

```bash
jupyter notebook
```

## 📊 Current Analysis Focus
* **Statistical Concepts:** Implementing and verifying measures of central tendency and dispersion (standard deviation, variance) using `numpy`.
* **Titanic Dataset EDA:** Exploring survival rates based on passenger class, age, and fare. Key tasks include handling missing data, type conversion (categorical data), and identifying outliers using the Interquartile Range (IQR) method.