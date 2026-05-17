# 🏠 AI/ML Internship — Week 4: Supervised Learning: Regression

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn)
![Week](https://img.shields.io/badge/Week-4%20of%208-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

**Name:** Usman Asif
**Date:** 17th May, 2026  
**Program:** AI/ML Internship — Digitech Offerings  
**Instructor:** Zain Ul Abideen

---

## 📌 Project Overview

This project trains, evaluates, and compares **5 regression algorithms** on the California Housing dataset to predict median house values. It covers the full ML workflow — from data preprocessing and feature engineering to residual diagnostics, cross-validation, hyperparameter tuning, and model deployment via `joblib`.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Dataset** | California Housing (sklearn built-in) |
| **Samples** | 20,640 |
| **Features used** | 8 (top correlated with target) |
| **Target** | Median House Value (log-transformed) |
| **Source** | `sklearn.datasets.fetch_california_housing` |

---

## 🤖 Models Trained

| # | Model | Regularization |
|---|---|---|
| 1 | Linear Regression | None (baseline) |
| 2 | Ridge Regression | L2 — shrinks all coefficients |
| 3 | Lasso Regression | L1 — automatic feature selection |
| 4 | ElasticNet | L1 + L2 combined |
| 5 | Polynomial + Ridge Pipeline | Degree-2 expansion + L2 |

---

## 🏆 Best Model Results

> **Best Model: Polynomial + Ridge Pipeline**

| Metric | Value |
|---|---|
| **Test R²** | *see notebook output* |
| **Test RMSE** | *see notebook output* |
| **CV R² (5-fold)** | *see notebook output* |

**Key Insight:** Degree-2 polynomial feature expansion captured non-linear interactions (e.g., income × location) that linear models cannot represent. Ridge regularisation prevented overfitting from the expanded feature space, producing the strongest generalisation across all five models.

---

## 📁 Repository Structure

```
AIML-Internship-Week4-UsmanAsif/
│
├── Week4_SyedAliTariq.ipynb       # Main notebook — all 18 steps with outputs
├── week4_dashboard.png            # 6-chart model evaluation dashboard
├── week4_best_model.pkl           # Saved best pipeline (joblib)
└── README.md                      # This file
```

---

## 📋 Notebook Contents (18 Steps)

### Part A — Data Preparation & Baseline Model
- **Step 1** — Environment setup, library imports, dataset loading
- **Step 2** — Feature selection (top 8 by correlation), log-transform target
- **Step 3** — Train/test split (80/20), StandardScaler fit on train only
- **Step 4** — `evaluate_model()` and `plot_actual_vs_predicted()` utility functions
- **Step 5** — Linear Regression baseline, coefficient analysis

### Part B — Polynomial & Regularized Models
- **Step 6** — Polynomial Regression degree comparison (1, 2, 3)
- **Step 7** — Ridge alpha sweep + GridSearchCV
- **Step 8** — Lasso feature elimination analysis
- **Step 9** — ElasticNet alpha × l1_ratio heatmap
- **Step 10** — Polynomial + Ridge sklearn Pipeline with GridSearchCV
- **Step 11** — 5-model comparison table (MAE, RMSE, R², Adj.R², MAPE, Dollar RMSE)

### Part C — Diagnostics & Cross-Validation
- **Step 12** — Residual diagnostic 2×2 plots + Shapiro-Wilk test
- **Step 13** — 5-Fold CV R² box plots for all models
- **Step 14** — Learning curves (bias-variance diagnosis)
- **Step 15** — Ridge & Lasso coefficient path plots

### Part D — Dashboard, Predictions & Report
- **Step 16** — 6-chart evaluation dashboard (`week4_dashboard.png`)
- **Step 17** — Dollar-value predictions, joblib save/reload, over/under prediction analysis
- **Step 18** — 7-section written analysis report (Executive Summary → Reflection)

---

## 📈 Dashboard Preview

<img width="2400" height="2700" alt="week4_dashboard" src="https://github.com/user-attachments/assets/2c60ad09-61eb-4d23-8956-5d64970c426a" />


---

## 🛠 Tools & Libraries

| Library | Version | Use |
|---|---|---|
| `scikit-learn` | 1.3+ | All models, Pipeline, GridSearchCV, CV |
| `numpy` | 1.24+ | Numerical operations, residuals |
| `pandas` | 2.0+ | Data manipulation, results tables |
| `matplotlib` | 3.7+ | All plots and dashboard |
| `seaborn` | 0.12+ | ElasticNet heatmap, comparison plots |
| `scipy` | 1.11+ | Shapiro-Wilk test, Q-Q plot |
| `joblib` | — | Model serialization |

---

## ▶️ How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `Week4_UsmanAsif.ipynb`
3. Go to **Runtime → Run all**
4. No external datasets needed — California Housing loads automatically via sklearn

---

## 📚 Key Concepts Covered

- **Bias-Variance Tradeoff** — diagnosed via learning curves
- **L1 vs L2 Regularization** — mathematical intuition and practical comparison
- **Data Leakage Prevention** — scaler fitted inside CV loop via Pipeline
- **Residual Analysis** — normality, homoscedasticity, Q-Q plots
- **Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation
- **Model Persistence** — joblib save and reload with verification

---

---

*Week 4 of 8 — Supervised Learning: Regression*
