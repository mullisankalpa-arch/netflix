<div align="center">

# 🎬 Netflix Customer Churn & Engagement Analytics Using AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*An end-to-end machine learning pipeline to predict subscriber churn, analyse engagement patterns, and surface actionable retention strategies from a 5,000-customer Netflix dataset.*

</div>

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Technologies](#-technologies)
- [Features](#-features)
- [Workflow](#-workflow)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Business Insights](#-business-insights)

---

## 📌 Project Description

Subscriber churn is one of the most costly challenges facing streaming platforms. Losing a customer means not only lost revenue but also the high cost of re-acquisition. This project applies **supervised machine learning** and **exploratory data analysis** to a Netflix-style subscriber dataset to:

- Identify *who* is likely to churn before they leave
- Understand *why* they churn by analysing behavioural and demographic patterns
- Quantify *how much revenue* is at risk
- Recommend *targeted retention strategies* backed by data

The full analysis — from raw CSV to a deployable churn-score per customer — is contained in a single, reproducible Jupyter Notebook.

---

## ❓ Problem Statement

> **How can Netflix proactively identify subscribers at high risk of cancelling their subscription, and what engagement or demographic signals best predict that decision?**

With a churn rate of ~50% in the dataset, even a moderate improvement in retention directly translates into significant monthly recurring revenue saved. The challenge is to build a model that is both **highly accurate** and **interpretable enough** to drive real business decisions.

---

## 🎯 Objectives

| # | Objective |
|---|-----------|
| 1 | Load, clean, and validate the customer dataset |
| 2 | Perform comprehensive EDA across all feature dimensions |
| 3 | Engineer new engagement and behavioural features |
| 4 | Train and compare three ML classifiers (Logistic Regression, Random Forest, Gradient Boosting) |
| 5 | Evaluate models via accuracy, F1-score, AUC-ROC, and 5-fold cross-validation |
| 6 | Identify the top drivers of churn using feature importance |
| 7 | Segment all customers into Low / Medium / High risk tiers |
| 8 | Quantify monthly revenue at risk and surface retention recommendations |

---

## 📊 Dataset

| Property | Details |
|---|---|
| **File** | `netflix_customer_churn.csv` |
| **Rows** | 5,000 customers |
| **Columns** | 14 features |
| **Target** | `churned` (1 = churned, 0 = retained) |
| **Missing values** | None |
| **Class balance** | 50.3% churned / 49.7% retained |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Unique UUID per customer |
| `age` | int | Customer age (18 – 70) |
| `gender` | categorical | Male / Female / Other |
| `subscription_type` | categorical | Basic / Standard / Premium |
| `watch_hours` | float | Total hours watched in the period |
| `last_login_days` | int | Days since last login (0 – 60) |
| `region` | categorical | Africa, Asia, Europe, North America, Oceania, South America |
| `device` | categorical | TV / Mobile / Laptop / Desktop / Tablet |
| `monthly_fee` | float | Monthly subscription fee (USD) |
| `churned` | int | **Target** — 1 = churned, 0 = retained |
| `payment_method` | categorical | Credit Card / Debit Card / PayPal / Crypto / Gift Card |
| `number_of_profiles` | int | Active sub-profiles (1 – 5) |
| `avg_watch_time_per_day` | float | Average hours watched per day |
| `favorite_genre` | categorical | Action / Comedy / Documentary / Drama / Horror / Romance / Sci-Fi |

---

## 🛠 Technologies

| Category | Library / Tool |
|---|---|
| Language | Python 3.8+ |
| Notebook | Jupyter Notebook |
| Data Manipulation | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Models | Logistic Regression, Random Forest, Gradient Boosting |
| Evaluation | AUC-ROC, Confusion Matrix, Cross-Validation |
| Environment | pip / venv |

---

## ✨ Features

- **Zero-missing-value dataset** with automatic outlier detection and capping
- **8 EDA visualisations** — distributions, box plots, heatmaps, scatter plots, categorical churn rates
- **5 engineered features** — `engagement_score`, `is_inactive`, `high_watch`, `age_group`, `fee_per_profile`
- **Three ML models** trained and compared on identical train/test splits
- **5-fold stratified cross-validation** for robust performance estimates
- **ROC curve overlay** and **confusion matrix** for every model
- **Feature importance rankings** from both Random Forest and Gradient Boosting
- **Customer risk tiering** (Low / Medium / High) with revenue-at-risk quantification
- **Top-20 at-risk customer table** ready for CRM export
- **Actionable business recommendations** grounded in model outputs

---

## 🔄 Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     END-TO-END PIPELINE                                 │
├──────────────┬──────────────┬──────────────┬───────────────┬────────────┤
│  1. INGEST   │  2. CLEAN    │  3. EXPLORE  │  4. ENGINEER  │  5. MODEL  │
│              │              │              │               │            │
│ Load CSV     │ Check nulls  │ Distributions│ engagement_   │ Logistic   │
│ Inspect      │ Find dupes   │ Box plots    │ score         │ Regression │
│ dtypes &     │ Outlier IQR  │ Heatmaps     │ is_inactive   │            │
│ cardinality  │ Cap extremes │ Scatter      │ high_watch    │ Random     │
│              │              │ Churn rates  │ age_group     │ Forest     │
│              │              │ by category  │ fee_per_      │            │
│              │              │              │ profile       │ Gradient   │
│              │              │              │               │ Boosting   │
├──────────────┴──────────────┴──────────────┴───────────────┴────────────┤
│  6. EVALUATE             │  7. INTERPRET            │  8. INSIGHTS      │
│                          │                          │                   │
│ Accuracy / F1 / AUC-ROC  │ Feature importances      │ Risk tiers        │
│ Confusion matrices       │ ROC curve comparison     │ Revenue at risk   │
│ 5-fold cross-validation  │ Probability distribution │ Recommendations   │
└──────────────────────────┴──────────────────────────┴───────────────────┘
```

---

## 📈 Key Results

### Model Performance (Test Set — 1,000 customers)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.88 | 0.87 | 0.90 | 0.89 | 0.9587 |
| Random Forest | 0.96 | 0.97 | 0.95 | 0.96 | 0.9946 |
| **Gradient Boosting** ✅ | **0.99** | **0.99** | **0.99** | **0.99** | **0.9979** |

> ✅ **Gradient Boosting** is the recommended model — highest accuracy, best AUC, and cleanest probability separation between retained and churned customers.

### Top Churn Drivers (by Feature Importance)

| Rank | Feature | Importance |
|:---:|---|:---:|
| 1 | `engagement_score` (watch hrs ÷ recency) | 0.257 |
| 2 | `avg_watch_time_per_day` | 0.227 |
| 3 | `last_login_days` | 0.136 |
| 4 | `watch_hours` | 0.123 |
| 5 | `number_of_profiles` | 0.067 |
| 6 | `payment_method` | 0.033 |

### Engagement Gap — Churned vs Retained

| Metric | Retained | Churned | Δ |
|---|:---:|:---:|:---:|
| Avg Watch Hours | 17.45 | 5.92 | **−66%** |
| Avg Days Since Login | 21.8 | 38.3 | **+76%** |
| Avg Profiles | 3.25 | 2.80 | **−14%** |

### Churn by Subscription Plan

| Plan | Churn Rate |
|---|:---:|
| Basic | **61.8%** ⚠️ |
| Standard | 45.4% |
| Premium | **43.7%** ✅ |

### Churn by Payment Method

| Payment Method | Churn Rate |
|---|:---:|
| Crypto | **59.7%** ⚠️ |
| Gift Card | **57.8%** ⚠️ |
| PayPal | 47.1% |
| Debit Card | 43.7% |
| Credit Card | **43.6%** ✅ |

---

## 💡 Business Insights

1. **Early-Warning Alerts** — Flag customers with >45 days inactivity AND <5 total watch hours. Trigger personalised re-engagement campaigns immediately.

2. **Basic Plan Upgrade Nudge** — Basic subscribers churn at 61.8% — 18 percentage points above Premium. Offer time-limited discounted upgrades to Standard/Premium.

3. **Payment Method Risk** — Crypto and Gift Card users churn at ~58–60%. Incentivise a switch to Credit/Debit Card via loyalty points or a free month.

4. **Profile Expansion** — Churned customers average 0.45 fewer profiles. Promote family/household plan features to single-profile users.

5. **Content Personalisation** — Action and Drama fans show the highest genre-level churn. Prioritise recommendation quality in these genres with "because you watched…" nudges.

---

## 🗂 Project Structure

```
IBM Project/
│
├── netflix_customer_churn.csv        # Source dataset (5,000 rows × 14 columns)
├── Netflix_Churn_Analytics.ipynb     # Main Jupyter Notebook (full pipeline)
└── README.md                         # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### 1 — Clone / Download the project

```bash
git clone https://github.com/your-username/netflix-churn-analytics.git
cd netflix-churn-analytics
```

> Or simply download the ZIP and extract it into your working directory.

### 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or install from the requirements file if provided:

```bash
pip install -r requirements.txt
```

#### Full dependency list

| Package | Version |
|---|---|
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
| scikit-learn | ≥ 1.3 |
| jupyter | ≥ 1.0 |

---

## ▶️ How to Run

### Option A — Jupyter Notebook (recommended)

```bash
# From the project directory, with the virtual environment activated:
jupyter notebook Netflix_Churn_Analytics.ipynb
```

Then, in the browser tab that opens:

1. Click **Kernel → Restart & Run All** to execute every cell from top to bottom.
2. All plots and tables will render inline.
3. The final summary cell prints the complete business insights report.

### Option B — JupyterLab

```bash
jupyter lab Netflix_Churn_Analytics.ipynb
```

### Option C — VS Code

Open `Netflix_Churn_Analytics.ipynb` in VS Code with the **Jupyter** extension installed. Select your Python interpreter and click **Run All**.

### Expected Runtime

| Step | Approx. Time |
|---|---|
| Data loading & EDA | < 10 seconds |
| Feature engineering | < 5 seconds |
| Logistic Regression training | < 5 seconds |
| Random Forest training | ~15 seconds |
| Gradient Boosting training | ~30 seconds |
| Cross-validation (all models) | ~60 seconds |
| **Total** | **~2 minutes** |

> ⚠️ Ensure `netflix_customer_churn.csv` is in the **same directory** as the notebook before running.

---

## 📄 License

This project is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it with attribution.

---

<div align="center">

**Built with ❤️ for Data Science & AI**

*Netflix Customer Churn & Engagement Analytics Using AI*

</div>
