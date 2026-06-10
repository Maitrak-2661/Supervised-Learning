# 🏦 Risk Alert Classifier — Early Warning System for Digital Banking

A supervised machine learning project that identifies high-risk customers likely to default on payments or engage in fraudulent behavior. Built using a full classification pipeline with imbalance handling, hyperparameter tuning, and ROC-based model evaluation.

---

## 📌 Problem Statement

A digital banking platform needs an **early-warning system** to flag customers who are likely to default or commit fraud — before it happens.

The challenge: the dataset is **highly imbalanced** (87% low-risk vs 13% high-risk customers), which makes standard accuracy metrics misleading and requires special handling.

---

## 📁 Project Structure

```
risk-alert-classifier/
│
├── risk_alert.ipynb              # Main Jupyter Notebook (full pipeline)
├── risk_alert_dataset.csv        # Dataset (4600 customers, 19 features)
└── README.md                     # This file
```

---

## 📊 Dataset Overview

| Property | Details |
|---|---|
| Total Records | 4,600 customers |
| Features | 16 input features (after dropping ID & date) |
| Target Variable | `risk_status` → 0 = Low Risk, 1 = High Risk |
| Class Distribution | 4,043 Low Risk (87.9%) vs 557 High Risk (12.1%) |
| Missing Values | Yes — handled using KNN Imputer |

### Features Used

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Customer age |
| `gender` | Categorical | Gender of customer |
| `region` | Categorical | Geographic region |
| `employment_type` | Categorical | Salaried / Self-employed / Unemployed |
| `annual_income_inr` | Numeric | Yearly income in INR |
| `credit_score` | Numeric | Credit score (higher = better) |
| `credit_utilization_ratio` | Numeric | % of credit limit being used |
| `missed_payments_12m` | Numeric | Missed payments in last 12 months |
| `avg_late_payment_days` | Numeric | Average days late on payments |
| `monthly_transaction_count` | Numeric | Number of transactions per month |
| `monthly_spend_inr` | Numeric | Monthly spending in INR |
| `cash_advance_count_6m` | Numeric | Cash advances in last 6 months |
| `complaints_last_6m` | Numeric | Complaints raised in last 6 months |
| `failed_login_attempts_3m` | Numeric | Failed login attempts in 3 months |
| `account_tenure_months` | Numeric | How long customer has been with bank |
| `debt_balance_inr` | Numeric | Current outstanding debt |

---

## 🔁 Project Pipeline

```
Raw Data
   ↓
Data Cleaning & Encoding
   ↓
KNN Imputation (missing values)
   ↓
Stratified Train-Test Split (80/20)
   ↓
Baseline Model → Logistic Regression
   ↓
Imbalance Handling → Under/Over Sampling, SMOTE, ADASYN
   ↓
Tree-Based Models → Decision Tree, Random Forest
   ↓
Hyperparameter Tuning → RandomizedSearchCV + GridSearchCV
   ↓
Final Evaluation → ROC Curves, AUC, Confusion Matrices
   ↓
Best Model Selection
```

---

## 🧪 Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline model |
| Decision Tree | With and without depth limits (to show overfitting) |
| Random Forest | 100 trees, with feature importance analysis |
| All models re-trained | With SMOTE-balanced data |
| Hyperparameter Tuned | Using RandomizedSearchCV + GridSearchCV |

---

## ⚖️ Handling Class Imbalance

Four techniques were tested and compared:

| Technique | What it does |
|---|---|
| Under-Sampling | Reduces majority class to match minority |
| Over-Sampling | Duplicates minority class samples |
| SMOTE | Generates synthetic minority samples |
| ADASYN | Adaptive synthetic sampling near boundaries |

**Winner: SMOTE** — achieved the best F1-Score and perfect Recall.

---

## 📈 Final Results

| Model | Accuracy | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| **Logistic Regression + SMOTE** | **0.9978** | **1.0000** | **0.9911** | **1.0000** |
| Random Forest + SMOTE | 0.9957 | 0.9820 | 0.9820 | 0.9999 |
| Decision Tree + SMOTE | 0.9793 | 0.9279 | 0.9156 | 0.9572 |

### ✅ Best Model: Logistic Regression + SMOTE

**Why:** In banking, the priority is to catch **every single high-risk customer** (maximize Recall). Missing a risky customer (False Negative / Type-II error) leads to direct financial loss. Logistic Regression with SMOTE achieved **perfect Recall of 1.0** — zero high-risk customers missed.

---

## 🔑 Top Predictive Features (from Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `avg_late_payment_days` | 0.277 |
| 2 | `missed_payments_12m` | 0.259 |
| 3 | `cash_advance_count_6m` | 0.144 |
| 4 | `credit_score` | 0.102 |
| 5 | `credit_utilization_ratio` | 0.080 |

Payment behavior (late days + missed payments) is the strongest signal for risk — more than income or credit score alone.

---

## 🧠 Key Concepts Covered

- Logistic Regression for binary classification
- Type-I Error (False Positive) vs Type-II Error (False Negative)
- Precision, Recall, F1-Score, AUC-ROC
- Why accuracy is misleading on imbalanced data
- KNN Imputer for multivariate missing value handling
- SMOTE and ADASYN for synthetic oversampling
- Decision Tree overfitting and how to control it
- Random Forest as an ensemble method
- RandomizedSearchCV and GridSearchCV for tuning
- ROC Curve interpretation

---

## 🛠️ Tech Stack

- **Python 3.10**
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **scikit-learn** — models, metrics, preprocessing, cross-validation
- **imbalanced-learn** — SMOTE, ADASYN, RandomUnderSampler
- **matplotlib / seaborn** — visualizations

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/risk-alert-classifier.git
cd risk-alert-classifier
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

**3. Launch Jupyter Notebook**
```bash
jupyter notebook risk_alert.ipynb
```

**4. Run all cells top to bottom**

Make sure `risk_alert_dataset.csv` is in the same folder as the notebook.

---

## 📋 Notebook Structure

| Section | Tasks | Description |
|---|---|---|
| Part A | Q1–Q6 | Theory — Logistic Regression, metrics, imbalanced data |
| Part B | 7–9 | Dataset loading, encoding, KNN imputation, train-test split |
| Part C | 10–12 | Baseline Logistic Regression, confusion matrix, Type-I/II errors |
| Part D | 13–15 | Under-sampling, over-sampling, SMOTE, ADASYN comparison |
| Part E | 16–19 | Decision Tree, Random Forest, feature importance |
| Part F | 20–22 | RandomizedSearchCV, GridSearchCV, tuned vs untuned |
| Part G | 23–25 | ROC curves, AUC comparison, final model selection |
| Part H | 26–27 | Final report, business interpretation, conclusions |

---

## 💡 Business Interpretation

| Error Type | Meaning | Business Impact |
|---|---|---|
| False Positive (Type-I) | Safe customer flagged as risky | Minor — unnecessary review |
| False Negative (Type-II) | Risky customer passes as safe | **Severe** — financial loss, fraud damage |

The model is optimized to **eliminate False Negatives** — the banking scenario where missing a high-risk customer has direct monetary consequences.

---

## 👤 Author

**[Your Name]**
Student — Data Science | Red & White Skill Education

---

## 📄 License

This project is for educational purposes only.
