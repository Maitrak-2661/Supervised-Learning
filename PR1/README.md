# 🏠 Predictive House Price System

> **Predictive Insight Engine** — Supervised Learning Project
> Red & White Skill Education | Duration: 6 Hours

---

## 📌 Overview

A complete end-to-end supervised machine learning pipeline for a real estate analytics firm. The system predicts house prices in INR using 10 property features and demonstrates why certain models outperform others — covering the full ML lifecycle from data exploration to business reporting.

**Pipeline:** Data Prep → Simple LR → Multiple LR → Polynomial LR → Gradient Descent → Bias-Variance Diagnostics → Final Report

---

## 📁 Project Structure

```
Predictive-House-Price-System/
│
├── Predictive_House_price_system.ipynb    # Main notebook — Parts A through I
├── RealEstate_HousePrice_Dataset_4200.csv # Dataset (4,200 houses, 12 columns)
└── README.md                              # This file
```

---

## 📊 Dataset — `RealEstate_HousePrice_Dataset_4200.csv`

| Property | Value |
|---|---|
| Total Records | 4,200 houses |
| Total Columns | 12 (1 ID + 10 features + 1 target) |
| Missing Values | **None** — dataset is clean |
| Price Range | ₹8,00,000 — ₹7,61,11,718 |
| Mean Price | ₹2,36,41,886 |
| Median Price | ₹2,21,45,004 |

### Feature Descriptions

| Column | Type | Role | Description |
|---|---|---|---|
| `house_id` | int | Identifier | Unique ID — not used in modelling |
| `area_sqft` | int | Feature | Total built-up area in square feet |
| `bedrooms` | int | Feature | Number of bedrooms |
| `bathrooms` | int | Feature | Number of bathrooms |
| `location_score` | float | Feature | Location desirability score |
| `age_years` | int | Feature | Age of the property in years |
| `distance_city_km` | float | Feature | Distance from nearest city centre (km) |
| `lot_size_sqft` | int | Feature | Total plot size in square feet |
| `has_garage` | int | Feature | Garage present: 1 = Yes, 0 = No |
| `has_pool` | int | Feature | Pool present: 1 = Yes, 0 = No |
| `renovation_years_ago` | int | Feature | Years since last renovation |
| `house_price_inr` | int | **Target** | House price in Indian Rupees |

### Feature Correlations with Price

```
area_sqft             +0.7554  ██████████████████████████
bedrooms              +0.6448  ██████████████████████
location_score        +0.5885  ████████████████████
lot_size_sqft         +0.5678  ███████████████████
bathrooms             +0.5270  ██████████████████
has_garage            +0.1769  ██████
has_pool              +0.1020  ████
renovation_years_ago  -0.0374  █
age_years             -0.0895  ███
distance_city_km      -0.4694  ████████████████  (negative)
```

---

## 🧩 Notebook Parts — A through I

### Part A — Conceptual Understanding

Six theory questions answered in the notebook:

| Q | Topic | Key Takeaway |
|---|---|---|
| Q1 | Supervised Learning | Learns from labelled data: f(X) → y |
| Q2 | Regression vs Classification | Regression = continuous output; Classification = category output |
| Q3 | Simple Linear Regression | y = β₀ + β₁·X + ε; one feature, one line |
| Q4 | Assumptions of Linear Regression | Linearity, Independence, Homoscedasticity, Normality, No multicollinearity |
| Q5 | Bias-Variance Trade-off | Total error = Bias² + Variance + Irreducible noise |
| Q6 | Overfitting vs Underfitting | Underfit = high bias; Overfit = high variance |

---

### Part B — Dataset Understanding & Preparation

- Loaded dataset: 4,200 rows × 12 columns
- Confirmed **zero missing values** across all columns
- Visualised all 10 features vs price (scatter plots)
- Generated correlation heatmap
- Identified top correlations: `area_sqft` (0.75), `bedrooms` (0.64), `location_score` (0.59)
- Identified strongest negative: `distance_city_km` (-0.47)

**Train / Test Split:**

```
Total  : 4,200 samples
Train  : 3,360 samples (80%)
Test   :   840 samples (20%)
Seed   : random_state = 18
```

---

### Part C — Simple Linear Regression

Used **only `area_sqft`** to predict price.

```
Model     : y = β₀ + β₁ · area_sqft
Slope     : ₹14,804 per sq.ft
Intercept : -₹9,89,589  (no real-world meaning)
```

**Assumption Validation:**
- Residuals plotted vs predicted values — spread above and below zero ✅
- Residual histogram — roughly bell-shaped ✅
- Wide spread confirms the model is directionally correct but imprecise due to missing features

---

### Part D — Model Evaluation Metrics

Metrics used throughout the project:

| Metric | What It Measures | Better When |
|---|---|---|
| **MAE** | Average absolute error in ₹ | Lower |
| **MSE** | Mean squared error — punishes large errors more | Lower |
| **RMSE** | √MSE — same unit as price (₹) | Lower |
| **R²** | % of price variation the model explains | Higher (max = 1.0) |

**SLR Results (confirmed from notebook):**

```
MAE  : ₹61,31,759  (₹61.32 Lakhs)
RMSE : ₹79,17,171  (₹79.17 Lakhs)
R²   : 0.5809
```

> SLR explains only 58% of price variation. With average errors of ₹61L, this model is not usable for a real estate business.

---

### Part E — Multiple Linear Regression

Used **all 10 features** simultaneously.

**MLR Coefficients — Price Impact per Unit:**

| Feature | Coefficient | Business Meaning |
|---|---|---|
| `location_score` | +₹30,83,386 | 1 point better location = +₹30.8L |
| `has_pool` | +₹3,84,723 | Pool present = +₹3.8L |
| `bedrooms` | +₹1,96,843 | Each extra bedroom = +₹2.0L |
| `bathrooms` | +₹1,84,599 | Each extra bathroom = +₹1.8L |
| `has_garage` | +₹1,73,520 | Garage present = +₹1.7L |
| `area_sqft` | +₹13,945 | Each sq.ft = +₹14K |
| `lot_size_sqft` | +₹56 | Minimal impact |
| `renovation_years_ago` | -₹27,117 | Each year since renovation = -₹27K |
| `age_years` | -₹66,695 | Each year of age = -₹67K |
| `distance_city_km` | -₹1,02,945 | Each km farther from city = -₹1.03L |

**MLR Results (confirmed from dataset):**

```
Train R² : 0.9230
Test  R² : 0.9247
MAE      : ₹25.19 Lakhs
RMSE     : ₹33.56 Lakhs
```

> MAE dropped from ₹61.3L to ₹25.2L — a **59% reduction** just by adding 9 more features.

---

### Part F — Polynomial Regression

Tested on `area_sqft` only, degrees 2 and 3.

| Degree | Train R² | Test R² | MAE | RMSE | Verdict |
|---|---|---|---|---|---|
| 1 (Linear) | 0.5682 | 0.5809 | ₹61.32L | ₹79.17L | Underfitting |
| 2 (Quadratic) | 0.5687 | 0.5789 | ₹61.47L | ₹79.36L | Underfitting |
| 3 (Cubic) | 0.5689 | 0.5789 | ₹61.46L | ₹79.37L | Underfitting |

**Key finding:** R² stays at ~0.58 regardless of degree. Increasing polynomial degree on a single feature does not fix underfitting. The problem is missing features, not model complexity. MLR already solves this by using all 10 features.

---

### Part G — Gradient Descent Optimization

All three variants implemented **from scratch using NumPy** (no sklearn).

**Data preparation:**
- Features scaled with `StandardScaler`
- Target scaled: `y / 1,000,000` (₹ millions — for numerical stability)
- Bias column (ones) prepended to X matrix

**Implementations:**

```python
# Batch GD — uses full dataset per weight update
def batch_gd(X, y, lr=0.01, epochs=1000)

# Stochastic GD — uses 1 random sample per weight update
def stochastic_gd(X, y, lr=0.01, epochs=50)

# Mini-Batch GD — uses 32 samples per weight update
def mini_batch_gd(X, y, lr=0.01, epochs=100, batch_size=32)
```

**Comparison:**

| Method | Learning Rate | Epochs | Batch Size | Loss Curve | Speed |
|---|---|---|---|---|---|
| Batch GD | 0.01 | 1,000 | 4,200 (full) | Very smooth | Slow |
| SGD | 0.01 | 50 | 1 | Very noisy | Fast per epoch |
| Mini-Batch GD | 0.01 | 100 | 32 | Moderate noise | Balanced |

> **Mini-Batch GD** is the production-ready choice — converges in 100 epochs with stable, smooth-enough loss reduction.

---

### Part H — Bias-Variance & Model Diagnostics

Analysis across all three model types:

| Model | Features | Train R² | Test R² | Gap | Bias | Variance | Diagnosis |
|---|---|---|---|---|---|---|---|
| Simple LR | 1 | 0.5682 | 0.5809 | 0.0127 | **High** | Low | Underfitting |
| Polynomial deg=2 | 1 | 0.5687 | 0.5789 | 0.0102 | **High** | Low | Underfitting |
| **Multiple LR** | **10** | **0.9230** | **0.9247** | **0.0017** | **Low** | **Low** | **✅ Best Balance** |

**How complexity affects error:**
- Low complexity (1 feature, any degree) → high bias, train and test R² both stuck at ~0.58
- Optimal complexity (10 features, linear) → low bias, near-zero train-test gap
- Excessive complexity (high-degree polynomial on single feature) → train R² climbs while test R² drops → overfitting begins

**Winner: Multiple Linear Regression** — highest R², lowest errors, no overfitting.

---

### Part I — Final Analysis & Reporting

#### Complete Model Comparison

| Model | Train R² | Test R² | MAE | RMSE | Status |
|---|---|---|---|---|---|
| Simple Linear Regression | 0.5682 | 0.5809 | ₹61.32L | ₹79.17L | ❌ Underfitting |
| Polynomial Regression (deg=2) | 0.5687 | 0.5789 | ₹61.47L | ₹79.36L | ❌ Underfitting |
| Polynomial Regression (deg=3) | 0.5689 | 0.5789 | ₹61.46L | ₹79.37L | ❌ Underfitting |
| **Multiple Linear Regression** | **0.9230** | **0.9247** | **₹25.19L** | **₹33.56L** | **✅ Winner** |

#### Gradient Descent — Final Verdict

| Method | Convergence | Stability | Recommended For |
|---|---|---|---|
| Batch GD | Guaranteed but slow | Very stable | Small datasets, debugging |
| SGD | Fast but erratic | Noisy | Very large datasets |
| **Mini-Batch GD** | **Fast and reliable** | **Stable** | **Production use** |

#### Business Interpretation

MLR coefficients translate directly into pricing decisions:

- **Location is the #1 driver.** A 1-point improvement in `location_score` = +₹30.83L — more impactful than any physical feature of the house.
- **Distance from city penalises value.** Every additional km from the city = -₹1.03L. A property 20 km farther is worth ~₹20.6L less, all else equal.
- **Age erodes value at ₹66,695 per year.** A 20-year-old house is worth ~₹13.3L less than a new one.
- **Area adds ₹13,945 per sq.ft.** A 500 sq.ft larger house is worth ~₹69.7L more.
- **Amenities add moderate value.** Pool (+₹3.8L), garage (+₹1.7L) — useful for pricing but not primary drivers.
- **Practical accuracy:** Average error of ₹25.19L on a median price of ₹2.21 Crore = **~11.4% average error** — operationally useful for property valuation, loan assessments, and investment decisions.

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
# 1. Place both files in the same directory
# 2. Launch Jupyter
jupyter notebook Predictive_House_price_system.ipynb

# 3. Run all cells: Kernel → Restart & Run All
```

> ⚠️ **Important:** Run cells strictly top-to-bottom. Parts H and I depend on variables (`slr`, `mlr`, `model_d2`, `all_result`, `losses`, `sgd_losses`, `mini_losses`) defined in earlier parts.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Python 3.x | Core language |
| NumPy | Gradient descent from scratch, matrix operations |
| Pandas | Data loading, EDA, coefficient analysis |
| Matplotlib | All plots and visualisations |
| Seaborn | Correlation heatmap |
| Scikit-learn | LinearRegression, PolynomialFeatures, StandardScaler, metrics, train_test_split |
| Jupyter Notebook | Interactive development environment |

---

## ✅ Submission Checklist

- [x] **Part A** — Theory: Supervised learning, regression vs classification, SLR, assumptions, bias-variance, overfitting
- [x] **Part B** — EDA: scatter plots, correlation heatmap, train-test split (80/20, seed=18)
- [x] **Part C** — SLR: regression line plotted, residual assumption validation
- [x] **Part D** — Metrics: MAE, MSE, RMSE, R² defined and computed for SLR
- [x] **Part E** — MLR: all 10 features, coefficients analysed, metrics computed
- [x] **Part F** — Polynomial deg=2 and deg=3: compared with linear, overfitting checked
- [x] **Part G** — Gradient Descent from scratch: Batch GD, SGD, Mini-Batch GD, convergence compared
- [x] **Part H** — Bias-Variance: train vs test R² gap across all models, best model identified
- [x] **Part I** — Final report: model comparison, GD analysis, business interpretation
- [x] All charts saved with labels and interpretations
- [x] GitHub repository with notebook, dataset, and README

---



