# 🎓 Smart Outcome Predictor

**Ensemble Learning system that predicts student course outcomes — completion (classification) and final score (regression) — using Bagging, Boosting, Voting, and Stacking.**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-enabled-success)
![LightGBM](https://img.shields.io/badge/LightGBM-enabled-success)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📌 Problem Statement

An EdTech analytics platform needs to predict two things for every student, using engagement, demographic, and assessment data:

| Task | Target Column | Type |
|---|---|---|
| Will the student finish the course? | `completion_status` (0/1) | Classification |
| What will their final score be? | `final_score` (0–100) | Regression |

A single model gave unstable results, so this project builds and compares **ensemble methods** — models that combine multiple learners — to get more accurate, robust predictions.

---

## 🗂️ Repository Structure

```
smart-outcome-predictor/
├── Smart_Outcome_Predictor_Simplified.ipynb   # main notebook (all parts A–G)
├── student_data.csv                           # dataset
└── README.md
```

---

## 📊 Dataset

5,200 student records with 19 columns, including:

- **Demographics** — age, country region, device type, education background
- **Engagement** — sessions, time spent, videos watched, forum posts, attendance rate
- **Assessment** — quiz attempts, average quiz score, assignments submitted
- **Targets** — `completion_status`, `final_score`

A small number of rows (under 2%) had missing values in `time_spent_hours`, `avg_quiz_score`, and `attendance_rate` — handled with median imputation.

---

## 🧠 Methods Implemented

| Category | Models |
|---|---|
| **Baseline** | Decision Tree |
| **Bagging** | Bagging Classifier / Regressor |
| **Boosting** | AdaBoost, Gradient Boosting, LightGBM, XGBoost |
| **Voting** | Hard Voting, Soft Voting |
| **Stacking** | Stacking Classifier / Regressor (meta-learner: Logistic/Linear Regression) |

Every model is trained on an 80/20 train-test split and scored using:
- **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Regression:** MAE, RMSE, R²

---

## 🏆 Results

### Classification — predicting `completion_status`

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **Gradient Boosting** ⭐ | 0.737 | **0.624** | 0.784 |
| Stacking Classifier | 0.740 | 0.620 | 0.787 |
| Voting (Soft) | 0.734 | 0.619 | 0.778 |
| Voting (Hard) | 0.741 | 0.618 | — |
| XGBoost | 0.725 | 0.616 | 0.771 |
| Bagging | 0.732 | 0.613 | 0.776 |
| AdaBoost | 0.738 | 0.608 | 0.786 |
| LightGBM | 0.712 | 0.593 | 0.765 |
| Decision Tree (Base) | 0.681 | 0.551 | 0.704 |

**Best model: Gradient Boosting Classifier (highest F1)**

### Regression — predicting `final_score`

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Stacking Regressor** ⭐ | 7.88 | 9.80 | **0.486** |
| Gradient Boosting | 7.91 | 9.82 | 0.484 |
| Voting Regressor | 7.94 | 9.89 | 0.476 |
| Bagging Regressor | 7.97 | 9.93 | 0.472 |
| XGBoost | 8.06 | 10.01 | 0.464 |
| LightGBM | 8.18 | 10.16 | 0.448 |
| AdaBoost | 8.55 | 10.51 | 0.409 |
| Decision Tree (Base) | 8.88 | 11.23 | 0.325 |

**Best model: Stacking Regressor (highest R²)**

> Every ensemble method beats the single Decision Tree baseline on both tasks — confirming that combining models reduces error and improves stability, as expected.

---

## 💡 Key Takeaways

- **Bagging vs Boosting:** Bagging mainly reduces variance (stabilizes an already-decent model). Boosting reduces bias (turns weak learners into a strong one), and edged out bagging here on both tasks.
- **XGBoost vs LightGBM vs Gradient Boosting:** Plain Gradient Boosting was competitive with the optimized libraries on this dataset size; XGBoost/LightGBM's speed advantage matters more on larger data.
- **Voting vs Stacking:** Stacking won on regression, voting was competitive on classification — stacking's extra meta-model training doesn't always pay off, and is worth validating rather than assuming.
- **Deployment pick:** Gradient Boosting (classification) and Stacking Regressor (regression) — both within margin-of-error of simpler alternatives, so production teams should also weigh training/inference cost, not just the leaderboard score.

---

## ⚠️ Known Limitation

Categorical features (`country_region`, `device_type`, `education_background`, `course_level`, `course_category`) are encoded with `LabelEncoder` for simplicity. This is fine for tree-based models but technically incorrect for the linear models used inside Voting/Stacking (Logistic/Linear Regression), since it implies a false numeric ordering between categories. Results are still reasonable, but a production version should switch to one-hot encoding for those models.

---

## 🛠️ Setup & Usage

```bash
# clone the repo
git clone https://github.com/<your-username>/smart-outcome-predictor.git
cd smart-outcome-predictor

# install dependencies
pip install pandas numpy scikit-learn matplotlib lightgbm xgboost jupyter

# run the notebook
jupyter notebook Smart_Outcome_Predictor_Simplified.ipynb
```

---

## 📁 Submission Checklist

- [x] Practical implementation in Jupyter Notebook
- [x] All models implemented (Bagging, Boosting, Voting, Stacking)
- [x] Evaluation tables and plots with interpretations under each result
- [x] Final model recommendation
- [x] README with project overview and results

---

*Built as part of the Supervised Learning / Ensemble Methods module.*
