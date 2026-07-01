# Credit Card Fraud Detection

A supervised machine learning project that detects fraudulent credit card transactions in a highly imbalanced dataset (0.17% fraud rate), using XGBoost tuned for business cost rather than raw accuracy.

## Why This Project Is Different From a Basic "Fraud Classifier"

Most beginner fraud projects optimize for accuracy. That metric is useless here: a model that never flags fraud is already 99.83% accurate while catching zero fraud. This project instead:

- Uses **Precision, Recall, F1, and PR-AUC** as the real scoring metrics.
- Tests multiple ways to handle class imbalance (not just one).
- Picks a **decision threshold based on cost**, not the default 0.5 cutoff.
- Translates model performance into **money saved/lost**, not just a confusion matrix.

## Dataset
📁  Dataset:  Credit Card Fraud Detection
🔗  Kaggle Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


- Source: anonymized European credit card transactions (PCA-transformed features `V1`–`V28`, plus `Time` and `Amount`).
- 284,807 total transactions; only 492 are fraud (0.17%).
- This notebook trains on a 50,000-row stratified sample for speed. See [Limitations](#limitations--next-steps).

## Approach

### 1. Handling Class Imbalance
Three strategies were tested on a baseline Logistic Regression model:

| Strategy | Result |
|---|---|
| Class-weighting (no resampling) | Perfect recall, but precision of only 0.065 — flags almost everything as fraud, unusable in practice |
| Random undersampling | Discards majority-class data, weaker overall signal |
| **SMOTE (oversampling)** | Best precision/recall balance — chosen approach |

Random Forest was then trained on SMOTE-resampled data. XGBoost used its built-in `scale_pos_weight` parameter instead, since gradient boosting already handles imbalance internally without needing resampled data.

### 2. Model Selection
Four models were compared: Logistic Regression, Random Forest, baseline XGBoost, and tuned XGBoost (via `RandomizedSearchCV`, optimized directly for PR-AUC).

### 3. Threshold Selection
Instead of using the default 0.5 classification cutoff, the threshold was tuned to maximize F1-score, and a cost-benefit analysis was run across multiple thresholds to confirm the choice also made business sense.

## Results

**Recommended model: Tuned XGBoost, threshold = 0.644**

| Metric | Score |
|---|---|
| Precision | 1.00 |
| Recall | 0.88 |
| F1-score | 0.94 |
| PR-AUC | 0.93 |

This outperformed the untuned XGBoost baseline (PR-AUC 0.90) and all other models tested.

An alternative high-recall threshold (≈0.007) was also calculated for use cases where catching every fraud case matters more than analyst workload — it reaches Recall = 0.90 but precision drops to 0.46 (more false alarms).

### Business Impact (per 10,000 transactions, 17 actual fraud cases)

Using assumptions of ₹4,500 average fraud value and ₹150 cost per investigated false alarm:

| Outcome | Count | Value |
|---|---|---|
| Fraud correctly caught | 15 | ₹67,500 |
| False alarms investigated | 0 | -₹2,250 |
| Fraud missed | 2 | -₹9,000 |
| **Net benefit** | | **₹56,250** (≈ ₹5.63 per transaction scored) |

A threshold sensitivity analysis (testing 0.1, 0.2, 0.3, 0.5, and the F1-optimal value) confirmed the F1-optimal threshold also maximized net business value in this run — though the report notes this isn't guaranteed in general, and the threshold should be re-validated periodically against real cost data.

## Project Structure

```
.
├── FraudDetection_SupervisedLearning.ipynb   # Full analysis, modeling, and evaluation
├── creditcard.csv                            # Raw transaction dataset
├── fraud_detection_model.pkl                 # Saved trained pipeline (tuned XGBoost)
├── summary_report.md                         # Business-facing write-up of findings
├── requirements.txt                          # Python dependencies
└── README.md
```

## How to Run

```bash
# 1. Clone the repo and enter the folder
git clone <repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook FraudDetection_SupervisedLearning.ipynb
```

To use the saved model directly instead of re-running training:

```python
import joblib
model = joblib.load("fraud_detection_model.pkl")
predictions = model.predict_proba(X_new)[:, 1]  # fraud probability
fraud_flag = predictions >= 0.644  # use the F1-optimal threshold
```

## Limitations & Next Steps

This is a trained model, not a production system. Before deployment, it would need:

1. **Real-time scoring** — current pipeline is batch/offline; production needs a low-latency API (e.g. FastAPI) scoring transactions at authorization time.
2. **Drift monitoring** — fraud patterns change over time; feature distributions and PR-AUC should be tracked weekly against fresh labelled data.
3. **Feedback loop** — confirmed outcomes from investigated cases should feed back into periodic retraining, not a static one-time model.
4. **Updated cost assumptions** — average fraud value and investigation cost will change over time; the threshold should be recalculated when they do.
5. **Full dataset training** — this notebook trained on a 50,000-row sample for speed; production should use the full 284,807-row dataset (and more historical data if available).

## Tech Stack

Python, pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, matplotlib, seaborn, joblib.

## video link
https://drive.google.com/drive/u/0/folders/1G97R_D0Z4Hzmu_eLgGLRcCuF687l7tBL
