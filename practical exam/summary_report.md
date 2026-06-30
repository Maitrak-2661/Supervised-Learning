# Summary Report — Credit Card Fraud Detection

## Business Problem and Why Standard Accuracy Fails

Fraud detection on this dataset is an extreme class-imbalance problem: only 0.17% of transactions are fraudulent. A model that predicts "not fraud" for every transaction would be 99.83% accurate while catching zero fraud and providing zero business value. Accuracy treats every misclassification as equally costly, but in fraud detection a missed fraud case (False Negative) and a false alarm (False Positive) have very different financial consequences. For this reason, the project used Precision, Recall, F1-score, and PR-AUC (Precision-Recall Area Under Curve) as the primary metrics instead of accuracy or ROC-AUC, since PR-AUC is far more sensitive to performance on the rare fraud class.

## Imbalance Handling Strategy Chosen and Why

Three strategies were tested on Logistic Regression: class-weighting on the original data, SMOTE oversampling, and random undersampling. SMOTE gave the best F1-score balance between precision and recall, because it provides the model with more (synthetic) examples of fraud to learn a decision boundary from, rather than just up-weighting the very small number of real fraud cases. The class-weighted model on raw data achieved perfect recall but extremely poor precision (0.065), meaning it flagged almost everything as fraud — useless in practice due to the sheer volume of false alarms it would generate. Random Forest was then trained on the SMOTE-resampled data as the best identified strategy. XGBoost used its native `scale_pos_weight` parameter instead of resampling, since gradient boosting handles class weighting well internally.

## Model and Threshold Recommended

The recommended model is **tuned XGBoost** (via RandomizedSearchCV, optimised for PR-AUC), evaluated at its **F1-optimal threshold of 0.644** rather than the default 0.5 cutoff. On the held-out test set this configuration achieved Precision = 1.00, Recall = 0.88, F1 = 0.94, and PR-AUC = 0.93 — the best balance of any model tested, including Logistic Regression, Random Forest, and the untuned XGBoost baseline (PR-AUC 0.90 before tuning). An alternative threshold was also computed for scenarios requiring Recall ≥ 0.90 (threshold ≈ 0.007), which catches more fraud cases at the cost of significantly lower precision (0.46) — useful only if leadership explicitly prioritises catching fraud over reducing analyst workload.

## Cost-Benefit Result

Using assumptions of ₹4,500 average fraud value and ₹150 per investigated false alarm, the F1-optimal threshold on the test set (10,000 transactions, 17 actual fraud cases) produced: 15 true positives, 0 false positives, 2 false negatives. This translates to ₹67,500 in fraud caught, ₹2,250 in investigation cost, and ₹9,000 in fraud still missed — a **net benefit of ₹56,250 per 10,000 transactions**, or roughly **₹5.63 per transaction** scored. A threshold sensitivity analysis across 0.1, 0.2, 0.3, 0.5, and the F1-optimal value confirmed this threshold also maximises net business benefit in this run, though this is not guaranteed in general — the F1-optimal and business-optimal thresholds can diverge, and in production the threshold should be re-validated against actual cost assumptions periodically rather than assumed to be fixed.

## What Would Be Improved for Production Deployment

1. **Real-time scoring:** the current pipeline is batch/offline; production would need a low-latency API (e.g. wrapping the saved pipeline in a FastAPI service) scoring transactions in milliseconds at authorization time, not after the fact.
2. **Model drift monitoring:** fraud patterns evolve as fraudsters adapt. Feature distributions and PR-AUC should be tracked weekly against fresh labelled data, with automatic alerts if performance degrades.
3. **Feedback loop:** confirmed fraud/non-fraud outcomes from investigated cases should be fed back into a retraining pipeline on a regular cadence (e.g. monthly), rather than relying on a static model trained once.
4. **Cost assumptions should be revisited regularly** — average fraud value and investigation cost are not static, and the chosen threshold should be recalculated as these change.
5. **Full dataset training:** this notebook used a 50,000-row stratified sample for speed; the production model should be retrained on the full 284,807-row dataset (and ideally far more historical data) before deployment.
