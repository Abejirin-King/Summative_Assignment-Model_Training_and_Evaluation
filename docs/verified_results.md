# Verified Results Summary

These are the results actually executed in the available runtime. They are not the final deep-learning results.

## 28-day held-out test

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7650 | 0.8102 | 0.7213 | 0.7632 | 0.8497 | 0.8829 |
| Random Forest | 0.7653 | 0.8245 | 0.7140 | 0.7586 | 0.8530 | 0.8874 |
| HistGradientBoosting | 0.7690 | 0.8154 | 0.7240 | 0.7670 | 0.8584 | 0.8922 |

Validation-selected threshold = 0.35 for HistGradientBoosting:
- Accuracy: 0.7503
- Precision: 0.7222
- Recall: 0.8523
- F1: 0.7819
- ROC-AUC: 0.8600
- PR-AUC: 0.8930

The lower threshold intentionally increases sensitivity to the at-risk class.

## Feature-group experiment

| Feature set | ROC-AUC |
|---|---:|
| Static demographic/registration | 0.6759 |
| Static + early engagement | 0.8587 |
| Full 28-day features | 0.8584 |

Early engagement is the dominant incremental signal in this experiment. Assessment variables do not materially improve the already strong engagement-based model.

## Timeliness experiment

The best traditional model at 56 days achieved:
- Accuracy: 0.7968
- Precision: 0.8475
- Recall: 0.7475
- F1: 0.7944
- ROC-AUC: 0.8827

The 56-day model is more predictive, but it delays the point at which intervention can occur.

## Important limitation

TensorFlow was not installed in the development runtime. The repository therefore contains the complete TensorFlow/Keras implementation and no fabricated DL numbers. Run the notebook in Google Colab, save the .keras model, and insert the observed DL results into the final report before submission.