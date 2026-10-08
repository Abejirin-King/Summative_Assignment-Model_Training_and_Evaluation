# From Prediction to Intervention

## Early Identification of At-Risk Learners Using Traditional ML and Deep Learning

This repository implements a leakage-aware learning-analytics project using the Open University Learning Analytics Dataset (OULAD).

### Problem definition
The target is binary: At risk = 1 for Fail or Withdrawn; Not at risk = 0 for Pass or Distinction.
The purpose is early identification for supportive intervention, not punitive student labeling.

### Research questions
1. How accurately can academic risk be predicted from information available during the initial weeks of a course?
2. How does traditional machine learning compare with a TensorFlow/Keras neural network?
3. How does the amount of observed early-course information affect predictive performance and intervention timeliness?

### Experimental design
- Primary prediction cutoff: 28 days
- Secondary cutoff: 56 days
- Unit: student-course-presentation
- Student-level grouped train/validation/test splitting
- id_student excluded from predictors
- VLE features restricted to the cutoff
- Assessment features restricted to assessments due and submitted by the cutoff
- Traditional models: Logistic Regression, Random Forest, HistGradientBoosting
- Deep learning: TensorFlow/Keras MLP using tf.data
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices and learning curves

### Verified traditional-model findings
The best 28-day traditional model is HistGradientBoosting. On the held-out test set its ROC-AUC is approximately 0.860. A validation-selected threshold of 0.35 gives approximately 0.852 recall for the at-risk class and 0.722 precision.
The 56-day experiment reaches approximately 0.883 ROC-AUC, demonstrating the trade-off between earlier intervention and additional evidence.
Feature ablation shows that static demographic/registration variables alone are much weaker (ROC-AUC approximately 0.676), while adding early engagement raises ROC-AUC to approximately 0.859. Adding early assessment features does not materially improve that result.

### Deep learning
The development runtime used to prepare this repository did not have TensorFlow available, so no neural-network result is fabricated. The notebook contains the complete TensorFlow/Keras experiment, including tf.data, hyperparameter comparison, early stopping, learning curves and .keras model saving. Run it in Google Colab or another TensorFlow environment and add the resulting DL metrics to the report.

### Repository structure
- notebooks/: end-to-end Colab/Jupyter workflow
- src/: feature engineering, preprocessing, models and evaluation
- models/: saved local model artifacts
- reports/: tables and figures
- docs/: methodology/project notes

### Data policy
The OULAD raw CSV files are not committed to GitHub. Keep anonymisedData.zip in the execution environment or Google Drive.

### Reproducibility
Install dependencies with pip install -r requirements.txt.

Dataset: Open University Learning Analytics Dataset (OULAD).