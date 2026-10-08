# Verified Results Summary

These are the final results recorded from the current Colab notebook run. The values below should be used consistently in the report, presentation, and project documentation.

## 28-day held-out test

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.752588 | 0.811655 | 0.688640 | 0.745104 | 0.830670 | 0.865894 | 0.50 |
| Random Forest | 0.754134 | 0.825117 | 0.674809 | 0.742432 | 0.837007 | 0.874417 | 0.50 |
| HistGradientBoosting | 0.760779 | 0.806901 | 0.715715 | 0.758578 | 0.842505 | 0.878211 | 0.50 |
| HGB, threshold tuned | 0.727863 | 0.697276 | 0.851383 | 0.766662 | 0.842505 | 0.878211 | 0.35 |

The threshold of 0.35 was selected using the validation set. It increases recall for the at-risk class from 0.715715 to 0.851383, while reducing precision and overall accuracy. This is relevant because the educational objective prioritises identifying students who may benefit from earlier support.

## Feature-group experiment

| Feature set | ROC-AUC |
|---|---:|
| Static demographic/registration | 0.675909 |
| Static + early engagement | 0.841454 |
| Full 28-day features | 0.842505 |

The ablation experiment shows that early engagement provides the major improvement over static demographic and registration information. Adding the remaining assessment-related features produces only a small additional improvement in this run.

## Deep-learning experiment

Three MLP configurations were evaluated on the same 28-day prediction task.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Threshold | Best validation ROC-AUC | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP_128_64_32 | 0.719518 | 0.684714 | 0.863449 | 0.763764 | 0.841226 | 0.877304 | 0.30 | 0.827556 | 9 |
| MLP_128_64 | 0.713800 | 0.680607 | 0.857269 | 0.758791 | 0.841660 | 0.877704 | 0.30 | 0.826079 | 10 |
| MLP_256_128_64 | 0.713491 | 0.674661 | 0.877575 | 0.762855 | 0.842383 | 0.878269 | 0.30 | 0.825344 | 16 |

The final deep-learning model selected by validation performance was **MLP_128_64_32**, with the highest validation ROC-AUC of 0.827556 and its best validation epoch at epoch 9. The test-set results should be interpreted separately from the validation score because model selection was based on validation performance.

The deep-learning configurations used TensorFlow/Keras with a tf.data input pipeline, dropout, batch normalization, early stopping, and learning-rate reduction.

## Timeliness experiment

The 56-day experiment used HistGradientBoosting with the full 56-day feature set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting, 56-day | 0.797404 | 0.847486 | 0.748970 | 0.795188 | 0.881874 | 0.910445 | 0.50 |

The 56-day model is more predictive than the 28-day models, but it uses substantially more information from the course. This creates an important practical trade-off: later prediction can improve performance while reducing the time available for intervention.

## Main findings

- On the 28-day test set, HistGradientBoosting achieved the strongest default-threshold traditional ML performance, with ROC-AUC **0.842505** and F1 **0.758578**.
- Lowering the HGB threshold to **0.35** increased at-risk recall to **0.851383**, at the cost of precision and accuracy.
- The deep-learning models achieved test ROC-AUC values between **0.841226 and 0.842383**, which were very close to the traditional HGB result.
- The selected MLP configuration was **MLP_128_64_32**, based on validation ROC-AUC.
- The 56-day HGB model achieved the highest overall ROC-AUC in the experiments, **0.881874**, demonstrating the predictive value of additional course information.
- Early engagement was the strongest incremental feature group in the ablation experiment, raising ROC-AUC from **0.675909** with static features alone to **0.841454** when early engagement was added.

## Reproducibility note

The results above come from the current notebook execution and should be treated as the authoritative numerical results for the final report and presentation. The raw OULAD dataset and trained model files are stored separately from the GitHub source repository, as required by the project submission structure.
