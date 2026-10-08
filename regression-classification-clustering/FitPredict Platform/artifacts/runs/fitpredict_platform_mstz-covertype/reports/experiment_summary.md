# Experiment Summary: `reg_class_platform_mstz-covertype`

## Objective

Provide a guided machine-learning workflow for selecting, optimizing, and evaluating a regression or classification model for a user-chosen dataset and target variable.

## Experiment Results

### Evaluation Metrics

The classification model demonstrated excellent overall performance (F1-Score: 0.916645). The high Balanced Accuracy (0.894950) confirms that the model handles both the majority and minority classes with outstanding precision (0.942116) and high operational recall (0.894950).

### Diagnostics

The model has excellent ability to distinguish positive cases from negative cases. Positive cases are generally assigned higher prediction scores than negative cases (ROC-AUC score: 0.9949).
The model demonstrates exccellent positive-case detection, with a PR-AUC of 0.9901. This corresponds to 0.9× the positive-class prevalence baseline.

### Statistical Analysis

Class 0: The model shows approximately unbiased prediction frequency. Predicted frequency for class 0 is close to observed prevalence (Prediction Bias score: -0.0044). The error pattern is relatively balanced for class 0. FP and FN occur at broadly similar frequencies (FP/FN ratio: 0.7756).
Class 1: The model shows approximately unbiased prediction frequency. Predicted frequency for class 1 is close to observed prevalence (Prediction Bias score: +0.0107). The error pattern is moderately FP-dominant. Class 1 has more false positives than false negatives (FP/FN ratio: 1.7118).
Class 2: The model shows approximately unbiased prediction frequency. Predicted frequency for class 2 is close to observed prevalence (Prediction Bias score: +0.0007). The error pattern is relatively balanced for class 2. FP and FN occur at broadly similar frequencies (FP/FN ratio: 1.2410).
Class 3: The model shows approximately unbiased prediction frequency. Predicted frequency for class 3 is close to observed prevalence (Prediction Bias score: -0.0003). The error pattern is moderately FN-dominant. Class 3 has more false negatives than false positives (FP/FN ratio: 0.6022).
Class 4: The model shows approximately unbiased prediction frequency. Predicted frequency for class 4 is close to observed prevalence (Prediction Bias score: -0.0033). The error pattern is very strongly FN-dominant. Class 4 is missed substantially more often than it is falsely predicted (FP/FN ratio: 0.1626).
Class 5: The model shows approximately unbiased prediction frequency. Predicted frequency for class 5 is close to observed prevalence (Prediction Bias score: -0.0017). The error pattern is moderately FN-dominant. Class 5 has more false negatives than false positives (FP/FN ratio: 0.5524).
Class 6: The model shows approximately unbiased prediction frequency. Predicted frequency for class 6 is close to observed prevalence (Prediction Bias score: -0.0017). The error pattern is strongly FN-dominant. False negatives for class 6 clearly dominate (FP/FN ratio: 0.2671).
The model demonstrates a very strong correspondence between predicted and observed class status (Matthews Correlation Coefficient: 0.9217).

## Reproducibility

| Artifact | Location |
|---|---|
| Configuration | `config/run_config.json` |
| Exact split & random state | `manifests/split_manifest.csv` |
| Final model configuration | `checkpoints/model_default_profile.joblib` |
