# Experiment Summary: `reg_class_platform_yelp-yelp_review_full`

## Objective

Provide a guided machine-learning workflow for selecting, optimizing, and evaluating a regression or classification model for a user-chosen dataset and target variable.

## Experiment Results

### Evaluation Metrics

The classification model demonstrated inadequate predictive performance (F1-Score: 0.591608). The current feature architecture fails to reliably distinguish between target categories, rendering the predictions structurally unsafe for automated deployment or production environments.

### Diagnostics

The model has good discriminative ability and can generally distinguish positive from negative cases effectively, although some overlap between the two groups remains (ROC-AUC score: 0.8793).
The model demonstrates moderate positive-case detection, with a PR-AUC of 0.6326. This corresponds to 0.3× the positive-class prevalence baseline.

### Statistical Analysis

Class 0: The model shows approximately unbiased prediction frequency. Predicted frequency for class 0 is close to observed prevalence (Prediction Bias score: +0.0124). The error pattern is relatively balanced for class 0. FP and FN occur at broadly similar frequencies (FP/FN ratio: 1.2562).
Class 1: The model shows approximately unbiased prediction frequency. Predicted frequency for class 1 is close to observed prevalence (Prediction Bias score: -0.0080). The error pattern is relatively balanced for class 1. FP and FN occur at broadly similar frequencies (FP/FN ratio: 0.9181).
Class 2: The model shows approximately unbiased prediction frequency. Predicted frequency for class 2 is close to observed prevalence (Prediction Bias score: -0.0087). The error pattern is relatively balanced for class 2. FP and FN occur at broadly similar frequencies (FP/FN ratio: 0.9147).
Class 3: The model shows approximately unbiased prediction frequency. Predicted frequency for class 3 is close to observed prevalence (Prediction Bias score: -0.0034). The error pattern is relatively balanced for class 3. FP and FN occur at broadly similar frequencies (FP/FN ratio: 0.9660).
Class 4: The model shows approximately unbiased prediction frequency. Predicted frequency for class 4 is close to observed prevalence (Prediction Bias score: +0.0077). The error pattern is relatively balanced for class 4. FP and FN occur at broadly similar frequencies (FP/FN ratio: 1.1285).
The model demonstrates a limited correspondence between predicted and observed class status (Matthews Correlation Coefficient: 0.4922).

## Reproducibility

| Artifact | Location |
|---|---|
| Configuration | `config/run_config.json` |
| Exact split & random state | `manifests/split_manifest.csv` |
| Final model configuration | `checkpoints/model_optimized_profile.joblib` |
