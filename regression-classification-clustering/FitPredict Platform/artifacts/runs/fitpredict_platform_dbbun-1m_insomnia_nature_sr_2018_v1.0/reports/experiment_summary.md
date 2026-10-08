# Experiment Summary: `reg_class_platform_dbbun-1m_insomnia_nature_sr_2018_v1.0`

## Objective

Provide a guided machine-learning workflow for selecting, optimizing, and evaluating a regression or classification model for a user-chosen dataset and target variable.

## Experiment Results

### Evaluation Metrics

The classification model demonstrated good overall performance (F1-Score: 0.770591). The model shows stable predictive capabilities across classes, though minor classification errors persist. Precision stands at 0.776380 with a recall threshold of 0.766559.

### Diagnostics

The model has good discriminative ability and can generally distinguish positive from negative cases effectively, although some overlap between the two groups remains (ROC-AUC score: 0.8112).
The model demonstrates strong positive-case detection, with a PR-AUC of 0.7748. This corresponds to 2.0× the positive-class prevalence baseline.

### Statistical Analysis

The model shows mild underprediction. It predicts the class somewhat less frequently than observed (Prediction Bias score: -0.0308).
The error pattern is relatively balanced. FP and FN occur at broadly similar frequencies (FP/FN ratio: 0.7469).
The model demonstrates a moderate correspondence between predicted and observed class status (Matthews Correlation Coefficient: 0.5428).

## Reproducibility

| Artifact | Location |
|---|---|
| Configuration | `config/run_config.json` |
| Exact split & random state | `manifests/split_manifest.csv` |
| Final model configuration | `checkpoints/model_optimized_profile.joblib` |
