# Experiment Summary: `reg_class_platform_electricsheepafrica-africa-synth-education-waec-results-nigeria`

## Objective

Provide a guided machine-learning workflow for selecting, optimizing, and evaluating a regression or classification model for a user-chosen dataset and target variable.

## Experiment Results

### Evaluation Metrics

The regression model demonstrated moderate overall performance. The model was able to capture meaningful patterns in the target variable, as reflected by the R² (0.658019), but a considerable proportion of the variability remains unexplained. The MAE (5.256633) indicates noticeable prediction errors, while the higher RMSE (9.116415) suggests that some larger errors remain. Overall, the results indicate that the available features contain meaningful predictive information, but a considerable proportion of the target variability remains unexplained by the model.

### Diagnostics

In general, the model is slightly underpredicting, but the systematic bias is negligible relative to the overall error. Residuals are closely balanced around zero (Relative residual bias:0.0004). Residuals show little evidence of first-order autocorrelation and appear reasonably independent in sequence (Durbin-Watson:2.0029).

### Statistical Analysis

The statistical distribution of the errors are strongly consistent with a non-normal distribution. The Shapiro-Wilk test rejects normality and the W statistic shows a relatively large departure from 1 (W=0.8404, p=1.7044718604619243e-125). This may be caused by strong curvature, heavy tails, skewness or extreme outliers.

## Reproducibility

| Artifact | Location |
|---|---|
| Configuration | `config/run_config.json` |
| Exact split & random state | `manifests/split_manifest.csv` |
| Final model configuration | `checkpoints/model_optimized_profile.joblib` |
