# Experiment Summary: `reg_class_platform_einae-zinc_standard_agent`

## Objective

Provide a guided machine-learning workflow for selecting, optimizing, and evaluating a regression or classification model for a user-chosen dataset and target variable.

## Experiment Results

### Evaluation Metrics

The regression model demonstrated excellent overall performance. The high R² (0.946830) indicates that the model was able to explain a very large proportion of the variability in the target variable. The low MAE (0.221094) indicates that predictions were, on average, very close to the observed values, while the low RMSE (0.372084) suggests that large prediction errors were also limited. Overall, the results indicate that the available features contain strong predictive information, allowing the model to capture most of the underlying patterns in the target variable with a high degree of accuracy.

### Diagnostics

In general, the model is slightly underpredicting, but the systematic bias is negligible relative to the overall error. Residuals are closely balanced around zero (Relative residual bias:0.0031). Residuals show little evidence of first-order autocorrelation and appear reasonably independent in sequence (Durbin-Watson:1.9926).

### Statistical Analysis

The statistical distribution of the errors are strongly consistent with a non-normal distribution. The Shapiro-Wilk test rejects normality and the W statistic shows a relatively large departure from 1 (W=0.7569, p=3.8748989320846123e-169). This may be caused by strong curvature, heavy tails, skewness or extreme outliers.

## Reproducibility

| Artifact | Location |
|---|---|
| Configuration | `config/run_config.json` |
| Exact split & random state | `manifests/split_manifest.csv` |
| Final model configuration | `checkpoints/model_optimized_profile.joblib` |
