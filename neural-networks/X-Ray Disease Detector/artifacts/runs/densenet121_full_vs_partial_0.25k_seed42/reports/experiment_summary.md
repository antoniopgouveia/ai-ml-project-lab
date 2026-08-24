# Experiment summary: densenet121_full_vs_partial_0.25k_seed42

## Objective
Determine whether the partial DenseNet-121 fine-tuning model reaches comparable validation auc-roc sooner than full DenseNet-121 fine-tuning model, while reducing training, classification and Grad-CAM cost.

## Comparable-quality definition
Target auc-roc = full FP32 best validation auc-roc (0.5778) - tolerance (0.0050) = 0.5728.

## Result
The partially optimized model reached the comparable-quality target in 72.8 seconds.
Full FT FP32 total training time: 364.4 seconds. Partial FT optimized total training time: 254.6 seconds.

## Reproducibility
Configuration: `config/run_config.json`  
Exact split and seed: `manifests/split_manifest.csv`  
Class prevalence: `metrics/class_prevalence.csv`