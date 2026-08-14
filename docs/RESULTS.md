# Final result interpretation

| Experiment | Spearman | Pearson | RMSE | MAE | R² | Role |
|---|---:|---:|---:|---:|---:|---|
| XGBoost | 0.768234 | 0.749732 | **0.423146** | **0.327855** | **0.558381** | Primary |
| CatBoost | 0.758194 | 0.739726 | 0.432975 | 0.339590 | 0.537626 | Supporting |
| Equal 50/50 | 0.768762 | 0.750118 | 0.424913 | 0.330791 | 0.554685 | Sensitivity |
| OOF 59/41 | **0.769344** | **0.750783** | 0.424124 | 0.329809 | 0.556336 | Exploratory |

The weighted ensemble has the numerically highest ranking correlation; XGBoost has the best absolute-error and explained-variance metrics. The ensemble gain is too small to support a winner switch: paired target-cluster bootstrap (2,000 resamples, 1,796 targets) gives ΔSpearman 95% CI `[-0.002010, 0.004255]` for weighted minus XGBoost and `[-0.003236, 0.004421]` for equal minus XGBoost.

The final XGBoost and CatBoost predictions are highly correlated (Pearson approximately `0.972`), as are their residuals (approximately `0.971`). They therefore tend to make the same mistakes. Averaging can smooth small differences, but it cannot create substantial complementary information.

`results/recomputed_metrics.csv` and `results/ensemble_vs_xgb_target_bootstrap.csv` are regenerated from the saved per-record predictions by `scripts/reproduce_metrics.py`.
