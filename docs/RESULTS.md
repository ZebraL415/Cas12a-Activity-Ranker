# v1.5 result interpretation

## Primary outcomes

SCC/Spearman and PCC/Pearson are co-primary outcomes: SCC measures candidate ordering, while PCC measures whether predicted activity values track the observed continuous scale.

| Protocol | Model | SCC ↑ | PCC ↑ | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---:|---:|---:|---:|---:|
| 5-fold target-group OOF | XGBoost | 0.7361 | 0.7119 | 0.4545 | 0.3503 | 0.5058 |
| 5-fold target-group OOF | LightGBM | 0.7377 | 0.7155 | 0.4520 | 0.3478 | 0.5112 |
| 5-fold target-group OOF | MLP | 0.6091 | 0.5777 | 0.5444 | 0.4238 | 0.2909 |
| 5-fold target-group OOF | **D ensemble** | **0.7410** | **0.7179** | **0.4507** | **0.3471** | **0.5140** |
| Historical fixed validation | XGBoost | 0.7652 | 0.7476 | 0.4247 | 0.3312 | 0.5550 |
| Historical fixed validation | LightGBM | 0.7690 | 0.7527 | 0.4207 | **0.3257** | 0.5635 |
| Historical fixed validation | MLP | 0.6347 | 0.6147 | 0.5175 | 0.4099 | 0.3395 |
| Historical fixed validation | **D ensemble** | **0.7709** | **0.7538** | **0.4205** | 0.3260 | **0.5639** |

The D ensemble is the v1.5 deployment model. It is numerically strongest on both primary outcomes and on RMSE/R² in the historical fixed validation, while LightGBM has a marginally lower MAE. This supports one combined continuous-score-and-ranking output rather than separate production models.

## Weight selection

The frozen prediction is:

```text
D = 0.35 × XGBoost + 0.59 × LightGBM + 0.06 × MLP
```

All nonnegative weights in 0.01 steps that sum to 1 were evaluated using training-set target-group OOF predictions. The selected row maximized OOF SCC; fixed-validation labels were not used. The complete table is `results/v1_5_d_ensemble/oof_weight_grid.csv`.

## Evidence boundary

The fixed-validation set contains 2,217 records and 1,796 target clusters. It is suitable for a locked comparison under one protocol, but it was observed during earlier project development. Results therefore describe historical fixed validation, not untouched external validation.

`scripts/reproduce_metrics.py` independently rebuilds the saved metrics and calculates a 2,000-resample paired target-cluster comparison against the v1.0 XGBoost model. A small numerical improvement should not be described as a universal biological gain without an independent new-guide/external assay evaluation.

For D minus v1.0 XGBoost, the observed changes are SCC `+0.002708`, PCC `+0.004048` and RMSE `-0.002659` (negative is better). The respective 95% target-cluster bootstrap intervals are `[-0.001819, 0.007377]`, `[-0.001050, 0.009246]` and `[-0.006135, 0.000748]`; all include zero.

## Frozen evidence

- `metrics.csv`: OOF and fixed-validation metrics for all three components and D;
- `oof_predictions.csv`: 8,417 training-set OOF predictions;
- `fixed_validation_predictions.csv`: 2,217 frozen fixed-validation predictions;
- `oof_weight_grid.csv`: every tested weight combination;
- `run_manifest.json`: data sizes, features, weights, versions and run metadata;
- `recomputed_fixed_validation_metrics.csv`: independently regenerated fixed metrics;
- `d_vs_v1_xgb_target_bootstrap.csv`: paired target-cluster uncertainty comparison.
