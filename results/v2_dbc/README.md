# v2 frozen evidence

This directory preserves the decisions behind the released `0.20 D + 0.47 B + 0.33 C` predictor.

| File | Content |
|---|---|
| `meta_oof_predictions.csv` | 8,417 cross-fitted module/system predictions |
| `fixed_validation_predictions.csv` | 2,217 historical fixed predictions |
| `integration_metrics.csv` / `integration_decisions.json` | compared integration schemes and promotion audit |
| `recomputed_metrics.csv` | independent SCC/PCC/error recalculation |
| `v2_vs_v1_target_cluster_bootstrap.csv` | paired SCC, PCC and RMSE changes with 2,000 resamples |
| `weight_audit/coarse_dbc_weight_grid_5151.csv` | every 0.01-step D/B/C combination |
| `weight_audit/fine_pooled_dbc_weight_grid_20301.csv` | every 0.005-step pooled combination |
| `weight_audit/fine_crossfitted_weight_grids_101505.csv` | every fold-specific fine-grid choice |
| `weight_audit/fine_weight_decision.json` | near-optimal ranges, selected fold weights, bootstrap and decision |
| `weight_audit/fine_weight_audit.svg/png` | visual summary of the fine audit |

The pooled optimum is not the release rule. Cross-fitted reweighting performs below the locked replacement, so fixed-validation results are not used to choose a new ratio.
