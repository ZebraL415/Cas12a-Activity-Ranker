# Model card: v1.5 D ensemble

## Intended use

Predict a continuous fluorescence-derived Cas12a molecular-detection activity score for an already aligned 25-position crRNA–target pair, and rank multiple rows within the same input file. SCC/Spearman and PCC/Pearson are co-primary evaluation outcomes.

This model is not intended for genome-editing efficiency, patient-level diagnosis, clinical decision-making or unaligned sequence search.

## Model

```text
D = 0.35 × XGBoost + 0.59 × LightGBM + 0.06 × MLP
```

All three components use the same 183 active sequence-derived features. The MLP artifact includes its fitted `StandardScaler`. Artifact paths, SHA-256 values and the frozen software environment are recorded in `d_model_metadata.json`.

## Inputs

The file interface requires `record_id`, `crRNA_sequence` and `target_aligned_25`. The two sequence columns represent exactly 25 aligned positions. The feature builder derives 188 alignment, position, substitution, composition and context variables; five training constants are filtered before model inference.

U is normalized to T. Ambiguous bases and incorrect lengths are rejected with a row-level error report. v1.5 does not infer alignment or accept precomputed mapping columns.

## Training and selection

- Training partition: 8,417 records.
- Weight-selection protocol: five frozen target-grouped OOF folds.
- Active deployed inputs: 183, in a frozen order.
- Weight grid: nonnegative 0.01 increments summing to 1.
- Selected weights: XGBoost 0.35, LightGBM 0.59, MLP 0.06.
- Evaluation: 2,217 historical fixed-validation records across 1,796 target clusters.

Fixed-validation labels were not used in the v1.5 weight search. The split had nevertheless been observed during earlier project development and is not an untouched external test.

## Performance

| Protocol | SCC | PCC | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| Training target-group OOF | 0.7410 | 0.7179 | 0.4507 | 0.3471 | 0.5140 |
| Historical fixed validation | 0.7709 | 0.7538 | 0.4205 | 0.3260 | 0.5639 |

The D ensemble is numerically best on SCC, PCC, RMSE and R² among its three components in the fixed-validation comparison. LightGBM has a marginally lower MAE.

## Limitations

- Activity is assay- and normalization-specific.
- Most fixed-validation rows use guides seen in training, so new-guide evidence is limited.
- Within-file ranks are relative and are not probabilities or calibrated experimental thresholds.
- New orthologs, assay scales or preprocessing protocols need independent validation.
- The historical fixed validation is not an external prospective cohort.

## Artifacts

- `primary/d_xgboost.json`: native XGBoost Booster;
- `primary/d_lightgbm.txt`: native LightGBM Booster;
- `primary/d_mlp_pipeline.joblib`: fitted StandardScaler + MLP pipeline;
- `d_model_metadata.json`: weights, metrics, hashes, environment and scope;
- `model_input_metadata.json`: ordered active features;
- `training_medians.csv`: frozen missing-value fallback.

All three native component predictions and the combined D output are checked against all 2,217 frozen validation rows with maximum absolute difference below `1e-6`.
