# Model card

## Intended use

Rank candidate 25-position Cas12a guide–target pairs by a fluorescence-derived continuous diagnostic reaction activity score, so higher-priority candidates can be tested first. The supported deployment artifact is `primary/xgboost_final.json`.

## Inputs and preprocessing

The external interface accepts `crRNA_sequence` and an aligned target. The feature builder creates 188 pair-alignment, positional, substitution, composition and sequence-context variables. Five training constants are absent from the deployed matrix; the ordered 183 active inputs and missing-value medians are frozen in `model_input_metadata.json` and `training_medians.csv`.

For gap-containing targets, callers must supply the 25-position alignment. U is normalized to T. Ambiguous bases are rejected rather than silently imputed.

## Training and evaluation

- Training: 8,417 records.
- Evaluation: 2,217 historical fixed-validation records, 1,796 distinct target sequences.
- Final OOF procedure: five frozen target-grouped folds used for ensemble-weight selection.
- Primary XGBoost: 1,100 trees, learning rate 0.02, max depth 7, histogram method, seed 42.
- Supporting CatBoost: 1,000 iterations, learning rate 0.02, depth 9, seed 42.

The fixed validation set did not participate in the final 59/41 weight search, but it had been inspected in earlier development and is not an untouched test set.

## Model choice

XGBoost is primary because it has the best RMSE, MAE and R², while the weighted ensemble's Spearman gain is only `0.001111` and its paired target-cluster bootstrap interval crosses zero. CatBoost, equal averaging and OOF weighting are retained for sensitivity analysis.

## Limitations

- The score is assay-specific and not editing efficiency, patient-level diagnostic accuracy or a mechanistic causal effect.
- 95.4% of fixed-validation records use guides observed during training; evidence for wholly unseen guides is limited.
- Predictions compress the extreme activity range.
- Model complementarity is weak because XGBoost and CatBoost predictions/residuals are highly correlated.
- Generalization to a new assay scale, ortholog or sequence-preparation protocol requires independent calibration and validation.

## Artifact formats

- `xgboost_final.json`: supported version-stable Booster artifact.
- `catboost_final.cbm`: supported native supporting-model artifact.

Native predictions were checked against all 2,217 saved validation predictions with maximum absolute difference below `1e-6`.
