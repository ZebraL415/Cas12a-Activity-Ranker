# Data and result definitions

Paths below are relative to the extracted asset directory.

| Field | Definition |
|---|---|
| `record_id` | Stable EasyDesign source-table record identifier; joins are by ID |
| `cohort_id` | `core_dev`: 8,417 development records; `common_dev`: 5,995 unambiguous complete contexts |
| `protocol_id` | `target`, `guide`, `atomic_template`, `dual_atomic` (Joint) |
| `outer_fold` / `fold` | Zero-based formal fold, 0–4 |
| `group_id` / `cluster_id` | Protocol-specific isolation group; all member records stay together |
| `y_true` / `true_y` | Published processed 30-minute activity label, unchanged |
| `y_pred` / `pred_y` | Model prediction on the same label scale |
| `model_id` | Original run model identity; display names follow `result_index.csv` |
| `run_id` | Original accepted run identifier, retained for traceability |
| `evaluation_kind` | Fixed recipe, nested grouped selection, or historical continuity |
| `selection_id` | Accepted nested selector configuration, distinct from model version |
| `mapping_found` | Whether label-free reference mapping was found; controls the fusion fallback |
| `guide_support_n` | Training-fold count supporting the exact guide history key |
| `template_support_n` | Training-fold count supporting the combined template key |
| `mapping_candidate_template_count` | Number of candidate templates; not the preceding historical count |
| `seed` / `permutation_seed` | Original perturbation random seed, 137/271/409 |
| `SCC`, `PCC`, `RMSE`, `MAE`, `R2` | Spearman, Pearson, root mean square error, mean absolute error, coefficient of determination |

Missing/undefined correlations are blank with a status/reason where available.
Zero remains a genuine measured value. RMSE/MAE use the inherited label scale,
not raw fluorescence units. The relation between that negative numeric scale
and an absolute physical fluorescence scale is not reconstructed from the
available public measurements.

## Identity definitions

Target folds isolate the normalized target identity. Guide folds isolate the
crRNA sequence. Atomic-template folds isolate connected components created by
shared candidate-template membership. Joint folds join target, guide and
atomic-template memberships, so none of those identities bridges the outer
fit and query sets. The core cohort has 6,825/1,341/21/20 groups respectively;
the common cohort has 5,923/1,304/20/19.

History encodings are rebuilt using only the current fit records. Query labels
are removed before feature transformation. In nested validation the current
outer query and current inner query are both excluded from the inner fit.

## Statistical streams

Fixed: 2,000 group bootstrap samples and 2,000 paired whole-group prediction
swaps, seed 20261012 for each cohort/protocol. Core/common Holm families: 16/4.
Nested: seed 20261013 + cohort index × 10 + protocol index, in the order
core/common and target/guide/atomic-template/joint. Swap seed adds 1,000;
primary families: SELECTED_POLICY against 3 core comparators or EasyDesign,
12/4 tests. L02: 2,000 target-cluster samples with the original int64 random
stream, seed 20260906, paired with each perturbation's own intact predictor.

All differences are model A minus model B, or perturbed minus intact as
labelled. Positive SCC/PCC/R² and negative RMSE/MAE differences favour A.
The 95% intervals use percentile bounds. Groups, folds and random seeds do
not establish independent biological replication.
