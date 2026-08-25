# v2.0 result interpretation

## Decision question

v2 asks one controlled question: if the previous sequence module A is replaced by the manifest-safe D ensemble, should the established 20%/47%/33% D/B/C system be released, or should its module weights be re-optimized?

SCC/Spearman and PCC/Pearson are co-primary. SCC measures ordering; PCC measures continuous-value agreement.

## Main comparison

| Protocol | System | SCC ↑ | PCC ↑ | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---:|---:|---:|---:|---:|
| Cross-fitted meta-OOF | Previous A+B+C | 0.818989 | 0.811902 | 0.378254 | **0.288858** | 0.657701 |
| Cross-fitted meta-OOF | **v2 D+B+C** | **0.821347** | **0.813272** | **0.378139** | 0.290669 | **0.657909** |
| Historical fixed validation | Previous A+B+C | 0.842370 | 0.832479 | 0.353557 | **0.269166** | 0.691690 |
| Historical fixed validation | **v2 D+B+C** | **0.846156** | **0.835481** | **0.352253** | 0.270721 | **0.693960** |

The controlled replacement improves both primary point estimates. On cross-fitted meta-OOF, SCC rises by `0.002358`; its paired target-cluster bootstrap interval is `[0.001061, 0.003650]`. PCC rises by `0.001370`, but its interval is `[-0.000043, 0.002750]`. RMSE changes by `-0.000115` and its interval crosses zero; MAE is slightly worse. The supported conclusion is therefore specific: v2 has reproducible ranking evidence and a positive PCC point estimate, not a blanket claim that every error measure is significantly better.

Historical fixed validation shows SCC `+0.003786`, PCC `+0.003002`, RMSE `-0.001304`, and R² `+0.002270`; MAE is `+0.001556` worse. This split confirms the numerical direction but is not treated as untouched external evidence.

## What each module contributes

| Protocol | Module/system | SCC | PCC | RMSE |
|---|---|---:|---:|---:|
| Cross-fitted meta-OOF | D alone | 0.740669 | 0.717570 | 0.450967 |
| Cross-fitted meta-OOF | B alone | 0.814765 | 0.807582 | 0.381351 |
| Cross-fitted meta-OOF | C alone | 0.814168 | 0.807501 | 0.381551 |
| Cross-fitted meta-OOF | **D+B+C** | **0.821347** | **0.813272** | **0.378139** |
| Historical fixed validation | D alone | 0.770942 | 0.753780 | 0.420487 |
| Historical fixed validation | B alone | 0.840515 | 0.827767 | 0.357350 |
| Historical fixed validation | C alone | 0.841647 | 0.832070 | 0.353974 |
| Historical fixed validation | **D+B+C** | **0.846156** | **0.835481** | **0.352253** |

D is a sequence-only safety route. B and C gain most of their predictive strength from training-history and template-mapping context. Their performance does not prove that history-derived signals are causal biological properties, and it motivates the explicit unseen-guide/template diagnostics and sequence fallback in the tool.

## Weight audit

The released formula is:

```text
v2 = 0.20 × D + 0.47 × B + 0.33 × C
```

Three searches were retained:

1. 5,151 coarse pooled combinations at 0.01 steps;
2. 20,301 fine pooled combinations at 0.005 steps;
3. 101,505 fine combinations selected separately from the other four folds for each held-out fold.

The pooled fine optimum is 22.5%/47%/30.5% with SCC 0.821459, only `0.000112` above the locked system. A broad near-optimal region exists: 423 combinations lie within 0.0001 SCC. More importantly, cross-fitted reweighting reaches SCC 0.820691, below locked 20%/47%/33% at 0.821347. The fixed-validation labels were not used to select the released weights. Hence v2 retains the locked weights.

## Evidence boundary

- History references use only 8,417 training rows; fixed-validation labels are excluded.
- OOF folds isolate target groups, not guides. Most fixed-validation rows use guides seen in training.
- The 2,217-row fixed split had been examined during project development and is historical validation.
- Automatic mapping is exact/IUPAC template-window matching within the frozen EasyDesign reference. It is not a universal genomic search.
- The released artifact supports EasyDesign-compatible aligned pairs and needs independent validation for new assays, organisms, orthologs or activity scales.

## Frozen evidence

- `results/v2_dbc/meta_oof_predictions.csv`: 8,417 cross-fitted module and system predictions;
- `results/v2_dbc/fixed_validation_predictions.csv`: 2,217 historical fixed predictions;
- `results/v2_dbc/recomputed_metrics.csv`: independently recalculated metrics;
- `results/v2_dbc/v2_vs_v1_target_cluster_bootstrap.csv`: SCC, PCC and RMSE paired intervals;
- `results/v2_dbc/weight_audit/`: all coarse, fine and fold-specific weight grids, decisions and charts;
- `models/mapping/`: frozen B/C models, 1,191-feature manifest, templates and training-only history references.
