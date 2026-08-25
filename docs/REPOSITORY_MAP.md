# Repository map

```text
Cas12a-Activity-Ranker/
├── README.md / README_zh.md       User-first entry points
├── data/
│   ├── raw/                       Unmodified upstream repository files
│   ├── processed/v2_2/            Authoritative table, manifest and folds
│   ├── examples/                  Minimal input and frozen v2 output
│   └── metadata/                  Data roles and public checksums
├── src/cas12a_ml/
│   ├── features.py                Aligned pair -> 188 sequence features
│   ├── mapping.py                 All-candidate automatic template mapping
│   ├── v2_features.py             Training-history + 1,191 B/C matrix
│   ├── predict.py                 v2, D fallback and legacy inference
│   ├── io.py                      Validated table-in/table-out interface
│   └── cli.py                     Command and self-test
├── models/
│   ├── primary/                   D XGBoost, LightGBM and MLP
│   ├── mapping/                   B/C models, templates, histories, manifest
│   ├── supporting/                Retained v1.0 CatBoost
│   ├── v2_model_metadata.json     v2 weights, metrics and hashes
│   └── d_model_metadata.json      Sequence-only D metadata
├── results/
│   ├── v2_dbc/                    OOF/fixed predictions and full weight audit
│   └── v1_5_d_ensemble/           Retained D evidence
├── scripts/
│   ├── build_v2_references.py     Freeze training-only history/template data
│   ├── reproduce_v2_metrics.py    Recompute v2 metrics and uncertainty
│   ├── verify_repository.py       Full mapping/model regression
│   └── predict_external.py        Backward-compatible command wrapper
├── tests/                         Data, mapping, interface and inference tests
└── docs/                          Usage, data, results and reproducibility
```

## Active execution chain

The user supplies an aligned table. `io.py` validates every row; `features.py` builds D's sequence matrix; `mapping.py` finds every frozen template candidate; `v2_features.py` combines sequence, mapping and training-only history inputs; `predict.py` calculates D, B, C and the 20%/47%/33% score. If mapping fails, the same file records the D fallback and warnings rather than inventing unavailable inputs.

`results/v2_dbc/fixed_validation_predictions.csv` is the 2,217-row regression oracle. `results/v2_dbc/weight_audit/` preserves all coarse, fine and cross-fitted weight combinations. `data/metadata/sha256_manifest.tsv` tracks public file integrity.

## Local-only preservation

Working copies may contain `_local_only/` for original submissions, internal logs and legacy artifacts. It is excluded from Git and is not part of the public runtime.
