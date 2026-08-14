# Repository map

```text
Cas12a-Activity-Ranker/
├── README.md / README_zh.md       Public project entry points
├── LICENSE                        Apache-2.0 software license
├── NOTICE                         Required attribution summary
├── THIRD_PARTY_NOTICES*.md        Data/source license boundaries
├── CONTRIBUTING.md                Contribution and protocol rules
├── CODE_OF_CONDUCT.md             Community expectations
├── SECURITY.md                    Responsible disclosure
├── pyproject.toml                 Installable Python package metadata
├── requirements.txt              Frozen runtime dependencies
├── data/
│   ├── raw/                       Unmodified upstream repository data files
│   ├── processed/v2_2/            Final model-ready table and metadata
│   ├── examples/                  Traceable inputs and expected predictions
│   └── metadata/                  Public file checksums and data roles
├── src/cas12a_ml/
│   ├── features.py                Sequence pair -> 188 frozen features
│   └── predict.py                 Native XGBoost/CatBoost inference
├── scripts/
│   ├── predict_external.py        External-use CLI
│   ├── train_final_four.py        Full final comparison/retraining
│   ├── reproduce_metrics.py       Metrics + target-cluster bootstrap
│   ├── verify_repository.py       End-to-end artifact validation
│   └── generate_sha256_manifest.py
├── models/
│   ├── primary/                   Supported XGBoost JSON model
│   ├── supporting/                Supporting CatBoost CBM model
│   ├── training_medians.csv       Frozen missing-value preprocessing
│   └── model_input_metadata.json  Ordered 183 active features
├── results/                       Frozen and independently recomputed evidence
├── tests/                         Integrity, feature and inference tests
├── docs/                          Usage, data, results and reproducibility
├── reports/                       Presentation resources
└── .github/                       CI, issue forms and PR template
```

## Active execution chain

`data/processed/v2_2/feature_manifest.csv` defines 188 candidate inputs. Five training-constant columns are removed; `models/model_input_metadata.json` freezes the remaining order of 183 features. `src/cas12a_ml/features.py` reconstructs the feature matrix from a sequence pair, and `src/cas12a_ml/predict.py` applies the stored medians and native models.

`results/fixed_validation_predictions.csv` is the regression oracle used to test model portability. Public integrity is tracked by `data/metadata/sha256_manifest.tsv`.

## Local-only preservation layer

The working copy may also contain `_local_only/`, which stores original submissions, internal experiment logs, legacy joblib payloads and previous data-package layouts. The entire directory is excluded through `.gitignore` and does not belong to the public GitHub repository or active runtime.
