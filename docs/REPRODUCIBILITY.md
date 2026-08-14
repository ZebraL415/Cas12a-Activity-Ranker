# Reproducibility

## Frozen environment

The supported environment is Python 3.12 with versions pinned in `requirements.txt`. On macOS, XGBoost also needs `libomp`.

## Acceptance sequence

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/verify_repository.py
python scripts/train_final_four.py --verify-only
python scripts/train_final_four.py --smoke-test
python scripts/reproduce_metrics.py
```

The verification script checks:

- the formal V2-2 SHA-256 and record counts;
- 188-feature manifest and 183 active deployed inputs;
- target isolation across frozen OOF folds;
- exact reconstruction of stored features;
- native-model predictions against all 2,217 frozen validation outputs;
- independently recomputed performance metrics;
- every file listed in the public checksum manifest.

## Full training

```bash
python scripts/train_final_four.py --output reproduced_run
```

The full run trains XGBoost and CatBoost in five target-grouped OOF folds, searches 101 ensemble weights on OOF predictions, then trains both final models and evaluates four outputs on the historical fixed validation set.

## Evidence boundary

The frozen per-record predictions make the reported metrics auditable without retraining. Full model-training reproducibility still depends on the pinned software environment, operating system numerical libraries and CPU implementation. Small floating-point differences may occur across platforms; model rankings and metrics should be compared with explicit tolerances.

The validation split is historical fixed validation, not an untouched external cohort. The OOF-weighted ensemble is reported as exploratory because its ranking gain is small and its target-cluster interval crosses zero.
