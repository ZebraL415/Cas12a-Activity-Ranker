# Reproducibility

## Supported environment

v1.5 targets Python 3.12. Exact package versions are pinned in `requirements.txt`. Model binaries are stored with Git LFS.

```bash
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .
```

## Release acceptance sequence

```bash
cas12a-ranker self-test
python -m unittest discover -s tests -v
python scripts/verify_repository.py
python scripts/reproduce_metrics.py
```

The acceptance checks cover:

- CSV, TSV and XLSX table round trips;
- preservation of source rows, order and columns;
- row-level invalid-input reporting;
- the authoritative data SHA-256 and split counts;
- 188-feature reconstruction and 183-feature deployed order;
- target isolation across the five frozen OOF folds;
- hashes for all three D model artifacts;
- all 2,217 XGBoost, LightGBM, MLP and ensemble predictions within `1e-6`;
- independent SCC, PCC, RMSE, MAE and R² reproduction within `1e-8`;
- the OOF weight-grid optimum at 0.35/0.59/0.06.

## What is and is not reproducible

Frozen predictions, model inference, metrics, feature reconstruction and weight selection are auditable from this release. The three deployment artifacts were exported from the frozen v1.5 training run, and their exact package environment is recorded in `models/d_model_metadata.json`.

The repository does not claim that the D training run can be reconstructed from an arbitrary environment with bit-identical trees and neural-network weights. Cross-platform numerical libraries and model-library implementations may differ. The supported reproducibility claim is exact released-artifact inference plus auditable saved training/validation evidence.

## Evaluation boundary

Weights were selected only from the 8,417-record training partition using five target-grouped OOF folds. The 2,217-record fixed validation was not used for this weight search, but it had been observed during earlier project development. It is therefore historical fixed validation, not an untouched external test.
