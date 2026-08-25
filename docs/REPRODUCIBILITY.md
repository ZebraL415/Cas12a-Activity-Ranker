# Reproducibility

## Supported environment

v2.0 targets Python 3.12. Exact package versions are pinned in `requirements.txt`; large model files use Git LFS.

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
python scripts/reproduce_v2_metrics.py
```

Acceptance covers:

- CSV, TSV and XLSX round trips with row, order and source-column preservation;
- row-level validation, forbidden manual mapping columns and explicit fallback behavior;
- authoritative data and all deployed model/reference SHA-256 values;
- exact automatic mapping reproduction across 10,634 train/validation rows;
- 188 sequence-feature reconstruction, 183 D inputs and ordered 1,191 B/C inputs;
- training-only guide/mapping history references;
- target isolation across frozen OOF folds;
- all 2,217 D, B, C and v2 predictions within `1e-6` of frozen values;
- independent SCC, PCC, RMSE, MAE and R² reproduction within `1e-8`;
- 20,301 pooled and 101,505 cross-fitted fine-weight records and the locked 20%/47%/33% decision.

## Reference construction

`scripts/build_v2_references.py` creates the deployment references from the authoritative V2-2 table and the EasyDesign combined source workbook. It enforces 8,417 `baseline_train` rows and writes:

- guide sums/counts by all, exact/nonexact, difference class and gap class;
- mapping-key sums/counts for six categorical fields;
- 198 cleaned Table S2 templates;
- provenance, dimensions, smoothing constants and source hashes.

No fixed-validation label enters these references. `models/mapping/reference_metadata.json` records the exact canonical-table and source-workbook SHA-256 values.

## What is and is not reproduced

Released-artifact inference, automatic mapping, feature construction, reference lookup, per-record predictions, metrics, uncertainty and weight enumeration are auditable from the repository.

The repository does not claim bit-identical retraining across arbitrary platforms. Tree and neural-network training may vary with libraries and numerical runtimes. The supported claim is exact inference from frozen artifacts plus auditable saved OOF/fixed evidence and environment pins.

## Evaluation boundary

The v2 decision uses five target-grouped cross-fitted meta-OOF predictions. The 2,217-row fixed validation was not used to select the released D/B/C weights, but it had been observed earlier during project development. It is historical fixed validation, not untouched external validation. OOF grouping isolates targets rather than guides, so unseen-guide generalization remains a limitation.
