# Usage guide

## 1. Prepare an input CSV

```csv
candidate_id,crRNA_sequence,target_aligned_25
candidate_01,TTTGTTGGGGCGTCCTTAGACGCCA,TTTGTTGGGGCGTCCTTAGACGCCA
candidate_02,TTTGGGCTAGGTGTGATAGGAATGG,TTTGGGCTAGGAGTGATAGGAATGG
```

Both biological columns represent 25 aligned positions. `crRNA_sequence` accepts A/C/G/T or U. `target_aligned_25` additionally accepts `-` for an existing target gap. Input identifiers and other metadata columns are copied to the output.

## 2. Run inference

```bash
python scripts/predict_external.py --input candidates.csv --output predictions.csv
```

Output columns:

| Column | Interpretation |
|---|---|
| `prediction_xgboost_primary` | Supported primary activity score |
| `prediction_catboost_supporting` | Supporting-model score |
| `prediction_equal_50_50_sensitivity` | Equal-average sensitivity result |
| `prediction_oof_weighted_exploratory` | 59/41 exploratory ensemble result |
| `rank_xgboost_descending` | Within-file candidate rank; 1 is highest |

Scores are meaningful for ranking within the studied assay representation. They are not calibrated probabilities and should not be interpreted as patient-level diagnostic accuracy.

## 3. Handle gaps correctly

For a no-gap target, `target_sequence` may replace `target_aligned_25` if it is exactly 25 nt. For gap-containing examples, supply `target_aligned_25` explicitly. This repository performs feature extraction after alignment; it is not an alignment program.

## 4. Validate an installation

```bash
python scripts/predict_external.py \
  --input data/examples/external_sequence_pairs.csv \
  --output /tmp/cas12a_predictions.csv

python scripts/verify_repository.py
```

The example predictions are frozen in `data/examples/expected_predictions.csv`.

## 5. Use the Python API

```python
import pandas as pd
from cas12a_ml import Cas12aPredictor, build_feature_frame

pairs = pd.read_csv("candidates.csv")
features = build_feature_frame(pairs)
predictions = Cas12aPredictor(".").predict(pairs)
```

When the package has not been installed, run from the repository root with `PYTHONPATH=src` or use the CLI script, which configures the source path automatically.
