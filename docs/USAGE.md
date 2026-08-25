# Usage guide

## Input contract

Use one row per aligned crRNA–target pair in CSV, TSV or XLSX format.

```csv
record_id,crRNA_sequence,target_aligned_25,sample_group
candidate_01,TTTGTTGGGGCGTCCTTAGACGCCA,TTTGTTGGGGCGTCCTTAGACGCCA,screen_1
candidate_02,TTTGTTGGGGCGTCCTTAGACGCCA,TTTGTTGGGGCGTCCTTAGTCGCCA,screen_1
```

Required columns:

- `record_id`: nonempty and unique within the file;
- `crRNA_sequence`: exactly 25 aligned positions using A/C/G/T; U is accepted as T;
- `target_aligned_25`: exactly 25 aligned positions using A/C/G/T and optional `-` gaps.

Headers are case-sensitive. All other columns and the original row order are preserved. v1.5 does not accept precomputed mapping columns and does not infer a biological alignment.

## Run prediction

```bash
cas12a-ranker predict --input candidates.csv --output predictions.csv
cas12a-ranker predict --input candidates.tsv --output predictions.tsv
cas12a-ranker predict --input candidates.xlsx --sheet Sheet1 --output predictions.xlsx
```

If `--output` is omitted, the result is written next to the input as `<stem>_cas12a_predictions.<extension>`.

The main columns are:

| Column | Interpretation |
|---|---|
| `cas12a_activity_score` | Continuous v1.5 D-ensemble activity prediction |
| `cas12a_activity_rank` | Descending within-file rank; 1 is highest |
| `cas12a_prediction_status` | `success` for valid rows |
| `cas12a_model_route` | `sequence_d_v1_5` for the default path |
| `cas12a_model_version` | Frozen release version |
| `cas12a_prediction_xgboost` | D-model XGBoost component |
| `cas12a_prediction_lightgbm` | D-model LightGBM component |
| `cas12a_prediction_mlp` | D-model MLP component |
| `cas12a_warning_codes` | Empty for a normal prediction; explanation for a retained invalid row |

Activity scores are assay-scale regression outputs, not probabilities. Ranking compares only rows supplied in the same file.

## Input errors

The default mode stops before model inference and writes `<stem>_input_errors.csv` if any row is invalid. The report identifies `record_id`, source row, field and reason.

For data triage only, valid rows may be predicted while invalid rows are retained:

```bash
cas12a-ranker predict --input candidates.csv --on-invalid keep
```

Invalid rows receive `cas12a_prediction_status=invalid_input` and no activity score. Missing headers and reserved `cas12a_` output columns remain fatal. To intentionally replace old `cas12a_` results, pass `--overwrite-results`.

## Minimal verification

```bash
cas12a-ranker self-test
cas12a-ranker predict \
  --input data/examples/minimal_input.csv \
  --output minimal_output.csv
```

The frozen expected file is `data/examples/minimal_expected_output.csv`.

## Python API

```python
import pandas as pd
from cas12a_ml import predict_file

predict_file("candidates.csv", "predictions.csv")
predictions = pd.read_csv("predictions.csv")
print(predictions[["record_id", "cas12a_activity_score", "cas12a_activity_rank"]])
```

Advanced users can call `Cas12aPredictor.predict(frame)` directly. The supported end-user interface remains table-in/table-out so identifiers, metadata and row order stay auditable.
