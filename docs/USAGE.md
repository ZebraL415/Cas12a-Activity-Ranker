# Usage guide

## Input contract

Use one row per already aligned crRNA–target pair in CSV, TSV or XLSX format:

```csv
record_id,crRNA_sequence,target_aligned_25,sample_group
candidate_01,TTTGTTGGGGCGTCCTTAGACGCCA,TTTGTTGGGGCGTCCTTAGACGCCA,screen_1
candidate_02,TTTGGGCTAGGTGTGATAGGAATGG,TTTGGGCTAGGAGTGATAGGAATGG,screen_1
```

Required, case-sensitive columns:

- `record_id`: nonempty and unique;
- `crRNA_sequence`: exactly 25 A/C/G/T positions; U is normalized to T;
- `target_aligned_25`: exactly 25 aligned A/C/G/T/`-` positions.

All other columns and row order are preserved. The program does not create the biological alignment. It does automatically map the supplied aligned target to its frozen EasyDesign template reference, so user-supplied `mapping_*` columns are rejected.

## Commands

```bash
cas12a-ranker predict --input candidates.csv --output predictions.csv
cas12a-ranker predict --input candidates.tsv --output predictions.tsv
cas12a-ranker predict --input candidates.xlsx --sheet Sheet1 --output predictions.xlsx
```

If `--output` is omitted, the result is written as `<stem>_cas12a_predictions.<extension>`.

The default model is `v2`. Retained comparison routes are explicit:

```bash
cas12a-ranker predict --input candidates.csv --model d
cas12a-ranker predict --input candidates.csv --model xgboost-legacy
```

## Output contract

| Column | Interpretation |
|---|---|
| `cas12a_activity_score` | Final continuous score used for this row |
| `cas12a_activity_rank` | Global descending rank, or blank when full/fallback routes are mixed |
| `cas12a_rank_within_route` | Descending rank among comparable rows using the same route |
| `cas12a_prediction_d/b/c` | Module-level predictions |
| `cas12a_prediction_full_dbc` | Full `0.20 D + 0.47 B + 0.33 C` score; blank after mapping failure |
| `cas12a_prediction_status` | `success`, `fallback_success` or `invalid_input` |
| `cas12a_model_route` | `mapping_dbc_v2`, `sequence_d_fallback_v2`, or an explicitly selected old route |
| `cas12a_mapping_*` | Mapping status, confidence, counts and every retained candidate key |
| `cas12a_guide_seen` | Whether this guide occurs in the frozen training history |
| `cas12a_template_key_seen` | Whether this combined mapping key occurs in training history |
| `cas12a_*_reference_count` | Number of training records supporting that history key |
| `cas12a_warning_codes` | Semicolon-separated row-level cautions |

Scores are assay-scale regression outputs, not probabilities or clinical thresholds.

## Mapping and ranking controls

By default, an unmapped row receives the sequence-only D prediction and explicit warnings:

```bash
cas12a-ranker predict --input candidates.csv --fallback-policy sequence
```

To treat any unmapped row as an error:

```bash
cas12a-ranker predict --input candidates.csv --fallback-policy error
```

If a file mixes `mapping_dbc_v2` and `sequence_d_fallback_v2`, the program withholds the global rank. It still provides `cas12a_rank_within_route`. To deliberately rank unlike routes together:

```bash
cas12a-ranker predict --input candidates.csv --allow-mixed-ranking
```

## Invalid inputs

Default behavior stops before inference and writes `<stem>_input_errors.csv`. For triage, keep invalid rows while predicting valid rows:

```bash
cas12a-ranker predict --input candidates.csv --on-invalid keep
```

Invalid rows receive `cas12a_prediction_status=invalid_input` and no score. Missing headers, user mapping columns and pre-existing reserved `cas12a_` columns are fatal. Use `--overwrite-results` only to replace old `cas12a_` output columns.

## Minimal verification and Python API

```bash
cas12a-ranker self-test
cas12a-ranker predict --input data/examples/minimal_input.csv --output minimal_output.csv
```

```python
import pandas as pd
from cas12a_ml import predict_file

predict_file("candidates.csv", "predictions.csv")
predictions = pd.read_csv("predictions.csv")
print(predictions[["record_id", "cas12a_activity_score", "cas12a_model_route"]])
```
