"""Validated table-in/table-out interface for batch prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import sequence_pair_features
from .predict import Cas12aPredictor


SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx"}
REQUIRED_COLUMNS = ("record_id", "crRNA_sequence", "target_aligned_25")
RESULT_COLUMNS = (
    "cas12a_activity_score",
    "cas12a_activity_rank",
    "cas12a_prediction_status",
    "cas12a_model_route",
    "cas12a_model_version",
    "cas12a_prediction_xgboost",
    "cas12a_prediction_lightgbm",
    "cas12a_prediction_mlp",
    "cas12a_warning_codes",
    "prediction_xgboost_primary",
    "prediction_catboost_supporting",
    "prediction_equal_50_50_sensitivity",
    "prediction_oof_weighted_exploratory",
    "rank_xgboost_descending",
)
TEXT_RESULT_COLUMNS = {
    "cas12a_prediction_status",
    "cas12a_model_route",
    "cas12a_model_version",
    "cas12a_warning_codes",
}


@dataclass
class InputValidationError(ValueError):
    """Raised after writing a row-level input error report."""

    error_path: Path
    error_count: int

    def __str__(self) -> str:
        return f"Input validation failed with {self.error_count} error(s); see {self.error_path}"


def read_table(path: str | Path, *, sheet: str | int = 0) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported input format {suffix!r}; use CSV, TSV or XLSX")
    if suffix == ".csv":
        return pd.read_csv(source, dtype=str, keep_default_na=False)
    if suffix == ".tsv":
        return pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
    return pd.read_excel(source, sheet_name=sheet, dtype=str, keep_default_na=False)


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported output format {suffix!r}; use CSV, TSV or XLSX")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        frame.to_csv(destination, index=False)
    elif suffix == ".tsv":
        frame.to_csv(destination, sep="\t", index=False)
    else:
        frame.to_excel(destination, index=False)
    return destination


def default_output_path(input_path: str | Path) -> Path:
    source = Path(input_path)
    return source.with_name(f"{source.stem}_cas12a_predictions{source.suffix.lower()}")


def _error_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_input_errors.csv")


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a row-level error table; an empty result means the input is valid."""
    errors: list[dict[str, object]] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    for column in missing:
        errors.append(
            {
                "record_id": "",
                "source_row": 1,
                "field": column,
                "error": f"Missing required column: {column}",
            }
        )
    reserved = [column for column in frame.columns if column.startswith("cas12a_")]
    for column in reserved:
        errors.append(
            {
                "record_id": "",
                "source_row": 1,
                "field": column,
                "error": "Input contains a reserved cas12a_ result column",
            }
        )
    if missing:
        return pd.DataFrame(errors)

    record_ids = frame["record_id"].astype(str).str.strip()
    for index in frame.index[record_ids.eq("")]:
        errors.append(
            {
                "record_id": "",
                "source_row": int(frame.index.get_loc(index)) + 2,
                "field": "record_id",
                "error": "record_id must be nonempty",
            }
        )
    duplicate = record_ids.duplicated(keep=False)
    for index in frame.index[duplicate]:
        errors.append(
            {
                "record_id": record_ids.loc[index],
                "source_row": int(frame.index.get_loc(index)) + 2,
                "field": "record_id",
                "error": "record_id must be unique",
            }
        )
    for position, (index, row) in enumerate(frame.iterrows(), start=2):
        try:
            sequence_pair_features(row["crRNA_sequence"], row["target_aligned_25"])
        except ValueError as exc:
            message = str(exc)
            field = "crRNA_sequence" if "crRNA_sequence" in message else "target_aligned_25"
            errors.append(
                {
                    "record_id": record_ids.loc[index],
                    "source_row": position,
                    "field": field,
                    "error": message,
                }
            )
    return pd.DataFrame(errors, columns=["record_id", "source_row", "field", "error"])


def predict_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    primary_model: str = "d",
    sheet: str | int = 0,
    on_invalid: str = "error",
    overwrite_results: bool = False,
) -> Path:
    """Read a candidate table, append predictions, and write one output row per input row."""
    source = Path(input_path)
    destination = Path(output_path) if output_path else default_output_path(source)
    if on_invalid not in {"error", "keep"}:
        raise ValueError("on_invalid must be 'error' or 'keep'")
    frame = read_table(source, sheet=sheet)
    reserved = [column for column in frame.columns if column.startswith("cas12a_")]
    if overwrite_results and reserved:
        frame = frame.drop(columns=reserved)
    errors = validate_input(frame)
    fatal_header_error = bool(
        not errors.empty
        and errors["record_id"].astype(str).eq("").any()
        and errors["source_row"].eq(1).any()
    )
    if not errors.empty and (on_invalid == "error" or fatal_header_error):
        error_path = _error_path(source)
        errors.to_csv(error_path, index=False)
        raise InputValidationError(error_path=error_path, error_count=len(errors))

    if errors.empty:
        predictor = Cas12aPredictor(repository_root, primary_model=primary_model)
        return write_table(predictor.predict(frame), destination)

    invalid_ids = set(errors.loc[errors["record_id"].ne(""), "record_id"].astype(str))
    valid_mask = ~frame["record_id"].astype(str).isin(invalid_ids)
    output = frame.copy()
    for column in RESULT_COLUMNS:
        output[column] = "" if column in TEXT_RESULT_COLUMNS else np.nan
    if valid_mask.any():
        predictor = Cas12aPredictor(repository_root, primary_model=primary_model)
        valid_predictions = predictor.predict(frame.loc[valid_mask].copy())
        for column in RESULT_COLUMNS:
            output.loc[valid_mask, column] = valid_predictions[column].to_numpy()
    output.loc[~valid_mask, "cas12a_prediction_status"] = "invalid_input"
    warning_by_id = errors.groupby("record_id")["error"].apply(lambda values: " | ".join(values))
    output.loc[~valid_mask, "cas12a_warning_codes"] = (
        output.loc[~valid_mask, "record_id"].astype(str).map(warning_by_id)
    )
    return write_table(output, destination)
