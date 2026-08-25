"""Command-line interface for file-based Cas12a activity prediction."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .io import InputValidationError, predict_file, read_table


def _repository_root(value: str | Path | None = None) -> Path:
    return Path(value).resolve() if value else Path(__file__).resolve().parents[2]


def run_self_test(repository_root: str | Path | None = None) -> None:
    root = _repository_root(repository_root)
    input_path = root / "data" / "examples" / "minimal_input.csv"
    expected_path = root / "data" / "examples" / "minimal_expected_output.csv"
    expected = read_table(expected_path)
    with tempfile.TemporaryDirectory(prefix="cas12a-ranker-self-test-") as temporary:
        actual_path = Path(temporary) / "minimal_output.csv"
        predict_file(input_path, actual_path, repository_root=root)
        actual = read_table(actual_path)
    if list(actual.columns) != list(expected.columns):
        raise AssertionError("Self-test output columns differ from the frozen example")
    if actual["record_id"].tolist() != expected["record_id"].tolist():
        raise AssertionError("Self-test record_id order differs from the frozen example")
    numeric = [
        "cas12a_activity_score",
        "cas12a_activity_rank",
        "cas12a_rank_within_route",
        "cas12a_prediction_xgboost",
        "cas12a_prediction_lightgbm",
        "cas12a_prediction_mlp",
        "cas12a_prediction_d",
        "cas12a_prediction_b",
        "cas12a_prediction_c",
        "cas12a_prediction_full_dbc",
        "cas12a_mapping_hit_count",
        "cas12a_mapping_template_count",
        "cas12a_mapping_group_count",
        "cas12a_mapping_orientation_count",
        "cas12a_guide_reference_count",
        "cas12a_template_reference_count",
        "prediction_xgboost_primary",
        "prediction_catboost_supporting",
        "prediction_equal_50_50_sensitivity",
        "prediction_oof_weighted_exploratory",
        "rank_xgboost_descending",
    ]
    for column in numeric:
        if not np.allclose(
            pd.to_numeric(actual[column], errors="coerce").to_numpy(float),
            pd.to_numeric(expected[column], errors="coerce").to_numpy(float),
            atol=1e-6,
            rtol=0,
            equal_nan=True,
        ):
            raise AssertionError(f"Self-test prediction mismatch: {column}")
    categorical = [column for column in expected.columns if column not in numeric]
    for column in categorical:
        if actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise AssertionError(f"Self-test value mismatch: {column}")
    print("PASS  Cas12a Activity Ranker self-test")
    print("Model version: 2.0.0")
    print(f"Validated examples: {len(actual)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cas12a-ranker")
    parser.add_argument("--repository-root", type=Path, default=None, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="append activity predictions to CSV, TSV or XLSX")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path)
    predict.add_argument("--sheet", default=0, help="XLSX sheet name or zero-based index")
    predict.add_argument("--model", choices=["v2", "d", "xgboost-legacy"], default="v2")
    predict.add_argument("--on-invalid", choices=["error", "keep"], default="error")
    predict.add_argument(
        "--fallback-policy",
        choices=["sequence", "error"],
        default="sequence",
        help="use sequence-only D or stop when automatic mapping finds no template",
    )
    predict.add_argument(
        "--allow-mixed-ranking",
        action="store_true",
        help="rank mapping-aware and sequence-fallback rows together despite different routes",
    )
    predict.add_argument("--overwrite-results", action="store_true")

    subparsers.add_parser("self-test", help="run the bundled minimal end-to-end example")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = _repository_root(args.repository_root)
    if args.command == "self-test":
        run_self_test(root)
        return
    sheet: str | int = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    try:
        destination = predict_file(
            args.input,
            args.output,
            repository_root=root,
            primary_model=args.model,
            sheet=sheet,
            on_invalid=args.on_invalid,
            overwrite_results=args.overwrite_results,
            fallback_policy=args.fallback_policy,
            allow_mixed_ranking=args.allow_mixed_ranking,
        )
    except (InputValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    rows = len(read_table(destination))
    print(f"Predicted {rows} sequence pairs -> {destination.resolve()}")


if __name__ == "__main__":
    main()
