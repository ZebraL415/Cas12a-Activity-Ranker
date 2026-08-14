#!/usr/bin/env python3
"""Run end-to-end integrity, feature-reconstruction and inference checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DEFAULT / "src"))

from cas12a_ml import Cas12aPredictor, build_feature_frame  # noqa: E402

EXPECTED_DATA_SHA = "39cda8368c216784507ac002df687b28a4f9cc6f81e2b0e84043e45eddb4c1c0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    profile = root / "data" / "processed" / "v2_2"
    table_path = profile / "EasyDesign_2024_V2-2_core_context_feature_table.csv"
    manifest_path = profile / "feature_manifest.csv"
    folds_path = profile / "frozen_target_grouped_folds.csv"
    required = [
        table_path,
        manifest_path,
        folds_path,
        root / "models" / "primary" / "xgboost_final.json",
        root / "models" / "supporting" / "catboost_final.cbm",
        root / "models" / "training_medians.csv",
        root / "models" / "model_input_metadata.json",
        root / "results" / "fixed_validation_predictions.csv",
        root / "data" / "examples" / "expected_predictions.csv",
    ]
    check(all(path.is_file() for path in required), "all required active files exist")
    check(sha256(table_path) == EXPECTED_DATA_SHA, "authoritative V2-2 SHA-256 matches")

    table = pd.read_csv(table_path, low_memory=False)
    counts = table["baseline_split"].value_counts().to_dict()
    check(len(table) == 11992, "authoritative table has 11,992 records")
    check(counts.get("baseline_train") == 8417, "training partition has 8,417 records")
    check(counts.get("baseline_validation") == 2217, "fixed validation has 2,217 records")
    check(table["record_id"].is_unique, "record_id is unique")

    manifest = pd.read_csv(manifest_path)["feature_name"].tolist()
    metadata = json.loads((root / "models" / "model_input_metadata.json").read_text(encoding="utf-8"))
    active = metadata["active_features"]
    check(len(manifest) == 188 and len(set(manifest)) == 188, "feature manifest has 188 unique inputs")
    check(len(active) == 183 and set(active).issubset(manifest), "deployment has 183 ordered active inputs")

    folds = pd.read_csv(folds_path)
    train = table.loc[table["baseline_split"].eq("baseline_train"), ["record_id", "target_sequence"]]
    joined = train.merge(folds[["record_id", "context_cv_fold"]], on="record_id", validate="one_to_one")
    check(len(joined) == 8417 and set(joined["context_cv_fold"]) == set(range(5)), "frozen OOF folds cover all training records")
    check(joined.groupby("target_sequence")["context_cv_fold"].nunique().max() == 1, "target sequences never cross OOF folds")

    gap_rows = table.loc[table["target_aligned_25"].str.contains("-", regex=False)]
    no_gap_sample = table.loc[~table["target_aligned_25"].str.contains("-", regex=False)].iloc[::97]
    sample = pd.concat([gap_rows, no_gap_sample], ignore_index=True)
    rebuilt = build_feature_frame(sample, manifest=manifest).to_numpy(float)
    expected_features = sample[manifest].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    check(np.isclose(rebuilt, expected_features, atol=1e-12, rtol=0, equal_nan=True).all(), "external feature builder exactly reproduces frozen V2-2 features")

    validation = table.loc[table["baseline_split"].eq("baseline_validation"), ["record_id", "crRNA_sequence", "target_aligned_25"]]
    expected_predictions = pd.read_csv(root / "results" / "fixed_validation_predictions.csv")
    validation = validation.merge(expected_predictions, on="record_id", validate="one_to_one")
    predictions = Cas12aPredictor(root).predict(validation[["record_id", "crRNA_sequence", "target_aligned_25"]])
    mapping = {
        "prediction_xgboost_primary": "XGBoost",
        "prediction_catboost_supporting": "CatBoost",
        "prediction_equal_50_50_sensitivity": "Equal_50_50",
        "prediction_oof_weighted_exploratory": "OOF_weighted",
    }
    for actual, expected in mapping.items():
        difference = np.abs(predictions[actual].to_numpy(float) - validation[expected].to_numpy(float))
        check(bool((difference < 1e-6).all()), f"native {expected} predictions match all 2,217 frozen rows within 1e-6")

    examples = pd.read_csv(root / "data" / "examples" / "external_sequence_pairs.csv", dtype=str)
    expected_examples = pd.read_csv(root / "data" / "examples" / "expected_predictions.csv")
    actual_examples = Cas12aPredictor(root).predict(examples)
    for column in mapping:
        check(np.allclose(actual_examples[column], expected_examples[column], atol=1e-6, rtol=0), f"example output column {column} is reproducible")

    final_metrics = pd.read_csv(root / "results" / "final_metrics.csv").set_index("experiment")
    recomputed = pd.read_csv(root / "results" / "recomputed_metrics.csv").set_index("experiment")
    for column in ["spearman", "pearson", "rmse", "mae", "r2"]:
        check(np.allclose(final_metrics.loc[recomputed.index, column], recomputed[column], atol=1e-8, rtol=0), f"recomputed {column} matches frozen metrics")

    checksum_path = root / "data" / "metadata" / "sha256_manifest.tsv"
    if checksum_path.exists():
        checksum_table = pd.read_csv(checksum_path, sep="\t")
        checksum_ok = True
        for relative_path, size_bytes, expected_sha in checksum_table.itertuples(index=False, name=None):
            path = root / relative_path
            if not path.is_file() or path.stat().st_size != int(size_bytes) or sha256(path) != expected_sha:
                checksum_ok = False
                print(f"FAIL  checksum mismatch: {relative_path}")
                break
        check(checksum_ok, f"repository checksum manifest verifies {len(checksum_table)} files")

    print("\nRepository verification completed successfully.")


if __name__ == "__main__":
    main()
