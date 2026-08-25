#!/usr/bin/env python3
"""Run end-to-end integrity, mapping, v2 model and inference checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DEFAULT / "src"))

from cas12a_ml import Cas12aPredictor, build_feature_frame  # noqa: E402
from cas12a_ml.cli import run_self_test  # noqa: E402
from cas12a_ml.mapping import MAPPING_FIELDS, TemplateMapper  # noqa: E402

EXPECTED_DATA_SHA = "39cda8368c216784507ac002df687b28a4f9cc6f81e2b0e84043e45eddb4c1c0"
EXPECTED_D_MODEL_SHA = {
    "models/primary/d_xgboost.json": "cbd7d6f8e0fc6223c8e46fdad0635567056d04d92e92d8334ebb5b84f1b36d3b",
    "models/primary/d_lightgbm.txt": "fe68bbca8631a3cda084b35c78a48e9a399de3bd3b3ad0841c3873b0170d2602",
    "models/primary/d_mlp_pipeline.joblib": "5b2f5bde6d8b08b05b3d298b64312961c1482456a2c27e3c3e59440675f753a6",
}
EXPECTED_D_METRICS = {
    "spearman": 0.7709418405865481,
    "pearson": 0.7537795124035377,
    "rmse": 0.4204865852942966,
    "mae": 0.32604111754079396,
    "r2": 0.5639133779010039,
}
EXPECTED_V2_METRICS = {
    "spearman": 0.8461558326118818,
    "pearson": 0.8354807207247226,
    "rmse": 0.35225309183393855,
    "mae": 0.27072122932068565,
    "r2": 0.693960064933054,
}


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


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    profile = root / "data" / "processed" / "v2_2"
    table_path = profile / "EasyDesign_2024_V2-2_core_context_feature_table.csv"
    manifest_path = profile / "feature_manifest.csv"
    folds_path = profile / "frozen_target_grouped_folds.csv"
    d_result_dir = root / "results" / "v1_5_d_ensemble"
    v2_result_dir = root / "results" / "v2_dbc"
    required = [
        table_path,
        manifest_path,
        folds_path,
        root / "models" / "training_medians.csv",
        root / "models" / "model_input_metadata.json",
        root / "models" / "d_model_metadata.json",
        root / "models" / "v2_model_metadata.json",
        root / "models" / "mapping" / "feature_manifest.csv",
        root / "models" / "mapping" / "guide_history_reference.csv",
        root / "models" / "mapping" / "mapping_history_reference.csv",
        root / "models" / "mapping" / "table_s2_template_reference.csv",
        d_result_dir / "fixed_validation_predictions.csv",
        d_result_dir / "oof_predictions.csv",
        d_result_dir / "oof_weight_grid.csv",
        v2_result_dir / "fixed_validation_predictions.csv",
        v2_result_dir / "meta_oof_predictions.csv",
        v2_result_dir / "weight_audit" / "fine_pooled_dbc_weight_grid_20301.csv",
        v2_result_dir / "weight_audit" / "fine_crossfitted_weight_grids_101505.csv",
        root / "data" / "examples" / "minimal_input.csv",
        root / "data" / "examples" / "minimal_expected_output.csv",
        *[root / path for path in EXPECTED_D_MODEL_SHA],
    ]
    check(all(path.is_file() for path in required), "all required v2.0 files exist")
    check(sha256(table_path) == EXPECTED_DATA_SHA, "authoritative V2-2 SHA-256 matches")
    for relative_path, expected_sha in EXPECTED_D_MODEL_SHA.items():
        check(sha256(root / relative_path) == expected_sha, f"{relative_path} SHA-256 matches")

    table = pd.read_csv(table_path, low_memory=False)
    counts = table["baseline_split"].value_counts().to_dict()
    check(len(table) == 11992, "authoritative table has 11,992 records")
    check(counts.get("baseline_train") == 8417, "training partition has 8,417 records")
    check(counts.get("baseline_validation") == 2217, "fixed validation has 2,217 records")
    check(table["record_id"].is_unique, "record_id is unique")

    manifest = pd.read_csv(manifest_path)["feature_name"].tolist()
    metadata = json.loads((root / "models" / "model_input_metadata.json").read_text(encoding="utf-8"))
    d_metadata = json.loads((root / "models" / "d_model_metadata.json").read_text(encoding="utf-8"))
    v2_metadata = json.loads((root / "models" / "v2_model_metadata.json").read_text(encoding="utf-8"))
    active = metadata["active_features"]
    check(len(manifest) == 188 and len(set(manifest)) == 188, "feature manifest has 188 unique inputs")
    check(len(active) == 183 and set(active).issubset(manifest), "deployment has 183 ordered active inputs")
    check(d_metadata["weights"] == {"xgboost": 0.35, "lightgbm": 0.59, "mlp": 0.06}, "D weights are frozen at 0.35/0.59/0.06")
    check(v2_metadata["weights"] == {"d": 0.2, "b": 0.47, "c": 0.33}, "v2 D/B/C weights are frozen at 0.20/0.47/0.33")
    for relative_path, expected_sha in v2_metadata["artifact_sha256"].items():
        check(
            sha256(root / "models" / "mapping" / relative_path) == expected_sha,
            f"models/mapping/{relative_path} SHA-256 matches",
        )

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

    mapped_source = table.loc[
        table["baseline_split"].isin(["baseline_train", "baseline_validation"]),
        ["target_aligned_25", *MAPPING_FIELDS],
    ].reset_index(drop=True)
    remapped = TemplateMapper(
        root / "models" / "mapping" / "table_s2_template_reference.csv"
    ).map_frame(mapped_source)
    mapping_equal = True
    for column in MAPPING_FIELDS:
        expected = mapped_source[column].fillna("").astype(str).to_numpy()
        actual = remapped[column].fillna("").astype(str).to_numpy()
        mapping_equal = mapping_equal and bool(np.array_equal(expected, actual))
    check(mapping_equal, "automatic mapping exactly reproduces all 10,634 frozen train/validation rows")

    validation_pairs = table.loc[
        table["baseline_split"].eq("baseline_validation"),
        ["record_id", "crRNA_sequence", "target_aligned_25"],
    ]
    frozen = pd.read_csv(d_result_dir / "fixed_validation_predictions.csv")
    validation_pairs = validation_pairs.merge(frozen, on="record_id", validate="one_to_one")
    check(len(validation_pairs) == 2217, "frozen D predictions cover all 2,217 validation rows")
    d_predictions = Cas12aPredictor(root, primary_model="d").predict(
        validation_pairs[["record_id", "crRNA_sequence", "target_aligned_25"]]
    )
    mapping = {
        "cas12a_prediction_xgboost": "xgb",
        "cas12a_prediction_lightgbm": "lightgbm",
        "cas12a_prediction_mlp": "mlp",
        "cas12a_activity_score": "heterogeneous_ensemble",
    }
    for actual, expected in mapping.items():
        difference = np.abs(d_predictions[actual].to_numpy(float) - validation_pairs[expected].to_numpy(float))
        check(bool((difference < 1e-6).all()), f"native {expected} predictions match all 2,217 frozen rows within 1e-6")

    recomputed = metric_values(
        validation_pairs["label_normalized"].to_numpy(float),
        d_predictions["cas12a_activity_score"].to_numpy(float),
    )
    for metric_name, expected in EXPECTED_D_METRICS.items():
        check(abs(recomputed[metric_name] - expected) < 1e-8, f"v1.5 D {metric_name} matches the frozen result")

    weight_grid = pd.read_csv(d_result_dir / "oof_weight_grid.csv")
    best = weight_grid.loc[weight_grid["spearman"].idxmax()]
    check(
        np.allclose(
            [best["xgb_weight"], best["lightgbm_weight"], best["mlp_weight"]],
            [0.35, 0.59, 0.06],
            atol=1e-12,
            rtol=0,
        ),
        "training OOF grid independently identifies the frozen D weights",
    )

    frozen_v2 = pd.read_csv(v2_result_dir / "fixed_validation_predictions.csv")
    validation_v2 = table.loc[
        table["baseline_split"].eq("baseline_validation"),
        ["record_id", "crRNA_sequence", "target_aligned_25"],
    ].merge(frozen_v2, on="record_id", validate="one_to_one")
    v2_predictions = Cas12aPredictor(root).predict(
        validation_v2[["record_id", "crRNA_sequence", "target_aligned_25"]]
    )
    for actual, expected in {
        "cas12a_prediction_d": "candidate",
        "cas12a_prediction_b": "b",
        "cas12a_prediction_c": "c",
        "cas12a_prediction_full_dbc": "replace_a_fixed",
        "cas12a_activity_score": "replace_a_fixed",
    }.items():
        difference = np.abs(v2_predictions[actual].to_numpy(float) - validation_v2[expected].to_numpy(float))
        check(bool((difference < 1e-6).all()), f"v2 {expected} predictions match all 2,217 frozen rows within 1e-6")
    recomputed_v2 = metric_values(
        validation_v2["label_normalized"].to_numpy(float),
        v2_predictions["cas12a_activity_score"].to_numpy(float),
    )
    for metric_name, expected in EXPECTED_V2_METRICS.items():
        check(abs(recomputed_v2[metric_name] - expected) < 1e-8, f"v2 {metric_name} matches the frozen result")
    fine_grid = pd.read_csv(v2_result_dir / "weight_audit" / "fine_pooled_dbc_weight_grid_20301.csv")
    crossfold_grid = pd.read_csv(v2_result_dir / "weight_audit" / "fine_crossfitted_weight_grids_101505.csv")
    fine_decision = json.loads(
        (v2_result_dir / "weight_audit" / "fine_weight_decision.json").read_text(encoding="utf-8")
    )
    check(len(fine_grid) == 20301, "fine pooled D/B/C audit contains all 20,301 weight combinations")
    check(len(crossfold_grid) == 101505, "cross-fitted D/B/C audit contains all 101,505 fold-specific combinations")
    check(bool(fine_decision["decision"]["keep_locked_20_47_33"]), "weight audit retains the locked 0.20/0.47/0.33 system")

    run_self_test(root)

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
