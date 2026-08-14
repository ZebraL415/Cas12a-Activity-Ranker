#!/usr/bin/env python3
"""Recompute final metrics and paired target-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_delta(frame: pd.DataFrame, comparator: str, repeats: int, seed: int) -> dict[str, float | int | str]:
    y = frame["label_normalized"].to_numpy(float)
    xgb = frame["XGBoost"].to_numpy(float)
    other = frame[comparator].to_numpy(float)
    # Preserve first-appearance order to match the frozen audit implementation.
    codes, targets = pd.factorize(frame["target_sequence"], sort=False)
    rng = np.random.default_rng(seed)
    delta_spearman = np.empty(repeats)
    delta_rmse = np.empty(repeats)
    row_ids = np.arange(len(frame))
    for iteration in range(repeats):
        target_draws = rng.integers(0, len(targets), size=len(targets))
        target_counts = np.bincount(target_draws, minlength=len(targets))
        sampled = np.repeat(row_ids, target_counts[codes])
        ys, xs, os = y[sampled], xgb[sampled], other[sampled]
        delta_spearman[iteration] = spearmanr(ys, os).statistic - spearmanr(ys, xs).statistic
        delta_rmse[iteration] = mean_squared_error(ys, os) ** 0.5 - mean_squared_error(ys, xs) ** 0.5
    original_s = spearmanr(y, other).statistic - spearmanr(y, xgb).statistic
    original_r = mean_squared_error(y, other) ** 0.5 - mean_squared_error(y, xgb) ** 0.5
    return {
        "comparator_minus_xgboost": comparator,
        "records": len(frame),
        "target_clusters": len(targets),
        "resamples": repeats,
        "seed": seed,
        "delta_spearman_observed": original_s,
        "delta_spearman_bootstrap_mean": float(delta_spearman.mean()),
        "delta_spearman_ci95_low": float(np.quantile(delta_spearman, 0.025)),
        "delta_spearman_ci95_high": float(np.quantile(delta_spearman, 0.975)),
        "delta_rmse_observed_positive_is_worse": original_r,
        "delta_rmse_bootstrap_mean": float(delta_rmse.mean()),
        "delta_rmse_ci95_low": float(np.quantile(delta_rmse, 0.025)),
        "delta_rmse_ci95_high": float(np.quantile(delta_rmse, 0.975)),
    }


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=root_default)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    predictions = pd.read_csv(root / "results" / "fixed_validation_predictions.csv")
    table = pd.read_csv(
        root / "data" / "processed" / "v2_2" / "EasyDesign_2024_V2-2_core_context_feature_table.csv",
        usecols=["record_id", "baseline_split", "target_sequence"],
        low_memory=False,
    )
    validation = table.loc[table["baseline_split"].eq("baseline_validation"), ["record_id", "target_sequence"]]
    frame = predictions.merge(validation, on="record_id", validate="one_to_one")
    if len(frame) != 2217:
        raise ValueError(f"Expected 2,217 validation records; got {len(frame)}")

    rows = []
    for experiment in ["XGBoost", "CatBoost", "Equal_50_50", "OOF_weighted"]:
        rows.append({"experiment": experiment, "n_validation": len(frame), **metrics(frame["label_normalized"].to_numpy(float), frame[experiment].to_numpy(float))})
    metric_table = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    metric_table.to_csv(root / "results" / "recomputed_metrics.csv", index=False)

    bootstrap = pd.DataFrame([
        bootstrap_delta(frame, "Equal_50_50", args.bootstrap, args.seed),
        bootstrap_delta(frame, "OOF_weighted", args.bootstrap, args.seed),
    ])
    bootstrap.to_csv(root / "results" / "ensemble_vs_xgb_target_bootstrap.csv", index=False)
    print(metric_table.to_string(index=False))
    print("\nPaired target-cluster bootstrap:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
