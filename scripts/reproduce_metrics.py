#!/usr/bin/env python3
"""Recompute v1.5 metrics and paired target-cluster uncertainty intervals."""

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


def bootstrap_delta(
    frame: pd.DataFrame,
    comparator: str,
    baseline: str,
    repeats: int,
    seed: int,
) -> dict[str, float | int | str]:
    y = frame["label_normalized"].to_numpy(float)
    base = frame[baseline].to_numpy(float)
    other = frame[comparator].to_numpy(float)
    codes, targets = pd.factorize(frame["target_sequence"], sort=False)
    rng = np.random.default_rng(seed)
    delta_spearman = np.empty(repeats)
    delta_pearson = np.empty(repeats)
    delta_rmse = np.empty(repeats)
    row_ids = np.arange(len(frame))
    for iteration in range(repeats):
        target_draws = rng.integers(0, len(targets), size=len(targets))
        target_counts = np.bincount(target_draws, minlength=len(targets))
        sampled = np.repeat(row_ids, target_counts[codes])
        ys, bs, os = y[sampled], base[sampled], other[sampled]
        delta_spearman[iteration] = spearmanr(ys, os).statistic - spearmanr(ys, bs).statistic
        delta_pearson[iteration] = pearsonr(ys, os).statistic - pearsonr(ys, bs).statistic
        delta_rmse[iteration] = mean_squared_error(ys, os) ** 0.5 - mean_squared_error(ys, bs) ** 0.5

    def observed(metric: str) -> float:
        if metric == "spearman":
            return float(spearmanr(y, other).statistic - spearmanr(y, base).statistic)
        if metric == "pearson":
            return float(pearsonr(y, other).statistic - pearsonr(y, base).statistic)
        return float(mean_squared_error(y, other) ** 0.5 - mean_squared_error(y, base) ** 0.5)

    result: dict[str, float | int | str] = {
        "comparison": f"{comparator}_minus_{baseline}",
        "records": len(frame),
        "target_clusters": len(targets),
        "resamples": repeats,
        "seed": seed,
    }
    for metric_name, samples in {
        "spearman": delta_spearman,
        "pearson": delta_pearson,
        "rmse_positive_is_worse": delta_rmse,
    }.items():
        observed_name = "rmse" if metric_name.startswith("rmse") else metric_name
        result[f"delta_{metric_name}_observed"] = observed(observed_name)
        result[f"delta_{metric_name}_bootstrap_mean"] = float(samples.mean())
        result[f"delta_{metric_name}_ci95_low"] = float(np.quantile(samples, 0.025))
        result[f"delta_{metric_name}_ci95_high"] = float(np.quantile(samples, 0.975))
    return result


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=root_default)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output_dir = root / "results" / "v1_5_d_ensemble"
    d_predictions = pd.read_csv(output_dir / "fixed_validation_predictions.csv")
    legacy_predictions = pd.read_csv(root / "results" / "fixed_validation_predictions.csv")
    table = pd.read_csv(
        root / "data" / "processed" / "v2_2" / "EasyDesign_2024_V2-2_core_context_feature_table.csv",
        usecols=["record_id", "baseline_split", "target_sequence"],
        low_memory=False,
    )
    validation = table.loc[table["baseline_split"].eq("baseline_validation"), ["record_id", "target_sequence"]]
    frame = d_predictions.merge(validation, on=["record_id", "target_sequence"], validate="one_to_one")
    frame = frame.merge(
        legacy_predictions[["record_id", "XGBoost"]].rename(columns={"XGBoost": "v1_xgboost"}),
        on="record_id",
        validate="one_to_one",
    )
    if len(frame) != 2217:
        raise ValueError(f"Expected 2,217 validation records; got {len(frame)}")

    rows = []
    for model in ["xgb", "lightgbm", "mlp", "heterogeneous_ensemble"]:
        rows.append(
            {
                "protocol": "historical_fixed_validation",
                "model": model,
                "n_validation": len(frame),
                **metrics(frame["label_normalized"].to_numpy(float), frame[model].to_numpy(float)),
            }
        )
    metric_table = pd.DataFrame(rows).sort_values(["spearman", "pearson"], ascending=False)
    metric_table.to_csv(output_dir / "recomputed_fixed_validation_metrics.csv", index=False)

    bootstrap = pd.DataFrame(
        [
            bootstrap_delta(
                frame,
                "heterogeneous_ensemble",
                "v1_xgboost",
                args.bootstrap,
                args.seed,
            )
        ]
    )
    bootstrap.to_csv(output_dir / "d_vs_v1_xgb_target_bootstrap.csv", index=False)
    print(metric_table.to_string(index=False))
    print("\nPaired target-cluster bootstrap (D minus v1.0 XGBoost):")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
