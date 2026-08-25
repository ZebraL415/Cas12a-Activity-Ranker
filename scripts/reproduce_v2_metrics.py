#!/usr/bin/env python3
"""Recompute v2 metrics and paired cluster-bootstrap changes from frozen predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, prediction).statistic),
        "pearson": float(pearsonr(y, prediction).statistic),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }


def paired_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    repeats: int,
    seed: int,
    protocol: str,
) -> dict[str, float | int | str]:
    y = frame["label_normalized"].to_numpy(float)
    baseline = frame["current_final"].to_numpy(float)
    candidate = frame["replace_a_fixed"].to_numpy(float)
    codes, clusters = pd.factorize(frame[cluster_column], sort=False)
    rows = np.arange(len(frame))
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(repeats) for name in ("spearman", "pearson", "rmse")}
    for iteration in range(repeats):
        draws = rng.integers(0, len(clusters), size=len(clusters))
        counts = np.bincount(draws, minlength=len(clusters))
        selected = np.repeat(rows, counts[codes])
        ys, old, new = y[selected], baseline[selected], candidate[selected]
        samples["spearman"][iteration] = spearmanr(ys, new).statistic - spearmanr(ys, old).statistic
        samples["pearson"][iteration] = pearsonr(ys, new).statistic - pearsonr(ys, old).statistic
        samples["rmse"][iteration] = mean_squared_error(ys, new) ** 0.5 - mean_squared_error(ys, old) ** 0.5
    observed = {
        "spearman": spearmanr(y, candidate).statistic - spearmanr(y, baseline).statistic,
        "pearson": pearsonr(y, candidate).statistic - pearsonr(y, baseline).statistic,
        "rmse": mean_squared_error(y, candidate) ** 0.5 - mean_squared_error(y, baseline) ** 0.5,
    }
    output: dict[str, float | int | str] = {
        "protocol": protocol,
        "comparison": "v2_DBC_minus_v1_ABC",
        "records": len(frame),
        "clusters": len(clusters),
        "resamples": repeats,
        "seed": seed,
    }
    for name, values in samples.items():
        suffix = "_positive_is_worse" if name == "rmse" else ""
        output[f"delta_{name}{suffix}_observed"] = float(observed[name])
        output[f"delta_{name}{suffix}_bootstrap_mean"] = float(values.mean())
        output[f"delta_{name}{suffix}_ci95_low"] = float(np.quantile(values, 0.025))
        output[f"delta_{name}{suffix}_ci95_high"] = float(np.quantile(values, 0.975))
    return output


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=root_default)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    result_dir = root / "results" / "v2_dbc"
    oof = pd.read_csv(result_dir / "meta_oof_predictions.csv")
    fixed = pd.read_csv(result_dir / "fixed_validation_predictions.csv")
    source = pd.read_csv(
        root / "data" / "processed" / "v2_2" / "EasyDesign_2024_V2-2_core_context_feature_table.csv",
        usecols=["record_id", "baseline_split", "target_sequence"],
        low_memory=False,
    )
    fixed = fixed.merge(
        source.loc[source["baseline_split"].eq("baseline_validation"), ["record_id", "target_sequence"]],
        on=["record_id", "target_sequence"],
        validate="one_to_one",
    )

    metric_rows: list[dict[str, float | int | str]] = []
    for protocol, frame in (("crossfitted_meta_oof", oof), ("historical_fixed_validation", fixed)):
        for scheme in ("current_final", "candidate", "b", "c", "replace_a_fixed"):
            metric_rows.append(
                {
                    "protocol": protocol,
                    "scheme": scheme,
                    "records": len(frame),
                    **metrics(frame["label_normalized"].to_numpy(float), frame[scheme].to_numpy(float)),
                }
            )
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(result_dir / "recomputed_metrics.csv", index=False)

    bootstrap = pd.DataFrame(
        [
            paired_bootstrap(
                oof,
                cluster_column="target_group_id",
                repeats=args.bootstrap,
                seed=args.seed,
                protocol="crossfitted_meta_oof",
            ),
            paired_bootstrap(
                fixed,
                cluster_column="target_sequence",
                repeats=args.bootstrap,
                seed=args.seed,
                protocol="historical_fixed_validation",
            ),
        ]
    )
    bootstrap.to_csv(result_dir / "v2_vs_v1_target_cluster_bootstrap.csv", index=False)
    print(metric_table.to_string(index=False))
    print("\nPaired target-cluster bootstrap:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
