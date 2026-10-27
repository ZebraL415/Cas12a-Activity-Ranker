from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "last_week_experiments" / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

from run_last_week_experiments import (  # noqa: E402
    BASE_SEED,
    FOLD,
    LABEL,
    SEEDS,
    atomic_csv,
    atomic_json,
    load_data,
    make_xgb,
    prepare,
    raw_features,
)
from run_guide_architecture import make_guide_target_encoding  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"
ALPHABET = ("A", "C", "G", "T", "-", "N")
PAIR_CATEGORIES = tuple(left + right for left in ALPHABET for right in ALPHABET)


def normalize_sequence(value: Any, length: int = 25) -> str:
    text = str(value).upper() if pd.notna(value) else ""
    text = "".join(base if base in ALPHABET else "N" for base in text)
    return (text + "N" * length)[:length]


def pair_onehot(frame: pd.DataFrame, length: int = 25) -> pd.DataFrame:
    guide = frame["crRNA_sequence"].map(lambda value: normalize_sequence(value, length))
    target = frame["target_aligned_25"].map(lambda value: normalize_sequence(value, length))
    matrix = np.zeros((len(frame), length * len(PAIR_CATEGORIES)), dtype=np.uint8)
    category_index = {category: index for index, category in enumerate(PAIR_CATEGORIES)}
    for row, (left, right) in enumerate(zip(guide, target)):
        for position, (a, b) in enumerate(zip(left, right)):
            matrix[row, position * len(PAIR_CATEGORIES) + category_index[a + b]] = 1
    columns = [
        f"pair_identity_pos_{position + 1:02d}_{category}"
        for position in range(length)
        for category in PAIR_CATEGORIES
    ]
    return pd.DataFrame(matrix, columns=columns, index=frame.index)


def metric_dict(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, prediction).statistic),
        "pearson": float(pearsonr(y, prediction).statistic),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }


def metric_rows(frame: pd.DataFrame, columns: list[str], stage: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        rows.append({"stage": stage, "experiment": column, "fold": "pooled", "n": len(frame), **metric_dict(frame[LABEL], frame[column])})
        for fold, subset in frame.groupby("fold", sort=True):
            rows.append({"stage": stage, "experiment": column, "fold": int(fold), "n": len(subset), **metric_dict(subset[LABEL], subset[column])})
    return pd.DataFrame(rows)


def load_parent(train: pd.DataFrame) -> pd.DataFrame:
    parent = pd.read_csv(RESULTS / "oof_guide_residual_confirmation_predictions.csv")
    result = train[["record_id", FOLD, LABEL]].merge(
        parent[["record_id", "fold", "guide_te_bagged", "guide_residual_bagged"]],
        on="record_id",
        validate="one_to_one",
    )
    if not np.array_equal(result[FOLD].to_numpy(int), result["fold"].to_numpy(int)):
        raise RuntimeError("Fold alignment failed")
    return result[["record_id", "fold", LABEL, "guide_te_bagged", "guide_residual_bagged"]]


def run_screen() -> str:
    print("Building deterministic position-specific pair identity features...", flush=True)
    train, _, features = load_data()
    train = train.reset_index(drop=True)
    parent = load_parent(train)
    base_x = raw_features(train, features)
    pair_x = pair_onehot(train)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    predictions = {
        "pair_guide_te_xgb_seed42": np.full(len(train), np.nan),
        "pair_guide_residual_xgb_seed42": np.full(len(train), np.nan),
    }
    active_pair_counts: list[int] = []

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        holdout_mask = folds == fold
        outer_train = train.loc[train_mask]
        holdout = train.loc[holdout_mask]
        prepared = prepare(base_x.loc[train_mask], base_x.loc[holdout_mask])
        train_te, holdout_te = make_guide_target_encoding(outer_train, holdout)
        pair_train = pair_x.loc[train_mask]
        pair_holdout = pair_x.loc[holdout_mask]
        keep_pair = pair_train.nunique(dropna=False).gt(1)
        active_pair_counts.append(int(keep_pair.sum()))
        pair_train = pair_train.loc[:, keep_pair]
        pair_holdout = pair_holdout.loc[:, keep_pair]
        augmented_train = pd.concat([prepared.train, train_te, pair_train], axis=1)
        augmented_holdout = pd.concat([prepared.validation, holdout_te, pair_holdout], axis=1)

        raw_model = make_xgb(BASE_SEED)
        raw_model.fit(augmented_train, y[train_mask])
        raw_prediction = raw_model.predict(augmented_holdout)
        predictions["pair_guide_te_xgb_seed42"][holdout_mask] = raw_prediction

        residual_target = y[train_mask] - train_te["te_guide_all"].to_numpy(float)
        residual_model = make_xgb(BASE_SEED)
        residual_model.fit(augmented_train, residual_target)
        residual_prediction = holdout_te["te_guide_all"].to_numpy(float) + residual_model.predict(augmented_holdout)
        predictions["pair_guide_residual_xgb_seed42"][holdout_mask] = residual_prediction
        print(
            f"pair fold={fold} active_pair={keep_pair.sum()} raw_rho={spearmanr(y[holdout_mask], raw_prediction).statistic:.6f} "
            f"residual_rho={spearmanr(y[holdout_mask], residual_prediction).statistic:.6f}",
            flush=True,
        )

    output = parent.copy()
    for name, prediction in predictions.items():
        output[name] = prediction
    atomic_csv(output, RESULTS / "oof_sequence_pair_screen_predictions.csv")
    columns = ["guide_te_bagged", "guide_residual_bagged", *predictions.keys()]
    metrics = metric_rows(output, columns, "position_pair_identity_screen")
    atomic_csv(metrics, RESULTS / "oof_sequence_pair_screen_metrics.csv")
    p = metrics.loc[metrics["fold"].eq("pooled")].set_index("experiment")
    best = max(predictions, key=lambda name: float(p.loc[name, "spearman"]))
    delta = float(p.loc[best, "spearman"] - p.loc["guide_residual_bagged", "spearman"])
    decision = {
        "pair_feature_definition": "25 positions x 36 fixed guide-target base-pair categories",
        "total_pair_columns": len(pair_x.columns),
        "active_pair_columns_per_fold": active_pair_counts,
        "best_pair_architecture": best,
        "best_spearman": float(p.loc[best, "spearman"]),
        "best_rmse": float(p.loc[best, "rmse"]),
        "delta_spearman_vs_guide_residual_bagged": delta,
        "run_five_seed_confirmation": bool(delta >= 0.002),
        "reached_0_8_oof": bool(p.loc[best, "spearman"] >= 0.8),
    }
    atomic_json(decision, RESULTS / "sequence_pair_screen_decision.json")
    print(metrics.loc[metrics["fold"].eq("pooled"), ["experiment", "spearman", "rmse", "mae", "r2"]].to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    return best


def run_confirm() -> None:
    decision = json.loads((RESULTS / "sequence_pair_screen_decision.json").read_text(encoding="utf-8"))
    if not decision["run_five_seed_confirmation"]:
        payload = {"skipped": True, "reason": "Pair architecture did not improve by at least 0.002"}
        atomic_json(payload, RESULTS / "sequence_pair_confirmation_decision.json")
        print(json.dumps(payload, indent=2), flush=True)
        return
    best = decision["best_pair_architecture"]
    residual_mode = best == "pair_guide_residual_xgb_seed42"
    print(f"Running five-seed pair architecture confirmation: {best}", flush=True)
    train, _, features = load_data()
    train = train.reset_index(drop=True)
    parent = load_parent(train)
    screen = pd.read_csv(RESULTS / "oof_sequence_pair_screen_predictions.csv")
    base_x = raw_features(train, features)
    pair_x = pair_onehot(train)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    predictions = {seed: np.full(len(train), np.nan) for seed in SEEDS}
    predictions[BASE_SEED] = screen[best].to_numpy(float)

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        holdout_mask = folds == fold
        outer_train = train.loc[train_mask]
        holdout = train.loc[holdout_mask]
        prepared = prepare(base_x.loc[train_mask], base_x.loc[holdout_mask])
        train_te, holdout_te = make_guide_target_encoding(outer_train, holdout)
        pair_train = pair_x.loc[train_mask]
        pair_holdout = pair_x.loc[holdout_mask]
        keep_pair = pair_train.nunique(dropna=False).gt(1)
        augmented_train = pd.concat([prepared.train, train_te, pair_train.loc[:, keep_pair]], axis=1)
        augmented_holdout = pd.concat([prepared.validation, holdout_te, pair_holdout.loc[:, keep_pair]], axis=1)
        target = y[train_mask]
        if residual_mode:
            target = target - train_te["te_guide_all"].to_numpy(float)
        for seed in SEEDS:
            if seed == BASE_SEED:
                continue
            model = make_xgb(seed)
            model.fit(augmented_train, target)
            pred = model.predict(augmented_holdout)
            if residual_mode:
                pred = holdout_te["te_guide_all"].to_numpy(float) + pred
            predictions[seed][holdout_mask] = pred
            print(f"pair confirm fold={fold} seed={seed} rho={spearmanr(y[holdout_mask], pred).statistic:.6f}", flush=True)

    output = parent.copy()
    for seed in SEEDS:
        output[f"pair_seed_{seed}"] = predictions[seed]
    output["pair_bagged"] = np.mean(np.column_stack([predictions[seed] for seed in SEEDS]), axis=1)
    atomic_csv(output, RESULTS / "oof_sequence_pair_confirmation_predictions.csv")
    columns = ["guide_residual_bagged"] + [f"pair_seed_{seed}" for seed in SEEDS] + ["pair_bagged"]
    metrics = metric_rows(output, columns, "position_pair_identity_five_seed")
    atomic_csv(metrics, RESULTS / "oof_sequence_pair_confirmation_metrics.csv")
    p = metrics.loc[metrics["fold"].eq("pooled")].set_index("experiment")
    payload = {
        "skipped": False,
        "selected_architecture": best,
        "pair_bagged_spearman": float(p.loc["pair_bagged", "spearman"]),
        "pair_bagged_rmse": float(p.loc["pair_bagged", "rmse"]),
        "delta_spearman_vs_guide_residual_bagged": float(
            p.loc["pair_bagged", "spearman"] - p.loc["guide_residual_bagged", "spearman"]
        ),
        "reached_0_8_oof": bool(p.loc["pair_bagged", "spearman"] >= 0.8),
    }
    atomic_json(payload, RESULTS / "sequence_pair_confirmation_decision.json")
    print(metrics.loc[metrics["fold"].eq("pooled"), ["experiment", "spearman", "rmse", "mae", "r2"]].to_string(index=False), flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def write_manifest() -> None:
    atomic_json(
        {
            "objective": "position-specific guide-target pair module for Spearman >= 0.8",
            "length": 25,
            "alphabet": list(ALPHABET),
            "pair_categories_per_position": len(PAIR_CATEGORIES),
            "total_pair_columns": 25 * len(PAIR_CATEGORIES),
            "category_universe": "fixed a priori; not learned from validation",
            "selection": "frozen training OOF only",
        },
        MANIFESTS / "sequence_pair_architecture_manifest.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run position-specific sequence pair architecture")
    parser.add_argument("--stage", choices=("screen", "confirm", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_manifest()
    if args.stage in ("screen", "all"):
        run_screen()
    if args.stage in ("confirm", "all"):
        run_confirm()


if __name__ == "__main__":
    main()
