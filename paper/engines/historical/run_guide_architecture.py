from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "last_week_experiments" / "scripts"))

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


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
MANIFESTS = ROOT / "manifests"
BASE_RESULTS = PROJECT / "last_week_experiments" / "results"
GUIDE = "crRNA_sequence"
SMOOTHING = 0.5


def ensure_dirs() -> None:
    for path in (RESULTS, MODELS, MANIFESTS):
        path.mkdir(parents=True, exist_ok=True)


def category_columns(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    diffcat = pd.Series(
        np.where(
            frame["aligned_difference_count"].eq(0),
            "exact",
            np.where(frame["aligned_difference_count"].eq(1), "one", "two_plus"),
        ),
        index=frame.index,
    )
    has_gap = frame["gap_count_target"].fillna(0).add(frame["gap_count_guide"].fillna(0)).gt(0)
    gapcat = pd.Series(np.where(has_gap, "gap", "no_gap"), index=frame.index)
    return diffcat, gapcat


def map_single_stats(
    keys: pd.Series,
    stats: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    sums = keys.map(stats["sum"]).fillna(0).to_numpy(float)
    counts = keys.map(stats["count"]).fillna(0).to_numpy(float)
    return sums, counts


def map_multi_stats(
    first: pd.Series,
    second: pd.Series,
    stats: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    index = pd.MultiIndex.from_arrays([first.to_numpy(), second.to_numpy()])
    sums = stats["sum"].reindex(index).fillna(0).to_numpy(float)
    counts = stats["count"].reindex(index).fillna(0).to_numpy(float)
    return sums, counts


def make_guide_target_encoding(
    outer_train: pd.DataFrame,
    holdout: pd.DataFrame,
    smoothing: float = SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create leakage-safe LOO train features and outer-train lookup holdout features."""
    train = outer_train.reset_index(drop=True).copy()
    test = holdout.reset_index(drop=True).copy()
    y = train[LABEL].to_numpy(float)
    n = len(train)
    global_sum = float(np.sum(y))
    global_mean = float(np.mean(y))
    global_loo = (global_sum - y) / (n - 1)
    train_guide = train[GUIDE].astype(str)
    test_guide = test[GUIDE].astype(str)
    train_diffcat, train_gapcat = category_columns(train)
    test_diffcat, test_gapcat = category_columns(test)

    all_stats = train.groupby(GUIDE)[LABEL].agg(["sum", "count"])
    train_sum, train_count = map_single_stats(train_guide, all_stats)
    test_sum, test_count = map_single_stats(test_guide, all_stats)
    train_count_excl = train_count - 1
    guide_all_train = (train_sum - y + smoothing * global_loo) / (train_count_excl + smoothing)
    guide_all_test = (test_sum + smoothing * global_mean) / (test_count + smoothing)

    train_features: dict[str, np.ndarray] = {
        "te_guide_all": guide_all_train,
        "te_guide_all_log_count": np.log1p(train_count_excl),
        "te_guide_seen": (train_count_excl > 0).astype(float),
    }
    test_features: dict[str, np.ndarray] = {
        "te_guide_all": guide_all_test,
        "te_guide_all_log_count": np.log1p(test_count),
        "te_guide_seen": (test_count > 0).astype(float),
    }

    def reference_feature(name: str, member_mask: np.ndarray) -> None:
        stats = train.loc[member_mask].groupby(GUIDE)[LABEL].agg(["sum", "count"])
        tr_sum, tr_count = map_single_stats(train_guide, stats)
        te_sum, te_count = map_single_stats(test_guide, stats)
        member = member_mask.astype(float)
        count_excl = tr_count - member
        train_features[f"te_guide_{name}_ref"] = (
            tr_sum - member * y + smoothing * guide_all_train
        ) / (count_excl + smoothing)
        train_features[f"te_guide_{name}_log_count"] = np.log1p(count_excl)
        test_features[f"te_guide_{name}_ref"] = (
            te_sum + smoothing * guide_all_test
        ) / (te_count + smoothing)
        test_features[f"te_guide_{name}_log_count"] = np.log1p(te_count)

    exact_mask = train["aligned_difference_count"].eq(0).to_numpy()
    reference_feature("exact", exact_mask)
    reference_feature("nonexact", ~exact_mask)

    def conditional_feature(name: str, train_category: pd.Series, test_category: pd.Series) -> None:
        temp = train.assign(_category=train_category.to_numpy())
        stats = temp.groupby([GUIDE, "_category"])[LABEL].agg(["sum", "count"])
        tr_sum, tr_count = map_multi_stats(train_guide, train_category, stats)
        te_sum, te_count = map_multi_stats(test_guide, test_category, stats)
        count_excl = tr_count - 1
        train_features[f"te_guide_{name}"] = (
            tr_sum - y + smoothing * guide_all_train
        ) / (count_excl + smoothing)
        train_features[f"te_guide_{name}_log_count"] = np.log1p(count_excl)
        test_features[f"te_guide_{name}"] = (
            te_sum + smoothing * guide_all_test
        ) / (te_count + smoothing)
        test_features[f"te_guide_{name}_log_count"] = np.log1p(te_count)

    conditional_feature("diffcat", train_diffcat, test_diffcat)
    conditional_feature("gapcat", train_gapcat, test_gapcat)

    train_te = pd.DataFrame(train_features, index=outer_train.index)
    test_te = pd.DataFrame(test_features, index=holdout.index)
    if train_te.isna().any().any() or test_te.isna().any().any():
        raise RuntimeError("Target encoding produced missing values")
    return train_te, test_te


def metric_dict(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    high = y >= np.quantile(y, 0.90)
    return {
        "spearman": float(spearmanr(y, prediction).statistic),
        "pearson": float(pearsonr(y, prediction).statistic),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
        "top_decile_bias": float(np.mean(prediction[high] - y[high])),
    }


def metric_rows(frame: pd.DataFrame, columns: list[str], stage: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        rows.append({"stage": stage, "experiment": column, "fold": "pooled", "n": len(frame), **metric_dict(frame[LABEL], frame[column])})
        for fold, subset in frame.groupby("fold", sort=True):
            rows.append({"stage": stage, "experiment": column, "fold": int(fold), "n": len(subset), **metric_dict(subset[LABEL], subset[column])})
    return pd.DataFrame(rows)


def load_base(train: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASE_RESULTS / "oof_seed_confirmation_predictions.csv")
    result = train[["record_id", FOLD, LABEL]].merge(
        base[["record_id", "fold", LABEL, "baseline_seed_42", "baseline_bagged"]],
        on="record_id",
        suffixes=("_frozen", "_base"),
        validate="one_to_one",
    )
    if not np.array_equal(result[f"{LABEL}_frozen"], result[f"{LABEL}_base"]):
        raise RuntimeError("OOF label alignment failed")
    if not np.array_equal(result[FOLD].to_numpy(int), result["fold"].to_numpy(int)):
        raise RuntimeError("OOF fold alignment failed")
    return result.rename(columns={f"{LABEL}_frozen": LABEL})[
        ["record_id", "fold", LABEL, "baseline_seed_42", "baseline_bagged"]
    ]


def run_screen() -> bool:
    print("Running guide-prior and guide-target-encoding architecture screen...", flush=True)
    train, _, features = load_data()
    train = train.reset_index(drop=True)
    base = load_base(train)
    x = raw_features(train, features)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    guide_prior = np.full(len(train), np.nan)
    guide_te_xgb = np.full(len(train), np.nan)
    feature_names: list[str] | None = None
    started = time.time()

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        holdout_mask = folds == fold
        outer_train = train.loc[train_mask]
        holdout = train.loc[holdout_mask]
        prepared = prepare(x.loc[train_mask], x.loc[holdout_mask])
        train_te, holdout_te = make_guide_target_encoding(outer_train, holdout)
        augmented_train = pd.concat([prepared.train, train_te], axis=1)
        augmented_holdout = pd.concat([prepared.validation, holdout_te], axis=1)
        if augmented_train.isna().any().any() or augmented_holdout.isna().any().any():
            raise RuntimeError(f"Augmented features contain missing values in fold {fold}")
        feature_names = list(augmented_train.columns)
        guide_prior[holdout_mask] = holdout_te["te_guide_all"].to_numpy(float)
        model = make_xgb(BASE_SEED)
        fold_started = time.time()
        model.fit(augmented_train, y[train_mask])
        guide_te_xgb[holdout_mask] = model.predict(augmented_holdout)
        print(
            f"guide-TE fold={fold} features={len(feature_names)} "
            f"rho={spearmanr(y[holdout_mask], guide_te_xgb[holdout_mask]).statistic:.6f} "
            f"elapsed={time.time() - fold_started:.1f}s",
            flush=True,
        )

    output = base.copy()
    output["guide_prior"] = guide_prior
    output["guide_te_xgb_seed42"] = guide_te_xgb

    best = None
    grid_rows: list[dict[str, float]] = []
    for alpha in np.linspace(0, 0.60, 121):
        prediction = (1 - alpha) * output["baseline_bagged"].to_numpy(float) + alpha * guide_prior
        values = metric_dict(y, prediction)
        grid_rows.append({"guide_prior_weight": float(alpha), **values})
        key = (values["spearman"], -values["rmse"])
        if best is None or key > best[0]:
            best = (key, float(alpha), prediction.copy(), values)
    assert best is not None
    output["xgb_guide_prior_blend"] = best[2]
    atomic_csv(output, RESULTS / "oof_guide_screen_predictions.csv")
    atomic_csv(pd.DataFrame(grid_rows), RESULTS / "guide_prior_weight_grid.csv")
    metrics = metric_rows(
        output,
        ["baseline_seed_42", "baseline_bagged", "guide_prior", "xgb_guide_prior_blend", "guide_te_xgb_seed42"],
        "guide_architecture_screen",
    )
    atomic_csv(metrics, RESULTS / "oof_guide_screen_metrics.csv")
    p = metrics.loc[metrics["fold"].eq("pooled")].set_index("experiment")
    delta_te = float(p.loc["guide_te_xgb_seed42", "spearman"] - p.loc["baseline_seed_42", "spearman"])
    te_pass = bool(delta_te >= 0.005)
    decision = {
        "smoothing": SMOOTHING,
        "target_encoded_feature_count": 11,
        "augmented_active_feature_count": len(feature_names or []),
        "best_guide_prior_weight": best[1],
        "guide_prior_blend_metrics": best[3],
        "guide_te_xgb_delta_spearman_vs_seed42": delta_te,
        "guide_te_xgb_screen_pass": te_pass,
        "next_action": "run five-seed guide-TE XGBoost confirmation" if te_pass else "retain prior blend and move to categorical/deep module",
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(decision, RESULTS / "guide_screen_decision.json")
    print(metrics.loc[metrics["fold"].eq("pooled"), ["experiment", "spearman", "rmse", "mae", "r2"]].to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    return te_pass


def run_confirm(force: bool = False) -> None:
    decision = json.loads((RESULTS / "guide_screen_decision.json").read_text(encoding="utf-8"))
    if not decision["guide_te_xgb_screen_pass"] and not force:
        payload = {"skipped": True, "reason": "Guide-TE XGBoost did not improve seed42 OOF Spearman by at least 0.005"}
        atomic_json(payload, RESULTS / "guide_confirmation_decision.json")
        print(json.dumps(payload, indent=2), flush=True)
        return

    print("Running five-seed guide-target-encoding XGBoost confirmation...", flush=True)
    train, _, features = load_data()
    train = train.reset_index(drop=True)
    base = load_base(train)
    screen = pd.read_csv(RESULTS / "oof_guide_screen_predictions.csv")
    x = raw_features(train, features)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    predictions = {seed: np.full(len(train), np.nan) for seed in SEEDS}
    predictions[BASE_SEED] = screen["guide_te_xgb_seed42"].to_numpy(float)

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        holdout_mask = folds == fold
        outer_train = train.loc[train_mask]
        holdout = train.loc[holdout_mask]
        prepared = prepare(x.loc[train_mask], x.loc[holdout_mask])
        train_te, holdout_te = make_guide_target_encoding(outer_train, holdout)
        augmented_train = pd.concat([prepared.train, train_te], axis=1)
        augmented_holdout = pd.concat([prepared.validation, holdout_te], axis=1)
        for seed in SEEDS:
            if seed == BASE_SEED:
                continue
            model = make_xgb(seed)
            model.fit(augmented_train, y[train_mask])
            predictions[seed][holdout_mask] = model.predict(augmented_holdout)
            print(
                f"guide-TE confirm fold={fold} seed={seed} "
                f"rho={spearmanr(y[holdout_mask], predictions[seed][holdout_mask]).statistic:.6f}",
                flush=True,
            )

    output = base.copy()
    for seed in SEEDS:
        output[f"guide_te_seed_{seed}"] = predictions[seed]
    output["guide_te_bagged"] = np.mean(np.column_stack([predictions[seed] for seed in SEEDS]), axis=1)
    atomic_csv(output, RESULTS / "oof_guide_confirmation_predictions.csv")
    columns = ["baseline_bagged"] + [f"guide_te_seed_{seed}" for seed in SEEDS] + ["guide_te_bagged"]
    metrics = metric_rows(output, columns, "guide_te_five_seed_confirmation")
    atomic_csv(metrics, RESULTS / "oof_guide_confirmation_metrics.csv")
    p = metrics.loc[metrics["fold"].eq("pooled")].set_index("experiment")
    delta = float(p.loc["guide_te_bagged", "spearman"] - p.loc["baseline_bagged", "spearman"])
    decision = {
        "skipped": False,
        "delta_spearman_vs_baseline_bagged": delta,
        "guide_te_bagged_spearman": float(p.loc["guide_te_bagged", "spearman"]),
        "guide_te_bagged_rmse": float(p.loc["guide_te_bagged", "rmse"]),
        "reached_0_8_oof": bool(p.loc["guide_te_bagged", "spearman"] >= 0.8),
    }
    atomic_json(decision, RESULTS / "guide_confirmation_decision.json")
    print(metrics.loc[metrics["fold"].eq("pooled"), ["experiment", "spearman", "rmse", "mae", "r2"]].to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


def run_residual_screen() -> str:
    print("Running hierarchical guide-baseline + target-residual screen...", flush=True)
    train, _, features = load_data()
    train = train.reset_index(drop=True)
    base = load_base(train)
    x = raw_features(train, features)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    residual_plain = np.full(len(train), np.nan)
    residual_augmented = np.full(len(train), np.nan)

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        holdout_mask = folds == fold
        outer_train = train.loc[train_mask]
        holdout = train.loc[holdout_mask]
        prepared = prepare(x.loc[train_mask], x.loc[holdout_mask])
        train_te, holdout_te = make_guide_target_encoding(outer_train, holdout)
        residual_target = y[train_mask] - train_te["te_guide_all"].to_numpy(float)
        for name, augmented in (("plain", False), ("augmented", True)):
            if augmented:
                model_train = pd.concat([prepared.train, train_te], axis=1)
                model_holdout = pd.concat([prepared.validation, holdout_te], axis=1)
            else:
                model_train = prepared.train
                model_holdout = prepared.validation
            model = make_xgb(BASE_SEED)
            model.fit(model_train, residual_target)
            prediction = holdout_te["te_guide_all"].to_numpy(float) + model.predict(model_holdout)
            if augmented:
                residual_augmented[holdout_mask] = prediction
            else:
                residual_plain[holdout_mask] = prediction
            print(
                f"residual fold={fold} architecture={name} "
                f"rho={spearmanr(y[holdout_mask], prediction).statistic:.6f}",
                flush=True,
            )

    guide_confirm = pd.read_csv(RESULTS / "oof_guide_confirmation_predictions.csv")
    output = base.merge(
        guide_confirm[["record_id", "guide_te_bagged", "guide_te_seed_42"]],
        on="record_id",
        validate="one_to_one",
    )
    output["guide_residual_xgb_seed42"] = residual_plain
    output["guide_residual_augmented_xgb_seed42"] = residual_augmented
    atomic_csv(output, RESULTS / "oof_guide_residual_screen_predictions.csv")
    columns = [
        "baseline_bagged",
        "guide_te_bagged",
        "guide_residual_xgb_seed42",
        "guide_residual_augmented_xgb_seed42",
    ]
    metrics = metric_rows(output, columns, "hierarchical_guide_residual_screen")
    atomic_csv(metrics, RESULTS / "oof_guide_residual_screen_metrics.csv")
    p = metrics.loc[metrics["fold"].eq("pooled")].set_index("experiment")
    best = max(
        ("guide_residual_xgb_seed42", "guide_residual_augmented_xgb_seed42"),
        key=lambda name: float(p.loc[name, "spearman"]),
    )
    delta = float(p.loc[best, "spearman"] - p.loc["guide_te_bagged", "spearman"])
    decision = {
        "decomposition": "cross-fitted guide baseline + XGBoost-predicted target residual",
        "best_residual_architecture": best,
        "best_spearman": float(p.loc[best, "spearman"]),
        "best_rmse": float(p.loc[best, "rmse"]),
        "delta_spearman_vs_five_seed_guide_te": delta,
        "run_five_seed_confirmation": bool(delta >= 0.002),
        "reached_0_8_oof": bool(p.loc[best, "spearman"] >= 0.8),
    }
    atomic_json(decision, RESULTS / "guide_residual_screen_decision.json")
    print(metrics.loc[metrics["fold"].eq("pooled"), ["experiment", "spearman", "rmse", "mae", "r2"]].to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    return best


def run_residual_confirm() -> None:
    decision = json.loads((RESULTS / "guide_residual_screen_decision.json").read_text(encoding="utf-8"))
    if not decision["run_five_seed_confirmation"]:
        payload = {"skipped": True, "reason": "Residual architecture did not exceed Guide-TE bagging by 0.002"}
        atomic_json(payload, RESULTS / "guide_residual_confirmation_decision.json")
        print(json.dumps(payload, indent=2), flush=True)
        return
    print("Running five-seed augmented hierarchical residual confirmation...", flush=True)
    train, _, features = load_data()
    train = train.reset_index(drop=True)
    base = load_base(train)
    screen = pd.read_csv(RESULTS / "oof_guide_residual_screen_predictions.csv")
    x = raw_features(train, features)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    predictions = {seed: np.full(len(train), np.nan) for seed in SEEDS}
    predictions[BASE_SEED] = screen["guide_residual_augmented_xgb_seed42"].to_numpy(float)

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        holdout_mask = folds == fold
        outer_train = train.loc[train_mask]
        holdout = train.loc[holdout_mask]
        prepared = prepare(x.loc[train_mask], x.loc[holdout_mask])
        train_te, holdout_te = make_guide_target_encoding(outer_train, holdout)
        residual_target = y[train_mask] - train_te["te_guide_all"].to_numpy(float)
        augmented_train = pd.concat([prepared.train, train_te], axis=1)
        augmented_holdout = pd.concat([prepared.validation, holdout_te], axis=1)
        for seed in SEEDS:
            if seed == BASE_SEED:
                continue
            model = make_xgb(seed)
            model.fit(augmented_train, residual_target)
            predictions[seed][holdout_mask] = (
                holdout_te["te_guide_all"].to_numpy(float) + model.predict(augmented_holdout)
            )
            print(
                f"residual confirm fold={fold} seed={seed} "
                f"rho={spearmanr(y[holdout_mask], predictions[seed][holdout_mask]).statistic:.6f}",
                flush=True,
            )

    guide_confirm = pd.read_csv(RESULTS / "oof_guide_confirmation_predictions.csv")
    output = base.merge(
        guide_confirm[["record_id", "guide_te_bagged"]], on="record_id", validate="one_to_one"
    )
    for seed in SEEDS:
        output[f"guide_residual_seed_{seed}"] = predictions[seed]
    output["guide_residual_bagged"] = np.mean(
        np.column_stack([predictions[seed] for seed in SEEDS]), axis=1
    )
    atomic_csv(output, RESULTS / "oof_guide_residual_confirmation_predictions.csv")
    columns = ["guide_te_bagged"] + [f"guide_residual_seed_{seed}" for seed in SEEDS] + ["guide_residual_bagged"]
    metrics = metric_rows(output, columns, "guide_residual_five_seed_confirmation")
    atomic_csv(metrics, RESULTS / "oof_guide_residual_confirmation_metrics.csv")
    p = metrics.loc[metrics["fold"].eq("pooled")].set_index("experiment")
    payload = {
        "skipped": False,
        "guide_residual_bagged_spearman": float(p.loc["guide_residual_bagged", "spearman"]),
        "guide_residual_bagged_rmse": float(p.loc["guide_residual_bagged", "rmse"]),
        "delta_spearman_vs_guide_te_bagged": float(
            p.loc["guide_residual_bagged", "spearman"] - p.loc["guide_te_bagged", "spearman"]
        ),
        "reached_0_8_oof": bool(p.loc["guide_residual_bagged", "spearman"] >= 0.8),
    }
    atomic_json(payload, RESULTS / "guide_residual_confirmation_decision.json")
    print(metrics.loc[metrics["fold"].eq("pooled"), ["experiment", "spearman", "rmse", "mae", "r2"]].to_string(index=False), flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def write_manifest() -> None:
    payload = {
        "objective": "Spearman >= 0.8 with overall metric improvement",
        "protocol": "frozen five target-grouped training OOF folds",
        "guide_prior_smoothing": SMOOTHING,
        "guide_target_encoding": {
            "training_rows": "leave-one-out within each outer-training fold",
            "holdout_rows": "lookup from the corresponding outer-training partition only",
            "unseen_guide_fallback": "outer-training global mean",
            "features": [
                "guide all mean/count/seen",
                "guide exact reference/count",
                "guide non-exact reference/count",
                "guide difference-category mean/count",
                "guide gap-category mean/count",
            ],
        },
        "selection": "training OOF only",
        "validation_status": "not evaluated by this script",
    }
    atomic_json(payload, MANIFESTS / "guide_architecture_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guide-level collaborative architecture experiments")
    parser.add_argument(
        "--stage",
        choices=("screen", "confirm", "residual", "residual_confirm", "all"),
        default="all",
    )
    parser.add_argument("--force-confirm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    write_manifest()
    if args.stage in ("screen", "all"):
        run_screen()
    if args.stage in ("confirm", "all"):
        run_confirm(force=args.force_confirm)
    if args.stage in ("residual", "all"):
        run_residual_screen()
    if args.stage in ("residual_confirm", "all"):
        run_residual_confirm()


if __name__ == "__main__":
    main()
