from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, pearsonr, rankdata, spearmanr
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    PROJECT_ROOT
    / "final_repo_audit"
    / "rigorous"
    / "data_package"
    / "Cas12a_Data"
    / "Unified_Horizontal_V2-2_P1_V4"
)
DATA_FILE = PACKAGE_ROOT / "data" / "EasyDesign_2024_V2-2_core_context_feature_table.csv"
FEATURE_FILE = PACKAGE_ROOT / "metadata" / "EasyDesign_2024_V2-2_feature_manifest.csv"
FOLD_FILE = PACKAGE_ROOT / "metadata" / "EasyDesign_2024_V2-2_frozen_target_grouped_folds.csv"
REFERENCE_ROOT = (
    PROJECT_ROOT
    / "final_repo_audit"
    / "final_four_package"
    / "four_experiment_delivery"
    / "results"
)
REFERENCE_OOF_FILE = REFERENCE_ROOT / "oof_predictions.csv"
REFERENCE_VALIDATION_FILE = REFERENCE_ROOT / "fixed_validation_predictions.csv"

RESULTS = RUN_ROOT / "results"
MODELS = RUN_ROOT / "models"
MANIFESTS = RUN_ROOT / "manifests"

LABEL = "label_normalized"
FOLD = "context_cv_fold"
EXACT_COLUMN = "aligned_difference_count"
BASE_SEED = 42
SEEDS = (7, 21, 42, 84, 2024)
BOOTSTRAP_SEED = 20260812
N_BOOTSTRAP = 2000

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 1100,
    "learning_rate": 0.02,
    "max_depth": 7,
    "min_child_weight": 2,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.05,
    "reg_lambda": 2,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "n_jobs": 1,
}

EXPERIMENTS: dict[str, dict[str, str]] = {
    "baseline": {
        "parent": "final_xgboost",
        "unique_variable": "none; exact reproducibility control",
    },
    "rank_target": {
        "parent": "baseline",
        "unique_variable": "training target representation: raw activity -> empirical rank-normal score",
    },
    "two_expert": {
        "parent": "baseline",
        "unique_variable": "routing: one global XGBoost -> exact/non-exact XGBoost experts",
    },
    "seed_bagging": {
        "parent": "selected OOF pipeline",
        "unique_variable": "prediction aggregation: one seed -> mean of five fixed seeds",
    },
    "affine_calibration": {
        "parent": "locked OOF pipeline",
        "unique_variable": "post-hoc output map: identity -> OOF-fitted positive-slope affine map",
    },
}


@dataclass
class Prepared:
    train: pd.DataFrame
    validation: pd.DataFrame
    median: pd.Series
    active: list[str]


@dataclass
class RankMap:
    y_sorted: np.ndarray
    z_grid: np.ndarray

    @classmethod
    def fit(cls, y: np.ndarray) -> tuple[np.ndarray, "RankMap"]:
        y = np.asarray(y, dtype=float)
        n = len(y)
        q_train = (rankdata(y, method="average") - 0.5) / n
        z_train = norm.ppf(np.clip(q_train, 1e-6, 1 - 1e-6))
        q_grid = (np.arange(n, dtype=float) + 0.5) / n
        return z_train, cls(y_sorted=np.sort(y), z_grid=norm.ppf(q_grid))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(z, dtype=float),
            self.z_grid,
            self.y_sorted,
            left=float(self.y_sorted[0]),
            right=float(self.y_sorted[-1]),
        )


def ensure_dirs() -> None:
    for path in (RESULTS, MODELS, MANIFESTS):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(payload: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data = pd.read_csv(DATA_FILE, low_memory=False)
    features = pd.read_csv(FEATURE_FILE)["feature_name"].tolist()
    folds = pd.read_csv(FOLD_FILE)
    train = data.loc[data["baseline_split"].eq("baseline_train")].merge(
        folds[["record_id", "target_group_id", FOLD]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    validation = data.loc[data["baseline_split"].eq("baseline_validation")].copy()
    if (len(train), len(validation), len(features)) != (8417, 2217, 188):
        raise RuntimeError(
            f"Frozen dimensions changed: train={len(train)}, validation={len(validation)}, features={len(features)}"
        )
    if train[FOLD].isna().any() or train["record_id"].duplicated().any():
        raise RuntimeError("Fold mapping is incomplete or record_id is duplicated")
    if set(train["record_id"]) & set(validation["record_id"]):
        raise RuntimeError("Train/validation record overlap detected")
    if train.groupby("target_group_id")[FOLD].nunique().max() != 1:
        raise RuntimeError("A target group crosses OOF folds")
    if sha256(DATA_FILE) != "39cda8368c216784507ac002df687b28a4f9cc6f81e2b0e84043e45eddb4c1c0":
        raise RuntimeError("Frozen data SHA-256 mismatch")
    return train.reset_index(drop=True), validation.reset_index(drop=True), features


def raw_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].apply(pd.to_numeric, errors="coerce")


def prepare(train_x: pd.DataFrame, test_x: pd.DataFrame) -> Prepared:
    median = train_x.median()
    filled_train = train_x.fillna(median).fillna(0)
    filled_test = test_x.fillna(median).fillna(0)
    keep = filled_train.nunique(dropna=False).gt(1)
    active = list(filled_train.columns[keep])
    return Prepared(
        train=filled_train.loc[:, active],
        validation=filled_test.loc[:, active],
        median=median.loc[active],
        active=active,
    )


def make_xgb(seed: int) -> XGBRegressor:
    return XGBRegressor(random_state=seed, **XGB_PARAMS)


def fit_predict(
    experiment: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    exact_train: np.ndarray,
    exact_test: np.ndarray,
    seed: int,
    return_bundle: bool = False,
) -> tuple[np.ndarray, Any | None]:
    if experiment == "baseline":
        model = make_xgb(seed)
        model.fit(x_train, y_train)
        return model.predict(x_test), model if return_bundle else None

    if experiment == "rank_target":
        z_train, rank_map = RankMap.fit(y_train)
        model = make_xgb(seed)
        model.fit(x_train, z_train)
        prediction = rank_map.inverse(model.predict(x_test))
        bundle = {"model": model, "rank_map": rank_map} if return_bundle else None
        return prediction, bundle

    if experiment == "two_expert":
        prediction = np.full(len(x_test), np.nan, dtype=float)
        models: dict[str, XGBRegressor] = {}
        for name, train_mask, test_mask in (
            ("exact", exact_train, exact_test),
            ("non_exact", ~exact_train, ~exact_test),
        ):
            if train_mask.sum() < 30 or test_mask.sum() == 0:
                raise RuntimeError(
                    f"Insufficient rows for {name}: train={train_mask.sum()}, test={test_mask.sum()}"
                )
            model = make_xgb(seed)
            model.fit(x_train.loc[train_mask], y_train[train_mask])
            prediction[test_mask] = model.predict(x_test.loc[test_mask])
            models[name] = model
        if np.isnan(prediction).any():
            raise RuntimeError("Two-expert routing left missing predictions")
        return prediction, models if return_bundle else None

    raise ValueError(f"Unknown experiment: {experiment}")


def metric_dict(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    high = y >= np.quantile(y, 0.90)
    low = y <= np.quantile(y, 0.10)
    return {
        "spearman": float(spearmanr(y, prediction).statistic),
        "pearson": float(pearsonr(y, prediction).statistic),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
        "top_decile_bias_pred_minus_true": float(np.mean(prediction[high] - y[high])),
        "bottom_decile_bias_pred_minus_true": float(np.mean(prediction[low] - y[low])),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }


def metrics_by_fold(
    frame: pd.DataFrame,
    prediction_columns: Iterable[str],
    stage: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in prediction_columns:
        for fold_name, subset in [("pooled", frame)]:
            rows.append(
                {
                    "stage": stage,
                    "experiment": column,
                    "fold": fold_name,
                    "n": len(subset),
                    **metric_dict(subset[LABEL].to_numpy(float), subset[column].to_numpy(float)),
                }
            )
        if "fold" in frame:
            for fold, subset in frame.groupby("fold", sort=True):
                rows.append(
                    {
                        "stage": stage,
                        "experiment": column,
                        "fold": int(fold),
                        "n": len(subset),
                        **metric_dict(subset[LABEL].to_numpy(float), subset[column].to_numpy(float)),
                    }
                )
    return pd.DataFrame(rows)


def print_pooled(metrics: pd.DataFrame) -> None:
    columns = ["experiment", "n", "spearman", "rmse", "mae", "r2"]
    print(metrics.loc[metrics["fold"].eq("pooled"), columns].to_string(index=False), flush=True)


def validate_against_reference(screen: pd.DataFrame) -> dict[str, float]:
    reference = pd.read_csv(REFERENCE_OOF_FILE)[["record_id", "xgboost_oof"]]
    joined = screen[["record_id", "baseline"]].merge(reference, on="record_id", validate="one_to_one")
    delta = joined["baseline"].to_numpy() - joined["xgboost_oof"].to_numpy()
    result = {
        "max_abs_prediction_delta": float(np.max(np.abs(delta))),
        "mean_abs_prediction_delta": float(np.mean(np.abs(delta))),
    }
    # XGBoost CPU predictions can vary at the last bits across builds, but a substantive mismatch is not allowed.
    if result["max_abs_prediction_delta"] > 1e-4:
        raise RuntimeError(f"Baseline predictions do not reproduce the frozen OOF reference: {result}")
    return result


def run_screen() -> str:
    print("Loading frozen data for OOF screen...", flush=True)
    train, _, features = load_data()
    x = raw_features(train, features)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    exact = train[EXACT_COLUMN].to_numpy(float) == 0
    predictions = {name: np.full(len(train), np.nan) for name in ("baseline", "rank_target", "two_expert")}

    started = time.time()
    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        holdout_mask = folds == fold
        prepared = prepare(x.loc[train_mask], x.loc[holdout_mask])
        if fold == 0 and len(prepared.active) != 183:
            print(f"Note: fold 0 active features={len(prepared.active)} (full-train reference is 183)", flush=True)
        for experiment in predictions:
            fold_started = time.time()
            pred, _ = fit_predict(
                experiment=experiment,
                x_train=prepared.train,
                y_train=y[train_mask],
                x_test=prepared.validation,
                exact_train=exact[train_mask],
                exact_test=exact[holdout_mask],
                seed=BASE_SEED,
            )
            predictions[experiment][holdout_mask] = pred
            score = spearmanr(y[holdout_mask], pred).statistic
            print(
                f"screen fold={fold} experiment={experiment} n={holdout_mask.sum()} "
                f"rho={score:.6f} elapsed={time.time() - fold_started:.1f}s",
                flush=True,
            )

    screen = pd.DataFrame(
        {
            "record_id": train["record_id"],
            "fold": folds,
            LABEL: y,
            "is_exact": exact.astype(int),
            **predictions,
        }
    )
    if screen[["baseline", "rank_target", "two_expert"]].isna().any().any():
        raise RuntimeError("OOF screen contains missing predictions")
    atomic_csv(screen, RESULTS / "oof_screen_predictions.csv")
    metrics = metrics_by_fold(screen, predictions.keys(), stage="screen_seed42")
    atomic_csv(metrics, RESULTS / "oof_screen_metrics.csv")
    reference_check = validate_against_reference(screen)

    pooled = metrics.loc[metrics["fold"].eq("pooled")].set_index("experiment")
    baseline_rho = float(pooled.loc["baseline", "spearman"])
    eligible: list[str] = []
    candidate_summary: dict[str, Any] = {}
    for experiment in ("rank_target", "two_expert"):
        delta_rho = float(pooled.loc[experiment, "spearman"] - baseline_rho)
        delta_rmse = float(pooled.loc[experiment, "rmse"] - pooled.loc["baseline", "rmse"])
        fold_merged = metrics.loc[
            metrics["experiment"].isin(["baseline", experiment]) & ~metrics["fold"].eq("pooled"),
            ["experiment", "fold", "spearman"],
        ].pivot(index="fold", columns="experiment", values="spearman")
        positive_folds = int((fold_merged[experiment] > fold_merged["baseline"]).sum())
        candidate_summary[experiment] = {
            "delta_spearman": delta_rho,
            "delta_rmse": delta_rmse,
            "positive_folds": positive_folds,
            "fast_screen_pass": bool(delta_rho >= 0.002 and delta_rmse <= 0.003),
        }
        if candidate_summary[experiment]["fast_screen_pass"]:
            eligible.append(experiment)

    if eligible:
        selected = max(eligible, key=lambda name: float(pooled.loc[name, "spearman"]))
        reason = "highest pooled OOF Spearman among candidates passing the prespecified fast-screen gate"
    else:
        selected = "baseline"
        reason = "neither new module passed ΔSpearman >= 0.002 with RMSE guard; proceed with baseline seed bagging"

    selection = {
        "selected_for_seed_confirmation": selected,
        "reason": reason,
        "fast_screen_gate": {"delta_spearman_min": 0.002, "delta_rmse_max": 0.003},
        "candidates": candidate_summary,
        "baseline_reference_check": reference_check,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(selection, RESULTS / "screen_selection.json")
    print_pooled(metrics)
    print(json.dumps(selection, indent=2), flush=True)
    return selected


def load_screen_selection() -> str:
    path = RESULTS / "screen_selection.json"
    if not path.exists():
        raise FileNotFoundError("Run --stage screen first")
    return json.loads(path.read_text(encoding="utf-8"))["selected_for_seed_confirmation"]


def run_confirm(candidate: str | None = None) -> str:
    candidate = candidate or load_screen_selection()
    if candidate not in ("baseline", "rank_target", "two_expert"):
        raise ValueError(f"Unsupported candidate: {candidate}")
    print(f"Running five-seed OOF confirmation for candidate={candidate}...", flush=True)
    train, _, features = load_data()
    x = raw_features(train, features)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    exact = train[EXACT_COLUMN].to_numpy(float) == 0
    screen = pd.read_csv(RESULTS / "oof_screen_predictions.csv")
    screen = train[["record_id"]].merge(screen, on="record_id", validate="one_to_one")

    prediction_columns: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        if seed == BASE_SEED:
            prediction_columns[f"baseline_seed_{seed}"] = screen["baseline"].to_numpy(float)
            prediction_columns[f"candidate_seed_{seed}"] = screen[candidate].to_numpy(float)
        else:
            prediction_columns[f"baseline_seed_{seed}"] = np.full(len(train), np.nan)
            prediction_columns[f"candidate_seed_{seed}"] = np.full(len(train), np.nan)

    started = time.time()
    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        holdout_mask = folds == fold
        prepared = prepare(x.loc[train_mask], x.loc[holdout_mask])
        for seed in SEEDS:
            if seed == BASE_SEED:
                continue
            jobs = [("baseline", "baseline")]
            if candidate != "baseline":
                jobs.append((candidate, "candidate"))
            for experiment, prefix in jobs:
                fold_started = time.time()
                pred, _ = fit_predict(
                    experiment=experiment,
                    x_train=prepared.train,
                    y_train=y[train_mask],
                    x_test=prepared.validation,
                    exact_train=exact[train_mask],
                    exact_test=exact[holdout_mask],
                    seed=seed,
                )
                prediction_columns[f"{prefix}_seed_{seed}"][holdout_mask] = pred
                print(
                    f"confirm fold={fold} seed={seed} experiment={experiment} "
                    f"rho={spearmanr(y[holdout_mask], pred).statistic:.6f} "
                    f"elapsed={time.time() - fold_started:.1f}s",
                    flush=True,
                )
                if candidate == "baseline":
                    prediction_columns[f"candidate_seed_{seed}"][holdout_mask] = pred

    baseline_seed_cols = [f"baseline_seed_{seed}" for seed in SEEDS]
    candidate_seed_cols = [f"candidate_seed_{seed}" for seed in SEEDS]
    prediction_columns["baseline_bagged"] = np.mean(
        np.column_stack([prediction_columns[column] for column in baseline_seed_cols]), axis=1
    )
    prediction_columns["candidate_bagged"] = np.mean(
        np.column_stack([prediction_columns[column] for column in candidate_seed_cols]), axis=1
    )

    output = pd.DataFrame(
        {
            "record_id": train["record_id"],
            "fold": folds,
            LABEL: y,
            "is_exact": exact.astype(int),
            **prediction_columns,
        }
    )
    if output[list(prediction_columns)].isna().any().any():
        raise RuntimeError("OOF seed confirmation contains missing predictions")
    atomic_csv(output, RESULTS / "oof_seed_confirmation_predictions.csv")
    metrics = metrics_by_fold(output, prediction_columns.keys(), stage=f"confirm_{candidate}")
    atomic_csv(metrics, RESULTS / "oof_seed_confirmation_metrics.csv")

    pooled = metrics.loc[metrics["fold"].eq("pooled")].set_index("experiment")
    seed_deltas: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline_col = f"baseline_seed_{seed}"
        candidate_col = f"candidate_seed_{seed}"
        delta = float(pooled.loc[candidate_col, "spearman"] - pooled.loc[baseline_col, "spearman"])
        seed_deltas.append({"seed": seed, "delta_spearman": delta, "positive": bool(delta > 0)})
    mean_seed_delta = float(np.mean([row["delta_spearman"] for row in seed_deltas]))
    positive_seeds = int(sum(row["positive"] for row in seed_deltas))
    module_formally_promoted = bool(
        candidate != "baseline" and mean_seed_delta >= 0.005 and positive_seeds >= 4
    )

    # Lock the highest OOF option with an RMSE guard. This may be exploratory even when the formal gate fails.
    options = ["baseline_seed_42", "baseline_bagged"]
    if candidate != "baseline":
        options.extend(["candidate_seed_42", "candidate_bagged"])
    baseline_rmse = float(pooled.loc["baseline_seed_42", "rmse"])
    valid_options = [name for name in options if float(pooled.loc[name, "rmse"]) <= baseline_rmse + 0.003]
    locked = max(valid_options, key=lambda name: float(pooled.loc[name, "spearman"]))
    locked_pipeline = "baseline" if locked.startswith("baseline") else candidate
    locked_bagged = locked.endswith("bagged")
    locked_status = "formal" if module_formally_promoted and locked_pipeline == candidate else "exploratory"

    bagging_delta = {
        "baseline_delta_spearman": float(
            pooled.loc["baseline_bagged", "spearman"] - pooled.loc["baseline_seed_42", "spearman"]
        ),
        "baseline_delta_rmse": float(
            pooled.loc["baseline_bagged", "rmse"] - pooled.loc["baseline_seed_42", "rmse"]
        ),
        "candidate_delta_spearman": float(
            pooled.loc["candidate_bagged", "spearman"] - pooled.loc["candidate_seed_42", "spearman"]
        ),
        "candidate_delta_rmse": float(
            pooled.loc["candidate_bagged", "rmse"] - pooled.loc["candidate_seed_42", "rmse"]
        ),
    }
    selection = {
        "screen_candidate": candidate,
        "seed_deltas_vs_baseline": seed_deltas,
        "mean_seed_delta_spearman": mean_seed_delta,
        "positive_seed_count": positive_seeds,
        "formal_module_promotion_gate": {
            "mean_delta_spearman_min": 0.005,
            "positive_seed_count_min": 4,
        },
        "module_formally_promoted": module_formally_promoted,
        "bagging_effect": bagging_delta,
        "locked_oof_column": locked,
        "locked_pipeline": locked_pipeline,
        "locked_bagged": locked_bagged,
        "locked_status": locked_status,
        "selection_note": (
            "Selection uses training OOF only. No fixed-validation candidate prediction or metric "
            "was computed or inspected during screen/confirm."
        ),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(selection, RESULTS / "confirmation_selection.json")
    print_pooled(metrics)
    print(json.dumps(selection, indent=2), flush=True)
    return locked


def fit_affine(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = float(np.cov(prediction, y, ddof=0)[0, 1] / np.var(prediction))
    intercept = float(np.mean(y) - slope * np.mean(prediction))
    if not np.isfinite(slope) or slope <= 0:
        raise RuntimeError(f"Affine calibration slope must be positive, got {slope}")
    return intercept, slope


def cluster_bootstrap(
    frame: pd.DataFrame,
    candidate_columns: list[str],
    reference_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    groups = {key: values.index.to_numpy() for key, values in frame.groupby("target_sequence", sort=False)}
    group_ids = list(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for iteration in range(N_BOOTSTRAP):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        indices = np.concatenate([groups[key] for key in sampled])
        y = frame.loc[indices, LABEL].to_numpy(float)
        reference = frame.loc[indices, reference_column].to_numpy(float)
        reference_spearman = float(spearmanr(y, reference).statistic)
        reference_rmse = float(mean_squared_error(y, reference) ** 0.5)
        for column in candidate_columns:
            prediction = frame.loc[indices, column].to_numpy(float)
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": column,
                    "delta_spearman": float(spearmanr(y, prediction).statistic - reference_spearman),
                    "delta_rmse": float(mean_squared_error(y, prediction) ** 0.5 - reference_rmse),
                }
            )
        if (iteration + 1) % 250 == 0:
            print(f"bootstrap {iteration + 1}/{N_BOOTSTRAP}", flush=True)
    replicates = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "resamples": N_BOOTSTRAP,
        "seed": BOOTSTRAP_SEED,
        "target_clusters": len(group_ids),
        "reference": reference_column,
        "candidates": {},
    }
    for column, subset in replicates.groupby("candidate"):
        summary["candidates"][column] = {}
        for metric in ("delta_spearman", "delta_rmse"):
            values = subset[metric].to_numpy(float)
            summary["candidates"][column][metric] = {
                "mean": float(np.mean(values)),
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_97_5": float(np.quantile(values, 0.975)),
            }
    return replicates, summary


def run_final() -> None:
    selection_path = RESULTS / "confirmation_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run --stage confirm first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pipeline = selection["locked_pipeline"]
    bagged = bool(selection["locked_bagged"])
    locked_oof_column = selection["locked_oof_column"]
    seeds = SEEDS if bagged else (BASE_SEED,)
    print(
        f"Final locked evaluation: pipeline={pipeline}, bagged={bagged}, seeds={list(seeds)}. "
        "New candidate predictions on the fixed validation are generated for the first time now.",
        flush=True,
    )

    train, validation, features = load_data()
    x_train = raw_features(train, features)
    x_validation = raw_features(validation, features)
    prepared = prepare(x_train, x_validation)
    y_train = train[LABEL].to_numpy(float)
    y_validation = validation[LABEL].to_numpy(float)
    exact_train = train[EXACT_COLUMN].to_numpy(float) == 0
    exact_validation = validation[EXACT_COLUMN].to_numpy(float) == 0
    if len(prepared.active) != 183:
        raise RuntimeError(f"Expected 183 active full-training features, got {len(prepared.active)}")

    full_predictions: list[np.ndarray] = []
    model_files: list[str] = []
    for seed in seeds:
        started = time.time()
        prediction, bundle = fit_predict(
            experiment=pipeline,
            x_train=prepared.train,
            y_train=y_train,
            x_test=prepared.validation,
            exact_train=exact_train,
            exact_test=exact_validation,
            seed=seed,
            return_bundle=True,
        )
        full_predictions.append(prediction)
        model_path = MODELS / f"{pipeline}_seed_{seed}.joblib"
        joblib.dump(
            {
                "pipeline": pipeline,
                "seed": seed,
                "bundle": bundle,
                "median": prepared.median,
                "active_features": prepared.active,
            },
            model_path,
        )
        model_files.append(str(model_path.relative_to(RUN_ROOT)))
        print(f"full fit seed={seed} elapsed={time.time() - started:.1f}s", flush=True)
    locked_prediction = np.mean(np.column_stack(full_predictions), axis=1)

    confirmation = pd.read_csv(RESULTS / "oof_seed_confirmation_predictions.csv")
    oof_y = confirmation[LABEL].to_numpy(float)
    oof_locked = confirmation[locked_oof_column].to_numpy(float)
    intercept, slope = fit_affine(oof_y, oof_locked)
    confirmation["locked_candidate_calibrated"] = intercept + slope * oof_locked
    oof_metric_frame = metrics_by_fold(
        confirmation,
        ["baseline_seed_42", locked_oof_column, "locked_candidate_calibrated"],
        stage="locked_training_oof",
    )
    atomic_csv(oof_metric_frame, RESULTS / "locked_training_oof_metrics.csv")
    calibrated_prediction = intercept + slope * locked_prediction

    reference_validation = pd.read_csv(REFERENCE_VALIDATION_FILE)[
        ["record_id", "XGBoost", "OOF_weighted"]
    ].rename(columns={"XGBoost": "baseline_xgboost", "OOF_weighted": "previous_oof_weighted"})
    output = validation[
        ["record_id", "target_sequence", "crRNA_sequence", LABEL, EXACT_COLUMN, "gap_count_target", "gap_count_guide"]
    ].merge(reference_validation, on="record_id", validate="one_to_one")
    output["locked_candidate"] = locked_prediction
    output["locked_candidate_calibrated"] = calibrated_prediction
    atomic_csv(output, RESULTS / "fixed_validation_new_experiment_predictions.csv")

    metrics = metrics_by_fold(
        output,
        ["baseline_xgboost", "previous_oof_weighted", "locked_candidate", "locked_candidate_calibrated"],
        stage="fixed_validation_exploratory",
    )
    atomic_csv(metrics, RESULTS / "fixed_validation_new_experiment_metrics.csv")

    replicates, bootstrap_summary = cluster_bootstrap(
        output,
        ["locked_candidate", "locked_candidate_calibrated"],
        reference_column="baseline_xgboost",
    )
    atomic_csv(replicates, RESULTS / "fixed_validation_target_cluster_bootstrap_replicates.csv")
    atomic_json(bootstrap_summary, RESULTS / "fixed_validation_target_cluster_bootstrap_summary.json")
    weighted_replicates, weighted_bootstrap_summary = cluster_bootstrap(
        output,
        ["locked_candidate", "locked_candidate_calibrated"],
        reference_column="previous_oof_weighted",
    )
    atomic_csv(
        weighted_replicates,
        RESULTS / "fixed_validation_vs_previous_weighted_bootstrap_replicates.csv",
    )
    atomic_json(
        weighted_bootstrap_summary,
        RESULTS / "fixed_validation_vs_previous_weighted_bootstrap_summary.json",
    )

    pooled = metrics.loc[metrics["fold"].eq("pooled")].set_index("experiment")
    oof_pooled = oof_metric_frame.loc[oof_metric_frame["fold"].eq("pooled")].set_index("experiment")
    final_summary = {
        "locked_pipeline": pipeline,
        "locked_bagged": bagged,
        "locked_oof_column": locked_oof_column,
        "locked_status_before_validation": selection["locked_status"],
        "seeds": list(seeds),
        "affine_calibration": {"intercept": intercept, "slope": slope, "fit_source": "locked training OOF predictions only"},
        "locked_training_oof_metrics": {
            name: {key: float(oof_pooled.loc[name, key]) for key in ("spearman", "pearson", "rmse", "mae", "r2")}
            for name in ("baseline_seed_42", locked_oof_column, "locked_candidate_calibrated")
        },
        "validation_metrics": {
            name: {key: float(pooled.loc[name, key]) for key in ("spearman", "pearson", "rmse", "mae", "r2")}
            for name in (
                "baseline_xgboost",
                "previous_oof_weighted",
                "locked_candidate",
                "locked_candidate_calibrated",
            )
        },
        "validation_deltas_vs_baseline": {
            name: {
                "spearman": float(pooled.loc[name, "spearman"] - pooled.loc["baseline_xgboost", "spearman"]),
                "rmse": float(pooled.loc[name, "rmse"] - pooled.loc["baseline_xgboost", "rmse"]),
                "mae": float(pooled.loc[name, "mae"] - pooled.loc["baseline_xgboost", "mae"]),
                "r2": float(pooled.loc[name, "r2"] - pooled.loc["baseline_xgboost", "r2"]),
            }
            for name in ("locked_candidate", "locked_candidate_calibrated")
        },
        "validation_deltas_vs_previous_oof_weighted": {
            name: {
                "spearman": float(pooled.loc[name, "spearman"] - pooled.loc["previous_oof_weighted", "spearman"]),
                "rmse": float(pooled.loc[name, "rmse"] - pooled.loc["previous_oof_weighted", "rmse"]),
                "mae": float(pooled.loc[name, "mae"] - pooled.loc["previous_oof_weighted", "mae"]),
                "r2": float(pooled.loc[name, "r2"] - pooled.loc["previous_oof_weighted", "r2"]),
            }
            for name in ("locked_candidate", "locked_candidate_calibrated")
        },
        "bootstrap": bootstrap_summary,
        "bootstrap_vs_previous_oof_weighted": weighted_bootstrap_summary,
        "interpretation_rule": "A CI crossing zero is numerical/exploratory evidence, not a stable or significant improvement.",
        "model_files": model_files,
    }
    atomic_json(final_summary, RESULTS / "final_experiment_summary.json")
    print_pooled(metrics)
    print(json.dumps(final_summary, indent=2), flush=True)


def write_manifest() -> None:
    train, validation, features = load_data()
    manifest = {
        "created_by": Path(__file__).name,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "data": {
            "path": str(DATA_FILE),
            "sha256": sha256(DATA_FILE),
            "train_rows": len(train),
            "fixed_validation_rows": len(validation),
            "label": LABEL,
        },
        "features": {
            "manifest_path": str(FEATURE_FILE),
            "manifest_sha256": sha256(FEATURE_FILE),
            "manifest_count": len(features),
            "expected_full_train_active_count": 183,
        },
        "folds": {
            "path": str(FOLD_FILE),
            "sha256": sha256(FOLD_FILE),
            "column": FOLD,
            "group": "target_group_id",
            "count": int(train[FOLD].nunique()),
        },
        "xgboost_parameters": XGB_PARAMS,
        "base_seed": BASE_SEED,
        "confirmation_seeds": list(SEEDS),
        "experiments": EXPERIMENTS,
        "governance": {
            "screen_and_selection_source": "training OOF only",
            "fast_screen_delta_spearman_min": 0.002,
            "formal_mean_seed_delta_spearman_min": 0.005,
            "formal_positive_seed_count_min": 4,
            "fixed_validation_status": "historical fixed validation; secondary exploratory confirmation, not untouched test",
        },
    }
    atomic_json(manifest, MANIFESTS / "experiment_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prespecified final-week Cas12a output experiments")
    parser.add_argument(
        "--stage",
        choices=("screen", "confirm", "final", "all"),
        default="all",
        help="Run one resumable stage or the entire workflow",
    )
    parser.add_argument(
        "--candidate",
        choices=("baseline", "rank_target", "two_expert"),
        default=None,
        help="Optional explicit OOF candidate for confirm; default uses screen_selection.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    write_manifest()
    if args.stage in ("screen", "all"):
        run_screen()
    if args.stage in ("confirm", "all"):
        run_confirm(args.candidate)
    if args.stage in ("final", "all"):
        run_final()


if __name__ == "__main__":
    main()
