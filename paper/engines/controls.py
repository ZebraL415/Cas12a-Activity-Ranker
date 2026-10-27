#!/usr/bin/env python3
"""Run the preregistered L02 history-label and mapping-block controls.

This script reads the accepted target-fold run and canonical project inputs but
never writes to the source project/repository. It intentionally contains no
search, tuning, model-selection, or plotting logic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[0]
WORK = PROJECT
CANONICAL_DATA = REPO / "data/processed/v2_2/EasyDesign_2024_V2-2_core_context_feature_table.csv"
FOLD_FILE = REPO / "data/processed/v2_2/frozen_target_grouped_folds.csv"
FEATURE_FILE = REPO / "data/processed/v2_2/feature_manifest.csv"
ACCEPTED_RUN = PROJECT / "data" / "development"
ACCEPTED_PREDICTIONS = ACCEPTED_RUN / "predictions.csv"
SCRIPT_PATH = Path(__file__).resolve()
CONTRACT_PATH = WORK / "03_protocols/L02_perturbation_contract.md"
PERMUTATION_SEEDS = (137, 271, 409)
OUTER_FOLDS = (0, 1, 2, 3, 4)
BOOTSTRAP_SEED = 20260906
BOOTSTRAP_RESAMPLES = 2000
N_RECORDS = 8417
N_GROUPS = 6825
LABEL = "label_normalized"
FOLD = "context_cv_fold"
MAP_COLUMNS: tuple[str, ...] = (
    "mapping_candidate_group_ids",
    "mapping_status",
    "mapping_confidence",
    "mapping_candidate_template_nos",
    "mapping_match_modes",
    "mapping_candidate_orientations",
)

for helper_dir in (Path(__file__).resolve().parent / 'historical',):
    sys.path.insert(0, str(helper_dir))

from run_last_week_experiments import XGB_PARAMS, prepare  # noqa: E402
from run_guide_architecture import make_guide_target_encoding  # noqa: E402
from run_mapping_metadata_architecture import (  # noqa: E402
    CORE_CATEGORICAL,
    TEMPLATE_CATEGORICAL,
    crossfitted_mapping_te,
    mapping_onehot,
    numeric_mapping,
)
from run_sequence_pair_architecture import pair_onehot  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

assert CORE_CATEGORICAL + TEMPLATE_CATEGORICAL == MAP_COLUMNS


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    variable = np.std(y) > 1e-14 and np.std(pred) > 1e-14
    return {
        "spearman": float(spearmanr(y, pred).statistic) if variable else np.nan,
        "pearson": float(pearsonr(y, pred).statistic) if variable else np.nan,
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
        "r2": float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def load_inputs() -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray]]:
    """Load canonical data, fixed folds and accepted predictions with identity checks."""
    data = pd.read_csv(CANONICAL_DATA, low_memory=False)
    folds = pd.read_csv(FOLD_FILE)
    features = pd.read_csv(FEATURE_FILE)["feature_name"].tolist()
    train = data.loc[data["baseline_split"].eq("baseline_train")].merge(
        folds[["record_id", "target_group_id", FOLD]],
        on="record_id",
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)
    assert len(train) == N_RECORDS
    assert len(features) == 188
    assert not train["record_id"].duplicated().any()
    assert not train[FOLD].isna().any()
    assert train.groupby("target_group_id")[FOLD].nunique().max() == 1
    assert train["target_group_id"].nunique() == N_GROUPS

    accepted = pd.read_csv(ACCEPTED_PREDICTIONS, low_memory=False)
    needed_models = ("D", "B", "C", "full_v2", "block_mapping")
    assert set(needed_models).issubset(set(accepted["model_id"].unique()))
    accepted_by_model: dict[str, np.ndarray] = {}
    ref: pd.DataFrame | None = None
    for model_name in needed_models:
        part = accepted.loc[accepted["model_id"].eq(model_name)].copy()
        assert len(part) == N_RECORDS
        assert part["record_id"].nunique() == N_RECORDS
        part = train[["record_id", LABEL, FOLD]].merge(
            part[["record_id", "fold", "y_true", "y_pred"]],
            on="record_id",
            how="left",
            validate="one_to_one",
        )
        assert not part["y_pred"].isna().any()
        assert np.array_equal(part[LABEL].to_numpy(float), part["y_true"].to_numpy(float))
        assert np.array_equal(part[FOLD].to_numpy(int), part["fold"].to_numpy(int))
        assert np.isfinite(part["y_pred"].to_numpy(float)).all()
        if ref is None:
            ref = part[["record_id", LABEL, FOLD]].copy()
        else:
            assert np.array_equal(ref["record_id"].to_numpy(), part["record_id"].to_numpy())
            assert np.array_equal(ref[LABEL].to_numpy(float), part[LABEL].to_numpy(float))
            assert np.array_equal(ref[FOLD].to_numpy(int), part[FOLD].to_numpy(int))
        accepted_by_model[model_name] = part["y_pred"].to_numpy(float)
    assert ref is not None
    assert np.array_equal(train["record_id"].to_numpy(), ref["record_id"].to_numpy())
    assert np.array_equal(train[LABEL].to_numpy(float), ref[LABEL].to_numpy(float))
    assert np.array_equal(train[FOLD].to_numpy(int), ref[FOLD].to_numpy(int))
    return train, features, accepted_by_model


def build_blocks(
    outer_train: pd.DataFrame,
    holdout: pd.DataFrame,
    features: list[str],
    history_train: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Recreate frozen sequence, history and mapping blocks with requested history labels."""
    tr = outer_train.reset_index(drop=True).copy()
    te = holdout.reset_index(drop=True).copy()
    history = (history_train if history_train is not None else tr).reset_index(drop=True).copy()
    te_for_features = te.copy()
    te_for_features[LABEL] = np.nan
    prepared = prepare(
        tr[features].apply(pd.to_numeric, errors="coerce"),
        te_for_features[features].apply(pd.to_numeric, errors="coerce"),
    )
    pair_tr = pair_onehot(tr)
    pair_te = pair_onehot(te_for_features)
    keep_pair = pair_tr.nunique(dropna=False).gt(1)
    pair_tr = pair_tr.loc[:, keep_pair]
    pair_te = pair_te.reindex(columns=pair_tr.columns, fill_value=0)
    guide_tr, guide_te = make_guide_target_encoding(history, te_for_features)
    map_te_tr, map_te_te = crossfitted_mapping_te(history, te_for_features, MAP_COLUMNS, False)
    onehot_tr, onehot_te = mapping_onehot(tr, te_for_features, MAP_COLUMNS)
    numeric_tr, numeric_te = numeric_mapping(tr), numeric_mapping(te_for_features)
    seq_tr = pd.concat([prepared.train.reset_index(drop=True), pair_tr.reset_index(drop=True)], axis=1)
    seq_te = pd.concat([prepared.validation.reset_index(drop=True), pair_te.reset_index(drop=True)], axis=1)
    mapping_tr = pd.concat([numeric_tr.reset_index(drop=True), onehot_tr.reset_index(drop=True)], axis=1)
    mapping_te = pd.concat([numeric_te.reset_index(drop=True), onehot_te.reset_index(drop=True)], axis=1)
    full_tr = pd.concat(
        [prepared.train.reset_index(drop=True), guide_tr.reset_index(drop=True), pair_tr.reset_index(drop=True),
         numeric_tr.reset_index(drop=True), map_te_tr.reset_index(drop=True), onehot_tr.reset_index(drop=True)], axis=1)
    full_te = pd.concat(
        [prepared.validation.reset_index(drop=True), guide_te.reset_index(drop=True), pair_te.reset_index(drop=True),
         numeric_te.reset_index(drop=True), map_te_te.reset_index(drop=True), onehot_te.reset_index(drop=True)], axis=1)
    assert full_tr.columns.equals(full_te.columns)
    assert seq_tr.columns.is_unique and mapping_tr.columns.is_unique
    assert np.isfinite(full_tr.to_numpy()).all() and np.isfinite(full_te.to_numpy()).all()
    return {"seq_tr": seq_tr, "seq_te": seq_te, "mapping_tr": mapping_tr, "mapping_te": mapping_te,
            "full_tr": full_tr, "full_te": full_te, "guide_tr": guide_tr.reset_index(drop=True),
            "guide_te": guide_te.reset_index(drop=True), "mapping_te_tr": map_te_tr.reset_index(drop=True),
            "mapping_te_te": map_te_te.reset_index(drop=True)}


def deterministic_permutation(seed: int, fold: int, stream: int, n: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, fold, stream])).permutation(n)


def row_multiset_hashes(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame.reset_index(drop=True), index=False).to_numpy(dtype=np.uint64)


def assert_no_holdout_label_path(outer_train: pd.DataFrame, holdout: pd.DataFrame, features: list[str], history_train: pd.DataFrame) -> None:
    holdout_nan = holdout.copy()
    holdout_nan[LABEL] = np.nan
    a = build_blocks(outer_train, holdout, features, history_train)
    b = build_blocks(outer_train, holdout_nan, features, history_train)
    for key in a:
        pd.testing.assert_frame_equal(a[key], b[key], check_exact=True)


def fit_history_shuffle(train: pd.DataFrame, features: list[str], accepted_by_model: dict[str, np.ndarray], preflight: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    assertions: dict[str, int] = {}
    folds_to_run = (0,) if preflight else OUTER_FOLDS
    seeds_to_run = (137,) if preflight else PERMUTATION_SEEDS
    for fold in folds_to_run:
        test_mask = train[FOLD].eq(fold).to_numpy()
        outer_train = train.loc[~test_mask].reset_index(drop=True)
        holdout = train.loc[test_mask].reset_index(drop=True)
        y = outer_train[LABEL].to_numpy(float)
        y_holdout = holdout[LABEL].to_numpy(float)
        for seed in seeds_to_run:
            permutation = deterministic_permutation(seed, fold, 0, len(outer_train))
            shuffled_y = y[permutation]
            assert np.array_equal(np.sort(shuffled_y), np.sort(y))
            assert np.mean(permutation != np.arange(len(y))) > 0.5
            assertions["H_history_multiset_and_changed_mapping"] = assertions.get("H_history_multiset_and_changed_mapping", 0) + 1
            history = outer_train.copy()
            history[LABEL] = shuffled_y
            assert outer_train.drop(columns=[LABEL]).reset_index(drop=True).equals(history.drop(columns=[LABEL]).reset_index(drop=True))
            assertions["H_nonhistory_inputs_unchanged"] = assertions.get("H_nonhistory_inputs_unchanged", 0) + 1
            assert_no_holdout_label_path(outer_train, holdout, features, history)
            assertions["H_holdout_label_not_in_feature_path"] = assertions.get("H_holdout_label_not_in_feature_path", 0) + 1
            blocks = build_blocks(outer_train, holdout, features, history)
            guide_tr = blocks["guide_tr"]["te_guide_all"].to_numpy(float)
            guide_te = blocks["guide_te"]["te_guide_all"].to_numpy(float)
            temp_tr = blocks["mapping_te_tr"]["mapping_te_mapping_candidate_template_nos"].to_numpy(float)
            temp_te = blocks["mapping_te_te"]["mapping_te_mapping_candidate_template_nos"].to_numpy(float)
            b_predictions: list[np.ndarray] = []
            for model_seed in (7, 21, 42, 84, 2024):
                model = XGBRegressor(**dict(XGB_PARAMS, max_depth=7, min_child_weight=8), random_state=model_seed)
                model.fit(blocks["full_tr"], y - guide_tr)
                b_predictions.append(guide_te + model.predict(blocks["full_te"]))
            b_pred = np.mean(np.column_stack(b_predictions), axis=1)
            c_model = XGBRegressor(**dict(XGB_PARAMS, max_depth=7, min_child_weight=2), random_state=42)
            c_model.fit(blocks["full_tr"], y - (guide_tr + temp_tr) / 2.0)
            c_pred = (guide_te + temp_te) / 2.0 + c_model.predict(blocks["full_te"])
            d_pred = accepted_by_model["D"][test_mask]
            miss = holdout["mapping_hit_count"].fillna(0).eq(0).to_numpy()
            full_pred = np.where(miss, d_pred, 0.20 * d_pred + 0.47 * b_pred + 0.33 * c_pred)
            assert np.isfinite(b_pred).all() and np.isfinite(c_pred).all() and np.isfinite(full_pred).all()
            assertions["H_finite_predictions"] = assertions.get("H_finite_predictions", 0) + 1
            if preflight:
                print(json.dumps({"experiment": "H", "seed": seed, "fold": fold, "n_test": len(holdout), "n_train": len(outer_train), "full_features": blocks["full_tr"].shape[1]}, indent=2))
                return pd.DataFrame(), pd.DataFrame(), assertions
            base = pd.DataFrame({"record_id": holdout["record_id"].to_numpy(), "dataset_id": "baseline_train", "protocol_id": "target", "permutation_seed": seed, "fold": fold, "y_true": y_holdout, "target_group_id": holdout["target_group_id"].to_numpy(), "prediction_route": np.where(miss, "sequence_fallback", "mapping_aware")})
            for model_id, prediction in (("B_shuffled", b_pred), ("C_shuffled", c_pred), ("full_v2_shuffled", full_pred)):
                frame = base.copy(); frame["model_id"] = model_id; frame["y_pred"] = prediction; frame["prediction_status"] = "ok"; rows.append(frame)
                metric_rows.append({"experiment": "H_history_label_shuffle", "permutation_seed": seed, "fold": fold, "model_id": model_id, "n": len(frame), "group_n": frame["target_group_id"].nunique(), **metrics(y_holdout, prediction)})
    result = pd.concat(rows, ignore_index=True)
    assert len(result) == len(PERMUTATION_SEEDS) * N_RECORDS * 3
    assert result.groupby(["permutation_seed", "model_id"])["record_id"].nunique().eq(N_RECORDS).all()
    assert np.isfinite(result["y_pred"].to_numpy(float)).all()
    assertions["H_unique_record_predictions"] = int(result.groupby(["permutation_seed", "model_id"])["record_id"].nunique().eq(N_RECORDS).all())
    metrics_frame = pd.DataFrame(metric_rows)
    pooled = []
    for (seed, model_id), part in result.groupby(["permutation_seed", "model_id"], sort=True):
        pooled.append({"experiment": "H_history_label_shuffle", "permutation_seed": seed, "fold": "pooled", "model_id": model_id, "n": len(part), "group_n": part["target_group_id"].nunique(), **metrics(part["y_true"].to_numpy(float), part["y_pred"].to_numpy(float))})
    return result, pd.concat([metrics_frame, pd.DataFrame(pooled)], ignore_index=True), assertions


def fit_mapping_permutation(train: pd.DataFrame, features: list[str], accepted_by_model: dict[str, np.ndarray], preflight: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    assertions: dict[str, int] = {}
    folds_to_run = (0,) if preflight else OUTER_FOLDS
    seeds_to_run = (137,) if preflight else PERMUTATION_SEEDS
    for fold in folds_to_run:
        test_mask = train[FOLD].eq(fold).to_numpy()
        outer_train = train.loc[~test_mask].reset_index(drop=True)
        holdout = train.loc[test_mask].reset_index(drop=True)
        y = outer_train[LABEL].to_numpy(float)
        y_holdout = holdout[LABEL].to_numpy(float)
        blocks = build_blocks(outer_train, holdout, features, None)
        seq_tr, seq_te = blocks["seq_tr"], blocks["seq_te"]
        mapping_tr, mapping_te = blocks["mapping_tr"], blocks["mapping_te"]
        assert mapping_tr.columns.equals(mapping_te.columns)
        baseline_train_hash = np.sort(row_multiset_hashes(mapping_tr)); baseline_holdout_hash = np.sort(row_multiset_hashes(mapping_te))
        for seed in seeds_to_run:
            train_perm = deterministic_permutation(seed, fold, 1, len(mapping_tr)); holdout_perm = deterministic_permutation(seed, fold, 2, len(mapping_te))
            assert np.mean(train_perm != np.arange(len(train_perm))) > 0.5 and np.mean(holdout_perm != np.arange(len(holdout_perm))) > 0.5
            map_tr_perm = mapping_tr.iloc[train_perm].reset_index(drop=True); map_te_perm = mapping_te.iloc[holdout_perm].reset_index(drop=True)
            assert np.array_equal(np.sort(row_multiset_hashes(map_tr_perm)), baseline_train_hash); assert np.array_equal(np.sort(row_multiset_hashes(map_te_perm)), baseline_holdout_hash)
            assert seq_tr.equals(blocks["seq_tr"]) and seq_te.equals(blocks["seq_te"])
            assert np.array_equal(y, outer_train[LABEL].to_numpy(float))
            assertions["I_block_shape_and_row_multiset"] = assertions.get("I_block_shape_and_row_multiset", 0) + 1
            assertions["I_sequence_and_y_unchanged"] = assertions.get("I_sequence_and_y_unchanged", 0) + 1
            xtr = pd.concat([seq_tr.reset_index(drop=True), map_tr_perm], axis=1); xte = pd.concat([seq_te.reset_index(drop=True), map_te_perm], axis=1)
            model = XGBRegressor(**dict(XGB_PARAMS, max_depth=7, min_child_weight=8), random_state=42)
            model.fit(xtr, y - y.mean()); pred = y.mean() + model.predict(xte)
            assert np.isfinite(pred).all(); assertions["I_finite_predictions"] = assertions.get("I_finite_predictions", 0) + 1
            if preflight:
                print(json.dumps({"experiment": "I", "seed": seed, "fold": fold, "n_test": len(holdout), "n_train": len(outer_train), "sequence_features": seq_tr.shape[1], "mapping_features": mapping_tr.shape[1]}, indent=2))
                return pd.DataFrame(), pd.DataFrame(), assertions
            frame = pd.DataFrame({"record_id": holdout["record_id"].to_numpy(), "dataset_id": "baseline_train", "protocol_id": "target", "permutation_seed": seed, "fold": fold, "y_true": y_holdout, "target_group_id": holdout["target_group_id"].to_numpy(), "model_id": "block_mapping_permuted", "y_pred": pred, "prediction_status": "ok"})
            rows.append(frame); metric_rows.append({"experiment": "I_mapping_block_permutation", "permutation_seed": seed, "fold": fold, "model_id": "block_mapping_permuted", "n": len(frame), "group_n": frame["target_group_id"].nunique(), **metrics(y_holdout, pred)})
    result = pd.concat(rows, ignore_index=True)
    assert len(result) == len(PERMUTATION_SEEDS) * N_RECORDS
    assert result.groupby("permutation_seed")["record_id"].nunique().eq(N_RECORDS).all(); assert np.isfinite(result["y_pred"].to_numpy(float)).all()
    assertions["I_unique_record_predictions"] = int(result.groupby("permutation_seed")["record_id"].nunique().eq(N_RECORDS).all())
    metrics_frame = pd.DataFrame(metric_rows); pooled = []
    for seed, part in result.groupby("permutation_seed", sort=True):
        pooled.append({"experiment": "I_mapping_block_permutation", "permutation_seed": seed, "fold": "pooled", "model_id": "block_mapping_permuted", "n": len(part), "group_n": part["target_group_id"].nunique(), **metrics(part["y_true"].to_numpy(float), part["y_pred"].to_numpy(float))})
    return result, pd.concat([metrics_frame, pd.DataFrame(pooled)], ignore_index=True), assertions


def bootstrap_deltas(frame: pd.DataFrame, perturbed: np.ndarray, baseline: np.ndarray) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Paired target-cluster bootstrap with one shared resample stream.

    One resampled target-group index set is generated per bootstrap replicate
    and is then used for all five metrics.  This avoids regenerating the same
    resamples five times and preserves the preregistered paired comparison.
    """
    y = frame["y_true"].to_numpy(float)
    groups = frame["target_group_id"].to_numpy()
    codes, group_values = pd.factorize(groups, sort=True)
    row_ids = np.arange(len(frame))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_names = ("spearman", "pearson", "rmse", "mae", "r2")
    point_perturbed = metrics(y, perturbed)
    point_baseline = metrics(y, baseline)
    point_delta = {name: float(point_perturbed[name] - point_baseline[name]) for name in metric_names}
    boot = {name: np.empty(BOOTSTRAP_RESAMPLES, dtype=float) for name in metric_names}
    n_groups = len(group_values)
    for i in range(BOOTSTRAP_RESAMPLES):
        draws = rng.integers(0, n_groups, size=n_groups)
        counts = np.bincount(draws, minlength=n_groups)
        idx = np.repeat(row_ids, counts[codes])
        resampled_perturbed = metrics(y[idx], perturbed[idx])
        resampled_baseline = metrics(y[idx], baseline[idx])
        for name in metric_names:
            boot[name][i] = resampled_perturbed[name] - resampled_baseline[name]
    intervals = {
        name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for name, values in boot.items()
    }
    return point_delta, intervals


def analyze_outputs(train: pd.DataFrame, accepted_by_model: dict[str, np.ndarray], h: pd.DataFrame, i: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []; bootstrap_rows: list[dict[str, Any]] = []; panel_rows: list[dict[str, Any]] = []
    metric_names = ("spearman", "pearson", "rmse", "mae", "r2")
    matching = {"B_shuffled": "B", "C_shuffled": "C", "full_v2_shuffled": "full_v2"}
    accepted_series = {k: pd.Series(v, index=train["record_id"]) for k, v in accepted_by_model.items()}
    for (seed, model_id), part in h.groupby(["permutation_seed", "model_id"], sort=True):
        baseline_model = matching[model_id]; y = part["y_true"].to_numpy(float); pred = part["y_pred"].to_numpy(float); base = accepted_series[baseline_model].reindex(part["record_id"]).to_numpy(float)
        mp, mb = metrics(y, pred), metrics(y, base)
        seed_rows.append({"experiment": "H_history_label_shuffle", "permutation_seed": seed, "evaluation_scope": "pooled", "model_id": model_id, "baseline_model": baseline_model, "n": len(part), "group_n": part["target_group_id"].nunique(), **mp, **{f"delta_{m}": mp[m] - mb[m] for m in metric_names}})
    for seed, part in i.groupby("permutation_seed", sort=True):
        y = part["y_true"].to_numpy(float); pred = part["y_pred"].to_numpy(float); base = accepted_series["block_mapping"].reindex(part["record_id"]).to_numpy(float); mp, mb = metrics(y, pred), metrics(y, base)
        seed_rows.append({"experiment": "I_mapping_block_permutation", "permutation_seed": seed, "evaluation_scope": "pooled", "model_id": "block_mapping_permuted", "baseline_model": "block_mapping", "n": len(part), "group_n": part["target_group_id"].nunique(), **mp, **{f"delta_{m}": mp[m] - mb[m] for m in metric_names}})

    for experiment, source, perturbed_model, baseline_model in (("H_history_label_shuffle", h, "full_v2_shuffled", "full_v2"), ("I_mapping_block_permutation", i, "block_mapping_permuted", "block_mapping")):
        for seed, part in source.loc[source["model_id"].eq(perturbed_model)].groupby("permutation_seed", sort=True):
            base = accepted_series[baseline_model].reindex(part["record_id"]).to_numpy(float); pred = part["y_pred"].to_numpy(float); point = metrics(part["y_true"].to_numpy(float), pred); base_point = metrics(part["y_true"].to_numpy(float), base)
            delta_point, intervals = bootstrap_deltas(part, pred, base)
            for metric_name in metric_names:
                estimate = delta_point[metric_name]; lo, hi = intervals[metric_name]; higher = metric_name in {"spearman", "pearson", "r2"}; direction = "perturbed - baseline; positive is better" if higher else "perturbed - baseline; negative is better"
                bootstrap_rows.append({"experiment": experiment, "permutation_seed": seed, "perturbed_model": perturbed_model, "baseline_model": baseline_model, "metric": metric_name, "delta_estimate": estimate, "ci_low": lo, "ci_high": hi, "n": len(part), "group_n": part["target_group_id"].nunique(), "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED, "higher_is_better": higher, "direction_definition": direction})
                panel_rows.append({"experiment": experiment, "permutation_seed": seed, "series": perturbed_model, "metric": metric_name, "estimate": point[metric_name], "baseline_estimate": base_point[metric_name], "delta": estimate, "ci_low": lo, "ci_high": hi, "n": len(part), "group_n": part["target_group_id"].nunique(), "baseline_model": baseline_model, "higher_is_better": higher, "direction_definition": direction, "panel_role": "primary_paired_target_bootstrap"})
    return pd.DataFrame(seed_rows), pd.DataFrame(bootstrap_rows), pd.DataFrame(panel_rows)


def preflight() -> None:
    train, features, accepted = load_inputs()
    print("L02 preflight: accepted identity checks PASS")
    fit_history_shuffle(train, features, accepted, preflight=True)
    fit_mapping_permutation(train, features, accepted, preflight=True)
    print("L02 preflight: H and I fold=0 seed=137 PASS")


def run_full() -> None:
    start = time.time(); started = now_iso(); train, features, accepted = load_inputs()
    history_root = WORK / "04_runs/L02_history_shuffle"; mapping_root = WORK / "04_runs/L02_mapping_permutation"; history_root.mkdir(parents=True, exist_ok=True); mapping_root.mkdir(parents=True, exist_ok=True)
    history_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_history_shuffle"; mapping_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_mapping_permutation"; history_dir = history_root / history_id; mapping_dir = mapping_root / mapping_id
    for path in (history_dir, mapping_dir):
        if path.exists(): raise FileExistsError(path)
        path.mkdir(parents=True)
    validation: dict[str, Any] = {"status": "running", "started": started, "protocol": "L02_perturbation_contract", "permutation_seeds": list(PERMUTATION_SEEDS), "outer_folds": list(OUTER_FOLDS), "n": N_RECORDS, "group_n": N_GROUPS, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "assertions": {}, "runs": {}, "versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}}
    write_json(history_dir / "status.json", {"status": "running", "started": started}); write_json(mapping_dir / "status.json", {"status": "running", "started": started})
    def log(path: Path, message: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {message}\n"; (path / "run.log").open("a", encoding="utf-8").write(line); print(line, end="", flush=True)
    try:
        log(history_dir, "full history-label shuffle started"); h_pred, h_metrics, h_assertions = fit_history_shuffle(train, features, accepted); log(history_dir, f"history predictions complete rows={len(h_pred)}")
        log(mapping_dir, "full mapping-block permutation started"); i_pred, i_metrics, i_assertions = fit_mapping_permutation(train, features, accepted); log(mapping_dir, f"mapping predictions complete rows={len(i_pred)}")
        h_pred.to_csv(history_dir / "predictions.csv", index=False); h_metrics.to_csv(history_dir / "metrics.csv", index=False); i_pred.to_csv(mapping_dir / "predictions.csv", index=False); i_metrics.to_csv(mapping_dir / "metrics.csv", index=False)
        seed_metrics, bootstrap, panel = analyze_outputs(train, accepted, h_pred, i_pred); analysis_dir = WORK / "05_analysis"; analysis_dir.mkdir(parents=True, exist_ok=True); seed_metrics.to_csv(analysis_dir / "l02_seed_metrics.csv", index=False); bootstrap.to_csv(analysis_dir / "l02_paired_target_bootstrap.csv", index=False); panel.to_csv(analysis_dir / "l02_panel_source_data.csv", index=False)
        common = {"data_path": str(CANONICAL_DATA), "fold_path": str(FOLD_FILE), "feature_manifest_path": str(FEATURE_FILE), "accepted_predictions_path": str(ACCEPTED_PREDICTIONS), "accepted_prediction_models": ["D", "B", "C", "full_v2", "block_mapping"], "n": N_RECORDS, "group_n": N_GROUPS, "permutation_seeds": list(PERMUTATION_SEEDS), "outer_folds": list(OUTER_FOLDS), "unique_variables": {"H": "outer-training history labels only; global row permutation per seed/fold", "I": "whole-row label-free mapping block permutation on train and holdout separately"}, "no_search": True, "no_release_modification": True, "script": str(SCRIPT_PATH), "contract": str(CONTRACT_PATH)}
        for folder, run_id, experiment, pred, met in ((history_dir, history_id, "H_history_label_shuffle", h_pred, h_metrics), (mapping_dir, mapping_id, "I_mapping_block_permutation", i_pred, i_metrics)):
            write_json(folder / "manifest.json", dict(common, run_id=run_id, experiment=experiment, n_predictions=len(pred), metrics_rows=len(met))); write_json(folder / "status.json", {"status": "complete", "finished": now_iso(), "n_predictions": len(pred), "n_metrics": len(met)})
        registration = pd.DataFrame([{"experiment": "H_history_label_shuffle", "run_id": history_id, "status": "complete", "predictions_path": str(history_dir / "predictions.csv"), "metrics_path": str(history_dir / "metrics.csv"), "n_predictions": len(h_pred), "n": N_RECORDS, "group_n": N_GROUPS, "permutation_seeds": ";".join(map(str, PERMUTATION_SEEDS))}, {"experiment": "I_mapping_block_permutation", "run_id": mapping_id, "status": "complete", "predictions_path": str(mapping_dir / "predictions.csv"), "metrics_path": str(mapping_dir / "metrics.csv"), "n_predictions": len(i_pred), "n": N_RECORDS, "group_n": N_GROUPS, "permutation_seeds": ";".join(map(str, PERMUTATION_SEEDS))}]); accepted_runs = WORK / "04_runs/accepted_l02_runs.csv"
        if accepted_runs.exists():
            prior = pd.read_csv(accepted_runs)
            assert not set(registration["run_id"]).intersection(set(prior.get("run_id", []))), "run_id collision; refusing to overwrite accepted L02 registry"
            registration = pd.concat([prior, registration], ignore_index=True)
        registration.to_csv(accepted_runs, index=False)
        validation["assertions"] = dict(h_assertions, **{f"{k}_I": v for k, v in i_assertions.items()}); validation["runs"] = {"H": history_id, "I": mapping_id}; validation["output_rows"] = {"H": len(h_pred), "I": len(i_pred)}; validation["analysis_rows"] = {"seed_metrics": len(seed_metrics), "bootstrap": len(bootstrap), "panel": len(panel)}; validation["sha256_final_checkpoint"] = {"script": sha256(SCRIPT_PATH), "contract": sha256(CONTRACT_PATH), "history_predictions": sha256(history_dir / "predictions.csv"), "mapping_predictions": sha256(mapping_dir / "predictions.csv"), "panel_source_data": sha256(analysis_dir / "l02_panel_source_data.csv")}; validation["finished"] = now_iso(); validation["seconds"] = time.time() - start; validation["status"] = "complete"; write_json(WORK / "08_review/statistics/l02_validation.json", validation); log(history_dir, "COMPLETE history-label shuffle"); log(mapping_dir, "COMPLETE mapping-block permutation")
    except Exception as exc:
        validation["status"] = "failed"; validation["error"] = str(exc); validation["traceback"] = traceback.format_exc(); validation["finished"] = now_iso(); write_json(WORK / "08_review/statistics/l02_validation.json", validation); write_json(history_dir / "status.json", validation); write_json(mapping_dir / "status.json", validation); raise


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("preflight", "run")); args = parser.parse_args(); preflight() if args.action == "preflight" else run_full()


if __name__ == "__main__":
    main()
