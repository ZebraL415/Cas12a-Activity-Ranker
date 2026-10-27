from __future__ import annotations

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
    atomic_csv,
    atomic_json,
    load_data,
    make_xgb,
    prepare,
    raw_features,
)
from run_guide_architecture import make_guide_target_encoding  # noqa: E402
from run_sequence_pair_architecture import pair_onehot  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"
SMOOTHING = 20.0
CORE_CATEGORICAL = (
    "mapping_candidate_group_ids",
    "mapping_status",
    "mapping_confidence",
)
TEMPLATE_CATEGORICAL = (
    "mapping_candidate_template_nos",
    "mapping_match_modes",
    "mapping_candidate_orientations",
)
MAPPING_NUMERIC = (
    "mapping_hit_count",
    "mapping_template_count",
    "mapping_group_count",
    "mapping_orientation_count",
)


def metric_dict(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, prediction).statistic),
        "pearson": float(pearsonr(y, prediction).statistic),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }


def normalized_key(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("missing").astype(str)


def lookup_encoding(
    source_key: pd.Series,
    query_key: pd.Series,
    source_y: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    temp = pd.DataFrame({"key": source_key.astype(str).to_numpy(), "y": source_y})
    stats = temp.groupby("key", sort=False)["y"].agg(["sum", "count"])
    sums = query_key.astype(str).map(stats["sum"]).fillna(0).to_numpy(float)
    counts = query_key.astype(str).map(stats["count"]).fillna(0).to_numpy(float)
    prior = float(np.mean(source_y))
    return (sums + smoothing * prior) / (counts + smoothing), np.log1p(counts)


def crossfitted_mapping_te(
    outer_train: pd.DataFrame,
    holdout: pd.DataFrame,
    columns: tuple[str, ...],
    include_interaction: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = outer_train.reset_index(drop=True)
    test = holdout.reset_index(drop=True)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    keys_train = {column: normalized_key(train, column) for column in columns}
    keys_test = {column: normalized_key(test, column) for column in columns}
    if include_interaction:
        diff_train = np.where(
            train["aligned_difference_count"].eq(0),
            "exact",
            np.where(train["aligned_difference_count"].eq(1), "one", "multi"),
        )
        diff_test = np.where(
            test["aligned_difference_count"].eq(0),
            "exact",
            np.where(test["aligned_difference_count"].eq(1), "one", "multi"),
        )
        keys_train["mapping_group_x_difference_class"] = (
            normalized_key(train, "mapping_candidate_group_ids") + "||" + pd.Series(diff_train, index=train.index)
        )
        keys_test["mapping_group_x_difference_class"] = (
            normalized_key(test, "mapping_candidate_group_ids") + "||" + pd.Series(diff_test, index=test.index)
        )
    train_features: dict[str, np.ndarray] = {}
    holdout_features: dict[str, np.ndarray] = {}
    for name in keys_train:
        train_value = np.full(len(train), np.nan)
        train_count = np.full(len(train), np.nan)
        for inner_fold in sorted(np.unique(folds)):
            source_mask = folds != inner_fold
            query_mask = folds == inner_fold
            value, count = lookup_encoding(
                keys_train[name].loc[source_mask],
                keys_train[name].loc[query_mask],
                y[source_mask],
                SMOOTHING,
            )
            train_value[query_mask] = value
            train_count[query_mask] = count
        holdout_value, holdout_count = lookup_encoding(
            keys_train[name],
            keys_test[name],
            y,
            SMOOTHING,
        )
        train_features[f"mapping_te_{name}"] = train_value
        train_features[f"mapping_te_{name}_log_count"] = train_count
        holdout_features[f"mapping_te_{name}"] = holdout_value
        holdout_features[f"mapping_te_{name}_log_count"] = holdout_count
    return (
        pd.DataFrame(train_features, index=outer_train.index),
        pd.DataFrame(holdout_features, index=holdout.index),
    )


def mapping_onehot(
    outer_train: pd.DataFrame,
    holdout: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    for column in columns:
        train_dummy = pd.get_dummies(normalized_key(outer_train, column), prefix=f"mapping_cat_{column}", dtype=np.uint8)
        holdout_dummy = pd.get_dummies(normalized_key(holdout, column), prefix=f"mapping_cat_{column}", dtype=np.uint8)
        holdout_dummy = holdout_dummy.reindex(columns=train_dummy.columns, fill_value=0)
        train_parts.append(train_dummy.set_axis(outer_train.index))
        holdout_parts.append(holdout_dummy.set_axis(holdout.index))
    return pd.concat(train_parts, axis=1), pd.concat(holdout_parts, axis=1)


def numeric_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(MAPPING_NUMERIC)].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)


def main() -> None:
    train, _, features = load_data()
    train = train.reset_index(drop=True)
    y = train[LABEL].to_numpy(float)
    folds = train[FOLD].to_numpy(int)
    base_x = raw_features(train, features)
    pair_x = pair_onehot(train)
    variants = {
        "mapping_core_xgb": (CORE_CATEGORICAL, False),
        "mapping_template_xgb": (CORE_CATEGORICAL + TEMPLATE_CATEGORICAL, False),
        "mapping_interaction_xgb": (CORE_CATEGORICAL + TEMPLATE_CATEGORICAL, True),
    }
    predictions = {name: np.full(len(train), np.nan) for name in variants}

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        holdout_mask = folds == fold
        outer_train = train.loc[train_mask]
        holdout = train.loc[holdout_mask]
        prepared = prepare(base_x.loc[train_mask], base_x.loc[holdout_mask])
        train_guide_te, holdout_guide_te = make_guide_target_encoding(outer_train, holdout)
        keep_pair = pair_x.loc[train_mask].nunique(dropna=False).gt(1)
        base_train = pd.concat([prepared.train, train_guide_te, pair_x.loc[train_mask, keep_pair]], axis=1)
        base_holdout = pd.concat([prepared.validation, holdout_guide_te, pair_x.loc[holdout_mask, keep_pair]], axis=1)
        target = y[train_mask] - train_guide_te["te_guide_all"].to_numpy(float)
        prior = holdout_guide_te["te_guide_all"].to_numpy(float)
        for name, (categorical_columns, include_interaction) in variants.items():
            train_mapping_te, holdout_mapping_te = crossfitted_mapping_te(
                outer_train,
                holdout,
                categorical_columns,
                include_interaction,
            )
            train_onehot, holdout_onehot = mapping_onehot(outer_train, holdout, categorical_columns)
            x_train = pd.concat(
                [base_train, numeric_mapping(outer_train), train_mapping_te, train_onehot], axis=1
            )
            x_holdout = pd.concat(
                [base_holdout, numeric_mapping(holdout), holdout_mapping_te, holdout_onehot], axis=1
            )
            model = make_xgb(BASE_SEED)
            model.fit(x_train, target)
            prediction = prior + model.predict(x_holdout)
            predictions[name][holdout_mask] = prediction
            print(
                f"mapping fold={fold} variant={name} features={x_train.shape[1]} "
                f"rho={spearmanr(y[holdout_mask], prediction).statistic:.6f}",
                flush=True,
            )

    ensemble = pd.read_csv(RESULTS / "oof_seed_bagged_components_predictions.csv")
    output = train[["record_id", FOLD, LABEL]].merge(
        ensemble[["record_id", "fixed_weight_seed42_ensemble"]], on="record_id", validate="one_to_one"
    )
    base = output["fixed_weight_seed42_ensemble"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    best: tuple[float, str, float, np.ndarray] | None = None
    for name, prediction in predictions.items():
        output[name] = prediction
        rows.append({"experiment": name, **metric_dict(y, prediction)})
        for weight in np.linspace(0.0, 0.80, 161):
            blend = (1.0 - weight) * base + weight * prediction
            rho = float(spearmanr(y, blend).statistic)
            if best is None or rho > best[0]:
                best = (rho, name, float(weight), blend)
    assert best is not None
    output["best_mapping_blend"] = best[3]
    rows.append({"experiment": "fixed_weight_seed42_ensemble", **metric_dict(y, base)})
    rows.append({"experiment": "best_mapping_blend", **metric_dict(y, best[3])})
    atomic_csv(output, RESULTS / "oof_mapping_metadata_predictions.csv")
    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, RESULTS / "oof_mapping_metadata_metrics.csv")
    base_rho = float(spearmanr(y, base).statistic)
    decision = {
        "selected_model": best[1],
        "selected_weight": best[2],
        "best_blend_spearman": best[0],
        "delta_vs_static_ensemble": float(best[0] - base_rho),
        "promote": bool(best[0] >= base_rho + 0.001),
        "reached_0_8_oof": bool(best[0] >= 0.8),
        "scope_restriction": "valid only when EasyDesign mapping/template metadata is available; not a sequence-only model",
    }
    atomic_json(decision, RESULTS / "mapping_metadata_decision.json")
    atomic_json(
        {
            "outer_protocol": "frozen five target-group folds",
            "core_categorical": list(CORE_CATEGORICAL),
            "template_categorical": list(TEMPLATE_CATEGORICAL),
            "numeric": list(MAPPING_NUMERIC),
            "target_encoding": "inner-fold cross-fitted within every outer training partition",
            "validation_labels_used": False,
            "risk": "mapping groups may encode dataset/template batch effects",
        },
        MANIFESTS / "mapping_metadata_manifest.json",
    )
    print(metrics.sort_values("spearman", ascending=False).to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
