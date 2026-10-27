from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = AUDIT_ROOT.parent
LIGHTGBM_PACKAGES = PROJECT / "architecture_search" / "lightgbm_packages"
sys.path.append(str(LIGHTGBM_PACKAGES))
from lightgbm import LGBMRegressor  # noqa: E402
import lightgbm  # noqa: E402


PACKAGE = (
    PROJECT
    / "final_repo_audit"
    / "rigorous"
    / "data_package"
    / "Cas12a_Data"
    / "Unified_Horizontal_V2-2_P1_V4"
)
DATA_FILE = PACKAGE / "data" / "EasyDesign_2024_V2-2_core_context_feature_table.csv"
FEATURE_FILE = PACKAGE / "metadata" / "EasyDesign_2024_V2-2_feature_manifest.csv"
FOLD_FILE = PACKAGE / "metadata" / "EasyDesign_2024_V2-2_frozen_target_grouped_folds.csv"
SEED = 42
MODEL_NAMES = ("xgb", "lightgbm", "mlp")


TEAMMATE_EXCLUDE = {
    "baseline_split",
    "label_normalized",
    "label_raw_name",
    "label_raw_value",
    "label_scale_group",
    "label_is_primary_baseline",
    "crRNA_sequence",
    "target_sequence",
}


def metric_dict(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, prediction).statistic),
        "pearson": float(pearsonr(y, prediction).statistic),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }


def make_models() -> dict[str, object]:
    return {
        "xgb": XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=4,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=4,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.001,
                        max_iter=500,
                        early_stopping=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def prepare_exact(
    train_frame: pd.DataFrame, holdout_frame: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_x = train_frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    holdout_x = holdout_frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return train_x, holdout_x, features


def prepare_manifest(
    train_frame: pd.DataFrame, holdout_frame: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_x = train_frame[features].apply(pd.to_numeric, errors="coerce")
    holdout_x = holdout_frame[features].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0).fillna(0.0)
    train_x = train_x.fillna(medians)
    holdout_x = holdout_x.fillna(medians)
    active = train_x.nunique(dropna=False).gt(1)
    active_features = active[active].index.tolist()
    return train_x[active_features], holdout_x[active_features], active_features


def select_weights(
    y: np.ndarray, predictions: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray]:
    rows: list[dict[str, float]] = []
    best: tuple[float, float, float, float, np.ndarray] | None = None
    for ix in range(101):
        w_xgb = ix / 100.0
        for il in range(101 - ix):
            w_lgbm = il / 100.0
            w_mlp = 1.0 - w_xgb - w_lgbm
            prediction = (
                w_xgb * predictions["xgb"]
                + w_lgbm * predictions["lightgbm"]
                + w_mlp * predictions["mlp"]
            )
            score = float(spearmanr(y, prediction).statistic)
            rows.append(
                {
                    "xgb_weight": w_xgb,
                    "lightgbm_weight": w_lgbm,
                    "mlp_weight": w_mlp,
                    "spearman": score,
                }
            )
            if best is None or score > best[0]:
                best = (score, w_xgb, w_lgbm, w_mlp, prediction.copy())
    assert best is not None
    weights = {"xgb": best[1], "lightgbm": best[2], "mlp": best[3]}
    return pd.DataFrame(rows), weights, best[4]


def main(variant: str) -> None:
    start = time.time()
    output = AUDIT_ROOT / "output" / f"clean_{variant}"
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_FILE, low_memory=False)
    folds = pd.read_csv(FOLD_FILE)
    train = data.loc[data["baseline_split"].eq("baseline_train")].merge(
        folds[["record_id", "target_group_id", "context_cv_fold"]],
        on="record_id",
        validate="one_to_one",
    )
    validation = data.loc[data["baseline_split"].eq("baseline_validation")].copy()
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    if variant == "teammate_exact":
        features = [column for column in data.columns if column not in TEAMMATE_EXCLUDE]
        prepare = prepare_exact
    elif variant == "manifest_safe":
        features = pd.read_csv(FEATURE_FILE)["feature_name"].tolist()
        prepare = prepare_manifest
    else:
        raise ValueError(variant)

    y = train["label_normalized"].to_numpy(float)
    y_validation = validation["label_normalized"].to_numpy(float)
    oof_predictions = {name: np.full(len(train), np.nan) for name in MODEL_NAMES}
    fold_rows: list[dict[str, object]] = []
    active_by_fold: dict[str, list[str]] = {}

    for fold in sorted(train["context_cv_fold"].unique()):
        fit_mask = train["context_cv_fold"].ne(fold).to_numpy()
        holdout_mask = ~fit_mask
        x_fit, x_holdout, active = prepare(
            train.loc[fit_mask], train.loc[holdout_mask], features
        )
        active_by_fold[str(int(fold))] = active
        models = make_models()
        for name, model in models.items():
            model.fit(x_fit, y[fit_mask])
            prediction = np.asarray(model.predict(x_holdout), dtype=float)
            oof_predictions[name][holdout_mask] = prediction
            fold_rows.append(
                {
                    "fold": int(fold),
                    "model": name,
                    "records": int(holdout_mask.sum()),
                    "active_features": len(active),
                    **metric_dict(y[holdout_mask], prediction),
                }
            )
        print(f"{variant}: completed fold {fold}", flush=True)

    if any(np.isnan(values).any() for values in oof_predictions.values()):
        raise RuntimeError("OOF predictions contain missing values")
    grid, weights, oof_ensemble = select_weights(y, oof_predictions)
    grid.to_csv(output / "oof_weight_grid.csv", index=False)

    x_train, x_validation, active_full = prepare(train, validation, features)
    validation_predictions: dict[str, np.ndarray] = {}
    full_models = make_models()
    for name, model in full_models.items():
        model.fit(x_train, y)
        validation_predictions[name] = np.asarray(model.predict(x_validation), dtype=float)
        joblib.dump(model, output / f"{name}.joblib")
        print(f"{variant}: completed full {name}", flush=True)
    validation_ensemble = sum(
        weights[name] * validation_predictions[name] for name in MODEL_NAMES
    )

    oof_frame = train[
        ["record_id", "target_group_id", "context_cv_fold", "label_normalized"]
    ].copy()
    fixed_frame = validation[
        ["record_id", "target_sequence", "label_normalized"]
    ].copy()
    for name in MODEL_NAMES:
        oof_frame[name] = oof_predictions[name]
        fixed_frame[name] = validation_predictions[name]
    oof_frame["heterogeneous_ensemble"] = oof_ensemble
    fixed_frame["heterogeneous_ensemble"] = validation_ensemble
    oof_frame.to_csv(output / "oof_predictions.csv", index=False)
    fixed_frame.to_csv(output / "fixed_validation_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    for protocol, target, prediction_set, ensemble in (
        ("training_target_group_oof", y, oof_predictions, oof_ensemble),
        (
            "historical_fixed_validation",
            y_validation,
            validation_predictions,
            validation_ensemble,
        ),
    ):
        for name in MODEL_NAMES:
            metric_rows.append(
                {"protocol": protocol, "model": name, **metric_dict(target, prediction_set[name])}
            )
        metric_rows.append(
            {
                "protocol": protocol,
                "model": "heterogeneous_ensemble",
                **metric_dict(target, ensemble),
            }
        )
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)

    manifest = {
        "variant": variant,
        "strict_change_from_teammate": (
            "weight selection moved from fixed validation labels to five frozen target-group OOF folds"
            if variant == "teammate_exact"
            else "feature inputs restricted to the audited 188-feature manifest; all weights selected on frozen target-group OOF"
        ),
        "n_train": len(train),
        "n_fixed_validation": len(validation),
        "manifest_features": len(features),
        "full_active_features": len(active_full),
        "weights_selected_on": "training_target_group_oof",
        "fixed_validation_labels_used_for_weight_selection": False,
        "weights": weights,
        "oof_metrics": metric_dict(y, oof_ensemble),
        "fixed_validation_metrics": metric_dict(y_validation, validation_ensemble),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
        },
        "seconds": time.time() - start,
        "active_features": active_full,
        "active_features_by_fold": active_by_fold,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"active_features", "active_features_by_fold"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("teammate_exact", "manifest_safe"), required=True
    )
    args = parser.parse_args()
    main(args.variant)
