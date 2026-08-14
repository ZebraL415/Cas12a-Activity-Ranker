#!/usr/bin/env python3
"""Cas12a final four locked comparisons — one-file implementation.

Runs four strictly comparable experiments on the frozen V2-2 protocol:
XGBoost, CatBoost, 50/50 ensemble, and OOF-selected weighted ensemble.

Expected repository layout:
  scripts/train_final_four.py
  data/processed/v2_2/...

Usage:
  python scripts/train_final_four.py --verify-only
  python scripts/train_final_four.py --output reproduced_run

The OOF-selected 59/41 ensemble is an exploratory comparison.  The historical
fixed validation set was not used to select its weight in this final run, but
it had been inspected during earlier project development and is not described
as an untouched external test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
EXPECTED_SHA256 = "39cda8368c216784507ac002df687b28a4f9cc6f81e2b0e84043e45eddb4c1c0"
EXPECTED_COUNTS = {"train": 8417, "validation": 2217, "manifest": 188}

XGB_PARAMS = dict(
    n_estimators=1100, learning_rate=0.02, max_depth=7,
    min_child_weight=2, subsample=0.85, colsample_bytree=0.80,
    reg_alpha=0.05, reg_lambda=2,
)
CAT_PARAMS = dict(
    iterations=1000, learning_rate=0.02, depth=9,
    l2_leaf_reg=5, random_strength=0.5,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_xgboost():
    from xgboost import XGBRegressor
    return XGBRegressor(
        objective="reg:squarederror", random_state=SEED, n_jobs=1,
        tree_method="hist", eval_metric="rmse", **XGB_PARAMS,
    )


def make_catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        loss_function="RMSE", random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=1, **CAT_PARAMS,
    )


def preprocess(train: pd.DataFrame, other: pd.DataFrame):
    medians = train.median()
    train = train.fillna(medians).fillna(0)
    other = other.fillna(medians).fillna(0)
    active = train.nunique(dropna=False).gt(1)
    names = list(train.columns[active])
    return train.loc[:, active], other.loc[:, active], medians[active], names


def locate_files(data_root: Path) -> tuple[Path, Path, Path]:
    profile = data_root / "v2_2"
    data = profile / "EasyDesign_2024_V2-2_core_context_feature_table.csv"
    manifest = profile / "feature_manifest.csv"
    folds = profile / "frozen_target_grouped_folds.csv"
    missing = [str(p) for p in (data, manifest, folds) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    return data, manifest, folds


def load_and_verify(data_root: Path):
    data_path, manifest_path, folds_path = locate_files(data_root)
    actual_hash = file_sha256(data_path)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"Data SHA-256 mismatch: {actual_hash}")
    frame = pd.read_csv(data_path, low_memory=False)
    features = pd.read_csv(manifest_path)["feature_name"].tolist()
    folds = pd.read_csv(folds_path)
    train = frame.loc[frame["baseline_split"].eq("baseline_train")].merge(
        folds[["record_id", "context_cv_fold"]], on="record_id", validate="one_to_one"
    )
    validation = frame.loc[frame["baseline_split"].eq("baseline_validation")].copy()
    actual = (len(train), len(validation), len(features))
    expected = (EXPECTED_COUNTS["train"], EXPECTED_COUNTS["validation"], EXPECTED_COUNTS["manifest"])
    if actual != expected:
        raise ValueError(f"Protocol count mismatch: got {actual}, expected {expected}")
    print(f"Verified data: {data_path}")
    print(f"SHA-256: {actual_hash}; train={actual[0]}, validation={actual[1]}, manifest={actual[2]}")
    return data_path, train, validation, features


def run(data_root: Path, output: Path, resume: bool = True) -> None:
    import joblib
    data_path, train, validation, features = load_and_verify(data_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)

    x = train[features].apply(pd.to_numeric, errors="coerce")
    y = train["label_normalized"].to_numpy(float)
    fold_ids = train["context_cv_fold"].to_numpy(int)
    oof_path = output / "oof_predictions.csv"

    if resume and oof_path.exists():
        saved = pd.read_csv(oof_path)
        if len(saved) != len(train) or saved[["xgboost_oof", "catboost_oof"]].isna().any().any():
            raise ValueError("Existing OOF file is incomplete or incompatible; rerun with --no-resume.")
        oof_xgb = saved["xgboost_oof"].to_numpy()
        oof_cat = saved["catboost_oof"].to_numpy()
        print("Resumed complete OOF predictions.")
    else:
        oof_xgb = np.full(len(train), np.nan)
        oof_cat = np.full(len(train), np.nan)
        for fold in range(5):
            fit_mask, hold_mask = fold_ids != fold, fold_ids == fold
            fit_x, hold_x, _, _ = preprocess(x.loc[fit_mask], x.loc[hold_mask])
            xgb_model, cat_model = make_xgboost(), make_catboost()
            xgb_model.fit(fit_x, y[fit_mask])
            cat_model.fit(fit_x, y[fit_mask])
            oof_xgb[hold_mask] = xgb_model.predict(hold_x)
            oof_cat[hold_mask] = cat_model.predict(hold_x)
            print(f"Completed OOF fold {fold + 1}/5")
        pd.DataFrame({
            "record_id": train["record_id"], "fold": fold_ids,
            "label_normalized": y, "xgboost_oof": oof_xgb,
            "catboost_oof": oof_cat,
        }).to_csv(oof_path, index=False)

    weight_rows = []
    for weight in np.linspace(0, 1, 101):
        pred = weight * oof_xgb + (1 - weight) * oof_cat
        weight_rows.append({"xgboost_weight": weight, "catboost_weight": 1 - weight, **metrics(y, pred)})
    weight_table = pd.DataFrame(weight_rows)
    weight_table.to_csv(output / "oof_weight_search.csv", index=False)
    best = weight_table.sort_values(["spearman", "rmse"], ascending=[False, True]).iloc[0]
    xgb_weight = float(best["xgboost_weight"])

    valid_x = validation[features].apply(pd.to_numeric, errors="coerce")
    fit_x, valid_x, medians, active_features = preprocess(x, valid_x)
    valid_y = validation["label_normalized"].to_numpy(float)
    xgb_model, cat_model = make_xgboost(), make_catboost()
    xgb_model.fit(fit_x, y)
    cat_model.fit(fit_x, y)
    pred_xgb, pred_cat = xgb_model.predict(valid_x), cat_model.predict(valid_x)
    joblib.dump({"model": xgb_model, "median": medians, "features": active_features}, output / "models" / "xgboost_final.joblib")
    joblib.dump({"model": cat_model, "median": medians, "features": active_features}, output / "models" / "catboost_final.joblib")
    xgb_model.get_booster().save_model(output / "models" / "xgboost_final.json")
    cat_model.save_model(output / "models" / "catboost_final.cbm")
    pd.DataFrame({"feature_name": active_features, "median": medians.to_numpy(float)}).to_csv(
        output / "models" / "training_medians.csv", index=False
    )

    predictions = {
        "XGBoost": pred_xgb,
        "CatBoost": pred_cat,
        "Equal_50_50": 0.5 * pred_xgb + 0.5 * pred_cat,
        "OOF_weighted": xgb_weight * pred_xgb + (1 - xgb_weight) * pred_cat,
    }
    result_rows = [{
        "experiment": name, "n_train": len(train), "n_validation": len(validation),
        "features": len(active_features), **metrics(valid_y, pred),
    } for name, pred in predictions.items()]
    result_table = pd.DataFrame(result_rows).sort_values("spearman", ascending=False)
    result_table.to_csv(output / "final_metrics.csv", index=False)
    pd.DataFrame({"record_id": validation["record_id"], "label_normalized": valid_y, **predictions}).to_csv(output / "fixed_validation_predictions.csv", index=False)
    run_manifest = {
        "data_file": data_path.name, "data_sha256": file_sha256(data_path),
        "seed": SEED, "train": len(train), "validation": len(validation),
        "manifest_features": len(features), "active_features": len(active_features),
        "oof_folds": 5, "xgboost_params": XGB_PARAMS, "catboost_params": CAT_PARAMS,
        "ensemble_weight": {"xgboost": xgb_weight, "catboost": 1 - xgb_weight},
        "fixed_validation_used_for_weight_selection": False,
        "validation_role": "historical_fixed_internal_validation",
        "weighted_ensemble_role": "exploratory_oof_tuned_comparator",
        "primary_model_recommendation": "XGBoost",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(result_table.to_string(index=False))
    print(f"OOF weight: XGBoost={xgb_weight:.2f}, CatBoost={1-xgb_weight:.2f}")
    print(f"Results written to: {output.resolve()}")


def smoke_test(data_root: Path) -> None:
    """Fit very small versions of both learners to verify the local runtime."""
    _, train, validation, features = load_and_verify(data_root)
    train = train.iloc[:512].copy()
    validation = validation.iloc[:128].copy()
    x = train[features].apply(pd.to_numeric, errors="coerce")
    valid_x = validation[features].apply(pd.to_numeric, errors="coerce")
    fit_x, valid_x, _, _ = preprocess(x, valid_x)
    y = train["label_normalized"].to_numpy(float)
    xgb_model = make_xgboost().set_params(n_estimators=5)
    cat_model = make_catboost().set_params(iterations=5)
    xgb_model.fit(fit_x, y)
    cat_model.fit(fit_x, y)
    xgb_pred = np.asarray(xgb_model.predict(valid_x), dtype=float)
    cat_pred = np.asarray(cat_model.predict(valid_x), dtype=float)
    if not (np.isfinite(xgb_pred).all() and np.isfinite(cat_pred).all()):
        raise RuntimeError("Smoke-test predictions contain non-finite values")
    print(f"Smoke test passed: XGBoost and CatBoost each predicted {len(valid_x)} rows.")


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=base / "data" / "processed")
    parser.add_argument("--output", type=Path, default=base / "reproduced_run")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        load_and_verify(args.data_root)
    elif args.smoke_test:
        smoke_test(args.data_root)
    else:
        run(args.data_root, args.output, resume=not args.no_resume)


if __name__ == "__main__":
    main()
