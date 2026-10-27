"""Frozen TriCasRank model factories and one-fit component reuse.

This module deliberately consumes the feature bundle by attribute only.  It
does not import the feature implementation, so the model recipe can be
audited and constructed independently of feature-generation changes.
"""

from __future__ import annotations

import math
from importlib import metadata
from typing import Any, Mapping

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


D_WEIGHTS: dict[str, float] = {"xgb": 0.35, "lightgbm": 0.59, "mlp": 0.06}
FUSION_WEIGHTS: dict[str, float] = {"D": 0.20, "B": 0.47, "C": 0.33}
D_B_WEIGHTS: dict[str, float] = {"D": 0.20 / 0.67, "B": 0.47 / 0.67}
B_SEEDS: tuple[int, ...] = (7, 21, 42, 84, 2024)
C_SEED = 42
GUIDE_SMOOTHING = 0.5
MAPPING_SMOOTHING = 20.0
MISSING_MAPPING_ROUTE = "D"

# Copied from last_week_experiments/scripts/run_last_week_experiments.py.
# B and C then apply only the overrides present in paper_pipeline.py.
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


def _new_d_models() -> dict[str, Any]:
    """Mirror the source ``make_models`` constructor, including defaults."""
    return {
        "xgb": XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
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
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _new_residual_xgb(*, seed: int, min_child_weight: int) -> XGBRegressor:
    """Mirror the B/C/block_mapping XGBRegressor call in paper_pipeline.py."""
    return XGBRegressor(
        **dict(XGB_PARAMS, max_depth=7, min_child_weight=min_child_weight),
        random_state=seed,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "get_params"):
        return {
            "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "params": _jsonable(value.get_params(deep=False)),
        }
    if callable(value):
        return f"{value.__module__}.{getattr(value, '__qualname__', value.__class__.__name__)}"
    return str(value)


def _snapshot(estimator: Any) -> dict[str, Any]:
    return {
        "class": f"{estimator.__class__.__module__}.{estimator.__class__.__qualname__}",
        "resolved_get_params_deep_true": _jsonable(estimator.get_params(deep=True)),
    }


def resolved_recipe() -> dict[str, Any]:
    """Return the frozen recipe plus library-resolved constructor defaults.

    Calling this function only instantiates estimators and reads their
    ``get_params`` values; it never fits a model.
    """
    d = _new_d_models()
    b = {str(seed): _new_residual_xgb(seed=seed, min_child_weight=8) for seed in B_SEEDS}
    c = _new_residual_xgb(seed=C_SEED, min_child_weight=2)
    mapping = _new_residual_xgb(seed=42, min_child_weight=8)
    versions = {}
    for distribution in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "joblib",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "recipe_id": "TriCasRank-fixed-2026-10-12",
        "source_constructors": {
            "D": "parallel_model_optimization_audit/work/run_clean_heterogeneous.py::make_models",
            "B": "paper_results_discussion/04_analysis/scripts/paper_pipeline.py; XGB_PARAMS from last_week_experiments/scripts/run_last_week_experiments.py",
            "C": "paper_results_discussion/04_analysis/scripts/paper_pipeline.py; XGB_PARAMS from last_week_experiments/scripts/run_last_week_experiments.py",
            "SEQ_MAPPING_XGB": "paper_results_discussion/04_analysis/scripts/paper_pipeline.py::block_mapping",
        },
        "weights": {
            "D_components": dict(D_WEIGHTS),
            "TriCasRank": dict(FUSION_WEIGHTS),
            "D_B_FIXED": dict(D_B_WEIGHTS),
        },
        "seeds": {"D": 42, "B": list(B_SEEDS), "C": C_SEED, "SEQ_MAPPING_XGB": 42},
        "smoothing": {"guide": GUIDE_SMOOTHING, "mapping": MAPPING_SMOOTHING},
        "fallback": {"when_mapping_not_found": MISSING_MAPPING_ROUTE},
        "residual_targets": {
            "B": "y_fit - guide_anchor_fit",
            "C": "y_fit - (guide_anchor_fit + template_anchor_fit) / 2",
            "SEQ_MAPPING_XGB": "y_fit - mean(y_fit); add mean(y_fit) to query predictions",
        },
        "components": {
            "D": {name: _snapshot(model) for name, model in d.items()},
            "B": {seed: _snapshot(model) for seed, model in b.items()},
            "C": _snapshot(c),
            "SEQ_MAPPING_XGB": _snapshot(mapping),
        },
        "library_versions": versions,
    }


def _as_matrix(value: Any, name: str, *, expected_rows: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix; got shape {array.shape}")
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ValueError(f"{name} row count {array.shape[0]} != expected {expected_rows}")
    if array.shape[1] == 0:
        raise ValueError(f"{name} has no feature columns")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite feature values")
    return array


def _as_vector(value: Any, name: str, *, expected_rows: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector; got shape {array.shape}")
    if expected_rows is not None and len(array) != expected_rows:
        raise ValueError(f"{name} length {len(array)} != expected {expected_rows}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _bundle_value(bundle: Any, name: str) -> Any:
    if isinstance(bundle, Mapping):
        if name not in bundle:
            raise ValueError(f"feature bundle is missing {name!r}")
        return bundle[name]
    if not hasattr(bundle, name):
        raise ValueError(f"feature bundle is missing {name!r}")
    return getattr(bundle, name)


def _validate_matching_columns(fit_value: Any, query_value: Any, name: str) -> None:
    fit_columns = getattr(fit_value, "columns", None)
    query_columns = getattr(query_value, "columns", None)
    if fit_columns is None and query_columns is None:
        return
    if fit_columns is None or query_columns is None:
        raise ValueError(f"{name} fit/query matrices must use the same representation")
    if not fit_columns.equals(query_columns):
        raise ValueError(f"{name} fit/query columns must match in both names and order")


def _query_mapping_found(bundle: Any, n_query: int) -> np.ndarray:
    diagnostics = _bundle_value(bundle, "query_diagnostics")
    if hasattr(diagnostics, "columns") and hasattr(diagnostics, "__getitem__"):
        if "mapping_found" not in diagnostics.columns:
            raise ValueError("query_diagnostics must contain schema field 'mapping_found'")
        values = diagnostics["mapping_found"].to_numpy()
    elif isinstance(diagnostics, Mapping):
        if "mapping_found" not in diagnostics:
            raise ValueError("query_diagnostics must contain schema field 'mapping_found'")
        values = diagnostics["mapping_found"]
    else:
        raise ValueError("query_diagnostics must be a DataFrame or mapping with mapping_found")
    found = np.asarray(values)
    if found.ndim != 1 or len(found) != n_query:
        raise ValueError("query_diagnostics.mapping_found must align one-to-one with query rows")
    if pd_isna(found).any():
        raise ValueError("query_diagnostics.mapping_found contains missing values")
    if not np.isin(found, (True, False, 0, 1)).all():
        raise ValueError("query_diagnostics.mapping_found must be boolean")
    return found.astype(bool)


def pd_isna(values: np.ndarray) -> np.ndarray:
    """Null check without importing pandas into the model module."""
    try:
        return np.asarray([bool(value is None or value != value) for value in values])
    except (TypeError, ValueError):
        return np.asarray([value is None for value in values], dtype=bool)


def _copy_diagnostics(bundle: Any, n_query: int) -> Any:
    diagnostics = _bundle_value(bundle, "query_diagnostics")
    if len(diagnostics) != n_query:
        raise ValueError("query_diagnostics row count does not align with query matrices")
    return diagnostics.copy() if hasattr(diagnostics, "copy") else diagnostics


def _compose_predictions(
    *,
    d: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    history_prior: np.ndarray,
    mapping_found: np.ndarray,
    mapping_prediction: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n = len(d)
    if any(len(values) != n for values in (b, c, history_prior, mapping_found)):
        raise ValueError("component predictions and mapping diagnostics must have equal length")
    tri_unrouted = 0.20 * d + 0.47 * b + 0.33 * c
    db_unrouted = (0.20 * d + 0.47 * b) / 0.67
    dc_unrouted = (0.20 * d + 0.33 * c) / 0.53
    bc_unrouted = (0.47 * b + 0.33 * c) / 0.80
    routed_fallback = np.where(mapping_found, "context", "sequence_fallback")
    predictions: dict[str, np.ndarray] = {
        "D": d,
        "B": b,
        "C": c,
        "TriCasRank": np.where(mapping_found, tri_unrouted, d),
        "HISTORY_PRIOR": history_prior,
        "D_B_FIXED": np.where(mapping_found, db_unrouted, d),
        "D_C_FIXED": np.where(mapping_found, dc_unrouted, d),
        "B_C_FIXED": np.where(mapping_found, bc_unrouted, d),
        "WITHOUT_D": np.where(mapping_found, bc_unrouted, d),
        "WITHOUT_B": np.where(mapping_found, dc_unrouted, d),
        "WITHOUT_C": np.where(mapping_found, db_unrouted, d),
        "TriCasRank_UNROUTED": tri_unrouted,
        "D_B_FIXED_UNROUTED": db_unrouted,
        "D_C_FIXED_UNROUTED": dc_unrouted,
        "B_C_FIXED_UNROUTED": bc_unrouted,
    }
    routes: dict[str, np.ndarray] = {
        "D": np.full(n, "sequence", dtype=object),
        "B": np.full(n, "context", dtype=object),
        "C": np.full(n, "context", dtype=object),
        "TriCasRank": routed_fallback.copy(),
        "HISTORY_PRIOR": np.full(n, "history_prior", dtype=object),
        "D_B_FIXED": routed_fallback.copy(),
        "D_C_FIXED": routed_fallback.copy(),
        "B_C_FIXED": routed_fallback.copy(),
        "WITHOUT_D": routed_fallback.copy(),
        "WITHOUT_B": routed_fallback.copy(),
        "WITHOUT_C": routed_fallback.copy(),
        "TriCasRank_UNROUTED": np.full(n, "context", dtype=object),
        "D_B_FIXED_UNROUTED": np.full(n, "context", dtype=object),
        "D_C_FIXED_UNROUTED": np.full(n, "context", dtype=object),
        "B_C_FIXED_UNROUTED": np.full(n, "context", dtype=object),
    }
    if mapping_prediction is not None:
        predictions["SEQ_MAPPING_XGB"] = mapping_prediction
        # Schema route is context even when a row has no matching mapping.
        routes["SEQ_MAPPING_XGB"] = np.full(n, "context", dtype=object)
    return predictions, routes


def _predict_components(components: Mapping[str, Any], bundle: Any) -> dict[str, Any]:
    d_query = _as_matrix(_bundle_value(bundle, "d_query"), "d_query")
    n_query = d_query.shape[0]
    context_query = _as_matrix(
        _bundle_value(bundle, "context_query"), "context_query", expected_rows=n_query
    )
    guide_query = _as_vector(
        _bundle_value(bundle, "guide_anchor_query"), "guide_anchor_query", expected_rows=n_query
    )
    template_query = _as_vector(
        _bundle_value(bundle, "template_anchor_query"),
        "template_anchor_query",
        expected_rows=n_query,
    )
    mapping_found = _query_mapping_found(bundle, n_query)

    d_parts = {
        name: _as_vector(model.predict(d_query), f"D.{name} prediction", expected_rows=n_query)
        for name, model in components["D"].items()
    }
    if set(d_parts) != set(D_WEIGHTS):
        raise ValueError(f"saved D components must be exactly {sorted(D_WEIGHTS)}")
    d = sum(D_WEIGHTS[name] * d_parts[name] for name in D_WEIGHTS)
    b_parts = {
        seed: _as_vector(
            model.predict(context_query), f"B.{seed} residual prediction", expected_rows=n_query
        )
        for seed, model in components["B"].items()
    }
    if set(b_parts) != {str(seed) for seed in B_SEEDS}:
        raise ValueError("saved B components must contain the five frozen seeds")
    b = guide_query + np.mean(np.column_stack([b_parts[str(seed)] for seed in B_SEEDS]), axis=1)
    c_residual = _as_vector(
        components["C"].predict(context_query), "C residual prediction", expected_rows=n_query
    )
    c = (guide_query + template_query) / 2.0 + c_residual
    history_prior = (guide_query + template_query) / 2.0

    mapping_prediction = None
    if "SEQ_MAPPING_XGB" in components:
        mapping_query = _as_matrix(
            _bundle_value(bundle, "seq_mapping_query"), "seq_mapping_query", expected_rows=n_query
        )
        center = float(components["SEQ_MAPPING_XGB"]["fit_target_mean"])
        mapping_prediction = center + _as_vector(
            components["SEQ_MAPPING_XGB"]["estimator"].predict(mapping_query),
            "SEQ_MAPPING_XGB residual prediction",
            expected_rows=n_query,
        )
    predictions, routes = _compose_predictions(
        d=d,
        b=b,
        c=c,
        history_prior=history_prior,
        mapping_found=mapping_found,
        mapping_prediction=mapping_prediction,
    )
    for model_id, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{model_id} prediction contains non-finite values")
    return {
        "predictions": predictions,
        "prediction_routes": routes,
        "diagnostics": _copy_diagnostics(bundle, n_query),
    }


def predict_components(components: Mapping[str, Any], bundle: Any) -> dict[str, Any]:
    """Predict from already-fitted components without fitting or refitting."""
    return _predict_components(components, bundle)


def fit_predict(
    bundle: Any,
    y_fit: Any,
    *,
    include_mapping_baseline: bool = True,
    return_models: bool = False,
) -> dict[str, Any]:
    """Fit the frozen D/B/C components once and derive all comparisons.

    The supplied feature bundle must carry train-only matrices and anchors.
    No search, model-specific feature transformation, or query-label access is
    performed here.
    """
    y = _as_vector(y_fit, "y_fit")
    d_fit_value = _bundle_value(bundle, "d_fit")
    d_query_value = _bundle_value(bundle, "d_query")
    context_fit_value = _bundle_value(bundle, "context_fit")
    context_query_value = _bundle_value(bundle, "context_query")
    _validate_matching_columns(d_fit_value, d_query_value, "D")
    _validate_matching_columns(context_fit_value, context_query_value, "context")
    d_fit = _as_matrix(d_fit_value, "d_fit", expected_rows=len(y))
    context_fit = _as_matrix(context_fit_value, "context_fit", expected_rows=len(y))
    d_query = _as_matrix(d_query_value, "d_query")
    context_query = _as_matrix(
        context_query_value,
        "context_query",
        expected_rows=d_query.shape[0],
    )
    if d_fit.shape[1] != d_query.shape[1]:
        raise ValueError("d_fit and d_query must have the same ordered feature count")
    if context_fit.shape[1] != context_query.shape[1]:
        raise ValueError("context_fit and context_query must have the same ordered feature count")
    guide_fit = _as_vector(
        _bundle_value(bundle, "guide_anchor_fit"), "guide_anchor_fit", expected_rows=len(y)
    )
    template_fit = _as_vector(
        _bundle_value(bundle, "template_anchor_fit"),
        "template_anchor_fit",
        expected_rows=len(y),
    )
    _as_vector(
        _bundle_value(bundle, "guide_anchor_query"),
        "guide_anchor_query",
        expected_rows=d_query.shape[0],
    )
    _as_vector(
        _bundle_value(bundle, "template_anchor_query"),
        "template_anchor_query",
        expected_rows=d_query.shape[0],
    )
    _query_mapping_found(bundle, d_query.shape[0])

    components: dict[str, Any] = {"D": {}, "B": {}, "C": None}
    for name, model in _new_d_models().items():
        model.fit(d_fit, y)
        components["D"][name] = model

    for seed in B_SEEDS:
        model = _new_residual_xgb(seed=seed, min_child_weight=8)
        model.fit(context_fit, y - guide_fit)
        components["B"][str(seed)] = model

    c_model = _new_residual_xgb(seed=C_SEED, min_child_weight=2)
    c_model.fit(context_fit, y - (guide_fit + template_fit) / 2.0)
    components["C"] = c_model

    if include_mapping_baseline:
        mapping_fit_value = _bundle_value(bundle, "seq_mapping_fit")
        mapping_query_value = _bundle_value(bundle, "seq_mapping_query")
        _validate_matching_columns(mapping_fit_value, mapping_query_value, "seq_mapping")
        mapping_fit = _as_matrix(
            mapping_fit_value, "seq_mapping_fit", expected_rows=len(y)
        )
        mapping_query = _as_matrix(
            mapping_query_value,
            "seq_mapping_query",
            expected_rows=d_query.shape[0],
        )
        if mapping_fit.shape[1] != mapping_query.shape[1]:
            raise ValueError("seq_mapping_fit and seq_mapping_query must have the same ordered feature count")
        target_mean = float(np.mean(y))
        mapping_model = _new_residual_xgb(seed=42, min_child_weight=8)
        mapping_model.fit(mapping_fit, y - target_mean)
        components["SEQ_MAPPING_XGB"] = {
            "estimator": mapping_model,
            "fit_target_mean": target_mean,
        }

    output = _predict_components(components, bundle)
    output["fit_target_n"] = len(y)
    output["resolved_recipe"] = resolved_recipe()
    if return_models:
        output["models"] = components
    return output


__all__ = [
    "B_SEEDS",
    "C_SEED",
    "D_WEIGHTS",
    "FUSION_WEIGHTS",
    "GUIDE_SMOOTHING",
    "MAPPING_SMOOTHING",
    "XGB_PARAMS",
    "fit_predict",
    "predict_components",
    "resolved_recipe",
]
