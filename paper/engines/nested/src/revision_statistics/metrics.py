"""Metric definitions shared by the runner and the final analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata


def _as_finite_pair(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.size != pred.size:
        raise ValueError(f"y_true and y_pred have different lengths ({true.size} != {pred.size})")
    if true.size == 0:
        raise ValueError("metrics require at least one observation")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("metrics require finite y_true and y_pred values")
    return true, pred


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Pearson correlation, or None when undefined by a constant vector."""
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def basic_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | None]:
    """Compute SCC, PCC, RMSE, MAE and R² on the supplied processed-label scale.

    Spearman correlation uses average ranks for ties. Undefined correlations are
    represented as ``None`` (rather than zero); R² is also ``None`` when the
    observed labels are constant, for which its denominator is zero.
    Non-finite input is rejected so that incomplete predictions cannot be
    silently scored on a reduced subset.
    """
    true, pred = _as_finite_pair(y_true, y_pred)
    scc = _correlation(rankdata(true, method="average"), rankdata(pred, method="average"))
    pcc = _correlation(true, pred)
    residual = pred - true
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    total = float(np.sum(np.square(true - np.mean(true))))
    r2 = None if total == 0 else float(1.0 - np.sum(np.square(residual)) / total)
    return {"SCC": scc, "PCC": pcc, "RMSE": rmse, "MAE": mae, "R2": r2}
