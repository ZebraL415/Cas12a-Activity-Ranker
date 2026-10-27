"""Exact protocol-group resampling utilities for paired OOF predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


def _paired_arrays(
    y_true: Sequence[float], predictions: Mapping[str, Sequence[float]], group_ids: Sequence[object]
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, int]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    groups = np.asarray(group_ids, dtype=str).reshape(-1)
    pred = {name: np.asarray(values, dtype=float).reshape(-1) for name, values in predictions.items()}
    if not pred:
        raise ValueError("at least one prediction vector is required")
    if len(y) == 0 or len(groups) != len(y) or any(len(values) != len(y) for values in pred.values()):
        raise ValueError("truth, prediction, and group vectors must have the same non-zero length")
    if not np.isfinite(y).all() or any(not np.isfinite(values).all() for values in pred.values()):
        raise ValueError("resampling requires finite truth and prediction values")
    _, group_codes = np.unique(groups, return_inverse=True)
    return y, pred, group_codes, int(group_codes.max()) + 1


def _sample_group_counts(rng: np.random.Generator, n_draws: int, n_groups: int) -> np.ndarray:
    """Draw n_groups clusters with replacement, represented as multiplicities."""
    selected = rng.integers(0, n_groups, size=(n_draws, n_groups), dtype=np.int32)
    counts = np.stack([np.bincount(row, minlength=n_groups) for row in selected])
    return counts.astype(np.uint16, copy=False)


def _weighted_average_ranks(values: np.ndarray, row_weights: np.ndarray) -> np.ndarray:
    """Average ranks in each expanded weighted sample, without materializing repeats."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    tie_sizes = np.diff(np.r_[starts, values.size])
    tie_codes = np.repeat(np.arange(starts.size), tie_sizes)
    weights_sorted = row_weights[:, order]
    tie_weights = np.add.reduceat(weights_sorted, starts, axis=1)
    cumulative = np.cumsum(tie_weights, axis=1, dtype=np.float64)
    # For a tie with frequency f, the average 1-based rank is cum_f - (f - 1) / 2.
    tie_ranks = cumulative - (tie_weights - 1.0) / 2.0
    ranks_sorted = tie_ranks[:, tie_codes]
    ranks = np.empty_like(ranks_sorted)
    ranks[:, order] = ranks_sorted
    return ranks


def _weighted_row_correlations(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    totals = np.sum(weights, axis=1)
    safe_totals = np.where(totals > 0, totals, 1.0)
    mean_x = np.sum(weights * x, axis=1) / safe_totals
    mean_y = np.sum(weights * y, axis=1) / safe_totals
    centered_x = x - mean_x[:, None]
    centered_y = y - mean_y[:, None]
    covariance = np.sum(weights * centered_x * centered_y, axis=1)
    variance_x = np.sum(weights * np.square(centered_x), axis=1)
    variance_y = np.sum(weights * np.square(centered_y), axis=1)
    denominator = np.sqrt(variance_x * variance_y)
    result = np.full(totals.shape, np.nan, dtype=float)
    valid = (totals > 0) & (denominator > 0)
    result[valid] = covariance[valid] / denominator[valid]
    return result


def _weighted_scc_draws(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    row_weights: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    y_rank = _weighted_average_ranks(y_true, row_weights)
    totals = np.sum(row_weights, axis=1)
    safe_totals = np.where(totals > 0, totals, 1.0)
    y_mean = np.sum(row_weights * y_true, axis=1) / safe_totals
    y_centered = y_true[None, :] - y_mean[:, None]
    y_ss = np.sum(row_weights * np.square(y_centered), axis=1)
    output: dict[str, dict[str, np.ndarray]] = {}
    for name, values in predictions.items():
        scc = _weighted_row_correlations(y_rank, _weighted_average_ranks(values, row_weights), row_weights)
        pcc = _weighted_row_correlations(
            np.broadcast_to(y_true, row_weights.shape),
            np.broadcast_to(values, row_weights.shape),
            row_weights,
        )
        residual = values[None, :] - y_true[None, :]
        squared_error = np.sum(row_weights * np.square(residual), axis=1)
        rmse = np.sqrt(squared_error / safe_totals)
        mae = np.sum(row_weights * np.abs(residual), axis=1) / safe_totals
        r2 = np.full(totals.shape, np.nan, dtype=float)
        r2_valid = (totals > 0) & (y_ss > 0)
        r2[r2_valid] = 1.0 - squared_error[r2_valid] / y_ss[r2_valid]
        output[name] = {"SCC": scc, "PCC": pcc, "RMSE": rmse, "MAE": mae, "R2": r2}
    return output


def bootstrap_metric_draws(
    y_true: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    group_ids: Sequence[object],
    *,
    draws: int = 2_000,
    seed: int = 20261012,
    chunk_size: int = 64,
) -> dict[str, dict[str, np.ndarray]]:
    """Return group-bootstrap metric draws, sharing samples across all models.

    Whole groups are sampled with replacement, with the original number of groups
    drawn each time. Repeated groups retain all member records. Weighted average
    ranks are algebraically identical to ranking the explicitly expanded sample.
    ``NaN`` marks a draw with an undefined correlation.
    """
    y, pred, group_codes, n_groups = _paired_arrays(y_true, predictions, group_ids)
    if draws <= 0 or chunk_size <= 0:
        raise ValueError("draws and chunk_size must be positive")
    rng = np.random.default_rng(seed)
    metric_names = ("SCC", "PCC", "RMSE", "MAE", "R2")
    output = {
        name: {metric: np.full(draws, np.nan, dtype=float) for metric in metric_names}
        for name in pred
    }
    for begin in range(0, draws, chunk_size):
        end = min(begin + chunk_size, draws)
        group_counts = _sample_group_counts(rng, end - begin, n_groups)
        row_weights = group_counts[:, group_codes].astype(np.float64, copy=False)
        scores = _weighted_scc_draws(y, pred, row_weights)
        for name, metrics in scores.items():
            for metric, values in metrics.items():
                output[name][metric][begin:end] = values
    return output


def bootstrap_scc_draws(
    y_true: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    group_ids: Sequence[object],
    *,
    draws: int = 2_000,
    seed: int = 20261012,
    chunk_size: int = 64,
) -> dict[str, np.ndarray]:
    """Compatibility helper returning only the SCC component of each model."""
    return {
        name: metrics["SCC"]
        for name, metrics in bootstrap_metric_draws(
            y_true,
            predictions,
            group_ids,
            draws=draws,
            seed=seed,
            chunk_size=chunk_size,
        ).items()
    }


def percentile_interval(
    draws: Sequence[float], *, minimum_valid_fraction: float = 0.95
) -> dict[str, float | int | str | None]:
    values = np.asarray(draws, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one bootstrap draw is required")
    if not 0 <= minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be between 0 and 1")
    valid = values[np.isfinite(values)]
    fraction = float(valid.size / values.size)
    result: dict[str, float | int | str | None] = {
        "lower": None,
        "upper": None,
        "valid_draws": int(valid.size),
        "total_draws": int(values.size),
        "valid_fraction": fraction,
        "status": "insufficient_valid_draws",
    }
    if fraction < minimum_valid_fraction:
        return result
    if valid.size < 2:
        result["status"] = "fewer_than_two_valid_draws"
        return result
    result["lower"] = float(np.quantile(valid, 0.025))
    result["upper"] = float(np.quantile(valid, 0.975))
    result["status"] = "available"
    return result


def _rowwise_correlations(y_rank: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    pred_rank = rankdata(predictions, method="average", axis=1)
    y_centered = y_rank - np.mean(y_rank)
    pred_centered = pred_rank - np.mean(pred_rank, axis=1, keepdims=True)
    numerator = pred_centered @ y_centered
    denom = np.sqrt(np.sum(np.square(y_centered)) * np.sum(np.square(pred_centered), axis=1))
    result = np.full(denom.shape, np.nan, dtype=float)
    valid = denom > 0
    result[valid] = numerator[valid] / denom[valid]
    return result


def make_group_swap_flags(n_groups: int, *, draws: int = 2_000, seed: int = 20261012) -> np.ndarray:
    """Create reusable group-level swap assignments for paired contrasts."""
    if n_groups <= 0 or draws <= 0:
        raise ValueError("n_groups and draws must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(draws, n_groups), dtype=np.int8).astype(bool)


def prediction_swap_test(
    y_true: Sequence[float],
    prediction_a: Sequence[float],
    prediction_b: Sequence[float],
    group_ids: Sequence[object],
    *,
    draws: int = 2_000,
    seed: int = 20261012,
    chunk_size: int = 32,
    swap_flags: np.ndarray | None = None,
) -> dict[str, float | int | None | str]:
    """Two-sided protocol-group prediction-swap test for the SCC difference.

    Each draw independently swaps the two methods' predictions for each whole
    group, leaving labels fixed. The reported P value uses the add-one correction.
    """
    y, pred, group_codes, n_groups = _paired_arrays(
        y_true, {"a": prediction_a, "b": prediction_b}, group_ids
    )
    if draws <= 0 or chunk_size <= 0:
        raise ValueError("draws and chunk_size must be positive")
    a, b = pred["a"], pred["b"]
    y_rank = rankdata(y, method="average")
    observed_a = _rowwise_correlations(y_rank, a.reshape(1, -1))[0]
    observed_b = _rowwise_correlations(y_rank, b.reshape(1, -1))[0]
    observed = float(observed_a - observed_b) if np.isfinite(observed_a) and np.isfinite(observed_b) else np.nan
    if not np.isfinite(observed):
        return {
            "p_value": None,
            "observed_delta": None,
            "valid_draws": 0,
            "total_draws": int(draws),
            "status": "undefined_observed_scc",
        }

    if swap_flags is None:
        flags = make_group_swap_flags(n_groups, draws=draws, seed=seed)
    else:
        flags = np.asarray(swap_flags, dtype=bool)
        if flags.shape != (draws, n_groups):
            raise ValueError(f"swap_flags must have shape {(draws, n_groups)}, got {flags.shape}")
    permutation_delta = np.full(draws, np.nan, dtype=float)
    for begin in range(0, draws, chunk_size):
        end = min(begin + chunk_size, draws)
        swap_by_group = flags[begin:end]
        swap_rows = swap_by_group[:, group_codes]
        perm_a = np.where(swap_rows, b[None, :], a[None, :])
        perm_b = np.where(swap_rows, a[None, :], b[None, :])
        corr_a = _rowwise_correlations(y_rank, perm_a)
        corr_b = _rowwise_correlations(y_rank, perm_b)
        permutation_delta[begin:end] = corr_a - corr_b
    valid = permutation_delta[np.isfinite(permutation_delta)]
    if valid.size != draws:
        return {
            "p_value": None,
            "observed_delta": observed,
            "valid_draws": int(valid.size),
            "total_draws": int(draws),
            "status": "undefined_permutation_scc",
        }
    exceedances = int(np.count_nonzero(np.abs(valid) >= abs(observed) - 1e-12))
    p_value = float((exceedances + 1) / (draws + 1))
    return {
        "p_value": p_value,
        "observed_delta": observed,
        "valid_draws": int(valid.size),
        "total_draws": int(draws),
        "status": "available",
    }


def holm_adjust(
    p_values: Sequence[float | None], *, family_size: int | None = None
) -> list[float | None]:
    """Holm-adjust a fixed family; missing tests occupy ranks as p=1, not omitted."""
    raw = list(p_values)
    target_size = len(raw) if family_size is None else int(family_size)
    if target_size < len(raw) or target_size < 0:
        raise ValueError("family_size cannot be smaller than the supplied p-value list")
    finite_values: list[float | None] = []
    for value in raw:
        if value is None:
            finite_values.append(None)
            continue
        number = float(value)
        if not np.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"p values must be finite and in [0, 1], got {value!r}")
        finite_values.append(number)
    effective = [1.0 if value is None else value for value in finite_values]
    effective.extend([1.0] * (target_size - len(effective)))
    order = np.argsort(effective, kind="stable")
    sorted_values = np.asarray(effective, dtype=float)[order]
    adjusted_sorted = np.maximum.accumulate((target_size - np.arange(target_size)) * sorted_values)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(target_size, dtype=float)
    adjusted[order] = adjusted_sorted
    return [None if value is None else float(adjusted[i]) for i, value in enumerate(finite_values)]
