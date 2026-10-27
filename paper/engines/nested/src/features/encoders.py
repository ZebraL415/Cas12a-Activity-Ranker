"""Fit-partition-only history encoders used by the fixed B/C recipe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


GUIDE_COLUMN = "crRNA_sequence"
LABEL_COLUMNS = ("label_normalized", "y_true")
GUIDE_SMOOTHING = 0.5
MAPPING_SMOOTHING = 20.0
MAPPING_CATEGORICAL = (
    "mapping_candidate_group_ids",
    "mapping_status",
    "mapping_confidence",
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


def _category_columns(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    diffcat = pd.Series(
        np.where(
            frame["aligned_difference_count"].eq(0),
            "exact",
            np.where(frame["aligned_difference_count"].eq(1), "one", "two_plus"),
        ),
        index=frame.index,
    )
    gaps = frame["gap_count_target"].fillna(0).add(frame["gap_count_guide"].fillna(0))
    gapcat = pd.Series(np.where(gaps.gt(0), "gap", "no_gap"), index=frame.index)
    return diffcat, gapcat


def _map_single_stats(keys: pd.Series, stats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sums = keys.map(stats["sum"]).fillna(0).to_numpy(float)
    counts = keys.map(stats["count"]).fillna(0).to_numpy(float)
    return sums, counts


def _map_multi_stats(
    first: pd.Series, second: pd.Series, stats: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    index = pd.MultiIndex.from_arrays([first.to_numpy(), second.to_numpy()])
    sums = stats["sum"].reindex(index).fillna(0).to_numpy(float)
    counts = stats["count"].reindex(index).fillna(0).to_numpy(float)
    return sums, counts


def guide_target_encoding(
    fit_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    *,
    smoothing: float = GUIDE_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Guide LOO for fit rows; current-fit lookup for query rows.

    The conditional references and category definitions port
    ``run_guide_architecture.py::make_guide_target_encoding``.
    """
    if smoothing <= 0:
        raise ValueError("guide smoothing must be positive")
    required = {GUIDE_COLUMN, "aligned_difference_count", "gap_count_target", "gap_count_guide"}
    missing = sorted(required - set(fit_frame.columns))
    if missing:
        raise ValueError(f"Fit frame is missing guide-history inputs: {missing}")
    missing_query = sorted(required - set(query_frame.columns))
    if missing_query:
        raise ValueError(f"Query frame is missing guide-history inputs: {missing_query}")
    label_column = next((name for name in LABEL_COLUMNS if name in fit_frame.columns), None)
    if label_column is None:
        raise ValueError("Fit frame must contain label_normalized or y_true for history encoding")

    train = fit_frame.reset_index(drop=True)
    query = query_frame.reset_index(drop=True)
    y = pd.to_numeric(train[label_column], errors="coerce").to_numpy(float)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError("Fit labels for guide encoding must be non-empty and finite")
    n = len(y)
    if n < 2:
        raise ValueError("Guide LOO requires at least two fit records")
    global_sum = float(np.sum(y))
    global_mean = float(np.mean(y))
    global_loo = (global_sum - y) / (n - 1)
    train_guide = train[GUIDE_COLUMN].astype(str)
    query_guide = query[GUIDE_COLUMN].astype(str)
    train_diffcat, train_gapcat = _category_columns(train)
    query_diffcat, query_gapcat = _category_columns(query)

    all_stats = pd.DataFrame({"guide": train_guide, "y": y}).groupby("guide", sort=False)["y"].agg(["sum", "count"])
    train_sum, train_count = _map_single_stats(train_guide, all_stats)
    query_sum, query_count = _map_single_stats(query_guide, all_stats)
    count_excluding_self = train_count - 1
    guide_all_train = (train_sum - y + smoothing * global_loo) / (count_excluding_self + smoothing)
    guide_all_query = (query_sum + smoothing * global_mean) / (query_count + smoothing)

    train_features: dict[str, np.ndarray] = {
        "te_guide_all": guide_all_train,
        "te_guide_all_log_count": np.log1p(count_excluding_self),
        "te_guide_seen": (count_excluding_self > 0).astype(float),
    }
    query_features: dict[str, np.ndarray] = {
        "te_guide_all": guide_all_query,
        "te_guide_all_log_count": np.log1p(query_count),
        "te_guide_seen": (query_count > 0).astype(float),
    }

    def reference_feature(name: str, member_mask: np.ndarray) -> None:
        stats = (
            pd.DataFrame({"guide": train_guide.loc[member_mask].to_numpy(), "y": y[member_mask]})
            .groupby("guide", sort=False)["y"]
            .agg(["sum", "count"])
        )
        tr_sum, tr_count = _map_single_stats(train_guide, stats)
        q_sum, q_count = _map_single_stats(query_guide, stats)
        member = member_mask.astype(float)
        count_excluding_self = tr_count - member
        train_features[f"te_guide_{name}_ref"] = (
            tr_sum - member * y + smoothing * guide_all_train
        ) / (count_excluding_self + smoothing)
        train_features[f"te_guide_{name}_log_count"] = np.log1p(count_excluding_self)
        query_features[f"te_guide_{name}_ref"] = (q_sum + smoothing * guide_all_query) / (q_count + smoothing)
        query_features[f"te_guide_{name}_log_count"] = np.log1p(q_count)

    exact_mask = train["aligned_difference_count"].eq(0).to_numpy()
    reference_feature("exact", exact_mask)
    reference_feature("nonexact", ~exact_mask)

    def conditional_feature(name: str, train_category: pd.Series, query_category: pd.Series) -> None:
        fit_keys = pd.MultiIndex.from_arrays([train_guide.to_numpy(), train_category.to_numpy()])
        stats = pd.DataFrame({"y": y}, index=fit_keys).groupby(level=[0, 1], sort=False)["y"].agg(["sum", "count"])
        tr_sum, tr_count = _map_multi_stats(train_guide, train_category, stats)
        q_sum, q_count = _map_multi_stats(query_guide, query_category, stats)
        count_excluding_self = tr_count - 1
        train_features[f"te_guide_{name}"] = (
            tr_sum - y + smoothing * guide_all_train
        ) / (count_excluding_self + smoothing)
        train_features[f"te_guide_{name}_log_count"] = np.log1p(count_excluding_self)
        query_features[f"te_guide_{name}"] = (q_sum + smoothing * guide_all_query) / (q_count + smoothing)
        query_features[f"te_guide_{name}_log_count"] = np.log1p(q_count)

    conditional_feature("diffcat", train_diffcat, query_diffcat)
    conditional_feature("gapcat", train_gapcat, query_gapcat)

    encoded_fit = pd.DataFrame(train_features, index=fit_frame.index)
    encoded_query = pd.DataFrame(query_features, index=query_frame.index)
    if not np.isfinite(encoded_fit.to_numpy(float)).all() or not np.isfinite(encoded_query.to_numpy(float)).all():
        raise RuntimeError("Guide history encoding produced non-finite values")
    return encoded_fit, encoded_query


def normalized_mapping_key(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("missing").astype(str)


def _lookup_encoding(
    source_key: pd.Series,
    query_key: pd.Series,
    source_y: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(source_y):
        raise ValueError("Mapping target encoding needs at least one reference record")
    stats = pd.DataFrame({"key": source_key.astype(str).to_numpy(), "y": source_y}).groupby(
        "key", sort=False
    )["y"].agg(["sum", "count"])
    query_keys = query_key.astype(str)
    sums = query_keys.map(stats["sum"]).fillna(0).to_numpy(float)
    counts = query_keys.map(stats["count"]).fillna(0).to_numpy(float)
    prior = float(np.mean(source_y))
    return (sums + smoothing * prior) / (counts + smoothing), np.log1p(counts)


def mapping_target_encoding(
    fit_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    fit_mapping: pd.DataFrame,
    query_mapping: pd.DataFrame,
    encoding_fold_by_record_id: Mapping[str, Any] | pd.Series,
    *,
    smoothing: float = MAPPING_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit fit-row mapping histories by explicit current-protocol folds."""
    if smoothing <= 0:
        raise ValueError("mapping smoothing must be positive")
    label_column = next((name for name in LABEL_COLUMNS if name in fit_frame.columns), None)
    if label_column is None:
        raise ValueError("Fit frame must contain label_normalized or y_true for mapping encoding")
    if "record_id" not in fit_frame or "record_id" not in query_frame:
        raise ValueError("Fit and query frames must contain record_id")
    fit_ids = fit_frame["record_id"].astype(str)
    fold_map = encoding_fold_by_record_id
    if isinstance(fold_map, pd.Series):
        if fold_map.index.name == "record_id" or fold_map.index.is_unique:
            fold_values = fold_map.copy()
        else:
            raise ValueError("encoding fold Series must have a unique index")
    else:
        fold_values = pd.Series(dict(fold_map))
    fold_values.index = fold_values.index.astype(str)
    if not fold_values.index.is_unique:
        raise ValueError("encoding_fold_by_record_id contains duplicate record IDs")
    folds = fit_ids.map(fold_values)
    if folds.isna().any():
        missing = fit_ids.loc[folds.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Encoding folds are missing fit record IDs: {missing}")
    folds = folds.reset_index(drop=True)
    unique_folds = list(pd.unique(folds))
    if len(unique_folds) < 2:
        raise ValueError("Mapping cross-fit requires at least two explicit encoding folds")

    y = pd.to_numeric(fit_frame[label_column], errors="coerce").to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Fit labels for mapping encoding must be finite")
    keys_fit = {column: normalized_mapping_key(fit_mapping, column).reset_index(drop=True) for column in MAPPING_CATEGORICAL}
    keys_query = {column: normalized_mapping_key(query_mapping, column).reset_index(drop=True) for column in MAPPING_CATEGORICAL}
    train_features: dict[str, np.ndarray] = {}
    query_features: dict[str, np.ndarray] = {}

    for column in MAPPING_CATEGORICAL:
        train_value = np.full(len(fit_frame), np.nan, dtype=float)
        train_count = np.full(len(fit_frame), np.nan, dtype=float)
        for fold in unique_folds:
            source_mask = folds.ne(fold).to_numpy()
            query_mask = folds.eq(fold).to_numpy()
            value, count = _lookup_encoding(
                keys_fit[column].loc[source_mask],
                keys_fit[column].loc[query_mask],
                y[source_mask],
                smoothing,
            )
            train_value[query_mask] = value
            train_count[query_mask] = count
        query_value, query_count = _lookup_encoding(keys_fit[column], keys_query[column], y, smoothing)
        train_features[f"mapping_te_{column}"] = train_value
        train_features[f"mapping_te_{column}_log_count"] = train_count
        query_features[f"mapping_te_{column}"] = query_value
        query_features[f"mapping_te_{column}_log_count"] = query_count

    encoded_fit = pd.DataFrame(train_features, index=fit_frame.index)
    encoded_query = pd.DataFrame(query_features, index=query_frame.index)
    if not np.isfinite(encoded_fit.to_numpy(float)).all() or not np.isfinite(encoded_query.to_numpy(float)).all():
        raise RuntimeError("Mapping history cross-fit produced non-finite values")
    return encoded_fit, encoded_query


def mapping_onehot(
    fit_mapping: pd.DataFrame,
    query_mapping: pd.DataFrame,
    columns: tuple[str, ...] = MAPPING_CATEGORICAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_parts: list[pd.DataFrame] = []
    query_parts: list[pd.DataFrame] = []
    for column in columns:
        fit_keys = normalized_mapping_key(fit_mapping, column)
        query_keys = normalized_mapping_key(query_mapping, column)
        fit_dummy = pd.get_dummies(fit_keys, prefix=f"mapping_cat_{column}", dtype=np.uint8)
        query_dummy = pd.get_dummies(query_keys, prefix=f"mapping_cat_{column}", dtype=np.uint8)
        query_dummy = query_dummy.reindex(columns=fit_dummy.columns, fill_value=0)
        fit_parts.append(fit_dummy.set_axis(fit_mapping.index))
        query_parts.append(query_dummy.set_axis(query_mapping.index))
    return pd.concat(fit_parts, axis=1), pd.concat(query_parts, axis=1)


def numeric_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(MAPPING_NUMERIC) - set(frame.columns))
    if missing:
        raise ValueError(f"Mapping frame is missing numeric fields: {missing}")
    result = frame.loc[:, list(MAPPING_NUMERIC)].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    return result


def query_support_counts(
    fit_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    fit_mapping: pd.DataFrame,
    query_mapping: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mapping-found flag and available fit-record support per query row."""
    guide_counts = fit_frame[GUIDE_COLUMN].astype(str).value_counts(dropna=False)
    guide_support = query_frame[GUIDE_COLUMN].astype(str).map(guide_counts).fillna(0).to_numpy(int)

    fit_hit = pd.to_numeric(fit_mapping["mapping_hit_count"], errors="coerce").fillna(0).gt(0)
    query_hit = pd.to_numeric(query_mapping["mapping_hit_count"], errors="coerce").fillna(0).gt(0).to_numpy(bool)
    fit_key = normalized_mapping_key(fit_mapping, "mapping_candidate_template_nos")
    query_key = normalized_mapping_key(query_mapping, "mapping_candidate_template_nos")
    valid_fit = fit_hit & fit_key.ne("") & fit_key.ne("missing")
    valid_query = query_hit & query_key.ne("") & query_key.ne("missing")
    template_counts = fit_key.loc[valid_fit].value_counts(dropna=False)
    template_support = np.where(
        valid_query.to_numpy(),
        query_key.map(template_counts).fillna(0).to_numpy(int),
        0,
    ).astype(int)
    return query_hit, guide_support, template_support
