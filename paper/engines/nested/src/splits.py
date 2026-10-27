"""Identity-checked outer/inner grouped splits for R03."""

from __future__ import annotations

from typing import Any

import pandas as pd


def make_nested_splits(frame: pd.DataFrame, protocol_id: str, outer_fold: int) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    group_column = f"{protocol_id}_group"
    fold_column = f"{protocol_id}_fold"
    missing = sorted({"record_id", group_column, fold_column} - set(frame.columns))
    if missing:
        raise ValueError(f"Nested split frame is missing fields: {missing}")
    if frame.record_id.isna().any() or frame[group_column].isna().any() or frame[fold_column].isna().any():
        raise ValueError("Nested split frame contains missing ID, group, or fold values")
    ids = frame.record_id.astype(str)
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Nested split frame needs unique nonempty record IDs")
    folds = pd.to_numeric(frame[fold_column], errors="raise").astype(int)
    if sorted(folds.unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected fixed five-fold assignment 0..4; got {sorted(folds.unique().tolist())}")
    if int(outer_fold) not in range(5):
        raise ValueError("outer_fold must be in 0..4")
    group_fold_count = pd.DataFrame({"group": frame[group_column].astype(str), "fold": folds}).groupby("group").fold.nunique()
    if group_fold_count.gt(1).any():
        raise ValueError(f"{protocol_id} isolation groups cross assigned folds")

    normalized = frame.copy().reset_index(drop=True)
    normalized[fold_column] = folds.to_numpy()
    outer_query = normalized.loc[normalized[fold_column].eq(int(outer_fold))].reset_index(drop=True)
    outer_fit = normalized.loc[~normalized[fold_column].eq(int(outer_fold))].reset_index(drop=True)
    _assert_disjoint(outer_fit, outer_query, group_column, "outer")

    inner_splits = []
    for inner_fold in [fold for fold in range(5) if fold != int(outer_fold)]:
        validation = normalized.loc[normalized[fold_column].eq(inner_fold)].reset_index(drop=True)
        fit = normalized.loc[~normalized[fold_column].isin({int(outer_fold), inner_fold})].reset_index(drop=True)
        _assert_disjoint(fit, validation, group_column, f"inner {inner_fold}")
        if set(outer_query.record_id.astype(str)) & set(fit.record_id.astype(str)):
            raise ValueError(f"Outer query ID leaked into inner {inner_fold} fit")
        if set(outer_query.record_id.astype(str)) & set(validation.record_id.astype(str)):
            raise ValueError(f"Outer query ID leaked into inner {inner_fold} validation")
        if set(outer_query[group_column].astype(str)) & set(fit[group_column].astype(str)):
            raise ValueError(f"Outer query group leaked into inner {inner_fold} fit")
        if set(outer_query[group_column].astype(str)) & set(validation[group_column].astype(str)):
            raise ValueError(f"Outer query group leaked into inner {inner_fold} validation")
        inner_splits.append({"inner_fold": inner_fold, "fit": fit, "validation": validation})
    return outer_fit, outer_query, inner_splits


def _assert_disjoint(fit: pd.DataFrame, query: pd.DataFrame, group_column: str, label: str) -> None:
    fit_ids, query_ids = set(fit.record_id.astype(str)), set(query.record_id.astype(str))
    if fit_ids & query_ids:
        raise ValueError(f"Record IDs overlap in {label} split")
    fit_groups, query_groups = set(fit[group_column].astype(str)), set(query[group_column].astype(str))
    if fit_groups & query_groups:
        raise ValueError(f"Isolation groups overlap in {label} split")


__all__ = ["make_nested_splits"]
