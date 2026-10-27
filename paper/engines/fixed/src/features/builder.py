"""Fit/query feature construction for the frozen TriCasRank recipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .encoders import (
    GUIDE_COLUMN,
    GUIDE_SMOOTHING,
    LABEL_COLUMNS,
    MAPPING_CATEGORICAL,
    MAPPING_NUMERIC,
    MAPPING_SMOOTHING,
    guide_target_encoding,
    mapping_onehot,
    mapping_target_encoding,
    numeric_mapping,
    query_support_counts,
)
from .mapping import MAPPING_FIELDS, TemplateMapper, mapping_frame_from_input
from .sequence import (
    SequenceManifest,
    build_feature_frame,
    coerce_sequence_manifest,
    pair_onehot,
)


PROTOCOLS = ("target", "guide", "dual_atomic", "atomic_template")
FORBIDDEN_QUERY_LABELS = ("label_normalized", "y_true")


@dataclass
class FeatureState:
    """Serializable preprocessing and fit-reference state; contains no source paths."""

    protocol_id: str
    sequence_manifest: SequenceManifest
    numeric_medians: dict[str, float]
    active_numeric_columns: tuple[str, ...]
    active_pair_columns: tuple[str, ...]
    mapping_onehot_columns: tuple[str, ...]
    global_y_mean: float
    guide_smoothing: float
    mapping_smoothing: float
    guide_lookup: dict[str, dict[Any, tuple[float, int]]]
    mapping_lookup: dict[str, dict[str, tuple[float, int]]]
    template_mapper: TemplateMapper | None
    feature_manifest: dict[str, Any]


@dataclass
class FeatureBundle:
    d_fit: pd.DataFrame
    d_query: pd.DataFrame
    context_fit: pd.DataFrame
    context_query: pd.DataFrame
    seq_mapping_fit: pd.DataFrame
    seq_mapping_query: pd.DataFrame
    guide_anchor_fit: np.ndarray
    guide_anchor_query: np.ndarray
    template_anchor_fit: np.ndarray
    template_anchor_query: np.ndarray
    query_diagnostics: pd.DataFrame
    feature_manifest: dict[str, Any]
    state: FeatureState


def _require_ids(frame: pd.DataFrame, name: str) -> pd.Series:
    if "record_id" not in frame:
        raise ValueError(f"{name} frame must contain record_id")
    ids = frame["record_id"]
    if ids.isna().any():
        raise ValueError(f"{name} frame contains missing record_id")
    ids = ids.astype(str)
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"{name} frame has empty or duplicate record_id")
    return ids


def _query_label_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        {name for name in frame.columns if name in FORBIDDEN_QUERY_LABELS or name.startswith("label_")}
    )


def _labels(fit_frame: pd.DataFrame) -> np.ndarray:
    label_column = next((name for name in LABEL_COLUMNS if name in fit_frame.columns), None)
    if label_column is None:
        raise ValueError("fit_frame must contain label_normalized or y_true")
    values = pd.to_numeric(fit_frame[label_column], errors="coerce").to_numpy(float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("fit labels must be non-empty and finite")
    return values


def _sequence_values(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> tuple[pd.DataFrame, str]:
    if not feature_names:
        raise ValueError("sequence_manifest contains no approved feature names")
    present = [name for name in feature_names if name in frame.columns]
    if len(present) == len(feature_names):
        values = frame.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
        source = "approved_sequence_columns"
    else:
        generated = build_feature_frame(frame, manifest=feature_names)
        values = generated
        if present:
            values.loc[:, present] = frame.loc[:, present].apply(pd.to_numeric, errors="coerce").to_numpy()
            source = f"sequence_pair_plus_{len(present)}_approved_columns"
        else:
            source = "generated_from_25nt_pair"
    return values.replace([np.inf, -np.inf], np.nan), source


def _guide_input_frame(frame: pd.DataFrame, sequence_values: pd.DataFrame) -> pd.DataFrame:
    """Materialize alignment summaries for sequence-only candidate inputs."""
    required = ("aligned_difference_count", "gap_count_target", "gap_count_guide")
    result = frame.copy()
    missing = [column for column in required if column not in result.columns]
    if missing:
        for column in missing:
            if column in sequence_values.columns:
                result[column] = sequence_values[column].to_numpy()
        missing = [column for column in required if column not in result.columns]
    if missing:
        # A caller may provide a deliberately reduced manifest; recover the
        # three label-free summaries from the 25-position pair itself.
        generated = build_feature_frame(frame)
        for column in missing:
            result[column] = generated[column].to_numpy()
    return result


def _fit_numeric_transform(fit_values: pd.DataFrame, query_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], tuple[str, ...]]:
    medians = fit_values.median(axis=0).fillna(0.0)
    filled_fit = fit_values.fillna(medians).fillna(0.0)
    filled_query = query_values.fillna(medians).fillna(0.0)
    active = tuple(filled_fit.columns[filled_fit.nunique(dropna=False).gt(1)].astype(str))
    fit = filled_fit.loc[:, list(active)].astype(float)
    query = filled_query.reindex(columns=list(active)).astype(float)
    if not np.isfinite(fit.to_numpy(float)).all() or not np.isfinite(query.to_numpy(float)).all():
        raise ValueError("Sequence feature transform contains non-finite values")
    median_state = {str(name): float(medians[name]) for name in medians.index}
    return fit, query, median_state, active


def _fit_guide_lookup(fit_frame: pd.DataFrame, y: np.ndarray) -> dict[str, dict[Any, tuple[float, int]]]:
    guide = fit_frame[GUIDE_COLUMN].astype(str).reset_index(drop=True)
    temp = fit_frame.reset_index(drop=True)
    diffcat = np.where(
        temp["aligned_difference_count"].eq(0),
        "exact",
        np.where(temp["aligned_difference_count"].eq(1), "one", "two_plus"),
    )
    gap_total = temp["gap_count_target"].fillna(0) + temp["gap_count_guide"].fillna(0)
    gapcat = np.where(gap_total.gt(0), "gap", "no_gap")

    def grouped(keys: pd.Series | pd.MultiIndex, values: np.ndarray = y) -> dict[Any, tuple[float, int]]:
        source = pd.DataFrame({"key": list(keys), "y": values})
        stats = source.groupby("key", sort=False)["y"].agg(["sum", "count"])
        return {key: (float(row["sum"]), int(row["count"])) for key, row in stats.iterrows()}

    lookup: dict[str, dict[Any, tuple[float, int]]] = {
        "all": grouped(guide),
    }
    for name, mask in (
        ("exact", temp["aligned_difference_count"].eq(0).to_numpy()),
        ("nonexact", ~temp["aligned_difference_count"].eq(0).to_numpy()),
    ):
        lookup[name] = grouped(guide.loc[mask], y[mask]) if mask.any() else {}
    lookup["diffcat"] = grouped(pd.MultiIndex.from_arrays([guide.to_numpy(), diffcat]))
    lookup["gapcat"] = grouped(pd.MultiIndex.from_arrays([guide.to_numpy(), gapcat]))
    return lookup


def _fit_mapping_lookup(mapping: pd.DataFrame, y: np.ndarray) -> dict[str, dict[str, tuple[float, int]]]:
    lookup: dict[str, dict[str, tuple[float, int]]] = {}
    for column in MAPPING_CATEGORICAL:
        temp = pd.DataFrame({"key": mapping[column].fillna("missing").astype(str).to_numpy(), "y": y})
        stats = temp.groupby("key", sort=False)["y"].agg(["sum", "count"])
        lookup[column] = {
            str(key): (float(row["sum"]), int(row["count"])) for key, row in stats.iterrows()
        }
    return lookup


def _guide_query_features(query_frame: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    query = query_frame.reset_index(drop=True)
    guide = query[GUIDE_COLUMN].astype(str)
    diffcat = np.where(
        query["aligned_difference_count"].eq(0),
        "exact",
        np.where(query["aligned_difference_count"].eq(1), "one", "two_plus"),
    )
    gap_total = query["gap_count_target"].fillna(0) + query["gap_count_guide"].fillna(0)
    gapcat = np.where(gap_total.gt(0), "gap", "no_gap")
    smoothing = state.guide_smoothing
    all_stats = state.guide_lookup["all"]
    all_sum = np.asarray([all_stats.get(key, (0.0, 0))[0] for key in guide], dtype=float)
    all_count = np.asarray([all_stats.get(key, (0.0, 0))[1] for key in guide], dtype=float)
    guide_all = (all_sum + smoothing * state.global_y_mean) / (all_count + smoothing)
    features: dict[str, np.ndarray] = {
        "te_guide_all": guide_all,
        "te_guide_all_log_count": np.log1p(all_count),
        "te_guide_seen": (all_count > 0).astype(float),
    }

    for name in ("exact", "nonexact"):
        stats = state.guide_lookup[name]
        sums = np.asarray([stats.get(key, (0.0, 0))[0] for key in guide], dtype=float)
        counts = np.asarray([stats.get(key, (0.0, 0))[1] for key in guide], dtype=float)
        features[f"te_guide_{name}_ref"] = (sums + smoothing * guide_all) / (counts + smoothing)
        features[f"te_guide_{name}_log_count"] = np.log1p(counts)

    for name, categories in (("diffcat", diffcat), ("gapcat", gapcat)):
        stats = state.guide_lookup[name]
        pairs = list(zip(guide.to_numpy(), categories))
        sums = np.asarray([stats.get(key, (0.0, 0))[0] for key in pairs], dtype=float)
        counts = np.asarray([stats.get(key, (0.0, 0))[1] for key in pairs], dtype=float)
        features[f"te_guide_{name}"] = (sums + smoothing * guide_all) / (counts + smoothing)
        features[f"te_guide_{name}_log_count"] = np.log1p(counts)
    return pd.DataFrame(features, index=query_frame.index)


def _mapping_query_features(query_mapping: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    features: dict[str, np.ndarray] = {}
    smoothing = state.mapping_smoothing
    for column in MAPPING_CATEGORICAL:
        values = query_mapping[column].fillna("missing").astype(str)
        lookup = state.mapping_lookup[column]
        sums = np.asarray([lookup.get(key, (0.0, 0))[0] for key in values], dtype=float)
        counts = np.asarray([lookup.get(key, (0.0, 0))[1] for key in values], dtype=float)
        features[f"mapping_te_{column}"] = (sums + smoothing * state.global_y_mean) / (counts + smoothing)
        features[f"mapping_te_{column}_log_count"] = np.log1p(counts)
    return pd.DataFrame(features, index=query_mapping.index)


def _onehot_query(query_mapping: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    values: list[pd.DataFrame] = []
    for column in MAPPING_CATEGORICAL:
        keys = query_mapping[column].fillna("missing").astype(str)
        dummies = pd.get_dummies(keys, prefix=f"mapping_cat_{column}", dtype=np.uint8)
        values.append(dummies.set_axis(query_mapping.index))
    result = pd.concat(values, axis=1)
    return result.reindex(columns=list(columns), fill_value=0)


def _transform_query_details(query_frame: pd.DataFrame, state: FeatureState) -> dict[str, Any]:
    if not isinstance(query_frame, pd.DataFrame):
        raise TypeError("query_frame must be a pandas DataFrame")
    forbidden = _query_label_columns(query_frame)
    if forbidden:
        raise ValueError(f"Query frame must not contain labels: {forbidden}")
    query_ids = _require_ids(query_frame, "query")
    required = {GUIDE_COLUMN}
    missing = sorted(required - set(query_frame.columns))
    if missing:
        raise ValueError(f"Query frame is missing required guide/sequence inputs: {missing}")

    query_values, query_sequence_source = _sequence_values(query_frame, state.sequence_manifest.feature_names)
    guide_query_frame = _guide_input_frame(query_frame, query_values)
    medians = pd.Series(state.numeric_medians)
    query_filled = query_values.fillna(medians).fillna(0.0)
    d_query = query_filled.reindex(columns=list(state.active_numeric_columns)).astype(float)
    if not np.isfinite(d_query.to_numpy(float)).all():
        raise ValueError("Query D matrix contains non-finite values")

    query_pairs_all = pair_onehot(query_frame)
    pair_query = query_pairs_all.reindex(columns=list(state.active_pair_columns), fill_value=0)
    query_mapping = mapping_frame_from_input(query_frame, state.template_mapper)
    query_numeric_mapping = numeric_mapping(query_mapping)
    query_onehot = _onehot_query(query_mapping, state.mapping_onehot_columns)
    guide_query = _guide_query_features(guide_query_frame, state)
    mapping_query = _mapping_query_features(query_mapping, state)

    if len(query_frame) != len(query_ids):
        raise RuntimeError("Query transformation changed row count")
    context_query = pd.concat(
        [d_query, guide_query, pair_query, query_numeric_mapping, mapping_query, query_onehot], axis=1
    )
    seq_mapping_query = pd.concat([d_query, pair_query, query_numeric_mapping, query_onehot], axis=1)
    if not np.isfinite(context_query.to_numpy(float)).all() or not np.isfinite(seq_mapping_query.to_numpy(float)).all():
        raise ValueError("Query feature matrix contains non-finite values")
    mapping_found, guide_support, template_support = query_support_counts(
        state._support_fit_frame, query_frame, state._support_fit_mapping, query_mapping
    )
    diagnostics = pd.DataFrame(
        {
            "record_id": query_ids.to_numpy(),
            "mapping_found": mapping_found.astype(bool),
            "guide_support_n": guide_support.astype(int),
            "template_support_n": template_support.astype(int),
        },
        index=query_frame.index,
    )
    guide_anchor = guide_query["te_guide_all"].to_numpy(float)
    template_anchor = mapping_query["mapping_te_mapping_candidate_template_nos"].to_numpy(float)
    return {
        "d_query": d_query,
        "context_query": context_query,
        "seq_mapping_query": seq_mapping_query,
        "guide_anchor_query": guide_anchor,
        "template_anchor_query": template_anchor,
        "query_diagnostics": diagnostics,
        "query_sequence_source": query_sequence_source,
    }


def _manifest(
    protocol_id: str,
    sequence_names: tuple[str, ...],
    active_numeric: tuple[str, ...],
    active_pair: tuple[str, ...],
    guide_columns: tuple[str, ...],
    mapping_history_columns: tuple[str, ...],
    mapping_onehot_columns: tuple[str, ...],
    mapping_numeric_columns: tuple[str, ...],
    context_columns: tuple[str, ...],
    seq_mapping_columns: tuple[str, ...],
    fit_sequence_source: str,
    query_sequence_source: str,
) -> dict[str, Any]:
    return {
        "protocol_id": protocol_id,
        "sequence_feature_names": list(sequence_names),
        "active_numeric_columns": list(active_numeric),
        "active_pair_columns": list(active_pair),
        "guide_history_columns": list(guide_columns),
        "mapping_numeric_columns": list(mapping_numeric_columns),
        "mapping_history_columns": list(mapping_history_columns),
        "mapping_onehot_columns": list(mapping_onehot_columns),
        "d_columns": list(active_numeric),
        "context_columns": list(context_columns),
        "seq_mapping_columns": list(seq_mapping_columns),
        "fit_sequence_source": fit_sequence_source,
        "query_sequence_source": query_sequence_source,
        "guide_smoothing": GUIDE_SMOOTHING,
        "mapping_smoothing": MAPPING_SMOOTHING,
        "mapping_history_crossfit": "current_protocol_remaining_outer_fold_ids",
    }


def build_features(
    fit_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    *,
    sequence_manifest: Any,
    protocol_id: str,
    encoding_fold_by_record_id: Any,
    recipe: dict[str, Any] | None = None,
) -> FeatureBundle:
    """Build the fixed D, B/C-context, and no-history sequence+mapping inputs."""
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"Unsupported protocol_id {protocol_id!r}; expected one of {PROTOCOLS}")
    if not isinstance(fit_frame, pd.DataFrame) or not isinstance(query_frame, pd.DataFrame):
        raise TypeError("fit_frame and query_frame must be pandas DataFrames")
    query_labels = _query_label_columns(query_frame)
    if query_labels:
        raise ValueError(f"query_frame must not contain labels: {query_labels}")
    fit_ids = _require_ids(fit_frame, "fit")
    query_ids = _require_ids(query_frame, "query")
    overlap = set(fit_ids) & set(query_ids)
    if overlap:
        raise ValueError(f"fit/query record_id overlap is forbidden: {sorted(overlap)[:5]}")
    y = _labels(fit_frame)
    if not len(fit_frame):
        raise ValueError("fit_frame must not be empty")
    if recipe is None:
        recipe = {}
    guide_smoothing = float(recipe.get("guide_smoothing", GUIDE_SMOOTHING))
    mapping_smoothing = float(recipe.get("mapping_smoothing", MAPPING_SMOOTHING))
    if guide_smoothing != GUIDE_SMOOTHING or mapping_smoothing != MAPPING_SMOOTHING:
        raise ValueError("Guide and mapping smoothing values are frozen at 0.5 and 20.0")
    manifest = coerce_sequence_manifest(sequence_manifest)
    template_mapper = TemplateMapper(manifest.template_records) if manifest.template_records else None
    fold_column = f"{protocol_id}_fold"
    if fold_column in fit_frame:
        if isinstance(encoding_fold_by_record_id, pd.Series):
            fold_lookup = encoding_fold_by_record_id.copy()
        else:
            fold_lookup = pd.Series(dict(encoding_fold_by_record_id))
        fold_lookup.index = fold_lookup.index.astype(str)
        actual_folds = fit_ids.map(fold_lookup)
        expected_folds = pd.to_numeric(fit_frame[fold_column], errors="coerce").reset_index(drop=True)
        if actual_folds.isna().any() or expected_folds.isna().any():
            raise ValueError(f"encoding_fold_by_record_id must cover fit rows for {fold_column}")
        actual_values = pd.to_numeric(actual_folds, errors="coerce").reset_index(drop=True)
        if not actual_values.astype(int).equals(expected_folds.astype(int)):
            raise ValueError(f"encoding_fold_by_record_id does not match current protocol column {fold_column}")

    required_sequences = {GUIDE_COLUMN}
    for frame_name, frame in (("fit", fit_frame), ("query", query_frame)):
        missing = sorted(required_sequences - set(frame.columns))
        if missing:
            raise ValueError(f"{frame_name} frame is missing required sequence/history fields: {missing}")

    fit_values, fit_sequence_source = _sequence_values(fit_frame, manifest.feature_names)
    query_values, query_sequence_source = _sequence_values(query_frame, manifest.feature_names)
    guide_fit_frame = _guide_input_frame(fit_frame, fit_values)
    guide_query_frame = _guide_input_frame(query_frame, query_values)
    d_fit, d_query, numeric_medians, active_numeric = _fit_numeric_transform(fit_values, query_values)

    fit_pair_all = pair_onehot(fit_frame)
    query_pair_all = pair_onehot(query_frame)
    active_pair = tuple(fit_pair_all.columns[fit_pair_all.nunique(dropna=False).gt(1)].astype(str))
    pair_fit = fit_pair_all.loc[:, list(active_pair)]
    pair_query = query_pair_all.reindex(columns=list(active_pair), fill_value=0)

    fit_mapping = mapping_frame_from_input(fit_frame, template_mapper)
    query_mapping = mapping_frame_from_input(query_frame, template_mapper)
    fit_numeric_mapping = numeric_mapping(fit_mapping)
    query_numeric_mapping = numeric_mapping(query_mapping)
    fit_onehot, query_onehot = mapping_onehot(fit_mapping, query_mapping)
    onehot_columns = tuple(fit_onehot.columns.astype(str))
    fit_guide_history, query_guide_history = guide_target_encoding(
        guide_fit_frame, guide_query_frame, smoothing=guide_smoothing
    )
    fit_mapping_history, query_mapping_history = mapping_target_encoding(
        fit_frame,
        query_frame,
        fit_mapping,
        query_mapping,
        encoding_fold_by_record_id,
        smoothing=mapping_smoothing,
    )

    y_mean = float(np.mean(y))
    guide_lookup = _fit_guide_lookup(guide_fit_frame, y)
    mapping_lookup = _fit_mapping_lookup(fit_mapping, y)
    state_manifest = {
        "protocol_id": protocol_id,
        "sequence_feature_names": tuple(manifest.feature_names),
        "active_numeric_columns": active_numeric,
        "active_pair_columns": active_pair,
        "guide_history_columns": tuple(fit_guide_history.columns.astype(str)),
        "mapping_numeric_columns": tuple(fit_numeric_mapping.columns.astype(str)),
        "mapping_history_columns": tuple(fit_mapping_history.columns.astype(str)),
        "mapping_onehot_columns": onehot_columns,
        "d_columns": tuple(d_fit.columns.astype(str)),
        "context_columns": (),
        "seq_mapping_columns": (),
        "fit_sequence_source": fit_sequence_source,
        "query_sequence_source": query_sequence_source,
        "guide_smoothing": guide_smoothing,
        "mapping_smoothing": mapping_smoothing,
        "mapping_history_crossfit": "current_protocol_remaining_outer_fold_ids",
    }
    state = FeatureState(
        protocol_id=protocol_id,
        sequence_manifest=manifest,
        numeric_medians=numeric_medians,
        active_numeric_columns=active_numeric,
        active_pair_columns=active_pair,
        mapping_onehot_columns=onehot_columns,
        global_y_mean=y_mean,
        guide_smoothing=guide_smoothing,
        mapping_smoothing=mapping_smoothing,
        guide_lookup=guide_lookup,
        mapping_lookup=mapping_lookup,
        template_mapper=template_mapper,
        feature_manifest=state_manifest,
    )
    # These compact fit references supply cross-model support counts in transform_query.
    # They contain data/statistics only, never a historical workspace path.
    state._support_fit_frame = fit_frame.loc[:, ["record_id", GUIDE_COLUMN]].copy()
    state._support_fit_mapping = fit_mapping.copy()

    context_fit = pd.concat(
        [d_fit, fit_guide_history, pair_fit, fit_numeric_mapping, fit_mapping_history, fit_onehot], axis=1
    )
    seq_mapping_fit = pd.concat([d_fit, pair_fit, fit_numeric_mapping, fit_onehot], axis=1)
    context_query = pd.concat(
        [d_query, query_guide_history, pair_query, query_numeric_mapping, query_mapping_history, query_onehot], axis=1
    )
    seq_mapping_query = pd.concat([d_query, pair_query, query_numeric_mapping, query_onehot], axis=1)
    if not context_fit.columns.equals(context_query.columns):
        raise RuntimeError("B/C context fit/query columns are not identical and ordered")
    if not seq_mapping_fit.columns.equals(seq_mapping_query.columns):
        raise RuntimeError("SEQ_MAPPING fit/query columns are not identical and ordered")
    for name, matrix in (
        ("context_fit", context_fit),
        ("context_query", context_query),
        ("seq_mapping_fit", seq_mapping_fit),
        ("seq_mapping_query", seq_mapping_query),
    ):
        if not np.isfinite(matrix.to_numpy(float)).all():
            raise ValueError(f"{name} contains non-finite values")

    mapping_found, guide_support, template_support = query_support_counts(
        fit_frame, query_frame, fit_mapping, query_mapping
    )
    query_diagnostics = pd.DataFrame(
        {
            "record_id": query_ids.to_numpy(),
            "mapping_found": mapping_found.astype(bool),
            "guide_support_n": guide_support.astype(int),
            "template_support_n": template_support.astype(int),
        },
        index=query_frame.index,
    )
    state_manifest["context_columns"] = tuple(context_fit.columns.astype(str))
    state_manifest["seq_mapping_columns"] = tuple(seq_mapping_fit.columns.astype(str))
    bundle_manifest = _manifest(
        protocol_id,
        tuple(manifest.feature_names),
        active_numeric,
        active_pair,
        tuple(fit_guide_history.columns.astype(str)),
        tuple(fit_mapping_history.columns.astype(str)),
        onehot_columns,
        tuple(fit_numeric_mapping.columns.astype(str)),
        tuple(context_fit.columns.astype(str)),
        tuple(seq_mapping_fit.columns.astype(str)),
        fit_sequence_source,
        query_sequence_source,
    )
    state.feature_manifest = state_manifest

    return FeatureBundle(
        d_fit=d_fit,
        d_query=d_query,
        context_fit=context_fit,
        context_query=context_query,
        seq_mapping_fit=seq_mapping_fit,
        seq_mapping_query=seq_mapping_query,
        guide_anchor_fit=fit_guide_history["te_guide_all"].to_numpy(float),
        guide_anchor_query=query_guide_history["te_guide_all"].to_numpy(float),
        template_anchor_fit=fit_mapping_history["mapping_te_mapping_candidate_template_nos"].to_numpy(float),
        template_anchor_query=query_mapping_history["mapping_te_mapping_candidate_template_nos"].to_numpy(float),
        query_diagnostics=query_diagnostics,
        feature_manifest=bundle_manifest,
        state=state,
    )


def transform_query(query_frame: pd.DataFrame, state: FeatureState) -> dict[str, Any]:
    """Transform a label-free query using only the saved fit state."""
    if not isinstance(state, FeatureState):
        raise TypeError("state must be a FeatureState returned by build_features")
    return _transform_query_details(query_frame, state)
