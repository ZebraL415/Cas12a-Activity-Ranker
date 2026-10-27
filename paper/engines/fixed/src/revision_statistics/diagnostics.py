"""Descriptive, fit-free summaries from saved component_diagnostics.csv files."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analyzer import AnalysisValidationError, CORE_MODELS
from .metrics import basic_metrics


DIAGNOSTIC_PROFILES: tuple[str, ...] = (
    "D",
    "B",
    "C",
    "TriCasRank",
    "HISTORY_PRIOR",
    "SEQ_MAPPING_XGB",
    "D_B_FIXED",
    "D_C_FIXED",
    "B_C_FIXED",
    "WITHOUT_D",
    "WITHOUT_B",
    "WITHOUT_C",
    "TriCasRank_UNROUTED",
    "D_B_FIXED_UNROUTED",
    "D_C_FIXED_UNROUTED",
    "B_C_FIXED_UNROUTED",
)
COMPONENTS = ("D", "B", "C")

PROFILE_CLASS = {
    "D": "component_prediction",
    "B": "component_prediction",
    "C": "component_prediction",
    "TriCasRank": "official_routed_full_model",
    "HISTORY_PRIOR": "baseline",
    "SEQ_MAPPING_XGB": "strong_mapping_baseline",
    "D_B_FIXED": "fixed_fusion_diagnostic",
    "D_C_FIXED": "fixed_fusion_diagnostic",
    "B_C_FIXED": "fixed_fusion_diagnostic",
    "WITHOUT_D": "fixed_removal_diagnostic",
    "WITHOUT_B": "fixed_removal_diagnostic",
    "WITHOUT_C": "fixed_removal_diagnostic",
    "TriCasRank_UNROUTED": "unrouted_diagnostic_only",
    "D_B_FIXED_UNROUTED": "unrouted_diagnostic_only",
    "D_C_FIXED_UNROUTED": "unrouted_diagnostic_only",
    "B_C_FIXED_UNROUTED": "unrouted_diagnostic_only",
}


def _stable_ids_hash(record_ids: list[str] | pd.Series) -> str:
    canonical = "\n".join(sorted(map(str, record_ids)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bool_series(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise AnalysisValidationError(f"component diagnostics contain missing {name}")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    mapped = values.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise AnalysisValidationError(f"component diagnostics contain non-boolean {name}")
    return mapped.astype(bool)


def _read_diagnostic_fold(
    root: Path,
    run_metadata: dict[str, Any],
    prediction_file: dict[str, Any],
    official_oof: pd.DataFrame,
    assignments: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol_id = str(run_metadata["protocol_id"])
    fold = int(prediction_file["outer_fold"])
    prediction_path = Path(prediction_file["path"]).resolve()
    diagnostic_path = (prediction_path.parent / "component_diagnostics.csv").resolve()
    runs_root = (root / "runs").resolve()
    if not diagnostic_path.is_relative_to(runs_root) or not diagnostic_path.is_file():
        raise AnalysisValidationError(f"missing or out-of-scope component diagnostics file: {diagnostic_path}")
    diagnostic_sha = hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
    frame = pd.read_csv(diagnostic_path, dtype={"record_id": "string"})
    required = {
        "record_id",
        "mapping_found",
        "guide_support_n",
        "template_support_n",
        "y_true",
        "outer_fold",
        "protocol_id",
        "cohort_id",
        *DIAGNOSTIC_PROFILES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AnalysisValidationError(f"{diagnostic_path} lacks diagnostic fields: {missing}")
    frame["record_id"] = frame["record_id"].astype(str)
    if frame.empty or frame["record_id"].duplicated().any():
        raise AnalysisValidationError(f"{diagnostic_path} is empty or contains duplicate record_id values")
    frame["mapping_found"] = _bool_series(frame["mapping_found"], "mapping_found")
    for column in ("guide_support_n", "template_support_n", "outer_fold"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.equal(values.to_numpy(), np.floor(values.to_numpy())).all():
            raise AnalysisValidationError(f"{diagnostic_path}: {column} must be integer-valued")
        frame[column] = values.astype(int)
    if (frame[["guide_support_n", "template_support_n"]] < 0).any().any():
        raise AnalysisValidationError(f"{diagnostic_path}: support counts must be non-negative")
    numeric_columns = ["y_true", *DIAGNOSTIC_PROFILES]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=float)
        if frame[column].isna().any() or not np.isfinite(values).all():
            raise AnalysisValidationError(f"{diagnostic_path}: {column} contains missing or non-finite values")
    if set(frame["cohort_id"].astype(str)) != {"core_dev"}:
        raise AnalysisValidationError(f"{diagnostic_path}: cohort_id is not core_dev")
    if set(frame["protocol_id"].astype(str)) != {protocol_id}:
        raise AnalysisValidationError(f"{diagnostic_path}: protocol_id does not match the ledger")
    if set(frame["outer_fold"].astype(int)) != {fold}:
        raise AnalysisValidationError(f"{diagnostic_path}: outer_fold does not match the ledger")

    official = official_oof.loc[
        (official_oof["cohort_id"].astype(str) == "core_dev")
        & (official_oof["protocol_id"].astype(str) == protocol_id)
        & (official_oof["outer_fold"].astype(int) == fold)
    ].copy()
    if official.empty or official.duplicated(["model_id", "record_id"]).any():
        raise AnalysisValidationError(f"official prediction source is empty or duplicated for {protocol_id}/fold_{fold}")
    official["record_id"] = official["record_id"].astype(str)
    base = official.loc[official["model_id"].astype(str) == "D"].set_index("record_id").sort_index()
    diagnostic = frame.set_index("record_id").sort_index()
    if set(base.index) != set(diagnostic.index):
        raise AnalysisValidationError(f"{diagnostic_path}: record IDs do not equal official query IDs for fold {fold}")
    diagnostic = diagnostic.loc[base.index].reset_index()
    base = base.loc[diagnostic["record_id"]]
    if not np.allclose(diagnostic["y_true"].to_numpy(), base["y_true"].to_numpy(dtype=float), rtol=0, atol=1e-12):
        raise AnalysisValidationError(f"{diagnostic_path}: labels do not match official OOF predictions")
    if not _bool_series(base["mapping_found"], "official mapping_found").reset_index(drop=True).equals(
        diagnostic["mapping_found"].reset_index(drop=True)
    ):
        raise AnalysisValidationError(f"{diagnostic_path}: mapping_found does not match official OOF predictions")
    for support_column in ("guide_support_n", "template_support_n"):
        if not np.array_equal(
            diagnostic[support_column].to_numpy(dtype=int),
            base[support_column].to_numpy(dtype=int),
        ):
            raise AnalysisValidationError(f"{diagnostic_path}: {support_column} does not match official OOF predictions")
    official_wide = official.pivot(index="record_id", columns="model_id", values="y_pred").loc[diagnostic["record_id"]]
    for model in CORE_MODELS:
        if model not in official_wide.columns or model not in diagnostic.columns:
            raise AnalysisValidationError(f"{diagnostic_path}: official prediction comparison lacks {model}")
        if not np.allclose(
            diagnostic[model].to_numpy(dtype=float),
            official_wide[model].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        ):
            raise AnalysisValidationError(f"{diagnostic_path}: saved {model} values differ from official OOF predictions")

    assignment = assignments.set_index("record_id")
    group_column = f"{protocol_id}_group"
    fold_column = f"{protocol_id}_fold"
    expected_groups = assignment.loc[diagnostic["record_id"], group_column].astype(str).to_numpy()
    expected_folds = assignment.loc[diagnostic["record_id"], fold_column].astype(int).to_numpy()
    if not np.array_equal(diagnostic["outer_fold"].to_numpy(dtype=int), expected_folds):
        raise AnalysisValidationError(f"{diagnostic_path}: outer_fold differs from source assignment by record_id")
    diagnostic["group_id"] = expected_groups
    expected_labels = labels.set_index("record_id").loc[diagnostic["record_id"], "y_true"].to_numpy(dtype=float)
    if not np.allclose(diagnostic["y_true"].to_numpy(dtype=float), expected_labels, rtol=0, atol=1e-12):
        raise AnalysisValidationError(f"{diagnostic_path}: labels differ from the processed development-label source")
    metadata = {
        "outer_fold": fold,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_file["sha256"],
        "component_diagnostics_path": str(diagnostic_path),
        "component_diagnostics_sha256": diagnostic_sha,
        "record_n": int(len(diagnostic)),
        "record_id_set_sha256": _stable_ids_hash(diagnostic["record_id"]),
    }
    return diagnostic, metadata


def _profile_rows(frame: pd.DataFrame, profiles: tuple[str, ...] = DIAGNOSTIC_PROFILES) -> list[dict[str, Any]]:
    cohort_id = str(frame["cohort_id"].iloc[0])
    protocol_id = str(frame["protocol_id"].iloc[0])
    result = []
    constant_fractions = {
        profile: float(
            np.mean(
                [
                    np.ptp(frame.loc[frame["outer_fold"] == fold, profile].to_numpy(dtype=float)) == 0
                    for fold in range(5)
                ]
            )
        )
        for profile in profiles
    }
    subsets = [("pooled_oof", None, frame)] + [
        ("outer_fold", int(fold), frame.loc[frame["outer_fold"] == fold]) for fold in range(5)
    ]
    for level, outer_fold, subset in subsets:
        record_ids = subset["record_id"].astype(str)
        for profile in profiles:
            score = basic_metrics(subset["y_true"].to_numpy(dtype=float), subset[profile].to_numpy(dtype=float))
            result.append(
                {
                    "cohort_id": cohort_id,
                    "protocol_id": protocol_id,
                    "evaluation_level": level,
                    "outer_fold": outer_fold,
                    "profile_id": profile,
                    "profile_class": PROFILE_CLASS[profile],
                    "prediction_scale": "supplied_processed_label_scale",
                    "record_n": int(len(subset)),
                    "group_n": int(subset["group_id"].nunique()),
                    "record_id_set_sha256": _stable_ids_hash(record_ids),
                    "constant_prediction": bool(np.ptp(subset[profile].to_numpy(dtype=float)) == 0),
                    "constant_fold_fraction": constant_fractions[profile],
                    **score,
                }
            )
    return result


def _correlation_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cohort_id = str(frame["cohort_id"].iloc[0])
    protocol_id = str(frame["protocol_id"].iloc[0])
    result = []
    subsets = [("pooled_oof", None, frame)] + [
        ("outer_fold", int(fold), frame.loc[frame["outer_fold"] == fold]) for fold in range(5)
    ]
    for level, outer_fold, subset in subsets:
        for pair in combinations(COMPONENTS, 2):
            for value_type in ("prediction", "residual"):
                if value_type == "prediction":
                    x = subset[pair[0]].to_numpy(dtype=float)
                    y = subset[pair[1]].to_numpy(dtype=float)
                else:
                    x = subset["y_true"].to_numpy(dtype=float) - subset[pair[0]].to_numpy(dtype=float)
                    y = subset["y_true"].to_numpy(dtype=float) - subset[pair[1]].to_numpy(dtype=float)
                corr = basic_metrics(x, y)
                result.append(
                    {
                        "cohort_id": cohort_id,
                        "protocol_id": protocol_id,
                        "evaluation_level": level,
                        "outer_fold": outer_fold,
                        "value_type": value_type,
                        "component_a": pair[0],
                        "component_b": pair[1],
                        "n": int(len(subset)),
                        "group_n": int(subset["group_id"].nunique()),
                        "record_id_set_sha256": _stable_ids_hash(subset["record_id"]),
                        "SCC": corr["SCC"],
                        "PCC": corr["PCC"],
                        "constant_a": bool(np.ptp(x) == 0),
                        "constant_b": bool(np.ptp(y) == 0),
                        "residual_definition": "y_true - component_prediction" if value_type == "residual" else None,
                        "inferential_test": None,
                    }
                )
    return result


def _activity_quintile_rows(frame: pd.DataFrame, quintile_map: pd.Series) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cohort_id = str(frame["cohort_id"].iloc[0])
    protocol_id = str(frame["protocol_id"].iloc[0])
    one_per_id = frame.drop_duplicates("record_id").set_index("record_id")
    records = []
    edges = []
    for quintile in range(1, 6):
        ids = quintile_map.index[quintile_map == quintile]
        subset = one_per_id.loc[ids]
        edges.append(
            {
                "cohort_id": cohort_id,
                "protocol_id": protocol_id,
                "quintile": quintile,
                "quintile_label": f"Q{quintile}",
                "definition": "equal-frequency qcut on unique record_id y_true, ascending supplied processed-label value",
                "record_n": int(len(subset)),
                "record_id_set_sha256": _stable_ids_hash(ids),
                "y_true_min": float(subset["y_true"].min()),
                "y_true_max": float(subset["y_true"].max()),
                "y_true_mean": float(subset["y_true"].mean()),
            }
        )
        for profile in DIAGNOSTIC_PROFILES:
            prediction = subset[profile].to_numpy(dtype=float)
            truth = subset["y_true"].to_numpy(dtype=float)
            records.append(
                {
                    "cohort_id": cohort_id,
                    "protocol_id": protocol_id,
                    "quintile": quintile,
                    "quintile_label": f"Q{quintile}",
                    "profile_id": profile,
                    "profile_class": PROFILE_CLASS[profile],
                    "record_n": int(len(subset)),
                    "group_n": int(subset["group_id"].nunique()),
                    "record_id_set_sha256": _stable_ids_hash(ids),
                    "y_true_min": float(truth.min()),
                    "y_true_max": float(truth.max()),
                    "y_true_mean": float(truth.mean()),
                    "MAE": float(np.mean(np.abs(truth - prediction))),
                    "RMSE": float(np.sqrt(np.mean(np.square(truth - prediction)))),
                    "inferential_test": None,
                }
            )
    return records, edges


def _support_rows(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cohort_id = str(frame["cohort_id"].iloc[0])
    protocol_id = str(frame["protocol_id"].iloc[0])
    id_columns = [
        "cohort_id",
        "protocol_id",
        "record_id",
        "group_id",
        "outer_fold",
        "y_true",
        "mapping_found",
        "guide_support_n",
        "template_support_n",
    ]
    ids = frame[id_columns].drop_duplicates("record_id").sort_values(["outer_fold", "record_id"], kind="stable")
    if len(ids) != frame["record_id"].nunique():
        raise AnalysisValidationError(f"{cohort_id}/{protocol_id}: diagnostic support IDs are not unique")
    summary_rows = []
    distribution_rows = []
    metric_rows = []
    for support_type in ("guide_support_n", "template_support_n"):
        values = ids[support_type].to_numpy(dtype=int)
        summary_rows.append(
            {
                "cohort_id": cohort_id,
                "protocol_id": protocol_id,
                "support_type": support_type,
                "record_n": int(len(ids)),
                "group_n": int(ids["group_id"].nunique()),
                "zero_support_n": int(np.count_nonzero(values == 0)),
                "zero_support_fraction": float(np.mean(values == 0)),
                "positive_support_n": int(np.count_nonzero(values > 0)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "max": int(np.max(values)),
                "record_id_set_sha256": _stable_ids_hash(ids["record_id"]),
            }
        )
        group_cols = [support_type, "mapping_found"]
        for (support_value, mapping_found), index in ids.groupby(group_cols, sort=True).groups.items():
            record_ids = ids.loc[index, "record_id"].astype(str).tolist()
            stratum = frame.loc[frame["record_id"].astype(str).isin(record_ids)]
            distribution_rows.append(
                {
                    "cohort_id": cohort_id,
                    "protocol_id": protocol_id,
                    "support_type": support_type,
                    "support_n": int(support_value),
                    "mapping_found": bool(mapping_found),
                    "record_n": int(len(record_ids)),
                    "group_n": int(ids.loc[index, "group_id"].nunique()),
                    "record_id_set_sha256": _stable_ids_hash(record_ids),
                }
            )
            if not record_ids:
                continue
            for profile in DIAGNOSTIC_PROFILES:
                subset = stratum.drop_duplicates("record_id").set_index("record_id").loc[record_ids]
                score = basic_metrics(subset["y_true"].to_numpy(dtype=float), subset[profile].to_numpy(dtype=float))
                metric_rows.append(
                    {
                        "cohort_id": cohort_id,
                        "protocol_id": protocol_id,
                        "support_type": support_type,
                        "support_n": int(support_value),
                        "mapping_found": bool(mapping_found),
                        "profile_id": profile,
                        "profile_class": PROFILE_CLASS[profile],
                        "record_n": int(len(subset)),
                        "group_n": int(subset["group_id"].nunique()),
                        "record_id_set_sha256": _stable_ids_hash(subset.index.astype(str)),
                        **score,
                        "inferential_test": None,
                    }
                )
    return ids, summary_rows, distribution_rows, metric_rows


def _quintile_map(frames: dict[str, pd.DataFrame]) -> tuple[pd.Series, list[float]]:
    first = next(iter(frames.values()))
    base = first.drop_duplicates("record_id").set_index("record_id")["y_true"].sort_index()
    for protocol_id, frame in frames.items():
        current = frame.drop_duplicates("record_id").set_index("record_id")["y_true"].sort_index()
        if not base.index.equals(current.index) or not np.allclose(base.to_numpy(), current.to_numpy(), rtol=0, atol=1e-12):
            raise AnalysisValidationError(f"core component diagnostics have inconsistent label IDs/values in {protocol_id}")
    codes, edges = pd.qcut(base, 5, labels=False, retbins=True, duplicates="drop")
    if codes.isna().any() or int(codes.max()) + 1 != 5:
        raise AnalysisValidationError("processed-label quintile binning did not produce five non-empty bins")
    return codes.astype(int) + 1, [float(x) for x in edges]


def analyze_core_component_diagnostics(
    execution_root: str | Path,
    complete_protocols: list[dict[str, Any]],
    official_oof: pd.DataFrame,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and summarize component diagnostic CSVs for complete core protocols only.

    Outputs are descriptive: no bootstrap, P values or additional significance
    family is created. Component profiles and support strata use paired query IDs.
    """
    root = Path(execution_root).resolve()
    label_table = labels.copy()
    assignment_table = assignments.copy()
    label_table["record_id"] = label_table["record_id"].astype(str)
    assignment_table["record_id"] = assignment_table["record_id"].astype(str)
    core_metadata = [item for item in complete_protocols if item["cohort_id"] == "core_dev"]
    protocol_frames: dict[str, pd.DataFrame] = {}
    diagnostic_hashes: list[dict[str, Any]] = []
    validation = {
        "complete_core_protocols_used_only": True,
        "diagnostics_match_official_oof_ids_labels_fold_support": True,
        "official_model_predictions_match_diagnostic_columns": True,
        "protocol_group_assignment_matches_by_record_id": True,
        "all_profile_comparisons_share_identical_record_ids_within_protocol": True,
        "models_fitted_or_refit": False,
        "additional_inferential_families_created": False,
    }
    for metadata in core_metadata:
        protocol_id = str(metadata["protocol_id"])
        fold_frames = []
        expected_n = int(metadata["record_n"])
        for file_info in metadata["prediction_files"]:
            fold, diag_meta = _read_diagnostic_fold(
                root,
                metadata,
                file_info,
                official_oof,
                assignment_table,
                label_table,
            )
            fold_frames.append(fold)
            diagnostic_hashes.append(
                {"protocol_id": protocol_id, **diag_meta}
            )
        pooled = pd.concat(fold_frames, ignore_index=True)
        if pooled.duplicated("record_id").any() or pooled["record_id"].nunique() != expected_n:
            raise AnalysisValidationError(
                f"core_dev/{protocol_id}: component diagnostics are not exactly one row per expected OOF record"
            )
        if set(pooled["outer_fold"].astype(int)) != {0, 1, 2, 3, 4}:
            raise AnalysisValidationError(f"core_dev/{protocol_id}: diagnostics do not cover all five outer folds")
        protocol_frames[protocol_id] = pooled
    quintiles, quintile_edges = _quintile_map(protocol_frames) if protocol_frames else (pd.Series(dtype=int), [])

    profile_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    quintile_rows: list[dict[str, Any]] = []
    quintile_edge_rows: list[dict[str, Any]] = []
    support_id_frames = []
    support_summary_rows: list[dict[str, Any]] = []
    support_distribution_rows: list[dict[str, Any]] = []
    support_metric_rows: list[dict[str, Any]] = []
    for protocol_id, frame in protocol_frames.items():
        profile_rows.extend(_profile_rows(frame))
        correlation_rows.extend(_correlation_rows(frame))
        quintile_values, edge_rows = _activity_quintile_rows(frame, quintiles)
        quintile_rows.extend(quintile_values)
        quintile_edge_rows.extend(edge_rows)
        ids, summaries, distributions, support_metrics = _support_rows(frame)
        support_id_frames.append(ids)
        support_summary_rows.extend(summaries)
        support_distribution_rows.extend(distributions)
        support_metric_rows.extend(support_metrics)
    tables = {
        "component_metric_profiles": profile_rows,
        "component_correlations": correlation_rows,
        "activity_quintile_mae": quintile_rows,
        "activity_quintile_edges": quintile_edge_rows,
        "diagnostic_support_ids": pd.concat(support_id_frames, ignore_index=True)
        if support_id_frames
        else pd.DataFrame(
            columns=["cohort_id", "protocol_id", "record_id", "group_id", "outer_fold", "y_true", "mapping_found", "guide_support_n", "template_support_n"]
        ),
        "diagnostic_support_summary": support_summary_rows,
        "diagnostic_support_distribution": support_distribution_rows,
        "diagnostic_support_metric_profiles": support_metric_rows,
    }
    summary = {
        "status": "available" if core_metadata else "pending_no_complete_core_protocol",
        "source": "runs/<complete-ledger-core-run>/fold_<0..4>/component_diagnostics.csv",
        "scope": "core_dev completed protocols only; no partial common_dev diagnostics included",
        "complete_protocols": [
            {"protocol_id": x["protocol_id"], "record_n": x["record_n"], "group_n": x["group_n"]}
            for x in core_metadata
        ],
        "diagnostic_file_hashes": diagnostic_hashes,
        "profile_definition": {
            "D": "sequence-only ensemble prediction",
            "B": "guide-anchor plus fitted residual prediction",
            "C": "mean guide/template anchor plus fitted residual prediction",
            "TriCasRank": "routed frozen full prediction; missing mapping falls back to D",
            "D_B_FIXED/D_C_FIXED/B_C_FIXED": "fixed-weight fusion diagnostics with the same missing-mapping fallback",
            "WITHOUT_D/WITHOUT_B/WITHOUT_C": "fixed removal profiles with the same missing-mapping fallback",
            "*_UNROUTED": "unrouted component blend for diagnostic comparison only; not a deployable route",
            "residual_correlation": "observed processed y_true minus the saved component prediction",
        },
        "activity_quintile": {
            "definition": "five equal-frequency qcut bins on one unique record_id y_true per core_dev record, ordered by ascending supplied processed-label value",
            "edges": quintile_edges,
            "bin_record_counts": [int((quintiles == q).sum()) for q in range(1, 6)] if len(quintiles) else [],
        },
        "inferential_boundary": "descriptive metric/correlation profiles only; no P values, confidence intervals or new significance family",
        "checks": validation,
    }
    return tables, summary


def _number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "not estimable"
    return f"{float(value):.{digits}f}"


def draft_component_diagnostics_summary(
    tables: dict[str, Any],
    summary: dict[str, Any],
    source_prefix: str | Path,
) -> tuple[str, str]:
    """Create short EN/ZH evidence notes for the descriptive diagnostic tables."""
    profiles = pd.DataFrame(tables["component_metric_profiles"])
    correlations = pd.DataFrame(tables["component_correlations"])
    quintiles = pd.DataFrame(tables["activity_quintile_mae"])
    support = pd.DataFrame(tables["diagnostic_support_summary"])
    prefix = str(source_prefix)
    en_lines: list[str] = []
    zh_lines: list[str] = []
    selected_profiles = ("D", "B", "C", "TriCasRank", "D_B_FIXED", "WITHOUT_D", "WITHOUT_B", "WITHOUT_C", "TriCasRank_UNROUTED")
    for item in summary.get("complete_protocols", []):
        protocol_id = item["protocol_id"]
        pooled = profiles.loc[
            (profiles["protocol_id"] == protocol_id)
            & (profiles["evaluation_level"] == "pooled_oof")
        ].set_index("profile_id")
        available = [name for name in selected_profiles if name in pooled.index]
        scc_profile = "; ".join(f"{name}={_number(pooled.at[name, 'SCC'])}" for name in available)
        mae_profile = "; ".join(f"{name}={_number(pooled.at[name, 'MAE'])}" for name in available)
        en_lines.append(
            f"- **{protocol_id}:** pooled diagnostic SCCs: {scc_profile}. Corresponding MAEs: {mae_profile}. "
            "These are same-query descriptive profiles; routed and unrouted diagnostic variants are not independent fits."
        )
        zh_lines.append(
            f"- **{protocol_id}：** pooled 诊断 SCC：{scc_profile}。对应 MAE：{mae_profile}。"
            "这些是同一查询记录上的描述性 profile；routed/unrouted 诊断变体不是独立拟合的模型。"
        )
        full = pooled.loc["TriCasRank"]
        removal_en = []
        removal_zh = []
        for profile in ("WITHOUT_D", "WITHOUT_B", "WITHOUT_C"):
            if profile not in pooled.index:
                continue
            variant = pooled.loc[profile]
            delta_scc = float(variant["SCC"] - full["SCC"])
            delta_mae = float(variant["MAE"] - full["MAE"])
            removal_en.append(f"{profile} ΔSCC={delta_scc:+.4f}, ΔMAE={delta_mae:+.4f}")
            removal_zh.append(f"{profile} ΔSCC={delta_scc:+.4f}，ΔMAE={delta_mae:+.4f}")
        if removal_en:
            en_lines.append(
                f"  Fixed removal deltas (variant minus TriCasRank; no inferential test): {'; '.join(removal_en)}."
            )
            zh_lines.append(
                f"  固定移除差值（变体减 TriCasRank；未作推断检验）：{'；'.join(removal_zh)}。"
            )
        protocol_corr = correlations.loc[
            (correlations["protocol_id"] == protocol_id)
            & (correlations["evaluation_level"] == "pooled_oof")
        ]
        component_corr = []
        for value_type in ("prediction", "residual"):
            for pair in ("D-B", "D-C", "B-C"):
                a, b = pair.split("-")
                row = protocol_corr.loc[
                    (protocol_corr["value_type"] == value_type)
                    & (protocol_corr["component_a"] == a)
                    & (protocol_corr["component_b"] == b)
                ]
                if not row.empty:
                    record = row.iloc[0]
                    component_corr.append(
                        f"{value_type} {pair} (SCC={_number(record['SCC'])}, PCC={_number(record['PCC'])})"
                    )
        if component_corr:
            en_lines.append(f"  D/B/C correlation profile: {'; '.join(component_corr)}. Residuals are y_true minus each saved component prediction.")
            zh_lines.append(f"  D/B/C 相关 profile：{'；'.join(component_corr)}。残差定义为 y_true 减去对应保存的分量预测。")
        q = quintiles.loc[
            (quintiles["protocol_id"] == protocol_id)
            & quintiles["profile_id"].isin(["TriCasRank", "D", "WITHOUT_B", "WITHOUT_C"])
        ]
        q_lines_en = []
        q_lines_zh = []
        for profile in ("TriCasRank", "D", "WITHOUT_B", "WITHOUT_C"):
            rows = q.loc[q["profile_id"] == profile].sort_values("quintile")
            if not rows.empty:
                q_lines_en.append(f"{profile}: " + ", ".join(f"Q{int(r.quintile)}={_number(r.MAE)}" for r in rows.itertuples()))
                q_lines_zh.append(f"{profile}：" + "，".join(f"Q{int(r.quintile)}={_number(r.MAE)}" for r in rows.itertuples()))
        if q_lines_en:
            en_lines.append(f"  Processed-label quintile MAE (Q1 lowest numeric label to Q5 highest): {'; '.join(q_lines_en)}.")
            zh_lines.append(f"  processed-label 分位 MAE（Q1 数值最低至 Q5 数值最高）：{'；'.join(q_lines_zh)}。")
        support_rows = support.loc[support["protocol_id"] == protocol_id].set_index("support_type")
        if not support_rows.empty:
            guide = support_rows.loc["guide_support_n"]
            template = support_rows.loc["template_support_n"]
            en_lines.append(
                f"  Shared-ID support: guide zero={int(guide['zero_support_n'])}/{int(guide['record_n'])} "
                f"({float(guide['zero_support_fraction']):.3f}), median={_number(guide['median'])}; "
                f"template zero={int(template['zero_support_n'])}/{int(template['record_n'])} "
                f"({float(template['zero_support_fraction']):.3f}), median={_number(template['median'])}."
            )
            zh_lines.append(
                f"  共同ID的支持度：guide 零支持={int(guide['zero_support_n'])}/{int(guide['record_n'])} "
                f"（{float(guide['zero_support_fraction']):.3f}），中位数={_number(guide['median'])}；"
                f"template 零支持={int(template['zero_support_n'])}/{int(template['record_n'])} "
                f"（{float(template['zero_support_fraction']):.3f}），中位数={_number(template['median'])}。"
            )
    completed = ", ".join(x["protocol_id"] for x in summary.get("complete_protocols", [])) or "none"
    pending_scope = summary.get("scope", "core_dev completed protocols only")
    en = f"""# Component diagnostics summary (descriptive draft)

These profiles were generated from saved `component_diagnostics.csv` files for complete core_dev protocols only ({completed}). Each file was checked against the same-fold official OOF predictions, labels, support values, and record_id-derived protocol group/fold assignments. No model was fitted or refitted.

## Component and fixed-removal profiles

{chr(10).join(en_lines) if en_lines else 'No complete core_dev component diagnostic file is available.'}

`WITHOUT_D`, `WITHOUT_B` and `WITHOUT_C` are fixed-weight removal profiles with the prescribed D fallback when mapping is unavailable. `*_UNROUTED` profiles deliberately omit that fallback and are diagnostics only. Component correlations describe prediction and residual co-variation, not independent evidence or mechanism.

## Activity and support diagnostics

Activity bins are five equal-frequency quantiles of the supplied processed `y_true` values, ordered from the lowest to highest numeric value. The label transform direction is not inferred. Support summaries preserve the same query `record_id` set across profiles; support counts describe reference availability and are not additional model inputs for D or EasyDesign.

All tables are descriptive and add no P values, confidence intervals or significance family. Full numeric outputs and the ID-level source table are under `{prefix}`. Validation scope: {pending_scope}.
"""
    zh = f"""# 分量诊断总结（描述性草案）

以下 profile 来自已完成 core_dev 协议的保存文件 `component_diagnostics.csv`（{completed}）。每个文件均与同折官方 OOF 预测、标签、support 值以及按 record_id 回查的协议 group/fold assignment 核对；未拟合或重拟合模型。

## 分量与固定移除 profile

{chr(10).join(zh_lines) if zh_lines else '当前没有完整 core_dev 分量诊断文件。'}

`WITHOUT_D`、`WITHOUT_B` 和 `WITHOUT_C` 是固定权重移除 profile；mapping 不可用时按既定规则回退至 D。`*_UNROUTED` profile 特意不使用该 fallback，仅作诊断，不是部署预测。分量 prediction/residual 相关仅描述共变，不能视为独立证据或机制检验。

## 活性与支持度诊断

分箱基于所提供的 processed `y_true` 五个等频分位，从数值最低到最高排序；不推断标签变换方向。支持度汇总保持 profile 间共同 query `record_id` 集合；support 计数描述 reference 可用性，并非 D 或 EasyDesign 的额外模型输入。

所有表格均为描述性结果，不新增 P 值、置信区间或显著性检验家族。完整数值输出及 ID 级 source table 位于 `{prefix}`。验证范围：{pending_scope}。
"""
    return en, zh
