"""Ledger-driven validation and analysis of completed out-of-fold predictions."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import basic_metrics
from .resampling import (
    bootstrap_metric_draws,
    holm_adjust,
    make_group_swap_flags,
    percentile_interval,
    prediction_swap_test,
)


PROTOCOLS = ("target", "guide", "atomic_template", "dual_atomic")
COHORTS = ("core_dev", "common_dev")
CORE_MODELS = ("D", "HISTORY_PRIOR", "SEQ_MAPPING_XGB", "D_B_FIXED", "TriCasRank")
COMMON_MODELS = ("D", "TriCasRank", "EasyDesign-adapted")
METRIC_NAMES = ("SCC", "PCC", "RMSE", "MAE", "R2")
CACHE_SCHEMA = "revision_statistics_protocol_cache_v1"
PREDICTION_COLUMNS = (
    "run_id",
    "cohort_id",
    "protocol_id",
    "outer_fold",
    "group_id",
    "record_id",
    "model_id",
    "y_true",
    "y_pred",
    "prediction_route",
    "guide_support_n",
    "template_support_n",
    "config_sha256",
)


class AnalysisValidationError(ValueError):
    """Raised when a completed ledger run violates the prediction contract."""


def _to_python(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_python(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_python(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _comparison_output_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "pending_reason" not in frame.columns:
        frame["pending_reason"] = pd.Series(index=frame.index, dtype="object")
    return frame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        return json.loads(Path(config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("config must be a dict or a path to config/run.json")
    return config


def _model_roster(config: dict[str, Any], cohort_id: str) -> tuple[str, ...]:
    expected = CORE_MODELS if cohort_id == "core_dev" else COMMON_MODELS
    configured = config.get("cohorts", {}).get(cohort_id, {}).get("models")
    if configured is not None and set(map(str, configured)) != set(expected):
        raise AnalysisValidationError(
            f"{cohort_id} model roster in config differs from the frozen contract: "
            f"expected {list(expected)}, got {list(configured)}"
        )
    return expected


def _expected_n(config: dict[str, Any], cohort_id: str) -> int:
    value = config.get("cohorts", {}).get(cohort_id, {}).get("n")
    if value is None:
        return 8_417 if cohort_id == "core_dev" else 5_995
    return int(value)


def _expected_group_count(config: dict[str, Any], cohort_id: str, protocol_id: str) -> int | None:
    key = "group_counts_core" if cohort_id == "core_dev" else "group_counts_common"
    value = config.get(key, {}).get(protocol_id)
    return None if value is None else int(value)


def _load_source_tables(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    data_root = root / "data"
    labels_path = data_root / "development_labels.csv"
    assignments_path = data_root / "group_assignments.csv"
    common_ids_path = data_root / "common_record_ids.csv"
    for path in (labels_path, assignments_path, common_ids_path):
        if not path.is_file():
            raise AnalysisValidationError(f"required validation source is missing: {path}")
    labels = pd.read_csv(labels_path, dtype={"record_id": "string"})
    assignments = pd.read_csv(assignments_path, dtype={"record_id": "string"})
    common = pd.read_csv(common_ids_path, dtype={"record_id": "string"})
    if not {"record_id", "y_true"}.issubset(labels.columns):
        raise AnalysisValidationError(f"development label table lacks record_id/y_true: {labels_path}")
    if "record_id" not in assignments or "record_id" not in common:
        raise AnalysisValidationError("assignment/common-ID source tables must contain record_id")
    if labels["record_id"].duplicated().any() or assignments["record_id"].duplicated().any():
        raise AnalysisValidationError("development labels and group assignments must be unique by record_id")
    if common["record_id"].duplicated().any():
        raise AnalysisValidationError("common_record_ids.csv contains duplicate record_id values")
    labels["record_id"] = labels["record_id"].astype(str)
    assignments["record_id"] = assignments["record_id"].astype(str)
    common_ids = set(common["record_id"].astype(str))
    return labels, assignments, common_ids


def _expected_ids(
    cohort_id: str,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    common_ids: set[str],
    expected_n: int,
) -> set[str]:
    label_ids = set(labels["record_id"])
    assignment_ids = set(assignments["record_id"])
    ids = label_ids & assignment_ids
    if cohort_id == "common_dev":
        ids &= common_ids
    if len(ids) != expected_n:
        raise AnalysisValidationError(
            f"{cohort_id} source IDs do not match config: expected {expected_n}, found {len(ids)} "
            "after joining labels, assignments and cohort membership by record_id"
        )
    return ids


def _ledger_candidates(
    ledger: dict[str, Any],
    cohort_id: str,
    protocol_id: str,
    required_folds: tuple[int, ...],
    *,
    active_map_authoritative: bool = False,
    active_run_key: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    runs = ledger.get("runs")
    if not isinstance(runs, dict):
        raise AnalysisValidationError("memory/run_ledger.json must contain a runs object")
    if active_map_authoritative:
        if not active_run_key:
            return None, {
                "candidate_n": 0,
                "full_candidate_n": 0,
                "candidate_complete_folds": [],
                "reason": "no_active_run_mapping",
            }
        if active_run_key not in runs:
            return None, {
                "candidate_n": 0,
                "full_candidate_n": 0,
                "candidate_complete_folds": [],
                "reason": "active_run_key_not_found_in_ledger",
                "active_run_key": active_run_key,
            }
        active_entry = runs[active_run_key]
        if not isinstance(active_entry, dict):
            raise AnalysisValidationError(f"active ledger run {active_run_key!r} is not an object")
        if active_entry.get("cohort_id") != cohort_id or active_entry.get("protocol_id") != protocol_id:
            raise AnalysisValidationError(
                f"active ledger mapping {active_run_key!r} points to the wrong cohort/protocol; "
                f"expected {cohort_id}/{protocol_id}"
            )
    candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for run_key, entry in runs.items():
        if active_map_authoritative and str(run_key) != str(active_run_key):
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("cohort_id") != cohort_id or entry.get("protocol_id") != protocol_id:
            continue
        run_id = str(entry.get("run_id", ""))
        config_sha = str(entry.get("config_sha256", ""))
        input_sha = str(entry.get("input_sha256", ""))
        source_sha = str(entry.get("source_sha256", ""))
        if not all((run_id, config_sha, input_sha, source_sha)):
            raise AnalysisValidationError(f"ledger entry {run_key!r} lacks run_id or one of its SHA-256 fields")
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in (config_sha, input_sha, source_sha)):
            raise AnalysisValidationError(f"ledger entry {run_key!r} contains a malformed SHA-256 field")
        key = (run_id, config_sha, input_sha, source_sha)
        candidate = candidates.setdefault(
            key,
            {
                "cohort_id": cohort_id,
                "protocol_id": protocol_id,
                "run_id": run_id,
                "config_sha256": config_sha,
                "input_sha256": input_sha,
                "source_sha256": source_sha,
                "run_keys": [],
                "folds": {},
                "required_models": entry.get("required_models"),
                "run_states": [],
            },
        )
        candidate["run_keys"].append(str(run_key))
        candidate["run_states"].append(entry.get("state"))
        if candidate["required_models"] is not None and entry.get("required_models") is not None:
            if set(map(str, candidate["required_models"])) != set(map(str, entry["required_models"])):
                raise AnalysisValidationError(f"ledger entries grouped into {key!r} disagree on required_models")
        folds = entry.get("folds", {})
        if not isinstance(folds, dict):
            raise AnalysisValidationError(f"ledger entry {run_key!r} folds must be an object")
        for fold_key, fold in folds.items():
            if not isinstance(fold, dict) or fold.get("state") != "complete":
                continue
            try:
                fold_id = int(fold_key)
            except (TypeError, ValueError) as exc:
                raise AnalysisValidationError(f"ledger entry {run_key!r} has invalid fold key {fold_key!r}") from exc
            if fold_id not in required_folds:
                continue
            if fold_id in candidate["folds"]:
                old_path = candidate["folds"][fold_id].get("prediction_path")
                new_path = fold.get("prediction_path")
                if old_path != new_path:
                    raise AnalysisValidationError(
                        f"ledger has competing complete prediction paths for {cohort_id}/{protocol_id}/fold_{fold_id}"
                    )
            candidate["folds"][fold_id] = {**fold, "_ledger_run_key": str(run_key)}
    for candidate in candidates.values():
        candidate["run_state_complete"] = any(state == "complete" for state in candidate["run_states"])
    full = [
        candidate
        for candidate in candidates.values()
        if set(candidate["folds"]) == set(required_folds) and candidate["run_state_complete"]
    ]
    details = {
        "candidate_n": len(candidates),
        "full_candidate_n": len(full),
        "active_map_authoritative": active_map_authoritative,
        "active_run_key": active_run_key,
        "candidate_complete_folds": [
            {
                "run_id": candidate["run_id"],
                "complete_folds": sorted(candidate["folds"]),
                "run_keys": candidate["run_keys"],
                "run_states": candidate["run_states"],
                "run_state_complete": candidate["run_state_complete"],
            }
            for candidate in candidates.values()
        ],
    }
    if len(full) > 1:
        details["reason"] = "multiple_complete_ledger_lineages"
        return None, details
    if not full:
        full_folds_not_run_complete = any(
            set(candidate["folds"]) == set(required_folds) and not candidate["run_state_complete"]
            for candidate in candidates.values()
        )
        details["reason"] = "run_state_not_complete" if full_folds_not_run_complete else "fewer_than_all_required_folds_complete"
        return None, details
    chosen = full[0]
    roster = chosen.get("required_models")
    if roster is not None:
        expected = CORE_MODELS if cohort_id == "core_dev" else COMMON_MODELS
        if set(map(str, roster)) != set(expected):
            raise AnalysisValidationError(
                f"ledger run {chosen['run_id']} declares required_models {list(roster)}, expected {list(expected)}"
            )
    return chosen, details


def _read_fold_predictions(
    root: Path,
    candidate: dict[str, Any],
    fold_id: int,
    fold_metadata: dict[str, Any],
    expected_models: tuple[str, ...],
) -> pd.DataFrame:
    raw_path = fold_metadata.get("prediction_path")
    if not raw_path or not Path(str(raw_path)).is_absolute():
        raise AnalysisValidationError(
            f"ledger run {candidate['run_id']} fold {fold_id} must specify an absolute prediction_path"
        )
    path = Path(str(raw_path)).resolve()
    runs_root = (root / "runs").resolve()
    if not path.is_relative_to(runs_root):
        raise AnalysisValidationError(f"ledger prediction path escapes execution runs/: {path}")
    if path.name != "predictions.csv" or path.parent.name != f"fold_{fold_id}":
        raise AnalysisValidationError(
            f"ledger prediction path does not match fold_{fold_id}/predictions.csv: {path}"
        )
    if not path.is_file():
        raise AnalysisValidationError(f"ledger marks a missing prediction file complete: {path}")
    prediction_sha256 = _sha256_file(path)
    ledger_prediction_sha256 = fold_metadata.get("prediction_sha256")
    if ledger_prediction_sha256 is not None and prediction_sha256 != ledger_prediction_sha256:
        raise AnalysisValidationError(f"{path}: prediction file hash differs from the completed ledger record")
    frame = pd.read_csv(path, dtype={"record_id": "string", "group_id": "string"})
    if frame.empty:
        raise AnalysisValidationError(f"ledger marks an empty prediction file complete: {path}")
    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise AnalysisValidationError(f"prediction file {path} lacks required fields: {missing}")
    frame["record_id"] = frame["record_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["model_id"] = frame["model_id"].astype(str)
    frame.attrs["prediction_path"] = str(path)
    frame.attrs["prediction_sha256"] = prediction_sha256
    for col in ("outer_fold", "guide_support_n", "template_support_n"):
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.isna().any() or not np.all(numeric.to_numpy() == np.floor(numeric.to_numpy())):
            raise AnalysisValidationError(f"{path}: {col} must contain integers")
        frame[col] = numeric.astype(int)
    if (frame[["guide_support_n", "template_support_n"]] < 0).any().any():
        raise AnalysisValidationError(f"{path}: support counts must be non-negative")
    for col in ("y_true", "y_pred"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any() or not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise AnalysisValidationError(f"{path}: {col} contains missing or non-finite values")
    if frame[["record_id", "group_id", "model_id", "prediction_route", "config_sha256"]].isna().any().any():
        raise AnalysisValidationError(f"{path}: prediction identity or route fields contain missing values")
    for col in ("record_id", "group_id", "model_id", "prediction_route"):
        if frame[col].astype(str).str.len().eq(0).any():
            raise AnalysisValidationError(f"{path}: {col} contains empty strings")
    if frame["run_id"].astype(str).nunique() != 1 or frame["run_id"].astype(str).iloc[0] != candidate["run_id"]:
        raise AnalysisValidationError(f"{path}: row run_id does not match ledger lineage")
    if frame["config_sha256"].astype(str).nunique() != 1 or frame["config_sha256"].astype(str).iloc[0] != candidate["config_sha256"]:
        raise AnalysisValidationError(f"{path}: row config_sha256 does not match ledger lineage")
    if set(frame["cohort_id"].astype(str)) != {candidate["cohort_id"]}:
        raise AnalysisValidationError(f"{path}: cohort_id does not match ledger")
    if set(frame["protocol_id"].astype(str)) != {candidate["protocol_id"]}:
        raise AnalysisValidationError(f"{path}: protocol_id does not match ledger")
    if set(frame["outer_fold"].astype(int)) != {fold_id}:
        raise AnalysisValidationError(f"{path}: outer_fold does not match ledger fold key {fold_id}")
    if frame.duplicated(["model_id", "record_id"]).any():
        raise AnalysisValidationError(f"{path}: duplicate model_id/record_id prediction rows")
    actual_models = set(frame["model_id"].astype(str))
    if actual_models != set(expected_models):
        raise AnalysisValidationError(
            f"{path}: prediction model roster differs from contract; expected {list(expected_models)}, got {sorted(actual_models)}"
        )
    per_model_n = frame.groupby("model_id", sort=False)["record_id"].nunique()
    if per_model_n.nunique() != 1:
        raise AnalysisValidationError(f"{path}: model query ID counts do not match")
    query_n = int(per_model_n.iloc[0])
    if fold_metadata.get("n_query") is not None and int(fold_metadata["n_query"]) != query_n:
        raise AnalysisValidationError(
            f"{path}: distinct query IDs ({query_n}) differ from ledger n_query ({fold_metadata['n_query']})"
        )
    if len(frame) != query_n * len(expected_models):
        raise AnalysisValidationError(f"{path}: row count is inconsistent with query IDs and expected models")
    expected_routes = {
        "D": {"sequence"},
        "HISTORY_PRIOR": {"history_prior"},
        "SEQ_MAPPING_XGB": {"context"},
        "D_B_FIXED": {"context", "sequence_fallback"},
        "TriCasRank": {"context", "sequence_fallback"},
        "EasyDesign-adapted": {"easydesign"},
    }
    for model_id, route_values in frame.groupby("model_id")["prediction_route"]:
        allowed = expected_routes.get(model_id)
        if allowed is not None and not set(route_values.astype(str)).issubset(allowed):
            raise AnalysisValidationError(f"{path}: {model_id} has an invalid prediction_route")
    return frame


def _validate_fold_against_sources(
    frame: pd.DataFrame,
    *,
    cohort_id: str,
    protocol_id: str,
    fold_id: int,
    expected_models: tuple[str, ...],
    expected_cohort_ids: set[str],
    labels_by_id: dict[str, float],
    assignments: pd.DataFrame,
) -> None:
    group_col = f"{protocol_id}_group"
    fold_col = f"{protocol_id}_fold"
    if group_col not in assignments or fold_col not in assignments:
        raise AnalysisValidationError(f"assignment source lacks protocol columns {group_col}/{fold_col}")
    assignment = assignments.set_index("record_id")
    cohort_fold_ids = {
        record_id
        for record_id in expected_cohort_ids
        if int(assignment.at[record_id, fold_col]) == fold_id
    }
    model_id = expected_models[0]
    base = frame.loc[frame["model_id"] == model_id].copy()
    actual_ids = set(base["record_id"])
    if actual_ids != cohort_fold_ids:
        missing = sorted(cohort_fold_ids - actual_ids)[:10]
        extra = sorted(actual_ids - cohort_fold_ids)[:10]
        raise AnalysisValidationError(
            f"{cohort_id}/{protocol_id}/fold_{fold_id}: query IDs differ from assignment fold; "
            f"missing={missing}, extra={extra}"
        )
    base = base.set_index("record_id").sort_index()
    expected_groups = assignment.loc[base.index, group_col].astype(str)
    if not np.array_equal(base["group_id"].astype(str).to_numpy(), expected_groups.to_numpy()):
        raise AnalysisValidationError(f"{cohort_id}/{protocol_id}/fold_{fold_id}: group_id differs from source assignment")
    expected_y = np.asarray([labels_by_id[record_id] for record_id in base.index], dtype=float)
    actual_y = base["y_true"].to_numpy(dtype=float)
    if not np.allclose(actual_y, expected_y, rtol=0, atol=1e-12):
        mismatches = base.index[~np.isclose(actual_y, expected_y, rtol=0, atol=1e-12)].tolist()[:10]
        raise AnalysisValidationError(f"{cohort_id}/{protocol_id}/fold_{fold_id}: y_true mismatches source labels for {mismatches}")
    for col in ("group_id", "y_true", "outer_fold", "guide_support_n", "template_support_n"):
        distinct = frame.groupby("record_id", sort=False)[col].nunique(dropna=False)
        if (distinct > 1).any():
            raise AnalysisValidationError(
                f"{cohort_id}/{protocol_id}/fold_{fold_id}: compared methods disagree on {col} for a record"
            )
    if "mapping_found" in frame:
        distinct = frame.groupby("record_id", sort=False)["mapping_found"].nunique(dropna=False)
        if (distinct > 1).any():
            raise AnalysisValidationError(
                f"{cohort_id}/{protocol_id}/fold_{fold_id}: compared methods disagree on mapping_found"
            )


def _read_complete_protocol(
    root: Path,
    candidate: dict[str, Any],
    *,
    required_folds: tuple[int, ...],
    expected_models: tuple[str, ...],
    expected_cohort_ids: set[str],
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    if candidate["cohort_id"] == "common_dev":
        ed_hashes = [candidate["folds"][fold].get("ed_source_sha256") for fold in required_folds]
        if any(not value for value in ed_hashes):
            raise AnalysisValidationError(f"common_dev/{candidate['protocol_id']}: a complete fold lacks ed_source_sha256")
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in ed_hashes):
            raise AnalysisValidationError(f"common_dev/{candidate['protocol_id']}: malformed ed_source_sha256")
        if len(set(ed_hashes)) != 1:
            raise AnalysisValidationError(
                f"common_dev/{candidate['protocol_id']}: EasyDesign predictions come from mixed source hashes across folds"
            )
        candidate["easydesign_source_sha256"] = ed_hashes[0]
    labels_by_id = dict(zip(labels["record_id"].astype(str), pd.to_numeric(labels["y_true"], errors="raise")))
    fold_frames = []
    verified_prediction_files = []
    for fold_id in required_folds:
        fold_metadata = candidate["folds"][fold_id]
        frame = _read_fold_predictions(root, candidate, fold_id, fold_metadata, expected_models)
        verified_prediction_files.append(
            {
                "outer_fold": int(fold_id),
                "path": frame.attrs["prediction_path"],
                "sha256": frame.attrs["prediction_sha256"],
            }
        )
        _validate_fold_against_sources(
            frame,
            cohort_id=candidate["cohort_id"],
            protocol_id=candidate["protocol_id"],
            fold_id=fold_id,
            expected_models=expected_models,
            expected_cohort_ids=expected_cohort_ids,
            labels_by_id=labels_by_id,
            assignments=assignments,
        )
        fold_frames.append(frame)
    full = pd.concat(fold_frames, ignore_index=True)
    if full.duplicated(["model_id", "record_id"]).any():
        raise AnalysisValidationError(
            f"{candidate['cohort_id']}/{candidate['protocol_id']}: record appears in more than one outer fold"
        )
    per_model_ids = {
        model_id: set(full.loc[full["model_id"] == model_id, "record_id"])
        for model_id in expected_models
    }
    first = per_model_ids[expected_models[0]]
    if any(ids != first for ids in per_model_ids.values()):
        raise AnalysisValidationError(f"{candidate['cohort_id']}/{candidate['protocol_id']}: models do not share identical OOF IDs")
    if first != expected_cohort_ids:
        missing = sorted(expected_cohort_ids - first)[:10]
        extra = sorted(first - expected_cohort_ids)[:10]
        raise AnalysisValidationError(
            f"{candidate['cohort_id']}/{candidate['protocol_id']}: pooled OOF IDs differ from cohort; "
            f"missing={missing}, extra={extra}"
        )
    groups_per_record = full.groupby("record_id")["group_id"].nunique()
    if (groups_per_record > 1).any():
        raise AnalysisValidationError(f"{candidate['cohort_id']}/{candidate['protocol_id']}: group identity changes across folds")
    fold_per_record = full.groupby("record_id")["outer_fold"].nunique()
    if (fold_per_record > 1).any():
        raise AnalysisValidationError(f"{candidate['cohort_id']}/{candidate['protocol_id']}: record occurs in multiple folds")
    group_fold_counts = full.drop_duplicates(["record_id"])[["group_id", "outer_fold"]].groupby("group_id")["outer_fold"].nunique()
    if (group_fold_counts > 1).any():
        raise AnalysisValidationError(f"{candidate['cohort_id']}/{candidate['protocol_id']}: isolation group crosses outer folds")
    actual_group_count = int(full.drop_duplicates("record_id")["group_id"].nunique())
    if actual_group_count != len(set(full["group_id"])):
        raise AnalysisValidationError(f"{candidate['cohort_id']}/{candidate['protocol_id']}: inconsistent group identity")
    full.attrs["n_groups"] = actual_group_count
    full.attrs["run_metadata"] = {
        key: candidate[key]
        for key in ("run_id", "config_sha256", "input_sha256", "source_sha256", "run_keys")
    }
    candidate["verified_prediction_files"] = verified_prediction_files
    return full


def _metric_rows(frame: pd.DataFrame, models: tuple[str, ...], expected_folds: tuple[int, ...]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    pooled_constant: dict[str, float] = {}
    fold_constant: dict[str, list[bool]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    groups_by_id = frame.drop_duplicates("record_id").set_index("record_id")["group_id"]
    folds_by_id = frame.drop_duplicates("record_id").set_index("record_id")["outer_fold"]
    for model_id in models:
        model = frame.loc[frame["model_id"] == model_id].sort_values("record_id", kind="stable")
        y_true = model["y_true"].to_numpy(dtype=float)
        y_pred = model["y_pred"].to_numpy(dtype=float)
        constant = bool(np.ptp(y_pred) == 0)
        pooled_constant[model_id] = constant
        score = basic_metrics(y_true, y_pred)
        rows.append(
            {
                "evaluation_level": "pooled_oof",
                "outer_fold": None,
                "cohort_id": str(model["cohort_id"].iloc[0]),
                "protocol_id": str(model["protocol_id"].iloc[0]),
                "model_id": model_id,
                "record_n": int(len(model)),
                "group_n": int(model["group_id"].nunique()),
                "constant_prediction": constant,
                **score,
            }
        )
        for fold_id in expected_folds:
            fold = model.loc[model["outer_fold"] == fold_id]
            if fold.empty:
                raise AnalysisValidationError(f"completed protocol is missing {model_id}/fold_{fold_id} rows")
            fold_pred = fold["y_pred"].to_numpy(dtype=float)
            fold_is_constant = bool(np.ptp(fold_pred) == 0)
            fold_constant[model_id].append(fold_is_constant)
            fold_score = basic_metrics(fold["y_true"].to_numpy(dtype=float), fold_pred)
            rows.append(
                {
                    "evaluation_level": "outer_fold",
                    "outer_fold": int(fold_id),
                    "cohort_id": str(fold["cohort_id"].iloc[0]),
                    "protocol_id": str(fold["protocol_id"].iloc[0]),
                    "model_id": model_id,
                    "record_n": int(len(fold)),
                    "group_n": int(fold["group_id"].nunique()),
                    "constant_prediction": fold_is_constant,
                    **fold_score,
                }
            )
    fractions = {model: float(np.mean(fold_constant[model])) for model in models}
    for row in rows:
        row["constant_fold_fraction"] = fractions[row["model_id"]]
    return rows, fractions


def _diagnostic_rows(frame: pd.DataFrame, models: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    route_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    cohort_id = str(frame["cohort_id"].iloc[0])
    protocol_id = str(frame["protocol_id"].iloc[0])
    for model_id in models:
        model = frame.loc[frame["model_id"] == model_id].copy()
        route_counts = model["prediction_route"].value_counts(dropna=False).sort_index()
        for route, count in route_counts.items():
            route_rows.append(
                {
                    "cohort_id": cohort_id,
                    "protocol_id": protocol_id,
                    "model_id": model_id,
                    "prediction_route": str(route),
                    "record_n": int(count),
                    "fraction": float(count / len(model)),
                }
            )
        for column in ("guide_support_n", "template_support_n"):
            values = pd.to_numeric(model[column], errors="raise").to_numpy(dtype=float)
            support_rows.append(
                {
                    "cohort_id": cohort_id,
                    "protocol_id": protocol_id,
                    "model_id": model_id,
                    "support_type": column,
                    "record_n": int(values.size),
                    "zero_support_n": int(np.count_nonzero(values == 0)),
                    "zero_support_fraction": float(np.mean(values == 0)),
                    "min": float(np.min(values)),
                    "median": float(np.median(values)),
                    "max": float(np.max(values)),
                }
            )
        ordered = model.sort_values(["y_pred", "record_id"], kind="stable").reset_index(drop=True)
        bins = np.array_split(np.arange(len(ordered)), min(10, len(ordered)))
        for bin_index, positions in enumerate(bins, start=1):
            subset = ordered.iloc[positions]
            prediction = subset["y_pred"].to_numpy(dtype=float)
            truth = subset["y_true"].to_numpy(dtype=float)
            error = prediction - truth
            calibration_rows.append(
                {
                    "cohort_id": cohort_id,
                    "protocol_id": protocol_id,
                    "model_id": model_id,
                    "prediction_bin": int(bin_index),
                    "record_n": int(len(subset)),
                    "prediction_min": float(np.min(prediction)),
                    "prediction_max": float(np.max(prediction)),
                    "mean_prediction": float(np.mean(prediction)),
                    "mean_observed": float(np.mean(truth)),
                    "mean_error_prediction_minus_observed": float(np.mean(error)),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "mae": float(np.mean(np.abs(error))),
                }
            )
    return route_rows, support_rows, calibration_rows


def _comparison_pairs(cohort_id: str) -> tuple[str, ...]:
    return ("D", "HISTORY_PRIOR", "SEQ_MAPPING_XGB", "D_B_FIXED") if cohort_id == "core_dev" else ("EasyDesign-adapted",)


def _protocol_cache_identity(
    root: Path,
    config: dict[str, Any],
    run_metadata: dict[str, Any],
) -> tuple[str, Path, dict[str, Any]]:
    source_paths = [
        root / "data" / "development_labels.csv",
        root / "data" / "group_assignments.csv",
        root / "data" / "common_record_ids.csv",
    ]
    numerical_sources = [Path(__file__).with_name("metrics.py"), Path(__file__).with_name("resampling.py")]
    numerical_functions = {
        function.__name__: inspect.getsource(function)
        for function in (
            _comparison_pairs,
            _metric_rows,
            _diagnostic_rows,
            _comparison_rows_for_complete,
            _calibrate_holm,
        )
    }
    numerical_functions["constants"] = json.dumps(
        {
            "protocols": PROTOCOLS,
            "cohorts": COHORTS,
            "core_models": CORE_MODELS,
            "common_models": COMMON_MODELS,
            "metric_names": METRIC_NAMES,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    analyzer_numerical_digest = hashlib.sha256(
        json.dumps(numerical_functions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = {
        "schema": CACHE_SCHEMA,
        "cohort_id": run_metadata["cohort_id"],
        "protocol_id": run_metadata["protocol_id"],
        "run_lineage": {
            key: run_metadata.get(key)
            for key in (
                "run_id",
                "run_keys",
                "config_sha256",
                "input_sha256",
                "source_sha256",
                "easydesign_source_sha256",
                "prediction_files",
            )
        },
        "statistics_config": {
            key: config.get("statistics", {}).get(key)
            for key in (
                "primary_metric",
                "secondary_metrics",
                "bootstrap_draws",
                "permutation_draws",
                "seed",
                "cluster_unit",
                "core_holm_family_size",
                "common_holm_family_size",
                "constant_correlation",
                "bootstrap_min_valid_fraction",
            )
        },
        "model_roster": list(_model_roster(config, run_metadata["cohort_id"])),
        "outer_folds": int(config.get("outer_folds", 5)),
        "validation_sources": {str(path.relative_to(root)): _sha256_file(path) for path in source_paths},
        "statistics_code_sha256": {
            path.name: _sha256_file(path)
            for path in numerical_sources
        } | {"analyzer_numerical_functions": analyzer_numerical_digest},
    }
    serialized = json.dumps(_to_python(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    cache_path = root / "analysis" / "revision_statistics" / "cache" / f"{run_metadata['cohort_id']}__{run_metadata['protocol_id']}__{key}.json"
    return key, cache_path, identity


def _read_protocol_cache(path: Path, cache_key: str, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != CACHE_SCHEMA or payload.get("cache_key") != cache_key or payload.get("identity") != _to_python(identity):
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def _write_protocol_cache(
    path: Path,
    cache_key: str,
    identity: dict[str, Any],
    result: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {"schema": CACHE_SCHEMA, "cache_key": cache_key, "identity": identity, "result": result}
    temporary.write_text(
        json.dumps(_to_python(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _comparison_rows_for_complete(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    *,
    draws: int,
    seed: int,
    minimum_valid_fraction: float,
    cohort_id: str,
    protocol_id: str,
) -> list[dict[str, Any]]:
    ordered_ids = sorted(frame["record_id"].astype(str).unique())
    by_model = {
        model_id: frame.loc[frame["model_id"] == model_id].set_index("record_id").loc[ordered_ids]
        for model_id in models
    }
    reference = by_model["TriCasRank"]
    y_true = reference["y_true"].to_numpy(dtype=float)
    group_ids = reference["group_id"].astype(str).to_numpy()
    predictions = {model_id: by_model[model_id]["y_pred"].to_numpy(dtype=float) for model_id in models}
    # One draw stream per cohort/protocol is reused for all model contrasts and metrics.
    bootstrap = bootstrap_metric_draws(
        y_true,
        predictions,
        group_ids,
        draws=draws,
        seed=seed,
    )
    swap_flags = make_group_swap_flags(np.unique(group_ids).size, draws=draws, seed=seed)
    points = {model_id: basic_metrics(y_true, prediction) for model_id, prediction in predictions.items()}
    constant_fractions: dict[str, float] = {}
    for model_id in models:
        fold_flags = [
            bool(np.ptp(by_model[model_id].loc[by_model[model_id]["outer_fold"] == fold, "y_pred"].to_numpy(dtype=float)) == 0)
            for fold in range(5)
        ]
        constant_fractions[model_id] = float(np.mean(fold_flags))
    result = []
    for comparator in _comparison_pairs(cohort_id):
        row: dict[str, Any] = {
            "cohort_id": cohort_id,
            "protocol_id": protocol_id,
            "model_a": "TriCasRank",
            "model_b": comparator,
            "record_n": int(len(y_true)),
            "group_n": int(np.unique(group_ids).size),
            "constant_fold_fraction_a": constant_fractions["TriCasRank"],
            "constant_fold_fraction_b": constant_fractions[comparator],
            "core_family_size": 16 if cohort_id == "core_dev" else None,
            "common_family_size": 4 if cohort_id == "common_dev" else None,
        }
        for metric in METRIC_NAMES:
            a_point = points["TriCasRank"][metric]
            b_point = points[comparator][metric]
            delta = None if a_point is None or b_point is None else float(a_point - b_point)
            interval = percentile_interval(
                bootstrap["TriCasRank"][metric] - bootstrap[comparator][metric],
                minimum_valid_fraction=minimum_valid_fraction,
            )
            row[f"delta_{metric}"] = delta
            row[f"delta_{metric}_ci_low"] = interval["lower"]
            row[f"delta_{metric}_ci_high"] = interval["upper"]
            row[f"delta_{metric}_valid_draws"] = interval["valid_draws"]
            row[f"delta_{metric}_total_draws"] = interval["total_draws"]
            row[f"delta_{metric}_valid_fraction"] = interval["valid_fraction"]
            row[f"delta_{metric}_ci_status"] = interval["status"]
        swap = prediction_swap_test(
            y_true,
            predictions["TriCasRank"],
            predictions[comparator],
            group_ids,
            draws=draws,
            seed=seed,
            swap_flags=swap_flags,
        )
        row.update(
            {
                "p_raw": swap["p_value"],
                "p_holm": None,
                "permutation_valid_draws": swap["valid_draws"],
                "permutation_total_draws": swap["total_draws"],
                "permutation_status": swap["status"],
                "formal_family": "core_16_SCC" if cohort_id == "core_dev" else "common_4_SCC",
                "family_member_status": "tested" if swap["p_value"] is not None else "unavailable_undefined",
                "inference_scope": "protocol-group paired prediction-swap test; association difference, not mechanism",
            }
        )
        result.append(row)
    return result


def _pending_comparison_row(cohort_id: str, protocol_id: str, comparator: str, reason: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort_id": cohort_id,
        "protocol_id": protocol_id,
        "model_a": "TriCasRank",
        "model_b": comparator,
        "record_n": None,
        "group_n": None,
        "constant_fold_fraction_a": None,
        "constant_fold_fraction_b": None,
        "formal_family": "core_16_SCC" if cohort_id == "core_dev" else "common_4_SCC",
        "family_member_status": "protocol_pending",
        "pending_reason": reason,
        "inference_scope": "protocol-group paired prediction-swap test; association difference, not mechanism",
        "p_raw": None,
        "p_holm": None,
        "permutation_valid_draws": 0,
        "permutation_total_draws": None,
        "permutation_status": "protocol_pending",
    }
    for metric in METRIC_NAMES:
        row[f"delta_{metric}"] = None
        row[f"delta_{metric}_ci_low"] = None
        row[f"delta_{metric}_ci_high"] = None
        row[f"delta_{metric}_valid_draws"] = 0
        row[f"delta_{metric}_total_draws"] = None
        row[f"delta_{metric}_valid_fraction"] = None
        row[f"delta_{metric}_ci_status"] = "protocol_pending"
    return row


def _calibrate_holm(comparisons: list[dict[str, Any]], core_family_size: int, common_family_size: int) -> None:
    for cohort_id, family_size in (("core_dev", core_family_size), ("common_dev", common_family_size)):
        indices = [i for i, row in enumerate(comparisons) if row["cohort_id"] == cohort_id]
        adjusted = holm_adjust([comparisons[i].get("p_raw") for i in indices], family_size=family_size)
        for index, value in zip(indices, adjusted):
            comparisons[index]["p_holm"] = value
            comparisons[index]["family_size"] = family_size
            testable_n = sum(comparisons[j].get("p_raw") is not None for j in indices)
            comparisons[index]["family_testable_n"] = testable_n
            comparisons[index]["family_unavailable_n"] = family_size - testable_n
            comparisons[index]["family_complete"] = len(indices) == family_size and testable_n == family_size


def _execution_versions(root: Path) -> dict[str, Any]:
    return {
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "pytest": __import__("pytest").__version__ if __import__("importlib").util.find_spec("pytest") else None,
    }


def _method_drafts(config: dict[str, Any], versions: dict[str, Any], complete: list[dict[str, Any]], pending: list[dict[str, Any]]) -> tuple[str, str]:
    stats = config.get("statistics", {})
    bootstrap_n = int(stats.get("bootstrap_draws", 2_000))
    permutation_n = int(stats.get("permutation_draws", 2_000))
    seed = int(stats.get("seed", 20261012))
    complete_text = ", ".join(f"{x['cohort_id']}/{x['protocol_id']}" for x in complete) or "none"
    pending_text = ", ".join(f"{x['cohort_id']}/{x['protocol_id']}" for x in pending) or "none"
    sample_text = "; ".join(
        f"{x['cohort_id']}/{x['protocol_id']}: {x['record_n']:,} records in {x['group_n']:,} groups"
        for x in complete
    ) or "no complete cohort/protocol combination"
    en = f"""### Statistical analysis

Predictions were evaluated on the supplied processed-label scale using pooled out-of-fold (OOF) predictions and fold-specific summaries from five-fold fixed-recipe grouped cross-validation (CV). Each record contributed one query prediction per model; the protocol-specific isolation group, rather than the individual record, was the resampling unit. The complete combinations in this analysis contained {sample_text}. Spearman rank correlation (SCC; average ranks for ties) was the primary metric. Pearson correlation (PCC), root mean squared error (RMSE), mean absolute error (MAE) and the coefficient of determination (R²) were secondary metrics. Correlations were reported as missing when either vector was constant; R² was reported as missing when the observed labels were constant. The fraction of folds with constant predictions was recorded for each model.

For paired uncertainty, protocol-isolation groups were sampled with replacement {bootstrap_n:,} times using seed {seed}, drawing the original number of groups per replicate and retaining all records within each sampled group. The same resamples were used for all compared methods. Percentile 95% confidence intervals were computed for paired metric differences. An interval was withheld when fewer than 95% of bootstrap draws yielded a defined metric (the valid-draw count and reason are retained in the analysis tables). TriCasRank-versus-comparator SCC differences were tested using {permutation_n:,} two-sided protocol-group prediction-swap permutations with the same seed and add-one correction, p=(b+1)/(B+1), where b is the number of permuted absolute differences at least as large as the observed absolute difference. Predictions were swapped for whole groups while labels were held fixed. The core family comprised 16 planned comparisons (four protocols × four comparators: D, HISTORY_PRIOR, SEQ_MAPPING_XGB and D_B_FIXED); the common-context family comprised four comparisons with EasyDesign-adapted. Holm correction was applied separately to the prespecified families, with unavailable family members retained in the family denominator. These tests assess differences in group-level predictive association under the specified exchangeability procedure, not biological mechanisms. Prediction-ordered, near-equal-count bins are provided as descriptive calibration source data; no calibration model was fitted.

The analysis used Python {versions['python']}, NumPy {versions['numpy']}, pandas {versions['pandas']} and SciPy {versions['scipy']}. The present ledger snapshot contains complete analyses for {complete_text}; pending cohort/protocol combinations are {pending_text}. This fixed-recipe evaluation is not nested cross-validation and does not reconstruct the original model-selection process. All uncertainty is conditional on the frozen recipe and supplied isolation-group assignments.
"""
    zh = f"""### 统计分析

所有预测均在所提供的 processed-label 尺度上评估，并报告五折 fixed-recipe grouped cross-validation（CV）的 pooled out-of-fold（OOF）指标及逐折摘要。每条记录对每个模型提供一次查询预测；协议隔离组而非单条记录是重抽样单位。本次完整组合包含：{sample_text}。Spearman 秩相关系数（SCC；并列值采用平均秩）为主要指标；Pearson 相关系数（PCC）、均方根误差（RMSE）、平均绝对误差（MAE）及决定系数（R²）为次要指标。预测或真实标签为常数时，相关系数记为缺失；真实标签为常数时，R²记为缺失。另记录每个模型出现常数预测的折比例。

配对不确定性通过按协议隔离组有放回抽样估计：每次抽取与原数据相同数量的组，并保留所抽组内的全部记录，共进行 {bootstrap_n:,} 次抽样，随机种子为 {seed}；所有比较方法共用相同抽样。报告配对指标差值的百分位 95% 置信区间。相关系数或其他指标有效 bootstrap 抽样比例低于 95% 时不报告该区间，并在分析表中保留有效抽样数和缺失原因。TriCasRank 与比较方法的 SCC 差异采用 {permutation_n:,} 次双侧协议组 prediction-swap 置换检验（相同随机种子），并使用加一校正：p=(b+1)/(B+1)，其中 b 为绝对置换差值不小于观测绝对差值的次数。检验保持标签不变，对整组预测进行互换。核心比较家族预先包含 16 项（四种协议 × D、HISTORY_PRIOR、SEQ_MAPPING_XGB 与 D_B_FIXED 四个比较对象）；共同情境家族包含四项 TriCasRank 对 EasyDesign-adapted 比较。两家族分别进行 Holm 校正，未能检验的家族成员仍保留在家族分母中。该检验评估指定可交换条件下的组级预测关联差异，不检验生物学机制。另提供按预测值排序、近似等量分箱的描述性校准 source data；未拟合校准模型。

分析使用 Python {versions['python']}、NumPy {versions['numpy']}、pandas {versions['pandas']} 和 SciPy {versions['scipy']}。当前 ledger 中已完整分析的组合为：{complete_text}；仍待完成的组合为：{pending_text}。本固定配方评估并非嵌套交叉验证，也不重建原模型选择过程。所有不确定性均条件于冻结配方及所提供的隔离组划分。
"""
    return en, zh


def _fmt(value: Any, decimals: int = 3, *, signed: bool = True) -> str:
    number = _to_python(value)
    if number is None:
        return "not estimable"
    return f"{float(number):+.{decimals}f}" if signed else f"{float(number):.{decimals}f}"


def _metric_delta_evidence(row: dict[str, Any], metric: str) -> str:
    delta = row.get(f"delta_{metric}")
    lower = row.get(f"delta_{metric}_ci_low")
    upper = row.get(f"delta_{metric}_ci_high")
    valid = row.get(f"delta_{metric}_valid_draws")
    total = row.get(f"delta_{metric}_total_draws")
    if delta is None:
        return f"Δ{metric} not estimable"
    if lower is None or upper is None:
        interval = f"95% CI withheld ({valid}/{total} valid draws)"
    else:
        interval = f"95% CI [{float(lower):+.4f}, {float(upper):+.4f}]"
    return f"Δ{metric}={float(delta):+.4f} ({interval})"


def _comparison_evidence(row: dict[str, Any], metric: str = "SCC") -> str:
    p_holm = row.get("p_holm")
    constant_text = (
        f"constant-fold fraction (TriCasRank/comparator)="
        f"{_fmt(row.get('constant_fold_fraction_a'), signed=False)}/{_fmt(row.get('constant_fold_fraction_b'), signed=False)}"
    )
    p_text = "Holm-adjusted P not estimable" if p_holm is None else f"Holm-adjusted P={float(p_holm):.5f}"
    return "; ".join(
        [
            _metric_delta_evidence(row, metric),
            _metric_delta_evidence(row, "RMSE"),
            _metric_delta_evidence(row, "MAE"),
            p_text,
            constant_text,
        ]
    )


def _evidence_drafts(
    tables: dict[str, Any],
    complete: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    *,
    source_prefix: str,
) -> tuple[str, str, str]:
    metrics = pd.DataFrame(tables["metrics"])
    comparisons = pd.DataFrame(tables["comparisons"])
    background_en: list[str] = []
    background_zh: list[str] = []
    structure_en: list[str] = []
    structure_zh: list[str] = []
    context_en: list[str] = []
    context_zh: list[str] = []
    constant_warnings_en: list[str] = []
    constant_warnings_zh: list[str] = []
    common_en: list[str] = []
    common_zh: list[str] = []
    recommendation_lines: list[str] = []
    for item in complete:
        cohort_id = item["cohort_id"]
        protocol_id = item["protocol_id"]
        pooled = metrics.loc[
            (metrics["cohort_id"] == cohort_id)
            & (metrics["protocol_id"] == protocol_id)
            & (metrics["model_id"] == "TriCasRank")
            & (metrics["evaluation_level"] == "pooled_oof")
        ].iloc[0]
        absolute_en = (
            f"TriCasRank pooled OOF SCC={_fmt(pooled['SCC'], signed=False)}, PCC={_fmt(pooled['PCC'], signed=False)}, "
            f"RMSE={float(pooled['RMSE']):.3f}, MAE={float(pooled['MAE']):.3f} "
            f"(n={int(pooled['record_n']):,} records; {int(pooled['group_n']):,} protocol-isolation groups)."
        )
        absolute_zh = (
            f"TriCasRank pooled OOF SCC={_fmt(pooled['SCC'], signed=False)}，PCC={_fmt(pooled['PCC'], signed=False)}，"
            f"RMSE={float(pooled['RMSE']):.3f}，MAE={float(pooled['MAE']):.3f} "
            f"（n={int(pooled['record_n']):,} 条记录；{int(pooled['group_n']):,} 个协议隔离组）。"
        )
        c_rows = comparisons.loc[
            (comparisons["cohort_id"] == cohort_id)
            & (comparisons["protocol_id"] == protocol_id)
        ].to_dict(orient="records")
        by_model = {row["model_b"]: row for row in c_rows}
        pooled_rows = metrics.loc[
            (metrics["cohort_id"] == cohort_id)
            & (metrics["protocol_id"] == protocol_id)
            & (metrics["evaluation_level"] == "pooled_oof")
        ]
        for _, model_row in pooled_rows.loc[pooled_rows["constant_fold_fraction"] > 0].iterrows():
            constant_fold_n = int(round(float(model_row["constant_fold_fraction"]) * 5))
            constant_warnings_en.append(
                f"- **{cohort_id}/{protocol_id}/{model_row['model_id']}:** predictions were constant in {constant_fold_n}/5 folds "
                f"(constant_fold_fraction={float(model_row['constant_fold_fraction']):.3f}; pooled SCC={_fmt(model_row['SCC'], signed=False)}). "
                "A pooled correlation across fold-specific predictions should not be read as within-fold ranking ability for those constant-prediction folds."
            )
            constant_warnings_zh.append(
                f"- **{cohort_id}/{protocol_id}/{model_row['model_id']}：** 预测在 {constant_fold_n}/5 折为常数 "
                f"（constant_fold_fraction={float(model_row['constant_fold_fraction']):.3f}；pooled SCC={_fmt(model_row['SCC'], signed=False)}）。"
                "这些常数预测折不具备折内排序能力，跨折 pooled 相关不能解释为其折内排序能力。"
            )
        if cohort_id == "core_dev":
            context_en.append(f"- **{protocol_id}:** {absolute_en}")
            context_zh.append(f"- **{protocol_id}：** {absolute_zh}")
            bg_names = ("D", "HISTORY_PRIOR")
            structure_names = ("SEQ_MAPPING_XGB", "D_B_FIXED")
            bg_en = "; ".join(
                f"TriCasRank vs {name}: {_comparison_evidence(by_model[name])}"
                for name in bg_names
                if name in by_model
            )
            bg_zh = "；".join(
                f"TriCasRank 对 {name}：{_comparison_evidence(by_model[name])}"
                for name in bg_names
                if name in by_model
            )
            structure_text_en = "; ".join(
                f"TriCasRank vs {name}: {_comparison_evidence(by_model[name])}"
                for name in structure_names
                if name in by_model
            )
            structure_text_zh = "；".join(
                f"TriCasRank 对 {name}：{_comparison_evidence(by_model[name])}"
                for name in structure_names
                if name in by_model
            )
            background_en.append(f"- **{protocol_id}:** {bg_en}.")
            background_zh.append(f"- **{protocol_id}：** {bg_zh}。")
            structure_en.append(f"- **{protocol_id}:** {structure_text_en}.")
            structure_zh.append(f"- **{protocol_id}：** {structure_text_zh}。")
        else:
            row = by_model.get("EasyDesign-adapted")
            comparison_en = "" if row is None else f" Paired comparison with EasyDesign-adapted: {_comparison_evidence(row)}."
            comparison_zh = "" if row is None else f" 与 EasyDesign-adapted 的配对比较：{_comparison_evidence(row)}。"
            common_en.append(f"- **{protocol_id}:** {absolute_en}{comparison_en}")
            common_zh.append(f"- **{protocol_id}：** {absolute_zh}{comparison_zh}")
    pending_text_en = ", ".join(f"{x['cohort_id']}/{x['protocol_id']}" for x in pending) or "none"
    pending_text_zh = "、".join(f"{x['cohort_id']}/{x['protocol_id']}" for x in pending) or "无"
    if not common_en:
        common_en.append("No common_dev result is available yet; no core_dev prediction has been substituted for the cohort-matched common analysis.")
        common_zh.append("共同情境分析尚无完整结果；未用 core_dev 预测替代共同队列内的匹配分析。")
    en = f"""# Statistical evidence summary (draft)

## Background information value

The full fixed-recipe model is compared with the sequence-only D model and the HISTORY_PRIOR baseline in the core cohort. These paired differences quantify whether the full system improves ranking and error on the same protocol-isolated OOF records; they do not by themselves identify a biological mechanism.

{chr(10).join(background_en) if background_en else 'No complete core_dev protocol is available yet.'}

## Additional structure beyond mapping

Comparisons with SEQ_MAPPING_XGB and D_B_FIXED assess the evidence for gains beyond a strong single mapping model and the fixed D+B fusion. A confidence interval crossing zero is interpreted as uncertain additional ranking gain, not equivalence. Error differences are reported alongside SCC differences; negative RMSE/MAE differences favor TriCasRank.

{chr(10).join(structure_en) if structure_en else 'No complete core_dev structural-comparator result is available yet.'}

## Identity-context performance

Absolute pooled metrics are reported separately for target, guide, atomic-template and dual-atomic isolation. Differences across these protocols are descriptive; no additional between-protocol significance family was introduced.

{chr(10).join(context_en) if context_en else 'No complete core_dev protocol is available yet.'}

## Constant-fold interpretation

{chr(10).join(constant_warnings_en) if constant_warnings_en else 'No constant-prediction fold was detected in the currently complete cohort/protocol combinations.'}

Common-context analyses use predictions from the same 5,995-record cohort and are fitted within that cohort, rather than reusing core_dev predictions. Comparisons are with EasyDesign-adapted.

{chr(10).join(common_en)}

## Scope and source

Complete cohort/protocol combinations: {', '.join(f"{x['cohort_id']}/{x['protocol_id']}" for x in complete) or 'none'}. Pending combinations: {pending_text_en}. For unfinished families, unavailable comparisons remain in the prespecified 16/4 denominators as p=1 for Holm adjustment; `family_complete=false`, and interim adjusted P values will be recalculated when the remaining protocols complete. The fixed-recipe grouped evaluation does not remove model-selection bias inherited from the original development process (R03); it is not presented as complete nested cross-validation. This boundary applies to all comparisons and is stated here once.

Source tables: `{source_prefix}/metrics_pooled_and_fold.csv`, `{source_prefix}/paired_comparisons.csv`, and `{source_prefix}/oof_predictions.csv`. The last file is the record-level source data for C's plots and tables.
"""
    zh = f"""# 统计证据总结（草案）

## 背景信息价值

在 core cohort 中，将固定完整模型与仅序列模型 D、HISTORY_PRIOR 进行配对比较，用于判断完整系统是否在相同协议隔离 OOF 记录上改善排序和误差；这些比较本身不识别生物学机制。

{chr(10).join(background_zh) if background_zh else '当前尚无完整 core_dev 协议结果。'}

## 超越 mapping 的结构增益

与 SEQ_MAPPING_XGB 及固定 D+B 融合的比较，用于评估相对于强单一 mapping 模型和既定融合结构是否存在额外收益。若置信区间跨零，则解释为额外排序收益仍不确定，而非等效。误差差值与 SCC 差值并列报告；RMSE/MAE 差值为负表示 TriCasRank 误差较低。

{chr(10).join(structure_zh) if structure_zh else '当前尚无完整 core_dev 结构对照结果。'}

## 身份情境表现

分别报告 target、guide、atomic-template 与 dual-atomic 隔离下的 pooled 指标。不同协议之间的绝对指标差异仅作描述，不另增协议间显著性检验家族。

{chr(10).join(context_zh) if context_zh else '当前尚无完整 core_dev 协议结果。'}

## 常数预测折的解释

{chr(10).join(constant_warnings_zh) if constant_warnings_zh else '当前完整 cohort/protocol 组合中未发现常数预测折。'}

共同情境分析使用同一 5,995 条记录并在共同队列内拟合，不复用 core_dev 预测；比较对象为 EasyDesign-adapted。

{chr(10).join(common_zh)}

## 范围与来源

已完成的 cohort/protocol 组合：{', '.join(f"{x['cohort_id']}/{x['protocol_id']}" for x in complete) or '无'}。待完成组合：{pending_text_zh}。在尚未完成的比较家族中，不可用比较仍按预设的 16/4 家族分母以 p=1 参与 Holm 校正；`family_complete=false`，后续协议完成后会重新计算中间阶段的校正 P 值。固定配方分组评估并未消除原开发过程遗留的模型选择偏差（R03），也不被表述为完整嵌套交叉验证。该范围限制在此集中说明。

来源表：`{source_prefix}/metrics_pooled_and_fold.csv`、`{source_prefix}/paired_comparisons.csv` 与 `{source_prefix}/oof_predictions.csv`。最后一份文件提供 C 作图和制表所需的记录级 source data。
"""
    core_rows = comparisons.loc[comparisons.get("cohort_id") == "core_dev"].to_dict(orient="records") if not comparisons.empty else []
    structural = [
        row
        for row in core_rows
        if row.get("model_b") in ("SEQ_MAPPING_XGB", "D_B_FIXED")
        and row.get("family_member_status") != "protocol_pending"
    ]
    structurally_positive = [
        row
        for row in structural
        if row.get("delta_SCC") is not None
        and row.get("delta_SCC_ci_low") is not None
        and row.get("delta_SCC_ci_low") > 0
    ]
    structurally_uncertain = [
        row
        for row in structural
        if row.get("delta_SCC") is not None
        and (row.get("delta_SCC_ci_low") is None or row.get("delta_SCC_ci_low") <= 0 <= row.get("delta_SCC_ci_high"))
    ]
    structurally_negative = [
        row
        for row in structural
        if row.get("delta_SCC") is not None
        and row.get("delta_SCC_ci_high") is not None
        and row.get("delta_SCC_ci_high") < 0
    ]
    recommendation_lines.append(
        f"Among the {len(structural)} completed structural contrasts, {len(structurally_positive)} have 95% SCC-difference intervals entirely above zero, {len(structurally_negative)} entirely below zero, and {len(structurally_uncertain)} span zero or have an unavailable interval. This pattern supports protocol-specific interpretation rather than a universal incremental-ranking claim."
    )
    if structurally_negative:
        names = ", ".join(f"{row['protocol_id']} vs {row['model_b']}" for row in structurally_negative)
        recommendation_lines.append(f"The observed negative SCC-difference contrasts are {names}; these settings favor the comparator for rank correlation, while RMSE/MAE differences remain separately reported.")
    recommendation_lines.append(
        "The frozen TriCasRank recipe remains the model evaluated in this revision; this analysis does not select or refit a reduced model. If simplicity is a priority, use these protocol-specific estimates to motivate a future prespecified comparison, not a post hoc change to this run."
    )
    recommendation_lines.append(
        "Use the paired differences against D and HISTORY_PRIOR to describe background-information value, and SEQ_MAPPING_XGB/D_B_FIXED to describe incremental structure. Do not call a zero-crossing interval equivalence or use performance completion as a model-selection rule."
    )
    recommendation_lines.append(
        f"Evidence table: `{source_prefix}/paired_comparisons.csv`; absolute pooled/fold results: `{source_prefix}/metrics_pooled_and_fold.csv`. Pending combinations are {pending_text_en}."
    )
    recommendation = "# Model-retention evidence note\n\n" + "\n\n".join(recommendation_lines) + "\n"
    return en, zh, recommendation


def _write_outputs(root: Path, result: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    complete_metadata = result["complete_protocols"]
    diagnostic_lineage = json.dumps(
        result.get("component_diagnostics", {}).get("diagnostic_file_hashes", []),
        sort_keys=True,
        separators=(",", ":"),
    )
    lineage_text = "|".join(
        sorted(
            f"{x['cohort_id']}:{x['protocol_id']}:{x['run_id']}:{x['config_sha256']}:{x['input_sha256']}:{x['source_sha256']}:{x.get('easydesign_source_sha256')}:{x.get('statistics_cache_key')}"
            for x in complete_metadata
        )
    )
    fingerprint = hashlib.sha256(f"{lineage_text}|{diagnostic_lineage}".encode("utf-8")).hexdigest()[:12]
    analysis_dir = root / "analysis" / "revision_statistics" / fingerprint
    c_dir = root / "handoff" / "C" / "revision_statistics" / fingerprint
    d_dir = root / "handoff" / "D" / "revision_statistics" / fingerprint
    analysis_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)
    d_dir.mkdir(parents=True, exist_ok=True)
    outputs = result["_tables"]
    outputs["comparisons"] = _comparison_output_frame(outputs["comparisons"])
    table_paths = {
        "oof_predictions": analysis_dir / "oof_predictions.csv",
        "metrics": analysis_dir / "metrics_pooled_and_fold.csv",
        "comparisons": analysis_dir / "paired_comparisons.csv",
        "routes": analysis_dir / "prediction_routes.csv",
        "support": analysis_dir / "support_summary.csv",
        "calibration": analysis_dir / "calibration_bins.csv",
        "group_fold_index": analysis_dir / "group_fold_index.csv",
        "component_metric_profiles": analysis_dir / "component_metric_profiles.csv",
        "component_correlations": analysis_dir / "component_correlations.csv",
        "activity_quintile_mae": analysis_dir / "activity_quintile_mae.csv",
        "activity_quintile_edges": analysis_dir / "activity_quintile_edges.csv",
        "diagnostic_support_ids": analysis_dir / "diagnostic_support_ids.csv",
        "diagnostic_support_summary": analysis_dir / "diagnostic_support_summary.csv",
        "diagnostic_support_distribution": analysis_dir / "diagnostic_support_distribution.csv",
        "diagnostic_support_metric_profiles": analysis_dir / "diagnostic_support_metric_profiles.csv",
    }
    for key, path in table_paths.items():
        _write_csv(path, outputs[key])
    summary_path = analysis_dir / "analysis_summary.json"
    for key, path in table_paths.items():
        _write_csv(c_dir / path.name, outputs[key])
    versions = result["software"]
    en, zh = _method_drafts(config, versions, complete_metadata, result["pending_protocols"])
    (d_dir / "statistical_methods_en.md").write_text(en, encoding="utf-8")
    (d_dir / "statistical_methods_zh.md").write_text(zh, encoding="utf-8")
    terminology = """# Statistical terminology ledger

| Canonical term | Meaning in this analysis | Avoid conflating with |
|---|---|---|
| fixed-recipe grouped CV | Five-fold re-evaluation of the frozen model recipe with protocol-specific group isolation | Complete nested model-selection validation or independent external validation |
| OOF prediction | The prediction for a query record from the outer fold in which that record was held out | Prediction from a model fitted on the full cohort |
| protocol-isolation group | The group sampled as the independent cluster in paired bootstrap and prediction-swap tests | A biological replicate; independent biological replication was not established by these data |
| SCC | Spearman rank correlation, with average ranks for tied values | Percentage accuracy or classification accuracy |
| PCC | Pearson product-moment correlation | Rank correlation |
| RMSE / MAE / R² | Error and fit summaries computed on the supplied processed-label scale | Metrics on raw fluorescence or an inverse-transformed scale |
| constant-fold fraction | Fraction of the five outer folds with a constant prediction vector for a model | Pooled OOF correlation by itself |
| prediction-swap test | Two-sided group-level exchangeability test of a paired SCC difference with labels held fixed | Biological mechanism test |
| common_dev | Common-context query cohort in which the compared methods were refit on the same common training records | Core full-data predictions reused as common predictions |

Canonical model, cohort and protocol names follow the project terminology ledger.
"""
    (d_dir / "statistical_terminology_ledger.md").write_text(terminology, encoding="utf-8")
    evidence_en, evidence_zh, recommendation = _evidence_drafts(
        outputs,
        complete_metadata,
        result["pending_protocols"],
        source_prefix=str(analysis_dir.relative_to(root)),
    )
    (analysis_dir / "evidence_summary_en.md").write_text(evidence_en, encoding="utf-8")
    (analysis_dir / "evidence_summary_zh.md").write_text(evidence_zh, encoding="utf-8")
    (analysis_dir / "model_recommendation.md").write_text(recommendation, encoding="utf-8")
    from .diagnostics import draft_component_diagnostics_summary

    diagnostic_en, diagnostic_zh = draft_component_diagnostics_summary(
        outputs,
        result["component_diagnostics"],
        analysis_dir.relative_to(root),
    )
    (analysis_dir / "component_diagnostics_summary_en.md").write_text(diagnostic_en, encoding="utf-8")
    (analysis_dir / "component_diagnostics_summary_zh.md").write_text(diagnostic_zh, encoding="utf-8")
    (c_dir / "component_diagnostics_summary_en.md").write_text(diagnostic_en, encoding="utf-8")
    (c_dir / "component_diagnostics_summary_zh.md").write_text(diagnostic_zh, encoding="utf-8")
    (d_dir / "evidence_summary_en.md").write_text(evidence_en, encoding="utf-8")
    (d_dir / "evidence_summary_zh.md").write_text(evidence_zh, encoding="utf-8")
    (d_dir / "model_recommendation.md").write_text(recommendation, encoding="utf-8")
    (d_dir / "component_diagnostics_summary_en.md").write_text(diagnostic_en, encoding="utf-8")
    (d_dir / "component_diagnostics_summary_zh.md").write_text(diagnostic_zh, encoding="utf-8")
    qa_note = f"""# Statistical Methods drafting notes

Detected writing route: task=manuscript; paper_type=algorithmic; section=method; language=en with Chinese parallel text; journal=generic.

One-sentence argument: quantify fixed-recipe grouped out-of-fold performance and protocol-group paired uncertainty, with inference conditional on the frozen recipe and supplied isolation groups.

Section outline:

1. Define pooled/fold OOF evaluation and the record versus protocol-group units.
2. Define the five metrics and constant-vector handling.
3. Define group-paired bootstrap uncertainty, valid-draw rule, prediction-swap tests and separate Holm families.
4. State software, descriptive calibration outputs and the fixed-recipe inference boundary.

Assumptions or missing inputs: The supplied labels are retained on their inherited processed scale; no inverse transformation was inferred. No alpha threshold was supplied, so exact Holm-adjusted P values are reported without adding a threshold-based conclusion. The original selection process is not reconstructed.

Claim-evidence map:

- Claim: pooled and fold metrics are recomputed from complete OOF predictions. Evidence: `analysis_summary.json`, `metrics_pooled_and_fold.csv`, and `oof_predictions.csv`; status: supported for listed complete cohort/protocol combinations.
- Claim: pairwise uncertainty uses protocol groups and shared paired resamples. Evidence: `paired_comparisons.csv` and `statistics_config` in `analysis_summary.json`; status: supported by analysis implementation and synthetic tests.
- Claim: IDs, labels, groups, folds and compared-method support values are paired. Evidence: `group_fold_index.csv`, `oof_predictions.csv`, and source assignment/label files; status: validated for listed complete runs.
- Boundary: the fixed-recipe grouped evaluation does not remove the original model-selection bias (R03). Evidence: `DATA_VALIDATION_STATISTICS.md` and the frozen `fixed-recipe grouped CV` decision; status: explicitly retained.

Why this structure: It separates the prediction estimand, metric definitions, paired uncertainty and inference boundary so each number in C can be traced without re-fitting models.

To redirect me: Confirm any terminology changes before D merges this subsection with the full model Methods.
"""
    (d_dir / "statistical_methods_qa_notes.md").write_text(qa_note, encoding="utf-8")
    (root / "analysis" / "evidence_summary_en.md").write_text(evidence_en, encoding="utf-8")
    (root / "analysis" / "evidence_summary_zh.md").write_text(evidence_zh, encoding="utf-8")
    (root / "analysis" / "model_recommendation.md").write_text(recommendation, encoding="utf-8")
    c_readme = f"""# Statistical handoff for C

This bundle is generated from ledger-complete OOF prediction files only. Completed cohort/protocol pairs: {', '.join(f"{x['cohort_id']}/{x['protocol_id']}" for x in complete_metadata)}. Pending pairs are listed in `analysis_summary.json`; no partial-fold run was scored as a complete protocol.

- `oof_predictions.csv`: complete, ID-aligned prediction source data.
- `group_fold_index.csv`: record-level isolation group and outer-fold assignment.
- `metrics_pooled_and_fold.csv`: pooled and fold-specific metrics, including constant-fold fractions.
- `paired_comparisons.csv`: metric differences and group-bootstrap intervals, plus the fixed SCC test families and Holm adjustment.
- `prediction_routes.csv`, `support_summary.csv`: routing and support diagnostics.
- `calibration_bins.csv`: descriptive prediction-ordered bins; no inferential claim is attached.
- `component_metric_profiles.csv`, `component_correlations.csv`: fit-free metrics and prediction/residual correlations for saved D/B/C components, fixed removals and unrouted diagnostics.
- `activity_quintile_mae.csv`: MAE by processed-label quintile, descriptive only.
- `activity_quintile_edges.csv`: exact processed-label bin edges and record counts.
- `diagnostic_support_ids.csv`, `diagnostic_support_summary.csv`, `diagnostic_support_distribution.csv`, `diagnostic_support_metric_profiles.csv`: support and mapping availability on shared query IDs.
- `component_diagnostics_summary_en.md` and `component_diagnostics_summary_zh.md`: descriptive interpretation notes; no additional tests or plots.
- `analysis_summary.json`: source run IDs/hashes, completeness, pending items and checks.

Metric and comparison sources are under `{analysis_dir.relative_to(root)}`. No model was fitted or prediction changed by this analysis.
"""
    (c_dir / "README.md").write_text(c_readme, encoding="utf-8")
    result["artifacts"] = {
        "analysis_dir": str(analysis_dir),
        "handoff_c_dir": str(c_dir),
        "handoff_d_dir": str(d_dir),
        **{key: str(path) for key, path in table_paths.items()},
        "summary_json": str(summary_path),
        "statistical_methods_en": str(d_dir / "statistical_methods_en.md"),
        "statistical_methods_zh": str(d_dir / "statistical_methods_zh.md"),
        "statistical_terminology_ledger": str(d_dir / "statistical_terminology_ledger.md"),
        "statistical_methods_qa_notes": str(d_dir / "statistical_methods_qa_notes.md"),
        "evidence_summary_en": str(analysis_dir / "evidence_summary_en.md"),
        "evidence_summary_zh": str(analysis_dir / "evidence_summary_zh.md"),
        "model_recommendation": str(root / "analysis" / "model_recommendation.md"),
        "component_diagnostics_summary_en": str(analysis_dir / "component_diagnostics_summary_en.md"),
        "component_diagnostics_summary_zh": str(analysis_dir / "component_diagnostics_summary_zh.md"),
        "component_diagnostics_handoff_c_dir": str(c_dir),
        "component_diagnostics_dir": str(analysis_dir),
        "handoff_c_readme": str(c_dir / "README.md"),
    }
    _json_dump(summary_path, {key: value for key, value in result.items() if key != "_tables"})
    shutil.copy2(summary_path, c_dir / summary_path.name)
    result.pop("_tables", None)
    return result["artifacts"]


def analyze_execution(execution_root: str | Path, config: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Analyze only complete, ledger-recorded prediction folds in an execution root.

    The function verifies each prediction file against the ledger, cohort labels,
    protocol-specific source assignments and model roster before computing any
    pooled or paired statistics. A cohort/protocol is scored only when all five
    folds form one unambiguous complete run lineage.
    """
    root = Path(execution_root).resolve()
    cfg = _read_config(config)
    required_folds = tuple(range(int(cfg.get("outer_folds", 5))))
    if required_folds != tuple(range(5)):
        raise AnalysisValidationError(f"the frozen contract requires five folds (0–4), got {required_folds}")
    ledger_path = root / "memory" / "run_ledger.json"
    if not ledger_path.is_file():
        raise AnalysisValidationError(f"run ledger not found: {ledger_path}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    labels, assignments, common_ids = _load_source_tables(root, cfg)
    expected_ids = {
        cohort_id: _expected_ids(cohort_id, labels, assignments, common_ids, _expected_n(cfg, cohort_id))
        for cohort_id in COHORTS
    }
    labels_by_id = dict(zip(labels["record_id"].astype(str), pd.to_numeric(labels["y_true"], errors="raise")))
    completed_frames: dict[tuple[str, str], pd.DataFrame] = {}
    complete_metadata: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    lineage_audit: list[dict[str, Any]] = []
    for cohort_id in COHORTS:
        models = _model_roster(cfg, cohort_id)
        for protocol_id in PROTOCOLS:
            active_runs = ledger.get("active_runs")
            active_map_authoritative = isinstance(active_runs, dict)
            active_map_key = f"{cohort_id}:{protocol_id}"
            active_run_key = str(active_runs[active_map_key]) if active_map_authoritative and active_map_key in active_runs and active_runs[active_map_key] else None
            candidate, lineage = _ledger_candidates(
                ledger,
                cohort_id,
                protocol_id,
                required_folds,
                active_map_authoritative=active_map_authoritative,
                active_run_key=active_run_key,
            )
            lineage_audit.append({"cohort_id": cohort_id, "protocol_id": protocol_id, **lineage})
            if candidate is None:
                pending.append(
                    {
                        "cohort_id": cohort_id,
                        "protocol_id": protocol_id,
                        "status": "pending",
                        "reason": lineage.get("reason", "no_complete_ledger_lineage"),
                        "completed_folds_seen": sorted(
                            {
                                fold
                                for item in lineage.get("candidate_complete_folds", [])
                                for fold in item.get("complete_folds", [])
                            }
                        ),
                        "required_folds": list(required_folds),
                    }
                )
                continue
            frame = _read_complete_protocol(
                root,
                candidate,
                required_folds=required_folds,
                expected_models=models,
                expected_cohort_ids=expected_ids[cohort_id],
                labels=labels,
                assignments=assignments,
            )
            expected_groups = _expected_group_count(cfg, cohort_id, protocol_id)
            actual_groups = int(frame.attrs["n_groups"])
            if expected_groups is not None and actual_groups != expected_groups:
                raise AnalysisValidationError(
                    f"{cohort_id}/{protocol_id}: group count {actual_groups} differs from frozen expected {expected_groups}"
                )
            completed_frames[(cohort_id, protocol_id)] = frame
            complete_metadata.append(
                {
                    "cohort_id": cohort_id,
                    "protocol_id": protocol_id,
                    "run_id": candidate["run_id"],
                    "run_keys": candidate["run_keys"],
                    "config_sha256": candidate["config_sha256"],
                    "input_sha256": candidate["input_sha256"],
                    "source_sha256": candidate["source_sha256"],
                    "easydesign_source_sha256": candidate.get("easydesign_source_sha256"),
                    "record_n": int(frame["record_id"].nunique()),
                    "group_n": actual_groups,
                    "completed_folds": list(required_folds),
                    "prediction_paths": [candidate["folds"][fold]["prediction_path"] for fold in required_folds],
                    "prediction_files": candidate.get("verified_prediction_files", []),
                }
            )

    metric_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    protocol_cache_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    group_fold_frames: list[pd.DataFrame] = []
    stats = cfg.get("statistics", {})
    draws = int(stats.get("bootstrap_draws", 2_000))
    seed = int(stats.get("seed", 20261012))
    min_valid_fraction = float(stats.get("bootstrap_min_valid_fraction", 0.95))
    if draws != int(stats.get("permutation_draws", 2_000)):
        raise AnalysisValidationError("bootstrap and prediction-swap draws must follow the same frozen draw count")
    core_family_size = int(stats.get("core_holm_family_size", 16))
    common_family_size = int(stats.get("common_holm_family_size", 4))
    if core_family_size != 16 or common_family_size != 4:
        raise AnalysisValidationError("the frozen formal SCC families must contain 16 core and 4 common contrasts")
    for cohort_id in COHORTS:
        models = _model_roster(cfg, cohort_id)
        for protocol_id in PROTOCOLS:
            frame = completed_frames.get((cohort_id, protocol_id))
            if frame is None:
                reason = next(
                    item["reason"] for item in pending if item["cohort_id"] == cohort_id and item["protocol_id"] == protocol_id
                )
                comparisons.extend(
                    _pending_comparison_row(cohort_id, protocol_id, comparator, reason)
                    for comparator in _comparison_pairs(cohort_id)
                )
                continue
            run_metadata = next(
                item for item in complete_metadata if item["cohort_id"] == cohort_id and item["protocol_id"] == protocol_id
            )
            cache_key, cache_path, cache_identity = _protocol_cache_identity(root, cfg, run_metadata)
            cached_result = _read_protocol_cache(cache_path, cache_key, cache_identity)
            if cached_result is None:
                metric_result, _ = _metric_rows(frame, models, required_folds)
                routes, supports, calibration = _diagnostic_rows(frame, models)
                comparison_result = _comparison_rows_for_complete(
                    frame,
                    models,
                    draws=draws,
                    seed=seed,
                    minimum_valid_fraction=min_valid_fraction,
                    cohort_id=cohort_id,
                    protocol_id=protocol_id,
                )
                cached_result = {
                    "metrics": metric_result,
                    "routes": routes,
                    "support": supports,
                    "calibration": calibration,
                    "comparisons": comparison_result,
                }
                _write_protocol_cache(cache_path, cache_key, cache_identity, cached_result)
                cache_status = "computed_and_cached"
            else:
                cache_status = "reused_exact_lineage_cache"
            run_metadata["statistics_cache_key"] = cache_key
            run_metadata["statistics_cache_path"] = str(cache_path)
            run_metadata["statistics_cache_status"] = cache_status
            protocol_cache_records.append(
                {
                    "cohort_id": cohort_id,
                    "protocol_id": protocol_id,
                    "cache_key": cache_key,
                    "cache_path": str(cache_path),
                    "status": cache_status,
                    "statistics_code_sha256": cache_identity["statistics_code_sha256"],
                    "prediction_files": run_metadata["prediction_files"],
                }
            )
            metric_rows.extend(cached_result["metrics"])
            route_rows.extend(cached_result["routes"])
            support_rows.extend(cached_result["support"])
            calibration_rows.extend(cached_result["calibration"])
            comparisons.extend(cached_result["comparisons"])
            prediction_frames.append(frame)
            group_fold_frames.append(
                frame.drop_duplicates("record_id")[["cohort_id", "protocol_id", "record_id", "group_id", "outer_fold"]]
            )
    _calibrate_holm(comparisons, core_family_size, common_family_size)
    tables = {
        "oof_predictions": pd.concat(prediction_frames, ignore_index=True).sort_values(
            ["cohort_id", "protocol_id", "outer_fold", "record_id", "model_id"], kind="stable"
        )
        if prediction_frames
        else pd.DataFrame(columns=PREDICTION_COLUMNS),
        "metrics": metric_rows,
        "comparisons": comparisons,
        "routes": route_rows,
        "support": support_rows,
        "calibration": calibration_rows,
        "group_fold_index": pd.concat(group_fold_frames, ignore_index=True).sort_values(
            ["cohort_id", "protocol_id", "outer_fold", "record_id"], kind="stable"
        )
        if group_fold_frames
        else pd.DataFrame(columns=["cohort_id", "protocol_id", "record_id", "group_id", "outer_fold"]),
    }
    from .diagnostics import analyze_core_component_diagnostics

    component_tables, component_summary = analyze_core_component_diagnostics(
        root,
        complete_metadata,
        tables["oof_predictions"],
        labels,
        assignments,
    )
    tables.update(component_tables)
    versions = _execution_versions(root)
    result: dict[str, Any] = {
        "status": "complete" if not pending else ("partial" if complete_metadata else "pending"),
        "ledger_path": str(ledger_path),
        "source_paths": {
            "development_labels": str(root / "data" / "development_labels.csv"),
            "group_assignments": str(root / "data" / "group_assignments.csv"),
            "common_record_ids": str(root / "data" / "common_record_ids.csv"),
        },
        "software": versions,
        "statistics_config": {
            "primary_metric": "SCC",
            "secondary_metrics": list(METRIC_NAMES[1:]),
            "bootstrap_draws": draws,
            "permutation_draws": draws,
            "seed": seed,
            "cluster_unit": "protocol_isolation_group",
            "bootstrap_min_valid_fraction": min_valid_fraction,
            "core_holm_family_size": core_family_size,
            "common_holm_family_size": common_family_size,
            "correlation_tie_method": "average",
        },
        "complete_protocols": complete_metadata,
        "pending_protocols": pending,
        "ledger_lineage_audit": lineage_audit,
        "protocol_result_cache": protocol_cache_records,
        "component_diagnostics": component_summary,
        "checks": {
            "complete_folds_read_only_from_ledger": True,
            "validated_complete_protocol_n": len(completed_frames),
            "all_loaded_labels_match_development_labels": True if completed_frames else None,
            "all_loaded_ids_group_fold_assignments_match_source": True if completed_frames else None,
            "all_compared_methods_share_ids_labels_groups_support_and_fold": True if completed_frames else None,
            "protocol_group_isolation_validated": True if completed_frames else None,
            "pooled_or_fold_metrics_recomputed_from_predictions": True if completed_frames else None,
            "analysis_fitted_or_changed_models": False,
            "formal_family_sizes_preserved": True,
        },
        "_tables": tables,
    }
    if complete_metadata:
        result["artifacts"] = _write_outputs(root, result, cfg)
    else:
        result.pop("_tables")
    return _to_python(result)
