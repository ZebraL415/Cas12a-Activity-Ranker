"""ID-keyed input loading and pure audit helpers for the four fixed protocols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from features.sequence import SequenceManifest, sequence_manifest_from_frames


PROTOCOLS = ("target", "guide", "dual_atomic", "atomic_template")
GROUP_COLUMNS = {protocol: f"{protocol}_group" for protocol in PROTOCOLS}
FOLD_COLUMNS = {protocol: f"{protocol}_fold" for protocol in PROTOCOLS}
IDENTITY_COLUMNS = ("target_sequence", "crRNA_sequence", "mapping_candidate_template_nos")
ASSIGNMENT_COLUMNS = tuple(
    ["record_id"]
    + list(GROUP_COLUMNS.values())
    + list(FOLD_COLUMNS.values())
    + ["context_cv_fold"]
)


def _load_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    path = Path(config)
    return json.loads(path.read_text(encoding="utf-8"))


def _source_paths(config: dict[str, Any]) -> dict[str, Any]:
    paths = config.get("source_paths", config.get("inputs", {}))
    if not isinstance(paths, dict):
        raise ValueError("config.source_paths must be a mapping of input names to paths")
    return paths


def _resolve_path(raw_path: Any, *, key: str, paths: dict[str, Any]) -> Path:
    if raw_path is None:
        raise ValueError(f"config.source_paths.{key} is required")
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    base = paths.get("project_root", Path.cwd())
    return Path(str(base)).expanduser() / path


def _ids(frame: pd.DataFrame, source_name: str) -> pd.Series:
    if "record_id" not in frame:
        raise ValueError(f"{source_name} input is missing record_id")
    if frame["record_id"].isna().any():
        raise ValueError(f"{source_name} input contains missing record_id")
    result = frame["record_id"].astype(str)
    if result.eq("").any() or result.duplicated().any():
        raise ValueError(f"{source_name} input contains empty or duplicate record_id")
    return result


def _template_atoms(value: Any) -> set[str]:
    if pd.isna(value) or str(value).strip() in ("", "missing"):
        return set()
    atoms = {part.strip() for part in str(value).split(";") if part.strip()}
    invalid = sorted(atom for atom in atoms if not atom.isdigit())
    if invalid:
        raise ValueError(f"Invalid mapping template identifiers: {invalid[:5]}")
    return atoms


def _assignments_for_protocol(assignments: pd.DataFrame, protocol_id: str) -> tuple[str, str]:
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"Unknown protocol_id {protocol_id!r}; expected one of {PROTOCOLS}")
    group_col = GROUP_COLUMNS[protocol_id]
    fold_col = FOLD_COLUMNS[protocol_id]
    missing = sorted({group_col, fold_col} - set(assignments.columns))
    if missing:
        raise ValueError(f"Assignments are missing required columns for {protocol_id}: {missing}")
    return group_col, fold_col


def load_inputs(config: dict[str, Any] | str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SequenceManifest]:
    """Load development/history data and join assignments by ``record_id``.

    The returned development frame carries only the group/fold columns needed
    by the revision runner.  No source files are modified.
    """
    settings = _load_config(config)
    paths = _source_paths(settings)
    data_path = _resolve_path(paths.get("data"), key="data", paths=paths)
    assignment_path = _resolve_path(paths.get("assignments"), key="assignments", paths=paths)
    manifest_path = _resolve_path(paths.get("sequence_manifest"), key="sequence_manifest", paths=paths)
    template_path = _resolve_path(paths.get("template_reference"), key="template_reference", paths=paths)
    for name, path in (
        ("data", data_path),
        ("assignments", assignment_path),
        ("sequence_manifest", manifest_path),
        ("template_reference", template_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Configured {name} file does not exist: {path}")

    source = pd.read_csv(data_path, low_memory=False)
    if "baseline_split" not in source:
        raise ValueError("Development source is missing baseline_split")
    if "label_normalized" not in source:
        raise ValueError("Development source is missing label_normalized")
    source_ids = _ids(source, "source")
    source = source.copy()
    source["record_id"] = source_ids.to_numpy()
    dev_source = source.loc[source["baseline_split"].eq("baseline_train")].copy()
    historical = source.loc[source["baseline_split"].eq("baseline_validation")].copy()
    if dev_source.empty:
        raise ValueError("No baseline_train records were found")
    if historical.empty:
        raise ValueError("No baseline_validation records were found")

    assignments = pd.read_csv(assignment_path, dtype={"record_id": str}, low_memory=False)
    assignment_ids = _ids(assignments, "assignment")
    assignments = assignments.copy()
    assignments["record_id"] = assignment_ids.to_numpy()
    required_assignment_columns = {"record_id", *GROUP_COLUMNS.values(), *FOLD_COLUMNS.values()}
    missing_assignment = sorted(required_assignment_columns - set(assignments.columns))
    if missing_assignment:
        raise ValueError(f"Assignment input is missing required group/fold fields: {missing_assignment}")
    if "context_cv_fold" in assignments and not assignments["target_fold"].astype(int).equals(
        assignments["context_cv_fold"].astype(int)
    ):
        raise ValueError("target_fold and context_cv_fold disagree in the assignment source")

    dev_ids = set(dev_source["record_id"].astype(str))
    assignment_id_set = set(assignments["record_id"].astype(str))
    missing_assignments = sorted(dev_ids - assignment_id_set)
    extra_assignments = sorted(assignment_id_set - dev_ids)
    if missing_assignments or extra_assignments:
        raise ValueError(
            "Assignment coverage does not exactly match baseline_train record IDs: "
            f"missing={len(missing_assignments)}, extra={len(extra_assignments)}, "
            f"missing_examples={missing_assignments[:5]}, extra_examples={extra_assignments[:5]}"
        )

    assignment_payload = ["record_id", *GROUP_COLUMNS.values(), *FOLD_COLUMNS.values()]
    if "context_cv_fold" in assignments:
        assignment_payload.append("context_cv_fold")
    dev = dev_source.merge(
        assignments.loc[:, assignment_payload],
        on="record_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(dev) != len(dev_source) or dev[list(GROUP_COLUMNS.values()) + list(FOLD_COLUMNS.values())].isna().any().any():
        raise RuntimeError("ID-keyed assignment join changed row count or left group/fold values missing")
    for identity_column in IDENTITY_COLUMNS:
        if identity_column in assignments and identity_column in dev_source:
            check = dev_source[["record_id", identity_column]].merge(
                assignments[["record_id", identity_column]],
                on="record_id",
                how="left",
                validate="one_to_one",
                suffixes=("_source", "_assignment"),
                sort=False,
            )
            left = check[f"{identity_column}_source"].fillna("<NA>").astype(str)
            right = check[f"{identity_column}_assignment"].fillna("<NA>").astype(str)
            if not left.equals(right):
                raise ValueError(f"Assignment {identity_column} values do not match the source by record_id")

    feature_manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    template_reference = pd.read_csv(template_path, dtype=str, keep_default_na=False)
    sequence_manifest = sequence_manifest_from_frames(feature_manifest, template_reference)
    if not sequence_manifest.feature_names:
        raise ValueError("Sequence feature manifest is empty")
    return dev.reset_index(drop=True), historical.reset_index(drop=True), assignments.reset_index(drop=True), sequence_manifest


def make_encoding_fold_map(
    assignments: pd.DataFrame,
    protocol_id: str,
    *,
    outer_fold: int | None = None,
) -> dict[str, int]:
    """Return explicit protocol folds, optionally excluding the outer query fold."""
    group_column, fold_column = _assignments_for_protocol(assignments, protocol_id)
    del group_column
    selected = assignments
    if outer_fold is not None:
        selected = assignments.loc[pd.to_numeric(assignments[fold_column], errors="raise").ne(int(outer_fold))]
    values = pd.to_numeric(selected[fold_column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Assignment {fold_column} contains missing/non-numeric folds")
    ids = _ids(selected, "assignment fold map")
    result = dict(zip(ids.tolist(), values.astype(int).tolist()))
    if len(set(result.values())) < 2:
        raise ValueError("Mapping history cross-fit needs at least two remaining protocol folds")
    return result


def audit_protocol_assignments(dev: pd.DataFrame, assignments: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return pure data/group audit results; no writes or model fitting occur."""
    dev_ids = _ids(dev, "development")
    assignment_ids = _ids(assignments, "assignment")
    if set(dev_ids) != set(assignment_ids):
        raise ValueError("Development and assignment record_id sets differ")
    report: dict[str, Any] = {
        "n_dev_records": int(len(dev)),
        "n_unique_record_id": int(dev_ids.nunique()),
        "n_assignment_records": int(len(assignments)),
        "assignment_join_key": "record_id",
        "protocols": {},
        "protocol_rows": [],
    }
    rows: list[dict[str, Any]] = []
    for protocol_id in PROTOCOLS:
        group_column, fold_column = _assignments_for_protocol(assignments, protocol_id)
        if dev[group_column].isna().any() or dev[fold_column].isna().any():
            raise ValueError(f"Development frame contains missing {group_column}/{fold_column}")
        groups_per_fold = assignments.groupby(group_column, dropna=False)[fold_column].nunique(dropna=False)
        split_groups = int(groups_per_fold.gt(1).sum())
        if split_groups:
            raise ValueError(f"{protocol_id} isolation groups cross outer folds: {split_groups}")
        folds = pd.to_numeric(dev[fold_column], errors="coerce")
        if folds.isna().any():
            raise ValueError(f"{protocol_id} has missing/non-numeric outer fold values")
        fold_values = sorted(folds.astype(int).unique().tolist())
        by_fold = dev.assign(_audit_fold=folds.astype(int)).groupby("_audit_fold", sort=True)
        record_counts = {str(int(fold)): int(len(group)) for fold, group in by_fold}
        group_counts = {
            str(int(fold)): int(group[group_column].nunique()) for fold, group in by_fold
        }
        protocol_report = {
            "group_column": group_column,
            "fold_column": fold_column,
            "n_groups": int(dev[group_column].nunique()),
            "outer_folds": fold_values,
            "record_count_by_fold": record_counts,
            "group_count_by_fold": group_counts,
            "groups_crossing_folds": split_groups,
        }
        report["protocols"][protocol_id] = protocol_report

        for fold in fold_values:
            holdout_mask = folds.astype(int).eq(fold)
            fit = dev.loc[~holdout_mask]
            holdout = dev.loc[holdout_mask]
            row: dict[str, Any] = {
                "protocol_id": protocol_id,
                "outer_fold": int(fold),
                "n_fit": int(len(fit)),
                "n_query": int(len(holdout)),
                "fit_groups": int(fit[group_column].nunique()),
                "query_groups": int(holdout[group_column].nunique()),
                "group_overlap": int(len(set(fit[group_column]) & set(holdout[group_column]))),
                "record_id_overlap": int(len(set(fit["record_id"].astype(str)) & set(holdout["record_id"].astype(str)))),
                "target_sequence_overlap": None,
                "guide_sequence_overlap": None,
                "template_id_overlap": None,
            }
            if "target_sequence" in dev:
                row["target_sequence_overlap"] = int(
                    len(set(fit["target_sequence"].astype(str)) & set(holdout["target_sequence"].astype(str)))
                )
            if "crRNA_sequence" in dev:
                row["guide_sequence_overlap"] = int(
                    len(set(fit["crRNA_sequence"].astype(str)) & set(holdout["crRNA_sequence"].astype(str)))
                )
            if "mapping_candidate_template_nos" in dev:
                fit_templates = set().union(*(_template_atoms(v) for v in fit["mapping_candidate_template_nos"]))
                query_templates = set().union(*(_template_atoms(v) for v in holdout["mapping_candidate_template_nos"]))
                row["template_id_overlap"] = int(len(fit_templates & query_templates))
            rows.append(row)
    report["protocol_rows"] = rows
    return report, pd.DataFrame(rows)
