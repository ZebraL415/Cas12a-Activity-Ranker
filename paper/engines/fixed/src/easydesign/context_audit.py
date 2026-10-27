"""No-fit audit of the frozen common-context subset and protocol fold IDs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adapter import PROTOCOLS, encode_sequences, prepare_partition


EXPECTED_STATUS_COUNTS = {
    "ambiguous_context": 2279,
    "incomplete_flanks": 142,
    "no_mapping": 1,
    "ok": 5995,
}
EXPECTED_COMMON_GROUP_COUNTS = {
    "target": 5923,
    "guide": 1304,
    "atomic_template": 20,
    "dual_atomic": 19,
}
EXPECTED_COMMON_FOLD_COUNTS = {
    "target": (1199, 1198, 1197, 1202, 1199),
    "guide": (1193, 1186, 1206, 1178, 1232),
    "atomic_template": (1304, 1141, 1235, 1133, 1182),
    "dual_atomic": (1406, 1082, 1205, 1145, 1157),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(raw: Any, *, project_root: Path) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else project_root / path


def audit_common_context_partitions(
    config: Mapping[str, Any] | str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recheck context admission and all four common-cohort fold partitions.

    Admission is read from the frozen, sequence-derived ``context_status``;
    labels are used only to verify that the source joins are consistent and
    are not used to select records or compute the partition.
    """

    if isinstance(config, Mapping):
        settings = dict(config)
    else:
        settings = json.loads(Path(config).expanduser().read_text(encoding="utf-8"))
    paths = settings.get("source_paths", settings.get("inputs", {}))
    project_root = Path(str(paths.get("project_root", Path.cwd()))).expanduser()
    data_path = _resolve_path(paths.get("data"), project_root=project_root)
    assignment_path = _resolve_path(paths.get("assignments"), project_root=project_root)
    context_path = _resolve_path(paths.get("easydesign_contexts"), project_root=project_root)
    for name, path in (
        ("development data", data_path),
        ("assignments", assignment_path),
        ("EasyDesign context table", context_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Configured {name} is absent: {path}")

    contexts = pd.read_csv(context_path, dtype={"record_id": str}, low_memory=False)
    if contexts["record_id"].isna().any() or contexts["record_id"].astype(str).duplicated().any():
        raise ValueError("EasyDesign context table record_id must be complete and unique")
    baseline_train = contexts.loc[contexts["baseline_split"].eq("baseline_train")].copy()
    status_counts = {
        str(key): int(value)
        for key, value in baseline_train["context_status"].value_counts(dropna=False).items()
    }
    if status_counts != EXPECTED_STATUS_COUNTS:
        raise ValueError(
            "Frozen baseline_train context_status counts changed: "
            f"expected {EXPECTED_STATUS_COUNTS}, observed {status_counts}"
        )
    common_context = baseline_train.loc[baseline_train["context_status"].eq("ok")].copy()
    common_ids = set(common_context["record_id"].astype(str))

    dev = pd.read_csv(data_path, low_memory=False)
    dev = dev.loc[dev["baseline_split"].eq("baseline_train"), [
        "record_id",
        "label_normalized",
        "crRNA_sequence",
    ]].copy()
    dev = dev.rename(columns={"crRNA_sequence": "dev_crRNA_sequence"})
    dev["record_id"] = dev["record_id"].astype(str)
    if dev["record_id"].duplicated().any():
        raise ValueError("baseline_train development record_id is not unique")
    dev_ids = set(dev["record_id"])
    if common_ids - dev_ids:
        raise ValueError(f"Context-approved IDs absent from development data: {sorted(common_ids-dev_ids)[:5]}")

    assignments = pd.read_csv(assignment_path, dtype={"record_id": str}, low_memory=False)
    if assignments["record_id"].isna().any() or assignments["record_id"].duplicated().any():
        raise ValueError("Protocol assignment record_id must be complete and unique")
    assignment_ids = set(assignments["record_id"].astype(str))
    if common_ids - assignment_ids:
        raise ValueError(f"Context-approved IDs absent from protocol assignments: {sorted(common_ids-assignment_ids)[:5]}")
    if "crRNA_sequence" in assignments.columns:
        assignments = assignments.rename(columns={"crRNA_sequence": "assignment_crRNA_sequence"})

    joined = common_context[["record_id", "target_context_45", "guide", "y_true"]].merge(
        dev,
        on="record_id",
        how="left",
        validate="one_to_one",
    ).merge(
        assignments,
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    if len(joined) != len(common_ids) or joined["target_fold"].isna().any():
        raise ValueError("Common-context record_id joins changed row count or lost assignments")
    context_y = pd.to_numeric(joined["y_true"], errors="coerce").to_numpy(float)
    dev_y = pd.to_numeric(joined["label_normalized"], errors="coerce").to_numpy(float)
    if not np.isfinite(context_y).all() or not np.isfinite(dev_y).all():
        raise ValueError("Common context and development labels must be finite for join audit")
    if not np.array_equal(context_y.astype(np.float32), dev_y.astype(np.float32)):
        raise ValueError("Context y_true and development label_normalized disagree after ID join")
    guide = joined["guide"].astype(str).to_numpy()
    if not np.array_equal(guide, joined["dev_crRNA_sequence"].astype(str).to_numpy()):
        raise ValueError("Context guide and development crRNA_sequence disagree after ID join")
    if "assignment_crRNA_sequence" in joined.columns and not np.array_equal(
        guide, joined["assignment_crRNA_sequence"].astype(str).to_numpy()
    ):
        raise ValueError("Context guide and assignment crRNA_sequence disagree after ID join")

    common_groups: dict[str, int] = {}
    common_folds: dict[str, list[int]] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    overlap_check: dict[str, int] = {}
    for protocol in PROTOCOLS:
        fold_column = f"{protocol}_fold"
        group_column = f"{protocol}_group"
        missing = sorted({fold_column, group_column} - set(joined.columns))
        if missing:
            raise ValueError(f"Common assignment join misses {protocol} fields: {missing}")
        fold_values = pd.to_numeric(joined[fold_column], errors="coerce")
        if fold_values.isna().any() or not fold_values.isin(range(5)).all():
            raise ValueError(f"Invalid {protocol} fold values in common context")
        group_n = int(joined[group_column].nunique(dropna=True))
        fold_counts = tuple(
            int(fold_values.eq(fold).sum())
            for fold in range(5)
        )
        if group_n != EXPECTED_COMMON_GROUP_COUNTS[protocol]:
            raise ValueError(
                f"{protocol} common group count changed: expected "
                f"{EXPECTED_COMMON_GROUP_COUNTS[protocol]}, observed {group_n}"
            )
        if fold_counts != EXPECTED_COMMON_FOLD_COUNTS[protocol]:
            raise ValueError(
                f"{protocol} common fold row counts changed: expected "
                f"{EXPECTED_COMMON_FOLD_COUNTS[protocol]}, observed {fold_counts}"
            )
        group_fold_counts = joined.groupby(group_column, dropna=False)[fold_column].nunique(dropna=False)
        crossing = int(group_fold_counts.gt(1).sum())
        if crossing:
            raise ValueError(f"{crossing} {protocol} groups cross outer folds in common context")

        common_groups[protocol] = group_n
        common_folds[protocol] = list(fold_counts)
        split_rows[protocol] = []
        for outer_fold in range(5):
            expected_stop_fold = (outer_fold + 1) % 5
            fit = joined.loc[fold_values.ne(outer_fold)].copy().reset_index(drop=True)
            query_full = joined.loc[fold_values.eq(outer_fold)].copy().reset_index(drop=True)
            stop_ids = fit.loc[pd.to_numeric(fit[fold_column], errors="coerce").eq(expected_stop_fold), "record_id"].tolist()
            # Prevent query-label access at the adapter boundary exactly as in
            # the coordinator.  Fit rows retain only the approved processed label.
            query = query_full.drop(
                columns=[
                    column
                    for column in ("y_true", "label_normalized", "label_raw_value", "label_raw_name")
                    if column in query_full
                ],
            )
            partition = prepare_partition(
                fit,
                stop_ids,
                query,
                outer_fold=outer_fold,
                protocol_id=protocol,
            )
            split_rows[protocol].append(
                {
                    "outer_fold": outer_fold,
                    "stop_fold": expected_stop_fold,
                    "n_outer_fit": len(partition.fit_ids),
                    "n_inner_fit": len(partition.inner_fit_ids),
                    "n_stop": len(partition.stop_ids),
                    "n_query": len(partition.query_ids),
                    "n_groups_outer_fit": int(fit[group_column].nunique()),
                    "n_groups_inner_fit": int(
                        fit.loc[~pd.Series(partition.stop_mask), group_column].nunique()
                    ),
                    "n_groups_stop": int(
                        fit.loc[pd.Series(partition.stop_mask), group_column].nunique()
                    ),
                    "n_groups_query": int(query[group_column].nunique()),
                    "assignment_check": partition.assignment_check,
                    "query_label_columns_absent": True,
                }
            )
        overlap_check[protocol] = crossing

    encoded = encode_sequences(joined[["target_context_45", "guide"]])
    if encoded.shape != (len(joined), 45, 10) or not np.isfinite(encoded).all():
        raise ValueError("The 5,995 common records do not produce finite 45×10 tensors")

    report: dict[str, Any] = {
        "status": "pass",
        "admission_rule": "baseline_train AND context_status == ok; label values do not determine selection",
        "context_source": str(context_path),
        "context_source_sha256": _sha256(context_path),
        "development_source_sha256": _sha256(data_path),
        "assignment_source_sha256": _sha256(assignment_path),
        "baseline_train_context_status_counts": status_counts,
        "n_baseline_train_context_rows": len(baseline_train),
        "n_common_context_records": len(joined),
        "n_unique_common_record_ids": len(common_ids),
        "n_common_ids_missing_from_development": 0,
        "n_common_ids_missing_from_assignments": 0,
        "context_y_true_equals_development_label_normalized_after_record_id_join": True,
        "context_guide_equals_development_and_assignment_crRNA_sequence_after_record_id_join": True,
        "context_encoding": {
            "shape": list(encoded.shape[1:]),
            "n_records": int(encoded.shape[0]),
            "finite": bool(np.isfinite(encoded).all()),
            "gap_channel_index_within_each_alphabet": 4,
        },
        "common_group_counts": common_groups,
        "common_fold_record_counts": common_folds,
        "groups_crossing_outer_folds": overlap_check,
        "fold_partition_checks": split_rows,
        "training_was_run": False,
    }
    if output_path is not None:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite context audit: {destination}")
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return report
