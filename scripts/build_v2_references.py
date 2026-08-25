#!/usr/bin/env python3
"""Freeze v2 deployment references from the authoritative EasyDesign training split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


MAPPING_FIELDS = (
    "mapping_candidate_group_ids",
    "mapping_status",
    "mapping_confidence",
    "mapping_candidate_template_nos",
    "mapping_match_modes",
    "mapping_candidate_orientations",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_sequence(value: object) -> str:
    text = "" if pd.isna(value) else str(value).upper()
    return re.sub(r"[^ACGTRYSWKMBDHVN]", "", text)


def normalized_key(series: pd.Series) -> pd.Series:
    return series.fillna("missing").astype(str).replace("", "missing")


def aggregate(
    frame: pd.DataFrame,
    *,
    reference_type: str,
    category: str = "",
) -> pd.DataFrame:
    values = frame.groupby("crRNA_sequence", sort=True)["label_normalized"].agg(["sum", "count"])
    values = values.reset_index()
    values.insert(0, "reference_type", reference_type)
    values.insert(2, "category", category)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--canonical-table", type=Path, required=True)
    parser.add_argument("--source-workbook", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = root / "models" / "mapping"
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.canonical_table, low_memory=False)
    train = data.loc[data["baseline_split"].eq("baseline_train")].reset_index(drop=True)
    if len(train) != 8417:
        raise RuntimeError(f"Expected 8,417 frozen training rows, found {len(train)}")
    if train["record_id"].duplicated().any():
        raise RuntimeError("Frozen training record_id values are not unique")

    exact = train["aligned_difference_count"].eq(0)
    diffcat = np.where(exact, "exact", np.where(train["aligned_difference_count"].eq(1), "one", "two_plus"))
    has_gap = train["gap_count_target"].fillna(0).add(train["gap_count_guide"].fillna(0)).gt(0)
    guide_parts = [
        aggregate(train, reference_type="all"),
        aggregate(train.loc[exact], reference_type="exact"),
        aggregate(train.loc[~exact], reference_type="nonexact"),
    ]
    for category in ("exact", "one", "two_plus"):
        guide_parts.append(
            aggregate(train.loc[diffcat == category], reference_type="diffcat", category=category)
        )
    for category, mask in (("gap", has_gap), ("no_gap", ~has_gap)):
        guide_parts.append(
            aggregate(train.loc[mask], reference_type="gapcat", category=category)
        )
    guide_reference = pd.concat(guide_parts, ignore_index=True)
    guide_reference.to_csv(output / "guide_history_reference.csv", index=False)

    mapping_parts: list[pd.DataFrame] = []
    for field in MAPPING_FIELDS:
        temp = pd.DataFrame(
            {
                "field": field,
                "key": normalized_key(train[field]),
                "label_normalized": train["label_normalized"].to_numpy(float),
            }
        )
        stats = temp.groupby(["field", "key"], sort=True)["label_normalized"].agg(["sum", "count"])
        mapping_parts.append(stats.reset_index())
    mapping_reference = pd.concat(mapping_parts, ignore_index=True)
    mapping_reference.to_csv(output / "mapping_history_reference.csv", index=False)

    raw_templates = pd.read_excel(args.source_workbook, sheet_name="Table S2", header=1)
    template_rows: list[dict[str, object]] = []
    for source in raw_templates.itertuples(index=False):
        template_no = int(getattr(source, "_0"))
        sequence = clean_sequence(getattr(source, "Sequence"))
        group = (template_no - 1) // 9 + 1
        template_rows.append(
            {
                "template_no": template_no,
                "template_group_id": f"EasyDesign_2024_template_group_{group:02d}",
                "template_sequence_clean": sequence,
                "template_length": len(sequence),
            }
        )
    templates = pd.DataFrame(template_rows)
    if len(templates) != 198:
        raise RuntimeError(f"Expected 198 Table S2 templates, found {len(templates)}")
    templates.to_csv(output / "table_s2_template_reference.csv", index=False)

    metadata = {
        "version": "2.0.0",
        "reference_scope": "EasyDesign baseline_train only",
        "train_rows": int(len(train)),
        "unique_guides": int(train["crRNA_sequence"].nunique()),
        "global_train_label_mean": float(train["label_normalized"].mean()),
        "guide_smoothing": 0.5,
        "mapping_smoothing": 20.0,
        "template_rows": int(len(templates)),
        "canonical_table_sha256": sha256(args.canonical_table),
        "source_workbook_sha256": sha256(args.source_workbook),
        "guide_reference_rows": int(len(guide_reference)),
        "mapping_reference_rows": int(len(mapping_reference)),
        "leakage_control": "No baseline_validation labels are present in either history reference.",
    }
    (output / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
