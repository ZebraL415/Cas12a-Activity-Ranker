"""Deterministic mapping of aligned targets to frozen EasyDesign templates."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


IUPAC = {
    "A": frozenset("A"),
    "C": frozenset("C"),
    "G": frozenset("G"),
    "T": frozenset("T"),
    "R": frozenset("AG"),
    "Y": frozenset("CT"),
    "S": frozenset("GC"),
    "W": frozenset("AT"),
    "K": frozenset("GT"),
    "M": frozenset("AC"),
    "B": frozenset("CGT"),
    "D": frozenset("AGT"),
    "H": frozenset("ACT"),
    "V": frozenset("ACG"),
    "N": frozenset("ACGT"),
}

MAPPING_FIELDS = (
    "mapping_status",
    "mapping_confidence",
    "mapping_hit_count",
    "mapping_template_count",
    "mapping_group_count",
    "mapping_orientation_count",
    "mapping_candidate_template_nos",
    "mapping_candidate_group_ids",
    "mapping_candidate_orientations",
    "mapping_match_modes",
)


def reverse_complement(sequence: str) -> str:
    table = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")
    return sequence.translate(table)[::-1]


def compatible(query: str, template_window: str) -> bool:
    return len(query) == len(template_window) and all(
        bool(IUPAC.get(left, frozenset()) & IUPAC.get(right, frozenset()))
        for left, right in zip(query, template_window)
    )


class TemplateMapper:
    """Map every ungapped target against all frozen Table S2 template windows."""

    def __init__(self, reference_path: str | Path):
        templates = pd.read_csv(reference_path, dtype=str, keep_default_na=False)
        required = {
            "template_no",
            "template_group_id",
            "template_sequence_clean",
        }
        missing = sorted(required - set(templates.columns))
        if missing:
            raise ValueError(f"Template reference is missing columns: {missing}")
        self.templates = templates.reset_index(drop=True)
        self._exact_indexes: dict[int, dict[str, list[dict[str, Any]]]] = {}

    def _exact_index(self, length: int) -> dict[str, list[dict[str, Any]]]:
        if length in self._exact_indexes:
            return self._exact_indexes[length]
        index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for template in self.templates.itertuples(index=False):
            sequence = str(template.template_sequence_clean)
            for start in range(0, len(sequence) - length + 1):
                window = sequence[start : start + length]
                if re.fullmatch(r"[ACGT]+", window):
                    index[window].append(
                        {
                            "template_no": int(template.template_no),
                            "template_group_id": str(template.template_group_id),
                            "start_0based": start,
                        }
                    )
        self._exact_indexes[length] = index
        return index

    def _iupac_hits(self, query: str, orientation: str) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for template in self.templates.itertuples(index=False):
            sequence = str(template.template_sequence_clean)
            for start in range(0, len(sequence) - len(query) + 1):
                window = sequence[start : start + len(query)]
                if window != query and compatible(query, window):
                    hits.append(
                        {
                            "template_no": int(template.template_no),
                            "template_group_id": str(template.template_group_id),
                            "start_0based": start,
                            "orientation": orientation,
                            "match_mode": "iupac_compatible",
                        }
                    )
        return hits

    def map_target(self, target_aligned_25: Any) -> dict[str, Any]:
        target = str(target_aligned_25).strip().upper().replace("U", "T").replace("-", "")
        exact_index = self._exact_index(len(target))
        candidates: list[dict[str, Any]] = []
        for orientation, query in (
            ("forward", target),
            ("reverse_complement", reverse_complement(target)),
        ):
            for hit in exact_index.get(query, []):
                candidates.append(
                    {
                        **hit,
                        "orientation": orientation,
                        "match_mode": "exact_acgt",
                    }
                )
        if not candidates:
            for orientation, query in (
                ("forward", target),
                ("reverse_complement", reverse_complement(target)),
            ):
                candidates.extend(self._iupac_hits(query, orientation))

        template_nos = sorted({int(hit["template_no"]) for hit in candidates})
        group_ids = sorted({str(hit["template_group_id"]) for hit in candidates})
        orientations = sorted({str(hit["orientation"]) for hit in candidates})
        positions = {
            (int(hit["template_no"]), int(hit["start_0based"]), str(hit["orientation"]))
            for hit in candidates
        }
        modes = sorted({str(hit["match_mode"]) for hit in candidates})
        if not candidates:
            status, confidence = "unmapped_exact_or_iupac", "review"
        elif len(positions) == 1 and modes == ["exact_acgt"]:
            status, confidence = "unique_exact_window", "high"
        elif len(template_nos) == 1 and modes == ["exact_acgt"]:
            status, confidence = "ambiguous_position_same_template", "medium"
        elif len(group_ids) == 1 and modes == ["exact_acgt"]:
            status, confidence = "ambiguous_template_single_group", "medium"
        elif modes == ["iupac_compatible"] and len(group_ids) == 1:
            status, confidence = "iupac_compatible_single_group", "review"
        elif len(group_ids) > 1:
            status, confidence = "ambiguous_multiple_groups", "review"
        else:
            status, confidence = "ambiguous_source_mapping", "review"
        return {
            "mapping_status": status,
            "mapping_confidence": confidence,
            "mapping_hit_count": len(candidates),
            "mapping_template_count": len(template_nos),
            "mapping_group_count": len(group_ids),
            "mapping_orientation_count": len(orientations),
            "mapping_candidate_template_nos": ";".join(map(str, template_nos)),
            "mapping_candidate_group_ids": ";".join(group_ids),
            "mapping_candidate_orientations": ";".join(orientations),
            "mapping_match_modes": ";".join(modes),
        }

    def map_frame(self, pairs: pd.DataFrame) -> pd.DataFrame:
        rows = [self.map_target(value) for value in pairs["target_aligned_25"]]
        return pd.DataFrame(rows, index=pairs.index)
