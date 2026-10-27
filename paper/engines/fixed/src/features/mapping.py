"""Path-independent implementation of the frozen Table S2 template mapper."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

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
    """Map 25-nt aligned targets to embedded template records.

    ``records`` consists only of (template_no, template_group_id,
    template_sequence_clean), so an instance/state can be serialized without
    retaining the historical repository path.
    """

    def __init__(self, records: Iterable[tuple[str, str, str]]):
        cleaned = tuple((str(number), str(group), str(sequence)) for number, group, sequence in records)
        if not cleaned:
            raise ValueError("TemplateMapper requires at least one template record")
        self.records = cleaned
        self._exact_indexes: dict[int, dict[str, list[dict[str, Any]]]] = {}

    def _exact_index(self, length: int) -> dict[str, list[dict[str, Any]]]:
        if length in self._exact_indexes:
            return self._exact_indexes[length]
        index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for template_no, group_id, raw_sequence in self.records:
            sequence = raw_sequence.strip().upper().replace("U", "T")
            for start in range(0, len(sequence) - length + 1):
                window = sequence[start : start + length]
                if re.fullmatch(r"[ACGT]+", window):
                    index[window].append(
                        {
                            "template_no": int(template_no),
                            "template_group_id": group_id,
                            "start_0based": start,
                        }
                    )
        self._exact_indexes[length] = index
        return index

    def _iupac_hits(self, query: str, orientation: str) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for template_no, group_id, raw_sequence in self.records:
            sequence = raw_sequence.strip().upper().replace("U", "T")
            for start in range(0, len(sequence) - len(query) + 1):
                window = sequence[start : start + len(query)]
                if window != query and compatible(query, window):
                    hits.append(
                        {
                            "template_no": int(template_no),
                            "template_group_id": group_id,
                            "start_0based": start,
                            "orientation": orientation,
                            "match_mode": "iupac_compatible",
                        }
                    )
        return hits

    def map_target(self, target_aligned_25: Any) -> dict[str, Any]:
        if pd.isna(target_aligned_25):
            raise ValueError("Missing target_aligned_25 for mapping")
        aligned = str(target_aligned_25).strip().upper().replace("U", "T")
        if len(aligned) != 25:
            raise ValueError(f"target_aligned_25 must contain exactly 25 positions; got {len(aligned)}")
        invalid = sorted(set(aligned) - (set(IUPAC) | {"-"}))
        if invalid:
            raise ValueError(f"Invalid characters in target_aligned_25 for mapping: {invalid}")
        target = aligned.replace("-", "")
        if not target:
            raise ValueError("target_aligned_25 cannot contain gaps only")
        exact_index = self._exact_index(len(target))
        candidates: list[dict[str, Any]] = []
        for orientation, query in (("forward", target), ("reverse_complement", reverse_complement(target))):
            for hit in exact_index.get(query, []):
                candidates.append(
                    {
                        **hit,
                        "orientation": orientation,
                        "match_mode": "exact_acgt",
                    }
                )
        if not candidates:
            for orientation, query in (("forward", target), ("reverse_complement", reverse_complement(target))):
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
        target_column = "target_aligned_25" if "target_aligned_25" in pairs else "target_sequence"
        if target_column not in pairs:
            raise ValueError("Input must contain target_aligned_25 (or target_sequence) for mapping")
        rows = [self.map_target(value) for value in pairs[target_column]]
        return pd.DataFrame(rows, index=pairs.index, columns=list(MAPPING_FIELDS))


def mapping_frame_from_input(frame: pd.DataFrame, mapper: TemplateMapper | None) -> pd.DataFrame:
    """Use complete approved label-free mapping fields or generate all fields."""
    present = [column for column in MAPPING_FIELDS if column in frame.columns]
    if present and len(present) != len(MAPPING_FIELDS):
        missing = sorted(set(MAPPING_FIELDS) - set(present))
        raise ValueError(f"Partial mapping fields are not accepted; missing: {missing}")
    if len(present) == len(MAPPING_FIELDS):
        return frame.loc[:, list(MAPPING_FIELDS)].copy()
    if mapper is None:
        raise ValueError("Mapping fields are absent and no embedded template reference is available")
    return mapper.map_frame(frame)
