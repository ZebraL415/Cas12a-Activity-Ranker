"""Pure feature reconstruction for the frozen 25-position sequence profile.

The scalar feature definitions are a path-independent port of
``ranker_v1_5_worktree/src/cas12a_ml/features.py``.  This module performs no
I/O and does not infer an alignment for external candidates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DINUCLEOTIDES = tuple(a + b for a in "ACGT" for b in "ACGT")
TRINUCLEOTIDES = ("AAA", "CCC", "GGG", "TTT", "GCG", "CGC", "TAT", "ATA")
PAIR_ALPHABET = ("A", "C", "G", "T", "-", "N")
PAIR_CATEGORIES = tuple(left + right for left in PAIR_ALPHABET for right in PAIR_ALPHABET)
SEQUENCE_INPUT_COLUMNS = ("crRNA_sequence", "target_aligned_25")


@dataclass(frozen=True)
class SequenceManifest:
    """Approved sequence-feature names plus embedded, path-free map references."""

    feature_names: tuple[str, ...]
    feature_categories: tuple[tuple[str, str], ...] = ()
    template_records: tuple[tuple[str, str, str], ...] = ()


def sequence_manifest_from_frames(
    feature_manifest: pd.DataFrame,
    template_reference: pd.DataFrame | None = None,
) -> SequenceManifest:
    if "feature_name" not in feature_manifest:
        raise ValueError("Sequence feature manifest must include feature_name")
    names = tuple(feature_manifest["feature_name"].astype(str))
    if len(names) != len(set(names)):
        raise ValueError("Sequence feature manifest contains duplicate names")
    categories: tuple[tuple[str, str], ...] = ()
    if "category" in feature_manifest:
        categories = tuple(
            (str(name), str(category))
            for name, category in feature_manifest[["feature_name", "category"]].itertuples(
                index=False, name=None
            )
        )
    records: tuple[tuple[str, str, str], ...] = ()
    if template_reference is not None:
        required = {"template_no", "template_group_id", "template_sequence_clean"}
        missing = sorted(required - set(template_reference.columns))
        if missing:
            raise ValueError(f"Template reference is missing columns: {missing}")
        records = tuple(
            (str(template_no), str(group_id), str(sequence))
            for template_no, group_id, sequence in template_reference[
                ["template_no", "template_group_id", "template_sequence_clean"]
            ].itertuples(index=False, name=None)
        )
    return SequenceManifest(names, categories, records)


def coerce_sequence_manifest(value: Any) -> SequenceManifest:
    if isinstance(value, SequenceManifest):
        return value
    if isinstance(value, pd.DataFrame):
        return sequence_manifest_from_frames(value)
    if isinstance(value, Mapping):
        names = value.get("feature_names", value.get("feature_name", ()))
        if isinstance(names, pd.Series):
            names = names.tolist()
        categories = value.get("feature_categories", ())
        templates = value.get("template_records", ())
        return SequenceManifest(
            tuple(str(name) for name in names),
            tuple((str(name), str(category)) for name, category in categories),
            tuple((str(number), str(group), str(sequence)) for number, group, sequence in templates),
        )
    if isinstance(value, (list, tuple)):
        return SequenceManifest(tuple(str(name) for name in value))
    raise TypeError("sequence_manifest must be a SequenceManifest, mapping, DataFrame, or name list")


def _normalize(value: Any, field: str, *, allow_gap: bool) -> str:
    if pd.isna(value):
        raise ValueError(f"Missing {field}")
    sequence = str(value).strip().upper().replace("U", "T")
    allowed = set("ACGT-") if allow_gap else set("ACGT")
    invalid = sorted(set(sequence) - allowed)
    if invalid:
        raise ValueError(f"Invalid characters in {field}: {invalid}")
    if len(sequence) != 25:
        raise ValueError(f"{field} must contain exactly 25 aligned positions; got {len(sequence)}")
    if not allow_gap and "-" in sequence:
        raise ValueError(f"{field} must not contain alignment gaps")
    if allow_gap and sequence.replace("-", "") == "":
        raise ValueError(f"{field} cannot contain gaps only")
    return sequence


def normalize_sequence_pair(guide_value: Any, target_value: Any) -> tuple[str, str]:
    return (
        _normalize(guide_value, "crRNA_sequence", allow_gap=False),
        _normalize(target_value, "target_aligned_25", allow_gap=True),
    )


def _entropy(sequence: str) -> float | None:
    sequence = "".join(base for base in sequence if base in "ACGT")
    if not sequence:
        return None
    value = 0.0
    for base in "ACGT":
        fraction = sequence.count(base) / len(sequence)
        if fraction:
            value -= fraction * math.log2(fraction)
    return round(value, 6)


def _longest_homopolymer(sequence: str) -> int:
    sequence = "".join(base for base in sequence if base in "ACGT")
    if not sequence:
        return 0
    longest = current = 1
    for previous, current_base in zip(sequence, sequence[1:]):
        current = current + 1 if current_base == previous else 1
        longest = max(longest, current)
    return longest


def _longest_run(values: Iterable[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _base_features(prefix: str, sequence: str) -> dict[str, float | int | None]:
    acgt = "".join(base for base in sequence if base in "ACGT")
    result: dict[str, float | int | None] = {
        f"{prefix}_length_acgt": len(acgt),
        f"{prefix}_gc_content": round((acgt.count("G") + acgt.count("C")) / len(acgt), 6)
        if acgt
        else None,
        f"{prefix}_shannon_entropy": _entropy(acgt),
        f"{prefix}_longest_homopolymer": _longest_homopolymer(acgt),
    }
    for base in "ACGT":
        result[f"{prefix}_{base.lower()}_fraction"] = round(acgt.count(base) / len(acgt), 6) if acgt else None
    return result


def _pair_features(guide: str, target: str) -> dict[str, float | int | None]:
    events: list[str] = []
    substitutions = {f"sub_{a}_to_{b}_count": 0 for a in "ACGT" for b in "ACGT" if a != b}
    result: dict[str, float | int | None] = {}
    for index, (guide_base, target_base) in enumerate(zip(guide, target), start=1):
        if guide_base == "-" and target_base != "-":
            event = "gap_in_guide"
        elif target_base == "-" and guide_base != "-":
            event = "gap_in_target"
        elif guide_base == target_base:
            event = "match"
        else:
            event = "substitution"
            substitutions[f"sub_{guide_base}_to_{target_base}_count"] += 1
        events.append(event)
        result[f"difference_pos_{index:02d}"] = int(event in {"substitution", "gap_in_target", "gap_in_guide"})
        result[f"substitution_pos_{index:02d}"] = int(event == "substitution")
        result[f"target_gap_pos_{index:02d}"] = int(event == "gap_in_target")

    differences = [event != "match" for event in events]
    positions = [index + 1 for index, value in enumerate(differences) if value]
    result.update(substitutions)
    result.update(
        {
            "aligned_difference_count": sum(differences),
            "substitution_count": events.count("substitution"),
            "gap_count_target": events.count("gap_in_target"),
            "gap_count_guide": events.count("gap_in_guide"),
            "pam_difference_count": sum(differences[:4]),
            "spacer_difference_count": sum(differences[4:]),
            "first_difference_position_1based": min(positions) if positions else np.nan,
            "last_difference_position_1based": max(positions) if positions else np.nan,
            "longest_match_run": _longest_run(event == "match" for event in events),
            "longest_substitution_run": _longest_run(event == "substitution" for event in events),
            "longest_target_gap_run": _longest_run(event == "gap_in_target" for event in events),
        }
    )
    return result


def _fraction(sequence: str, bases: str) -> float:
    return sum(base in bases for base in sequence) / len(sequence)


def _kmer_frequency(sequence: str, kmer: str) -> float:
    denominator = len(sequence) - len(kmer) + 1
    if denominator <= 0:
        return 0.0
    return sum(sequence[index : index + len(kmer)] == kmer for index in range(denominator)) / denominator


def _context_features(sequence: str, prefix: str) -> dict[str, float | int]:
    midpoint = max(1, len(sequence) // 2)
    result: dict[str, float | int] = {
        f"{prefix}_context_unique_base_count": len(set(sequence)),
        f"{prefix}_context_gc_first_half": _fraction(sequence[:midpoint], "GC"),
        f"{prefix}_context_gc_second_half": _fraction(sequence[midpoint:], "GC"),
        f"{prefix}_context_gc_first_5nt": _fraction(sequence[:5], "GC"),
        f"{prefix}_context_gc_last_5nt": _fraction(sequence[-5:], "GC"),
    }
    for kmer in DINUCLEOTIDES:
        result[f"{prefix}_context_dinuc_{kmer}_frequency"] = _kmer_frequency(sequence, kmer)
    for kmer in TRINUCLEOTIDES:
        result[f"{prefix}_context_trinuc_{kmer}_frequency"] = _kmer_frequency(sequence, kmer)
    return result


def sequence_pair_features(guide_value: Any, target_value: Any) -> dict[str, float | int | None]:
    """Rebuild the 188 approved scalar sequence inputs from an aligned pair."""
    guide, target = normalize_sequence_pair(guide_value, target_value)
    target_ungapped = target.replace("-", "")
    guide_spacer = guide[4:]
    target_spacer_ungapped = target[4:].replace("-", "")
    features: dict[str, float | int | None] = {}
    features.update(_pair_features(guide, target))
    features.update(_base_features("guide", guide))
    features.update(_base_features("target_ungapped", target_ungapped))
    features.update(_base_features("guide_spacer", guide_spacer))
    features.update(_base_features("target_spacer_ungapped", target_spacer_ungapped))
    features.update(_context_features(guide, "guide"))
    features.update(_context_features(target_ungapped, "target"))
    return features


def build_feature_frame(pairs: pd.DataFrame, manifest: tuple[str, ...] | list[str] | None = None) -> pd.DataFrame:
    """Generate sequence features without relying on a package/workspace path."""
    if "crRNA_sequence" not in pairs:
        raise ValueError("Input must contain a crRNA_sequence column")
    target_column = "target_aligned_25" if "target_aligned_25" in pairs else "target_sequence"
    if target_column not in pairs:
        raise ValueError("Input must contain target_aligned_25 (or no-gap target_sequence)")
    rows = [
        sequence_pair_features(guide, target)
        for guide, target in pairs[["crRNA_sequence", target_column]].itertuples(index=False, name=None)
    ]
    frame = pd.DataFrame(rows, index=pairs.index)
    if manifest is not None:
        missing = sorted(set(manifest) - set(frame.columns))
        if missing:
            raise ValueError(f"Feature builder is missing manifest features: {missing}")
        frame = frame.loc[:, list(manifest)]
    return frame


def _pair_symbol(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper()
    return "".join(base if base in PAIR_ALPHABET else "N" for base in text)


def pair_onehot(frame: pd.DataFrame, length: int = 25) -> pd.DataFrame:
    """Position-specific guide/target identity encoding used by B/C and the mapping baseline."""
    if "crRNA_sequence" not in frame:
        raise ValueError("Input must contain crRNA_sequence for pair encoding")
    target_column = "target_aligned_25" if "target_aligned_25" in frame else "target_sequence"
    if target_column not in frame:
        raise ValueError("Input must contain target_aligned_25 (or target_sequence) for pair encoding")
    category_index = {category: index for index, category in enumerate(PAIR_CATEGORIES)}
    matrix = np.zeros((len(frame), length * len(PAIR_CATEGORIES)), dtype=np.uint8)
    for row, (guide_value, target_value) in enumerate(
        frame[["crRNA_sequence", target_column]].itertuples(index=False, name=None)
    ):
        guide = (_pair_symbol(guide_value) + "N" * length)[:length]
        target = (_pair_symbol(target_value) + "N" * length)[:length]
        for position, (left, right) in enumerate(zip(guide, target)):
            matrix[row, position * len(PAIR_CATEGORIES) + category_index[left + right]] = 1
    columns = [
        f"pair_identity_pos_{position + 1:02d}_{category}"
        for position in range(length)
        for category in PAIR_CATEGORIES
    ]
    return pd.DataFrame(matrix, columns=columns, index=frame.index)
