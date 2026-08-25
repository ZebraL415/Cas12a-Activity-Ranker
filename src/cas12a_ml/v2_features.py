"""Build the frozen 1,191-feature matrix used by mapping-aware modules B and C."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_feature_frame


MAPPING_CATEGORICAL = (
    "mapping_candidate_group_ids",
    "mapping_status",
    "mapping_confidence",
    "mapping_candidate_template_nos",
    "mapping_match_modes",
    "mapping_candidate_orientations",
)
MAPPING_NUMERIC = (
    "mapping_hit_count",
    "mapping_template_count",
    "mapping_group_count",
    "mapping_orientation_count",
)


def normalized_key(value: Any) -> str:
    if pd.isna(value) or str(value) == "":
        return "missing"
    return str(value)


def normalized_sequence(value: Any) -> str:
    return str(value).strip().upper().replace("U", "T")


class V2FeatureBuilder:
    """Recreate validation/deployment features from frozen training-only references."""

    def __init__(self, repository_root: str | Path):
        root = Path(repository_root)
        manifest = pd.read_csv(root / "models" / "mapping" / "feature_manifest.csv")
        self.manifest = manifest["feature_name"].astype(str).tolist()
        metadata = json.loads(
            (root / "models" / "mapping" / "reference_metadata.json").read_text(encoding="utf-8")
        )
        self.global_mean = float(metadata["global_train_label_mean"])
        self.guide_smoothing = float(metadata["guide_smoothing"])
        self.mapping_smoothing = float(metadata["mapping_smoothing"])
        self.numeric_features = manifest.loc[
            manifest["feature_block"].eq("frozen_manifest_numeric"), "feature_name"
        ].astype(str).tolist()
        medians = pd.read_csv(root / "models" / "training_medians.csv")
        self.numeric_medians = medians.set_index("feature_name")["median"].reindex(self.numeric_features)

        guide = pd.read_csv(
            root / "models" / "mapping" / "guide_history_reference.csv",
            dtype={"crRNA_sequence": str, "reference_type": str, "category": str},
            keep_default_na=False,
        )
        self.guide_stats = {
            (row.reference_type, row.crRNA_sequence, row.category): (float(row.sum), float(row.count))
            for row in guide.itertuples(index=False)
        }
        mapping = pd.read_csv(
            root / "models" / "mapping" / "mapping_history_reference.csv",
            dtype={"field": str, "key": str},
            keep_default_na=False,
        )
        self.mapping_stats = {
            (row.field, row.key): (float(row.sum), float(row.count))
            for row in mapping.itertuples(index=False)
        }

    def _guide_features(self, pairs: pd.DataFrame, numeric: pd.DataFrame) -> pd.DataFrame:
        output: list[dict[str, float]] = []
        for position, (_, row) in enumerate(pairs.iterrows()):
            guide = normalized_sequence(row["crRNA_sequence"])
            diff = float(numeric.iloc[position]["aligned_difference_count"])
            gap = float(numeric.iloc[position]["gap_count_target"]) + float(
                numeric.iloc[position].get("gap_count_guide", 0)
            )
            diffcat = "exact" if diff == 0 else "one" if diff == 1 else "two_plus"
            gapcat = "gap" if gap > 0 else "no_gap"
            total_sum, total_count = self.guide_stats.get(("all", guide, ""), (0.0, 0.0))
            guide_all = (total_sum + self.guide_smoothing * self.global_mean) / (
                total_count + self.guide_smoothing
            )
            values: dict[str, float] = {
                "te_guide_all": guide_all,
                "te_guide_all_log_count": float(np.log1p(total_count)),
                "te_guide_seen": float(total_count > 0),
            }
            for name, category in (("exact", ""), ("nonexact", "")):
                value_sum, count = self.guide_stats.get((name, guide, category), (0.0, 0.0))
                values[f"te_guide_{name}_ref"] = (
                    value_sum + self.guide_smoothing * guide_all
                ) / (count + self.guide_smoothing)
                values[f"te_guide_{name}_log_count"] = float(np.log1p(count))
            for name, category in (("diffcat", diffcat), ("gapcat", gapcat)):
                value_sum, count = self.guide_stats.get((name, guide, category), (0.0, 0.0))
                values[f"te_guide_{name}"] = (
                    value_sum + self.guide_smoothing * guide_all
                ) / (count + self.guide_smoothing)
                values[f"te_guide_{name}_log_count"] = float(np.log1p(count))
            output.append(values)
        return pd.DataFrame(output, index=pairs.index)

    def build(self, pairs: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        numeric = build_feature_frame(pairs, manifest=self.numeric_features)
        numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(self.numeric_medians).fillna(0)
        guide = self._guide_features(pairs, numeric)
        matrix = pd.DataFrame(0.0, index=pairs.index, columns=self.manifest)
        matrix.loc[:, self.numeric_features] = numeric.to_numpy()
        matrix.loc[:, guide.columns] = guide.to_numpy()

        pair_columns = {name for name in self.manifest if name.startswith("pair_identity_")}
        for row_number, (index, row) in enumerate(pairs.iterrows()):
            guide_sequence = normalized_sequence(row["crRNA_sequence"])
            target_sequence = normalized_sequence(row["target_aligned_25"])
            for position, (left, right) in enumerate(zip(guide_sequence, target_sequence), start=1):
                column = f"pair_identity_pos_{position:02d}_{left}{right}"
                if column in pair_columns:
                    matrix.at[index, column] = 1.0

        for column in MAPPING_NUMERIC:
            matrix.loc[:, column] = pd.to_numeric(mapping[column], errors="coerce").fillna(0).to_numpy()
        for field in MAPPING_CATEGORICAL:
            values: list[float] = []
            counts: list[float] = []
            for value in mapping[field]:
                key = normalized_key(value)
                value_sum, count = self.mapping_stats.get((field, key), (0.0, 0.0))
                values.append(
                    (value_sum + self.mapping_smoothing * self.global_mean)
                    / (count + self.mapping_smoothing)
                )
                counts.append(float(np.log1p(count)))
            matrix.loc[:, f"mapping_te_{field}"] = values
            matrix.loc[:, f"mapping_te_{field}_log_count"] = counts
            for index, value in mapping[field].items():
                column = f"mapping_cat_{field}_{normalized_key(value)}"
                if column in matrix.columns:
                    matrix.at[index, column] = 1.0

        guide_seen = guide["te_guide_seen"].astype(bool)
        template_seen = mapping["mapping_candidate_template_nos"].map(
            lambda value: self.mapping_stats.get(
                ("mapping_candidate_template_nos", normalized_key(value)), (0.0, 0.0)
            )[1]
            > 0
        )
        diagnostics = pd.DataFrame(
            {
                "guide_seen": guide_seen,
                "guide_reference_count": np.expm1(guide["te_guide_all_log_count"]),
                "template_key_seen": template_seen,
                "template_reference_count": mapping["mapping_candidate_template_nos"].map(
                    lambda value: self.mapping_stats.get(
                        ("mapping_candidate_template_nos", normalized_key(value)), (0.0, 0.0)
                    )[1]
                ),
                "guide_anchor": guide["te_guide_all"],
                "template_anchor": matrix["mapping_te_mapping_candidate_template_nos"],
            },
            index=pairs.index,
        )
        return matrix.astype(float), diagnostics
