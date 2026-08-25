from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cas12a_ml import Cas12aPredictor  # noqa: E402
from cas12a_ml.mapping import MAPPING_FIELDS, TemplateMapper  # noqa: E402


class V2MappingTest(unittest.TestCase):
    def test_mapper_reproduces_frozen_validation_fields(self):
        source = pd.read_csv(
            ROOT / "data" / "processed" / "v2_2" / "EasyDesign_2024_V2-2_core_context_feature_table.csv",
            low_memory=False,
        )
        sample = source.loc[source["baseline_split"].eq("baseline_validation")].iloc[::137]
        actual = TemplateMapper(
            ROOT / "models" / "mapping" / "table_s2_template_reference.csv"
        ).map_frame(sample)
        for column in MAPPING_FIELDS:
            self.assertEqual(
                actual[column].fillna("").astype(str).tolist(),
                sample[column].fillna("").astype(str).tolist(),
                column,
            )

    def test_full_score_uses_locked_weights(self):
        pairs = pd.read_csv(ROOT / "data" / "examples" / "minimal_input.csv", dtype=str).head(4)
        actual = Cas12aPredictor(ROOT).predict(pairs)
        expected = (
            0.20 * actual["cas12a_prediction_d"]
            + 0.47 * actual["cas12a_prediction_b"]
            + 0.33 * actual["cas12a_prediction_c"]
        )
        self.assertTrue(np.allclose(actual["cas12a_activity_score"], expected, atol=1e-10, rtol=0))
        self.assertIn("W_MAPPING_MULTIPLE_CANDIDATES", actual.loc[0, "cas12a_warning_codes"])

    def test_mixed_routes_withhold_global_rank_unless_requested(self):
        pairs = pd.read_csv(ROOT / "data" / "examples" / "minimal_input.csv", dtype=str)
        default = Cas12aPredictor(ROOT).predict(pairs)
        opted_in = Cas12aPredictor(ROOT, allow_mixed_ranking=True).predict(pairs)
        self.assertTrue(default["cas12a_activity_rank"].isna().all())
        self.assertTrue(default["cas12a_rank_within_route"].notna().all())
        self.assertTrue(opted_in["cas12a_activity_rank"].notna().all())

    def test_strict_fallback_policy_stops_on_unmapped_target(self):
        pairs = pd.DataFrame(
            {
                "record_id": ["unmapped"],
                "crRNA_sequence": ["A" * 25],
                "target_aligned_25": ["A" * 25],
            }
        )
        with self.assertRaisesRegex(ValueError, "Automatic mapping failed"):
            Cas12aPredictor(ROOT, fallback_policy="error").predict(pairs)

    def test_u_is_normalized_to_t_in_every_v2_module(self):
        pairs = pd.read_csv(ROOT / "data" / "examples" / "minimal_input.csv", dtype=str).head(1)
        with_u = pairs.copy()
        with_u["crRNA_sequence"] = with_u["crRNA_sequence"].str.replace("T", "U")
        with_u["target_aligned_25"] = with_u["target_aligned_25"].str.replace("T", "U")
        expected = Cas12aPredictor(ROOT).predict(pairs)
        actual = Cas12aPredictor(ROOT).predict(with_u)
        for column in (
            "cas12a_prediction_d",
            "cas12a_prediction_b",
            "cas12a_prediction_c",
            "cas12a_activity_score",
        ):
            self.assertAlmostEqual(actual.loc[0, column], expected.loc[0, column], places=10)

    def test_history_references_match_training_only_contract(self):
        metadata = json.loads(
            (ROOT / "models" / "mapping" / "reference_metadata.json").read_text(encoding="utf-8")
        )
        guide = pd.read_csv(ROOT / "models" / "mapping" / "guide_history_reference.csv")
        all_guide = guide.loc[guide["reference_type"].eq("all")]
        self.assertEqual(int(all_guide["count"].sum()), 8417)
        self.assertEqual(metadata["train_rows"], 8417)
        self.assertEqual(metadata["unique_guides"], 1341)
        self.assertIn("No baseline_validation labels", metadata["leakage_control"])


if __name__ == "__main__":
    unittest.main()
