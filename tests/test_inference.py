from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cas12a_ml import Cas12aPredictor  # noqa: E402
from cas12a_ml.cli import run_self_test  # noqa: E402


class InferenceTest(unittest.TestCase):
    def test_examples_match_expected_predictions(self):
        pairs = pd.read_csv(ROOT / "data" / "examples" / "minimal_input.csv", dtype=str)
        expected = pd.read_csv(ROOT / "data" / "examples" / "minimal_expected_output.csv")
        actual = Cas12aPredictor(ROOT).predict(pairs)
        columns = [
            "cas12a_activity_score",
            "cas12a_prediction_xgboost",
            "cas12a_prediction_lightgbm",
            "cas12a_prediction_mlp",
            "cas12a_prediction_d",
            "cas12a_prediction_b",
            "cas12a_prediction_c",
            "cas12a_prediction_full_dbc",
            "prediction_xgboost_primary",
            "prediction_catboost_supporting",
            "prediction_equal_50_50_sensitivity",
            "prediction_oof_weighted_exploratory",
        ]
        for column in columns:
            self.assertTrue(
                np.allclose(actual[column], expected[column], atol=1e-6, rtol=0, equal_nan=True),
                column,
            )

    def test_bundled_self_test(self):
        run_self_test(ROOT)

    def test_legacy_route_is_explicit(self):
        pairs = pd.read_csv(ROOT / "data" / "examples" / "minimal_input.csv", dtype=str).head(1)
        actual = Cas12aPredictor(ROOT, primary_model="xgboost-legacy").predict(pairs)
        self.assertEqual(actual.loc[0, "cas12a_model_route"], "sequence_xgboost_v1_0_legacy")
        self.assertEqual(actual.loc[0, "cas12a_model_version"], "1.0.0")
        self.assertAlmostEqual(
            actual.loc[0, "cas12a_activity_score"],
            actual.loc[0, "prediction_xgboost_primary"],
        )

    def test_sequence_only_d_route_remains_available(self):
        pairs = pd.read_csv(ROOT / "data" / "examples" / "minimal_input.csv", dtype=str).head(1)
        actual = Cas12aPredictor(ROOT, primary_model="d").predict(pairs)
        self.assertEqual(actual.loc[0, "cas12a_model_route"], "sequence_d_v1_5")
        self.assertEqual(actual.loc[0, "cas12a_model_version"], "1.5.0")
        self.assertAlmostEqual(actual.loc[0, "cas12a_activity_score"], actual.loc[0, "cas12a_prediction_d"])

    def test_unmapped_row_uses_explicit_sequence_fallback(self):
        pairs = pd.DataFrame(
            {
                "record_id": ["unmapped"],
                "crRNA_sequence": ["A" * 25],
                "target_aligned_25": ["A" * 25],
            }
        )
        actual = Cas12aPredictor(ROOT).predict(pairs)
        self.assertEqual(actual.loc[0, "cas12a_model_route"], "sequence_d_fallback_v2")
        self.assertEqual(actual.loc[0, "cas12a_prediction_status"], "fallback_success")
        self.assertTrue(np.isnan(actual.loc[0, "cas12a_prediction_full_dbc"]))
        self.assertAlmostEqual(actual.loc[0, "cas12a_activity_score"], actual.loc[0, "cas12a_prediction_d"])
        self.assertIn("W_MAPPING_NOT_FOUND", actual.loc[0, "cas12a_warning_codes"])


if __name__ == "__main__":
    unittest.main()
