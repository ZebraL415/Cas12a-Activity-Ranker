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
            "prediction_xgboost_primary",
            "prediction_catboost_supporting",
            "prediction_equal_50_50_sensitivity",
            "prediction_oof_weighted_exploratory",
        ]
        for column in columns:
            self.assertTrue(np.allclose(actual[column], expected[column], atol=1e-6, rtol=0), column)

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


if __name__ == "__main__":
    unittest.main()
