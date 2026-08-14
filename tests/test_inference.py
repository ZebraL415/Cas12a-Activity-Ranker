from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cas12a_ml import Cas12aPredictor  # noqa: E402


class InferenceTest(unittest.TestCase):
    def test_examples_match_expected_predictions(self):
        pairs = pd.read_csv(ROOT / "data" / "examples" / "external_sequence_pairs.csv", dtype=str)
        expected = pd.read_csv(ROOT / "data" / "examples" / "expected_predictions.csv")
        actual = Cas12aPredictor(ROOT).predict(pairs)
        columns = [
            "prediction_xgboost_primary",
            "prediction_catboost_supporting",
            "prediction_equal_50_50_sensitivity",
            "prediction_oof_weighted_exploratory",
        ]
        for column in columns:
            self.assertTrue(np.allclose(actual[column], expected[column], atol=1e-6, rtol=0), column)


if __name__ == "__main__":
    unittest.main()
