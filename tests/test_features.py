from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cas12a_ml.features import build_feature_frame  # noqa: E402


class FeatureReconstructionTest(unittest.TestCase):
    def test_manifest_features_match_source_rows(self):
        profile = ROOT / "data" / "processed" / "v2_2"
        table = pd.read_csv(profile / "EasyDesign_2024_V2-2_core_context_feature_table.csv", low_memory=False)
        manifest = pd.read_csv(profile / "feature_manifest.csv")["feature_name"].tolist()
        sample = pd.concat([
            table.iloc[[0, 10, 100, 1000, 5000, 9000]],
            table.loc[table["target_aligned_25"].str.contains("-", regex=False)].iloc[:6],
        ])
        actual = build_feature_frame(sample, manifest).to_numpy(float)
        expected = sample[manifest].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.assertTrue(np.isclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True).all())

    def test_unaligned_gap_input_is_rejected(self):
        pairs = pd.DataFrame({"crRNA_sequence": ["A" * 25], "target_aligned_25": ["A" * 24]})
        with self.assertRaises(ValueError):
            build_feature_frame(pairs)


if __name__ == "__main__":
    unittest.main()
