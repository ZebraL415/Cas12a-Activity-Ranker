from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "data" / "processed" / "v2_2"


class DataIntegrityTest(unittest.TestCase):
    def test_authoritative_sha_and_counts(self):
        path = PROFILE / "EasyDesign_2024_V2-2_core_context_feature_table.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, "39cda8368c216784507ac002df687b28a4f9cc6f81e2b0e84043e45eddb4c1c0")
        frame = pd.read_csv(path, usecols=["record_id", "baseline_split"])
        self.assertEqual(len(frame), 11992)
        self.assertTrue(frame["record_id"].is_unique)
        self.assertEqual((frame["baseline_split"] == "baseline_train").sum(), 8417)
        self.assertEqual((frame["baseline_split"] == "baseline_validation").sum(), 2217)


if __name__ == "__main__":
    unittest.main()
