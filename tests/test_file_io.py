from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cas12a_ml import InputValidationError, predict_file, read_table  # noqa: E402


class FileInterfaceTest(unittest.TestCase):
    def test_csv_and_xlsx_preserve_rows_columns_and_order(self):
        source = pd.read_csv(ROOT / "data" / "examples" / "minimal_input.csv", dtype=str)
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            for suffix in (".csv", ".tsv", ".xlsx"):
                input_path = temporary / f"input{suffix}"
                output_path = temporary / f"output{suffix}"
                if suffix == ".csv":
                    source.to_csv(input_path, index=False)
                elif suffix == ".tsv":
                    source.to_csv(input_path, sep="\t", index=False)
                else:
                    source.to_excel(input_path, index=False)
                predict_file(input_path, output_path, repository_root=ROOT)
                output = read_table(output_path)
                self.assertEqual(len(output), len(source))
                self.assertEqual(output["record_id"].tolist(), source["record_id"].tolist())
                self.assertEqual(list(output.columns[: len(source.columns)]), list(source.columns))
                self.assertIn("cas12a_activity_score", output.columns)

    def test_invalid_input_writes_error_report(self):
        source = pd.DataFrame(
            {
                "record_id": ["duplicate", "duplicate"],
                "crRNA_sequence": ["A" * 25, "A" * 25],
                "target_aligned_25": ["A" * 24, "A" * 25],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "bad.csv"
            source.to_csv(input_path, index=False)
            with self.assertRaises(InputValidationError) as context:
                predict_file(input_path, repository_root=ROOT)
            errors = pd.read_csv(context.exception.error_path)
            self.assertIn("record_id must be unique", set(errors["error"]))
            self.assertTrue(errors["error"].str.contains("25 aligned positions").any())

    def test_keep_mode_handles_a_file_with_no_valid_rows(self):
        source = pd.DataFrame(
            {
                "record_id": ["bad_01", "bad_02"],
                "crRNA_sequence": ["A" * 24, "N" * 25],
                "target_aligned_25": ["A" * 25, "A" * 25],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "all_bad.csv"
            output_path = Path(temporary) / "kept.csv"
            source.to_csv(input_path, index=False)
            predict_file(
                input_path,
                output_path,
                repository_root=ROOT,
                on_invalid="keep",
            )
            output = pd.read_csv(output_path)
            self.assertEqual(output["record_id"].tolist(), source["record_id"].tolist())
            self.assertTrue(output["cas12a_activity_score"].isna().all())
            self.assertEqual(set(output["cas12a_prediction_status"]), {"invalid_input"})
            self.assertTrue(output["cas12a_warning_codes"].notna().all())

    def test_user_supplied_mapping_columns_are_rejected(self):
        source = pd.DataFrame(
            {
                "record_id": ["manual_mapping"],
                "crRNA_sequence": ["A" * 25],
                "target_aligned_25": ["A" * 25],
                "mapping_status": ["unique_exact_window"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "manual.csv"
            source.to_csv(input_path, index=False)
            with self.assertRaises(InputValidationError) as context:
                predict_file(input_path, repository_root=ROOT)
            errors = pd.read_csv(context.exception.error_path)
            self.assertTrue(errors["error"].str.contains("calculates mapping automatically").any())


if __name__ == "__main__":
    unittest.main()
