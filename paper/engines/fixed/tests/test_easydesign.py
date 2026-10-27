"""Pure adapter checks; these tests never import TensorFlow or call fit."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import types
import unittest
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from easydesign import encode_sequences, prepare_partition
from easydesign.adapter import _resolve_recipe, _write_id_manifest, make_progress_callback


class EasyDesignEncodingTests(unittest.TestCase):
    def test_shape_gap_channel_and_fractional_iupac_encoding(self) -> None:
        context = list("A" * 45)
        context[1] = "-"
        context[2] = "N"
        guide = list("C" * 25)
        guide[0] = "R"
        frame = pd.DataFrame(
            {"target_context_45": ["".join(context)], "guide": ["".join(guide)]}
        )

        encoded = encode_sequences(frame)

        self.assertEqual(encoded.shape, (1, 45, 10))
        self.assertEqual(encoded.dtype, np.float32)
        np.testing.assert_array_equal(encoded[0, 0, :5], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(encoded[0, 1, :5], [0, 0, 0, 0, 1])
        np.testing.assert_allclose(encoded[0, 2, :5], [0.25, 0.25, 0.25, 0.25, 0])
        # Guide R=AG is placed in the second five-channel block at position 10.
        np.testing.assert_array_equal(encoded[0, 10, 5:10], [0.5, 0, 0.5, 0, 0])
        self.assertEqual(float(encoded[0, 9, 5:].sum()), 0.0)

    def test_rejects_wrong_context_length_and_unsupported_symbols(self) -> None:
        bad_length = pd.DataFrame({"target_context_45": ["A" * 44], "guide": ["C" * 25]})
        with self.assertRaisesRegex(ValueError, "length 45"):
            encode_sequences(bad_length)
        bad_symbol = pd.DataFrame(
            {"target_context_45": ["A" * 44 + "?"], "guide": ["C" * 25]}
        )
        with self.assertRaisesRegex(ValueError, "IUPAC symbol"):
            encode_sequences(bad_symbol)


class EasyDesignPartitionTests(unittest.TestCase):
    @staticmethod
    def _frame(ids: list[str], folds: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "record_id": ids,
                "target_fold": folds,
                "target_group": [f"g-{record_id}" for record_id in ids],
                "target_context_45": ["A" * 45] * len(ids),
                "guide": ["C" * 25] * len(ids),
            }
        )

    def test_next_fold_stop_is_a_subset_and_query_is_disjoint(self) -> None:
        fit = self._frame(["a", "b", "c", "d"], [1, 2, 3, 4])
        query = self._frame(["q", "r"], [0, 0])
        partition = prepare_partition(
            fit,
            ["a"],
            query,
            outer_fold=0,
            protocol_id="target",
        )
        self.assertEqual(partition.expected_stop_fold, 1)
        self.assertEqual(partition.stop_ids, ("a",))
        self.assertEqual(partition.inner_fit_ids, ("b", "c", "d"))
        self.assertEqual(partition.query_ids, ("q", "r"))
        self.assertTrue(partition.assignment_check.startswith("verified target_fold"))

    def test_query_label_is_rejected_instead_of_read(self) -> None:
        fit = self._frame(["a", "b"], [1, 2])
        query = self._frame(["q"], [0]).assign(y_true=[1000.0])
        with self.assertRaisesRegex(ValueError, "label-free"):
            prepare_partition(
                fit,
                ["a"],
                query,
                outer_fold=0,
                protocol_id="target",
            )

    def test_rejects_stopping_ids_outside_fit_and_fit_query_overlap(self) -> None:
        fit = self._frame(["a", "b"], [1, 2])
        query = self._frame(["q"], [0])
        with self.assertRaisesRegex(ValueError, "not contained"):
            prepare_partition(
                fit,
                ["not-fit"],
                query,
                outer_fold=0,
                protocol_id="target",
            )
        missing_id = fit.copy()
        missing_id.loc[1, "record_id"] = pd.NA
        with self.assertRaisesRegex(ValueError, "missing record_id"):
            prepare_partition(
                missing_id,
                ["a"],
                query,
                outer_fold=0,
                protocol_id="target",
            )
        overlap = self._frame(["b"], [0])
        with self.assertRaisesRegex(ValueError, "share record IDs"):
            prepare_partition(
                fit,
                ["a"],
                overlap,
                outer_fold=0,
                protocol_id="target",
            )

    def test_id_manifest_subsets_preserve_each_records_fold_and_group(self) -> None:
        fit = self._frame(["a", "b", "c", "d", "e"], [1, 2, 1, 3, 4])
        query = self._frame(["q"], [0])
        partition = prepare_partition(
            fit,
            ["a", "c"],
            query,
            outer_fold=0,
            protocol_id="target",
        )

        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_id_manifest(
                Path(directory),
                fit_frame=fit,
                query_frame=query,
                partition=partition,
            )
            with manifest.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        rows_by_key = {(row["record_id"], row["role"]): row for row in rows}
        expected = {
            "a": ("1", "g-a"),
            "b": ("2", "g-b"),
            "c": ("1", "g-c"),
            "d": ("3", "g-d"),
            "e": ("4", "g-e"),
            "q": ("0", "g-q"),
        }
        expected_roles = {
            "outer_fit": {"a", "b", "c", "d", "e"},
            "inner_fit": {"b", "d", "e"},
            "internal_stop": {"a", "c"},
            "outer_query": {"q"},
        }
        for role, record_ids in expected_roles.items():
            for record_id in record_ids:
                row = rows_by_key[(record_id, role)]
                self.assertEqual(row["assigned_fold"], expected[record_id][0])
                self.assertEqual(row["isolation_group"], expected[record_id][1])


class EasyDesignProgressTests(unittest.TestCase):
    def test_progress_callback_updates_atomically_and_prints_each_25_epochs(self) -> None:
        class FakeCallback:
            def __init__(self) -> None:
                pass

        fake_tf = types.SimpleNamespace(
            keras=types.SimpleNamespace(
                callbacks=types.SimpleNamespace(Callback=FakeCallback)
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            progress_path = Path(directory) / "progress.json"
            callback = make_progress_callback(
                fake_tf,
                progress_path,
                protocol_id="dual_atomic",
                outer_fold=2,
                stage="inner_stopping",
                max_epochs=450,
            )
            callback.on_train_begin()
            initial = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(initial["epoch"], 0)
            self.assertEqual(initial["status"], "running")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                for epoch in range(25):
                    callback.on_epoch_end(
                        epoch,
                        {"loss": np.float32(0.5), "val_loss": np.float32(0.4)},
                    )
            lines = [line for line in output.getvalue().splitlines() if line]
            self.assertEqual(len(lines), 1)
            self.assertIn('"stage": "inner_stopping"', lines[0])
            self.assertIn('"epoch": 25', lines[0])
            self.assertIn('"protocol": "dual_atomic"', lines[0])
            self.assertIn('"protocol_id": "dual_atomic"', lines[0])
            self.assertIn('"val_loss": 0.4000000059604645', lines[0])
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["fold"], 2)
            self.assertEqual(saved["loss"], float(np.float32(0.5)))
            self.assertEqual(saved["epoch"], 25)
            self.assertFalse(progress_path.with_name("progress.json.tmp").exists())

            callback.on_train_end()
            completed = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["epoch"], 25)


class EasyDesignRecipeTests(unittest.TestCase):
    def test_execution_config_resolves_frozen_model_and_recipe_hashes(self) -> None:
        data_root = os.environ.get('TRICASRANK_PAPER_DATA')
        if not data_root:
            self.skipTest('Set TRICASRANK_PAPER_DATA to the extracted paper assets for the real SavedModel check')
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from settings import settings
        recipe = _resolve_recipe(settings(data_root))
        self.assertEqual(
            recipe.model_pb_sha256,
            "f012985be29b90e86e45b351cf826de1f1f712709611ec0b3c81f07f24900309",
        )
        self.assertEqual(recipe.training["learning_rate"], 0.003684121184051037)
        self.assertEqual(recipe.training["batch_size"], 229)
        self.assertEqual(recipe.training["max_epochs"], 450)
        self.assertTrue(recipe.training_config_sha256)
        self.assertTrue(recipe.training_config_path.is_file())


if __name__ == "__main__":
    unittest.main()
