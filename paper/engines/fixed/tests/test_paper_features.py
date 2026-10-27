from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from features import SequenceManifest, build_features, sequence_pair_features, transform_query  # noqa: E402


BASE_GUIDE = "ACGTACGTACGTACGTACGTACGTA"


def _mutate(sequence: str, position: int, base: str) -> str:
    return sequence[:position] + base + sequence[position + 1 :]


def _synthetic_frames(n: int = 18) -> tuple[pd.DataFrame, pd.DataFrame, SequenceManifest, dict[str, int]]:
    rows: list[dict[str, object]] = []
    for index in range(n):
        guide = BASE_GUIDE if index % 2 == 0 else _mutate(BASE_GUIDE, 0, "T")
        target = guide
        for offset in range(index % 4):
            position = (5 + index + offset) % 25
            target = _mutate(target, position, "ACGT"[("ACGT".index(target[position]) + 1) % 4])
        scalar = sequence_pair_features(guide, target)
        rows.append(
            {
                "record_id": f"fit-{index:02d}",
                "crRNA_sequence": guide,
                "target_aligned_25": target,
                "label_normalized": float(index) / 10.0,
                **scalar,
                "mapping_status": "unique_exact_window" if index % 2 == 0 else "ambiguous_template_single_group",
                "mapping_confidence": "high" if index % 2 == 0 else "medium",
                "mapping_hit_count": 1 + index % 3,
                "mapping_template_count": 1 + index % 2,
                "mapping_group_count": 1,
                "mapping_orientation_count": 1,
                "mapping_candidate_template_nos": str(index % 3 + 1),
                "mapping_candidate_group_ids": f"group-{index % 2}",
                "mapping_candidate_orientations": "forward",
                "mapping_match_modes": "exact_acgt",
            }
        )
    fit = pd.DataFrame(rows)
    fit["guide_fold"] = [index % 3 for index in range(n)]
    query = pd.DataFrame(
        [
            {"record_id": "query-1", "crRNA_sequence": BASE_GUIDE, "target_aligned_25": BASE_GUIDE},
            {
                "record_id": "query-2",
                "crRNA_sequence": _mutate(BASE_GUIDE, 0, "T"),
                "target_aligned_25": _mutate(BASE_GUIDE, 10, "A"),
            },
        ]
    )
    query_targets = query["target_aligned_25"].tolist()
    template_records = tuple(
        (str(index + 1), f"group-{index + 1}", "G" * 30 + target + "C" * 30)
        for index, target in enumerate(query_targets)
    )
    manifest = SequenceManifest(tuple(sequence_pair_features(BASE_GUIDE, BASE_GUIDE).keys()), (), template_records)
    folds = {f"fit-{index:02d}": index % 3 for index in range(n)}
    return fit, query, manifest, folds


class FeatureBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fit, self.query, self.manifest, self.folds = _synthetic_frames()
        self.bundle = build_features(
            self.fit,
            self.query,
            sequence_manifest=self.manifest,
            protocol_id="guide",
            encoding_fold_by_record_id=self.folds,
        )

    def test_matrix_contract_and_saved_query_transform(self) -> None:
        bundle = self.bundle
        self.assertTrue(bundle.context_fit.columns.equals(bundle.context_query.columns))
        self.assertTrue(bundle.seq_mapping_fit.columns.equals(bundle.seq_mapping_query.columns))
        self.assertFalse(any(name.startswith("te_guide_") or name.startswith("mapping_te_") for name in bundle.seq_mapping_fit))
        self.assertTrue(any(name.startswith("te_guide_") for name in bundle.context_fit))
        self.assertTrue(any(name.startswith("mapping_te_") for name in bundle.context_fit))
        transformed = transform_query(self.query, bundle.state)
        pd.testing.assert_frame_equal(bundle.d_query, transformed["d_query"])
        pd.testing.assert_frame_equal(bundle.context_query, transformed["context_query"])
        pd.testing.assert_frame_equal(bundle.seq_mapping_query, transformed["seq_mapping_query"])
        np.testing.assert_allclose(bundle.guide_anchor_query, transformed["guide_anchor_query"])
        np.testing.assert_allclose(bundle.template_anchor_query, transformed["template_anchor_query"])
        self.assertEqual(bundle.query_diagnostics["mapping_found"].tolist(), [True, True])
        self.assertTrue(bundle.query_diagnostics["guide_support_n"].gt(0).all())
        self.assertTrue(bundle.query_diagnostics["template_support_n"].gt(0).all())

    def test_guide_loo_excludes_own_label_and_mapping_crossfit_excludes_fold(self) -> None:
        changed = self.fit.copy()
        changed.loc[0, "label_normalized"] += 1000.0
        changed_bundle = build_features(
            changed,
            self.query,
            sequence_manifest=self.manifest,
            protocol_id="guide",
            encoding_fold_by_record_id=self.folds,
        )
        self.assertAlmostEqual(
            self.bundle.context_fit.iloc[0]["te_guide_all"],
            changed_bundle.context_fit.iloc[0]["te_guide_all"],
        )

        changed_fold = self.fit.copy()
        fold_zero_ids = [record_id for record_id, fold in self.folds.items() if fold == 0]
        changed_fold.loc[changed_fold.record_id.isin(fold_zero_ids), "label_normalized"] += 500.0
        changed_fold_bundle = build_features(
            changed_fold,
            self.query,
            sequence_manifest=self.manifest,
            protocol_id="guide",
            encoding_fold_by_record_id=self.folds,
        )
        rows = self.fit.record_id.isin(fold_zero_ids).to_numpy()
        column = "mapping_te_mapping_candidate_template_nos"
        np.testing.assert_allclose(
            self.bundle.context_fit.loc[rows, column].to_numpy(),
            changed_fold_bundle.context_fit.loc[rows, column].to_numpy(),
        )

    def test_encoding_fold_map_must_match_current_protocol(self) -> None:
        wrong_folds = {record_id: (fold + 1) % 3 for record_id, fold in self.folds.items()}
        with self.assertRaisesRegex(ValueError, "does not match current protocol column guide_fold"):
            build_features(
                self.fit,
                self.query,
                sequence_manifest=self.manifest,
                protocol_id="guide",
                encoding_fold_by_record_id=wrong_folds,
            )

    def test_unseen_mapping_categories_are_zero_onehot_not_mapping_miss(self) -> None:
        query = self.query.iloc[[0]].copy()
        query["mapping_status"] = "new-status"
        query["mapping_confidence"] = "new-confidence"
        query["mapping_hit_count"] = 1
        query["mapping_template_count"] = 1
        query["mapping_group_count"] = 1
        query["mapping_orientation_count"] = 1
        query["mapping_candidate_template_nos"] = "999999"
        query["mapping_candidate_group_ids"] = "new-group"
        query["mapping_candidate_orientations"] = "new-orientation"
        query["mapping_match_modes"] = "new-match-mode"
        bundle = build_features(
            self.fit,
            query,
            sequence_manifest=self.manifest,
            protocol_id="target",
            encoding_fold_by_record_id=self.folds,
        )
        mapping_onehot_columns = [name for name in bundle.seq_mapping_query if name.startswith("mapping_cat_")]
        self.assertEqual(float(bundle.seq_mapping_query[mapping_onehot_columns].to_numpy().sum()), 0.0)
        self.assertTrue(bool(bundle.query_diagnostics.iloc[0]["mapping_found"]))
        self.assertEqual(int(bundle.query_diagnostics.iloc[0]["template_support_n"]), 0)

    def test_query_labels_are_rejected_and_state_has_no_source_path(self) -> None:
        query_with_label = self.query.assign(label_normalized=123.0)
        with self.assertRaisesRegex(ValueError, "must not contain labels"):
            transform_query(query_with_label, self.bundle.state)
        query_with_raw_label = self.query.assign(label_raw_value=123.0)
        with self.assertRaisesRegex(ValueError, "must not contain labels"):
            transform_query(query_with_raw_label, self.bundle.state)
        serialized = pickle.dumps(self.bundle.state)
        self.assertNotIn(b"ranker_v1_5_worktree", serialized)
        self.assertNotIn(b"/Users/", serialized)

    def test_candidate_sequence_pair_reconstructs_features_and_mapping(self) -> None:
        candidate = pd.DataFrame(
            [{"record_id": "fresh-candidate", "crRNA_sequence": BASE_GUIDE, "target_aligned_25": BASE_GUIDE}]
        )
        transformed = transform_query(candidate, self.bundle.state)
        self.assertGreater(transformed["d_query"].shape[1], 0)
        self.assertEqual(transformed["query_sequence_source"], "generated_from_25nt_pair")
        self.assertTrue(bool(transformed["query_diagnostics"].iloc[0]["mapping_found"]))

    def test_query_row_reordering_preserves_id_keyed_features(self) -> None:
        original = transform_query(self.query, self.bundle.state)
        shuffled_query = self.query.iloc[::-1].copy()
        shuffled = transform_query(shuffled_query, self.bundle.state)

        def keyed(frame: pd.DataFrame, result: dict[str, object]) -> pd.DataFrame:
            keyed_frame = frame.copy()
            keyed_frame.index = result["query_diagnostics"]["record_id"].astype(str).to_numpy()
            return keyed_frame.sort_index()

        for field in ("d_query", "context_query", "seq_mapping_query"):
            pd.testing.assert_frame_equal(keyed(original[field], original), keyed(shuffled[field], shuffled))
        for field in ("guide_anchor_query", "template_anchor_query"):
            original_by_id = dict(zip(original["query_diagnostics"].record_id, original[field]))
            shuffled_by_id = dict(zip(shuffled["query_diagnostics"].record_id, shuffled[field]))
            self.assertEqual(original_by_id, shuffled_by_id)


if __name__ == "__main__":
    unittest.main()
