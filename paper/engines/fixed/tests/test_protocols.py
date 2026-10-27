from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from protocols import audit_protocol_assignments, load_inputs, make_encoding_fold_map  # noqa: E402


def _write_inputs(root: Path) -> dict[str, object]:
    ids = ["rec-a", "rec-b", "rec-c", "rec-d"]
    identity = {
        record_id: {
            "target_sequence": target,
            "crRNA_sequence": guide,
            "mapping_candidate_template_nos": str(index + 1),
        }
        for index, (record_id, target, guide) in enumerate(
            zip(ids, ["A" * 25, "C" * 25, "G" * 25, "T" * 25], ["A" * 25, "C" * 25, "G" * 25, "T" * 25])
        )
    }
    source_rows = [
        {"record_id": record_id, **identity[record_id], "baseline_split": "baseline_train", "label_normalized": i / 10}
        for i, record_id in enumerate(reversed(ids))
    ]
    source_rows.append(
        {
            "record_id": "historical-1",
            "target_sequence": "A" * 25,
            "crRNA_sequence": "A" * 25,
            "mapping_candidate_template_nos": "1",
            "baseline_split": "baseline_validation",
            "label_normalized": 0.99,
        }
    )
    source = pd.DataFrame(source_rows)
    assignments: list[dict[str, object]] = []
    for index, record_id in enumerate(ids):
        row: dict[str, object] = {"record_id": record_id, **identity[record_id]}
        for protocol in ("target", "guide", "dual_atomic", "atomic_template"):
            row[f"{protocol}_group"] = f"{protocol}-{index}"
            row[f"{protocol}_fold"] = index
        row["context_cv_fold"] = index
        assignments.append(row)
    # Deliberately shuffled, to verify that load_inputs uses record_id rather than row order.
    assignment_frame = pd.DataFrame(assignments).iloc[[2, 0, 3, 1]].reset_index(drop=True)
    pd.DataFrame(
        {"feature_name": ["aligned_difference_count"], "category": ["pair_alignment"]}
    ).to_csv(root / "features.csv", index=False)
    pd.DataFrame(
        {
            "template_no": [1],
            "template_group_id": ["template-group-1"],
            "template_sequence_clean": ["A" * 60],
        }
    ).to_csv(root / "templates.csv", index=False)
    source.to_csv(root / "data.csv", index=False)
    assignment_frame.to_csv(root / "assignments.csv", index=False)
    return {
        "source_paths": {
            "project_root": str(root),
            "data": "data.csv",
            "assignments": "assignments.csv",
            "sequence_manifest": "features.csv",
            "template_reference": "templates.csv",
        }
    }


class ProtocolInputTests(unittest.TestCase):
    def test_load_inputs_joins_by_record_id_and_preserves_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _write_inputs(root)
            dev, historical, assignments, manifest = load_inputs(config)
            self.assertEqual(dev["record_id"].tolist(), ["rec-d", "rec-c", "rec-b", "rec-a"])
            self.assertEqual(dev["target_group"].tolist(), ["target-3", "target-2", "target-1", "target-0"])
            self.assertEqual(len(historical), 1)
            self.assertEqual(len(assignments), 4)
            self.assertEqual(manifest.feature_names, ("aligned_difference_count",))
            self.assertEqual(manifest.template_records[0][0], "1")

    def test_encoding_fold_map_excludes_outer_query_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _write_inputs(Path(temporary))
            dev, _, assignments, _ = load_inputs(config)
            mapping = make_encoding_fold_map(assignments, "guide", outer_fold=1)
            self.assertNotIn("rec-b", mapping)
            self.assertEqual(set(mapping), {"rec-a", "rec-c", "rec-d"})
            report, audit_rows = audit_protocol_assignments(dev, assignments)
            self.assertEqual(report["n_dev_records"], 4)
            self.assertEqual(set(report["protocols"]), {"target", "guide", "dual_atomic", "atomic_template"})
            self.assertEqual(len(audit_rows), 16)
            self.assertTrue(audit_rows["group_overlap"].eq(0).all())
            self.assertTrue(audit_rows["record_id_overlap"].eq(0).all())

    def test_audit_rejects_group_crossing_protocol_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _write_inputs(Path(temporary))
            dev, _, assignments, _ = load_inputs(config)
            bad_assignments = assignments.copy()
            bad_assignments.loc[bad_assignments.record_id.eq("rec-a"), "guide_group"] = "guide-1"
            bad_dev = dev.drop(columns=["guide_group"]).merge(
                bad_assignments[["record_id", "guide_group"]], on="record_id", validate="one_to_one"
            )
            bad_dev.loc[bad_dev.record_id.eq("rec-a"), "guide_fold"] = 1
            with self.assertRaisesRegex(ValueError, "cross outer folds"):
                audit_protocol_assignments(bad_dev, bad_assignments)

    def test_assignment_id_set_must_exactly_match_development_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _write_inputs(root)
            assignments = pd.read_csv(root / "assignments.csv", dtype={"record_id": str})
            assignments = assignments.iloc[:-1]
            assignments.to_csv(root / "assignments.csv", index=False)
            with self.assertRaisesRegex(ValueError, "coverage does not exactly match"):
                load_inputs(config)


if __name__ == "__main__":
    unittest.main()
