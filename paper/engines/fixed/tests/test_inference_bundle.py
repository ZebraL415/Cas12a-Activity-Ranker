from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import build_features, transform_query  # noqa: E402
from inference import (  # noqa: E402
    export_candidate_package,
    load_model_bundle,
    predict_candidate_bundle,
    predict_candidate_frame,
    save_model_bundle,
)
from models import resolved_recipe  # noqa: E402
from test_paper_features import _synthetic_frames  # noqa: E402


class ConstantPredictor:
    """A fit-free stand-in used to verify serialization and component reuse."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, matrix) -> np.ndarray:
        return np.full(len(matrix), self.value, dtype=float)


def test_model_bundle_round_trip_and_digest(tmp_path) -> None:
    components = {
        "D": {"xgb": ConstantPredictor(1), "lightgbm": ConstantPredictor(2), "mlp": ConstantPredictor(3)},
        "B": {str(seed): ConstantPredictor(1) for seed in (7, 21, 42, 84, 2024)},
        "C": ConstantPredictor(0),
        "SEQ_MAPPING_XGB": {"estimator": ConstantPredictor(0.5), "fit_target_mean": 10.0},
    }
    state = {"feature_columns": ["x1", "x2"], "lookup": {"g1": 1.2}}
    recipe = {"recipe_id": "unit-test-frozen-recipe"}
    manifest = save_model_bundle(
        components,
        tmp_path,
        feature_state=state,
        recipe=recipe,
        metadata={"no_fit_in_unit_test": True},
    )
    loaded = load_model_bundle(tmp_path)
    assert manifest["artifact_sha256"]
    assert set(loaded["components"]) == set(components)
    assert loaded["components"]["B"].keys() == components["B"].keys()
    assert loaded["feature_state"] == state
    assert loaded["resolved_recipe"] == recipe

    query = SimpleNamespace(
        d_query=np.ones((2, 2)),
        context_query=np.ones((2, 3)),
        seq_mapping_query=np.ones((2, 4)),
        guide_anchor_query=np.array([4.0, 5.0]),
        template_anchor_query=np.array([6.0, 7.0]),
        query_diagnostics=pd.DataFrame(
            {
                "record_id": ["mapped", "unmapped"],
                "mapping_found": [True, False],
                "guide_support_n": [3, 0],
                "template_support_n": [2, 0],
            }
        ),
    )
    prediction = predict_candidate_bundle(loaded, query)
    # D=.35*1 + .59*2 + .06*3 = 1.71; all frozen branches are reused.
    assert np.allclose(prediction["predictions"]["D"], [1.71, 1.71])
    assert np.allclose(prediction["predictions"]["B"], [5.0, 6.0])
    assert np.allclose(prediction["predictions"]["C"], [5.0, 6.0])
    assert np.allclose(prediction["predictions"]["SEQ_MAPPING_XGB"], [10.5, 10.5])
    assert np.allclose(prediction["predictions"]["TriCasRank"][1], 1.71)
    assert prediction["prediction_routes"]["TriCasRank"].tolist() == ["context", "sequence_fallback"]
    assert prediction["prediction_routes"]["SEQ_MAPPING_XGB"].tolist() == ["context", "context"]


def test_candidate_package_exports_minimal_fixture_and_runs_cli(tmp_path) -> None:
    fit, query, manifest, folds = _synthetic_frames()
    fixture = pd.DataFrame(
        [
            {
                "record_id": "mapped-query",
                "crRNA_sequence": query.iloc[0]["crRNA_sequence"],
                "target_aligned_25": query.iloc[0]["target_aligned_25"],
            },
            {
                "record_id": "unmapped-query",
                "crRNA_sequence": query.iloc[0]["crRNA_sequence"],
                "target_aligned_25": "A" * 25,
            },
        ],
        columns=["record_id", "crRNA_sequence", "target_aligned_25"],
    )
    feature_bundle = build_features(
        fit,
        fixture,
        sequence_manifest=manifest,
        protocol_id="guide",
        encoding_fold_by_record_id=folds,
    )
    components = {
        "D": {
            "xgb": ConstantPredictor(1),
            "lightgbm": ConstantPredictor(2),
            "mlp": ConstantPredictor(3),
        },
        "B": {str(seed): ConstantPredictor(0.25) for seed in (7, 21, 42, 84, 2024)},
        "C": ConstantPredictor(-0.5),
    }
    model_bundle = {"components": components, "feature_state": feature_bundle.state}
    reference = predict_candidate_frame(fixture, model_bundle, transform_query_fn=transform_query)
    assert reference["mapping_found"].tolist() == [True, False]
    assert reference.loc[1, "score"] == reference.loc[1, "D"]

    destination = tmp_path / "candidate"
    manifest_record = export_candidate_package(
        components,
        destination,
        feature_state=feature_bundle.state,
        recipe=resolved_recipe(),
        metadata={"run_id": "dummy-no-fit-test", "source_note": "/private/training/source.csv"},
        fixture_frame=fixture,
        reference_predictions=reference,
    )
    assert manifest_record["learner_count"] == 9
    assert manifest_record["standalone_mapping_baseline_included"] is False
    assert manifest_record["has_mapping_baseline"] is False
    assert set(pd.read_csv(destination / "fixture_input.csv").columns) == {
        "record_id",
        "crRNA_sequence",
        "target_aligned_25",
    }
    expected = pd.read_csv(destination / "expected_predictions.csv")
    assert expected["route"].tolist() == ["context", "sequence_fallback"]
    assert (destination / "requirements-core.txt").is_file()
    assert (destination / "README.md").is_file()
    for source_file in (destination / "src").rglob("*.py"):
        contents = source_file.read_text(encoding="utf-8")
        assert "/Users/" not in contents
        assert "/Downloads/" not in contents

    reversed_input = tmp_path / "reversed.csv"
    reversed_output = tmp_path / "reversed_predictions.csv"
    fixture.iloc[::-1].to_csv(reversed_input, index=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    subprocess.run(
        [
            sys.executable,
            str(destination / "predict_candidate.py"),
            "--input",
            str(reversed_input),
            "--output",
            str(reversed_output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    reversed_predictions = pd.read_csv(reversed_output, dtype={"record_id": str})
    assert reversed_predictions["record_id"].tolist() == ["unmapped-query", "mapped-query"]
    by_id = reference.set_index("record_id")
    assert np.allclose(
        reversed_predictions.set_index("record_id").loc[by_id.index, "score"].to_numpy(float),
        by_id["score"].to_numpy(float),
        rtol=0,
        atol=1e-6,
    )
