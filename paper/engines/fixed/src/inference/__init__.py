"""Save, load, and apply already-fitted TriCasRank components.

Candidate inference is intentionally separate from training.  The feature
transformer is supplied by the candidate package or injected by a caller;
this module never fits or updates a model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd

from models import predict_components


ARTIFACT_FILENAME = "candidate_model.joblib"
MANIFEST_FILENAME = "candidate_manifest.json"
FORBIDDEN_QUERY_LABEL_COLUMNS = {
    "label_normalized",
    "y_true",
    "label_raw_value",
    "label_raw_name",
    "label_scale_group",
    "label_is_primary_baseline",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_bundle(
    components: Mapping[str, Any],
    destination: str | Path,
    *,
    feature_state: Any,
    recipe: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist fitted estimators and the train-derived transform state.

    Existing artifact files are never overwritten.  The containing directory
    may already exist so the coordinator can place package documentation next
    to the model.
    """
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / ARTIFACT_FILENAME
    manifest_path = folder / MANIFEST_FILENAME
    if artifact.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite candidate files in {folder}")
    payload = {
        "format_version": 1,
        "components": dict(components),
        "feature_state": feature_state,
        "resolved_recipe": dict(recipe),
        "metadata": dict(metadata or {}),
    }
    temporary = folder / f".{ARTIFACT_FILENAME}.tmp"
    try:
        joblib.dump(payload, temporary, compress=3)
        temporary.replace(artifact)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "format_version": 1,
        "artifact": ARTIFACT_FILENAME,
        "artifact_sha256": _sha256(artifact),
        "required_components": ["D", "B", "C"],
        "has_mapping_baseline": "SEQ_MAPPING_XGB" in components,
        "has_feature_state": feature_state is not None,
        "recipe_id": recipe.get("recipe_id"),
        "metadata": dict(metadata or {}),
    }
    manifest_tmp = folder / f".{MANIFEST_FILENAME}.tmp"
    try:
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_tmp.replace(manifest_path)
    finally:
        if manifest_tmp.exists():
            manifest_tmp.unlink()
    return manifest


def load_model_bundle(source: str | Path) -> dict[str, Any]:
    """Load a saved candidate after verifying its on-disk digest."""
    folder = Path(source)
    if folder.is_file():
        artifact = folder
        manifest_path = folder.with_name(MANIFEST_FILENAME)
    else:
        artifact = folder / ARTIFACT_FILENAME
        manifest_path = folder / MANIFEST_FILENAME
    if not artifact.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"candidate artifact and manifest are required under {folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_digest = _sha256(artifact)
    if actual_digest != manifest.get("artifact_sha256"):
        raise ValueError("candidate artifact SHA-256 does not match its manifest")
    payload = joblib.load(artifact)
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported candidate format version: {payload.get('format_version')}")
    components = payload.get("components", {})
    missing = {"D", "B", "C"} - set(components)
    if missing:
        raise ValueError(f"candidate is missing fitted components: {sorted(missing)}")
    if payload.get("feature_state") is None:
        raise ValueError("candidate does not include train-derived feature state")
    return payload


def predict_candidate_bundle(model_bundle: Mapping[str, Any], query_features: Any) -> dict[str, Any]:
    """Predict from transformed query features and a loaded model package."""
    if "components" not in model_bundle:
        raise ValueError("model_bundle is missing fitted components")
    return predict_components(model_bundle["components"], query_features)


def predict_candidate_frame(
    query_frame: pd.DataFrame,
    model_bundle: Mapping[str, Any],
    *,
    transform_query_fn: Callable[[pd.DataFrame, Any], Any] | None = None,
) -> pd.DataFrame:
    """Transform raw rows with saved fit state and return schema-ready scores.

    A feature package may inject its ``transform_query`` function.  When it is
    omitted, the local execution package's ``features.transform_query`` is
    imported lazily.  Query labels are neither required nor passed downstream.
    """
    if "record_id" not in query_frame.columns:
        raise ValueError("candidate query input requires record_id")
    leaked_labels = FORBIDDEN_QUERY_LABEL_COLUMNS.intersection(query_frame.columns)
    if leaked_labels:
        raise ValueError(f"candidate query input must not contain label columns: {sorted(leaked_labels)}")
    if query_frame["record_id"].isna().any() or query_frame["record_id"].astype(str).duplicated().any():
        raise ValueError("candidate query record_id values must be nonmissing and unique")
    if transform_query_fn is None:
        try:
            from features import transform_query as transform_query_fn  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "candidate feature package is unavailable; pass its transform_query function"
            ) from exc
    state = model_bundle.get("feature_state")
    if state is None:
        raise ValueError("model_bundle does not contain saved feature state")
    query_features = transform_query_fn(query_frame.copy(), state)
    prediction = predict_candidate_bundle(model_bundle, query_features)
    values = prediction["predictions"]
    routes = prediction["prediction_routes"]
    if len(values["TriCasRank"]) != len(query_frame):
        raise ValueError("candidate predictions do not align with input rows")
    result = query_frame.copy()
    result["score"] = values["TriCasRank"]
    result["D"] = values["D"]
    result["B"] = values["B"]
    result["C"] = values["C"]
    result["route"] = routes["TriCasRank"]
    diagnostics = prediction["diagnostics"]
    for name in ("mapping_found", "guide_support_n", "template_support_n"):
        if hasattr(diagnostics, "columns") and name in diagnostics.columns:
            result[name] = diagnostics[name].to_numpy()
        elif isinstance(diagnostics, Mapping) and name in diagnostics:
            result[name] = np.asarray(diagnostics[name])
    return result


def _candidate_components(components: Mapping[str, Any]) -> None:
    if "SEQ_MAPPING_XGB" in components:
        raise ValueError("candidate export must not include the strong mapping baseline")
    if set(components) != {"D", "B", "C"}:
        raise ValueError("candidate export requires exactly the fitted D, B, and C components")
    if set(components["D"]) != {"xgb", "lightgbm", "mlp"}:
        raise ValueError("candidate D must contain exactly the three frozen learners")
    if set(map(str, components["B"])) != {"7", "21", "42", "84", "2024"}:
        raise ValueError("candidate B must contain the five frozen seed learners")
    if components["C"] is None:
        raise ValueError("candidate C learner is missing")


def _fixture_ids(frame: pd.DataFrame) -> list[str]:
    required = ["record_id", "crRNA_sequence", "target_aligned_25"]
    if list(frame.columns) != required:
        raise ValueError(f"fixture_frame columns must be exactly {required} in that order")
    if frame.empty:
        raise ValueError("fixture_frame must contain at least one row")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).duplicated().any():
        raise ValueError("fixture_frame record_id values must be nonmissing and unique")
    if frame["crRNA_sequence"].isna().any() or frame["target_aligned_25"].isna().any():
        raise ValueError("fixture sequences must be nonmissing")
    return frame["record_id"].astype(str).tolist()


def _aligned_expected_predictions(
    fixture_ids: list[str], reference_predictions: pd.DataFrame
) -> pd.DataFrame:
    if not isinstance(reference_predictions, pd.DataFrame):
        raise TypeError("reference_predictions must be the candidate prediction DataFrame")
    columns = [
        "record_id",
        "score",
        "D",
        "B",
        "C",
        "route",
        "mapping_found",
        "guide_support_n",
        "template_support_n",
    ]
    missing = sorted(set(columns) - set(reference_predictions.columns))
    if missing:
        raise ValueError(f"reference_predictions is missing candidate output columns: {missing}")
    reference = reference_predictions.loc[:, columns].copy()
    if reference["record_id"].isna().any():
        raise ValueError("reference_predictions contains missing record_id")
    reference["record_id"] = reference["record_id"].astype(str)
    if reference["record_id"].duplicated().any():
        raise ValueError("reference_predictions contains duplicate record_id")
    if set(reference["record_id"]) != set(fixture_ids):
        raise ValueError("fixture_frame and reference_predictions record_id sets differ")
    return reference.set_index("record_id").loc[fixture_ids].reset_index()


def _assert_prediction_parity(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    for column in ("score", "D", "B", "C"):
        actual_values = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
        expected_values = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
        if not np.isfinite(actual_values).all() or not np.isfinite(expected_values).all():
            raise ValueError(f"candidate {column} predictions must be finite")
        if not np.allclose(actual_values, expected_values, rtol=0.0, atol=1e-6):
            raise ValueError(f"saved candidate {column} predictions do not match reference_predictions")
    for column in ("record_id", "route", "mapping_found", "guide_support_n", "template_support_n"):
        if actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f"saved candidate {column} values do not match reference_predictions")


def _copy_candidate_sources(stage: Path) -> dict[str, str]:
    source_src = Path(__file__).resolve().parents[1]
    copied: dict[str, str] = {}
    package_src = stage / "src"
    for package in ("features", "models", "inference"):
        source_dir = source_src / package
        if not source_dir.is_dir():
            raise FileNotFoundError(f"required source package is missing: {source_dir}")
        for source in sorted(source_dir.rglob("*.py")):
            relative = source.relative_to(source_src)
            text = source.read_text(encoding="utf-8")
            private_path_markers = ("/" + "Users/", "/" + "Downloads/", "\\" + "Downloads\\")
            if any(marker in text for marker in private_path_markers):
                raise ValueError(f"candidate source contains a private workspace path: {relative}")
            output = package_src / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
            copied[relative.as_posix()] = _sha256(output)
    return copied


def _candidate_cli_text() -> str:
    return textwrap.dedent(
        '''\
        from __future__ import annotations

        import argparse
        import sys
        from pathlib import Path

        import pandas as pd

        PACKAGE_ROOT = Path(__file__).resolve().parent
        sys.path.insert(0, str(PACKAGE_ROOT / "src"))

        from inference import load_model_bundle, predict_candidate_frame


        def main() -> None:
            parser = argparse.ArgumentParser(description="Predict with the frozen TriCasRank candidate")
            parser.add_argument("--input", required=True, help="Input CSV with record_id, crRNA_sequence, target_aligned_25")
            parser.add_argument("--output", required=True, help="Output CSV path")
            args = parser.parse_args()
            input_path = Path(args.input)
            output_path = Path(args.output)
            if input_path.resolve() == output_path.resolve():
                raise ValueError("--output must not overwrite --input")
            query = pd.read_csv(input_path, dtype={"record_id": str, "crRNA_sequence": str, "target_aligned_25": str})
            model_bundle = load_model_bundle(PACKAGE_ROOT)
            predictions = predict_candidate_frame(query, model_bundle)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(output_path, index=False)


        if __name__ == "__main__":
            main()
        '''
    )


def _candidate_readme_text(metadata: Mapping[str, Any]) -> str:
    provenance = json.dumps(dict(metadata), indent=2, sort_keys=True, ensure_ascii=False)
    return textwrap.dedent(
        f'''\
        # TriCasRank candidate model

        This directory contains a loadable candidate fitted with the frozen
        TriCasRank recipe. It stores the original nine learned components:
        three D learners, five B seed learners, and one C learner. It does not
        contain or fit the standalone strong-mapping baseline.

        ## Run

        Install the pinned runtime dependencies from `requirements-core.txt`,
        then run from this directory:

        ```bash
        python predict_candidate.py --input fixture_input.csv --output predictions.csv
        ```

        Input requires `record_id`, `crRNA_sequence`, and `target_aligned_25`.
        The command retains input columns and appends `score`, component scores
        `D/B/C`, `route`, `mapping_found`, `guide_support_n`, and
        `template_support_n`. An unmapped row uses the frozen D fallback for
        `score`; an unknown mapping category by itself does not trigger fallback.

        `fixture_input.csv` is the minimal sequence-only example. Its expected
        output scores and routing diagnostics are in `expected_predictions.csv`.
        The expected file is a reference fixture, not an external test set.

        ## Contents

        - `candidate_model.joblib`: nine fitted estimators, fitted feature state,
          and resolved recipe.
        - `candidate_manifest.json`: integrity hashes, package versions and
          source-file hashes.
        - `src/`: copied path-independent feature, model, and inference code.
        - `requirements-core.txt`: pinned inference dependencies.

        ## Provenance metadata

        ```json
        {provenance}
        ```

        Provenance is informational only. Runtime inference reads the local
        candidate artifact and `src/`; it does not resolve source workspaces.
        '''
    )


def export_candidate_package(
    components: Mapping[str, Any],
    destination: str | Path,
    *,
    feature_state: Any,
    recipe: Mapping[str, Any],
    metadata: Mapping[str, Any],
    fixture_frame: pd.DataFrame,
    reference_predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Export an independent, loadable nine-learner candidate package.

    `reference_predictions` must be the output of
    `predict_candidate_frame(fixture_frame, bundle)` from the fitted training
    process. Export validates saved-model parity before publishing the package.
    The destination must not already exist; construction occurs in a sibling
    temporary directory and is renamed into place only after validation.
    """
    _candidate_components(components)
    if feature_state is None:
        raise ValueError("candidate export requires the fitted FeatureState")
    if "SEQ_MAPPING_XGB" in components:
        raise ValueError("candidate export must not include SEQ_MAPPING_XGB")
    expected_weights = {"D": 0.20, "B": 0.47, "C": 0.33}
    if dict(recipe.get("weights", {}).get("TriCasRank", {})) != expected_weights:
        raise ValueError("recipe does not contain the frozen TriCasRank fusion weights")
    fixture_ids = _fixture_ids(fixture_frame)
    fixture = fixture_frame.copy()
    fixture["record_id"] = fixture["record_id"].astype(str)
    expected = _aligned_expected_predictions(fixture_ids, reference_predictions)
    try:
        metadata_json = json.dumps(dict(metadata), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain only finite JSON-compatible provenance values") from exc
    metadata_jsonable = json.loads(metadata_json)

    folder = Path(destination).expanduser().resolve()
    source_src = Path(__file__).resolve().parents[1]
    if folder == source_src or source_src in folder.parents:
        raise ValueError("candidate destination must be outside the source package tree")
    if folder.exists():
        raise FileExistsError(f"candidate destination already exists: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{folder.name}.build-", dir=folder.parent) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        source_hashes = _copy_candidate_sources(stage)
        save_model_bundle(
            components,
            stage,
            feature_state=feature_state,
            recipe=recipe,
            metadata=metadata_jsonable,
        )
        (stage / "requirements-core.txt").write_text(
            "numpy==1.26.4\npandas==2.2.2\nscipy==1.13.1\nscikit-learn==1.5.1\n"
            "xgboost==2.1.4\nlightgbm==4.6.0\njoblib==1.4.2\n",
            encoding="utf-8",
        )
        (stage / "README.md").write_text(
            _candidate_readme_text(metadata_jsonable),
            encoding="utf-8",
        )
        (stage / "predict_candidate.py").write_text(_candidate_cli_text(), encoding="utf-8")
        fixture.to_csv(stage / "fixture_input.csv", index=False)
        expected.to_csv(stage / "expected_predictions.csv", index=False)

        loaded = load_model_bundle(stage)
        # This is a save/load check in the training process. The delivered CLI
        # resolves the copied package-local source from its own directory.
        from features import transform_query

        actual = predict_candidate_frame(
            fixture,
            loaded,
            transform_query_fn=transform_query,
        )
        actual = actual.set_index("record_id").loc[fixture_ids].reset_index()
        _assert_prediction_parity(actual, expected)

        manifest_path = stage / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "learner_count": 9,
                "candidate_model": "TriCasRank",
                "standalone_mapping_baseline_included": False,
                "source_files_sha256": source_hashes,
                "fixture_input_sha256": _sha256(stage / "fixture_input.csv"),
                "expected_predictions_sha256": _sha256(stage / "expected_predictions.csv"),
                "requirements_sha256": _sha256(stage / "requirements-core.txt"),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Catch an accidental incomplete copy before making the directory visible.
        compile((stage / "predict_candidate.py").read_text(encoding="utf-8"), str(stage / "predict_candidate.py"), "exec")
        stage.replace(folder)
        return manifest


__all__ = [
    "ARTIFACT_FILENAME",
    "MANIFEST_FILENAME",
    "export_candidate_package",
    "load_model_bundle",
    "predict_candidate_bundle",
    "predict_candidate_frame",
    "save_model_bundle",
]
