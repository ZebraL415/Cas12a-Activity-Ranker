"""One-process nested grouped selection runner for the approved R03 supplement."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

R03_ROOT = Path(__file__).resolve().parent
ROOT = R03_ROOT.parents[1]
PLAN_PATH = R03_ROOT / "selection.json"
IMPLEMENTATION_VERSION = "R03-nested-selection-20261013-v1"
EVALUATION_KIND = "nested_grouped_selection"
SELECTION_ID = "R03_grid005_inner4_SCC_20261013"
COMPONENT_COLUMNS = (
    "D_xgb", "D_lightgbm", "D_mlp", "B", "C", "SEQ_MAPPING_XGB", "HISTORY_PRIOR",
    "mapping_found", "guide_support_n", "template_support_n",
)
MODEL_ORDER = (
    "D_nested", "SEQ_MAPPING_XGB", "D_B_nested", "TriCasRank_nested", "HISTORY_PRIOR",
    "SELECTED_POLICY", "TriCasRank_fixed", "D_fixed",
)

sys.path.insert(0, str(R03_ROOT / "src"))

import numpy as np
import pandas as pd
from features import build_features
from models import fit_predict, resolved_recipe
from protocols import load_inputs
from protocols.inputs import make_encoding_fold_map
from selection import predictions_for_outer, select_policy
from splits import make_nested_splits


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_plan() -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    source_root = (PLAN_PATH.parent / plan["source_execution"]).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Configured source execution root is unavailable: {source_root}")
    source_config_path = source_root / "config" / "run.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_config["source_paths"]["project_root"] = str(source_root)
    return plan, source_root, source_config


def _verify_frozen_recipe(source_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    actual = resolved_recipe()
    frozen_path = source_root / "config" / "frozen_recipe_resolved.json"
    expected = json.loads(frozen_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("R03 copied learner recipe differs from the pre-fit frozen resolved recipe")
    weights = actual["weights"]
    d_original = plan["d_original"]
    db_original = plan["db_original"]
    tri_original = plan["tri_original"]
    if not np.allclose([weights["D_components"][k] for k in ("xgb", "lightgbm", "mlp")], d_original, atol=1e-12, rtol=0):
        raise RuntimeError("selection.json D original weights differ from the frozen model recipe")
    if not np.allclose([weights["D_B_FIXED"][k] for k in ("D", "B")], db_original, atol=1e-12, rtol=0):
        raise RuntimeError("selection.json D/B original weights differ from the frozen model recipe")
    if not np.allclose([weights["TriCasRank"][k] for k in ("D", "B", "C")], tri_original, atol=1e-12, rtol=0):
        raise RuntimeError("selection.json TriCasRank original weights differ from the frozen model recipe")
    if not np.allclose(plan["tri_extra_boundary"], [*db_original, 0.0], atol=1e-12, rtol=0):
        raise RuntimeError("tri_extra_boundary must equal the frozen D/B weights with C=0")
    return actual


def _load_cohort(
    cohort_id: str,
    source_root: Path,
    source_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    dev, _historical, assignments, sequence_manifest = load_inputs(dict(source_config))
    if cohort_id == "core_dev":
        frame = dev
        expected_n = 8417
    elif cohort_id == "common_dev":
        context_path = Path(source_config["source_paths"]["easydesign_contexts"])
        if not context_path.is_absolute():
            context_path = Path(source_config["source_paths"]["project_root"]) / context_path
        context = pd.read_csv(context_path, dtype={"record_id": str}, low_memory=False)
        context = context.loc[
            context.baseline_split.eq("baseline_train") & context.context_status.eq("ok"),
            ["record_id", "target_context_45", "guide"],
        ]
        context["record_id"] = context.record_id.astype(str)
        frame = dev.merge(context, on="record_id", how="inner", validate="one_to_one", sort=False)
        if "guide" not in frame or not frame.guide.astype(str).eq(frame.crRNA_sequence.astype(str)).all():
            raise RuntimeError("Common cohort guide values do not match the frozen development guide IDs")
        expected_n = 5995
    else:
        raise ValueError(f"Unknown cohort: {cohort_id}")
    frame = frame.reset_index(drop=True)
    if len(frame) != expected_n or frame.record_id.astype(str).nunique() != expected_n:
        raise RuntimeError(f"{cohort_id} expected {expected_n} unique records; got {len(frame)}")
    if cohort_id == "core_dev":
        expected = pd.read_csv(source_root / "data" / "development_labels.csv", dtype={"record_id": str})
        expected_ids = set(expected.record_id.astype(str))
        if len(expected) != expected_n or set(frame.record_id.astype(str)) != expected_ids:
            raise RuntimeError("core_dev IDs differ from the frozen supplement development-label roster")
        expected_labels = expected.set_index("record_id").y_true.astype(float)
    else:
        expected = pd.read_csv(source_root / "data" / "common_record_ids.csv", dtype={"record_id": str})
        expected_ids = set(expected.record_id.astype(str))
        if len(expected_ids) != expected_n or set(frame.record_id.astype(str)) != expected_ids:
            raise RuntimeError("common_dev IDs differ from the frozen common-cohort roster")
        labels = pd.read_csv(source_root / "data" / "development_labels.csv", dtype={"record_id": str})
        expected_labels = labels.loc[labels.record_id.astype(str).isin(expected_ids)].set_index("record_id").y_true.astype(float)
    observed_labels = frame.set_index(frame.record_id.astype(str))["label_normalized"].astype(float)
    observed_labels.index.name = "record_id"
    expected_labels.index = expected_labels.index.astype(str)
    if not expected_labels.index.is_unique or len(expected_labels) != expected_n:
        raise RuntimeError(f"Frozen {cohort_id} outcome roster is not one-to-one")
    aligned_labels = observed_labels.reindex(expected_labels.index)
    if aligned_labels.isna().any() or not np.allclose(
        aligned_labels.to_numpy(float), expected_labels.to_numpy(float), atol=1e-12, rtol=0
    ):
        raise RuntimeError(f"{cohort_id} normalized labels differ from the frozen R03 outcome roster")
    if frame.label_normalized.isna().any() or not np.isfinite(frame.label_normalized.to_numpy(float)).all():
        raise RuntimeError(f"{cohort_id} contains non-finite normalized labels")
    return frame, assignments, sequence_manifest


def _id_frame(frame: pd.DataFrame, fold_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": frame.record_id.astype(str).to_numpy(),
            "fold": pd.to_numeric(frame[fold_column], errors="raise").astype(int).to_numpy(),
        }
    )


def _id_file_matches(path: Path, expected: pd.DataFrame) -> bool:
    if not path.is_file():
        return False
    observed = pd.read_csv(path, dtype={"record_id": str})
    return list(observed.columns) == ["record_id", "fold"] and observed.equals(expected.reset_index(drop=True))


def _component_file_matches(path: Path, expected_ids: Sequence[str]) -> bool:
    if not path.is_file():
        return False
    observed = pd.read_csv(path, dtype={"record_id": str}, usecols=["record_id"])
    return observed.record_id.astype(str).tolist() == [str(value) for value in expected_ids]


def _query_without_labels(frame: pd.DataFrame) -> pd.DataFrame:
    forbidden = [name for name in frame.columns if name.startswith("label_") or name == "y_true"]
    result = frame.drop(columns=forbidden).copy()
    if any(name.startswith("label_") or name == "y_true" for name in result.columns):
        raise RuntimeError("Query label-column isolation failed")
    return result


def _fit_one_component_table(
    fit_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    sequence_manifest: Any,
    protocol_id: str,
    cohort_id: str,
    outer_fold: int,
    inner_fold: int | None,
    source_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, float]:
    fit_ids = set(fit_frame.record_id.astype(str))
    fit_assignments = assignments.loc[assignments.record_id.astype(str).isin(fit_ids)].copy()
    if len(fit_assignments) != len(fit_frame) or set(fit_assignments.record_id.astype(str)) != fit_ids:
        raise RuntimeError("Current inner/outer fit IDs do not match assignment records")
    encoding_folds = make_encoding_fold_map(fit_assignments, protocol_id)

    # Keep outcomes separate: feature transformation and every learner see only this label-free query.
    query_features = _query_without_labels(query_frame)
    query_outcomes = query_frame[["record_id", f"{protocol_id}_group", "label_normalized"]].copy()
    query_outcomes["record_id"] = query_outcomes.record_id.astype(str)
    query_outcomes = query_outcomes.rename(
        columns={f"{protocol_id}_group": "group_id", "label_normalized": "y_true"}
    )
    y_fit = pd.to_numeric(fit_frame.label_normalized, errors="raise").to_numpy(float)

    started = time.monotonic()
    bundle = build_features(
        fit_frame,
        query_features,
        sequence_manifest=sequence_manifest,
        protocol_id=protocol_id,
        encoding_fold_by_record_id=encoding_folds,
        recipe=source_config["model_recipe"],
    )
    output = fit_predict(bundle, y_fit, include_mapping_baseline=True, return_models=False)
    fit_seconds = time.monotonic() - started
    component_predictions = output["component_predictions"]
    missing_components = sorted(set(COMPONENT_COLUMNS[:7]) - set(component_predictions))
    if missing_components:
        raise RuntimeError(f"Fitted component output is missing: {missing_components}")
    diagnostics = output["diagnostics"]
    if diagnostics.record_id.astype(str).tolist() != query_frame.record_id.astype(str).tolist():
        raise RuntimeError("Feature diagnostics changed query ID order")
    components = pd.DataFrame({"record_id": diagnostics.record_id.astype(str).to_numpy()})
    for name in COMPONENT_COLUMNS[:7]:
        values = np.asarray(component_predictions[name], dtype=float).reshape(-1)
        if len(values) != len(query_frame) or not np.isfinite(values).all():
            raise RuntimeError(f"Component {name} has missing/non-finite query predictions")
        components[name] = values
    for name in ("mapping_found", "guide_support_n", "template_support_n"):
        components[name] = diagnostics[name].to_numpy()

    # Labels and protocol groups enter the table only after all predictions are materialized.
    components = components.merge(query_outcomes, on="record_id", how="left", validate="one_to_one", sort=False)
    if components.y_true.isna().any() or not np.isfinite(components.y_true.to_numpy(float)).all():
        raise RuntimeError("Post-prediction outcome join left missing query labels")
    if components.record_id.astype(str).tolist() != query_frame.record_id.astype(str).tolist():
        raise RuntimeError("Post-prediction outcome join changed query ID order")
    components.insert(0, "group_id", components.pop("group_id").astype(str))
    components.insert(0, "record_id", components.pop("record_id").astype(str))
    components.insert(0, "inner_fold", inner_fold if inner_fold is not None else pd.NA)
    components.insert(0, "outer_fold", int(outer_fold))
    components.insert(0, "protocol_id", protocol_id)
    components.insert(0, "cohort_id", cohort_id)
    ordered = [
        "cohort_id", "protocol_id", "outer_fold", "inner_fold", "record_id", "group_id", "y_true",
        *COMPONENT_COLUMNS,
    ]
    return components.loc[:, ordered], fit_seconds


def _load_easydesign_rows(
    source_root: Path,
    cohort_id: str,
    protocol_id: str,
    outer_fold: int,
    outer_fit: pd.DataFrame,
    outer_query: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_parent = source_root / "runs" / f"common_{protocol_id}"
    candidates = sorted(run_parent.glob(f"*/fold_{outer_fold}"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one existing common/{protocol_id}/fold_{outer_fold} run, got {len(candidates)}")
    fold_dir = candidates[0]
    ed_dir = fold_dir / "easydesign"
    fold_status = json.loads((fold_dir / "status.json").read_text(encoding="utf-8"))
    ed_status = json.loads((ed_dir / "status.json").read_text(encoding="utf-8"))
    if fold_status.get("state") != "complete" or ed_status.get("state") != "complete":
        raise RuntimeError(f"Existing EasyDesign outer fold is not complete: {fold_dir}")
    fit_ids = pd.read_csv(ed_dir / "fit_input.csv", dtype={"record_id": str}).record_id.astype(str)
    query_ids = pd.read_csv(ed_dir / "query_input.csv", dtype={"record_id": str}).record_id.astype(str)
    expected_fit_ids = set(outer_fit.record_id.astype(str))
    expected_query_ids = set(outer_query.record_id.astype(str))
    if len(fit_ids) != len(expected_fit_ids) or set(fit_ids) != expected_fit_ids:
        raise RuntimeError("Existing EasyDesign fit IDs differ from this common outer fit")
    if len(query_ids) != len(expected_query_ids) or set(query_ids) != expected_query_ids:
        raise RuntimeError("Existing EasyDesign query IDs differ from this common outer query")
    predictions = pd.read_csv(ed_dir / "predictions.csv", dtype={"record_id": str})
    predictions["record_id"] = predictions.record_id.astype(str)
    if predictions.record_id.duplicated().any() or set(predictions.record_id) != expected_query_ids:
        raise RuntimeError("Existing EasyDesign prediction IDs are not a one-to-one match to this query")
    query_meta = outer_query[["record_id", f"{protocol_id}_group", "label_normalized"]].copy()
    query_meta["record_id"] = query_meta.record_id.astype(str)
    query_meta = query_meta.rename(columns={f"{protocol_id}_group": "group_id", "label_normalized": "y_true"})
    rows = query_meta.merge(predictions, on="record_id", how="left", validate="one_to_one", sort=False)
    if rows.y_pred.isna().any() or not np.isfinite(rows.y_pred.to_numpy(float)).all():
        raise RuntimeError("Existing EasyDesign query predictions are missing or non-finite")
    meta = {
        "status": "complete",
        "source_run": str(fold_dir.relative_to(source_root)),
        "fit_ids_match": True,
        "query_ids_match": True,
        "n_fit": int(len(fit_ids)),
        "n_query": int(len(query_ids)),
        "source_fit_ids_sha256_recorded": ed_status.get("fit_ids_sha256"),
        "source_query_ids_sha256_recorded": ed_status.get("query_ids_sha256"),
    }
    return rows, meta


def _fixed_recipe(resolved: Mapping[str, Any]) -> dict[str, Any]:
    weights = resolved["weights"]
    d = weights["D_components"]
    db = weights["D_B_FIXED"]
    tri = weights["TriCasRank"]
    return {
        "xgb": float(d["xgb"]),
        "lightgbm": float(d["lightgbm"]),
        "mlp": float(d["mlp"]),
        "db_fusion": {key: float(db[key]) for key in ("D", "B")},
        "fusion": {key: float(tri[key]) for key in ("D", "B", "C")},
    }


def _prediction_rows(
    outer_components: pd.DataFrame,
    outer_query: pd.DataFrame,
    selection: Mapping[str, Any],
    fixed_recipe: Mapping[str, Any],
    cohort_id: str,
    protocol_id: str,
    outer_fold: int,
    easydesign_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    component_predictions = predictions_for_outer(outer_components, selection, fixed_recipe)
    query_meta = outer_query[["record_id", f"{protocol_id}_group", "label_normalized"]].copy()
    query_meta["record_id"] = query_meta.record_id.astype(str)
    query_meta = query_meta.rename(columns={f"{protocol_id}_group": "group_id", "label_normalized": "y_true"})
    tables = []
    model_order = list(MODEL_ORDER)
    if easydesign_rows is not None:
        model_order.append("EasyDesign-adapted")
    for model_id in model_order:
        if model_id == "EasyDesign-adapted":
            source = easydesign_rows[["record_id", "group_id", "y_true", "y_pred"]].copy()
            source["prediction_route"] = "easydesign"
            predicted = source
        else:
            y_pred, route = component_predictions[model_id]
            if len(y_pred) != len(outer_components) or len(route) != len(outer_components):
                raise RuntimeError(f"Output length mismatch for {model_id}")
            predicted = pd.DataFrame(
                {
                    "record_id": outer_components.record_id.astype(str).to_numpy(),
                    "y_pred": np.asarray(y_pred, dtype=float),
                    "prediction_route": np.asarray(route, dtype=str),
                }
            )
            predicted = predicted.merge(query_meta, on="record_id", how="left", validate="one_to_one", sort=False)
        if predicted.record_id.astype(str).tolist() != query_meta.record_id.astype(str).tolist():
            raise RuntimeError(f"Prediction ID order mismatch for {model_id}")
        if not np.isfinite(predicted.y_pred.to_numpy(float)).all():
            raise RuntimeError(f"Non-finite query predictions for {model_id}")
        table = predicted[["record_id", "group_id", "y_true", "y_pred", "prediction_route"]].copy()
        table.insert(0, "model_id", model_id)
        table.insert(0, "outer_fold", int(outer_fold))
        table.insert(0, "protocol_id", protocol_id)
        table.insert(0, "cohort_id", cohort_id)
        table["evaluation_kind"] = EVALUATION_KIND
        table["selection_id"] = SELECTION_ID
        if model_id in {"SELECTED_POLICY"}:
            table["selected_family"] = selection["selected_family"]
        else:
            table["selected_family"] = ""
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    expected_n = len(outer_query)
    counts = result.groupby("model_id", sort=False).size()
    if set(counts.index) != set(model_order) or not counts.eq(expected_n).all():
        raise RuntimeError("Final per-fold predictions do not have exact model/record coverage")
    if result.duplicated(["record_id", "model_id"]).any():
        raise RuntimeError("Final per-fold predictions contain duplicate record/model pairs")
    return result


def _numeric_regression_check(
    source_root: Path,
    cohort_id: str,
    protocol_id: str,
    outer_fold: int,
    outer_components: pd.DataFrame,
    selection: Mapping[str, Any],
    fixed_recipe: Mapping[str, Any],
) -> dict[str, Any] | None:
    if (cohort_id, protocol_id, outer_fold) != ("core_dev", "target", 0):
        return None
    candidates = sorted((source_root / "runs" / "core_target").glob("*/fold_0/predictions.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one old core/target/fold_0 prediction snapshot, got {len(candidates)}")
    old = pd.read_csv(candidates[0], dtype={"record_id": str})
    old = old.loc[old.model_id.isin(["D", "SEQ_MAPPING_XGB", "D_B_FIXED", "TriCasRank"])].copy()
    old_wide = old.pivot(index="record_id", columns="model_id", values="y_pred").add_prefix("old_")
    current = predictions_for_outer(outer_components, selection, fixed_recipe)
    current_frame = pd.DataFrame({"record_id": outer_components.record_id.astype(str)})
    for name, (values, _routes) in current.items():
        current_frame[name] = np.asarray(values, dtype=float)
    current_frame = current_frame.set_index("record_id")
    components_by_id = outer_components.set_index("record_id")
    db_weights = fixed_recipe["db_fusion"]
    current_frame["D_B_FIXED_reconstructed"] = np.where(
        components_by_id.mapping_found.astype(bool).to_numpy(),
        db_weights["D"] * current_frame.D_fixed.to_numpy() + db_weights["B"] * components_by_id.B.to_numpy(float),
        current_frame.D_fixed.to_numpy(),
    )
    joined = old_wide.join(current_frame, how="outer", validate="one_to_one")
    if joined.isna().any().any():
        raise RuntimeError("Old and reconstructed core/target/fold_0 regression IDs differ")
    comparisons = {
        "D_fixed_vs_old_D": (joined["D_fixed"], joined["old_D"]),
        "SEQ_MAPPING_XGB_vs_old": (joined["SEQ_MAPPING_XGB"], joined["old_SEQ_MAPPING_XGB"]),
        "D_B_fixed_reconstruction_vs_old": (joined["D_B_FIXED_reconstructed"], joined["old_D_B_FIXED"]),
        "TriCasRank_fixed_vs_old_TriCasRank": (joined["TriCasRank_fixed"], joined["old_TriCasRank"]),
    }
    deltas = {name: float(np.max(np.abs(left.to_numpy(float) - right.to_numpy(float)))) for name, (left, right) in comparisons.items()}
    tolerance = 1e-8
    passed = all(value <= tolerance for value in deltas.values())
    report = {
        "status": "pass" if passed else "fail",
        "old_prediction_file": str(candidates[0].relative_to(source_root)),
        "n_query": int(len(joined)),
        "tolerance": tolerance,
        "max_absolute_deltas": deltas,
    }
    _write_json(R03_ROOT / "runs" / cohort_id / protocol_id / f"outer_{outer_fold}" / "numeric_regression.json", report)
    if not passed:
        raise RuntimeError(f"Original-weight numeric regression failed: {deltas}")
    return report


def _run_outer(
    *,
    cohort_id: str,
    protocol_id: str,
    outer_fold: int,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    sequence_manifest: Any,
    source_root: Path,
    source_config: Mapping[str, Any],
    selection_config: Mapping[str, Any],
    fixed_recipe: Mapping[str, Any],
) -> dict[str, Any]:
    fold_column = f"{protocol_id}_fold"
    output_dir = R03_ROOT / "runs" / cohort_id / protocol_id / f"outer_{outer_fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_fit, outer_query, inner_splits = make_nested_splits(frame, protocol_id, outer_fold)
    outer_fit_ids = _id_frame(outer_fit, fold_column)
    outer_query_ids = _id_frame(outer_query, fold_column)
    _write_csv(output_dir / "outer_fit_ids.csv", outer_fit_ids)
    _write_csv(output_dir / "outer_query_ids.csv", outer_query_ids)

    status_path = output_dir / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("selection_config") != selection_config:
            raise RuntimeError(f"Saved selector configuration differs; refusing to reuse {output_dir}")
        if status.get("implementation_version") != IMPLEMENTATION_VERSION:
            raise RuntimeError(f"Saved implementation version differs; refusing to reuse {output_dir}")
    else:
        status = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "selection_config": selection_config,
            "cohort_id": cohort_id,
            "protocol_id": protocol_id,
            "outer_fold": int(outer_fold),
            "created_at": timestamp(),
            "inner_fits": {},
            "outer_fit": {"state": "not_started"},
        }
    if not _id_file_matches(output_dir / "outer_fit_ids.csv", outer_fit_ids) or not _id_file_matches(
        output_dir / "outer_query_ids.csv", outer_query_ids
    ):
        raise RuntimeError("Outer fit/query ID files do not match current assignment rows")
    status.update(state="running", updated_at=timestamp(), outer_fit_n=len(outer_fit), outer_query_n=len(outer_query))
    _write_json(status_path, status)

    inner_tables = []
    inner_fit_counts: dict[str, int] = {}
    inner_query_counts: dict[str, int] = {}
    for split in inner_splits:
        inner_fold = int(split["inner_fold"])
        inner_fit = split["fit"]
        inner_query = split["validation"]
        inner_key = str(inner_fold)
        inner_dir = output_dir / f"inner_{inner_fold}"
        inner_dir.mkdir(parents=True, exist_ok=True)
        expected_fit_ids = _id_frame(inner_fit, fold_column)
        expected_query_ids = _id_frame(inner_query, fold_column)
        _write_csv(inner_dir / "fit_ids.csv", expected_fit_ids)
        _write_csv(inner_dir / "validation_ids.csv", expected_query_ids)
        component_path = inner_dir / "components.csv"
        inner_state = status.setdefault("inner_fits", {}).get(inner_key, {})
        reusable = (
            inner_state.get("state") == "complete"
            and _id_file_matches(inner_dir / "fit_ids.csv", expected_fit_ids)
            and _id_file_matches(inner_dir / "validation_ids.csv", expected_query_ids)
            and _component_file_matches(component_path, inner_query.record_id.astype(str).tolist())
        )
        if reusable:
            components = pd.read_csv(component_path, dtype={"record_id": str, "group_id": str})
            print(f"REUSE {cohort_id}/{protocol_id}/outer_{outer_fold}/inner_{inner_fold}", flush=True)
        else:
            status["inner_fits"][inner_key] = {
                "state": "running", "started_at": timestamp(), "fit_n": len(inner_fit),
                "validation_n": len(inner_query), "fit_ids_file": "fit_ids.csv",
                "validation_ids_file": "validation_ids.csv", "component_path": f"inner_{inner_fold}/components.csv",
            }
            status["updated_at"] = timestamp()
            _write_json(status_path, status)
            print(
                f"START {cohort_id}/{protocol_id}/outer_{outer_fold}/inner_{inner_fold}: "
                f"fit={len(inner_fit)} validation={len(inner_query)}",
                flush=True,
            )
            started = time.monotonic()
            components, fit_seconds = _fit_one_component_table(
                inner_fit, inner_query, assignments, sequence_manifest, protocol_id,
                cohort_id, outer_fold, inner_fold, source_config,
            )
            _write_csv(component_path, components)
            elapsed = time.monotonic() - started
            status["inner_fits"][inner_key] = {
                "state": "complete", "started_at": status["inner_fits"][inner_key]["started_at"],
                "finished_at": timestamp(), "fit_seconds": float(fit_seconds), "elapsed_seconds": float(elapsed),
                "fit_n": len(inner_fit), "validation_n": len(inner_query),
                "fit_ids_file": "fit_ids.csv", "validation_ids_file": "validation_ids.csv",
                "component_path": f"inner_{inner_fold}/components.csv",
            }
            status["updated_at"] = timestamp()
            _write_json(status_path, status)
            print(
                f"COMPLETE {cohort_id}/{protocol_id}/outer_{outer_fold}/inner_{inner_fold}: "
                f"fit_seconds={fit_seconds:.1f} elapsed={elapsed:.1f}",
                flush=True,
            )
        inner_tables.append(components)
        inner_fit_counts[inner_key] = int(len(inner_fit))
        inner_query_counts[inner_key] = int(len(inner_query))

    # The selector is deliberately given inner validation predictions and labels only.
    selection, candidate_scores = select_policy(inner_tables, selection_config)
    selection.update(
        {
            "cohort_id": cohort_id,
            "protocol_id": protocol_id,
            "outer_fold": int(outer_fold),
            "evaluation_kind": EVALUATION_KIND,
            "selection_id": SELECTION_ID,
            "outer_fit_n": int(len(outer_fit)),
            "outer_query_n": int(len(outer_query)),
            "inner_fit_counts": inner_fit_counts,
            "inner_validation_counts": inner_query_counts,
            "fit_ids_file": "outer_fit_ids.csv",
            "query_ids_file": "outer_query_ids.csv",
            "fixed_recipe_weights": {
                "D": [fixed_recipe["xgb"], fixed_recipe["lightgbm"], fixed_recipe["mlp"]],
                "TriCasRank": fixed_recipe["fusion"],
            },
        }
    )
    _write_csv(output_dir / "candidate_scores.csv", candidate_scores)
    _write_json(output_dir / "selection.json", selection)
    status["selector"] = {
        "state": "complete", "selected_family": selection["selected_family"],
        "d_weights": selection["d_weights"], "db_weights": selection["db_weights"],
        "tri_weights": selection["tri_weights"], "candidate_scores_path": "candidate_scores.csv",
    }
    status["updated_at"] = timestamp()
    _write_json(status_path, status)

    outer_component_path = output_dir / "outer_components.csv"
    outer_state = status.get("outer_fit", {})
    outer_reusable = (
        outer_state.get("state") == "complete"
        and _id_file_matches(output_dir / "outer_fit_ids.csv", outer_fit_ids)
        and _id_file_matches(output_dir / "outer_query_ids.csv", outer_query_ids)
        and _component_file_matches(outer_component_path, outer_query.record_id.astype(str).tolist())
    )
    if outer_reusable:
        outer_components = pd.read_csv(outer_component_path, dtype={"record_id": str, "group_id": str})
        print(f"REUSE {cohort_id}/{protocol_id}/outer_{outer_fold}/outer_components", flush=True)
    else:
        status["outer_fit"] = {
            "state": "running", "started_at": timestamp(), "fit_n": len(outer_fit),
            "query_n": len(outer_query), "fit_ids_file": "outer_fit_ids.csv",
            "query_ids_file": "outer_query_ids.csv", "component_path": "outer_components.csv",
        }
        status["updated_at"] = timestamp()
        _write_json(status_path, status)
        print(
            f"START {cohort_id}/{protocol_id}/outer_{outer_fold}: fit={len(outer_fit)} query={len(outer_query)}",
            flush=True,
        )
        started = time.monotonic()
        outer_components, fit_seconds = _fit_one_component_table(
            outer_fit, outer_query, assignments, sequence_manifest, protocol_id,
            cohort_id, outer_fold, None, source_config,
        )
        _write_csv(outer_component_path, outer_components)
        status["outer_fit"] = {
            "state": "complete", "started_at": status["outer_fit"]["started_at"], "finished_at": timestamp(),
            "fit_seconds": float(fit_seconds), "elapsed_seconds": float(time.monotonic() - started),
            "fit_n": len(outer_fit), "query_n": len(outer_query),
            "fit_ids_file": "outer_fit_ids.csv", "query_ids_file": "outer_query_ids.csv",
            "component_path": "outer_components.csv",
        }
        status["updated_at"] = timestamp()
        _write_json(status_path, status)
        print(f"COMPLETE {cohort_id}/{protocol_id}/outer_{outer_fold}: fit_seconds={fit_seconds:.1f}", flush=True)

    regression = _numeric_regression_check(
        source_root, cohort_id, protocol_id, outer_fold, outer_components, selection, fixed_recipe
    )
    if regression is not None:
        status["numeric_regression"] = regression
        print(f"REGRESSION {cohort_id}/{protocol_id}/outer_{outer_fold}: {regression['max_absolute_deltas']}", flush=True)

    easydesign_rows = None
    easy_identity = None
    if cohort_id == "common_dev":
        easydesign_rows, easy_identity = _load_easydesign_rows(
            source_root, cohort_id, protocol_id, outer_fold, outer_fit, outer_query
        )
        _write_json(output_dir / "common_easydesign_identity.json", easy_identity)
    final_predictions = _prediction_rows(
        outer_components, outer_query, selection, fixed_recipe,
        cohort_id, protocol_id, outer_fold, easydesign_rows,
    )
    _write_csv(output_dir / "predictions.csv", final_predictions)

    fold_summary = {
        "cohort_id": cohort_id,
        "protocol_id": protocol_id,
        "outer_fold": int(outer_fold),
        "outer_fit_n": int(len(outer_fit)),
        "outer_query_n": int(len(outer_query)),
        "outer_fit_groups": int(outer_fit[f"{protocol_id}_group"].astype(str).nunique()),
        "outer_query_groups": int(outer_query[f"{protocol_id}_group"].astype(str).nunique()),
        "outer_record_id_overlap": 0,
        "outer_group_overlap": 0,
        "inner_splits": [
            {
                "inner_fold": int(split["inner_fold"]),
                "fit_n": int(len(split["fit"])),
                "validation_n": int(len(split["validation"])),
                "fit_group_n": int(split["fit"][f"{protocol_id}_group"].astype(str).nunique()),
                "validation_group_n": int(split["validation"][f"{protocol_id}_group"].astype(str).nunique()),
                "record_id_overlap": 0,
                "group_overlap": 0,
            }
            for split in inner_splits
        ],
        "n_inner_fits": int(len(inner_splits)),
        "n_outer_fits": 1,
        "input_id_files": [
            "outer_fit_ids.csv", "outer_query_ids.csv",
            *[f"inner_{split['inner_fold']}/fit_ids.csv" for split in inner_splits],
            *[f"inner_{split['inner_fold']}/validation_ids.csv" for split in inner_splits],
        ],
        "selection_id": SELECTION_ID,
        "evaluation_kind": EVALUATION_KIND,
        "selected_family": selection["selected_family"],
        "common_easydesign_identity": easy_identity,
    }
    _write_json(output_dir / "split_summary.json", fold_summary)
    status.update(
        state="complete", selected_family=selection["selected_family"],
        finished_at=timestamp(), updated_at=timestamp(), predictions_path="predictions.csv",
        outer_components_path="outer_components.csv", split_summary_path="split_summary.json",
        n_predictions=int(len(final_predictions)),
    )
    _write_json(status_path, status)
    _refresh_oof_aggregate()
    return status


def _refresh_oof_aggregate() -> dict[str, Any]:
    files = []
    for path in sorted((R03_ROOT / "runs").rglob("predictions.csv")):
        if path.parent.name.startswith("outer_"):
            status_path = path.parent / "status.json"
            if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("state") == "complete":
                files.append(path)
    predictions = [pd.read_csv(path, dtype={"record_id": str, "group_id": str}) for path in files]
    if predictions:
        aggregate = pd.concat(predictions, ignore_index=True)
        aggregate = aggregate.sort_values(
            ["cohort_id", "protocol_id", "outer_fold", "model_id", "record_id"], kind="mergesort"
        ).reset_index(drop=True)
    else:
        aggregate = pd.DataFrame()
    _write_csv(R03_ROOT / "predictions_oof.csv", aggregate)
    coverage = {
        "state": "partial",
        "completed_outer_folds": int(len(files)),
        "expected_outer_folds": 40,
        "prediction_rows": int(len(aggregate)),
        "prediction_file": "predictions_oof.csv",
        "tasks": [],
    }
    for path in files:
        status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
        coverage["tasks"].append(
            {
                "cohort_id": status["cohort_id"], "protocol_id": status["protocol_id"],
                "outer_fold": int(status["outer_fold"]), "selected_family": status["selected_family"],
                "n_predictions": int(status["n_predictions"]),
            }
        )
    if len(files) == 40:
        coverage["state"] = "complete"
        coverage["expected_inner_fits"] = 160
        coverage["observed_inner_fits"] = int(
            sum(sum(1 for item in json.loads((path.parent / "status.json").read_text(encoding="utf-8")).get("inner_fits", {}).values() if item.get("state") == "complete") for path in files)
        )
        if coverage["observed_inner_fits"] != 160:
            coverage["state"] = "partial"
    _write_json(R03_ROOT / "coverage.json", coverage)
    return coverage


@contextmanager
def _training_lock():
    R03_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = R03_ROOT / "training.lock"
    stream = lock_path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError("An R03 training run already holds the single-process lock") from exc
    try:
        yield
    finally:
        fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("core_dev", "common_dev"), help="Omit to run both cohorts sequentially")
    parser.add_argument("--protocol", choices=("target", "guide", "atomic_template", "dual_atomic"))
    parser.add_argument("--outer-fold", type=int, choices=range(5))
    args = parser.parse_args()
    plan, source_root, source_config = _read_plan()
    resolved = _verify_frozen_recipe(source_root, plan)
    fixed_recipe = _fixed_recipe(resolved)
    cohort_ids = [args.cohort] if args.cohort else ["core_dev", "common_dev"]
    protocol_ids = [args.protocol] if args.protocol else ["target", "guide", "atomic_template", "dual_atomic"]
    outer_folds = [args.outer_fold] if args.outer_fold is not None else list(range(5))
    with _training_lock():
        for cohort_id in cohort_ids:
            frame, assignments, sequence_manifest = _load_cohort(cohort_id, source_root, source_config)
            for protocol_id in protocol_ids:
                for outer_fold in outer_folds:
                    _run_outer(
                        cohort_id=cohort_id, protocol_id=protocol_id, outer_fold=outer_fold,
                        frame=frame, assignments=assignments, sequence_manifest=sequence_manifest,
                        source_root=source_root, source_config=source_config, selection_config=plan,
                        fixed_recipe=fixed_recipe,
                    )
        coverage = _refresh_oof_aggregate()
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
