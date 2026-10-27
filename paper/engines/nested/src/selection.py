"""Nested weight and model-family selection from inner-fold components only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from revision_statistics.metrics import basic_metrics


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    values: tuple[float, ...]


def _weight_tuple(values: Sequence[float], n: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != n or any(not np.isfinite(value) or value < 0 for value in result):
        raise ValueError(f"Expected {n} finite non-negative weights")
    if not np.isclose(sum(result), 1.0, atol=1e-12, rtol=0):
        raise ValueError(f"Weights must sum to one; got {sum(result)}")
    return result


def simplex_grid(n_components: int, *, step: float, extras: Iterable[Sequence[float]] = ()) -> list[tuple[float, ...]]:
    """Create a closed simplex grid and add the specified frozen weights."""
    if n_components not in (2, 3):
        raise ValueError("Only two- and three-component simplexes are authorized")
    if step <= 0:
        raise ValueError("Grid step must be positive")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0, atol=1e-12, rtol=0):
        raise ValueError("Grid step must divide one exactly")
    if n_components == 2:
        points = [(i / units, (units - i) / units) for i in range(units + 1)]
    else:
        points = [
            (i / units, j / units, (units - i - j) / units)
            for i in range(units + 1)
            for j in range(units - i + 1)
        ]
    points.extend(_weight_tuple(item, n_components) for item in extras)
    unique: dict[tuple[float, ...], tuple[float, ...]] = {}
    for item in points:
        checked = _weight_tuple(item, n_components)
        unique[tuple(round(x, 12) for x in checked)] = checked
    return sorted(unique.values())


def build_weight_grids(selection_config: Mapping[str, Any]) -> dict[str, list[tuple[float, ...]]]:
    step = float(selection_config["grid_step"])
    return {
        "D_weights": simplex_grid(3, step=step, extras=[selection_config["d_original"]]),
        "D_B_weights": simplex_grid(2, step=step, extras=[selection_config["db_original"]]),
        "TriCasRank_weights": simplex_grid(
            3,
            step=step,
            extras=[selection_config["tri_original"], selection_config["tri_extra_boundary"]],
        ),
    }


def _candidate_list(values: Sequence[Sequence[float]], prefix: str) -> list[Candidate]:
    return [Candidate(f"{prefix}_{index:04d}", tuple(float(x) for x in weights)) for index, weights in enumerate(values)]


def _metrics_for_candidate(
    inner_tables: Sequence[pd.DataFrame],
    predict: Callable[[pd.DataFrame], np.ndarray],
) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    for table in inner_tables:
        y_true = pd.to_numeric(table["y_true"], errors="raise").to_numpy(float)
        y_pred = np.asarray(predict(table), dtype=float).reshape(-1)
        metric = basic_metrics(y_true, y_pred)
        if "inner_fold" in table:
            fold_values = pd.to_numeric(table["inner_fold"], errors="raise").unique()
            if len(fold_values) != 1:
                raise ValueError("Each component table must contain exactly one inner_fold value")
            metric["inner_fold"] = int(fold_values[0])
        rows.append(metric)
    return rows


def _summarize_fold_metrics(fold_metrics: Sequence[Mapping[str, float | None]]) -> dict[str, Any]:
    if len(fold_metrics) != 4:
        raise ValueError(f"Every selection candidate must have exactly four inner folds; got {len(fold_metrics)}")
    sccs = [metric["SCC"] for metric in fold_metrics]
    maes = [metric["MAE"] for metric in fold_metrics]
    return {
        "inner_scc": sccs,
        "inner_mae": maes,
        "mean_SCC": float(np.mean(sccs)) if all(value is not None for value in sccs) else None,
        "mean_MAE": float(np.mean(maes)),
    }


def _score_records(
    stage: str,
    candidates: Sequence[Candidate],
    inner_tables: Sequence[pd.DataFrame],
    predictor: Callable[[Candidate, pd.DataFrame], np.ndarray],
    *,
    original: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        fold_metrics = _metrics_for_candidate(
            inner_tables,
            lambda table, candidate=candidate: predictor(candidate, table),
        )
        summary = _summarize_fold_metrics(fold_metrics)
        use_scc = summary["mean_SCC"] is not None
        record: dict[str, Any] = {
            "selection_stage": stage,
            "candidate_id": candidate.candidate_id,
            "weights": candidate.values,
            "mean_SCC": summary["mean_SCC"],
            "mean_MAE": summary["mean_MAE"],
            "ranking_metric": "SCC" if use_scc else "MAE",
            "ranking_value": summary["mean_SCC"] if use_scc else summary["mean_MAE"],
            "scc_valid_all_folds": use_scc,
            "fold_metrics": fold_metrics,
            "l1_to_original": (
                float(sum(abs(a - b) for a, b in zip(candidate.values, original)))
                if original is not None
                else None
            ),
        }
        records.append(record)
    # If no candidate defines SCC in all folds, the stage is explicitly MAE-ranked.
    if records and not any(record["scc_valid_all_folds"] for record in records):
        for record in records:
            record["ranking_metric"] = "MAE"
            record["ranking_value"] = record["mean_MAE"]
    return records


def _choose_weight_record(records: Sequence[Mapping[str, Any]], tolerance: float) -> Mapping[str, Any]:
    eligible = [record for record in records if record["scc_valid_all_folds"]]
    metric = "mean_SCC" if eligible else "mean_MAE"
    pool = eligible if eligible else list(records)
    if not pool:
        raise ValueError("No weight candidates were generated")
    if metric == "mean_SCC":
        optimum = max(float(record[metric]) for record in pool)
        tied = [record for record in pool if optimum - float(record[metric]) <= tolerance]
    else:
        optimum = min(float(record[metric]) for record in pool)
        tied = [record for record in pool if float(record[metric]) - optimum <= tolerance]
    mae_best = min(float(record["mean_MAE"]) for record in tied)
    tied = [record for record in tied if float(record["mean_MAE"]) - mae_best <= tolerance]
    l1_best = min(float(record["l1_to_original"]) for record in tied)
    tied = [record for record in tied if float(record["l1_to_original"]) - l1_best <= tolerance]
    return min(tied, key=lambda record: tuple(float(value) for value in record["weights"]))


def _compose_d(table: pd.DataFrame, weights: Sequence[float]) -> np.ndarray:
    return sum(float(weight) * pd.to_numeric(table[column], errors="raise").to_numpy(float)
               for weight, column in zip(weights, ("D_xgb", "D_lightgbm", "D_mlp")))


def _compose_fusion(
    table: pd.DataFrame,
    weights: Sequence[float],
    *,
    d_weights: Sequence[float],
    include_c: bool,
) -> np.ndarray:
    d = _compose_d(table, d_weights)
    mapping_found = table["mapping_found"].astype(bool).to_numpy()
    b = pd.to_numeric(table["B"], errors="raise").to_numpy(float)
    if include_c:
        c = pd.to_numeric(table["C"], errors="raise").to_numpy(float)
        mixed = weights[0] * d + weights[1] * b + weights[2] * c
    else:
        mixed = weights[0] * d + weights[1] * b
    return np.where(mapping_found, mixed, d)


def _family_prediction(
    family: str,
    table: pd.DataFrame,
    *,
    d_weights: Sequence[float],
    db_weights: Sequence[float],
    tri_weights: Sequence[float],
) -> np.ndarray:
    if family == "D":
        return _compose_d(table, d_weights)
    if family == "SEQ_MAPPING_XGB":
        return pd.to_numeric(table["SEQ_MAPPING_XGB"], errors="raise").to_numpy(float)
    if family == "D_B":
        return _compose_fusion(table, db_weights, d_weights=d_weights, include_c=False)
    if family == "TriCasRank":
        return _compose_fusion(table, tri_weights, d_weights=d_weights, include_c=True)
    raise ValueError(f"Unknown model family: {family}")


def _family_records(
    inner_tables: Sequence[pd.DataFrame],
    *,
    d_weights: Sequence[float],
    db_weights: Sequence[float],
    tri_weights: Sequence[float],
    family_order: Sequence[str],
    learners: Mapping[str, int],
) -> list[dict[str, Any]]:
    records = []
    for family in family_order:
        fold_metrics = _metrics_for_candidate(
            inner_tables,
            lambda table, family=family: _family_prediction(
                family,
                table,
                d_weights=d_weights,
                db_weights=db_weights,
                tri_weights=tri_weights,
            ),
        )
        summary = _summarize_fold_metrics(fold_metrics)
        records.append(
            {
                "selection_stage": "model_family",
                "candidate_id": family,
                "family": family,
                "weights": None,
                "mean_SCC": summary["mean_SCC"],
                "mean_MAE": summary["mean_MAE"],
                "ranking_metric": "SCC" if summary["mean_SCC"] is not None else "MAE",
                "ranking_value": summary["mean_SCC"] if summary["mean_SCC"] is not None else summary["mean_MAE"],
                "scc_valid_all_folds": summary["mean_SCC"] is not None,
                "fold_metrics": fold_metrics,
                "n_learners": int(learners[family]),
                "family_index": list(family_order).index(family),
            }
        )
    if records and not any(record["scc_valid_all_folds"] for record in records):
        for record in records:
            record["ranking_metric"] = "MAE"
            record["ranking_value"] = record["mean_MAE"]
    return records


def _choose_family_record(records: Sequence[Mapping[str, Any]], tolerance: float) -> Mapping[str, Any]:
    eligible = [record for record in records if record["scc_valid_all_folds"]]
    pool = eligible if eligible else list(records)
    metric = "mean_SCC" if eligible else "mean_MAE"
    if metric == "mean_SCC":
        optimum = max(float(record[metric]) for record in pool)
        tied = [record for record in pool if optimum - float(record[metric]) <= tolerance]
    else:
        optimum = min(float(record[metric]) for record in pool)
        tied = [record for record in pool if float(record[metric]) - optimum <= tolerance]
    mae_best = min(float(record["mean_MAE"]) for record in tied)
    tied = [record for record in tied if float(record["mean_MAE"]) - mae_best <= tolerance]
    return min(tied, key=lambda record: (int(record["n_learners"]), int(record["family_index"])))


def _record_rows(records: Sequence[Mapping[str, Any]], chosen_id: str) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        weights = record.get("weights")
        weights_json = json.dumps(list(weights), separators=(",", ":")) if weights is not None else ""
        for index, metrics in enumerate(record["fold_metrics"]):
            rows.append(
                {
                    "selection_stage": record["selection_stage"],
                    "candidate_id": record["candidate_id"],
                    "family": record.get("family", ""),
                    "weights": weights_json,
                    "inner_fold": int(metrics.get("inner_fold", index)),
                    "SCC": metrics["SCC"],
                    "MAE": metrics["MAE"],
                    "mean_SCC": record["mean_SCC"],
                    "mean_MAE": record["mean_MAE"],
                    "ranking_metric": record["ranking_metric"],
                    "ranking_value": record["ranking_value"],
                    "scc_valid_all_folds": bool(record["scc_valid_all_folds"]),
                    "selected": record["candidate_id"] == chosen_id,
                    "n_learners": record.get("n_learners", ""),
                }
            )
    return rows


def select_policy(
    inner_tables: Sequence[pd.DataFrame],
    selection_config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select weights and family using only the four labeled inner-fold tables.

    This API has no outer-query table or outer labels parameter by design.
    """
    if len(inner_tables) != 4:
        raise ValueError(f"Exactly four inner-fold tables are required; got {len(inner_tables)}")
    required = {
        "inner_fold", "record_id", "y_true", "D_xgb", "D_lightgbm", "D_mlp", "B", "C",
        "SEQ_MAPPING_XGB", "HISTORY_PRIOR", "mapping_found",
    }
    seen_folds: list[int] = []
    seen_ids: set[str] = set()
    for table in inner_tables:
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Inner component table is missing required fields: {missing}")
        if table["record_id"].astype(str).duplicated().any():
            raise ValueError("Inner component table contains duplicate record IDs")
        ids = set(table["record_id"].astype(str))
        if seen_ids & ids:
            raise ValueError("Inner validation tables must have disjoint record IDs")
        seen_ids.update(ids)
        if len(table) == 0:
            raise ValueError("Inner validation fold is empty")
        fold_values = pd.to_numeric(table["inner_fold"], errors="raise").unique()
        if len(fold_values) != 1:
            raise ValueError("Each component table must contain exactly one inner_fold value")
        seen_folds.append(int(fold_values[0]))
    if len(set(seen_folds)) != 4:
        raise ValueError("The four inner tables must represent four distinct folds")

    grids = build_weight_grids(selection_config)
    original_d = tuple(float(x) for x in selection_config["d_original"])
    original_db = tuple(float(x) for x in selection_config["db_original"])
    original_tri = tuple(float(x) for x in selection_config["tri_original"])
    tolerance = float(selection_config.get("tie_tolerance", 1e-12))

    d_candidates = _candidate_list(grids["D_weights"], "D")
    d_records = _score_records(
        "D_weights",
        d_candidates,
        inner_tables,
        lambda candidate, table: _compose_d(table, candidate.values),
        original=original_d,
    )
    d_choice = _choose_weight_record(d_records, tolerance)
    d_weights = tuple(float(x) for x in d_choice["weights"])

    db_candidates = _candidate_list(grids["D_B_weights"], "D_B")
    db_records = _score_records(
        "D_B_weights",
        db_candidates,
        inner_tables,
        lambda candidate, table: _compose_fusion(
            table, candidate.values, d_weights=d_weights, include_c=False
        ),
        original=original_db,
    )
    db_choice = _choose_weight_record(db_records, tolerance)
    db_weights = tuple(float(x) for x in db_choice["weights"])

    tri_candidates = _candidate_list(grids["TriCasRank_weights"], "TriCasRank")
    tri_records = _score_records(
        "TriCasRank_weights",
        tri_candidates,
        inner_tables,
        lambda candidate, table: _compose_fusion(
            table, candidate.values, d_weights=d_weights, include_c=True
        ),
        original=original_tri,
    )
    tri_choice = _choose_weight_record(tri_records, tolerance)
    tri_weights = tuple(float(x) for x in tri_choice["weights"])

    family_order = list(selection_config["family_order"])
    family_records = _family_records(
        inner_tables,
        d_weights=d_weights,
        db_weights=db_weights,
        tri_weights=tri_weights,
        family_order=family_order,
        learners=selection_config["learners"],
    )
    family_choice = _choose_family_record(family_records, tolerance)

    score_rows = []
    score_rows.extend(_record_rows(d_records, d_choice["candidate_id"]))
    score_rows.extend(_record_rows(db_records, db_choice["candidate_id"]))
    score_rows.extend(_record_rows(tri_records, tri_choice["candidate_id"]))
    score_rows.extend(_record_rows(family_records, family_choice["candidate_id"]))
    selection = {
        "d_weights": list(d_weights),
        "db_weights": list(db_weights),
        "tri_weights": list(tri_weights),
        "selected_family": family_choice["candidate_id"],
        "scoring_rule": "arithmetic_mean_of_four_inner_fold_SCC; require_all_four_defined; all_undefined_fallback_to_mean_MAE",
        "tie_tolerance": tolerance,
        "selected_scores": {
            "D_weights": {key: d_choice[key] for key in ("candidate_id", "mean_SCC", "mean_MAE", "ranking_metric", "ranking_value", "fold_metrics")},
            "D_B_weights": {key: db_choice[key] for key in ("candidate_id", "mean_SCC", "mean_MAE", "ranking_metric", "ranking_value", "fold_metrics")},
            "TriCasRank_weights": {key: tri_choice[key] for key in ("candidate_id", "mean_SCC", "mean_MAE", "ranking_metric", "ranking_value", "fold_metrics")},
            "model_family": {key: family_choice[key] for key in ("candidate_id", "mean_SCC", "mean_MAE", "ranking_metric", "ranking_value", "fold_metrics")},
        },
        "family_scores": [
            {key: record[key] for key in ("family", "mean_SCC", "mean_MAE", "ranking_metric", "ranking_value", "n_learners", "fold_metrics")}
            for record in family_records
        ],
        "selection_reason": {
            "d_weights": "selected from the authorized 0.05 simplex grid plus the original frozen D weights",
            "db_weights": "selected after D weights from the authorized 0.05 D/B simplex grid plus the original frozen D/B weights",
            "tri_weights": "selected after D weights from the authorized 0.05 D/B/C simplex grid plus the original TriCasRank and D/B boundary weights",
            "selected_family": "highest valid four-fold mean SCC; tie by lower mean MAE, fewer learners, then declared family order",
            "mapping_miss": "fusion routes use D_nested with selected D weights",
        },
        "inner_folds": seen_folds,
        "inner_validation_counts": {
            str(fold): int(len(table)) for fold, table in zip(seen_folds, inner_tables)
        },
    }
    return selection, pd.DataFrame(score_rows)


def predictions_for_outer(
    table: pd.DataFrame,
    selection: Mapping[str, Any],
    original_recipe: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compose final and frozen predictions from one outer component table."""
    d_weights = selection["d_weights"]
    db_weights = selection["db_weights"]
    tri_weights = selection["tri_weights"]
    d_nested = _compose_d(table, d_weights)
    d_fixed = _compose_d(table, [original_recipe["xgb"], original_recipe["lightgbm"], original_recipe["mlp"]])
    b = pd.to_numeric(table["B"], errors="raise").to_numpy(float)
    c = pd.to_numeric(table["C"], errors="raise").to_numpy(float)
    mapping_found = table["mapping_found"].astype(bool).to_numpy()
    routed_fusion = np.where(mapping_found, "context", "sequence_fallback")
    db = np.where(mapping_found, db_weights[0] * d_nested + db_weights[1] * b, d_nested)
    tri = np.where(mapping_found, tri_weights[0] * d_nested + tri_weights[1] * b + tri_weights[2] * c, d_nested)
    db_fixed = [float(original_recipe["fusion"]["D"]), float(original_recipe["fusion"]["B"])]
    tri_fixed = [float(original_recipe["fusion"][key]) for key in ("D", "B", "C")]
    db_fixed_pred = np.where(mapping_found, db_fixed[0] * d_fixed + db_fixed[1] * b, d_fixed)
    tri_fixed_pred = np.where(
        mapping_found,
        tri_fixed[0] * d_fixed + tri_fixed[1] * b + tri_fixed[2] * c,
        d_fixed,
    )
    families = {
        "D": d_nested,
        "SEQ_MAPPING_XGB": pd.to_numeric(table["SEQ_MAPPING_XGB"], errors="raise").to_numpy(float),
        "D_B": db,
        "TriCasRank": tri,
    }
    selected = families[str(selection["selected_family"])]
    selected_route = {
        "D": np.full(len(table), "sequence", dtype=object),
        "SEQ_MAPPING_XGB": np.full(len(table), "context", dtype=object),
        "D_B": routed_fusion.copy(),
        "TriCasRank": routed_fusion.copy(),
    }[str(selection["selected_family"])]
    return {
        "D_nested": (d_nested, np.full(len(table), "sequence", dtype=object)),
        "SEQ_MAPPING_XGB": (families["SEQ_MAPPING_XGB"], np.full(len(table), "context", dtype=object)),
        "D_B_nested": (db, routed_fusion.copy()),
        "TriCasRank_nested": (tri, routed_fusion.copy()),
        "HISTORY_PRIOR": (
            pd.to_numeric(table["HISTORY_PRIOR"], errors="raise").to_numpy(float),
            np.full(len(table), "history_prior", dtype=object),
        ),
        "SELECTED_POLICY": (selected, selected_route),
        "TriCasRank_fixed": (tri_fixed_pred, routed_fusion.copy()),
        "D_fixed": (d_fixed, np.full(len(table), "sequence", dtype=object)),
    }


__all__ = ["build_weight_grids", "predictions_for_outer", "select_policy", "simplex_grid"]
