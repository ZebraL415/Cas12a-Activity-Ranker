"""Root-owned validation and summary analysis for the nested-selection OOF table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
PLAN = json.loads((HERE / "config" / "selection.json").read_text(encoding="utf-8"))
PROTOCOLS = PLAN["protocols"]
COHORTS = PLAN["cohorts"]
CORE_PRIMARY = ("D_nested", "SEQ_MAPPING_XGB", "D_B_nested")
COMMON_PRIMARY = ("EasyDesign-adapted",)
METRICS = ("SCC", "PCC", "RMSE", "MAE", "R2")


def require_columns(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_expected(execution_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    data = execution_root / "data"
    labels = pd.read_csv(data / "development_labels.csv", dtype={"record_id": str})
    assignments = pd.read_csv(data / "group_assignments.csv", dtype={"record_id": str})
    common = pd.read_csv(data / "common_record_ids.csv", dtype={"record_id": str})
    require_columns(labels, {"record_id", "y_true"}, "development labels")
    require_columns(assignments, {"record_id"}, "group assignments")
    require_columns(common, {"record_id"}, "common cohort")
    for name, frame in (("labels", labels), ("assignments", assignments)):
        if frame.record_id.isna().any() or frame.record_id.duplicated().any():
            raise ValueError(f"{name} must have unique non-null record_id")
    return labels, assignments, set(common.record_id.astype(str))


def validate_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    common_ids: set[str],
    cohorts: tuple[str, ...] = tuple(COHORTS),
) -> pd.DataFrame:
    required = {
        "cohort_id", "protocol_id", "outer_fold", "group_id", "record_id",
        "y_true", "y_pred", "model_id", "evaluation_kind", "selection_id",
    }
    require_columns(predictions, required, "nested predictions")
    p = predictions.copy()
    p["record_id"] = p.record_id.astype(str)
    p["group_id"] = p.group_id.astype(str)
    if p.duplicated(["cohort_id", "protocol_id", "outer_fold", "record_id", "model_id", "evaluation_kind", "selection_id"]).any():
        raise ValueError("nested predictions contain duplicate model/record/fold rows")
    if set(p.cohort_id) != set(cohorts) or set(p.protocol_id) != set(PROTOCOLS):
        raise ValueError(f"nested prediction cohort/protocol coverage is incomplete for {cohorts}")

    label_map = labels.set_index("record_id").y_true.astype(float)
    assignment_map = assignments.set_index("record_id")
    checks = []
    for cohort in cohorts:
        expected_total = COHORTS[cohort]
        cohort_ids = set(assignments.record_id.astype(str))
        if cohort == "common_dev":
            cohort_ids &= common_ids
        if len(cohort_ids) != expected_total:
            raise ValueError(f"frozen {cohort} membership expected {expected_total}, got {len(cohort_ids)}")
        for protocol in PROTOCOLS:
            group_col, fold_col = f"{protocol}_group", f"{protocol}_fold"
            require_columns(assignments, {group_col, fold_col}, f"assignments/{protocol}")
            sub = p[(p.cohort_id == cohort) & (p.protocol_id == protocol)]
            for model in sub.model_id.unique():
                m = sub[sub.model_id == model]
                observed = set(m.record_id)
                if observed != cohort_ids:
                    raise ValueError(f"{cohort}/{protocol}/{model}: expected {len(cohort_ids)} IDs, got {len(observed)}")
                if set(m.outer_fold.astype(int)) != set(range(5)):
                    raise ValueError(f"{cohort}/{protocol}/{model}: incomplete outer folds")
                expected = m.record_id.map(label_map).to_numpy(float)
                if not np.allclose(expected, m.y_true.to_numpy(float), rtol=0, atol=1e-12):
                    raise ValueError(f"{cohort}/{protocol}/{model}: y_true differs from frozen labels")
                assigned_group = m.record_id.map(assignment_map[group_col]).astype(str).to_numpy()
                assigned_fold = m.record_id.map(assignment_map[fold_col]).to_numpy(int)
                if not np.array_equal(assigned_group, m.group_id.to_numpy(str)):
                    raise ValueError(f"{cohort}/{protocol}/{model}: group IDs differ from assignment")
                if not np.array_equal(assigned_fold, m.outer_fold.to_numpy(int)):
                    raise ValueError(f"{cohort}/{protocol}/{model}: outer fold differs from assignment")
                checks.append({"cohort_id": cohort, "protocol_id": protocol, "model_id": model, "n": len(m), "groups": int(pd.Series(assigned_group).nunique())})

            models = set(sub.model_id)
            must = {"D_nested", "SEQ_MAPPING_XGB", "D_B_nested", "TriCasRank_nested", "HISTORY_PRIOR", "SELECTED_POLICY", "TriCasRank_fixed", "D_fixed"}
            if cohort == "common_dev":
                must.add("EasyDesign-adapted")
            missing_models = must - models
            if missing_models:
                raise ValueError(f"{cohort}/{protocol} missing required models: {sorted(missing_models)}")

    return pd.DataFrame(checks)


def score_and_compare(predictions: pd.DataFrame, output_dir: Path, cohorts: tuple[str, ...] = tuple(COHORTS)) -> dict:
    model_source = HERE / "engines" / "fixed" / "src"
    if not model_source.is_dir():
        raise FileNotFoundError(f"portable statistics source not available: {model_source}")
    sys.path.insert(0, str(model_source))
    from revision_statistics.metrics import basic_metrics
    from revision_statistics.resampling import (
        bootstrap_metric_draws,
        holm_adjust,
        percentile_interval,
        prediction_swap_test,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, fold_metric_rows, comparison_rows, secondary_rows = [], [], [], []
    for ci, cohort in enumerate(cohorts):
        for pi, protocol in enumerate(PROTOCOLS):
            block = predictions[(predictions.cohort_id == cohort) & (predictions.protocol_id == protocol)]
            wide = block.pivot(index="record_id", columns="model_id", values="y_pred")
            meta = block.drop_duplicates("record_id").set_index("record_id").loc[wide.index]
            y, groups = meta.y_true.to_numpy(float), meta.group_id.astype(str).to_numpy()
            n_groups = int(pd.Series(groups).nunique())
            n_records = int(len(y))
            estimate = {name: basic_metrics(y, wide[name].to_numpy(float)) for name in wide.columns}
            seed = int(PLAN["statistics"]["seed"]) + ci * 10 + pi
            draws = bootstrap_metric_draws(y, {name: wide[name].to_numpy(float) for name in wide.columns}, groups, draws=PLAN["statistics"]["draws"], seed=seed)
            for model, by_metric in estimate.items():
                for fold in range(5):
                    fold_frame = block[(block.model_id == model) & (block.outer_fold.astype(int) == fold)]
                    if fold_frame.empty:
                        raise ValueError(f"Missing outer fold {fold} for {cohort}/{protocol}/{model}")
                    fold_score = basic_metrics(fold_frame.y_true.to_numpy(float), fold_frame.y_pred.to_numpy(float))
                    for metric in METRICS:
                        fold_metric_rows.append({
                            "cohort_id": cohort,
                            "protocol_id": protocol,
                            "outer_fold": fold,
                            "model_id": model,
                            "n": int(len(fold_frame)),
                            "metric": metric,
                            "estimate": fold_score[metric],
                        })
                for metric in METRICS:
                    interval = percentile_interval(draws[model][metric])
                    metric_rows.append({
                        "cohort_id": cohort,
                        "protocol_id": protocol,
                        "model_id": model,
                        "n_records": n_records,
                        "n_groups": n_groups,
                        "metric": metric,
                        "estimate": by_metric[metric],
                        **interval,
                    })

            comparators = CORE_PRIMARY if cohort == "core_dev" else COMMON_PRIMARY
            primary_model = PLAN["statistics"]["primary_model"]
            for comparator in comparators:
                delta = {metric: draws[primary_model][metric] - draws[comparator][metric] for metric in METRICS}
                ci_by_metric = {metric: percentile_interval(delta[metric]) for metric in METRICS}
                test = prediction_swap_test(y, wide[primary_model].to_numpy(float), wide[comparator].to_numpy(float), groups, draws=PLAN["statistics"]["draws"], seed=seed + 1000)
                primary_delta = estimate[primary_model]["SCC"] - estimate[comparator]["SCC"]
                comparison_rows.append({
                    "cohort_id": cohort, "protocol_id": protocol, "model_a": primary_model, "model_b": comparator,
                    "delta_scc": primary_delta,
                    "permutation_p": test["p_value"], "permutation_status": test["status"],
                    "draws": PLAN["statistics"]["draws"], "clusters": n_groups,
                    **{
                        name: None if estimate[primary_model][metric] is None or estimate[comparator][metric] is None
                        else estimate[primary_model][metric] - estimate[comparator][metric]
                        for metric in METRICS
                        for name in (f"delta_{metric.lower()}",)
                    },
                    **{
                        name: ci_by_metric[metric][bound]
                        for metric in METRICS
                        for bound, suffix in (("lower", "ci_lower"), ("upper", "ci_upper"))
                        for name in (f"delta_{metric.lower()}_{suffix}",)
                    },
                })

            secondary_pairs = [
                ("TriCasRank_nested", model) for model in CORE_PRIMARY
            ] if cohort == "core_dev" else [("TriCasRank_nested", "EasyDesign-adapted")]
            secondary_pairs.extend((("TriCasRank_nested", "TriCasRank_fixed"), ("SELECTED_POLICY", "TriCasRank_fixed")))
            for model_a, model_b in secondary_pairs:
                intervals = {
                    metric: percentile_interval(draws[model_a][metric] - draws[model_b][metric])
                    for metric in METRICS
                }
                secondary_rows.append({
                    "cohort_id": cohort,
                    "protocol_id": protocol,
                    "model_a": model_a,
                    "model_b": model_b,
                    **{
                        f"delta_{metric.lower()}": None if estimate[model_a][metric] is None or estimate[model_b][metric] is None
                        else estimate[model_a][metric] - estimate[model_b][metric]
                        for metric in METRICS
                    },
                    **{
                        f"delta_{metric.lower()}_{suffix}": intervals[metric][bound]
                        for metric in METRICS
                        for bound, suffix in (("lower", "ci_lower"), ("upper", "ci_upper"))
                    },
                    "draws": PLAN["statistics"]["draws"],
                    "clusters": n_groups,
                    "p_value": "descriptive_only",
                })

    metrics_frame = pd.DataFrame(metric_rows)
    fold_metrics_frame = pd.DataFrame(fold_metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    secondary = pd.DataFrame(secondary_rows)
    family_size_by_cohort = (("core_dev", PLAN["statistics"]["core_family_size"]), ("common_dev", PLAN["statistics"]["common_family_size"]))
    for cohort, family_size in family_size_by_cohort:
        if cohort not in cohorts:
            continue
        ix = comparisons.index[comparisons.cohort_id.eq(cohort)]
        comparisons.loc[ix, "holm_p"] = holm_adjust(comparisons.loc[ix, "permutation_p"].tolist(), family_size=family_size)
    metrics_frame.to_csv(output_dir / "nested_metrics_and_intervals.csv", index=False)
    fold_metrics_frame.to_csv(output_dir / "nested_fold_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "nested_primary_comparisons.csv", index=False)
    secondary.to_csv(output_dir / "nested_secondary_effects.csv", index=False)
    summary = {
        "status": "complete",
        "selection_ids": sorted(map(str, predictions.selection_id.dropna().unique())),
        "evaluation_kind": str(predictions.evaluation_kind.iloc[0]),
        "primary_model": PLAN["statistics"]["primary_model"],
        "n_models": int(predictions.model_id.nunique()),
        "cohorts": {cohort: int(predictions.loc[predictions.cohort_id.eq(cohort), "record_id"].nunique()) for cohort in cohorts},
        "protocols": list(PROTOCOLS),
        "comparison_rows": len(comparisons),
        "fold_metric_rows": len(fold_metrics_frame),
        "secondary_effect_rows": len(secondary),
        "core_holm_family": PLAN["statistics"]["core_family_size"],
        "common_holm_family": PLAN["statistics"]["common_family_size"],
        "bootstrap_draws": PLAN["statistics"]["draws"],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "integration" / "results")
    parser.add_argument("--cohorts", nargs="+", choices=tuple(COHORTS), default=list(COHORTS),
                        help="Validation scope. Use the default for the complete final analysis; a single cohort supports checkpoint review.")
    args = parser.parse_args()
    labels, assignments, common_ids = load_expected(args.execution_root.resolve())
    predictions = pd.read_csv(args.predictions, dtype={"record_id": str, "group_id": str})
    cohorts = tuple(args.cohorts)
    predictions = predictions[predictions.cohort_id.isin(cohorts)].copy()
    coverage = validate_predictions(predictions, labels, assignments, common_ids, cohorts)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output / "prediction_coverage.csv", index=False)
    print(json.dumps(score_and_compare(predictions, output, cohorts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
