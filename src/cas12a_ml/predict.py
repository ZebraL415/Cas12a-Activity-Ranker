"""Stable inference for Cas12a Ranker v2.0 and its retained sequence-only routes."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_feature_frame
from .mapping import TemplateMapper
from .v2_features import V2FeatureBuilder


PRIMARY_MODELS = {"v2", "d", "xgboost-legacy"}


class Cas12aPredictor:
    """Load frozen models and predict fluorescence-derived Cas12a activity."""

    def __init__(
        self,
        repository_root: str | Path | None = None,
        *,
        primary_model: str = "v2",
        fallback_policy: str = "sequence",
        allow_mixed_ranking: bool = False,
    ):
        if primary_model not in PRIMARY_MODELS:
            raise ValueError(f"primary_model must be one of {sorted(PRIMARY_MODELS)}")
        if fallback_policy not in {"sequence", "error"}:
            raise ValueError("fallback_policy must be 'sequence' or 'error'")
        self.root = (
            Path(repository_root).resolve()
            if repository_root
            else Path(__file__).resolve().parents[2]
        )
        self.primary_model = primary_model
        self.fallback_policy = fallback_policy
        self.allow_mixed_ranking = allow_mixed_ranking

        legacy_metadata = json.loads(
            (self.root / "models" / "model_input_metadata.json").read_text(encoding="utf-8")
        )
        d_metadata = json.loads(
            (self.root / "models" / "d_model_metadata.json").read_text(encoding="utf-8")
        )
        self.features: list[str] = legacy_metadata["active_features"]
        self.legacy_xgb_weight = float(legacy_metadata["ensemble_weights"]["xgboost"])
        self.d_weights = {key: float(value) for key, value in d_metadata["weights"].items()}
        self.d_model_version = str(d_metadata["version"])
        medians = pd.read_csv(self.root / "models" / "training_medians.csv")
        self.medians = medians.set_index("feature_name")["median"].reindex(self.features)

        from catboost import CatBoostRegressor
        from lightgbm import Booster as LightGBMBooster
        from xgboost import Booster

        self.d_xgboost = Booster()
        self.d_xgboost.load_model(self.root / "models" / "primary" / "d_xgboost.json")
        self.d_lightgbm = LightGBMBooster(
            model_file=str(self.root / "models" / "primary" / "d_lightgbm.txt")
        )
        self.d_mlp = joblib.load(self.root / "models" / "primary" / "d_mlp_pipeline.joblib")

        # v1.0 artifacts remain loadable for backward-compatible output columns.
        self.legacy_xgboost = Booster()
        self.legacy_xgboost.load_model(
            self.root / "models" / "primary" / "xgboost_final.json"
        )
        self.legacy_catboost = CatBoostRegressor()
        self.legacy_catboost.load_model(
            self.root / "models" / "supporting" / "catboost_final.cbm"
        )

        self.v2_metadata: dict[str, object] | None = None
        self.mapper: TemplateMapper | None = None
        self.v2_features: V2FeatureBuilder | None = None
        self.b_models: list[Booster] = []
        self.c_model: Booster | None = None
        if primary_model == "v2":
            self.v2_metadata = json.loads(
                (self.root / "models" / "v2_model_metadata.json").read_text(encoding="utf-8")
            )
            mapping_root = self.root / "models" / "mapping"
            self.mapper = TemplateMapper(mapping_root / "table_s2_template_reference.csv")
            self.v2_features = V2FeatureBuilder(self.root)
            for seed in self.v2_metadata["module_b_seeds"]:  # type: ignore[index]
                model = Booster()
                model.load_model(mapping_root / f"b_seed_{seed}.json")
                self.b_models.append(model)
            self.c_model = Booster()
            self.c_model.load_model(mapping_root / "c_guide_template_anchor.json")

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        features = build_feature_frame(pairs, manifest=self.features)
        return features.apply(pd.to_numeric, errors="coerce").fillna(self.medians).fillna(0)

    def _sequence_predictions(self, pairs: pd.DataFrame) -> dict[str, np.ndarray]:
        matrix = self.transform(pairs)
        from xgboost import DMatrix

        dmatrix = DMatrix(matrix, feature_names=self.features)
        d_xgb = np.asarray(self.d_xgboost.predict(dmatrix), dtype=float)
        d_lgbm = np.asarray(self.d_lightgbm.predict(matrix), dtype=float)
        d_mlp = np.asarray(self.d_mlp.predict(matrix), dtype=float)
        d = (
            self.d_weights["xgboost"] * d_xgb
            + self.d_weights["lightgbm"] * d_lgbm
            + self.d_weights["mlp"] * d_mlp
        )
        legacy_xgb = np.asarray(self.legacy_xgboost.predict(dmatrix), dtype=float)
        legacy_cat = np.asarray(self.legacy_catboost.predict(matrix), dtype=float)
        return {
            "d_xgb": d_xgb,
            "d_lgbm": d_lgbm,
            "d_mlp": d_mlp,
            "d": d,
            "legacy_xgb": legacy_xgb,
            "legacy_cat": legacy_cat,
        }

    @staticmethod
    def _rank(values: pd.Series) -> pd.Series:
        return values.rank(method="min", ascending=False).astype("Int64")

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Append activity predictions while preserving every input row and column."""
        sequence = self._sequence_predictions(pairs)
        output = pairs.copy()
        output["cas12a_prediction_xgboost"] = sequence["d_xgb"]
        output["cas12a_prediction_lightgbm"] = sequence["d_lgbm"]
        output["cas12a_prediction_mlp"] = sequence["d_mlp"]
        output["cas12a_prediction_d"] = sequence["d"]
        output["cas12a_prediction_b"] = np.nan
        output["cas12a_prediction_c"] = np.nan
        output["cas12a_prediction_full_dbc"] = np.nan
        output["cas12a_prediction_status"] = "success"
        output["cas12a_warning_codes"] = ""
        output["cas12a_mapping_source"] = "not_requested"
        for column in (
            "status",
            "confidence",
            "candidate_template_nos",
            "candidate_group_ids",
            "candidate_orientations",
            "match_modes",
        ):
            output[f"cas12a_mapping_{column}"] = ""
        for column in ("hit_count", "template_count", "group_count", "orientation_count"):
            output[f"cas12a_mapping_{column}"] = pd.array([pd.NA] * len(output), dtype="Int64")
        output["cas12a_guide_seen"] = pd.array([pd.NA] * len(output), dtype="boolean")
        output["cas12a_template_key_seen"] = pd.array([pd.NA] * len(output), dtype="boolean")
        output["cas12a_guide_reference_count"] = pd.array([pd.NA] * len(output), dtype="Int64")
        output["cas12a_template_reference_count"] = pd.array([pd.NA] * len(output), dtype="Int64")
        output["cas12a_reference_support"] = "not_requested"

        if self.primary_model == "v2":
            assert self.mapper is not None and self.v2_features is not None
            assert self.c_model is not None and self.v2_metadata is not None
            mapping = self.mapper.map_frame(pairs)
            for column in mapping.columns:
                output_column = f"cas12a_{column}"
                output[output_column] = mapping[column].to_numpy()
            output["cas12a_mapping_source"] = "automatic_frozen_easydesign_table_s2"
            mapped = mapping["mapping_hit_count"].astype(int).gt(0)
            if not mapped.all() and self.fallback_policy == "error":
                record_ids = output.loc[~mapped, "record_id"].astype(str).tolist()
                raise ValueError(
                    "Automatic mapping failed for record_id values: " + ", ".join(record_ids[:10])
                )
            if mapped.any():
                mapped_pairs = pairs.loc[mapped].copy()
                mapped_mapping = mapping.loc[mapped].copy()
                matrix, diagnostics = self.v2_features.build(mapped_pairs, mapped_mapping)
                from xgboost import DMatrix

                dmatrix = DMatrix(matrix, feature_names=list(matrix.columns))
                guide_anchor = diagnostics["guide_anchor"].to_numpy(float)
                b_predictions = [
                    guide_anchor + model.predict(dmatrix) for model in self.b_models
                ]
                prediction_b = np.mean(np.vstack(b_predictions), axis=0)
                mixed_anchor = 0.5 * guide_anchor + 0.5 * diagnostics["template_anchor"].to_numpy(float)
                prediction_c = mixed_anchor + self.c_model.predict(dmatrix)
                weights = self.v2_metadata["weights"]  # type: ignore[index]
                prediction_full = (
                    float(weights["d"]) * sequence["d"][mapped.to_numpy()]
                    + float(weights["b"]) * prediction_b
                    + float(weights["c"]) * prediction_c
                )
                output.loc[mapped, "cas12a_prediction_b"] = prediction_b
                output.loc[mapped, "cas12a_prediction_c"] = prediction_c
                output.loc[mapped, "cas12a_prediction_full_dbc"] = prediction_full
                output.loc[mapped, "cas12a_guide_seen"] = diagnostics["guide_seen"].to_numpy()
                output.loc[mapped, "cas12a_template_key_seen"] = diagnostics["template_key_seen"].to_numpy()
                output.loc[mapped, "cas12a_guide_reference_count"] = diagnostics[
                    "guide_reference_count"
                ].round().astype(int).to_numpy()
                output.loc[mapped, "cas12a_template_reference_count"] = diagnostics[
                    "template_reference_count"
                ].round().astype(int).to_numpy()
                for index in output.index[mapped]:
                    guide_count = int(output.at[index, "cas12a_guide_reference_count"])
                    template_count = int(output.at[index, "cas12a_template_reference_count"])
                    output.at[index, "cas12a_reference_support"] = (
                        f"guide_count={guide_count};template_key_count={template_count}"
                    )
                    warnings: list[str] = []
                    if not bool(output.at[index, "cas12a_guide_seen"]):
                        warnings.append("W_GUIDE_NOT_IN_TRAIN_REFERENCE")
                    if not bool(output.at[index, "cas12a_template_key_seen"]):
                        warnings.append("W_TEMPLATE_KEY_NOT_IN_TRAIN_REFERENCE")
                    if int(output.at[index, "cas12a_mapping_template_count"]) > 1:
                        warnings.append("W_MAPPING_MULTIPLE_CANDIDATES")
                    if int(output.at[index, "cas12a_mapping_group_count"]) > 1:
                        warnings.append("W_MAPPING_MULTIPLE_GROUPS")
                    if str(output.at[index, "cas12a_mapping_confidence"]) == "review":
                        warnings.append("W_MAPPING_REVIEW")
                    output.at[index, "cas12a_warning_codes"] = ";".join(warnings)
            output["cas12a_activity_score"] = output["cas12a_prediction_full_dbc"]
            output.loc[~mapped, "cas12a_activity_score"] = sequence["d"][~mapped.to_numpy()]
            output["cas12a_model_route"] = "mapping_dbc_v2"
            output.loc[~mapped, "cas12a_model_route"] = "sequence_d_fallback_v2"
            output.loc[~mapped, "cas12a_prediction_status"] = "fallback_success"
            output.loc[~mapped, "cas12a_reference_support"] = "mapping_not_found"
            output.loc[~mapped, "cas12a_warning_codes"] = "W_MAPPING_NOT_FOUND;W_SEQUENCE_D_FALLBACK"
            output["cas12a_model_version"] = str(self.v2_metadata["version"])
        elif self.primary_model == "d":
            output["cas12a_activity_score"] = sequence["d"]
            output["cas12a_model_route"] = "sequence_d_v1_5"
            output["cas12a_model_version"] = self.d_model_version
        else:
            output["cas12a_activity_score"] = sequence["legacy_xgb"]
            output["cas12a_model_route"] = "sequence_xgboost_v1_0_legacy"
            output["cas12a_model_version"] = "1.0.0"

        output["cas12a_rank_within_route"] = pd.array([pd.NA] * len(output), dtype="Int64")
        for _, index in output.groupby("cas12a_model_route", sort=False).groups.items():
            output.loc[index, "cas12a_rank_within_route"] = self._rank(
                output.loc[index, "cas12a_activity_score"].astype(float)
            ).to_numpy()
        mixed_routes = output["cas12a_model_route"].nunique() > 1
        if mixed_routes and not self.allow_mixed_ranking:
            output["cas12a_activity_rank"] = pd.array([pd.NA] * len(output), dtype="Int64")
            output["cas12a_warning_codes"] = output["cas12a_warning_codes"].map(
                lambda value: ";".join(filter(None, [str(value), "W_GLOBAL_RANK_WITHHELD_MIXED_ROUTES"]))
            )
        else:
            output["cas12a_activity_rank"] = self._rank(output["cas12a_activity_score"].astype(float))

        legacy_equal = 0.5 * sequence["legacy_xgb"] + 0.5 * sequence["legacy_cat"]
        legacy_weighted = (
            self.legacy_xgb_weight * sequence["legacy_xgb"]
            + (1.0 - self.legacy_xgb_weight) * sequence["legacy_cat"]
        )
        output["prediction_xgboost_primary"] = sequence["legacy_xgb"]
        output["prediction_catboost_supporting"] = sequence["legacy_cat"]
        output["prediction_equal_50_50_sensitivity"] = legacy_equal
        output["prediction_oof_weighted_exploratory"] = legacy_weighted
        output["rank_xgboost_descending"] = self._rank(
            pd.Series(sequence["legacy_xgb"], index=output.index)
        )
        return output
