"""Stable inference for the v1.5 Cas12a sequence-model ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_feature_frame


PRIMARY_MODELS = {"d", "xgboost-legacy"}


class Cas12aPredictor:
    """Load frozen models and predict fluorescence-derived Cas12a activity."""

    def __init__(
        self,
        repository_root: str | Path | None = None,
        *,
        primary_model: str = "d",
    ):
        if primary_model not in PRIMARY_MODELS:
            raise ValueError(f"primary_model must be one of {sorted(PRIMARY_MODELS)}")
        self.root = (
            Path(repository_root).resolve()
            if repository_root
            else Path(__file__).resolve().parents[2]
        )
        self.primary_model = primary_model

        legacy_metadata = json.loads(
            (self.root / "models" / "model_input_metadata.json").read_text(encoding="utf-8")
        )
        d_metadata = json.loads(
            (self.root / "models" / "d_model_metadata.json").read_text(encoding="utf-8")
        )
        self.features: list[str] = legacy_metadata["active_features"]
        self.legacy_xgb_weight = float(legacy_metadata["ensemble_weights"]["xgboost"])
        self.d_weights = {key: float(value) for key, value in d_metadata["weights"].items()}
        self.model_version = str(d_metadata["version"])
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

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        features = build_feature_frame(pairs, manifest=self.features)
        return features.apply(pd.to_numeric, errors="coerce").fillna(self.medians).fillna(0)

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Append activity predictions while preserving every input row and column."""
        matrix = self.transform(pairs)
        from xgboost import DMatrix

        dmatrix = DMatrix(matrix, feature_names=self.features)
        pred_d_xgb = np.asarray(self.d_xgboost.predict(dmatrix), dtype=float)
        pred_d_lgbm = np.asarray(self.d_lightgbm.predict(matrix), dtype=float)
        pred_d_mlp = np.asarray(self.d_mlp.predict(matrix), dtype=float)
        pred_d = (
            self.d_weights["xgboost"] * pred_d_xgb
            + self.d_weights["lightgbm"] * pred_d_lgbm
            + self.d_weights["mlp"] * pred_d_mlp
        )

        pred_legacy_xgb = np.asarray(self.legacy_xgboost.predict(dmatrix), dtype=float)
        pred_legacy_cat = np.asarray(self.legacy_catboost.predict(matrix), dtype=float)
        pred_legacy_equal = 0.5 * pred_legacy_xgb + 0.5 * pred_legacy_cat
        pred_legacy_weighted = (
            self.legacy_xgb_weight * pred_legacy_xgb
            + (1.0 - self.legacy_xgb_weight) * pred_legacy_cat
        )

        if self.primary_model == "d":
            primary = pred_d
            route = "sequence_d_v1_5"
            reported_version = self.model_version
        else:
            primary = pred_legacy_xgb
            route = "sequence_xgboost_v1_0_legacy"
            reported_version = "1.0.0"

        output = pairs.copy()
        output["cas12a_activity_score"] = primary
        output["cas12a_activity_rank"] = (
            pd.Series(primary, index=output.index)
            .rank(method="min", ascending=False)
            .astype(int)
        )
        output["cas12a_prediction_status"] = "success"
        output["cas12a_model_route"] = route
        output["cas12a_model_version"] = reported_version
        output["cas12a_prediction_xgboost"] = pred_d_xgb
        output["cas12a_prediction_lightgbm"] = pred_d_lgbm
        output["cas12a_prediction_mlp"] = pred_d_mlp
        output["cas12a_warning_codes"] = ""

        # Deprecated v1.0 names are retained so existing users are not broken.
        output["prediction_xgboost_primary"] = pred_legacy_xgb
        output["prediction_catboost_supporting"] = pred_legacy_cat
        output["prediction_equal_50_50_sensitivity"] = pred_legacy_equal
        output["prediction_oof_weighted_exploratory"] = pred_legacy_weighted
        output["rank_xgboost_descending"] = (
            pd.Series(pred_legacy_xgb, index=output.index)
            .rank(method="min", ascending=False)
            .astype(int)
        )
        return output
