"""Stable native-model inference for external Cas12a sequence pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_feature_frame


class Cas12aPredictor:
    """Load the frozen native models and predict diagnostic activity scores."""

    def __init__(self, repository_root: str | Path | None = None):
        self.root = Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[2]
        metadata_path = self.root / "models" / "model_input_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.features: list[str] = metadata["active_features"]
        self.xgb_weight = float(metadata["ensemble_weights"]["xgboost"])
        medians = pd.read_csv(self.root / "models" / "training_medians.csv")
        self.medians = medians.set_index("feature_name")["median"].reindex(self.features)

        from xgboost import Booster
        from catboost import CatBoostRegressor

        self.xgboost = Booster()
        self.xgboost.load_model(self.root / "models" / "primary" / "xgboost_final.json")
        self.catboost = CatBoostRegressor()
        self.catboost.load_model(self.root / "models" / "supporting" / "catboost_final.cbm")

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        features = build_feature_frame(pairs, manifest=self.features)
        return features.apply(pd.to_numeric, errors="coerce").fillna(self.medians).fillna(0)

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        matrix = self.transform(pairs)
        from xgboost import DMatrix

        pred_xgb = np.asarray(self.xgboost.predict(DMatrix(matrix, feature_names=self.features)), dtype=float)
        pred_cat = np.asarray(self.catboost.predict(matrix), dtype=float)
        pred_equal = 0.5 * pred_xgb + 0.5 * pred_cat
        pred_weighted = self.xgb_weight * pred_xgb + (1.0 - self.xgb_weight) * pred_cat
        output = pairs.copy()
        output["prediction_xgboost_primary"] = pred_xgb
        output["prediction_catboost_supporting"] = pred_cat
        output["prediction_equal_50_50_sensitivity"] = pred_equal
        output["prediction_oof_weighted_exploratory"] = pred_weighted
        output["rank_xgboost_descending"] = pd.Series(pred_xgb, index=output.index).rank(method="min", ascending=False).astype(int)
        return output
