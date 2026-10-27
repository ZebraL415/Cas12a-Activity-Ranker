from __future__ import annotations

import numpy as np

from models import B_SEEDS, _compose_predictions, resolved_recipe


def test_recipe_captures_frozen_weights_seeds_and_resolved_estimators() -> None:
    recipe = resolved_recipe()
    assert recipe["weights"]["D_components"] == {"xgb": 0.35, "lightgbm": 0.59, "mlp": 0.06}
    assert recipe["weights"]["TriCasRank"] == {"D": 0.20, "B": 0.47, "C": 0.33}
    assert recipe["seeds"]["B"] == list(B_SEEDS)
    assert recipe["seeds"]["C"] == 42

    d_xgb = recipe["components"]["D"]["xgb"]["resolved_get_params_deep_true"]
    assert d_xgb["n_estimators"] == 800
    assert d_xgb["learning_rate"] == 0.03
    assert d_xgb["max_depth"] == 6
    assert d_xgb["random_state"] == 42
    d_lgbm = recipe["components"]["D"]["lightgbm"]["resolved_get_params_deep_true"]
    assert d_lgbm["deterministic"] is True
    assert d_lgbm["force_col_wise"] is True
    b42 = recipe["components"]["B"]["42"]["resolved_get_params_deep_true"]
    assert b42["n_estimators"] == 1100
    assert b42["max_depth"] == 7
    assert b42["min_child_weight"] == 8
    assert b42["random_state"] == 42
    c = recipe["components"]["C"]["resolved_get_params_deep_true"]
    assert c["min_child_weight"] == 2
    assert c["random_state"] == 42
    mapping = recipe["components"]["SEQ_MAPPING_XGB"]["resolved_get_params_deep_true"]
    assert mapping["min_child_weight"] == 8
    assert mapping["random_state"] == 42


def test_fixed_routes_fall_back_only_for_formal_routed_fusions() -> None:
    d = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    c = np.array([5.0, 6.0])
    prior = np.array([7.0, 8.0])
    predictions, routes = _compose_predictions(
        d=d,
        b=b,
        c=c,
        history_prior=prior,
        mapping_found=np.array([True, False]),
        mapping_prediction=np.array([9.0, 10.0]),
    )
    assert np.isclose(predictions["TriCasRank"][0], 0.20 * 1 + 0.47 * 3 + 0.33 * 5)
    assert predictions["TriCasRank"][1] == d[1]
    assert np.isclose(predictions["D_B_FIXED"][0], (0.20 * 1 + 0.47 * 3) / 0.67)
    assert predictions["D_B_FIXED"][1] == d[1]
    assert predictions["TriCasRank_UNROUTED"][1] != d[1]
    assert routes["TriCasRank"].tolist() == ["context", "sequence_fallback"]
    assert routes["D"].tolist() == ["sequence", "sequence"]
    assert routes["HISTORY_PRIOR"].tolist() == ["history_prior", "history_prior"]
    assert routes["SEQ_MAPPING_XGB"].tolist() == ["context", "context"]
    assert routes["B"].tolist() == ["context", "context"]
