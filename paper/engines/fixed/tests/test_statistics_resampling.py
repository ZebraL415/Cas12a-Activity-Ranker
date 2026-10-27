import numpy as np
import pytest
from scipy.stats import rankdata

from revision_statistics import (
    bootstrap_metric_draws,
    bootstrap_scc_draws,
    holm_adjust,
    percentile_interval,
    prediction_swap_test,
)
from revision_statistics.resampling import _weighted_average_ranks


def test_weighted_ranks_match_an_explicitly_expanded_sample():
    values = np.array([1, 1, 2, 4], dtype=float)
    weights = np.array([[1, 2, 1, 0]], dtype=float)
    got = _weighted_average_ranks(values, weights)[0]
    expanded = np.repeat(values, weights[0].astype(int))
    expected = rankdata(expanded, method="average")
    assert got[:3] == pytest.approx(expected[[0, 0, 3]])


def test_bootstrap_uses_shared_protocol_group_draws_and_is_reproducible():
    truth = np.array([0, 1, 2, 3], dtype=float)
    predictions = {"perfect": truth.copy(), "reverse": -truth}
    groups = ["A", "A", "B", "B"]
    first = bootstrap_scc_draws(truth, predictions, groups, draws=40, seed=7, chunk_size=9)
    second = bootstrap_scc_draws(truth, predictions, groups, draws=40, seed=7, chunk_size=9)
    assert np.allclose(first["perfect"], 1.0)
    assert np.allclose(first["reverse"], -1.0)
    assert np.allclose(first["perfect"], second["perfect"], equal_nan=True)
    assert np.allclose(first["reverse"], second["reverse"], equal_nan=True)


def test_bootstrap_metric_draws_include_descriptive_errors_without_extra_tests():
    truth = np.array([0, 1, 2, 3], dtype=float)
    result = bootstrap_metric_draws(
        truth,
        {"perfect": truth.copy()},
        ["A", "A", "B", "B"],
        draws=12,
        seed=9,
    )["perfect"]
    assert np.allclose(result["SCC"], 1.0)
    assert np.allclose(result["PCC"], 1.0)
    assert np.allclose(result["RMSE"], 0.0)
    assert np.allclose(result["MAE"], 0.0)
    assert np.allclose(result["R2"], 1.0)


def test_percentile_interval_requires_minimum_valid_fraction():
    result = percentile_interval([0.1, 0.2, 0.3, np.nan], minimum_valid_fraction=0.95)
    assert result["status"] == "insufficient_valid_draws"
    assert result["valid_draws"] == 3
    assert result["lower"] is None and result["upper"] is None

    valid = percentile_interval([0.1, 0.2, 0.3, 0.4], minimum_valid_fraction=0.95)
    assert valid["status"] == "available"
    assert valid["lower"] == pytest.approx(0.1075)
    assert valid["upper"] == pytest.approx(0.3925)


def test_prediction_swap_is_group_level_and_uses_add_one_correction():
    truth = [0, 1, 2, 3]
    groups = ["A", "A", "B", "B"]
    result = prediction_swap_test(truth, [0, 1, 2, 3], [0, 1, 2, 3], groups, draws=32, seed=11)
    assert result["p_value"] == 1.0
    assert result["valid_draws"] == 32

    different = prediction_swap_test(truth, [0, 1, 2, 3], [3, 2, 1, 0], groups, draws=32, seed=11)
    assert different["status"] == "available"
    assert different["p_value"] >= 1 / 33
    assert different["p_value"] <= 1


def test_holm_keeps_the_prespecified_family_size_when_a_test_is_missing():
    assert holm_adjust([0.01, None, 0.03], family_size=4) == [0.04, None, 0.09]
