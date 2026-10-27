import math

import numpy as np
import pytest

from revision_statistics import basic_metrics


def test_basic_metrics_uses_average_ranks_for_ties():
    result = basic_metrics([1, 1, 2, 3], [1, 2, 2, 4])
    expected = np.corrcoef([1.5, 1.5, 3.0, 4.0], [1.0, 2.5, 2.5, 4.0])[0, 1]
    assert result["SCC"] == pytest.approx(expected)
    assert result["PCC"] == pytest.approx(np.corrcoef([1, 1, 2, 3], [1, 2, 2, 4])[0, 1])
    assert result["RMSE"] == pytest.approx(math.sqrt(0.5))
    assert result["MAE"] == pytest.approx(0.5)
    assert result["R2"] == pytest.approx(1.0 - 2.0 / 2.75)


def test_constant_vectors_return_missing_correlations():
    pred_constant = basic_metrics([1, 2, 3], [2, 2, 2])
    assert pred_constant["SCC"] is None
    assert pred_constant["PCC"] is None
    assert pred_constant["R2"] == pytest.approx(0.0)

    true_constant = basic_metrics([2, 2, 2], [1, 2, 3])
    assert true_constant["SCC"] is None
    assert true_constant["PCC"] is None
    assert true_constant["R2"] is None


def test_single_observation_has_errors_but_no_correlation_or_r2():
    result = basic_metrics([2], [3])
    assert result == {"SCC": None, "PCC": None, "RMSE": 1.0, "MAE": 1.0, "R2": None}


@pytest.mark.parametrize(
    "truth,prediction",
    [([1, 2], [1]), ([1, float("nan")], [1, 2]), ([1, 2], [1, float("inf")]), ([], [])],
)
def test_invalid_input_is_not_silently_dropped(truth, prediction):
    with pytest.raises(ValueError):
        basic_metrics(truth, prediction)
