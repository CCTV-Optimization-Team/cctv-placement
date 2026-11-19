
import numpy as np
import pytest

from visibility import Scorer


def test_uniform_weights():
    scorer = Scorer("tiny")

    assert np.all(scorer.weights[scorer.open_cells] == 1.0)
    assert np.all(scorer.weights[~scorer.open_cells] == 0.0)
    assert scorer.total == float(scorer.n_open_cells)


def test_external_priority_weights():
    weights = np.ones((5, 5))
    weights[4, 4] = 3.0

    scorer = Scorer("tiny", weights=weights)

    assert scorer.weights[4, 4] == 3.0
    assert scorer.total == 25.0


def test_wall_weights_are_ignored():
    weights = np.ones((5, 5))
    weights[1, 2] = 1000.0

    scorer = Scorer("tiny", weights=weights)

    assert scorer.weights[1, 2] == 0.0
    assert scorer.total == 23.0


def test_invalid_negative_weights():
    weights = np.ones((5, 5))
    weights[2, 2] = -1.0

    with pytest.raises(ValueError):
        Scorer("tiny", weights=weights)


def test_zero_weights():
    weights = np.zeros((5, 5))

    scorer = Scorer("tiny", weights=weights)

    assert scorer.total == 0.0
    assert scorer.score([]) == 0.0
