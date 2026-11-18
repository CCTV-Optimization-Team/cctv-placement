
import numpy as np
from visibility import Scorer


def test_uniform_weights():
    scorer = Scorer("tiny")

    assert np.all(scorer.weights[scorer.open_cells] == 1.0)
    assert np.all(scorer.weights[~scorer.open_cells] == 0.0)



def test_external_priority_weights():
    weights = np.ones((5, 5))
    weights[4, 4] = 3.0

    scorer = Scorer("tiny", weights=weights)

    assert scorer.weights[24] == 3.0
    assert scorer.total == 25.0


def test_wall_weights_are_ignored():
    weights = np.ones((5, 5))
    weights[1, 2] = 1000.0

    scorer = Scorer("tiny", weights=weights)

    assert scorer.weights[7] == 0.0
    assert scorer.total == 23.0

def test_invalid_negative_weights():
    weights = np.ones((5, 5))
    weights[0, 0] = -1

    try:
        Scorer("tiny", weights=weights)
        assert False, "Negative weights should raise an error"
    except AssertionError:
        pass


def test_zero_weights():
    weights = np.zeros((5, 5))

    scorer = Scorer("tiny", weights=weights)

    assert scorer.total == 0.0
    assert scorer.score([]) == 0.0
