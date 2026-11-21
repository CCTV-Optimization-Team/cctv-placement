
import numpy as np
from visibility import Scorer


def test_gain_is_non_negative():
    scorer = Scorer("tiny")

    covered = np.zeros_like(scorer.open_cells, dtype=bool)
    gain = scorer.gain(covered, 0)

    assert gain >= 0


def test_gain_becomes_zero_when_already_covered():
    scorer = Scorer("tiny")

    covered = scorer.masks[0].copy()
    gain = scorer.gain(covered, 0)

    assert gain == 0


def test_covered_grid_has_correct_shape():
    scorer = Scorer("tiny")

    result = scorer.covered_grid([0, 1, 2])

    assert result.shape == (scorer.rows, scorer.cols)
    assert result.dtype == bool
