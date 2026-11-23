import numpy as np
import pytest

from visibility import Scorer, build_masks, save_masks


@pytest.fixture
def tiny_map(tmp_path):
    grid = np.zeros((5, 5), dtype=np.uint8)

    # Two walls, giving 23 open cells
    grid[1, 2] = 1
    grid[0, 0] = 1

    gmap = {"name": "test_tiny", "grid": grid}

    candidates, masks = build_masks(
        grid, cam_range=4, fov_deg=90
    )

    save_masks(
        tmp_path / "test_tiny.npz",
        grid,
        candidates,
        masks,
        4, 90, 8
    )

    return gmap, tmp_path


def test_uniform_weights(tiny_map):
    gmap, masks_dir = tiny_map
    scorer = Scorer(gmap, masks_dir=masks_dir)

    assert np.all(scorer.weights[scorer.open_cells] == 1.0)
    assert np.all(scorer.weights[~scorer.open_cells] == 0.0)
    assert scorer.total == float(scorer.n_open_cells)


def test_external_priority_weights(tiny_map):
    gmap, masks_dir = tiny_map

    weights = np.ones((5, 5))
    weights[4, 4] = 3.0

    scorer = Scorer(gmap, weights=weights, masks_dir=masks_dir)

    assert scorer.weights[4, 4] == 3.0
    assert scorer.total == 25.0


def test_wall_weights_are_ignored(tiny_map):
    gmap, masks_dir = tiny_map

    weights = np.ones((5, 5))
    weights[1, 2] = 1000.0

    scorer = Scorer(gmap, weights=weights, masks_dir=masks_dir)

    assert scorer.weights[1, 2] == 0.0
    assert scorer.total == 23.0


def test_invalid_negative_weights(tiny_map):
    gmap, masks_dir = tiny_map

    weights = np.ones((5, 5))
    weights[2, 2] = -1.0

    with pytest.raises(ValueError):
        Scorer(gmap, weights=weights, masks_dir=masks_dir)


def test_zero_weights(tiny_map):
    gmap, masks_dir = tiny_map

    weights = np.zeros((5, 5))

    scorer = Scorer(gmap, weights=weights, masks_dir=masks_dir)

    assert scorer.total == 0.0
    assert scorer.score([]) == 0.0