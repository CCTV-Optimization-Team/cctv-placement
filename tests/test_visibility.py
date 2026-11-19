
import numpy as np
from visibility import (
    build_masks,
    has_line_of_sight,
    Scorer,
    save_masks,
)


def get_candidate_index(candidates, r, c, direction):
    matches = np.where(
        np.all(candidates == [r, c, direction], axis=1)
    )[0]
    return int(matches[0])


def test_open_room_camera_sees_its_wedge():
    grid = np.zeros((9, 9), dtype=np.uint8)

    candidates, masks = build_masks(
        grid, cam_range=4, fov_deg=90
    )

    # Direction 2 = East in the shared config
    idx = get_candidate_index(candidates, 4, 4, 2)
    visible = masks[idx].reshape(9, 9)

    assert visible[4, 4]       # Camera's own cell
    assert visible[4, 6]       # East
    assert visible[2, 6]       # North-east
    assert not visible[2, 4]    # North is outside the wedge


def test_wall_blocks_visibility():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 3] = 1

    candidates, masks = build_masks(
        grid, cam_range=5, fov_deg=90
    )

    # Direction 2 = East
    idx = get_candidate_index(candidates, 2, 1, 2)
    visible = masks[idx].reshape(5, 5)

    assert visible[2, 2]
    assert not visible[2, 4]



def test_east_camera_does_not_see_west():
    grid = np.zeros((5, 5), dtype=np.uint8)

    candidates, masks = build_masks(
        grid, cam_range=4, fov_deg=90
    )

    # Direction 2 = East
    idx = get_candidate_index(candidates, 2, 2, 2)
    visible = masks[idx]

    assert visible[2, 4]       # East is visible
    assert not visible[2, 0]   # West is not visible


def test_overlapping_cameras_do_not_double_count(tmp_path):
    grid = np.zeros((5, 5), dtype=np.uint8)

    candidates, masks = build_masks(
        grid, cam_range=4, fov_deg=90
    )

    save_masks(
        tmp_path / "tiny.npz",
        grid,
        candidates,
        masks,
        4, 90, 8
    )

    scorer = Scorer("tiny", masks_dir=tmp_path)

    a = get_candidate_index(candidates, 2, 1, 2)
    b = get_candidate_index(candidates, 2, 3, 2)

    combined = scorer.score([a, b])
    individual_sum = scorer.score([a]) + scorer.score([b])

    assert combined <= 1.0
    assert combined < individual_sum


def test_line_of_sight_is_symmetric_in_open_space():
    grid = np.zeros((5, 5), dtype=np.uint8)

    a = (1, 1)
    b = (3, 3)

    assert has_line_of_sight(grid, a, b)
    assert has_line_of_sight(grid, b, a)
