
import numpy as np

from visibility import (
    bresenham,
    has_line_of_sight,
    visible_cells_from,
    build_candidates,
)


def test_camera_at_map_corner():
    """Visibility calculation should work at the top-left boundary."""
    grid = np.zeros((5, 5), dtype=int)

    visible = visible_cells_from(grid, 0, 0, 3)

    visible_positions = [(r, c) for r, c, angle in visible]

    assert (0, 0) in visible_positions

    for r, c, angle in visible:
        assert 0 <= r < 5
        assert 0 <= c < 5


def test_wall_blocks_line_of_sight():
    """A wall between two cells should block direct visibility."""
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1

    assert not has_line_of_sight(grid, (0, 0), (4, 4))


def test_open_path_has_line_of_sight():
    """An unobstructed horizontal path should remain visible."""
    grid = np.zeros((5, 5), dtype=int)

    assert has_line_of_sight(grid, (2, 0), (2, 4))


def test_candidates_are_only_on_open_cells():
    """Camera candidates should not be generated on walls."""
    grid = np.zeros((5, 5), dtype=int)
    grid[1, 2] = 1
    grid[2, 2] = 1

    candidates = build_candidates(grid)

    for r, c, direction in candidates:
        assert grid[r, c] == 0
        assert 0 <= direction < 8
