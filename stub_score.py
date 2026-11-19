"""
stub_score.py  --  Person 2's stand-in for Person 1's visibility model.

A camera "sees" a plain square around it and ignores walls and direction.
Use it to build masks the algorithms can run on from day 1; later swap in the
real masks (masks/<name>.npz) with NO change to algorithms.py.
"""
import numpy as np


def make_tiny_map(n=20, seed=0):
    """A 20x20 map: border walls plus a few internal wall segments. True = open."""
    grid = np.ones((n, n), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = False
    grid[8, 3:12] = False       # horizontal wall
    grid[3:9, 14] = False       # vertical wall
    return grid


def stub_candidates(grid, stride=1, n_dirs=1):
    """(row, col, direction) for every open cell on a stride grid."""
    rows, cols = np.nonzero(grid)
    return [(int(r), int(c), d)
            for r, c in zip(rows, cols) if r % stride == 0 and c % stride == 0
            for d in range(n_dirs)]


def stub_masks(grid, candidates, half_width=3):
    """Boolean masks (N, H, W): open cells in a square of +/- half_width."""
    H, W = grid.shape
    masks = np.zeros((len(candidates), H, W), dtype=bool)
    for i, (r, c, _d) in enumerate(candidates):
        r0, r1 = max(r - half_width, 0), min(r + half_width + 1, H)
        c0, c1 = max(c - half_width, 0), min(c + half_width + 1, W)
        masks[i, r0:r1, c0:c1] = grid[r0:r1, c0:c1]
    return masks


def stub_weights(grid, kind="uniform", seed=0):
    """Per-cell weights, 0 on walls. kind: 'uniform' or 'hotspot' (fake priority)."""
    w = grid.astype(float)
    if kind == "hotspot":
        H, W = grid.shape
        rr, cc = np.mgrid[:H, :W]
        w = w * (1 + 9 * np.exp(-(((rr - 4) ** 2 + (cc - 4) ** 2) / 8.0)))
    return w


def build_stub(half_width=3, stride=1):
    """Convenience: returns (grid, candidates, masks)."""
    g = make_tiny_map()
    c = stub_candidates(g, stride=stride)
    return g, c, stub_masks(g, c, half_width)
