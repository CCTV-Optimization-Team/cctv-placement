"""
stub_score.py  --  Person 2's stand-in for Person 1's Scorer, used only for tests/demo.

MaskScorer has the same surface the algorithms rely on:
    .candidates, .n_candidates, .score(selected), .raw_coverage(selected), .evaluations
A camera "sees" a plain square around it and ignores walls and direction.
"""
import numpy as np


def make_tiny_map(n=20):
    """A 20x20 map: border walls plus a few internal wall segments. True = open."""
    grid = np.ones((n, n), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = False
    grid[8, 3:12] = False
    grid[3:9, 14] = False
    return grid


def stub_candidates(grid, stride=1, n_dirs=1):
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


def stub_weights(grid, kind="uniform"):
    """Per-cell weights, 0 on walls. kind: 'uniform' or 'hotspot' (fake priority)."""
    w = grid.astype(float)
    if kind == "hotspot":
        H, W = grid.shape
        rr, cc = np.mgrid[:H, :W]
        w = w * (1 + 9 * np.exp(-(((rr - 4) ** 2 + (cc - 4) ** 2) / 8.0)))
    return w


class MaskScorer:
    def __init__(self, masks, weights=None, candidates=None):
        masks = np.asarray(masks, dtype=bool)
        self.n_candidates = masks.shape[0]
        self._M = masks.reshape(self.n_candidates, -1)
        C = self._M.shape[1]
        self._w = np.ones(C) if weights is None else np.asarray(weights, float).reshape(-1)
        self._total = self._w.sum()
        self.candidates = candidates or [(i, 0, 0) for i in range(self.n_candidates)]
        self.evaluations = 0

    def _cov(self, selected):
        return self._M[list(selected)].any(axis=0) if len(selected) else np.zeros(self._M.shape[1], bool)

    def score(self, selected):
        self.evaluations += 1
        return float(self._w[self._cov(selected)].sum() / self._total)

    def raw_coverage(self, selected):
        pos = self._w > 0
        return float(self._cov(selected)[pos].sum() / max(pos.sum(), 1))


def stub_scorer(weighting="uniform", half_width=3, stride=2):
    grid = make_tiny_map()
    cands = stub_candidates(grid, stride=stride)
    return MaskScorer(stub_masks(grid, cands, half_width), stub_weights(grid, weighting), cands)
