"""Day-1 STUB so Persons 2, 3, 4 can code before the real masks exist.

StubScorer has the same methods as visibility.Scorer, but "coverage" is just a
plain square around each camera and walls are ignored. It is wrong on purpose.
Use:   from interfaces import StubScorer
Later: from visibility import Scorer   (same methods, nothing else changes)

Usage:  s = StubScorer(rows=20, cols=20, grid=None)
"""
import numpy as np

import config


class StubScorer:
    def __init__(self, rows=20, cols=20, grid=None, weights=None, half_side=None):
        self.rows, self.cols = rows, cols
        self.half = half_side if half_side is not None else config.CAMERA_RANGE // 2
        self.open_cells = (np.ones((rows, cols), bool) if grid is None
                           else (np.asarray(grid) == 0)).ravel()
        # candidates: every cell x every direction (direction is ignored by the stub)
        self.candidates = np.array(
            [(r, c, d) for r in range(rows) for c in range(cols)
             for d in range(config.N_DIRECTIONS)], dtype=np.int32)
        w = self.open_cells.astype(float) if weights is None \
            else np.asarray(weights, float).reshape(-1)
        self.weights = np.where(self.open_cells, w, 0.0)
        self.total = self.weights.sum()
        self.n_evals = 0

    @property
    def n_candidates(self):
        return len(self.candidates)

    def _mask(self, c):
        r, col, _ = self.candidates[c]
        m = np.zeros((self.rows, self.cols), bool)
        m[max(0, r - self.half):r + self.half + 1,
          max(0, col - self.half):col + self.half + 1] = True
        return m.ravel()

    def covered_mask(self, selected):
        out = np.zeros(self.rows * self.cols, bool)
        for c in selected:
            out |= self._mask(c)
        return out

    def score(self, selected):
        self.n_evals += 1

        if self.total == 0:
            return 0.0

        return float(
            self.weights[self.covered_mask(selected)].sum()
            / self.total
        )

    def gain(self, covered, c):
        if self.total == 0:
            return 0.0

        new = self._mask(c) & ~covered
        return float(self.weights[new].sum() / self.total)

    def covered_grid(self, selected):
        return self.covered_mask(selected).reshape(self.rows, self.cols)