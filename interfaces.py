
"""interfaces.py - the contract between the four modules.

Every function below already returns valid dummy output, so anyone can develop
against it from day 1. When your real module is ready, swap the body for your
real code (or make it call your module) WITHOUT changing the signature.
Do not change a signature without telling the whole team.

Data shapes:
    GridMap      map.py (.grid, .entrances, .pois, .zones, .rows, .cols)
    candidates    list of (row, col, direction)
    masks        bool array (n_candidates, rows, cols)
                 masks[i, r, c] True = candidate i sees cell
    selected     list of indices into candidates, length K
    weights      float array (rows, cols), walls = 0
    history      list of best-score-so-far, one value per score evaluation
"""

import time
from dataclasses import dataclass, field

import numpy as np

import config
from map import GridMap, build_map


# ---------------------------------------------------------------------------
# Person 1


def build_candidates(gmap: GridMap, cfg=config):
    """Return candidates and masks. STUB: square coverage, walls ignored."""
    candidates, masks = [], []
    R = cfg.CAMERA_RANGE

    for r, c in gmap.open_cells():
        for d in range(cfg.N_DIRECTIONS):
            m = np.zeros(gmap.grid.shape, dtype=bool)
            m[max(0, r - R):r + R + 1, max(0, c - R):c + R + 1] = True
            m &= gmap.grid == 0
            candidates.append((int(r), int(c), d))
            masks.append(m)

    return candidates, np.array(masks)


class Scorer:
    """Stub weighted-coverage scorer. Person 1's real Scorer replaces this.

    Scorer(gmap, weights=None): weights None -> uniform

    Attributes:
        candidates: list of (row, col, direction)
        n_candidates: number of candidates
        evaluations: number of score() calls
    """

    def __init__(self, gmap: GridMap, weights=None, cfg=config):
        self.gmap = gmap
        self.candidates, self.masks = build_candidates(gmap, cfg)
        self.weights = (
            uniform_weights(gmap)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        self.n_candidates = len(self.candidates)
        self.evaluations = 0

    def _covered(self, selected):
        return self.masks[list(selected)].any(axis=0)

    def score(self, selected) -> float:
        self.evaluations += 1
        total = self.weights.sum()
        return (
            float(self.weights[self._covered(selected)].sum() / total)
            if total
            else 0.0
        )

    def raw_coverage(self, selected) -> float:
        open_cells = self.gmap.grid == 0
        return float(
            (self._covered(selected) & open_cells).sum() / open_cells.sum()
        )

    def gain(self, covered, c):
        total = self.weights.sum()
        if total == 0:
            return 0.0

        new = self.masks[c] & ~covered
        return float(self.weights[new].sum() / total)

    def covered_grid(self, selected):
        return self._covered(selected)


# ---------------------------------------------------------------------------
# Person 3


def uniform_weights(gmap: GridMap):
    return (gmap.grid == 0).astype(float)


def predict_priority(gmap: GridMap):
    """Return priority weights. STUB: uniform."""
    return uniform_weights(gmap)


def true_priority(gmap: GridMap):
    """Return true priority weights. STUB: uniform."""
    return uniform_weights(gmap)


def get_weights(gmap: GridMap, weighting: str):
    """Return the requested weight array."""
    return {
        "uniform": uniform_weights,
        "predicted": predict_priority,
        "true": true_priority,
    }[weighting](gmap)


# ---------------------------------------------------------------------------
# Person 2


@dataclass
class RunResult:
    placement: list
    final_score: float
    history: list = field(default_factory=list)
    runtime: float = 0.0
    evaluations: int = 0


def run(
    algorithm: str,
    scorer: Scorer,
    K: int,
    budget: int,
    seed: int,
) -> RunResult:
    """STUB: random placement. Algorithms are defined in config.ALGORITHMS."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    start_evals = scorer.evaluations

    best, best_sel, history = -1.0, None, []

    for _ in range(budget):
        sel = rng.choice(scorer.n_candidates, size=K, replace=False).tolist()
        s = scorer.score(sel)

        if s > best:
            best, best_sel = s, sel

        history.append(best)

    return RunResult(
        best_sel,
        best,
        history,
        time.perf_counter() - t0,
        scorer.evaluations - start_evals,
    )


# ---------------------------------------------------------------------------
# Person 4


def run_experiments(cfg=config, **kwargs):
    """Return a DataFrame with config.RESULT_COLUMNS. Real code is in runner.py."""
    from runner import run_experiments as _run

    return _run(cfg, **kwargs)
