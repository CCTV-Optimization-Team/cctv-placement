"""interfaces.py - the contract between the four modules.

Every function below already returns VALID dummy output, so anyone can develop
against it from day 1. When your real module is ready, swap the body for your
real code (or make it call your module) WITHOUT changing the signature.
Do not change a signature without telling the whole team.

Data shapes
  GridMap      map.py  (.grid, .entrances, .pois, .zones, .rows, .cols)
  candidates   list of (row, col, direction)
  masks        bool array (n_candidates, rows, cols); masks[i, r, c] True = candidate i sees cell
  selected     list of INDICES into candidates, length K
  weights      float array (rows, cols), walls = 0
  history      list of best-score-so-far, one value per score evaluation
"""
import time
from dataclasses import dataclass, field

import numpy as np

import config
from map import GridMap, build_map  # build_map(name) -> GridMap   (Person 4, real)


# ---------------------------------------------------------------- Person 1
def build_candidates(gmap: GridMap, cfg=config):
    """-> (candidates, masks).  STUB: a plain square around the camera, walls ignored."""
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
    """Weighted-coverage scorer.  STUB version; Person 1's real Scorer replaces it.

    Scorer(gmap, weights=None)   weights None -> uniform
      .candidates                list of (row, col, direction)
      .n_candidates              int
      .score(selected) -> float  weighted coverage in [0, 1]
      .raw_coverage(selected)    fraction of open cells covered (unweighted)
      .evaluations               number of score() calls so far (Person 2 uses it for budgets)
    """

    def __init__(self, gmap: GridMap, weights=None, cfg=config):
        self.gmap = gmap
        self.candidates, self.masks = build_candidates(gmap, cfg)
        self.weights = uniform_weights(gmap) if weights is None else np.asarray(weights, float)
        self.n_candidates = len(self.candidates)
        self.evaluations = 0

    def _covered(self, selected):
        return self.masks[list(selected)].any(axis=0)

    def score(self, selected) -> float:
        self.evaluations += 1
        total = self.weights.sum()
        return float(self.weights[self._covered(selected)].sum() / total) if total else 0.0

    def raw_coverage(self, selected) -> float:
        open_ = self.gmap.grid == 0
        return float((self._covered(selected) & open_).sum() / open_.sum())


# ---------------------------------------------------------------- Person 3
def uniform_weights(gmap: GridMap):
    return (gmap.grid == 0).astype(float)


def predict_priority(gmap: GridMap):
    """-> float array (rows, cols), walls 0.  STUB: uniform."""
    return uniform_weights(gmap)


def true_priority(gmap: GridMap):
    """-> float array (rows, cols), walls 0.  STUB: uniform. Real one loads priority/<name>_true.npy."""
    return uniform_weights(gmap)


def get_weights(gmap: GridMap, weighting: str):
    """weighting in config.WEIGHTINGS -> weight array. Used by the runner."""
    return {"uniform": uniform_weights, "predicted": predict_priority,
            "true": true_priority}[weighting](gmap)


# ---------------------------------------------------------------- Person 2
@dataclass
class RunResult:
    placement: list                   # selected indices, length K
    final_score: float
    history: list = field(default_factory=list)   # best score so far after each evaluation
    runtime: float = 0.0              # seconds
    evaluations: int = 0


def run(algorithm: str, scorer: Scorer, K: int, budget: int, seed: int) -> RunResult:
    """STUB: random placement.  Algorithms: see config.ALGORITHMS.

    NOTE: weights are baked into the scorer (Scorer(gmap, weights=...)), so the
    plan's run(algorithm, map, K, weights, budget, seed) becomes run(algorithm, scorer, K, budget, seed).
    """
    rng = np.random.default_rng(seed)
    t0, start_evals = time.perf_counter(), scorer.evaluations
    best, best_sel, history = -1.0, None, []
    for _ in range(budget):
        sel = rng.choice(scorer.n_candidates, size=K, replace=False).tolist()
        s = scorer.score(sel)
        if s > best:
            best, best_sel = s, sel
        history.append(best)
    return RunResult(best_sel, best, history, time.perf_counter() - t0,
                     scorer.evaluations - start_evals)


# ---------------------------------------------------------------- Person 4
def run_experiments(cfg=config, **kwargs):
    """-> pandas DataFrame with columns config.RESULT_COLUMNS. The real code is in runner.py."""
    from runner import run_experiments as _run
    return _run(cfg, **kwargs)
