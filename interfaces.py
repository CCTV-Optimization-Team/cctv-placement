
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
from ml_priority import predict_priority, true_priority


# ---------------------------------------------------------------------------
# Person 1  (real visibility model: Bresenham line of sight, range and field of view)

from visibility import Scorer, build_masks  # noqa: E402  (Scorer loads masks/<map>.npz)


def build_candidates(gmap: GridMap, cfg=config):
    """Return (candidates, masks): list of (row, col, direction) and bool (n, rows, cols)."""
    candidates, masks = build_masks(
        gmap.grid, cfg.CAMERA_RANGE, cfg.FOV_DEGREES, cfg.N_DIRECTIONS
    )
    return [tuple(int(v) for v in c) for c in candidates], masks


# ---------------------------------------------------------------------------
# Person 3


def uniform_weights(gmap: GridMap):
    return (gmap.grid == 0).astype(float)


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
    """Delegates to algorithms.py (Person 2)."""
    from algorithms import run as _run

    return _run(algorithm, scorer, K, budget, seed)


# ---------------------------------------------------------------------------
# Person 4


def run_experiments(cfg=config, **kwargs):
    """Return a DataFrame with config.RESULT_COLUMNS. Real code is in runner.py."""
    from runner import run_experiments as _run

    return _run(cfg, **kwargs)
