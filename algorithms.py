"""
algorithms.py  --  Person 2 (Search)

Search algorithms for CCTV camera placement:
    random, greedy, hill climbing (steepest / first-improvement),
    hill climbing with random restarts, simulated annealing.

Everything works on PRECOMPUTED visibility masks (Person 1's masks/<name>.npz)
and a per-cell weight array. No line of sight is ever computed in here.

----------------------------------------------------------------------------
INTERFACE (agreed contract)
----------------------------------------------------------------------------
    run(algorithm, map, K, weights, budget, seed, masks=None, **params) -> RunResult

    algorithm : "random" | "greedy" | "hill_climb" | "hill_climb_first"
                | "hill_climb_restarts" | "annealing"
    map       : whatever build_map() returns. Only used to build masks when
                `masks` is not given (via visibility.build_candidates).
    K         : number of cameras to place
    weights   : per-cell priority weights (flat or H x W). None = uniform.
                Use 0 on wall cells so raw coverage is measured on open cells.
    budget    : max number of score evaluations (None = cost of one greedy run)
    seed      : int, makes every run reproducible
    masks     : bool array (N, H, W) or (N, H*W): one visibility mask per candidate

Candidates are referred to by their INDEX 0..N-1 in the candidate list, so a
"placement" is a list of K candidate indices.

What counts as one evaluation
    * random / hill climbing / annealing : one evaluation = scoring one full
      K-camera placement.
    * greedy : one evaluation = scoring one "current set + candidate" option,
      so a full greedy run costs about N*K evaluations (see greedy_cost()).
    * best-score history has one entry per evaluation (best seen so far), and
      is padded with its last value up to `budget` if an algorithm stops early.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

ALGORITHMS = (
    "random",
    "greedy",
    "hill_climb",
    "hill_climb_first",
    "hill_climb_restarts",
    "annealing",
)

_CHUNK = 2048  # candidates per matrix product (keeps memory small)


# --------------------------------------------------------------------------
# Scorer: objective + evaluation counting + history
# --------------------------------------------------------------------------
class BudgetExhausted(Exception):
    pass


class Scorer:
    """Weighted coverage objective over stored masks, with budget accounting."""

    def __init__(self, masks, weights=None, budget=None, K=None):
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim < 2:
            raise ValueError("masks must have shape (N, H, W) or (N, H*W)")
        self.N = masks.shape[0]
        self.M = masks.reshape(self.N, -1)
        self.C = self.M.shape[1]

        if weights is None:
            w = np.ones(self.C, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.size != self.C:
                raise ValueError(f"weights has {w.size} cells but masks have {self.C}")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        self.w = w
        self.total = float(w.sum())
        if self.total <= 0:
            raise ValueError("weights sum to zero")
        self.w32 = (w / self.total).astype(np.float32)  # normalised, for chunked gains
        self.wn = w / self.total

        self.K = K
        self.budget = int(budget) if budget is not None else None
        self.evals = 0
        self._hist = np.zeros(self.budget if self.budget is not None else 0)
        self.best_value = 0.0
        self.best_sel = None

    # ---- budget / history ------------------------------------------------
    def remaining(self):
        return np.inf if self.budget is None else self.budget - self.evals

    def exhausted(self):
        return self.remaining() <= 0

    def _log(self, values, sels=None):
        """Log a batch of evaluation values (a numpy array) into the history."""
        values = np.asarray(values, dtype=np.float64)
        n = values.size
        if n == 0:
            return
        if self.budget is not None and self.evals + n > self.budget:
            raise BudgetExhausted
        run_best = np.maximum.accumulate(np.maximum(values, self.best_value))
        if self.budget is not None:
            self._hist[self.evals:self.evals + n] = run_best
        self.evals += n
        self.best_value = float(run_best[-1])

    # ---- exact (uncounted) scoring ---------------------------------------
    def covered_mask(self, sel):
        if len(sel) == 0:
            return np.zeros(self.C, dtype=bool)
        return self.M[list(sel)].any(axis=0)

    def value(self, sel):
        """Score of a placement WITHOUT counting an evaluation."""
        return float(self.wn[self.covered_mask(sel)].sum())

    # ---- counted scoring --------------------------------------------------
    def evaluate(self, sel):
        """Score one full placement. Costs 1 evaluation."""
        if self.remaining() < 1:
            raise BudgetExhausted
        v = self.value(sel)
        prev_best = self.best_value
        self._log(np.array([v]))
        if v > prev_best or self.best_sel is None:
            self.best_sel = list(sel)
        return v

    def values_for_all(self, base_cov, exclude, limit=None):
        """
        Value of (base_cov UNION mask[c]) for every candidate c not in `exclude`.
        Each candidate costs 1 evaluation. If the remaining budget is smaller
        than the number of candidates, only the first `remaining` are scored.
        Returns (candidate_indices, values), already logged.
        """
        ex = np.zeros(self.N, dtype=bool)
        if len(exclude):
            ex[list(exclude)] = True
        cand = np.flatnonzero(~ex)
        room = self.remaining()
        if limit is not None:
            room = min(room, limit)
        if room < cand.size:
            cand = cand[: int(room)]
        if cand.size == 0:
            raise BudgetExhausted
        base_val = float(self.wn[base_cov].sum())
        residual = self.w32 * (~base_cov)
        gains = np.empty(cand.size, dtype=np.float64)
        for s in range(0, cand.size, _CHUNK):
            blk = cand[s:s + _CHUNK]
            gains[s:s + blk.size] = self.M[blk].astype(np.float32) @ residual
        values = base_val + gains
        self._log(values)
        return cand, values

    def history(self):
        """Best-so-far score after each evaluation, padded to `budget`."""
        h = self._hist.copy()
        if self.evals < len(h):
            h[self.evals:] = self.best_value
        return h


# --------------------------------------------------------------------------
# Algorithms. Each takes a Scorer and an rng and returns a list of K indices.
# --------------------------------------------------------------------------
def _random_sel(rng, N, K):
    return [int(i) for i in rng.choice(N, size=K, replace=False)]


def random_placement(sc: Scorer, K, rng, **_):
    """Sample random K-sets until the budget is spent; keep the best."""
    best, best_v = None, -1.0
    while sc.remaining() >= 1:
        sel = _random_sel(rng, sc.N, K)
        v = sc.evaluate(sel)
        if v > best_v:
            best, best_v = sel, v
    return best if best is not None else _random_sel(rng, sc.N, K)


def greedy(sc: Scorer, K, rng=None, **_):
    """Add the candidate with the largest new weighted coverage, K times."""
    sel = []
    covered = np.zeros(sc.C, dtype=bool)
    try:
        for _ in range(K):
            cand, vals = sc.values_for_all(covered, sel)
            j = int(np.argmax(vals))  # ties -> lowest index (deterministic)
            sel.append(int(cand[j]))
            covered |= sc.M[sel[-1]]
    except BudgetExhausted:
        # budget too small for a full run: keep what we have, fill with the
        # best of what was scored in the last round if possible
        pass
    sc.best_sel = list(sel)
    return sel


def _climb_steepest(sc: Scorer, sel, K, eps=1e-12):
    """Steepest ascent: look at every one-camera swap, take the best."""
    sel = list(sel)
    cur = sc.value(sel)
    while True:
        best_move, best_val = None, cur + eps
        try:
            for j in range(K):
                others = sel[:j] + sel[j + 1:]
                base = sc.covered_mask(others)
                cand, vals = sc.values_for_all(base, sel)
                k = int(np.argmax(vals))
                if vals[k] > best_val:
                    best_val, best_move = float(vals[k]), (j, int(cand[k]))
        except BudgetExhausted:
            if best_move is not None:  # use what we scored before running out
                sel[best_move[0]] = best_move[1]
                cur = best_val
            return sel, cur
        if best_move is None:
            return sel, cur  # local maximum
        sel[best_move[0]] = best_move[1]
        cur = best_val


def hill_climb(sc: Scorer, K, rng, **_):
    """One steepest-ascent climb from a random start. May stop early (local max)."""
    sel = _random_sel(rng, sc.N, K)
    sc.evaluate(sel)
    sel, _ = _climb_steepest(sc, sel, K)
    sc.best_sel = list(sel)
    return sel


def hill_climb_first(sc: Scorer, K, rng, patience=None, **_):
    """
    First-improvement climb: try random single-camera swaps, accept the first
    that improves. Stops after `patience` consecutive failures (default
    min(K*(N-K), 1000)), i.e. when it is probably at a local maximum.
    """
    N = sc.N
    patience = patience or min(K * (N - K), 1000)
    sel = _random_sel(rng, N, K)
    cur = sc.evaluate(sel)
    fails = 0
    while fails < patience and sc.remaining() >= 1:
        j = int(rng.integers(K))
        c = int(rng.integers(N))
        if c in sel:
            continue
        trial = sel.copy()
        trial[j] = c
        v = sc.evaluate(trial)
        if v > cur + 1e-12:
            sel, cur, fails = trial, v, 0
        else:
            fails += 1
    return sel


def hill_climb_restarts(sc: Scorer, K, rng, **_):
    """Steepest-ascent climbs from fresh random starts until the budget is gone."""
    best, best_v = None, -1.0
    while sc.remaining() >= 1:
        sel = _random_sel(rng, sc.N, K)
        try:
            sc.evaluate(sel)
        except BudgetExhausted:
            break
        sel, v = _climb_steepest(sc, sel, K)
        if v > best_v:
            best, best_v = list(sel), v
    sc.best_sel = list(best)
    return best


def annealing(sc: Scorer, K, rng, T0=0.02, T_end=1e-4, **_):
    """
    Simulated annealing over K-subsets. Neighbour = swap one camera for a
    random unused candidate. Scores live in [0, 1], so T0 is a score difference:
    T0=0.02 means a 2-point drop is accepted with probability 1/e at the start.
    Geometric cooling from T0 to T_end across the evaluation budget.
    """
    if sc.budget is None:
        raise ValueError("annealing needs a finite budget (cooling is tied to it)")
    N = sc.N
    sel = _random_sel(rng, N, K)
    cur = sc.evaluate(sel)
    best, best_v = list(sel), cur
    ratio = T_end / T0
    while sc.remaining() >= 1:
        T = T0 * ratio ** (sc.evals / sc.budget)
        j = int(rng.integers(K))
        c = int(rng.integers(N))
        if c in sel:
            continue
        trial = sel.copy()
        trial[j] = c
        v = sc.evaluate(trial)
        d = v - cur
        if d >= 0 or rng.random() < np.exp(d / T):
            sel, cur = trial, v
            if cur > best_v:
                best, best_v = list(sel), cur
    sc.best_sel = best
    return best


_DISPATCH = {
    "random": random_placement,
    "greedy": greedy,
    "hill_climb": hill_climb,
    "hill_climb_first": hill_climb_first,
    "hill_climb_restarts": hill_climb_restarts,
    "annealing": annealing,
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    algorithm: str
    K: int
    seed: int
    placement: list               # K candidate indices
    score: float                  # weighted coverage under the weights used in the search
    raw_coverage: float           # covered cells / cells with positive weight
    runtime: float                # seconds
    evaluations: int              # evaluations actually used (<= budget)
    budget: int
    history: np.ndarray = field(repr=False, default=None)  # best score after each evaluation, length = budget
    map_name: str = ""
    weighting: str = ""
    params: dict = field(default_factory=dict)

    def row(self):
        """One results/<run>.csv row (format agreed on day 1)."""
        return {
            "algorithm": self.algorithm,
            "map": self.map_name,
            "K": self.K,
            "weighting": self.weighting,
            "seed": self.seed,
            "score": self.score,
            "runtime": self.runtime,
            "evaluations": self.evaluations,
        }


def greedy_cost(N, K):
    """Evaluations one full greedy run needs: N + (N-1) + ... + (N-K+1)."""
    return K * N - K * (K - 1) // 2


def _get_masks(map, masks):
    if masks is not None:
        return masks
    try:
        import config
        from visibility import build_candidates
    except ImportError as e:  # pragma: no cover
        raise ValueError(
            "No masks given and visibility.build_candidates is not importable. "
            "Pass masks=... (bool array, one mask per candidate)."
        ) from e
    _cands, m = build_candidates(map, config)
    return m


def run(algorithm, map=None, K=3, weights=None, budget=None, seed=0,
        masks=None, map_name="", weighting="", **params):
    """Run one algorithm once. See the module docstring for the contract."""
    if algorithm not in _DISPATCH:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")
    masks = np.asarray(_get_masks(map, masks), dtype=bool)
    N = masks.shape[0]
    if not 1 <= K <= N:
        raise ValueError(f"K={K} must be between 1 and N={N}")
    if budget is None:
        budget = greedy_cost(N, K)
    budget = int(budget)

    sc = Scorer(masks, weights, budget=budget, K=K)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    sel = _DISPATCH[algorithm](sc, K, rng, **params)
    runtime = time.perf_counter() - t0

    sel = [int(i) for i in sel]
    cov = sc.covered_mask(sel)
    pos = sc.w > 0
    raw = float(cov[pos].sum() / max(pos.sum(), 1))
    return RunResult(
        algorithm=algorithm, K=K, seed=seed, placement=sel,
        score=sc.value(sel), raw_coverage=raw, runtime=runtime,
        evaluations=sc.evals, budget=budget, history=sc.history(),
        map_name=map_name, weighting=weighting, params=dict(params),
    )


def run_equal_budget(algorithms: Sequence[str], map=None, K=3, weights=None,
                     budget=None, seed=0, masks=None, params=None, **labels):
    """
    Run several algorithms with the SAME evaluation budget and seed.
    Default budget = cost of one greedy run, so greedy is not truncated.
    params: optional {algorithm: {param: value}} (e.g. tuned annealing settings).
    Returns {algorithm: RunResult}.
    """
    masks = np.asarray(_get_masks(map, masks), dtype=bool)
    if budget is None:
        budget = greedy_cost(masks.shape[0], K)
    params = params or {}
    return {
        a: run(a, map, K, weights, budget, seed, masks=masks,
               **labels, **params.get(a, {}))
        for a in algorithms
    }


def evaluate_placement(placement, masks, weights=None):
    """
    Score a finished placement under ANY weights (e.g. the TRUE priority map
    after searching with predicted weights). Not counted as an evaluation.
    """
    sc = Scorer(masks, weights)
    return sc.value(placement)


def save_history_csv(results, path):
    """Convergence data: one column per run, one row per evaluation."""
    import pandas as pd
    cols = {}
    for r in results if not isinstance(results, dict) else results.values():
        key = f"{r.algorithm}|seed{r.seed}"
        cols[key] = r.history
    pd.DataFrame(cols).rename_axis("evaluation").to_csv(path)


def results_table(results):
    """List/dict of RunResult -> pandas DataFrame in the results CSV format."""
    import pandas as pd
    rs = results.values() if isinstance(results, dict) else results
    return pd.DataFrame([r.row() for r in rs])
