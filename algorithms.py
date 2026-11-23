"""
algorithms.py  --  Person 2 (Search)

Search algorithms for CCTV camera placement: random, greedy, hill climbing
(steepest-ascent / first-improvement), hill climbing with random restarts,
simulated annealing.

The algorithms never touch masks or weights. They only call the scorer
(Person 1's Scorer), so they work with any scorer that has:
    scorer.n_candidates          number of candidate placements N
    scorer.score(selected)       weighted coverage in [0, 1] of a list of candidate indices

----------------------------------------------------------------------------
CONTRACT
----------------------------------------------------------------------------
    run(algorithm, scorer, K, budget, seed) -> RunResult
        placement    list of K candidate indices (0..N-1)
        final_score  scorer.score(placement)
        history      best score so far after each evaluation (list, length = budget)
        runtime      seconds
        evaluations  evaluations actually used (<= budget)

    algorithm : a name from config.ALGORITHMS (spelling variants such as
                "hill-climbing" / "simulated_annealing" are accepted)
    budget    : max number of scorer.score() calls (None = cost of one greedy run)
    seed      : int; same seed -> same result

What counts as one evaluation = one call to scorer.score().
    random / hill climbing / annealing : scoring one full K-camera placement
    greedy                             : scoring "current set + one candidate"
                                         (one full run costs greedy_cost(N, K))
History entry i = best score seen among the first i+1 evaluations. For greedy this
includes partial sets (fewer than K cameras). If an algorithm stops early (hill
climbing at a local maximum) the history is padded with its last value.
"""

from __future__ import annotations

import re
import time
from typing import Sequence

import numpy as np

try:  # use the team's shared result type if interfaces.py defines it
    from interfaces import RunResult
except ImportError:  # standalone use / tests
    from dataclasses import dataclass

    @dataclass
    class RunResult:
        placement: list
        final_score: float
        history: list
        runtime: float
        evaluations: int


# Tuned settings go here once task 2.7 is done (picked up automatically by run()).
DEFAULT_PARAMS = {
    "annealing": {"T0": 0.02, "T_end": 1e-4},
}

ALGORITHMS = ("random", "greedy", "hill_climb", "hill_climb_first",
              "hill_climb_restarts", "annealing")

_ALIASES = {
    "random": "random", "random_placement": "random", "random_search": "random",
    "greedy": "greedy", "greedy_selection": "greedy",
    "hill_climb": "hill_climb", "hill_climbing": "hill_climb", "hillclimb": "hill_climb",
    "hillclimbing": "hill_climb", "steepest_ascent": "hill_climb",
    "steepest_ascent_hill_climbing": "hill_climb", "hill_climbing_steepest": "hill_climb",
    "hill_climb_steepest": "hill_climb",
    "hill_climb_first": "hill_climb_first", "hill_climbing_first": "hill_climb_first",
    "first_improvement": "hill_climb_first", "first_ascent": "hill_climb_first",
    "hill_climb_restarts": "hill_climb_restarts", "hill_climbing_restarts": "hill_climb_restarts",
    "hill_climb_random_restarts": "hill_climb_restarts",
    "hill_climbing_random_restarts": "hill_climb_restarts",
    "hill_climbing_with_restarts": "hill_climb_restarts",
    "random_restarts": "hill_climb_restarts", "restarts": "hill_climb_restarts",
    "hc_restarts": "hill_climb_restarts",
    "annealing": "annealing", "simulated_annealing": "annealing", "sa": "annealing",
}


def canonical(name: str) -> str:
    """Map a config.ALGORITHMS name to one of ALGORITHMS (raises if unknown)."""
    key = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    if key not in _ALIASES:
        raise ValueError(f"unknown algorithm {name!r}; supported: {sorted(set(_ALIASES))}")
    return _ALIASES[key]


def check_names(names: Sequence[str]) -> dict:
    """{name: canonical name or None}. Use on config.ALGORITHMS to find unmapped names."""
    out = {}
    for n in names:
        try:
            out[n] = canonical(n)
        except ValueError:
            out[n] = None
    return out


def greedy_cost(N: int, K: int) -> int:
    """Evaluations one full greedy run needs: N + (N-1) + ... + (N-K+1)."""
    return K * N - K * (K - 1) // 2


# --------------------------------------------------------------------------
# Evaluation counter / history, wrapped around the scorer
# --------------------------------------------------------------------------
class BudgetExhausted(Exception):
    pass


class _Ev:
    def __init__(self, scorer, budget):
        self.s = scorer
        self.budget = int(budget)
        self.n = 0
        self.best = 0.0
        self._hist = np.zeros(self.budget)

    @property
    def left(self):
        return self.budget - self.n

    def score(self, sel):
        """Counted evaluation."""
        if self.n >= self.budget:
            raise BudgetExhausted
        v = float(self.s.score(list(sel)))
        if v > self.best:
            self.best = v
        self._hist[self.n] = self.best
        self.n += 1
        return v

    def free_score(self, sel):
        """Not counted in our budget/history (only used to fill in a final score)."""
        return float(self.s.score(list(sel)))

    def history(self):
        h = self._hist.copy()
        if self.n < self.budget:
            h[self.n:] = self.best
        return h


def _rand_sel(rng, N, K):
    return [int(i) for i in rng.choice(N, size=K, replace=False)]


# --------------------------------------------------------------------------
# Algorithms: each returns (placement, its score)
# --------------------------------------------------------------------------
def _random(ev, N, K, rng, **_):
    """Sample random K-sets until the budget is spent; keep the best."""
    best, bv = None, -1.0
    while ev.left >= 1:
        sel = _rand_sel(rng, N, K)
        v = ev.score(sel)
        if v > bv:
            best, bv = sel, v
    return best, bv


def _greedy(ev, N, K, rng=None, **_):
    """Add the candidate that raises the score most, K times (ties -> lowest index)."""
    sel = []
    val = 0.0
    best_c, best_v = None, -1.0
    try:
        for _r in range(K):
            best_c, best_v = None, -1.0
            chosen = set(sel)
            for c in range(N):
                if c in chosen:
                    continue
                v = ev.score(sel + [c])
                if v > best_v:
                    best_c, best_v = c, v
            sel.append(best_c)
            val = best_v
        return sel, val
    except BudgetExhausted:
        # budget smaller than greedy_cost: keep the best option of the unfinished
        # round, then fill up to K with the lowest unused indices
        if best_c is not None:
            sel.append(best_c)
        chosen = set(sel)
        for c in range(N):
            if len(sel) >= K:
                break
            if c not in chosen:
                sel.append(c)
        return sel, ev.free_score(sel)


def _climb(ev, sel, cur, K, N, eps=1e-12):
    """Steepest ascent: try every one-camera swap, take the best, repeat."""
    sel = list(sel)
    while True:
        best_move, best_val = None, cur + eps
        try:
            for j in range(K):
                chosen = set(sel)
                for c in range(N):
                    if c in chosen:
                        continue
                    trial = sel.copy()
                    trial[j] = c
                    v = ev.score(trial)
                    if v > best_val:
                        best_val, best_move = v, (j, c)
        except BudgetExhausted:
            if best_move is not None:
                sel[best_move[0]] = best_move[1]
                cur = best_val
            return sel, cur
        if best_move is None:
            return sel, cur  # local maximum
        sel[best_move[0]] = best_move[1]
        cur = best_val


def _hill_climb(ev, N, K, rng, **_):
    """One steepest-ascent climb from a random start (may stop early at a local max)."""
    sel = _rand_sel(rng, N, K)
    cur = ev.score(sel)
    return _climb(ev, sel, cur, K, N)


def _hill_climb_first(ev, N, K, rng, patience=None, **_):
    """First-improvement climb: accept the first random swap that helps; stop after
    `patience` consecutive failures (default min(K*(N-K), 1000))."""
    patience = patience or min(K * (N - K), 1000)
    sel = _rand_sel(rng, N, K)
    cur = ev.score(sel)
    fails = 0
    while fails < patience and ev.left >= 1:
        j = int(rng.integers(K))
        c = int(rng.integers(N))
        if c in sel:
            continue
        trial = sel.copy()
        trial[j] = c
        v = ev.score(trial)
        if v > cur + 1e-12:
            sel, cur, fails = trial, v, 0
        else:
            fails += 1
    return sel, cur


def _hill_climb_restarts(ev, N, K, rng, **_):
    """Steepest-ascent climbs from fresh random starts until the budget is gone."""
    best, bv = None, -1.0
    while ev.left >= 1:
        sel = _rand_sel(rng, N, K)
        cur = ev.score(sel)
        sel, cur = _climb(ev, sel, cur, K, N)
        if cur > bv:
            best, bv = list(sel), cur
    return best, bv


def _annealing(ev, N, K, rng, T0=0.02, T_end=1e-4, **_):
    """
    Simulated annealing. Neighbour = swap one camera for a random unused candidate.
    Scores lie in [0, 1], so T0 is a score difference (0.02: a 2-point drop is accepted
    with probability 1/e at the start). Geometric cooling from T0 to T_end over the budget.
    """
    sel = _rand_sel(rng, N, K)
    cur = ev.score(sel)
    best, bv = list(sel), cur
    ratio = T_end / T0
    while ev.left >= 1:
        T = T0 * ratio ** (ev.n / ev.budget)
        j = int(rng.integers(K))
        c = int(rng.integers(N))
        if c in sel:
            continue
        trial = sel.copy()
        trial[j] = c
        v = ev.score(trial)
        d = v - cur
        if d >= 0 or rng.random() < np.exp(d / T):
            sel, cur = trial, v
            if cur > bv:
                best, bv = list(sel), cur
    return best, bv


_DISPATCH = {
    "random": _random,
    "greedy": _greedy,
    "hill_climb": _hill_climb,
    "hill_climb_first": _hill_climb_first,
    "hill_climb_restarts": _hill_climb_restarts,
    "annealing": _annealing,
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def run(algorithm, scorer, K, budget=None, seed=0, **params):
    """Run one algorithm once. See the module docstring for the contract.
    **params override DEFAULT_PARAMS (e.g. T0=..., T_end=..., patience=...)."""
    name = canonical(algorithm)
    N = int(scorer.n_candidates)
    K = int(K)
    if not 1 <= K <= N:
        raise ValueError(f"K={K} must be between 1 and n_candidates={N}")
    budget = greedy_cost(N, K) if budget is None else int(budget)
    if budget < 1:
        raise ValueError("budget must be at least 1")

    p = {**DEFAULT_PARAMS.get(name, {}), **params}
    ev = _Ev(scorer, budget)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    sel, val = _DISPATCH[name](ev, N, K, rng, **p)
    runtime = time.perf_counter() - t0

    return RunResult(
        placement=[int(i) for i in sel],
        final_score=float(val),
        history=ev.history().tolist(),
        runtime=float(runtime),
        evaluations=int(ev.n),
    )


def run_equal_budget(algorithms, scorer, K, budget=None, seed=0, params=None):
    """Run several algorithms with the SAME budget and seed -> {name: RunResult}.
    Default budget = cost of one greedy run. params: {algorithm: {param: value}}."""
    N = int(scorer.n_candidates)
    budget = greedy_cost(N, K) if budget is None else int(budget)
    params = params or {}
    return {a: run(a, scorer, K, budget, seed, **params.get(a, {})) for a in algorithms}
