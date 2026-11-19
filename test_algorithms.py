"""Tests for algorithms.py. Run:  python test_algorithms.py   (or pytest)."""
import itertools
import numpy as np
from algorithms import (ALGORITHMS, run, run_equal_budget, greedy_cost,
                        evaluate_placement, Scorer)
from stub_score import build_stub, stub_weights

GRID, CANDS, MASKS = build_stub(half_width=3, stride=2)
N = len(CANDS)
W_UNI = stub_weights(GRID)
W_HOT = stub_weights(GRID, "hotspot")


def test_all_return_k_unique_valid_indices():
    for a in ALGORITHMS:
        r = run(a, K=4, weights=W_UNI, budget=1500, seed=1, masks=MASKS)
        assert len(r.placement) == 4 and len(set(r.placement)) == 4, a
        assert all(0 <= i < N for i in r.placement), a


def test_budget_never_exceeded_and_history_shape():
    for a in ALGORITHMS:
        r = run(a, K=3, weights=W_UNI, budget=777, seed=2, masks=MASKS)
        assert r.evaluations <= 777, (a, r.evaluations)
        assert len(r.history) == 777
        assert (np.diff(r.history) >= -1e-12).all(), a  # best-so-far never drops


def test_history_matches_final_score():
    for a in ("random", "hill_climb_restarts", "annealing", "hill_climb"):
        r = run(a, K=3, weights=W_HOT, budget=600, seed=3, masks=MASKS)
        assert abs(r.history[-1] - r.score) < 1e-6, (a, r.history[-1], r.score)


def test_reproducible_with_seed():
    for a in ("random", "hill_climb_restarts", "annealing", "hill_climb_first"):
        r1 = run(a, K=3, weights=W_HOT, budget=500, seed=7, masks=MASKS)
        r2 = run(a, K=3, weights=W_HOT, budget=500, seed=7, masks=MASKS)
        assert r1.placement == r2.placement and r1.score == r2.score, a


def test_score_matches_direct_formula():
    r = run("greedy", K=3, weights=W_HOT, masks=MASKS)
    cov = MASKS[r.placement].any(axis=0)
    expect = W_HOT[cov].sum() / W_HOT.sum()
    assert abs(r.score - expect) < 1e-9
    assert abs(evaluate_placement(r.placement, MASKS, W_HOT) - expect) < 1e-9


def test_greedy_default_budget_is_its_cost():
    r = run("greedy", K=4, weights=W_UNI, masks=MASKS)
    assert r.evaluations == greedy_cost(N, 4)


def test_greedy_meets_approximation_guarantee_vs_brute_force():
    sub = MASKS[::6]                      # small enough to brute force
    K = 3
    opt = max(Scorer(sub, W_HOT).value(c) for c in itertools.combinations(range(len(sub)), K))
    g = run("greedy", K=K, weights=W_HOT, masks=sub)
    assert g.score >= (1 - 1 / np.e) * opt - 1e-9
    assert g.score <= opt + 1e-9


def test_hill_climb_ends_at_local_maximum():
    r = run("hill_climb", K=3, weights=W_HOT, budget=greedy_cost(N, 3) * 20, seed=5, masks=MASKS)
    sc = Scorer(MASKS, W_HOT)
    cur = sc.value(r.placement)
    for j in range(3):
        for c in range(N):
            if c in r.placement:
                continue
            trial = r.placement.copy(); trial[j] = c
            assert sc.value(trial) <= cur + 1e-9


def test_search_beats_random_on_average():
    K, B = 4, 5 * greedy_cost(N, 4)   # steepest ascent needs ~K*(N-K) evals per step
    rnd = np.mean([run("random", K=K, weights=W_HOT, budget=B, seed=s, masks=MASKS).score for s in range(5)])
    for a in ("greedy", "hill_climb_restarts", "annealing"):
        sc = np.mean([run(a, K=K, weights=W_HOT, budget=B, seed=s, masks=MASKS).score for s in range(5)])
        assert sc >= rnd - 1e-9, (a, sc, rnd)


def test_equal_budget_wrapper():
    out = run_equal_budget(ALGORITHMS, K=3, weights=W_UNI, budget=900, seed=0, masks=MASKS,
                           map_name="tiny", weighting="uniform")
    assert set(out) == set(ALGORITHMS)
    for r in out.values():
        assert r.budget == 900 and r.evaluations <= 900 and len(r.history) == 900
        assert set(r.row()) == {"algorithm", "map", "K", "weighting", "seed", "score", "runtime", "evaluations"}


def test_mask_shapes_flat_or_2d():
    flat = MASKS.reshape(N, -1)
    a = run("greedy", K=3, weights=W_UNI, masks=MASKS).placement
    b = run("greedy", K=3, weights=W_UNI.reshape(-1), masks=flat).placement
    assert a == b


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
