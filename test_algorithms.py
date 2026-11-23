"""Tests for algorithms.py using a stub scorer. Run: python test_algorithms.py (or pytest)."""
import itertools
import numpy as np
from algorithms import (ALGORITHMS, run, run_equal_budget, greedy_cost, canonical, check_names)
from stub_score import stub_scorer

RESULT_FIELDS = {"placement", "final_score", "history", "runtime", "evaluations"}


def test_result_has_contract_fields_and_types():
    s = stub_scorer("hotspot")
    r = run("greedy", s, 3, None, 0)
    assert RESULT_FIELDS <= set(vars(r)), vars(r).keys()
    assert isinstance(r.placement, list) and all(isinstance(i, int) for i in r.placement)
    assert isinstance(r.history, list) and isinstance(r.final_score, float)


def test_all_return_k_unique_valid_indices():
    for a in ALGORITHMS:
        s = stub_scorer()
        r = run(a, s, 4, 1500, 1)
        assert len(r.placement) == 4 and len(set(r.placement)) == 4, a
        assert all(0 <= i < s.n_candidates for i in r.placement), a


def test_budget_never_exceeded_and_history_shape():
    for a in ALGORITHMS:
        r = run(a, stub_scorer(), 3, 777, 2)
        assert r.evaluations <= 777, (a, r.evaluations)
        assert len(r.history) == 777
        assert (np.diff(r.history) >= -1e-12).all(), a


def test_evaluations_match_scorer_counter():
    for a in ALGORITHMS:
        s = stub_scorer("hotspot")
        r = run(a, s, 3, 800, 4)
        assert r.evaluations == s.evaluations, (a, r.evaluations, s.evaluations)


def test_final_score_matches_scorer_and_history():
    for a in ALGORITHMS:
        s = stub_scorer("hotspot")
        r = run(a, s, 3, 700, 3)
        assert abs(r.final_score - s.score(r.placement)) < 1e-9, a
        assert abs(r.history[-1] - r.final_score) < 1e-9, (a, r.history[-1], r.final_score)


def test_reproducible_with_seed():
    for a in ("random", "hill_climb_restarts", "annealing", "hill_climb_first"):
        r1 = run(a, stub_scorer("hotspot"), 3, 500, 7)
        r2 = run(a, stub_scorer("hotspot"), 3, 500, 7)
        assert r1.placement == r2.placement and r1.final_score == r2.final_score, a


def test_greedy_default_budget_is_its_cost():
    s = stub_scorer()
    r = run("greedy", s, 4, None, 0)
    assert r.evaluations == greedy_cost(s.n_candidates, 4)


def test_greedy_small_budget_still_returns_k_cameras():
    r = run("greedy", stub_scorer("hotspot"), 4, 100, 0)
    assert len(set(r.placement)) == 4 and r.evaluations == 100


def test_greedy_meets_guarantee_vs_brute_force():
    full = stub_scorer("hotspot")
    keep = list(range(0, full.n_candidates, 6))
    from stub_score import MaskScorer
    sub = MaskScorer(full._M[keep], full._w)
    K = 3
    opt = max(MaskScorer(full._M[keep], full._w).score(c)
              for c in itertools.combinations(range(sub.n_candidates), K))
    g = run("greedy", sub, K, None, 0)
    assert g.final_score >= (1 - 1 / np.e) * opt - 1e-9 and g.final_score <= opt + 1e-9


def test_hill_climb_ends_at_local_maximum():
    s = stub_scorer("hotspot")
    r = run("hill_climb", s, 3, greedy_cost(s.n_candidates, 3) * 20, 5)
    chk = stub_scorer("hotspot")
    cur = chk.score(r.placement)
    for j in range(3):
        for c in range(chk.n_candidates):
            if c in r.placement:
                continue
            t = r.placement.copy(); t[j] = c
            assert chk.score(t) <= cur + 1e-9


def test_search_beats_random_on_average():
    K = 4
    B = 5 * greedy_cost(stub_scorer().n_candidates, K)   # steepest ascent needs ~K*(N-K) evals/step
    rnd = np.mean([run("random", stub_scorer("hotspot"), K, B, s).final_score for s in range(5)])
    for a in ("greedy", "hill_climb_restarts", "annealing"):
        sc = np.mean([run(a, stub_scorer("hotspot"), K, B, s).final_score for s in range(5)])
        assert sc >= rnd - 1e-9, (a, sc, rnd)


def test_equal_budget_wrapper():
    out = run_equal_budget(ALGORITHMS, stub_scorer(), 3, 900, 0)
    assert set(out) == set(ALGORITHMS)
    for r in out.values():
        assert r.evaluations <= 900 and len(r.history) == 900


def test_name_variants_accepted():
    for n, want in [("Hill-Climbing", "hill_climb"), ("simulated annealing", "annealing"),
                    ("Random Restarts", "hill_climb_restarts"), ("greedy", "greedy"), ("random", "random")]:
        assert canonical(n) == want, n
    assert check_names(["greedy", "nonsense"])["nonsense"] is None


def test_config_algorithm_names_all_supported():
    try:
        import config
        names = config.ALGORITHMS
    except Exception:
        return  # config not available in standalone use
    bad = [n for n, c in check_names(names).items() if c is None]
    assert not bad, f"config.ALGORITHMS names not supported by algorithms.py: {bad}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
