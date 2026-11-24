"""runner.py (Person 4) - runs every algorithm on every map / K / weighting and saves results.

Usage (from the repo root):
    python3 runner.py --quick                 # tiny smoke test, a few seconds
    python3 runner.py --tag full              # everything in config.py
    python3 runner.py --maps tiny college --weightings uniform true --tag demo

Writes, in results/:
    results_<tag>.csv   one row per run, columns = config.RESULT_COLUMNS
    details_<tag>.json  per run: chosen cameras [row, col, direction] and a downsampled
                        best-so-far history (for heatmaps and convergence charts)

The runner only calls functions from interfaces.py, so it keeps working when the stubs
are replaced by the real modules.

Fair comparison rules
  * every algorithm gets the same evaluation budget on a given map and K:
    --budget N, else --budget-factor F * K * n_candidates, else config.EVAL_BUDGET,
    else config.BUDGET_FACTOR * K * n_candidates (greedy needs about 1.0 * K * n_candidates)
  * the optimiser is scored with the weighting under test (uniform / predicted / true)
  * the FINAL placement is always re-scored with the TRUE weights, so the three
    weightings can be compared on the same yardstick
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config
import interfaces as I


def _downsample(history, max_points=100):
    """Keep at most max_points of the best-so-far curve (always including the last)."""
    n = len(history)
    if n <= max_points:
        idx = list(range(n))
    else:
        idx = sorted({round(i * (n - 1) / (max_points - 1)) for i in range(max_points)})
    return {"n": n, "idx": idx, "best": [history[i] for i in idx]}


def run_experiments(cfg=config, maps=None, weightings=None, algorithms=None, k_values=None,
                    n_seeds=None, budget=None, budget_factor=None, tag="run", out_dir=None,
                    verbose=True):
    """Run the full grid and return a DataFrame with columns cfg.RESULT_COLUMNS."""
    maps = maps or cfg.MAP_NAMES
    weightings = weightings or cfg.WEIGHTINGS
    algorithms = algorithms or cfg.ALGORITHMS
    k_values = k_values or cfg.K_VALUES
    n_seeds = n_seeds if n_seeds is not None else cfg.N_SEEDS
    deterministic = set(getattr(cfg, "DETERMINISTIC_ALGORITHMS", ["greedy"]))
    out_dir = Path(out_dir or cfg.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, details = [], {}
    t_start = time.perf_counter()
    for name in maps:
        gmap = I.build_map(name)
        true_scorer = I.Scorer(gmap, weights=I.true_priority(gmap))  # the yardstick
        for weighting in weightings:
            scorer = I.Scorer(gmap, weights=I.get_weights(gmap, weighting))
            for K in k_values:
                # same budget for every algorithm on this map and K
                factor = budget_factor if budget_factor is not None else cfg.BUDGET_FACTOR
                run_budget = (budget or (None if budget_factor is not None else cfg.EVAL_BUDGET)
                              or max(1, int(factor * K * scorer.n_candidates)))
                for algo in algorithms:
                    seeds = [0] if algo in deterministic else range(n_seeds)
                    for i in seeds:
                        seed = cfg.BASE_SEED + i
                        try:
                            res = I.run(algo, scorer, K, run_budget, seed)
                        except Exception as e:
                            raise RuntimeError(
                                f"run failed: algorithm={algo} map={name} K={K} "
                                f"weighting={weighting} seed={seed}") from e
                        rows.append([algo, name, K, weighting, seed,
                                     true_scorer.score(res.placement),
                                     true_scorer.raw_coverage(res.placement),
                                     res.runtime, res.evaluations])
                        details[f"{algo}|{name}|{K}|{weighting}|{seed}"] = {
                            "placement": [list(scorer.candidates[j]) for j in res.placement],
                            "history": _downsample(res.history),
                        }
                if verbose:
                    print(f"[{time.perf_counter() - t_start:7.1f}s] {name} {weighting} K={K} done "
                          f"({len(rows)} runs so far, budget {run_budget} evaluations each)", flush=True)
        # checkpoint after every map so a crash does not lose everything
        _save(rows, details, cfg, out_dir, tag)
        if verbose:
            print(f"[{time.perf_counter() - t_start:7.1f}s] finished map {name} "
                  f"({len(rows)} runs so far)")
    return pd.DataFrame(rows, columns=cfg.RESULT_COLUMNS)


def _save(rows, details, cfg, out_dir, tag):
    pd.DataFrame(rows, columns=cfg.RESULT_COLUMNS).to_csv(out_dir / f"results_{tag}.csv", index=False)
    (out_dir / f"details_{tag}.json").write_text(json.dumps(details))


def summarize(df):
    """Mean true-weighted score per algorithm and weighting (quick sanity view)."""
    return (df.groupby(["algorithm", "weighting"])["score"].mean().unstack().round(3))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="tiny smoke test")
    p.add_argument("--maps", nargs="+")
    p.add_argument("--weightings", nargs="+")
    p.add_argument("--algorithms", nargs="+")
    p.add_argument("--K", nargs="+", type=int, dest="k_values")
    p.add_argument("--seeds", type=int, dest="n_seeds")
    p.add_argument("--budget", type=int)
    p.add_argument("--budget-factor", type=float, dest="budget_factor",
                   help="evaluations per run = F * K * n_candidates (default: config.BUDGET_FACTOR)")
    p.add_argument("--tag", default=None)
    a = p.parse_args()

    kw = dict(maps=a.maps, weightings=a.weightings, algorithms=a.algorithms,
              k_values=a.k_values, n_seeds=a.n_seeds, budget=a.budget,
              budget_factor=a.budget_factor)
    if a.quick:
        kw.update(maps=a.maps or ["tiny"], k_values=a.k_values or [2, 3],
                  n_seeds=a.n_seeds or 2, budget=a.budget or 50)
    tag = a.tag or ("quick" if a.quick else "run")
    df = run_experiments(tag=tag, **kw)
    print(f"\n{len(df)} runs saved to {config.RESULTS_DIR}/results_{tag}.csv\n")
    print(summarize(df))


if __name__ == "__main__":
    main()