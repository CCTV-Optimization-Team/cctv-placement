# Search Algorithms: CCTV Camera Placement (Person 2)

Part of the CCTV camera placement project. This module chooses **K cameras** from a list of
candidate placements so that the **priority-weighted share of covered cells is as high as
possible**, and compares five search strategies under the same evaluation budget.

| Algorithm name | What it does |
| --- | --- |
| `random` | Samples random K-sets until the budget runs out, keeps the best |
| `greedy` | Adds one camera at a time, each time the one that adds the most new weighted coverage |
| `hill_climb` | One steepest-ascent climb from a random start (stops at a local maximum) |
| `hill_climb_first` | First-improvement climb: accepts the first random swap that helps |
| `hill_climb_restarts` | Repeated steepest-ascent climbs from random starts until the budget runs out |
| `annealing` | Simulated annealing; accepts worse moves with a probability that shrinks as it cools |

The module never computes line of sight. It works only on **precomputed visibility masks**.

---

## Files

| File | Purpose |
| --- | --- |
| `algorithms.py` | The module: all algorithms, `run()`, `run_equal_budget()`, logging helpers |
| `stub_score.py` | Stand-in for the visibility model (plain square around each camera, ignores walls and direction) |
| `test_algorithms.py` | 11 tests (`python test_algorithms.py`, or `pytest`) |
| `run_stub_demo.py` | Runs everything on the stub and writes CSVs and a convergence chart to `results/` |

Requirements: Python 3 and NumPy. The demo and the helper tables also use pandas and Matplotlib.

---

## Quick start

```bash
python test_algorithms.py      # should print "11 tests passed"
python run_stub_demo.py        # writes results/stub_demo.csv, stub_convergence.csv, stub_convergence.png
```

```python
from algorithms import run, run_equal_budget, evaluate_placement, greedy_cost

# masks: bool array (N, H, W) or (N, H*W), one visibility mask per candidate
r = run("greedy", map=None, K=4, weights=None, budget=None, seed=0, masks=masks,
        map_name="campus", weighting="uniform")

r.placement      # [12, 87, 140, 33]  -> indices into the candidate list
r.score          # weighted coverage in [0, 1]
r.raw_coverage   # covered cells / cells with weight > 0
r.runtime        # seconds
r.evaluations    # score evaluations used (<= budget)
r.history        # best score after each evaluation (length = budget)
r.row()          # one results-CSV row (see below)
```

Compare several algorithms fairly (same budget, same seed):

```python
out = run_equal_budget(["random", "greedy", "hill_climb_restarts", "annealing"],
                       K=4, weights=w, budget=5 * greedy_cost(masks.shape[0], 4),
                       seed=0, masks=masks, map_name="campus", weighting="uniform",
                       params={"annealing": {"T0": 0.02, "T_end": 1e-4}})
```

---

## Input and output with other people

This module talks to the rest of the team **only through files and the agreed function
shapes**, never by importing or editing someone else's code.

```
 Person 4 ──maps/<n>.json──▶ Person 1 ──masks/<n>.npz──▶ ┌────────────┐
                                                         │  Person 2  │──results rows──▶ Person 4 (runner)
 Person 3 ──priority/<n>_{true,pred}.npy── (via P4) ───▶ │ algorithms │──convergence───▶ Person 4 (charts)
                                                         └────────────┘──run("greedy")─▶ Person 1 (sensitivity)
```

### What this module receives

| From | What | Format | How it enters `run()` |
| --- | --- | --- | --- |
| **Person 1** (Visibility) | Candidate list and one boolean visibility mask per candidate | `masks/<n>.npz` | `masks=` argument, shape `(N, H, W)` or `(N, H*W)`. Candidate `i` is row `i` of the array |
| **Person 3** (ML) | Priority weight per cell: true map and ML-predicted map | `priority/<n>_true.npy`, `priority/<n>_pred.npy`, shape `(H, W)` | `weights=` argument. Uniform weights = `weights=None` |
| **Person 4** (Maps) | Map grid, entrances, zones | `maps/<n>.json` | Only needed if `masks` is omitted: then `visibility.build_candidates(map, config)` is called |

Loading the real inputs (adjust the array names to match the key names Person 1 saved in the `.npz`):

```python
import numpy as np
data = np.load("masks/campus.npz")
masks = data["masks"]                       # <- key name to confirm with Person 1
w_true = np.load("priority/campus_true.npy")
w_pred = np.load("priority/campus_pred.npy")
```

### What this module gives out

| To | What | Format |
| --- | --- | --- |
| **Person 4** (experiment runner) | One row per run | `r.row()` gives `algorithm, map, K, weighting, seed, score, runtime, evaluations`. Build a table with `results_table(list_of_results)` and write it to `results/<run>.csv` |
| **Person 4** (charts) | Convergence data, best score vs evaluations | `save_history_csv(results, path)`: one column per run, one row per evaluation. Or use `r.history` directly |
| **Person 1** (sensitivity study, task 1.10) | Greedy as a function | Build masks for each (range, field of view) setting, call `run("greedy", ..., masks=masks)` once per setting. Greedy is deterministic, so one run is enough |
| **Person 3** (ML table, task 3.7) | Placements to score under another weighting | `evaluate_placement(r.placement, masks, w_true)` |

### Rules for scoring ML placements (Persons 3 and 4)

1. **Search** with the weighting being tested (`uniform`, `pred` or `true`).
2. **Report** every final placement under the **true** priority map:
   `evaluate_placement(r.placement, masks, w_true)`. The score `r.score` is under the weights used in the search, so for `pred` it is not the number to report.
3. `weighting` in the results CSV is just a label you pass in; the module does not interpret it.

---

## Conventions every caller must follow

- **Weights should be 0 on wall cells.** Otherwise "raw coverage" is measured over walls as well. Weights are normalised internally, so any positive scale works.
- **Candidates are referred to by index.** A placement is a list of K indices. Keep the candidate list from Person 1 so indices can be turned back into (row, column, direction).
- **Masks must be boolean** and all candidates must share one grid size.
- **Every run needs a `seed`.** Same seed, same result. Greedy is deterministic, so run it once per map, K and weighting. Use 20 or more seeds for the other algorithms.
- **Budget = number of score evaluations**, not seconds. Pass the same `budget` to every algorithm you compare.

### What counts as one evaluation

- `random`, hill climbing, annealing: scoring one full K-camera placement.
- `greedy`: scoring one "current set plus one candidate" option. One full run costs `K*N - K(K-1)/2` evaluations, available as `greedy_cost(N, K)`. With `budget=None`, `run()` uses exactly this.
- `r.history[i]` is the best score seen after `i+1` evaluations. For greedy it includes partial sets of fewer than K cameras. If an algorithm stops early (hill climbing at a local maximum) the history is padded with its last value.

### Choosing the budget

One steepest-ascent step costs about `K*(N-K)` evaluations, roughly one whole greedy run. On the stub map, at a budget equal to one greedy run, `hill_climb_restarts` loses to `random`; at 5 times that budget it matches greedy. Use a budget of at least a few times `greedy_cost` when comparing local search with greedy, and say which budget you used in the report.

### Algorithm settings

| Setting | Algorithm | Default | Meaning |
| --- | --- | --- | --- |
| `T0` | `annealing` | 0.02 | Start temperature, in score units (scores lie in [0, 1]) |
| `T_end` | `annealing` | 1e-4 | End temperature; cooling is geometric across the whole budget |
| `patience` | `hill_climb_first` | `min(K*(N-K), 1000)` | Consecutive failed swaps before it stops |

The annealing defaults are placeholders until tuning on the real maps (task 2.7).

---

## Current status

- Tested on the stub score only (tiny 20 by 20 map, plain-square visibility). Not yet run on Person 1's real masks.
- Switching to real masks needs **no change to `algorithms.py`**: pass the real `masks` array to `run()`.
- Open item: confirm the key names inside Person 1's `masks/<n>.npz` and the shape of Person 3's priority arrays.

## Contact

Person 2, Search algorithms. Message me if the mask, weight or results-CSV format you are using differs from what is described above.
