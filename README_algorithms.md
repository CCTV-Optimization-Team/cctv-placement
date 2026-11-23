# Search algorithms (Person 2)

`algorithms.py` chooses **K cameras** from the candidate list so that the priority-weighted
coverage is as high as possible, using six search strategies compared under the same
evaluation budget. It never computes line of sight: it only calls the scorer (Person 1's
`Scorer`).

| Algorithm | What it does |
| --- | --- |
| `random` | Samples random K-sets until the budget runs out, keeps the best |
| `greedy` | Adds one camera at a time, each time the one that raises the score most |
| `hill_climb` | One steepest-ascent climb from a random start (stops at a local maximum) |
| `hill_climb_first` | First-improvement climb: accepts the first random swap that helps |
| `hill_climb_restarts` | Steepest-ascent climbs from fresh random starts until the budget runs out |
| `annealing` | Simulated annealing: accepts worse moves with a probability that shrinks as it cools |

Algorithm names passed to `run()` come from `config.ALGORITHMS`. Spelling variants such as
`hill-climbing`, `simulated_annealing` or `random restarts` are mapped automatically
(`algorithms.check_names(config.ALGORITHMS)` shows the mapping).

## Interface

```python
from algorithms import run

r = run(algorithm, scorer, K, budget, seed)
```

| Argument | Meaning |
| --- | --- |
| `algorithm` | A name from `config.ALGORITHMS` |
| `scorer` | Person 1's `Scorer(gmap, weights=None)`. Only `.n_candidates` and `.score(selected)` are used |
| `K` | Number of cameras |
| `budget` | Max number of `scorer.score()` calls. `None` = cost of one greedy run |
| `seed` | Int; same seed gives the same result |

It returns a `RunResult`:

| Field | Meaning |
| --- | --- |
| `placement` | List of K candidate indices (index into `scorer.candidates`) |
| `final_score` | Weighted coverage of `placement` under the scorer's weights |
| `history` | Best score so far after each evaluation (list, length = `budget`) |
| `runtime` | Seconds |
| `evaluations` | Evaluations actually used (never more than `budget`) |

`RunResult` is imported from `interfaces.py` if it is defined there; otherwise a local
dataclass with the same five fields is used.

Extra: `run_equal_budget(algorithms, scorer, K, budget, seed, params=None)` runs several
algorithms with the same budget and seed and returns `{name: RunResult}`.
`run(..., T0=..., T_end=..., patience=...)` override the algorithm settings;
tuned values go into `DEFAULT_PARAMS` in `algorithms.py`.

## What counts as one evaluation

- One evaluation is one call to `scorer.score()`, so `RunResult.evaluations` equals the increase in `scorer.evaluations`.
- `random`, hill climbing, annealing: scoring one full K-camera placement.
- `greedy`: scoring "current set plus one candidate". A full run costs `greedy_cost(N, K) = K*N - K(K-1)/2`.
- `history` includes partial sets for greedy. If an algorithm stops early (hill climbing at a local maximum), its history is padded with its last value.
- If `budget` is smaller than greedy's cost, greedy keeps the best option of the unfinished round and fills the rest with the lowest unused indices, so it still returns K cameras.

## Choosing the budget

One steepest-ascent step costs about `K*(N-K)` evaluations, roughly one whole greedy run. On the stub map, at a budget equal to one greedy run `hill_climb_restarts` loses to `random`; at 5 times that budget it matches greedy. Use a budget of a few times `greedy_cost` when comparing local search with greedy, and say which budget was used.

## Notes for other people

- **Person 1:** the algorithms rely on `n_candidates` and `score(selected)` accepting a list of candidate indices of any length from 1 to K (greedy scores partial sets).
- **Person 4:** `RunResult.final_score` is under the weights the scorer was built with. To report a placement under the true priority map, build a second `Scorer(gmap, true_weights)` and call `.score(r.placement)` and `.raw_coverage(r.placement)` on it.
- Greedy is deterministic: one run per map, K and weighting is enough. Use 20+ seeds for the others.

## Files

| File | Purpose |
| --- | --- |
| `algorithms.py` | The module |
| `stub_score.py` | Stub scorer (plain square around each camera, ignores walls) for tests and the demo |
| `test_algorithms.py` | 14 tests: `python test_algorithms.py` |
| `run_stub_demo.py` | Runs everything on the stub and writes `results/stub_*` |

## Status

Tested on the stub scorer. Still to do with real data: run on the real maps (2.5), tune annealing and restarts (2.7), real convergence data (2.8), report section (2.9).
