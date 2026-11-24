
"""Shared project settings.

Conventions:
  * Cells are (row, col), zero-indexed, row 0 is the TOP row.
  * Grid: 0 = open cell, 1 = wall.
  * Candidate = (row, col, direction), direction 0..7, clockwise from North.
  * Weights are 2-D float arrays matching grid shape; walls = 0.
"""

# ---- Camera model (Person 1) ----
CAMERA_RANGE = 6
FOV_DEGREES = 90
N_DIRECTIONS = 8

# (row_delta, col_delta), clockwise from North
DIRECTION_VECTORS = [
    (-1,  0),  # 0: North
    (-1,  1),  # 1: North-East
    ( 0,  1),  # 2: East
    ( 1,  1),  # 3: South-East
    ( 1,  0),  # 4: South
    ( 1, -1),  # 5: South-West
    ( 0, -1),  # 6: West
    (-1, -1),  # 7: North-West
]

# ---- Priority zones (Person 3) ----
ZONE_RADIUS = 2

# ---- Experiments (Person 2 & Person 4) ----
MAP_NAMES = [
    "tiny", "college", "metro",
    "random_1", "random_2", "random_3",
    "random_4", "random_5"
]

K_VALUES = [2, 3, 4, 5, 6]
WEIGHTINGS = ["uniform", "predicted", "true"]

ALGORITHMS = [
    "random",
    "greedy",
    "hill_climbing",
    "hill_climbing_restarts",
    "annealing"
]

DETERMINISTIC_ALGORITHMS = ["greedy"]
BUDGET_FACTOR = 5         # evaluations per run = BUDGET_FACTOR * K * n_candidates (greedy needs ~K * n_candidates)
EVAL_BUDGET = None        # or a fixed number of evaluations for every run (overrides BUDGET_FACTOR)
N_SEEDS = 20
BASE_SEED = 42

# ---- Paths ----
MAPS_DIR = "maps"
MASKS_DIR = "masks"
PRIORITY_DIR = "priority"
RESULTS_DIR = "results"

# ---- Results CSV columns ----
RESULT_COLUMNS = [
    "algorithm",
    "map",
    "K",
    "weighting",
    "seed",
    "score",
    "raw_coverage",
    "runtime",
    "evaluations",
]
