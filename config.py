"""config.py - shared settings. Change values here, never hard-code them in a module.

Conventions shared by everyone:
  * Cells are (row, col), zero-indexed, row 0 is the TOP row.
  * grid: 0 = open cell, 1 = wall.
  * Candidate = (row, col, direction). direction is an integer 0..N_DIRECTIONS-1,
    0 = facing up (row decreases), increasing CLOCKWISE (see DIRECTION_VECTORS).
  * weights / priority maps are 2-D float arrays with the same shape as the grid,
    walls = 0. Uniform weighting = 1.0 on every open cell.
"""

# ---- camera model (Person 1 owns the meaning, everyone reads) ----
CAMERA_RANGE = 6          # cells
FOV_DEGREES = 90          # total field of view
N_DIRECTIONS = 8          # 8 -> 45 degree steps
# (d_row, d_col) of each direction index, clockwise from up
DIRECTION_VECTORS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# ---- experiments (Person 4 runner, Person 2 algorithms) ----
MAP_NAMES = ["tiny", "college", "metro", "random_1", "random_2", "random_3", "random_4", "random_5"]
K_VALUES = [2, 3, 4, 5, 6]
WEIGHTINGS = ["uniform", "predicted", "true"]
ALGORITHMS = ["random", "greedy", "hill_climbing", "hill_climbing_restarts", "annealing"]
EVAL_BUDGET = 2000        # score evaluations allowed per run, same for every algorithm
N_SEEDS = 20              # seeded runs for the random methods
BASE_SEED = 42

# ---- paths ----
MAPS_DIR = "maps"
MASKS_DIR = "masks"        # masks/<map>.npz        (Person 1 writes)
PRIORITY_DIR = "priority"  # priority/<map>_true.npy, <map>_pred.npy  (Person 3 writes)
RESULTS_DIR = "results"    # results/<run>.csv      (runner writes)

# columns of every results CSV row
RESULT_COLUMNS = ["algorithm", "map", "K", "weighting", "seed",
                  "score", "raw_coverage", "runtime", "evaluations"]
