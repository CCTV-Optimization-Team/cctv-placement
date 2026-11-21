# CCTV Camera Placement

Priority-aware CCTV camera placement using search algorithms and machine learning.

## Modules

| File | Owner |
|---|---|
| `map.py`, `maps/`, experiment runner, charts, report | Person 4 |
| `visibility.py` | Person 1 |
| `algorithms.py` | Person 2 |
| `ml_priority.py` | Person 3 |

## Conventions

- Cells are `(row, col)`, zero-indexed, row 0 is the top.
- Grid: `0 = open`, `1 = wall`.
- Direction: `0–7`, where 0 = North, clockwise.
- Weights: `(rows, cols)` float array, walls = 0.

## Workflow

One branch per person (`p<number>/<topic>`), merge through Pull Requests. No direct pushes to `main`.

## Setup

```bash
pip install -r requirements.txt
```

## Visibility Model (Person 1)

The visibility model generates possible CCTV camera locations and orientations, calculates line-of-sight visibility, and precomputes visibility masks for efficient camera placement.

### Main Components

- **Candidate Generation:** Generates camera candidates for open grid cells across all supported directions.
- **Line of Sight:** Uses Bresenham's line algorithm to determine whether obstacles block visibility between cells.
- **Visibility Masks:** Represents the visible cells for each camera candidate as a Boolean mask.
- **Mask Precomputation:** Saves candidate locations and visibility masks as `.npz` files for efficient reuse.
- **Weighted Scorer:** Calculates coverage scores using uniform cell weights or externally supplied priority weights.

### Scorer Interface

```python
from visibility import Scorer

# Uniform coverage
scorer = Scorer("tiny")

# Weighted coverage
scorer = Scorer("tiny", weights=priority_weights)

# Evaluate selected cameras
score = scorer.score(selected_candidates)

# Calculate additional coverage
gain = scorer.gain(covered, candidate_index)

# Get the covered grid
covered_grid = scorer.covered_grid(selected_candidates)
```

### Running Tests

```powershell
python -m pytest -v
```

### Performance Benchmark

The tiny map currently contains 20 × 20 cells. The number of camera candidates depends on the map dimensions, open cells, and supported camera orientations. Masks are precomputed and saved for reuse.

### Current Test Status

17 tests passed across visibility, scorer, priority-weight, and edge-case test modules.