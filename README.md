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

The tiny map (5 × 5) contains 184 camera candidates. Mask generation was benchmarked over three runs, with a median execution time of approximately 0.0065 seconds.

### Current Test Status

17 tests passed across visibility, scorer, priority-weight, and edge-case test modules.
