# cctv-placement
Priority-aware CCTV camera placement using search algorithms and machine learning
## Modules
| File | Owner |
|---|---|
| map.py, maps/, experiment runner, charts, report | Person 4 |
| visibility.py | Person 1 |
| algorithms.py | Person 2 |
| ml_priority.py | Person 3 |

## Conventions
- Cells are (row, col), zero-indexed, row 0 is the top
- Grid: 0 = open, 1 = wall
- Direction: 0..7, 0 = up, clockwise
- Weights: (rows, cols) float array, walls = 0

## Workflow
One branch per person (`p<number>/<topic>`), merge through Pull Requests. No direct pushes to `main`.

## Setup
`pip install -r requirements.txt`