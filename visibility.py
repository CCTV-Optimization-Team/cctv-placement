
"""Person 1: visibility model.

Pipeline:
    map JSON -> candidates (row, col, direction)
    -> boolean masks -> Scorer (fast weighted coverage)

Direction convention:
    0 = North (row decreases)
    directions increase clockwise

Cell index convention:
    idx = r * cols + c
"""

import json
import math
import os

import numpy as np

import config


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

def load_map(path):
    """Load a JSON map and return its data with a NumPy grid."""
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)

    m["grid"] = np.array(m["grid"], dtype=np.uint8)
    m["rows"], m["cols"] = m["grid"].shape
    return m


# ---------------------------------------------------------------------------
# Line of sight
# ---------------------------------------------------------------------------

def bresenham(r0, c0, r1, c1):
    """Return all cells on a line between two points, including endpoints."""
    cells = []

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)

    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1

    err = dc - dr
    r, c = r0, c0

    while True:
        cells.append((r, c))

        if r == r1 and c == c1:
            break

        e2 = 2 * err

        if e2 > -dr:
            err -= dr
            c += sc

        if e2 < dc:
            err += dc
            r += sr

    return cells


def has_line_of_sight(grid, a, b):
    """Return True if no wall blocks the cells strictly between a and b.

    The line is traced from the lexicographically smaller endpoint to ensure
    symmetric results.
    """
    if b < a:
        a, b = b, a

    intermediate = bresenham(a[0], a[1], b[0], b[1])[1:-1]

    for r, c in intermediate:
        if grid[r, c] == 1:
            return False

    return True


# ---------------------------------------------------------------------------
# Visible cells
# ---------------------------------------------------------------------------

def visible_cells_from(grid, r, c, cam_range):
    """Return open cells within Euclidean camera range and LOS.

    Returns:
        List of (row, col, angle_deg).

    Angle convention for geometry:
        0 degrees = East
        90 degrees = North
        180 degrees = West
        270 degrees = South

    This angle convention is only used for geometric calculations.
    Candidate directions follow config.DIRECTION_VECTORS.
    """
    rows, cols = grid.shape

    # Camera's own cell is always visible.
    out = [(r, c, 0.0)]

    R = int(math.ceil(cam_range))

    for rr in range(max(0, r - R), min(rows, r + R + 1)):
        for cc in range(max(0, c - R), min(cols, c + R + 1)):

            if (rr, cc) == (r, c):
                continue

            # Walls cannot be visible cells.
            if grid[rr, cc] == 1:
                continue

            # Euclidean range constraint.
            if (rr - r) ** 2 + (cc - c) ** 2 > cam_range ** 2:
                continue

            if has_line_of_sight(grid, (r, c), (rr, cc)):
                # Row decreases when moving upwards.
                angle = math.degrees(
                    math.atan2(-(rr - r), cc - c)
                ) % 360

                out.append((rr, cc, angle))

    return out


def angle_diff(a, b):
    """Return the smallest wrap-safe difference between two angles."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def build_candidates(grid, n_directions=None):
    """Generate candidates for every open cell and camera direction.

    Candidate format:
        (row, col, direction)

    Direction convention:
        0 = North
        1 = North-East
        2 = East
        3 = South-East
        4 = South
        5 = South-West
        6 = West
        7 = North-West
    """
    if n_directions is None:
        n_directions = config.N_DIRECTIONS

    rs, cs = np.nonzero(grid == 0)

    candidates = [
        (int(r), int(c), d)
        for r, c in zip(rs, cs)
        for d in range(n_directions)
    ]

    return np.asarray(candidates, dtype=np.int32).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Visibility masks
# ---------------------------------------------------------------------------

def build_masks(
    grid,
    cam_range=None,
    fov_deg=None,
    n_directions=None,
):
    """Build visibility masks.

    Returns:
        candidates: (N, 3)
        masks: (N, rows, cols), bool

    Direction 0 = North, increasing clockwise.
    """
    if cam_range is None:
        cam_range = config.CAMERA_RANGE

    if fov_deg is None:
        fov_deg = config.FOV_DEGREES

    if n_directions is None:
        n_directions = config.N_DIRECTIONS

    rows, cols = grid.shape

    candidates = build_candidates(grid, n_directions)

    masks = np.zeros(
        (len(candidates), rows, cols),
        dtype=bool,
    )

    half_fov = fov_deg / 2.0
    angle_step = 360.0 / n_directions

    # Visibility independent of direction, so cache by camera cell.
    cache = {}

    for i, (r, c, d) in enumerate(candidates):
        r, c, d = int(r), int(c), int(d)

        key = (r, c)

        if key not in cache:
            cache[key] = visible_cells_from(
                grid, r, c, cam_range
            )

        # North is 90 degrees in our geometric angle convention.
        # Directions increase clockwise.
        facing = (90.0 - d * angle_step) % 360

        for rr, cc, angle in cache[key]:

            # Camera's own cell is always visible.
            if (rr, cc) == key:
                masks[i, rr, cc] = True
                continue

            if angle_diff(angle, facing) <= half_fov + 1e-9:
                masks[i, rr, cc] = True

    return candidates, masks


def save_masks(
    path,
    grid,
    candidates,
    masks,
    cam_range=None,
    fov_deg=None,
    n_directions=None,
):
    """Save masks in the agreed project format.

    NPZ keys:
        candidates
        masks
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    np.savez_compressed(
        path,
        candidates=candidates,
        masks=masks.astype(bool, copy=False),
    )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class Scorer:
    """Fast weighted coverage Scorer using precomputed visibility masks.

    Usage:
        scorer = Scorer(gmap, weights=None)
        scorer.score(selected)
        scorer.raw_coverage(selected)
        scorer.gain(covered, candidate_index)

    Args:
        gmap: map dictionary containing name and grid.
        weights: optional 2D priority-weight array.

    selected:
        List of candidate indices.
    """

    def __init__(self, gmap, weights=None):
        self.gmap = gmap

        # Support the shared map dictionary returned by load_map().
        if isinstance(gmap, dict):
            self.name = gmap.get("name")
            self.grid = np.asarray(gmap["grid"])
        else:
            self.name = getattr(gmap, "name", None)
            self.grid = np.asarray(gmap.grid)

        if self.name is None:
            raise ValueError("Map must have a name.")

        self.rows, self.cols = self.grid.shape
        self.open_cells = (self.grid == 0)
        self.n_open_cells = int(self.open_cells.sum())

        mask_path = os.path.join(
            config.MASKS_DIR,
            f"{self.name}.npz",
        )

        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Visibility masks not found: {mask_path}. "
                "Run the mask precomputation script first."
            )

        with np.load(mask_path) as data:
            self.candidates = data["candidates"]
            self.masks = data["masks"]

        expected_shape = (
            len(self.candidates),
            self.rows,
            self.cols,
        )

        if self.masks.shape != expected_shape:
            raise ValueError(
                f"Mask shape mismatch. Expected {expected_shape}, "
                f"got {self.masks.shape}"
            )

        if self.masks.dtype != bool:
            self.masks = self.masks.astype(bool)

        self.n_candidates = len(self.candidates)
        self.evaluations = 0

        self.set_weights(weights)

    # ---------------------------------------------------------------------

    def set_weights(self, weights=None):
        """Set scoring weights. Wall weights are always ignored."""
        if weights is None:
            w = self.open_cells.astype(float)
        else:
            w = np.asarray(weights, dtype=float)

            if w.shape != (self.rows, self.cols):
                raise ValueError(
                    f"Weights must have shape {(self.rows, self.cols)}, "
                    f"got {w.shape}"
                )

            w = w.copy()
            w[~self.open_cells] = 0.0

        if not np.isfinite(w).all():
            raise ValueError("Weights must contain only finite values.")

        if (w < 0).any():
            raise ValueError("Weights must be non-negative.")

        self.weights = w.ravel()
        self.total = float(self.weights.sum())

    # ---------------------------------------------------------------------

    def covered_mask(self, selected):
        """Return a flattened boolean coverage array."""
        if len(selected) == 0:
            return np.zeros(self.rows * self.cols, dtype=bool)

        selected_masks = self.masks[list(selected)]

        # (num_selected, rows, cols) -> (rows, cols)
        covered = np.any(selected_masks, axis=0)

        return covered.ravel()

    # ---------------------------------------------------------------------

    def score(self, selected):
        """Return weighted coverage in [0, 1]."""
        self.evaluations += 1

        if self.total == 0:
            return 0.0

        covered = self.covered_mask(selected)

        return float(
            self.weights[covered].sum() / self.total
        )

    # ---------------------------------------------------------------------

    def raw_coverage(self, selected):
        """Return fraction of open cells covered, ignoring priority weights."""
        self.evaluations += 1

        if self.n_open_cells == 0:
            return 0.0

        covered = self.covered_mask(selected)

        # Walls are excluded regardless of what masks contain.
        covered &= self.open_cells.ravel()

        return float(covered.sum() / self.n_open_cells)

    # ---------------------------------------------------------------------

    def gain(self, covered, c):
        """Return additional weighted coverage from candidate c.

        covered may be flattened or shaped (rows, cols).
        """
        covered = np.asarray(covered, dtype=bool).reshape(-1)

        if covered.size != self.rows * self.cols:
            raise ValueError("covered array has incorrect size")

        if self.total == 0:
            return 0.0

        new_cells = self.masks[c].ravel() & ~covered
        new_cells &= self.open_cells.ravel()

        return float(
            self.weights[new_cells].sum() / self.total
        )

    # ---------------------------------------------------------------------

    def covered_grid(self, selected):
        """Return (rows, cols) boolean grid of covered cells."""
        return self.covered_mask(selected).reshape(
            self.rows, self.cols
        )
