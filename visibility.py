
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

    if "name" not in m:
        m["name"] = os.path.splitext(os.path.basename(path))[0]

    return m


def load_map_by_name(map_name):
    """Load a map by its name from the maps directory."""
    maps_dir = getattr(config, "MAPS_DIR", "maps")
    map_path = os.path.join(maps_dir, f"{map_name}.json")

    if not os.path.exists(map_path):
        project_maps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "maps",
        )
        map_path = os.path.join(project_maps_dir, f"{map_name}.json")

    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"Map '{map_name}' not found. Expected: {map_path}"
        )

    return load_map(map_path)


# ---------------------------------------------------------------------------
# Line of sight
# ---------------------------------------------------------------------------

def bresenham(r0, c0, r1, c1):
    """Return cells along a line, including both endpoints."""
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
    """Return True if no wall blocks cells strictly between endpoints."""
    if b < a:
        a, b = b, a

    intermediate = bresenham(
        a[0], a[1], b[0], b[1]
    )[1:-1]

    for r, c in intermediate:
        if grid[r, c] == 1:
            return False

    return True


# ---------------------------------------------------------------------------
# Visible cells
# ---------------------------------------------------------------------------

def visible_cells_from(grid, r, c, cam_range):
    """Return open cells within Euclidean range and line of sight.

    Returns:
        List of (row, col, angle_deg).

    Geometric angle convention:
        0   = East
        90  = North
        180 = West
        270 = South
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
                angle = math.degrees(
                    math.atan2(-(rr - r), cc - c)
                ) % 360

                out.append((rr, cc, angle))

    return out


def angle_diff(a, b):
    """Smallest wrap-safe difference between two angles."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def build_candidates(grid, n_directions=None):
    """Generate camera candidates for every open cell and direction.

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

    return np.asarray(
        candidates,
        dtype=np.int32,
    ).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Visibility masks
# ---------------------------------------------------------------------------

def build_masks(
    grid,
    cam_range=None,
    fov_deg=None,
    n_directions=None,
):
    """Build boolean visibility masks.

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

    candidates = build_candidates(
        grid,
        n_directions,
    )

    masks = np.zeros(
        (len(candidates), rows, cols),
        dtype=bool,
    )

    half_fov = fov_deg / 2.0
    angle_step = 360.0 / n_directions

    # Visibility is direction-independent; cache by camera cell.
    cache = {}

    for i, (r, c, d) in enumerate(candidates):
        r, c, d = int(r), int(c), int(d)

        key = (r, c)

        if key not in cache:
            cache[key] = visible_cells_from(
                grid,
                r,
                c,
                cam_range,
            )

        # North is 90 degrees in the geometric angle convention.
        # Camera directions increase clockwise.
        facing = (90.0 - d * angle_step) % 360

        for rr, cc, angle in cache[key]:

            # Camera's own cell is always visible.
            if (rr, cc) == key:
                masks[i, rr, cc] = True
                continue

            if angle_diff(angle, facing) <= half_fov + 1e-9:
                masks[i, rr, cc] = True

    return candidates, masks


# ---------------------------------------------------------------------------
# Save visibility masks
# ---------------------------------------------------------------------------

def save_masks(
    path,
    grid,
    candidates,
    masks,
    cam_range=None,
    fov_deg=None,
    n_directions=None,
):
    """Save masks in the agreed NPZ format.

    NPZ keys:
        candidates
        masks
    """
    parent = os.path.dirname(os.fspath(path))

    if parent:
        os.makedirs(parent, exist_ok=True)

    np.savez_compressed(
        path,
        candidates=candidates,
        masks=masks.astype(bool, copy=False),
    )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class Scorer:
    """Fast weighted coverage scorer using precomputed visibility masks.

    Supported interfaces:
        Scorer(gmap_dict, weights=None)
        Scorer("tiny", weights=None, masks_dir=None)

    Methods:
        score(selected)
        raw_coverage(selected)
        gain(covered, candidate_index)
        covered_mask(selected)
        covered_grid(selected)
    """

    def __init__(self, gmap, weights=None, masks_dir=None):

        self.gmap = gmap

        # Resolve map input
        if isinstance(gmap, str):
            loaded_map = load_map_by_name(gmap)

            self.name = loaded_map["name"]
            self.grid = np.asarray(loaded_map["grid"])

        elif isinstance(gmap, dict):
            self.name = gmap.get("name")
            self.grid = np.asarray(gmap["grid"])

        else:
            self.name = getattr(gmap, "name", None)
            self.grid = np.asarray(gmap.grid)

        if self.name is None:
            raise ValueError("Map must have a name.")

        # Grid properties
        self.rows, self.cols = self.grid.shape

        self.open_cells = self.grid == 0
        self.n_open_cells = int(self.open_cells.sum())

        # Resolve masks directory
        if masks_dir is None:
            masks_dir = config.MASKS_DIR

        mask_path = os.path.join(
            os.fspath(masks_dir),
            f"{self.name}.npz",
        )

        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Visibility masks not found: {mask_path}. "
                "Run the mask precomputation script first."
            )

        # Load masks
                # Load masks
        with np.load(mask_path) as data:
            candidate_array = data["candidates"]
            self.masks = data["masks"]

        # Contract (interfaces.py): candidates is a list of (row, col, direction)
        # tuples of plain ints, so it can be dumped to JSON.
        self.candidates = [tuple(int(v) for v in row) for row in candidate_array]
        
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

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def set_weights(self, weights=None):
        """Set priority weights as a 2D grid; walls always get zero."""

        if weights is None:
            self.weights = self.open_cells.astype(float)

        else:
            weights = np.asarray(weights, dtype=float)

            if weights.shape != self.grid.shape:
                raise ValueError(
                    f"Weights shape mismatch. Expected {self.grid.shape}, "
                    f"got {weights.shape}"
                )

            if np.any(weights < 0):
                raise ValueError("Weights cannot be negative.")

            if not np.all(np.isfinite(weights)):
                raise ValueError("Weights must be finite.")

            self.weights = weights.copy()
            self.weights[~self.open_cells] = 0.0

        self.total = float(self.weights.sum())

    # ------------------------------------------------------------------
    # Coverage mask
    # ------------------------------------------------------------------

    def covered_mask(self, selected):
        """Return a 2D boolean coverage grid."""

        covered = np.zeros(
            (self.rows, self.cols),
            dtype=bool,
        )

        if len(selected) == 0:
            return covered

        selected_masks = self.masks[list(selected)]

        covered = np.any(
            selected_masks,
            axis=0,
        )

        return covered

    # ------------------------------------------------------------------
    # Weighted coverage score
    # ------------------------------------------------------------------

    def score(self, selected):
        """Return weighted coverage score in [0, 1]."""

        self.evaluations += 1

        if self.total == 0:
            return 0.0

        covered = self.covered_mask(selected)

        # Walls must never contribute to the score.
        covered &= self.open_cells

        return float(
            self.weights[covered].sum() / self.total
        )

    # ------------------------------------------------------------------
    # Raw coverage
    # ------------------------------------------------------------------

    def raw_coverage(self, selected):
        """Return fraction of open cells covered, ignoring weights."""

        self.evaluations += 1

        if self.n_open_cells == 0:
            return 0.0

        covered = self.covered_mask(selected)

        # Exclude walls.
        covered &= self.open_cells

        return float(
            covered.sum() / self.n_open_cells
        )

    # ------------------------------------------------------------------
    # Marginal gain
    # ------------------------------------------------------------------

    def gain(self, covered, c):
        """Return additional weighted coverage from candidate c.

        covered may be either a flattened array or a 2D grid.
        """

        covered = np.asarray(
            covered,
            dtype=bool,
        ).reshape(
            self.rows,
            self.cols,
        )

        candidate_mask = self.masks[c]

        new_cells = (
            candidate_mask
            & ~covered
            & self.open_cells
        )

        if self.total == 0:
            return 0.0

        return float(
            self.weights[new_cells].sum() / self.total
        )

    # ------------------------------------------------------------------
    # Covered grid
    # ------------------------------------------------------------------

    def covered_grid(self, selected):
        """Return a (rows, cols) boolean coverage grid."""
        return self.covered_mask(selected)
