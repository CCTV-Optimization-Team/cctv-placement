"""Person 1: visibility model.

Pipeline:  map JSON -> candidates (row, col, direction) -> one boolean mask per
candidate -> Scorer (fast weighted-coverage score from stored masks).

Cell index convention everywhere: idx = r * cols + c.
Direction convention: 0 = east, counter-clockwise on screen, 2 = north (row decreases).
"""
import json
import math
import os

import numpy as np

import config


# ---------------------------------------------------------------- map loading
def load_map(path):
    """Load maps/<name>.json -> dict with numpy grid (0 open, 1 wall)."""
    with open(path) as f:
        m = json.load(f)
    m["grid"] = np.array(m["grid"], dtype=np.uint8)
    m["rows"], m["cols"] = m["grid"].shape
    return m


# ------------------------------------------------------------- line of sight
def bresenham(r0, c0, r1, c1):
    """All grid cells on the line from (r0,c0) to (r1,c1), both ends included."""
    cells = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
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
    """True if no wall lies strictly between cell a and cell b.

    Symmetric: the line is always traced from the lexicographically smaller
    cell, so has_line_of_sight(a, b) == has_line_of_sight(b, a).
    """
    if b < a:
        a, b = b, a
    for (r, c) in bresenham(a[0], a[1], b[0], b[1])[1:-1]:
        if grid[r, c] == 1:
            return False
    return True


# ------------------------------------------------------------- visible cells
def visible_cells_from(grid, r, c, cam_range):
    """Open cells within cam_range (Euclidean) with clear line of sight.
    Independent of direction, so it is computed once per camera cell.
    Returns list of (r, c, angle_deg) with angle in [0, 360), 0 = east,
    counter-clockwise on screen (row decreases = up).
    """
    rows, cols = grid.shape
    out = [(r, c, 0.0)]  # own cell is always visible
    R = int(math.floor(cam_range))
    for rr in range(max(0, r - R), min(rows, r + R + 1)):
        for cc in range(max(0, c - R), min(cols, c + R + 1)):
            if (rr, cc) == (r, c) or grid[rr, cc] == 1:
                continue
            if (rr - r) ** 2 + (cc - c) ** 2 > cam_range ** 2:
                continue
            if has_line_of_sight(grid, (r, c), (rr, cc)):
                # screen "up" is a decreasing row, so negate the row difference
                ang = math.degrees(math.atan2(-(rr - r), cc - c)) % 360
                out.append((rr, cc, ang))
    return out


def angle_diff(a, b):
    """Smallest absolute difference between two angles in degrees (wrap-safe)."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


# ----------------------------------------------------------------- candidates
def build_candidates(grid, n_directions=None):
    """Every open cell x every direction -> int array (N, 3): row, col, dir."""
    n_directions = n_directions or config.N_DIRECTIONS
    rs, cs = np.nonzero(grid == 0)
    cands = [(r, c, d) for r, c in zip(rs, cs) for d in range(n_directions)]
    return np.array(cands, dtype=np.int32)


def build_masks(grid, cam_range=None, fov_deg=None, n_directions=None):
    """Returns (candidates (N,3), masks (N, rows*cols) bool)."""
    cam_range = cam_range if cam_range is not None else config.CAMERA_RANGE
    fov_deg = fov_deg if fov_deg is not None else config.FOV_DEG
    n_directions = n_directions or config.N_DIRECTIONS
    rows, cols = grid.shape
    step = 360.0 / n_directions
    half = fov_deg / 2.0

    candidates = build_candidates(grid, n_directions)
    masks = np.zeros((len(candidates), rows * cols), dtype=bool)

    # cache per camera cell: line of sight does not depend on direction
    cache = {}
    for i, (r, c, d) in enumerate(candidates):
        key = (int(r), int(c))
        if key not in cache:
            cache[key] = visible_cells_from(grid, key[0], key[1], cam_range)
        facing = d * step
        for (rr, cc, ang) in cache[key]:
            if (rr, cc) == key or angle_diff(ang, facing) <= half + 1e-9:
                masks[i, rr * cols + cc] = True
    return candidates, masks


def save_masks(path, grid, candidates, masks, cam_range, fov_deg, n_directions):
    rows, cols = grid.shape
    np.savez_compressed(
        path,
        candidates=candidates,
        masks=masks,
        open_cells=(grid == 0).ravel(),
        meta=np.array([rows, cols, cam_range, fov_deg, n_directions], dtype=float),
    )


# --------------------------------------------------------------------- Scorer
class Scorer:
    """Fast weighted coverage from precomputed masks.

    scorer = Scorer("hall", weights)          # loads masks/hall.npz
    scorer.score([12, 407, 3051])             # candidate indices -> 0..1
    scorer.gain(covered, c)                   # extra weighted coverage of candidate c
    scorer.covered_grid([12, 407])            # (rows, cols) bool, for heatmaps
    weights: None (uniform over open cells) or array of shape (rows, cols).
    """

    def __init__(self, name, weights=None, masks_dir=None):
        masks_dir = masks_dir or config.MASKS_DIR
        data = np.load(os.path.join(masks_dir, f"{name}.npz"))
        self.candidates = data["candidates"]
        self.masks = data["masks"]
        self.open_cells = data["open_cells"]
        self.rows, self.cols = int(data["meta"][0]), int(data["meta"][1])
        self.set_weights(weights)
        self.n_evals = 0

    def set_weights(self, weights):
        if weights is None:
            w = self.open_cells.astype(float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            assert w.size == self.rows * self.cols, "weights must be (rows, cols)"
            w = np.where(self.open_cells, w, 0.0)  # walls are never scored
        assert (w >= 0).all(), "weights must be non-negative"
        assert np.isfinite(w).all(), "weights must contain only finite values"
        self.weights = w
        self.total = w.sum()

    @property
    def n_candidates(self):
        return len(self.candidates)

    def covered_mask(self, selected):
        if len(selected) == 0:
            return np.zeros(self.rows * self.cols, dtype=bool)
        return np.logical_or.reduce(self.masks[list(selected)], axis=0)

    def score(self, selected):
        """Weighted coverage in 0..1; a cell seen twice counts once."""
        self.n_evals += 1
        if self.total == 0:
            return 0.0
        return float(self.weights[self.covered_mask(selected)].sum() / self.total)

    def gain(self, covered, c):
        """Extra weighted coverage from adding candidate c to a boolean covered array."""
        new = self.masks[c] & ~covered
        return float(self.weights[new].sum() / self.total) if self.total else 0.0

    def covered_grid(self, selected):
        return self.covered_mask(selected).reshape(self.rows, self.cols)