"""map.py  (Person 4) - grid maps for the CCTV placement project.

Conventions (agreed for the whole team):
  * Coordinates are (row, col), zero-indexed, row 0 is the TOP row.
  * grid[row][col]: 0 = open cell, 1 = wall.
  * Entrances, points of interest and priority zones must sit on open cells.
  * Map files live in maps/<name>.json.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

MAPS_DIR = Path(__file__).parent / "maps"


@dataclass
class GridMap:
    name: str
    grid: np.ndarray                                   # (rows, cols) int8, 0 open / 1 wall
    entrances: list = field(default_factory=list)      # [(row, col), ...]
    pois: list = field(default_factory=list)           # [(name, row, col), ...]
    zones: list = field(default_factory=list)          # [(name, row, col, bonus), ...]

    @property
    def rows(self):
        return self.grid.shape[0]

    @property
    def cols(self):
        return self.grid.shape[1]

    def is_open(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0

    def open_cells(self):
        return [tuple(x) for x in np.argwhere(self.grid == 0)]

    def to_dict(self):
        return {
            "name": self.name,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "grid": self.grid.astype(int).tolist(),
            "entrances": [list(e) for e in self.entrances],
            "points_of_interest": [{"name": n, "row": r, "col": c} for n, r, c in self.pois],
            "priority_zones": [{"name": n, "row": r, "col": c, "bonus": b} for n, r, c, b in self.zones],
        }


def validate(m: GridMap):
    """Raise ValueError if the map breaks the shared rules. Returns m."""
    if m.grid.ndim != 2 or not set(np.unique(m.grid)) <= {0, 1}:
        raise ValueError(f"{m.name}: grid must be 2-D with only 0/1")
    points = [("entrance", "", r, c) for r, c in m.entrances]
    points += [("poi", n, r, c) for n, r, c in m.pois]
    points += [("zone", n, r, c) for n, r, c, _ in m.zones]
    for kind, n, r, c in points:
        if not m.is_open(r, c):
            raise ValueError(f"{m.name}: {kind} {n} at ({r},{c}) is not an open cell")
    if not m.entrances:
        raise ValueError(f"{m.name}: needs at least one entrance")
    # every open cell must be reachable from the first entrance (4-connected BFS)
    start = m.entrances[0]
    seen, queue = {start}, [start]
    while queue:
        r, c = queue.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt not in seen and m.is_open(*nxt):
                seen.add(nxt)
                queue.append(nxt)
    if len(seen) != len(m.open_cells()):
        raise ValueError(f"{m.name}: {len(m.open_cells()) - len(seen)} open cells are unreachable")
    for e in m.entrances[1:]:
        if tuple(e) not in seen:
            raise ValueError(f"{m.name}: entrance {e} is not connected")
    return m


def from_dict(d) -> GridMap:
    grid = np.array(d["grid"], dtype=np.int8)
    if grid.shape != (d["rows"], d["cols"]):
        raise ValueError(f"{d['name']}: rows/cols do not match grid shape {grid.shape}")
    return validate(GridMap(
        name=d["name"],
        grid=grid,
        entrances=[tuple(e) for e in d["entrances"]],
        pois=[(p["name"], p["row"], p["col"]) for p in d["points_of_interest"]],
        zones=[(z["name"], z["row"], z["col"], z["bonus"]) for z in d["priority_zones"]],
    ))


def save_map(m: GridMap, path=None):
    validate(m)
    path = Path(path) if path else MAPS_DIR / f"{m.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    d = m.to_dict()

    def items(key):  # one object per line keeps git diffs readable
        return ",\n    ".join(json.dumps(x) for x in d[key])

    grid_rows = ",\n    ".join(json.dumps(row) for row in d["grid"])
    text = (
        "{\n"
        f'  "name": {json.dumps(d["name"])},\n'
        f'  "rows": {d["rows"]},\n'
        f'  "cols": {d["cols"]},\n'
        f'  "entrances": {json.dumps(d["entrances"])},\n'
        f'  "points_of_interest": [\n    {items("points_of_interest")}\n  ],\n'
        f'  "priority_zones": [\n    {items("priority_zones")}\n  ],\n'
        f'  "grid": [\n    {grid_rows}\n  ]\n'
        "}\n"
    )
    path.write_text(text)
    return path


def load_map(path) -> GridMap:
    return from_dict(json.loads(Path(path).read_text()))


def build_map(name) -> GridMap:
    """Shared interface: build_map('tiny') -> GridMap loaded from maps/tiny.json."""
    return load_map(MAPS_DIR / f"{name}.json")


def zone_bonus_map(gmap: GridMap, radius=None):
    """Priority-zone bonus as a float array of shape (rows, cols).

    Input : a GridMap (from build_map(name)); uses gmap.zones = [(name, row, col, bonus), ...].
    radius: defaults to config.ZONE_RADIUS (Euclidean distance, in cells).
    Rule  : every OPEN cell with distance <= radius from a zone's (row, col) gets that zone's
            bonus added. Overlapping zones add up. Walls always stay 0.
    Note  : distance is straight-line and ignores walls, so a zone next to a wall can
            also cover open cells on the other side of it.
    Output: np.ndarray float64, same shape as gmap.grid, 0.0 where there is no zone.
    Use   : true_priority = traffic / traffic.max() + zone_bonus_map(gmap)
    """
    radius = config.ZONE_RADIUS if radius is None else radius
    bonus = np.zeros(gmap.grid.shape, dtype=float)
    rr, cc = np.indices(gmap.grid.shape)
    open_ = gmap.grid == 0
    for _, r, c, b in gmap.zones:
        bonus[open_ & ((rr - r) ** 2 + (cc - c) ** 2 <= radius ** 2)] += b
    return bonus


def to_ascii(m: GridMap) -> str:
    chars = np.where(m.grid == 1, "#", ".").astype(object)
    for r, c in m.entrances:
        chars[r, c] = "E"
    for _, r, c in m.pois:
        chars[r, c] = "P"
    for _, r, c, _ in m.zones:
        chars[r, c] = "Z"
    return "\n".join("".join(row) for row in chars)


def plot_map(m: GridMap, path=None, ax=None, legend=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.imshow(m.grid, cmap="Greys", origin="upper", vmin=0, vmax=1.6)
    ax.scatter([c for _, c in m.entrances], [r for r, _ in m.entrances],
               marker="s", s=60, c="tab:green", label="Entrance")
    ax.scatter([c for _, _, c in m.pois], [r for _, r, _ in m.pois],
               marker="o", s=40, c="tab:blue", label="Point of interest")
    ax.scatter([c for _, _, c, _ in m.zones], [r for _, r, _, _ in m.zones],
               marker="*", s=110, c="tab:red", label="Priority zone")
    ax.set_title(f"{m.name} ({m.rows}x{m.cols})", fontsize=9)
    ax.set_xticks(range(0, m.cols, 5))
    ax.set_yticks(range(0, m.rows, 5))
    ax.tick_params(labelsize=6)
    if legend and own_fig:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8)
    if own_fig:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=130)
    return fig


# --------------------------------------------------------------- generator
def generate_random_map(name, rows=28, cols=28, seed=0, n_rooms=7,
                        n_entrances=2, n_pois=4, n_zones=3) -> GridMap:
    """Random rooms-and-corridors map. Same seed -> same map. Always connected."""
    rng = np.random.default_rng(seed)
    g = np.ones((rows, cols), dtype=np.int8)

    # 1) non-overlapping rooms, at least 2 wall cells apart
    rooms, tries = [], 0
    while len(rooms) < n_rooms and tries < 2000:
        tries += 1
        h, w = int(rng.integers(4, 9)), int(rng.integers(4, 10))
        r0 = int(rng.integers(1, rows - h - 1))
        c0 = int(rng.integers(1, cols - w - 1))
        if any(r0 < b[0] + b[2] + 2 and b[0] < r0 + h + 2 and
               c0 < b[1] + b[3] + 2 and b[1] < c0 + w + 2 for b in rooms):
            continue
        rooms.append((r0, c0, h, w))
        g[r0:r0 + h, c0:c0 + w] = 0
    if len(rooms) < 3:
        raise ValueError("map too small for that many rooms")
    centers = [(r0 + h // 2, c0 + w // 2) for r0, c0, h, w in rooms]

    # 2) connect every room to its nearest earlier room with an L-shaped corridor
    def carve(r, c, wide):
        for dr in range(2 if wide else 1):
            for dc in range(2 if wide else 1):
                if 1 <= r + dr < rows - 1 and 1 <= c + dc < cols - 1:
                    g[r + dr, c + dc] = 0

    for i in range(1, len(centers)):
        (ar, ac) = centers[i]
        j = min(range(i), key=lambda k: abs(centers[k][0] - ar) + abs(centers[k][1] - ac))
        (br, bc) = centers[j]
        wide = bool(rng.integers(0, 2))
        for c in range(min(ac, bc), max(ac, bc) + 1):
            carve(ar, c, wide)
        for r in range(min(ar, br), max(ar, br) + 1):
            carve(r, bc, wide)

    # 3) entrances: shortest straight paths from the outer border to open space
    options = []
    for _ in range(60):
        side = int(rng.integers(0, 4))
        if side in (0, 1):
            pos = int(rng.integers(2, cols - 2))
            start, step = ((0, pos), (1, 0)) if side == 0 else ((rows - 1, pos), (-1, 0))
        else:
            pos = int(rng.integers(2, rows - 2))
            start, step = ((pos, 0), (0, 1)) if side == 2 else ((pos, cols - 1), (0, -1))
        path, (r, c) = [], start
        while 0 <= r < rows and 0 <= c < cols and g[r, c] == 1:
            path.append((r, c))
            r, c = r + step[0], c + step[1]
        if 0 <= r < rows and 0 <= c < cols and len(path) <= 6:
            options.append((len(path), start, path))
    options.sort(key=lambda o: o[0])
    entrances = []
    for _, start, path in options:
        if len(entrances) == n_entrances:
            break
        if all(abs(start[0] - e[0]) + abs(start[1] - e[1]) > 6 for e in entrances):
            for r, c in path:
                g[r, c] = 0
            entrances.append(start)
    if not entrances:
        raise ValueError("could not place an entrance")

    # 4) points of interest and priority zones, spread over different rooms
    names = ["Desk", "ATM", "Lift", "Shop", "Office", "Stairs"]
    order = rng.permutation(len(rooms))
    pois = []
    for k in range(min(n_pois, len(names))):
        r0, c0, h, w = rooms[order[k % len(rooms)]]
        r, c = int(rng.integers(r0, r0 + h)), int(rng.integers(c0, c0 + w))
        if all((r, c) != (p[1], p[2]) for p in pois):
            pois.append((names[k], r, c))
    zones = [(f"{n} zone", r, c, float(rng.choice([1.0, 1.5, 2.0]))) for n, r, c in pois[:n_zones]]
    er, ec = entrances[0]
    inward = (1 if er == 0 else -1 if er == rows - 1 else 0,
              1 if ec == 0 else -1 if ec == cols - 1 else 0)
    zones.append(("Entrance zone", er + 2 * inward[0], ec + 2 * inward[1], 2.0))
    zones = [z for z in zones if g[z[1], z[2]] == 0]

    return validate(GridMap(name, g, entrances, pois, zones))


if __name__ == "__main__":
    import sys
    for nm in sys.argv[1:] or [p.stem for p in sorted(MAPS_DIR.glob("*.json"))]:
        mp = build_map(nm)
        print(f"{nm}: OK ({mp.rows}x{mp.cols}, {len(mp.open_cells())} open cells)")
        print(to_ascii(mp))