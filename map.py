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


def to_ascii(m: GridMap) -> str:
    chars = np.where(m.grid == 1, "#", ".").astype(object)
    for r, c in m.entrances:
        chars[r, c] = "E"
    for _, r, c in m.pois:
        chars[r, c] = "P"
    for _, r, c, _ in m.zones:
        chars[r, c] = "Z"
    return "\n".join("".join(row) for row in chars)


def plot_map(m: GridMap, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(m.grid, cmap="Greys", origin="upper", vmin=0, vmax=1.6)
    ax.scatter([c for _, c in m.entrances], [r for r, _ in m.entrances],
               marker="s", s=90, c="tab:green", label="Entrance")
    ax.scatter([c for _, _, c in m.pois], [r for _, r, _ in m.pois],
               marker="o", s=70, c="tab:blue", label="Point of interest")
    ax.scatter([c for _, _, c, _ in m.zones], [r for _, r, _, _ in m.zones],
               marker="*", s=160, c="tab:red", label="Priority zone")
    for n, r, c in m.pois:
        ax.annotate(n, (c, r), xytext=(4, -6), textcoords="offset points", fontsize=7)
    for n, r, c, _ in m.zones:
        ax.annotate(n, (c, r), xytext=(4, -6), textcoords="offset points", fontsize=7)
    ax.set_title(f"Map: {m.name} ({m.rows}x{m.cols})")
    ax.set_xticks(range(0, m.cols, 5))
    ax.set_yticks(range(0, m.rows, 5))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


if __name__ == "__main__":
    import sys
    for nm in sys.argv[1:] or [p.stem for p in sorted(MAPS_DIR.glob("*.json"))]:
        mp = build_map(nm)
        print(f"{nm}: OK ({mp.rows}x{mp.cols}, {len(mp.open_cells())} open cells)")
        print(to_ascii(mp))
