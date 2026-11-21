
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visibility import load_map, build_masks


def main():
    map_data = load_map(ROOT / "maps" / "tiny.json")

    # Extract the NumPy grid from the map dictionary
    grid = map_data["grid"]

    times = []
    candidates = None
    masks = None

    for _ in range(3):
        start = time.perf_counter()

        candidates, masks = build_masks(grid)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    print("Map dimensions:", grid.shape)
    print("Number of candidates:", len(candidates))
    print("Mask shape:", masks.shape)
    print("Run times:", [round(t, 4) for t in times], "seconds")
    print("Median time:", round(median(times), 4), "seconds")


if __name__ == "__main__":
    main()
