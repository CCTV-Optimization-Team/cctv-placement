
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow importing visibility.py from the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visibility import load_map, build_masks


def main():
    # Load the tiny map
    map_data = load_map(ROOT / "maps" / "tiny.json")
    grid = map_data["grid"]

    # Generate candidates and visibility masks
    candidates, masks = build_masks(grid)

    rows, cols = grid.shape

    # Choose an open camera location
    camera_row, camera_col = 0, 0

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for direction in range(8):
        candidate = np.array([camera_row, camera_col, direction])

        matches = np.where(np.all(candidates == candidate, axis=1))[0]

        if len(matches) == 0:
            axes[direction].set_title(f"Direction {direction}: candidate missing")
            axes[direction].axis("off")
            continue

        candidate_index = matches[0]
        visible = masks[candidate_index].reshape(rows, cols)

        # 0 = not visible, 1 = visible; walls are overlaid separately
        display_grid = visible.astype(float)
        display_grid[grid == 1] = 2

        axes[direction].imshow(
            display_grid,
            cmap="viridis",
            vmin=0,
            vmax=2,
            origin="upper"
        )

        axes[direction].scatter(
            camera_col,
            camera_row,
            marker="*",
            s=180,
            color="red",
            label="Camera"
        )

        axes[direction].set_title(f"Direction {direction}")
        axes[direction].set_xticks(range(cols))
        axes[direction].set_yticks(range(rows))
        axes[direction].grid(True, alpha=0.3)

    fig.suptitle("Visibility masks for all 8 camera directions")
    plt.tight_layout()

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "tiny_all_directions.png"

    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
