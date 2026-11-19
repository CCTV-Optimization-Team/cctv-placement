
from pathlib import Path
import numpy as np

from visibility import load_map, build_candidates, build_masks, save_masks
import config


def main():
    maps_dir = Path(config.MAPS_DIR)
    masks_dir = Path(config.MASKS_DIR)
    masks_dir.mkdir(parents=True, exist_ok=True)

    map_files = sorted(maps_dir.glob("*.json"))

    if not map_files:
        print("No maps found.")
        return

    for map_path in map_files:
        print(f"\nProcessing: {map_path.name}")

        gmap = load_map(map_path)
        grid = gmap["grid"]

        candidates = build_candidates(grid)

        candidates,masks = build_masks(
            grid,
            cam_range=config.CAMERA_RANGE,
            fov_deg=config.FOV_DEGREES,
            n_directions=config.N_DIRECTIONS,
        )

        output_path = masks_dir / f"{map_path.stem}.npz"

        save_masks(
            output_path,
            grid,
            candidates,
            masks,
            cam_range=config.CAMERA_RANGE,
            fov_deg=config.FOV_DEGREES,
            n_directions=config.N_DIRECTIONS,
        )

        print(f"Map size: {grid.shape}")
        print(f"Candidates: {len(candidates)}")
        print(f"Masks shape: {masks.shape}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
