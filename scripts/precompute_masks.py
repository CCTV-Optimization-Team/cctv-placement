from pathlib import Path
from visibility import load_map, build_masks, save_masks
import config

maps_dir = Path(config.MAPS_DIR)
masks_dir = Path(config.MASKS_DIR)
masks_dir.mkdir(parents=True, exist_ok=True)

for path in maps_dir.glob("*.json"):
    m = load_map(path)
    grid = m["grid"]

    candidates, masks = build_masks(grid)

    save_masks(
        masks_dir / f"{path.stem}.npz",
        grid,
        candidates,
        masks,
        config.CAMERA_RANGE,
        config.FOV_DEG,
        config.N_DIRECTIONS
    )

    print(f"Saved {path.stem}: {len(candidates)} candidates")