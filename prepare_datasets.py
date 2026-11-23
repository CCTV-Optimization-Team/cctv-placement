
import os
import numpy as np

import config
from map import build_map
from features import extract_features, FEATURE_NAMES


def main():

    all_X = []
    all_y = []
    all_map_names = []
    all_coordinates = []

    os.makedirs("datasets", exist_ok=True)

    print("PREPARING ML DATASETS")

    for map_name in config.MAP_NAMES:

        print(f"\nProcessing: {map_name}")

        # 1. Load the map
        gmap = build_map(map_name)

        # 2. Extract POI coordinates
        pois = [
            (row, col)
            for _, row, col in gmap.pois
        ]

        # 3. Extract five features for every open cell
        X, cell_coordinates = extract_features(
            gmap.grid,
            gmap.entrances,
            pois
        )

        # 4. Load the corresponding true priority map
        priority_path = os.path.join(
            config.PRIORITY_DIR,
            f"{map_name}_true.npy"
        )

        if not os.path.exists(priority_path):
            raise FileNotFoundError(
                f"Priority map not found: {priority_path}"
            )

        true_priority = np.load(priority_path)

        if true_priority.shape != gmap.grid.shape:
            raise ValueError(
                f"{map_name}: Priority map dimensions do not match grid"
            )

        # 5. Match each feature row with its cell's true priority
        y = np.array([
            true_priority[row, col]
            for row, col in cell_coordinates
        ], dtype=float)

        # 6. Basic validation
        if X.shape[0] != len(y):
            raise ValueError(f"{map_name}: X and y sizes do not match")

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"{map_name}: Dataset contains invalid values")

        # 7. Save individual map dataset
        np.savez_compressed(
            f"datasets/{map_name}_dataset.npz",
            X=X,
            y=y,
            coordinates=np.array(cell_coordinates),
            feature_names=np.array(FEATURE_NAMES)
        )

        # 8. Add to combined dataset
        all_X.append(X)
        all_y.append(y)
        all_map_names.extend([map_name] * len(y))
        all_coordinates.extend(cell_coordinates)

        print("Feature matrix:", X.shape)
        print("Target array:", y.shape)
        print("Number of cells:", len(cell_coordinates))

    # 9. Combine datasets from all maps
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    np.savez_compressed(
        "datasets/all_maps_dataset.npz",
        X=X_combined,
        y=y_combined,
        map_names=np.array(all_map_names),
        coordinates=np.array(all_coordinates),
        feature_names=np.array(FEATURE_NAMES)
    )

    print("\nALL MAPS COMBINED")
    print("Combined X shape:", X_combined.shape)
    print("Combined y shape:", y_combined.shape)
    print("Feature names:", FEATURE_NAMES)
    print("Datasets saved in the datasets/ folder")


if __name__ == "__main__":
    main()
