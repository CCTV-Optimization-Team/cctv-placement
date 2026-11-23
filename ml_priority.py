
import os
import numpy as np

import config
from map import zone_bonus_map


def create_true_priority_map(gmap, visit_count):
    """
    Creates the true priority map using:
    Normalised Traffic + Priority Zone Bonuses

    Map convention:
    0 = Open cell
    1 = Wall
    """

    visit_count = np.asarray(visit_count, dtype=float)

    if visit_count.shape != gmap.grid.shape:
        raise ValueError("Visit count and grid dimensions do not match!")

    if not np.isfinite(visit_count).all() or np.any(visit_count < 0):
        raise ValueError("Visit counts must be finite and non-negative.")

    # Step 1: Normalise traffic so the busiest cell has value 1
    max_visits = np.max(visit_count[gmap.grid == 0])

    if max_visits > 0:
        traffic = visit_count / max_visits
    else:
        traffic = np.zeros(gmap.grid.shape, dtype=float)

    # Step 2: Get priority zone bonuses from P4's map.py
    bonuses = zone_bonus_map(gmap)

    # Step 3: Combine normalised traffic and zone bonuses
    true_priority = traffic + bonuses

    # Step 4: Walls must always have priority 0
    true_priority[gmap.grid == 1] = 0.0

    return true_priority


def save_priority_map(priority_map, map_name):
    """
    Saves the true priority map as a NumPy file.
    """

    os.makedirs(config.PRIORITY_DIR, exist_ok=True)

    file_path = os.path.join(
        config.PRIORITY_DIR,
        f"{map_name}_true.npy"
    )

    np.save(file_path, priority_map)

    print(f"True priority map saved at: {file_path}")


def predict_priority(gmap):
    """
    Loads the Gradient Boosting predicted priority map
    for the given GridMap.
    """

    file_path = os.path.join(
        config.PRIORITY_DIR,
        f"{gmap.name}_pred.npy"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Predicted priority map not found: {file_path}"
        )

    priority = np.load(file_path)

    if priority.shape != gmap.grid.shape:
        raise ValueError("Predicted priority map dimensions do not match!")

    if not np.isfinite(priority).all():
        raise ValueError("Predicted priority map contains NaN or infinity!")

    if np.any(priority[gmap.grid == 1] != 0):
        raise ValueError("Wall cells must have priority 0!")

    return priority.astype(float)


def true_priority(gmap):
    """
    Loads the true priority map for the given GridMap.
    """

    file_path = os.path.join(
        config.PRIORITY_DIR,
        f"{gmap.name}_true.npy"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"True priority map not found: {file_path}"
        )

    priority = np.load(file_path)

    if priority.shape != gmap.grid.shape:
        raise ValueError("True priority map dimensions do not match!")

    if not np.isfinite(priority).all():
        raise ValueError("True priority map contains NaN or infinity!")

    if np.any(priority[gmap.grid == 1] != 0):
        raise ValueError("Wall cells must have priority 0!")

    return priority.astype(float)

