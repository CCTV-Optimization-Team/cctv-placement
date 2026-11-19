
import numpy as np
import os


def create_true_priority_map(grid, visit_count, priority_zones):
    """
    Creates the true priority map using:
    Visit Count + Priority Zone Bonus

    Map convention:
    0 = Open cell
    1 = Wall
    """

    # Check that both matrices have the same dimensions
    if grid.shape != visit_count.shape:
        raise ValueError("Grid and visit count dimensions do not match!")

    # Start with visit counts as the base priority
    true_priority = visit_count.astype(float).copy()

    # Add bonuses for important priority zones
    for zone in priority_zones:
        row = zone["row"]
        col = zone["col"]
        bonus = zone["bonus"]

        # Check whether coordinates are valid
        if not (0 <= row < grid.shape[0] and
                0 <= col < grid.shape[1]):
            raise ValueError(f"Invalid priority zone coordinates: {zone}")

        # Add bonus only if the cell is open
        if grid[row, col] == 0:
            true_priority[row, col] += bonus

    # Walls must always have priority 0
    true_priority[grid == 1] = 0

    return true_priority


def save_priority_map(priority_map, map_name):
    """
    Saves the priority map as a NumPy .npy file.
    """

    os.makedirs("priority", exist_ok=True)

    file_path = f"priority/{map_name}_true.npy"

    np.save(file_path, priority_map)

    print(f"True priority map saved at: {file_path}")
