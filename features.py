
import numpy as np
from collections import deque


# Order of features in the final ML matrix
FEATURE_NAMES = [
    "distance_to_entrance",
    "distance_to_poi",
    "nearby_open_space",
    "corridor_like",
    "distance_to_wall"
]


def normalise_points(points):
    """
    Converts entrances or POIs into a list of (row, col) coordinates.

    Supports:
    - [(0, 5), (10, 3)]
    - {"ATM": (8, 15), "Reception": (3, 3)}
    - [{"name": "ATM", "row": 8, "col": 15}]
    """

    if isinstance(points, dict):
        points = list(points.values())

    coordinates = []

    for point in points:
        if isinstance(point, dict):
            coordinates.append((point["row"], point["col"]))
        else:
            coordinates.append((point[0], point[1]))

    return coordinates


def calculate_distance_map(grid, source_points):
    """
    Calculates shortest-path distance from every open cell
    to the nearest source point using multi-source BFS.

    Map convention:
    0 = Open cell
    1 = Wall
    """

    rows, cols = grid.shape

    distance = np.full((rows, cols), np.inf)

    queue = deque()

    for row, col in source_points:

        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"Invalid coordinate: {(row, col)}")

        if grid[row, col] == 1:
            raise ValueError(f"Source point {(row, col)} is on a wall.")

        if distance[row, col] == np.inf:
            distance[row, col] = 0
            queue.append((row, col))

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    while queue:

        row, col = queue.popleft()

        for dr, dc in directions:

            nr = row + dr
            nc = col + dc

            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and grid[nr, nc] == 0
                and distance[nr, nc] == np.inf
            ):

                distance[nr, nc] = distance[row, col] + 1
                queue.append((nr, nc))

    return distance


def calculate_nearby_open_space(grid, row, col):
    """
    Counts open cells in the surrounding 3x3 neighbourhood.
    The current cell is included in the count.
    """

    rows, cols = grid.shape

    count = 0

    for r in range(max(0, row - 1), min(rows, row + 2)):
        for c in range(max(0, col - 1), min(cols, col + 2)):

            if grid[r, c] == 0:
                count += 1

    return count


def is_corridor_like(grid, row, col):
    """
    A cell is considered corridor-like if it has exactly
    two opposite open neighbours (north-south or east-west).
    """

    rows, cols = grid.shape

    neighbours = {
        "up": (row - 1, col),
        "down": (row + 1, col),
        "left": (row, col - 1),
        "right": (row, col + 1)
    }

    open_neighbours = {}

    for direction, (r, c) in neighbours.items():

        if (
            0 <= r < rows
            and 0 <= c < cols
            and grid[r, c] == 0
        ):
            open_neighbours[direction] = True
        else:
            open_neighbours[direction] = False

    vertical_corridor = (
        open_neighbours["up"]
        and open_neighbours["down"]
        and not open_neighbours["left"]
        and not open_neighbours["right"]
    )

    horizontal_corridor = (
        open_neighbours["left"]
        and open_neighbours["right"]
        and not open_neighbours["up"]
        and not open_neighbours["down"]
    )

    return int(vertical_corridor or horizontal_corridor)


def calculate_distance_to_wall(grid, row, col):
    """
    Calculates Manhattan distance to the nearest wall.
    If the map has no walls, returns a finite fallback value.
    """

    wall_positions = np.argwhere(grid == 1)

    if len(wall_positions) == 0:
        return grid.shape[0] + grid.shape[1]

    distances = (
        np.abs(wall_positions[:, 0] - row)
        + np.abs(wall_positions[:, 1] - col)
    )

    return int(np.min(distances))


def extract_features(grid, entrances, pois):
    """
    Extracts five features for every open cell.

    Returns:
        X: Feature matrix (one row per open cell)
        cell_coordinates: Corresponding (row, col) for each row
    """

    grid = np.asarray(grid)

    if grid.ndim != 2:
        raise ValueError("Grid must be a 2D array.")

    if not np.isin(grid, [0, 1]).all():
        raise ValueError("Grid must contain only 0 (open) and 1 (wall).")

    entrances = normalise_points(entrances)
    pois = normalise_points(pois)

    if not entrances:
        raise ValueError("At least one entrance is required.")

    if not pois:
        raise ValueError("At least one POI is required.")

    entrance_distances = calculate_distance_map(grid, entrances)
    poi_distances = calculate_distance_map(grid, pois)

    X = []
    cell_coordinates = []

    rows, cols = grid.shape

    for row in range(rows):
        for col in range(cols):

            # Ignore wall cells
            if grid[row, col] == 1:
                continue

            # Skip cells that cannot reach an entrance or POI
            if (
                not np.isfinite(entrance_distances[row, col])
                or not np.isfinite(poi_distances[row, col])
            ):
                continue

            features = [
                entrance_distances[row, col],
                poi_distances[row, col],
                calculate_nearby_open_space(grid, row, col),
                is_corridor_like(grid, row, col),
                calculate_distance_to_wall(grid, row, col)
            ]

            X.append(features)
            cell_coordinates.append((row, col))

    return np.array(X, dtype=float), cell_coordinates
