
from collections import deque
#new imports for 100 poeple
import random
import numpy as np


def find_path(grid, start, goal, rng = None):

    #directions
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    # Randomise exploration order to vary equally
    # short paths between different simulations
    if rng is not None:
        rng.shuffle(directions)

    # Queue for BFS
    queue = deque([start])

    # Store visited cells
    visited = {start}

    # Store where each cell came from
    parent = {start: None}

    while queue:

        current = queue.popleft()

        # Check if destination is reached
        if current == goal:
            break

        row, col = current

        # Explore neighbouring cells
        for dr, dc in directions:

            new_row = row + dr
            new_col = col + dc

            # Check grid boundaries
            if (0 <= new_row < grid.shape[0] and
                0 <= new_col < grid.shape[1]):

                neighbour = (new_row, new_col)

                # Check if cell is walkable and unvisited
                if grid[new_row, new_col] == 0 and neighbour not in visited:

                    visited.add(neighbour)

                    parent[neighbour] = current

                    queue.append(neighbour)

    # If destination was not reached
    if goal not in parent:
        return None

    # Reconstruct shortest path
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    return path


def simulate_traffic(grid, entrances, pois, num_people=100, seed=42):

    rng = random.Random(seed)

    # Create an array to count visits to each cell
    visit_count = np.zeros(grid.shape, dtype=int)

    poi_locations = list(pois.values())

    for person in range(num_people):

        # Randomly select an entrance
        start = rng.choice(entrances)

        # Randomly select a destination
        goal = rng.choice(poi_locations)

        # Find the person's route
        path = find_path(grid, start, goal, rng)

        # If a valid path exists, count visits
        if path is not None:

            for row, col in path:
                visit_count[row, col] += 1

    return visit_count

