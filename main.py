
import numpy as np
import matplotlib.pyplot as plt

from simulation import find_path, simulate_traffic

from ml_priority import create_true_priority_map, save_priority_map

from features import extract_features, FEATURE_NAMES


# --------------------------------
# 1. CREATE A TEMPORARY 20x20 MAP
# --------------------------------

# Official convention (Person 4):
# 0 = Open cell
# 1 = Wall

grid = np.zeros((20, 20), dtype=int)

grid[5, 2:18] = 1
grid[5, 8:12] = 0

grid[10, 2:15] = 1
grid[10, 6:10] = 0

grid[15, 5:18] = 1
grid[15, 10:14] = 0


# --------------------------------
# 2. DEFINE ENTRANCES AND POIs
# --------------------------------

entrances = [(0, 10)]

pois = {
    "ATM": (8, 15),
    "Parking": (18, 15),
    "Reception": (3, 3)
}


# --------------------------------
# 3. TEST BFS PATHFINDING
# --------------------------------

start = entrances[0]
goal = pois["ATM"]

path = find_path(grid, start, goal)

print("Shortest path:", path)

if path is not None:
    print("Number of cells:", len(path))
else:
    print("No path found!")

# --------------------------------
# FEATURE EXTRACTION
# --------------------------------
# Extract ML features
X, cell_coordinates = extract_features(grid, entrances, pois)

print("\nFeature Extraction Results")
print("Feature names:", FEATURE_NAMES)
print("Feature matrix shape:", X.shape)
print("Number of cell coordinates:", len(cell_coordinates))

print("\nFirst 5 feature rows:")
print(X[:5])

print("\nFirst 5 cell coordinates:")
print(cell_coordinates[:5])


# --------------------------------
# 4. VISUALIZE THE MAP AND BFS PATH
# --------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(grid, cmap="gray")

if path is not None:
    rows = [cell[0] for cell in path]
    cols = [cell[1] for cell in path]

    plt.plot(cols, rows, color="red", linewidth=2, label="BFS Path")

plt.scatter(
    [col for row, col in entrances],
    [row for row, col in entrances],
    color="blue",
    label="Entrance"
)

plt.scatter(
    [col for row, col in pois.values()],
    [row for row, col in pois.values()],
    color="green",
    label="POIs"
)

plt.title("Temporary Floor Map with BFS Path")
plt.xlabel("Column")
plt.ylabel("Row")
plt.legend()
plt.show()


# --------------------------------
# 5. SIMULATE FOOT TRAFFIC
# --------------------------------

visit_count = simulate_traffic(
    grid,
    entrances,
    pois,
    num_people=100,
    seed=42
)

print("\nFoot Traffic Simulation Results")
print("Total visits:", np.sum(visit_count))
print("Maximum visits to a cell:", np.max(visit_count))
print("Shape of visit count map:", visit_count.shape)


# --------------------------------
# 6. VISUALIZE FOOT TRAFFIC
# --------------------------------

plt.figure(figsize=(8, 8))

# Mask wall cells (1).

traffic_map = np.ma.masked_where(grid == 1, visit_count)

plt.imshow(traffic_map, cmap="hot")
plt.colorbar(label="Number of Visits")

plt.title("Foot Traffic Simulation - 100 People")
plt.xlabel("Column")
plt.ylabel("Row")

plt.show()


# --------------------------------
# 7. DEFINE PRIORITY ZONES
# --------------------------------

# Temporary priority zones for testing.
# These are individual cells, not entire areas.

priority_zones = [
    {"name": "ATM Zone", "row": 8, "col": 15, "bonus": 20},
    {"name": "Reception Zone", "row": 3, "col": 3, "bonus": 15},
    {"name": "Parking Zone", "row": 18, "col": 15, "bonus": 25}
]


# --------------------------------
# 8. CREATE TRUE PRIORITY MAP
# --------------------------------

true_priority = create_true_priority_map(
    grid,
    visit_count,
    priority_zones
)

print("\nTrue Priority Map Results")
print("Maximum priority:", np.max(true_priority))
print("Total priority:", np.sum(true_priority))


# --------------------------------
# 9. SAVE TRUE PRIORITY MAP
# --------------------------------

save_priority_map(true_priority, "tiny_map")


# --------------------------------
# 10. VISUALIZE TRUE PRIORITY MAP
# --------------------------------

plt.figure(figsize=(8, 8))

priority_display = np.ma.masked_where(grid == 1, true_priority)

plt.imshow(priority_display, cmap="hot")
plt.colorbar(label="Priority Value")

plt.title("True Priority Map")
plt.xlabel("Column")
plt.ylabel("Row")

plt.show()

# Prepare ML Dataset

y = np.array([
    true_priority[row, col]
    for row, col in cell_coordinates
])

print("\nML Dataset Results")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("First 5 target values:", y[:5])