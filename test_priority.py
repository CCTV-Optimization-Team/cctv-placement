import numpy as np

from map import build_map
from simulation import simulate_traffic
from ml_priority import create_true_priority_map, save_priority_map

# Load actual map
gmap = build_map("college")

# Extract map information
grid = gmap.grid
entrances = gmap.entrances

# Convert P4's POI format into the format used by simulate_traffic
pois = {
    name: (row, col)
    for name, row, col in gmap.pois
}

# Simulate traffic
visit_count = simulate_traffic(
    grid,
    entrances,
    pois,
    num_people=100,
    seed=42
)

# Generate true priority map
true_priority = create_true_priority_map(gmap, visit_count)

# Save result
save_priority_map(true_priority, gmap.name)

# Display results
print("\nTrue Priority Map Results")
print("Map:", gmap.name)
print("Priority shape:", true_priority.shape)
print("Maximum priority:", np.max(true_priority))
print("Minimum priority:", np.min(true_priority))
print("Maximum traffic:", np.max(visit_count))
print("Wall priority values:", np.unique(true_priority[grid == 1]))
print("Saved successfully!")