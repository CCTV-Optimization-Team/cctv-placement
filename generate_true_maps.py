
import numpy as np

import config
from map import build_map
from simulation import simulate_traffic
from ml_priority import create_true_priority_map, save_priority_map


def main():

    print("GENERATING TRUE PRIORITY MAPS")

    for map_name in config.MAP_NAMES:

        print(f"\nProcessing map: {map_name}")

        # Load map
        gmap = build_map(map_name)

        # Extract map information
        grid = gmap.grid
        entrances = gmap.entrances

        pois = {
            name: (row, col)
            for name, row, col in gmap.pois
        }

        # Simulate foot traffic
        visit_count = simulate_traffic(
            grid,
            entrances,
            pois,
            num_people=100,
            seed=42
        )

        # Generate true priority map
        true_priority = create_true_priority_map(
            gmap,
            visit_count
        )

        # Save priority map
        save_priority_map(true_priority, map_name)

        print("Map dimensions:", grid.shape)
        print("Priority dimensions:", true_priority.shape)
        print("Maximum traffic:", np.max(visit_count))
        print("Maximum priority:", np.max(true_priority))


if __name__ == "__main__":
    main()
