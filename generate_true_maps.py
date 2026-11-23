
import numpy as np

import config
from map import build_map
from simulation import simulate_traffic
from ml_priority import create_true_priority_map, save_priority_map


# Simulation settings
NUM_PEOPLE = 500
SEEDS = [42, 43, 44]


def main():

    print("GENERATING TRUE PRIORITY MAPS")

    for map_name in config.MAP_NAMES:

        print(f"\nProcessing map: {map_name}")

        # Step 1: Load the map
        gmap = build_map(map_name)

        # Convert POIs into the format required by simulate_traffic()
        pois = {
            name: (row, col)
            for name, row, col in gmap.pois
        }

        # Step 2: Run traffic simulation multiple times
        visit_counts = []

        for seed in SEEDS:

            visit_count = simulate_traffic(
                gmap.grid,
                gmap.entrances,
                pois,
                num_people=NUM_PEOPLE,
                seed=seed
            )

            visit_counts.append(visit_count)

        # Step 3: Average traffic across all simulations
        average_visit_count = np.mean(visit_counts, axis=0)

        # Step 4: Generate true priority map
        true_priority = create_true_priority_map(
            gmap,
            average_visit_count
        )

        # Step 5: Save the true priority map
        save_priority_map(true_priority, map_name)

        # Step 6: Display basic results
        print("Map dimensions:", gmap.grid.shape)
        print("People per simulation:", NUM_PEOPLE)
        print("Number of simulations:", len(SEEDS))
        print("Seeds used:", SEEDS)
        print("Maximum average traffic:", np.max(average_visit_count))
        print("Maximum priority:", np.max(true_priority))
        print("Wall priority values:", np.unique(true_priority[gmap.grid == 1]))

    print("\nAll true priority maps generated successfully!")


if __name__ == "__main__":
    main()
