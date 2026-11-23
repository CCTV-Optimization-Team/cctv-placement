
import os
import numpy as np
import config

for map_name in config.MAP_NAMES:

    source = os.path.join(
        config.PRIORITY_DIR,
        f"{map_name}_pred_gb.npy"
    )

    destination = os.path.join(
        config.PRIORITY_DIR,
        f"{map_name}_pred.npy"
    )

    predictions = np.load(source)
    np.save(destination, predictions)

    print(f"Created {destination}")

print("\nAll eight prediction files created!")
