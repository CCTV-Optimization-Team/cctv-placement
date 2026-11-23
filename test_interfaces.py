
import numpy as np
from map import build_map
from interfaces import get_weights

map_names = [
    "tiny",
    "college",
    "metro",
    "random_1",
    "random_2",
    "random_3",
    "random_4",
    "random_5"
]

for name in map_names:

    print(f"\nTesting map: {name}")

    # Load map
    gmap = build_map(name)

    # Load all three weightings through interfaces.py
    uniform = get_weights(gmap, "uniform")
    predicted = get_weights(gmap, "predicted")
    true = get_weights(gmap, "true")

    # Check dimensions
    assert uniform.shape == gmap.grid.shape
    assert predicted.shape == gmap.grid.shape
    assert true.shape == gmap.grid.shape

    # Check for invalid values
    assert np.isfinite(predicted).all()
    assert np.isfinite(true).all()

    # Check wall cells
    assert np.all(predicted[gmap.grid == 1] == 0)
    assert np.all(true[gmap.grid == 1] == 0)

    print("Uniform:", uniform.shape)
    print("Predicted:", predicted.shape)
    print("True:", true.shape)
    print("PASSED!")

print("\nALL 8 MAPS PASSED!")
