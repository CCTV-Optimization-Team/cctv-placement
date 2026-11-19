"""Shared project settings."""

# --- Camera model ---
N_DIRECTIONS = 8
FOV_DEGREES = 90
CAMERA_RANGE = 6

# Directions: 0 = North; increasing clockwise.
# (row_delta, col_delta)
DIRECTION_VECTORS = [
    (-1,  0),  # 0: North
    (-1,  1),  # 1: North-East
    ( 0,  1),  # 2: East
    ( 1,  1),  # 3: South-East
    ( 1,  0),  # 4: South
    ( 1, -1),  # 5: South-West
    ( 0, -1),  # 6: West
    (-1, -1),  # 7: North-West
]

# --- Shared ---
SEED = 42
MAPS_DIR = "maps"
MASKS_DIR = "masks"
PRIORITY_DIR = "priority"
