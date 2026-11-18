"""Shared settings. Person 4 owns this file; Person 1 adds the camera constants."""

# --- Person 1: camera model ---
N_DIRECTIONS = 8      # 8 directions = one every 45 degrees
FOV_DEG = 90          # field of view in degrees
CAMERA_RANGE = 8      # range in grid cells

# Direction convention (write it in the README):
#   direction 0 faces EAST (column increases), indices go COUNTER-CLOCKWISE
#   as drawn on screen, so direction 2 faces NORTH (row decreases).

# --- Shared ---
SEED = 42
MAPS_DIR = "maps"
MASKS_DIR = "masks"
PRIORITY_DIR = "priority"