"""make_maps.py (Person 4) - builds the hand-made maps and the random maps.

Run from the repo root:  python3 make_maps.py
Writes maps/college.json, maps/metro.json, maps/random_1..5.json and maps/overview.png.
tiny.json is NOT regenerated (it is the shared test map).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from map import GridMap, save_map, build_map, generate_random_map, plot_map


def carve(g, r0, c0, r1, c1):
    """Open the rectangle rows r0..r1, cols c0..c1 (inclusive)."""
    g[r0:r1 + 1, c0:c1 + 1] = 0


def college():
    """30x30 college block: corridor, 4 classrooms, library, lab, canteen."""
    g = np.ones((30, 30), dtype=np.int8)
    carve(g, 13, 1, 15, 28)                                   # main corridor
    for c0, c1 in [(1, 6), (8, 13), (15, 20), (22, 28)]:      # classrooms (top)
        carve(g, 1, c0, 11, c1)
    carve(g, 17, 1, 28, 12)                                   # library
    carve(g, 17, 14, 28, 20)                                  # computer lab
    carve(g, 17, 22, 28, 28)                                  # canteen
    for r, c in [(12, 3), (12, 10), (12, 17), (12, 25),       # doors to classrooms
                 (16, 6), (16, 10), (16, 17), (16, 25)]:      # doors to lower rooms
        g[r, c] = 0
    g[19:26, 4] = 1                                           # bookshelves
    g[19:26, 8] = 1
    g[20, 15:20] = 1                                          # lab desks
    g[24, 15:20] = 1
    for r, c in [(14, 0), (14, 29), (29, 25), (0, 10)]:       # gates
        g[r, c] = 0
    return GridMap(
        "college", g,
        entrances=[(14, 0), (14, 29), (29, 25), (0, 10)],
        pois=[("Admin office", 4, 3), ("Seminar hall", 5, 25), ("Library desk", 22, 6),
              ("Computer lab", 22, 17), ("Canteen counter", 22, 25), ("Notice board", 14, 15)],
        zones=[("Main gate", 14, 2, 2.0), ("Canteen", 22, 25, 1.5),
               ("Library desk", 22, 6, 1.0), ("Computer lab", 22, 17, 1.0)],
    )


def metro():
    """24x40 metro concourse: street entrances, ticket counters, fare gates, two platforms."""
    g = np.ones((24, 40), dtype=np.int8)
    carve(g, 1, 1, 11, 38)                                    # upper concourse
    carve(g, 13, 1, 22, 38)                                   # platform level
    for c in (8, 9, 16, 17, 24, 25, 32, 33):                  # fare gates in row 12
        g[12, c] = 0
    g[3:5, 4:11] = 1                                          # ticket counters
    g[3:5, 29:36] = 1
    for c in range(5, 36, 6):                                 # pillars in the concourse
        g[8, c] = 1
    g[15:23, 19] = 1                                          # platform separator
    for c in (5, 11, 27, 33):                                 # pillars on platforms
        g[17, c] = 1
    for c in (8, 20, 32):                                     # street entrances
        g[0, c] = 0
    return GridMap(
        "metro", g,
        entrances=[(0, 8), (0, 20), (0, 32)],
        pois=[("Ticket counter W", 5, 7), ("Ticket counter E", 5, 32), ("Fare gates", 11, 16),
              ("Stairs W", 14, 3), ("Stairs E", 14, 36), ("Waiting area W", 18, 9),
              ("Waiting area E", 18, 30)],
        zones=[("Ticket counter W", 5, 7, 1.5), ("Ticket counter E", 5, 32, 1.5),
               ("Fare gates", 11, 8, 2.0), ("Fare gates E", 11, 24, 2.0),
               ("Platform W", 20, 10, 1.5), ("Platform E", 20, 30, 1.5)],
    )


RANDOM_SPECS = [  # name, rows, cols, seed, rooms
    ("random_1", 24, 24, 1, 6), ("random_2", 26, 26, 2, 7), ("random_3", 28, 28, 3, 8),
    ("random_4", 30, 30, 4, 9), ("random_5", 24, 30, 5, 7),
]


if __name__ == "__main__":
    names = ["tiny"]
    for m in (college(), metro()):
        print("saved", save_map(m)); names.append(m.name)
    for name, rows, cols, seed, rooms in RANDOM_SPECS:
        m = generate_random_map(name, rows, cols, seed, n_rooms=rooms)
        print("saved", save_map(m)); names.append(name)

    maps = [build_map(n) for n in names]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, m in zip(axes.flat, maps):
        plot_map(m, ax=ax)
    for ax in list(axes.flat)[len(maps):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("maps/overview.png", dpi=110)
    print("maps:", names)
