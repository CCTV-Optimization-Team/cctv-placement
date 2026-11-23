import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import config

for path in sorted(Path(config.MASKS_DIR).glob("*.npz")):
    print(f"\nChecking: {path.name}")

    data = np.load(path)

    print("Keys:", data.files)

    candidates = data["candidates"]
    masks = data["masks"]

    print("Candidates shape:", candidates.shape)
    print("Masks shape:", masks.shape)
    print("Masks dtype:", masks.dtype)

    assert set(data.files) == {"candidates", "masks"}
    assert candidates.ndim == 2 and candidates.shape[1] == 3
    assert masks.ndim == 3
    assert masks.shape[0] == candidates.shape[0]
    assert masks.dtype == bool

    print("VERIFIED OK")
