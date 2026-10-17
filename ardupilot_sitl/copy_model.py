"""
copy_model.py
-------------
Copy the trained Koopman model from Phase 2 into this simulation folder.

Run this once before running sitl_controller.py.

Usage:
  python copy_model.py
"""

import shutil
from pathlib import Path


# Phase 2 data directory (relative to this file's location)
PHASE2_DATA_DIR = Path(__file__).parent.parent / "v2_stable_koopman_mpc" / "data"

# Target directory in ardupilot_sitl/
TARGET_DATA_DIR = Path(__file__).parent / "data"


def copy_model():
    TARGET_DATA_DIR.mkdir(exist_ok=True)

    # Find koopman_model.npz in Phase 2 data/
    candidates = [
        PHASE2_DATA_DIR / "koopman_model.npz",
        PHASE2_DATA_DIR / "koopman_model_final.npz",
    ]

    found = None
    for candidate in candidates:
        if candidate.exists():
            found = candidate
            break

    if found is None:
        print("ERROR: Koopman model not found in Phase 2 data/ directory.")
        print(f"  Looked in: {PHASE2_DATA_DIR}")
        print()
        print("You need to train the Koopman model first:")
        print("  cd ../v2_stable_koopman_mpc")
        print("  python data_collection/collect_koopman_data.py")
        print("  python koopman_model/build_edmd_dataset.py")
        print("  python koopman_model/fit_koopman.py")
        print()
        print("Alternatively, if your model is at a different path, copy it manually:")
        print("  copy <path_to_model.npz> ardupilot_sitl\\data\\koopman_model.npz")
        return False

    dst = TARGET_DATA_DIR / "koopman_model.npz"
    shutil.copy2(str(found), str(dst))
    print(f"Copied: {found}")
    print(f"    -> {dst}")

    # Verify
    import numpy as np
    model = np.load(str(dst))
    print(f"\nModel loaded successfully:")
    print(f"  A: {model['A'].shape}")
    print(f"  B: {model['B'].shape}")
    print(f"\nReady to run sitl_controller.py!")
    return True


if __name__ == "__main__":
    copy_model()
