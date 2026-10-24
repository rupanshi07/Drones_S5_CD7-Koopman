# -*- coding: utf-8 -*-
"""run_all_comparisons.py

Convenient wrapper to generate the *comparison.png* image for a set of
simulation configurations.

The original ``run_sim.py`` script already creates ``comparison.png`` when it
runs with ``--ctrl all`` (and without ``--no-plot``).  This helper runs the
simulation with the requested parameters and then moves the generated PNG to a
dedicated folder (``analysis/comparisons``) so that all comparison images are
kept together.

Usage (run from the MuJoCo project root)::

    python analysis/run_all_comparisons.py \
        --terrain rolling \
        --payload 0.2 \
        --wind 0.015

The arguments correspond to the same flags accepted by ``run_sim.py``.
The script does **not** modify any CSV files.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

def run_simulation(root: Path, terrain: str, payload: float, wind: float) -> Path:
    """Execute ``run_sim.py`` with the given parameters.

    The command renders the GUI, saves ``comparison.png`` in the current working
    directory, and returns the absolute path to the generated PNG.
    """
    script = root / "run_sim.py"
    cmd = [
        sys.executable,
        str(script),
        "--ctrl",
        "all",
        "--gui",
        f"--terrain",
        terrain,
        f"--payload",
        str(payload),
        f"--wind",
        str(wind),
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    # Run in the MuJoCo project directory so that ``comparison.png`` lands there.
    subprocess.run(cmd, cwd=str(root), check=True)
    png_path = root / "comparison.png"
    if not png_path.is_file():
        sys.exit("[ERROR] comparison.png not found after run_sim.py execution.")
    return png_path.resolve()

def main():
    parser = argparse.ArgumentParser(description="Generate comparison PNGs for different simulation setups.")
    parser.add_argument("--terrain", default="rolling", help="Terrain type (default: rolling)")
    parser.add_argument("--payload", type=float, default=0.2, help="Payload increase as a fraction (e.g., 0.2 for +20%%).")
    parser.add_argument("--wind", type=float, default=0.015, help="Wind speed (N) for the gust period.")
    parser.add_argument("--out", default="analysis/comparisons", help="Folder where the PNG will be stored.")
    args = parser.parse_args()

    project_root = Path(__file__).parents[1]  # MuJoCo project root
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run the simulation and get the generated PNG.
    png_src = run_simulation(project_root, args.terrain, args.payload, args.wind)

    # Destination filename includes the parameters for easy identification.
    dest_name = f"comparison_{args.terrain}_payload{args.payload}_wind{args.wind}.png"
    dest_path = out_dir / dest_name
    shutil.move(str(png_src), str(dest_path))
    print(f"[INFO] Comparison image saved to {dest_path}")

if __name__ == "__main__":
    main()
