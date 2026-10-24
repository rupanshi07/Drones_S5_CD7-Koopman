# -*- coding: utf-8 -*-
"""run_batch_comparisons.py

Utility to generate *comparison.png* images for a grid of simulation
parameters (terrain, payload, wind). It repeatedly calls the existing
``run_sim.py`` script with ``--ctrl all --gui`` for each combination and moves
the resulting PNG into ``analysis/comparisons`` with a descriptive filename.

Usage (run from the MuJoCo project root)::

    python analysis/run_batch_comparisons.py \
        --terrains rolling flat mountain canyon \
        --payloads 0.0 0.2 0.4 \
        --winds 0.0 0.015 0.03

All arguments are optional – defaults cover a reasonable subset.
The script does **not** create or modify any CSV files.
"""

import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

def run_sim(root: Path, terrain: str, payload: float, wind: float) -> Path:
    """Execute ``run_sim.py`` with the given parameters.

    Returns the absolute path to the generated ``comparison.png``.
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
    subprocess.run(cmd, cwd=str(root), check=True)
    png_path = root / "comparison.png"
    if not png_path.is_file():
        sys.exit(f"[ERROR] comparison.png not found after run_sim for {terrain}, payload={payload}, wind={wind}")
    return png_path.resolve()

def main():
    parser = argparse.ArgumentParser(description="Batch generate comparison PNGs for various terrains, payloads and wind settings.")
    parser.add_argument(
        "--terrains",
        nargs="*",
        default=["rolling", "flat", "mountain", "canyon"],
        help="List of terrain names (default: rolling flat mountain canyon)",
    )
    parser.add_argument(
        "--payloads",
        type=float,
        nargs="*",
        default=[0.0, 0.2, 0.4],
        help="Payload fractions (e.g., 0.0 for nominal, 0.2 for +20%%).",
    )
    parser.add_argument(
        "--winds",
        type=float,
        nargs="*",
        default=[0.0, 0.015, 0.03],
        help="Wind magnitudes used during the 3‑5 s gust window.",
    )
    parser.add_argument(
        "--out",
        default="analysis/comparisons",
        help="Directory to store the generated PNGs.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parents[1]
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.terrains, args.payloads, args.winds))
    total = len(combos)
    print(f"[INFO] Generating {total} comparison images...")

    for idx, (terrain, payload, wind) in enumerate(combos, start=1):
        print(f"[INFO] ({idx}/{total}) terrain={terrain}, payload={payload}, wind={wind}")
        png_src = run_sim(project_root, terrain, payload, wind)
        dest_name = f"comparison_{terrain}_payload{payload}_wind{wind}.png"
        dest_path = out_dir / dest_name
        shutil.move(str(png_src), str(dest_path))
        print(f"[INFO] Saved: {dest_path}\n")

    print("[DONE] All comparison images have been generated and stored in:")
    print(out_dir)

if __name__ == "__main__":
    main()
