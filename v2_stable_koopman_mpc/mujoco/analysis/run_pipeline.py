# -*- coding: utf-8 -*-
"""run_pipeline.py

Convenient wrapper that

1. Executes the simulation/video‑generation script ``run_all_with_ffmpeg.py`` for all
   rows in ``demos/summary.csv``.
2. After the videos are produced, reads the (now‑populated) CSV and generates PNG
   plots using :pymod:`analysis.generate_plots`.
3. Builds a simple Markdown report (``analysis/report.md``) that links each video
   file, lists the corresponding simulation parameters, and shows the metric values.

The script is *self‑contained* – it does not write any intermediate CSV files; it
only reads the existing ``summary.csv`` and writes image files and the report.

Usage (run from the MuJoCo project root)::

    python analysis/run_pipeline.py

You can adjust the paths via the optional command‑line arguments.
"""

import argparse
import os
import subprocess
import sys
import pandas as pd
from pathlib import Path

# Import the plot‑generation helper (relative import works because we are in the same
# package folder).
import generate_plots  # type: ignore  # noqa: E402


def run_simulation(csv_path: str, out_dir: str, window_title: str) -> None:
    """Invoke ``run_all_with_ffmpeg.py``.

    The function blocks until the helper finishes. Any error propagates as a
    ``CalledProcessError`` which we surface to the caller.
    """
    script_path = Path(__file__).parents[1] / "run_all_with_ffmpeg.py"
    cmd = [sys.executable, str(script_path), "--csv", csv_path, "--out", out_dir, "--title", window_title]
    print(f"[INFO] Running simulation pipeline: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def generate_plots_from_csv(csv_path: str, plots_dir: str) -> None:
    """Generate the three performance plots.

    This simply forwards to the existing ``generate_plots.main`` function.
    """
    generate_plots.main(csv_path, plots_dir)


def build_report(csv_path: str, demos_dir: str, plots_dir: str, report_path: str) -> None:
    """Create ``report.md`` linking videos, parameters and metric values.

    The Markdown file contains a table where each row corresponds to a simulation
    configuration. Columns:
        * Video (link to the MP4 file)
        * Controller, Terrain, Payload, Wind
        * RMSE, Max Error, Min AGL
        * Plots (embedded images for the three metrics)
    """
    df = pd.read_csv(csv_path)
    # Normalise column names for case‑insensitive access
    df.columns = [c.lower() for c in df.columns]
    required_cols = {"controller", "terrain", "payload", "wind", "rmse_m", "max_err_m", "min_agl_m"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] CSV is missing expected columns {missing}. The report will contain empty fields for them.")

    # Ensure the report directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Simulation Report\n\n")
        f.write("Generated automatically by ``run_pipeline.py``.\n\n")
        # Header of the Markdown table
        f.write("| Video | Controller | Terrain | Payload | Wind | RMSE (m) | Max Err (m) | Min AGL (m) | Plots |
")
        f.write("|-------|------------|---------|---------|------|----------|-------------|------------|-------|\n")
        for _, row in df.iterrows():
            video_file = f"{row.get('controller','')}_{row.get('terrain','')}_payload{row.get('payload','')}_wind{row.get('wind','')}.mp4"
            video_path = Path(demos_dir) / video_file
            video_link = f"[📹]({video_path.as_posix()})"
            controller = row.get('controller', '')
            terrain = row.get('terrain', '')
            payload = row.get('payload', '')
            wind = row.get('wind', '')
            rmse = row.get('rmse_m', '')
            max_err = row.get('max_err_m', '')
            min_agl = row.get('min_agl_m', '')
            # Embed the three metric plots – they are located in ``plots_dir``
            plots_md = " ".join([
                f"![]({Path(plots_dir) / f'{metric}.png'} )" for metric in ["RMSE_m", "max_err_m", "min_AGL_m"]
            ])
            f.write(f"| {video_link} | {controller} | {terrain} | {payload} | {wind} | {rmse} | {max_err} | {min_agl} | {plots_md} |\n")

    print(f"[INFO] Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full simulation pipeline and produce plots + markdown report.")
    parser.add_argument("--csv", default="demos/summary.csv", help="Path to the CSV with simulation parameters.")
    parser.add_argument("--out", default="demos", help="Directory where MP4 videos will be stored.")
    parser.add_argument("--title", default=os.getenv("MUJOCO_WINDOW_TITLE", "MuJoCo Simulation"), help="MuJoCo window title for ffmpeg capture.")
    parser.add_argument("--plots", default="analysis/plots", help="Directory to store generated PNG plots.")
    parser.add_argument("--report", default="analysis/report.md", help="Path for the generated Markdown report.")
    args = parser.parse_args()

    # 1️⃣ Run the simulation + video capture
    run_simulation(args.csv, args.out, args.title)

    # 2️⃣ Generate plots from the (now‑filled) CSV
    generate_plots_from_csv(args.csv, args.plots)

    # 3️⃣ Build the markdown report linking everything together
    build_report(args.csv, args.out, args.plots, args.report)


if __name__ == "__main__":
    main()
