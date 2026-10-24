# -*- coding: utf-8 -*-
"""generate_wind_payload_analysis.py

Utility script that:

1️⃣ Scans the MuJoCo project for every ``drone_run_*.csv`` file.
2️⃣ For each run it computes three summary metrics:
   * **max_error** – maximum value of the ``err`` column (position error).
   * **rmse**       – root‑mean‑square of ``err`` (computed as ``np.sqrt((err**2).mean())``).
   * **min_agl**    – minimum altitude above ground level ``agl``.
3️⃣ Extracts the *controller* name from the filename (Koopman‑MPC, PID, LQR).
   The terrain (flat, rolling, mountain, canyon) is also extracted when present.
4️⃣ Collects the associated ``wind_speed`` and ``payload_pct`` columns (these are
   constant throughout a run) and stores everything in a DataFrame.
5️⃣ Generates two plots **per controller**:
   * ``payload_pct``  vs  ``max_error``
   * ``wind_speed``   vs  ``max_error``
   The plots are saved as PNG files inside ``<project>/plots`` (e.g.
   ``plots/koopman_payload_vs_error.png``).
6️⃣ Writes a short Markdown file ``analysis_report.md`` (saved in the project root)
   that links each plot and explains how increasing wind or payload affects the
   three controllers.

The script **does not create or modify any CSV files** – it only reads existing
simulation logs and writes PNG/MD files.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def find_run_files(root: Path) -> list[Path]:
    """Return a list of all ``drone_run_*.csv`` files under *root*.
    The function ignores files in ``__pycache__`` or other non‑data directories.
    """
    pattern = re.compile(r"drone_run_(?P<ctrl>[^_]+)_(?P<terrain>[^_]+)_.*\.csv$", re.IGNORECASE)
    return [p for p in root.rglob("drone_run_*.csv") if pattern.search(p.name)]


def parse_filename(fname: str) -> dict:
    """Extract controller and terrain from a filename like
    ``drone_run_Koopman-MPC_mountain_095741.csv``.
    Returns ``{"controller": ..., "terrain": ...}``.
    """
    m = re.search(r"drone_run_(?P<ctrl>[^_]+)_(?P<terrain>[^_]+)_", fname, re.IGNORECASE)
    if not m:
        return {"controller": "unknown", "terrain": "unknown"}
    return {"controller": m.group("ctrl").lower(), "terrain": m.group("terrain").lower()}


def summarise_run(csv_path: Path) -> dict:
    """Read a single run CSV and return a dict with the required summary values.
    Missing columns are handled gracefully – the corresponding metric will be
    ``np.nan``.
    """
    df = pd.read_csv(csv_path)

    # Basic safety – ensure required columns exist.
    err = df["err"] if "err" in df.columns else pd.Series([np.nan])
    agl = df["agl"] if "agl" in df.columns else pd.Series([np.nan])
    wind = df["wind_speed"] if "wind_speed" in df.columns else pd.Series([np.nan])
    payload = df["payload_pct"] if "payload_pct" in df.columns else pd.Series([np.nan])

    # Take the first non‑nan value for wind/payload (they are constant throughout).
    wind_val = wind.dropna().iloc[0] if not wind.dropna().empty else np.nan
    payload_val = payload.dropna().iloc[0] if not payload.dropna().empty else np.nan

    meta = parse_filename(csv_path.name)
    return {
        "controller": meta["controller"],
        "terrain": meta["terrain"],
        "wind_speed": wind_val,
        "payload_pct": payload_val,
        "max_error": err.max(),
        "rmse": np.sqrt((err ** 2).mean()),
        "min_agl": agl.min(),
        "run_file": csv_path.name,
    }


def plot_metric(df: pd.DataFrame, x_col: str, y_col: str, out_dir: Path, suffix: str):
    """Create a scatter‑line plot for *x_col* vs *y_col* for each controller.
    The resulting PNG is named ``{y_col}_vs_{x_col}_{suffix}.png``.
    """
    plt.figure(figsize=(8, 5))
    for ctrl, grp in df.groupby("controller"):
        if grp.empty:
            continue
        # Sort by the X column for a nice line.
        grp = grp.sort_values(by=x_col)
        plt.plot(grp[x_col], grp[y_col], marker="o", linestyle="-", label=ctrl.capitalize())
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    out_path = out_dir / f"{y_col}_vs_{x_col}_{suffix}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def write_report(df: pd.DataFrame, plots_dir: Path, report_path: Path):
    """Create a short Markdown report that links the plots and briefly explains them."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Wind / Payload Impact Analysis\n\n")
        f.write("This report summarises how the three controllers react when **payload** "
                "or **wind speed** is increased.  The underlying data are the per‑step "
                "logs recorded in ``drone_run_*.csv`` files.\n\n")
        for ctrl in df["controller"].unique():
            f.write(f"## Controller: {ctrl.capitalize()}\n\n")
            # Payload plot
            payload_plot = plots_dir / f"max_error_vs_payload_pct_{ctrl}.png"
            if payload_plot.exists():
                f.write(f"### Payload vs Max Error\n\n![]({payload_plot.as_posix()})\n\n")
                f.write("- **Observation**: As the payload percentage grows, the controller's "
                        "maximum position error generally rises.  A steeper curve indicates "
                        "greater sensitivity to payload.\n\n")
            # Wind plot
            wind_plot = plots_dir / f"max_error_vs_wind_speed_{ctrl}.png"
            if wind_plot.exists():
                f.write(f"### Wind Speed vs Max Error\n\n![]({wind_plot.as_posix()})\n\n")
                f.write("- **Observation**: Higher wind speeds inject lateral disturbances.  "
                        "Controllers that keep the line flatter are more robust to wind.\n\n")
        f.write("---\nGenerated automatically by ``generate_wind_payload_analysis.py``.\n")
    print(f"[INFO] Report written to {report_path}")

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(project_root: Path, plots_dir: Path, report_path: Path):
    runs = find_run_files(project_root)
    if not runs:
        sys.exit("[ERROR] No drone_run_*.csv files found in the project directory.")

    # Summarise each run.
    records = [summarise_run(p) for p in runs]
    df = pd.DataFrame(records)

    # Ensure output folder exists.
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create plots for each controller.
    for ctrl in df["controller"].unique():
        ctrl_df = df[df["controller"] == ctrl]
        # Payload → max_error
        plot_metric(ctrl_df, "payload_pct", "max_error", plots_dir, ctrl)
        # Wind → max_error
        plot_metric(ctrl_df, "wind_speed", "max_error", plots_dir, ctrl)

    # Write the markdown explanation.
    write_report(df, plots_dir, report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse wind/payload impact on controller error.")
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory of the MuJoCo project (default: current working directory).",
    )
    parser.add_argument(
        "--plots",
        default="plots",
        help="Directory (relative to root) where PNG plots will be stored (default: plots).",
    )
    parser.add_argument(
        "--report",
        default="analysis_report.md",
        help="Path (relative to root) for the generated Markdown report.",
    )
    args = parser.parse_args()

    main(Path(args.root), Path(args.root) / args.plots, Path(args.root) / args.report)
