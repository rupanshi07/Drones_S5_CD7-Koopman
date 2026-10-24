# -*- coding: utf-8 -*-
"""
generate_summary_plots.py

Utility that:
  1️⃣ Reads all `drone_run_*.csv` files in the MuJoCo project.
  2️⃣ Computes per‑run statistics (max error, mean error, min AGL).
  3️⃣ Produces PNG plots that illustrate how *payload* and *wind* affect the
     error for each controller (koopman, pid, lqr).
  4️⃣ Writes a short Markdown report (`analysis/explanation.md`) explaining
     the observed trends.

The script does **not** create or modify any CSV files.
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_run_csvs(root: Path) -> list[Path]:
    """Return a list of all CSV files that look like simulation runs."""
    pattern = re.compile(r"drone_run_(?P<controller>[\w-]+)_.+\.csv$", re.IGNORECASE)
    return [p for p in root.rglob("*.csv") if pattern.search(p.name)]


def extract_controller(fname: str) -> str:
    """Extract the controller name from a filename like
    `drone_run_Koopman-MPC_mountain_095741.csv`."""
    match = re.search(r"drone_run_(?P<ctrl>[\w-]+)_", fname, re.IGNORECASE)
    return match.group("ctrl").lower() if match else "unknown"


def summarise_run(csv_path: Path) -> dict:
    """Read a single run CSV and return a dict with the needed statistics."""
    df = pd.read_csv(csv_path)

    # Guard against missing columns.
    err = df["err"] if "err" in df.columns else pd.Series([float("nan")])
    agl = df["agl"] if "agl" in df.columns else pd.Series([float("nan")])
    wind = df["wind_speed"] if "wind_speed" in df.columns else pd.Series([float("nan")])
    payload = df["payload_pct"] if "payload_pct" in df.columns else pd.Series([float("nan")])

    wind_val = wind.dropna().iloc[0] if not wind.dropna().empty else float("nan")
    payload_val = payload.dropna().iloc[0] if not payload.dropna().empty else float("nan")

    return {
        "controller": extract_controller(csv_path.name),
        "max_error": err.max(),
        "mean_error": err.mean(),
        "min_agl": agl.min(),
        "wind_speed": wind_val,
        "payload_pct": payload_val,
        "run_file": csv_path.name,
    }


def make_plot(df: pd.DataFrame, x_col: str, y_col: str, ctrl: str, out_dir: Path):
    """Create and save a simple scatter‑line plot for a given controller."""
    plt.figure(figsize=(8, 5))
    for _, grp in df.groupby("controller"):
        if grp.empty:
            continue
        plt.plot(
            grp[x_col],
            grp[y_col],
            marker="o",
            linestyle="-",
            label=grp["controller"].iloc[0].capitalize(),
        )
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(f"{ctrl.capitalize()}: {y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    plt.legend()
    out_path = out_dir / f"{ctrl}_{y_col}_vs_{x_col}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Plot saved: {out_path}")


def build_explanation(plots_dir: Path, report_path: Path):
    """Write a markdown file explaining each plot."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# How Wind & Payload Influence Controller Performance\n\n")
        f.write(
            "The three controllers (Koopman‑MPC, PID, LQR) were evaluated on a set of "
            "simulations where *payload* (percentage of nominal payload) and *wind speed* "
            "were varied. For each run we recorded the position error `err` and the altitude "
            "above ground level `agl`. The plots below show the **maximum position error** "
            "observed during a flight as a function of the two key parameters.\n\n"
        )
        for img in sorted(plots_dir.glob("*.png")):
            f.write(f"## {img.stem.replace('_', ' ').title()}\n\n")
            f.write(f"![{img.name}]({img.as_posix()})\n\n")
            if "payload" in img.stem:
                f.write("- **Interpretation**: As the payload percentage grows, the drone must "
                        "carry more mass, which typically degrades tracking accuracy. The "
                        "steeper the curve, the more sensitive that controller is to payload.\n")
            if "wind" in img.stem:
                f.write("- **Interpretation**: Higher wind speeds inject lateral disturbances. "
                        "Controllers that maintain a flatter curve are more robust to wind.\n")
            f.write("\n")
    print(f"[INFO] Explanation written to {report_path}")

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(project_root: Path, plots_dir: Path, report_path: Path):
    """Run the whole pipeline: discover CSVs, compute stats, plot, write report."""
    run_files = find_run_csvs(project_root)
    if not run_files:
        print("[WARN] No `drone_run_*.csv` files found – nothing to plot.")
        return

    records = [summarise_run(p) for p in run_files]
    summary_df = pd.DataFrame(records)

    plots_dir.mkdir(parents=True, exist_ok=True)

    for ctrl in summary_df["controller"].unique():
        ctrl_df = summary_df[summary_df["controller"] == ctrl].dropna(subset=["max_error"])
        if ctrl_df.empty:
            continue
        make_plot(ctrl_df, "payload_pct", "max_error", ctrl, plots_dir)
        make_plot(ctrl_df, "wind_speed", "max_error", ctrl, plots_dir)

    build_explanation(plots_dir, report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate performance plots (payload/wind vs max error) for each controller."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory of the MuJoCo project (default: current directory).",
    )
    parser.add_argument(
        "--plots",
        default="analysis/plots",
        help="Directory where PNG plots will be saved (default: analysis/plots).",
    )
    parser.add_argument(
        "--report",
        default="analysis/explanation.md",
        help="Path for the generated Markdown explanation file.",
    )
    args = parser.parse_args()
    main(Path(args.root), Path(args.plots), Path(args.report))
