# -*- coding: utf-8 -*-
"""generate_plots.py

Utility script to read the simulation summary CSV and generate performance graphs.
The script creates a folder ``plots`` inside the ``analysis`` directory and saves
PNG images for each selected metric.  It **never writes any CSV files** – the
input CSV is only read.

Usage (run from the MuJoCo project root)::

    python analysis/generate_plots.py --csv demos/summary.csv

Arguments
---------
--csv   Path to the CSV file containing the summary data.
--out   Optional output directory for the plots (default: analysis/plots).

The CSV is expected to contain the following columns (at minimum)::

    controller, terrain, payload, wind_N, wind_dir_deg,
    RMSE_m, max_err_m, min_AGL_m

If any of the metric columns are missing the script will skip those plots.
"""


Utility script to read the simulation summary CSV and generate performance graphs.
The script creates a folder ``plots`` inside the ``analysis`` directory and saves
PNG images for each selected metric.

Usage (run from the MuJoCo project root)::

    python analysis/generate_plots.py --csv demos/summary.csv

Arguments
---------
--csv   Path to the CSV file containing the summary data.
--out   Optional output directory for the plots (default: analysis/plots).

The CSV is expected to contain the following columns (at minimum)::

    controller, terrain, payload, wind_N, wind_dir_deg,
    RMSE_m, max_err_m, min_AGL_m

If any of the metric columns are missing the script will skip those plots.
''' 

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> None:
    """Create ``path`` if it does not exist."""
    os.makedirs(path, exist_ok=True)

def plot_metric(df: pd.DataFrame, metric: str, out_dir: str) -> None:
    """Create a bar plot for *metric* grouped by controller and terrain.

    The plot shows the metric value for each (controller, terrain) combination.
    The resulting figure is saved as ``{metric}.png`` in *out_dir*.
    """
    if metric not in df.columns:
        print(f"[WARN] Column '{metric}' not found – skipping plot.")
        return
    # Drop rows where the metric is NaN
    data = df.dropna(subset=[metric])
    if data.empty:
        print(f"[WARN] No data for metric '{metric}' – skipping plot.")
        return

    # Create a pivot table: rows = terrain, columns = controller, values = metric
    pivot = data.pivot_table(index='terrain', columns='controller', values=metric, aggfunc='mean')
    ax = pivot.plot(kind='bar', figsize=(10, 6))
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f"{metric.replace('_', ' ')} by Controller and Terrain")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{metric}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")

def main(csv_path: str, out_dir: str) -> None:
    # Load CSV
    df = pd.read_csv(csv_path)
    # Ensure output directory exists
    ensure_dir(out_dir)
    # List of metric columns we want to visualise
    metrics = ['RMSE_m', 'max_err_m', 'min_AGL_m']
    for metric in metrics:
        plot_metric(df, metric, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance graphs from MuJoCo summary CSV.")
    parser.add_argument('--csv', required=True, help='Path to the summary CSV file')
    parser.add_argument('--out', default='analysis/plots', help='Directory to save the PNG plots')
    args = parser.parse_args()
    # Resolve paths relative to the project root for convenience
    csv_path = os.path.abspath(args.csv)
    out_dir = os.path.abspath(args.out)
    main(csv_path, out_dir)
