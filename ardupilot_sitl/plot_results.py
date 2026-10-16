"""
plot_results.py
---------------
Plot tracking results from ArduPilot SITL simulation runs.

Usage:
  python plot_results.py                           # plot latest result for each controller+scenario
  python plot_results.py --file results/foo.npz   # plot a specific file
  python plot_results.py --compare                 # side-by-side comparison of all controllers
  python plot_results.py --summary                 # bar chart from comparison_summary.csv

Saves plots to results/ directory as PNG files.
"""

import argparse
import numpy as np
import matplotlib
# Use non-interactive backend for saving PNGs without display issues
# Override with --show flag to pop up interactive window
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


COLORS = {
    "koopman": "#2196F3",   # blue
    "lqr":     "#4CAF50",   # green
    "pid":     "#FF5722",   # orange
}

SCENARIO_COLORS = {
    "clean":    "#333333",
    "wind":     "#2196F3",
    "payload":  "#FF9800",
    "combined": "#F44336",
}


# --------------------------------------------------------------------------- #
# Single episode plot
# --------------------------------------------------------------------------- #

def plot_single(npz_path, save_dir="results", show=False):
    d = np.load(npz_path, allow_pickle=True)
    t       = d["t"]
    pos     = d["pos"]
    ref_pos = d["ref_pos"]
    thrust  = d["thrust"]
    controller = str(d.get("controller", ["unknown"])[0])
    scenario   = str(d.get("scenario",   ["unknown"])[0])

    pos_err = np.linalg.norm(pos - ref_pos, axis=1)
    rmse    = np.sqrt(np.mean(pos_err**2))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"ArduPilot SITL – {controller.upper()} | Scenario: {scenario}\n"
        f"Tracking RMSE = {rmse:.4f} m",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    labels = ["X [m]", "Y [m]", "Z [m]"]

    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t, pos[:, i], color=COLORS.get(controller, "#555555"),
                linewidth=1.5, label="Actual")
        ax.plot(t, ref_pos[:, i], "r--", linewidth=1.5, label="Reference")
        ax.set_ylabel(labels[i])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel("Time [s]")

    # 3D trajectory
    ax3d = fig.add_subplot(gs[:, 1], projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2],
              color=COLORS.get(controller, "#555555"), linewidth=1.5, label="Actual")
    ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
              "r--", linewidth=1.5, label="Reference")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D Trajectory (ENU)")
    ax3d.legend(fontsize=8)

    save_path = Path(save_dir) / (Path(npz_path).stem + "_plot.png")
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return str(save_path)


# --------------------------------------------------------------------------- #
# Multi-controller comparison plot
# --------------------------------------------------------------------------- #

def plot_comparison(results_dir="results", scenario="clean", show=False):
    """
    Load all controller results for a given scenario and plot side-by-side.
    """
    controllers = ["koopman", "lqr", "pid"]
    data = {}
    for ctrl in controllers:
        files = sorted(Path(results_dir).glob(f"{ctrl}_{scenario}_*.npz"))
        if files:
            d = np.load(str(files[-1]), allow_pickle=True)
            data[ctrl] = d

    if not data:
        print(f"No results found for scenario '{scenario}' in {results_dir}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"ArduPilot SITL – Controller Comparison | Scenario: {scenario}",
        fontsize=13, fontweight="bold"
    )

    labels = ["X [m]", "Y [m]", "Z [m]"]
    for ctrl, d in data.items():
        for i in range(3):
            axes[i].plot(d["t"], d["pos"][:, i],
                         color=COLORS.get(ctrl, "#888"), linewidth=1.5,
                         label=ctrl.upper())

    # Reference (same for all)
    first = next(iter(data.values()))
    for i in range(3):
        axes[i].plot(first["t"], first["ref_pos"][:, i],
                     "k--", linewidth=1.5, label="Reference", alpha=0.5)
        axes[i].set_ylabel(labels[i])
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")

    save_path = Path(results_dir) / f"comparison_{scenario}.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary bar chart
# --------------------------------------------------------------------------- #

def plot_summary_bar(csv_path="results/comparison_summary.csv",
                     save_dir="results", show=False):
    import csv as csv_module
    rows = []
    with open(csv_path) as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No comparison summary found. Run run_comparison.py first.")
        return

    controllers = ["koopman", "lqr", "pid"]
    scenarios   = ["clean", "wind", "payload", "combined"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scenarios))
    width = 0.25

    for i, ctrl in enumerate(controllers):
        rmse_vals = []
        for scenario in scenarios:
            found = [r for r in rows if r["controller"] == ctrl and r["scenario"] == scenario]
            rmse_vals.append(float(found[0]["rmse_m"]) if found else float("nan"))
        bars = ax.bar(x + i * width, rmse_vals, width,
                      label=ctrl.upper(), color=COLORS.get(ctrl, "#888"),
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, rmse_vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels([s.upper() for s in scenarios])
    ax.set_ylabel("Tracking RMSE [m]")
    ax.set_title("ArduPilot SITL – Koopman-MPC vs LQR vs PID\nTracking RMSE by Scenario",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)

    save_path = Path(save_dir) / "rmse_comparison_bar.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return str(save_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ArduPilot SITL results")
    parser.add_argument("--file", default=None, help="Specific .npz file to plot")
    parser.add_argument("--compare", action="store_true",
                        help="Plot comparison across all controllers")
    parser.add_argument("--scenario", default="clean",
                        choices=["clean", "wind", "payload", "combined"],
                        help="Scenario for comparison plot")
    parser.add_argument("--summary", action="store_true",
                        help="Plot RMSE bar chart from comparison_summary.csv")
    parser.add_argument("--results-dir", default="results",
                        help="Results directory")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive plot window (requires display)")
    args = parser.parse_args()

    if args.file:
        plot_single(args.file, show=args.show)
    elif args.compare:
        for s in ["clean", "wind", "payload", "combined"]:
            plot_comparison(args.results_dir, s, show=args.show)
    elif args.summary:
        plot_summary_bar(save_dir=args.results_dir, show=args.show)
    else:
        # Plot all .npz files in results/ that don't have a matching plot
        results_dir = Path(args.results_dir)
        npz_files = sorted(results_dir.glob("*.npz"))
        if not npz_files:
            print(f"No .npz files found in {results_dir}")
            print("Run sitl_controller.py or run_comparison.py first.")
        else:
            for f in npz_files:
                plot_path = results_dir / (f.stem + "_plot.png")
                if not plot_path.exists():
                    print(f"Plotting {f.name} ...")
                    plot_single(str(f), show=args.show)
            print("\nAll plots saved to results/")
            print("For comparison bar chart: python plot_results.py --summary")
