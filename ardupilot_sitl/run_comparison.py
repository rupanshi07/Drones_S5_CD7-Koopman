"""
run_comparison.py
-----------------
Run the full controller x scenario comparison matrix for ArduPilot SITL.
Mirrors Phase-2's comparison/run_comparison.py EXACTLY.

Runs:
  Koopman-MPC x {clean, wind, payload, combined}
  LQR         x {clean, wind, payload, combined}
  PID         x {clean, wind, payload, combined}

= 12 simulation episodes total.

Metrics computed (identical to Phase-2 compute_metrics):
  - overall RMSE  (m)
  - max error     (m)
  - relative degradation vs clean (% increase in RMSE under disturbance)
  - control effort (mean squared thrust deviation from hover)

After all episodes complete, prints a summary table and saves results to
results/comparison_summary.csv

Usage:
  python run_comparison.py [--duration 15] [--ctrl-freq 20] [--no-gui]

Prerequisites:
  - SITL running (Mission Planner -> Simulation -> Run SITL)
  - setup_vehicle.py has been run (or re-run between episodes if needed)
"""

import argparse
import subprocess
import sys
import time
import csv
import os
from pathlib import Path

import numpy as np


CONTROLLERS = ["koopman", "lqr", "pid"]
SCENARIOS   = ["clean", "wind", "payload", "combined"]


def run_episode(controller, scenario, duration, ctrl_freq, port, no_gui):
    """Run one episode via subprocess and return the result npz path."""
    cmd = [
        sys.executable, "sitl_controller.py",
        "--controller", controller,
        "--scenario", scenario,
        "--duration", str(duration),
        "--ctrl-freq", str(ctrl_freq),
        "--port", str(port),
    ]
    if no_gui:
        cmd.append("--no-gui")

    print(f"\n{'='*60}")
    print(f"  Running: {controller.upper()} | {scenario}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: Episode returned non-zero exit code {result.returncode}")

    # Find the latest result file for this controller+scenario
    results_dir = Path("results")
    pattern = f"{controller}_{scenario}_*.npz"
    files = sorted(results_dir.glob(pattern))
    if files:
        return str(files[-1])
    return None


def load_metrics(npz_path):
    """
    Load tracking metrics from a result .npz file.
    Metrics are IDENTICAL to Phase-2's compute_metrics() in run_comparison.py.
    """
    if npz_path is None or not Path(npz_path).exists():
        return {"rmse": float("nan"), "max_err": float("nan"),
                "mean_thrust": float("nan"), "control_effort": float("nan")}
    d = np.load(npz_path, allow_pickle=True)
    pos_err = np.linalg.norm(d["pos"] - d["ref_pos"], axis=1)
    rmse = np.sqrt(np.mean(pos_err**2))

    # control_effort: mean squared THRUST deviation from hover (0.5)
    # Mirrors Phase-2's mean squared RPM deviation from hover:
    #   control_effort = mean((rpm - RPM_HOVER)^2)
    # Since we log thrust [0,1] instead of RPM, we compute deviation from 0.5
    thrust = d["thrust"] if "thrust" in d else np.full(len(pos_err), 0.5)
    ctrl_effort = float(np.mean((thrust - 0.5) ** 2))

    return {
        "rmse":           rmse,
        "max_err":        float(pos_err.max()),
        "mean_thrust":    float(thrust.mean()),
        "control_effort": ctrl_effort,
    }


def print_table(results):
    """Print a comparison table like Phase 2's analysis."""
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE: Koopman-MPC vs LQR vs PID  (ArduPilot SITL)")
    print("=" * 80)
    header = f"{'Scenario':<12} | {'Koopman RMSE':>13} | {'LQR RMSE':>10} | {'PID RMSE':>10}"
    print(header)
    print("-" * 55)
    for scenario in SCENARIOS:
        k_rmse = results.get(("koopman", scenario), {}).get("rmse", float("nan"))
        l_rmse = results.get(("lqr",     scenario), {}).get("rmse", float("nan"))
        p_rmse = results.get(("pid",     scenario), {}).get("rmse", float("nan"))
        row = (f"{scenario:<12} | {k_rmse:>12.4f}m | {l_rmse:>9.4f}m | "
               f"{p_rmse:>9.4f}m")
        print(row)
    print("=" * 80)

    # Relative degradation vs clean
    print("\n  Relative degradation vs clean scenario:")
    print(f"{'':12} | {'Koopman':>10} | {'LQR':>10} | {'PID':>10}")
    print("-" * 50)
    for scenario in SCENARIOS[1:]:  # skip clean
        for ctrl in CONTROLLERS:
            clean = results.get((ctrl, "clean"), {}).get("rmse", 1e-9)
            curr  = results.get((ctrl, scenario), {}).get("rmse", float("nan"))
            rel   = (curr - clean) / (clean + 1e-9) * 100
            results[(ctrl, scenario)]["rel_degradation"] = rel

        k_rel = results.get(("koopman", scenario), {}).get("rel_degradation", float("nan"))
        l_rel = results.get(("lqr",     scenario), {}).get("rel_degradation", float("nan"))
        p_rel = results.get(("pid",     scenario), {}).get("rel_degradation", float("nan"))
        print(f"{scenario:<12} | {k_rel:>+9.1f}% | {l_rel:>+9.1f}% | {p_rel:>+9.1f}%")
    print()


def save_csv(results, path="results/comparison_summary.csv"):
    Path(path).parent.mkdir(exist_ok=True)
    rows = []
    for (ctrl, scenario), metrics in results.items():
        rows.append({
            "controller":     ctrl,
            "scenario":       scenario,
            "rmse_m":         round(metrics.get("rmse",           float("nan")), 4),
            "max_err_m":      round(metrics.get("max_err",        float("nan")), 4),
            "mean_thrust":    round(metrics.get("mean_thrust",    float("nan")), 4),
            "control_effort": round(metrics.get("control_effort", float("nan")), 6),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Comparison] Summary saved to {path}")


def run_comparison(args):
    """Run the full comparison matrix."""
    print("\n" + "=" * 70)
    print("  FULL COMPARISON: ArduPilot SITL Koopman-MPC vs LQR vs PID")
    print(f"  {len(CONTROLLERS) * len(SCENARIOS)} episodes total")
    print(f"  Duration per episode: {args.duration}s  |  Freq: {args.ctrl_freq}Hz")
    print("=" * 70)
    print()
    print("Prerequisites:")
    print("  1. Mission Planner SITL is running (Simulation -> Multirotor -> Run SITL)")
    print("  2. setup_vehicle.py has been run at least once")
    print()
    input("Press ENTER to start the comparison run (or Ctrl+C to cancel)...")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_results = {}
    for controller in CONTROLLERS:
        for scenario in SCENARIOS:
            npz_path = run_episode(controller, scenario, args.duration,
                                   args.ctrl_freq, args.port, args.no_gui)
            metrics = load_metrics(npz_path)
            all_results[(controller, scenario)] = metrics
            print(f"  -> RMSE={metrics['rmse']:.4f}m  max_err={metrics['max_err']:.4f}m")

            # Brief pause between episodes (re-arm)
            if not (controller == CONTROLLERS[-1] and scenario == SCENARIOS[-1]):
                print(f"\n[Wait] Pausing 5s between episodes ...")
                print("  (Re-run setup_vehicle.py if drone is not responsive)")
                time.sleep(5.0)

    print_table(all_results)
    save_csv(all_results)

    # --- Relative degradation -- IDENTICAL to Phase-2 run_comparison.py output ---
    print("\n\n=== RELATIVE DEGRADATION (the core question this project asks) ===")
    print("How much worse does each controller get, percentage-wise, going from")
    print("clean conditions to the worst disturbance? This matters more than raw")
    print("tracking tightness, since PID and Koopman-MPC may simply be tuned to")
    print("different absolute aggressiveness.\n")
    print(f"{'controller':<14} {'clean RMSE':>12} {'worst RMSE':>12} {'degradation':>14}")
    for controller in CONTROLLERS:
        clean_rmse = all_results.get((controller, "clean"), {}).get("rmse", float("nan"))
        worst_rmse = all_results.get((controller, "combined"), {}).get("rmse", float("nan"))
        if not (float("nan") in (clean_rmse, worst_rmse)) and clean_rmse > 0:
            degradation_pct = 100 * (worst_rmse - clean_rmse) / clean_rmse
        else:
            degradation_pct = float("nan")
        print(f"{controller:<14} {clean_rmse:>12.4f} {worst_rmse:>12.4f} "
              f"{degradation_pct:>13.1f}%")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full controller comparison on ArduPilot SITL"
    )
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Episode duration seconds")
    parser.add_argument("--ctrl-freq", type=float, default=20.0,
                        help="Control frequency Hz")
    parser.add_argument("--port", type=int, default=5760)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    run_comparison(args)
