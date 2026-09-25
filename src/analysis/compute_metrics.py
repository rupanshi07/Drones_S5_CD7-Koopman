"""
Metrics, Analysis, and Comparison
----------------------------------
Loads the saved simulation traces for Koopman-MPC, PID, and LQR across all
six disturbance scenarios and computes the comparison metrics:

  - RMSE tracking error       (z-position vs. target, whole run)
  - Settling time             (time after disturbance onset until the
                                controller stays within tolerance of target)
  - Control effort            (mean squared motor-RPM deviation from hover)
  - Max transient deviation   (largest |z - target| in the window after
                                disturbance onset)

Outputs:
  results/analysis/metrics_table.csv   -- one row per (controller, scenario)
  results/analysis/metrics_summary.png -- 2x2 bar chart comparing controllers
  results/analysis/settling_plot_scenario_<n>.png -- z-tracking overlay
      with settling-tolerance band, per scenario

Usage:
    python src/analysis/compute_metrics.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

CTRL_FREQ = 48
DT = 1.0 / CTRL_FREQ

# Shared target height convention used by all three controllers
# (see src/mpc/mpc_controller.py / lqr_controller.py / run_scenario_pid.py)
FINAL_TARGET_Z = 0.1125

# Hover baseline RPM, used to compute "control effort" as effort *above*
# what's needed just to hover (so a controller isn't penalized for the
# baseline thrust every controller must apply).
MASS, GRAVITY, KF = 0.027, 9.8, 3.16e-10
HOVER_RPM = np.sqrt((MASS * GRAVITY) / (4 * KF))

# Settling tolerance band around target (meters). 5% of a 0.1125m target is
# tiny, so use an absolute band instead -- 1 cm is a reasonable "settled"
# criterion at this scale.
SETTLE_TOL = 0.01
# How long the trajectory must stay inside the band, continuously, to count
# as "settled" (avoids single-sample flukes counting as settling).
SETTLE_HOLD_STEPS = int(0.5 * CTRL_FREQ)  # 0.5s

CONTROLLERS = {
    "MPC": lambda n: f"scenario_{n}.npz",
    "PID": lambda n: f"pid_scenario_{n}.npz",
    "LQR": lambda n: f"lqr_scenario_{n}.npz",
}
SCENARIOS = [1, 2, 3, 4, 5, 6]
SCENARIO_LABELS = {
    1: "Nominal",
    2: "Windy, selector OFF",
    3: "Windy, selector ON",
    4: "Nominal->Windy transition",
    5: "Payload mid-flight",
    6: "Payload + wind combined",
}


def load_run(controller, scenario):
    path = os.path.join(RESULTS_DIR, CONTROLLERS[controller](scenario))
    if not os.path.exists(path):
        return None
    d = np.load(path)
    z = d["states"][:, 2]
    actions = d["actions"]
    onset = int(d["disturbance_onset_step"])
    return z, actions, onset


def rmse_tracking_error(z):
    return float(np.sqrt(np.mean((z - FINAL_TARGET_Z) ** 2)))


def control_effort(actions):
    dev = actions - HOVER_RPM
    return float(np.mean(dev ** 2))


def max_transient_deviation(z, onset):
    if onset is None or onset < 0:
        window = z
    else:
        window = z[onset:]
    if len(window) == 0:
        return float("nan")
    return float(np.max(np.abs(window - FINAL_TARGET_Z)))


def settling_time(z, onset):
    """Seconds from disturbance onset until |z - target| < SETTLE_TOL and
    stays there for SETTLE_HOLD_STEPS consecutive samples. Returns np.nan
    if the run never settles (e.g. it crashed and flatlined off-target)."""
    start = 0 if (onset is None or onset < 0) else onset
    err = np.abs(z[start:] - FINAL_TARGET_Z)
    within = err < SETTLE_TOL

    run_len = 0
    for i, ok in enumerate(within):
        run_len = run_len + 1 if ok else 0
        if run_len >= SETTLE_HOLD_STEPS:
            settle_idx = i - SETTLE_HOLD_STEPS + 1
            return settle_idx * DT
    return float("nan")


def main():
    rows = []
    for scenario in SCENARIOS:
        for controller in CONTROLLERS:
            run = load_run(controller, scenario)
            if run is None:
                print(f"  [skip] no data for {controller} scenario {scenario}")
                continue
            z, actions, onset = run
            rows.append({
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "controller": controller,
                "rmse_tracking_error_m": rmse_tracking_error(z),
                "settling_time_s": settling_time(z, onset),
                "control_effort_rpm2": control_effort(actions),
                "max_transient_deviation_m": max_transient_deviation(z, onset),
                "final_z_m": float(z[-1]),
                "min_z_m": float(z.min()),
                "flagged_possible_crash": bool(z.min() < 0.02),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "metrics_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    print(df.to_string(index=False))

    metrics = [
        ("rmse_tracking_error_m", "RMSE Tracking Error (m)"),
        ("settling_time_s", "Settling Time (s)"),
        ("control_effort_rpm2", "Control Effort (RPM^2 dev. from hover)"),
        ("max_transient_deviation_m", "Max Transient Deviation (m)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.25
    x = np.arange(len(SCENARIOS))
    colors = {"MPC": "tab:blue", "PID": "tab:orange", "LQR": "tab:green"}

    for ax, (col, title) in zip(axes.flat, metrics):
        for i, controller in enumerate(CONTROLLERS):
            sub = df[df["controller"] == controller].set_index("scenario")
            vals = [sub[col].get(s, np.nan) for s in SCENARIOS]
            ax.bar(x + (i - 1) * width, vals, width, label=controller,
                   color=colors[controller])
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in SCENARIOS])
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Koopman-MPC vs PID vs LQR across disturbance scenarios")
    fig.tight_layout()
    summary_path = os.path.join(OUT_DIR, "metrics_summary.png")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"Saved {summary_path}")

    for scenario in SCENARIOS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for controller in CONTROLLERS:
            run = load_run(controller, scenario)
            if run is None:
                continue
            z, _, onset = run
            t = np.arange(len(z)) * DT
            ax.plot(t, z, label=controller, color=colors[controller])
            if onset is not None and onset >= 0:
                ax.axvline(onset * DT, color="gray", linestyle=":", alpha=0.6)
        ax.axhline(FINAL_TARGET_Z, color="red", linestyle="--", label="Target Z")
        ax.axhspan(FINAL_TARGET_Z - SETTLE_TOL, FINAL_TARGET_Z + SETTLE_TOL,
                   color="red", alpha=0.08, label="Settle band")
        ax.set_title(f"Scenario {scenario}: {SCENARIO_LABELS[scenario]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z Position (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"settling_plot_scenario_{scenario}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    print(f"Saved per-scenario settling plots to {OUT_DIR}")


if __name__ == "__main__":
    main()