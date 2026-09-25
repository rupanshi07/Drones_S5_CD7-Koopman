"""
FINAL Metrics, Analysis, and Comparison
----------------------------------------

Compares:
    Koopman-MPC
    PID
    LQR

Across:
    1. Nominal
    2. Windy, selector OFF
    3. Windy, selector ON
    4. Nominal -> Windy transition
    5. Payload mid-flight
    6. Payload + Wind combined

Metrics:
    - RMSE tracking error
    - Settling time
    - Control effort RMS
    - Integrated squared control effort
    - Maximum transient deviation
    - Crash / failure status

Important:
    If a controller crashes during a run, metrics are calculated only
    up to the first detected failure. The run is separately marked
    as failed. Post-crash reset data is NOT treated as controller
    performance.

Run:
    python src/analysis/Compute_metrics_v2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "results"
OUT_DIR = os.path.join(RESULTS_DIR, "analysis")

os.makedirs(OUT_DIR, exist_ok=True)

CTRL_FREQ = 48
DT = 1.0 / CTRL_FREQ
DURATION_SEC_TOTAL = 15  # matches DURATION_SEC in run_scenario*.py
DISTURBANCE_ONSET_S = 7.5  # wind transition / payload drop trigger, matches
                            # NUM_STEPS // 2 in run_scenario*.py

FINAL_TARGET_Z = 0.3

# Physical hover RPM:
# sqrt(m*g / (4*KF))
MASS = 0.027
GRAVITY = 9.8
KF = 3.16e-10

HOVER_RPM = np.sqrt(
    (MASS * GRAVITY) / (4.0 * KF)
)

RPM_MIN = 0.0
RPM_MAX = 25000.0

# Failure thresholds
FLIP_THRESHOLD_RAD = np.pi / 2
GROUND_THRESHOLD_M = 0.02

# Settling definition
SETTLE_TOL = 0.01
SETTLE_HOLD_STEPS = int(0.5 * CTRL_FREQ)


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
    4: "Nominal -> Windy",
    5: "Payload mid-flight",
    6: "Payload + wind",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_run(controller, scenario):

    path = os.path.join(
        RESULTS_DIR,
        CONTROLLERS[controller](scenario)
    )

    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)

    states = data["states"]
    actions = data["actions"]

    onset = int(data["disturbance_onset_step"])

    return states, actions, onset


# ============================================================
# FAILURE DETECTION
# ============================================================

def detect_failure(states, actions):

    z = states[:, 2]
    roll = states[:, 7]
    pitch = states[:, 8]

    # Ground contact
    ground_idx = np.where(
        z < GROUND_THRESHOLD_M
    )[0]

    # Flip
    flip_idx = np.where(
        (np.abs(roll) > FLIP_THRESHOLD_RAD) |
        (np.abs(pitch) > FLIP_THRESHOLD_RAD)
    )[0]

    # Look for sudden altitude discontinuity.
    # This is a useful indicator of env.reset().
    z_jump_idx = np.where(
        np.abs(np.diff(z)) > 0.08
    )[0]

    candidates = []

    if len(ground_idx):
        candidates.append(int(ground_idx[0]))

    if len(flip_idx):
        candidates.append(int(flip_idx[0]))

    if len(z_jump_idx):
        candidates.append(int(z_jump_idx[0] + 1))

    if not candidates:
        return {
            "crash_flag": False,
            "failure_step": -1,
            "failure_time_s": np.nan,
            "failure_reason": "none"
        }

    failure_step = min(candidates)

    reasons = []

    if len(ground_idx) and ground_idx[0] == failure_step:
        reasons.append("ground_contact")

    if len(flip_idx) and flip_idx[0] == failure_step:
        reasons.append("flip")

    if len(z_jump_idx) and z_jump_idx[0] + 1 == failure_step:
        reasons.append("reset_jump")

    return {
        "crash_flag": True,
        "failure_step": failure_step,
        "failure_time_s": failure_step * DT,
        "failure_reason": "+".join(reasons)
    }


# ============================================================
# TRACKING ERROR
# ============================================================

def tracking_rmse(z):

    error = z - FINAL_TARGET_Z

    return float(
        np.sqrt(np.mean(error ** 2))
    )


# ============================================================
# CONTROL EFFORT
# ============================================================

def control_effort(actions):

    # Deviation from physical hover RPM
    deviation = actions - HOVER_RPM

    rms = float(
        np.sqrt(np.mean(deviation ** 2))
    )

    # Integral of squared control deviation
    # Units: RPM^2 * seconds
    ise = float(
        np.sum(deviation ** 2) * DT
    )

    max_deviation = float(
        np.max(np.abs(deviation))
    )

    return rms, ise, max_deviation


# ============================================================
# MAX TRANSIENT DEVIATION
# ============================================================

def max_transient_deviation(z, onset):

    if onset is None or onset < 0:
        window = z
    else:
        if onset >= len(z):
            return np.nan

        window = z[onset:]

    if len(window) == 0:
        return np.nan

    return float(
        np.max(
            np.abs(window - FINAL_TARGET_Z)
        )
    )


# ============================================================
# SETTLING TIME
# ============================================================

def settling_time(z, onset):

    if onset is None or onset < 0:
        start = 0
    else:
        start = onset

    if start >= len(z):
        return np.nan

    error = np.abs(
        z[start:] - FINAL_TARGET_Z
    )

    within_band = error <= SETTLE_TOL

    # Proper settling:
    # first point from which the signal remains inside
    # the tolerance band until the end of the valid run.
    for i in range(len(within_band)):

        if within_band[i]:

            remaining = within_band[i:]

            if np.all(remaining):

                return float(i * DT)

    return np.nan


# ============================================================
# BUILD METRICS TABLE
# ============================================================

def build_metrics_table():

    rows = []

    for scenario in SCENARIOS:

        for controller in CONTROLLERS:

            run = load_run(controller, scenario)

            if run is None:
                print(
                    f"[MISSING] {controller} scenario {scenario}"
                )
                continue

            states, actions, onset = run

            # ------------------------------------------------
            # Detect failure BEFORE calculating metrics
            # ------------------------------------------------

            failure = detect_failure(
                states,
                actions
            )

            if failure["crash_flag"]:

                valid_end = failure["failure_step"] + 1

            else:

                valid_end = len(states)

            valid_states = states[:valid_end]
            valid_actions = actions[:valid_end]

            z = valid_states[:, 2]

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            rmse = tracking_rmse(z)

            rms_effort, ise_effort, max_effort = \
                control_effort(valid_actions)

            max_dev = max_transient_deviation(
                z,
                onset
            )

            settle = settling_time(
                z,
                onset
            )

            # If failure happened before settling,
            # explicitly mark as NaN.
            if failure["crash_flag"]:

                if (
                    failure["failure_step"] <=
                    (onset if onset >= 0 else 0)
                ):
                    settle = np.nan

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------

            min_z = float(
                np.min(valid_states[:, 2])
            )

            max_roll = float(
                np.max(
                    np.abs(valid_states[:, 7])
                )
            )

            max_pitch = float(
                np.max(
                    np.abs(valid_states[:, 8])
                )
            )

            rpm_low = int(
                (valid_actions <= RPM_MIN + 1e-6).sum()
            )

            rpm_high = int(
                (valid_actions >= RPM_MAX - 1e-6).sum()
            )

            row = {

                "scenario":
                    scenario,

                "scenario_label":
                    SCENARIO_LABELS[scenario],

                "controller":
                    controller,

                # ------------------------------
                # Performance
                # ------------------------------

                "rmse_tracking_error_m":
                    rmse,

                "settling_time_s":
                    settle,

                "max_transient_deviation_m":
                    max_dev,

                # ------------------------------
                # Control effort
                # ------------------------------

                "control_effort_rms_rpm":
                    rms_effort,

                "control_effort_ise_rpm2_s":
                    ise_effort,

                "control_effort_max_dev_rpm":
                    max_effort,

                # ------------------------------
                # Final state
                # ------------------------------

                "final_z_m":
                    float(valid_states[-1, 2]),

                "min_z_m":
                    min_z,

                # ------------------------------
                # Diagnostics
                # ------------------------------

                "max_roll_deg":
                    np.degrees(max_roll),

                "max_pitch_deg":
                    np.degrees(max_pitch),

                "rpm_saturation_low":
                    rpm_low,

                "rpm_saturation_high":
                    rpm_high,

                "crash_flag":
                    failure["crash_flag"],

                "failure_step":
                    failure["failure_step"],

                "failure_time_s":
                    failure["failure_time_s"],

                "failure_reason":
                    failure["failure_reason"],

                "valid_steps":
                    valid_end,

                "valid_duration_s":
                    valid_end * DT
            }

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# PERCENTAGE COMPARISON
# ============================================================

def percentage_comparison(df):

    rows = []

    metrics = [

        (
            "rmse_tracking_error_m",
            "RMSE"
        ),

        (
            "settling_time_s",
            "Settling Time"
        ),

        (
            "control_effort_rms_rpm",
            "Control Effort RMS"
        ),

        (
            "max_transient_deviation_m",
            "Maximum Transient Deviation"
        )
    ]

    for scenario in SCENARIOS:

        sub = df[
            df["scenario"] == scenario
        ].set_index("controller")

        if "MPC" not in sub.index:
            continue

        for column, label in metrics:

            if column not in sub.columns:
                continue

            mpc = sub.loc["MPC", column]

            for controller in ["PID", "LQR"]:

                if controller not in sub.index:
                    continue

                value = sub.loc[
                    controller,
                    column
                ]

                # Do NOT claim percentage improvement
                # against a crashed MPC run.
                if (
                    bool(sub.loc["MPC", "crash_flag"])
                    or pd.isna(mpc)
                    or pd.isna(value)
                    or mpc == 0
                ):
                    improvement = np.nan
                else:
                    improvement = (
                        (mpc - value) /
                        mpc
                    ) * 100.0

                rows.append({

                    "scenario":
                        scenario,

                    "scenario_label":
                        SCENARIO_LABELS[scenario],

                    "metric":
                        label,

                    "comparison":
                        f"{controller} vs MPC",

                    "mpc_value":
                        mpc,

                    "comparison_value":
                        value,

                    "pct_improvement":
                        improvement
                })

    return pd.DataFrame(rows)


# ============================================================
# WINNER SUMMARY
# ============================================================

def winner_summary(df):

    metrics = [

        (
            "rmse_tracking_error_m",
            "RMSE"
        ),

        (
            "settling_time_s",
            "Settling Time"
        ),

        (
            "control_effort_rms_rpm",
            "Control Effort RMS"
        ),

        (
            "max_transient_deviation_m",
            "Maximum Transient Deviation"
        )
    ]

    rows = []

    for scenario in SCENARIOS:

        sub = df[
            df["scenario"] == scenario
        ]

        # Only successful controllers participate
        # in metric winner selection.
        sub = sub[
            sub["crash_flag"] == False
        ]

        for column, label in metrics:

            valid = sub[
                ["controller", column]
            ].dropna()

            if len(valid) == 0:

                winner = "No successful controller"

            else:

                idx = valid[column].idxmin()

                winner = valid.loc[
                    idx,
                    "controller"
                ]

            rows.append({

                "scenario":
                    scenario,

                "scenario_label":
                    SCENARIO_LABELS[scenario],

                "metric":
                    label,

                "best_controller":
                    winner
            })

    return pd.DataFrame(rows)


# ============================================================
# COMPLETION / CRASH-RATE SUMMARY
# ============================================================

def completion_summary(df):
    """
    One row per (scenario, controller) giving whether the run
    survived the full 15s, and if not, how much of the scenario
    it actually completed and why it failed.

    This matters here specifically because MPC crashes in every
    single scenario (see failure_reason), which silently makes
    several disturbance-specific metrics (settling_time_s,
    max_transient_deviation_m) come out as NaN -- not because of
    a data problem, but because MPC is down before the disturbance
    (wind transition / payload drop, both at t=7.5s) ever triggers
    in scenarios 4-6. This table makes that visible instead of
    leaving it implicit in blank cells elsewhere.
    """

    rows = []

    for _, row in df.iterrows():

        completed_pct = 100.0 * row["valid_duration_s"] / DURATION_SEC_TOTAL

        rows.append({

            "scenario":
                row["scenario"],

            "scenario_label":
                row["scenario_label"],

            "controller":
                row["controller"],

            "completed":
                not bool(row["crash_flag"]),

            "duration_completed_s":
                row["valid_duration_s"],

            "pct_scenario_completed":
                completed_pct,

            "failure_reason":
                row["failure_reason"],

            "failed_before_disturbance_onset":
                bool(row["crash_flag"]) and
                (row["failure_time_s"] < DISTURBANCE_ONSET_S)
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def create_plots(df):

    # --------------------------------------------------------
    # 1. RMSE comparison
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    x = np.arange(len(SCENARIOS))
    width = 0.25

    for i, controller in enumerate(
        ["MPC", "PID", "LQR"]
    ):

        sub = df[
            df["controller"] == controller
        ].set_index("scenario")

        values = [
            sub["rmse_tracking_error_m"].get(
                s,
                np.nan
            )
            for s in SCENARIOS
        ]

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=controller
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [f"S{s}" for s in SCENARIOS]
    )

    ax.set_ylabel(
        "RMSE Tracking Error (m)"
    )

    ax.set_xlabel(
        "Scenario"
    )

    ax.set_title(
        "RMSE Comparison: Koopman-MPC vs PID vs LQR"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUT_DIR,
            "rmse_comparison.png"
        ),
        dpi=200
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 2. Control effort
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for i, controller in enumerate(
        ["MPC", "PID", "LQR"]
    ):

        sub = df[
            df["controller"] == controller
        ].set_index("scenario")

        values = [
            sub["control_effort_rms_rpm"].get(
                s,
                np.nan
            )
            for s in SCENARIOS
        ]

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=controller
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [f"S{s}" for s in SCENARIOS]
    )

    ax.set_ylabel(
        "RMS RPM Deviation from Hover"
    )

    ax.set_xlabel(
        "Scenario"
    )

    ax.set_title(
        "Control Effort Comparison"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUT_DIR,
            "control_effort_comparison.png"
        ),
        dpi=200
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 3. Tracking plots
    # --------------------------------------------------------

    for scenario in SCENARIOS:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        for controller in [
            "MPC",
            "PID",
            "LQR"
        ]:

            run = load_run(
                controller,
                scenario
            )

            if run is None:
                continue

            states, actions, onset = run

            failure = detect_failure(
                states,
                actions
            )

            if failure["crash_flag"]:
                end = failure["failure_step"] + 1
            else:
                end = len(states)

            z = states[:end, 2]

            t = np.arange(len(z)) * DT

            ax.plot(
                t,
                z,
                label=controller
            )

        ax.axhline(
            FINAL_TARGET_Z,
            linestyle="--",
            label="Target Z"
        )

        ax.axhspan(
            FINAL_TARGET_Z - SETTLE_TOL,
            FINAL_TARGET_Z + SETTLE_TOL,
            alpha=0.10,
            label="Â±1 cm settling band"
        )

        ax.set_xlabel(
            "Time (s)"
        )

        ax.set_ylabel(
            "Altitude Z (m)"
        )

        ax.set_title(
            f"Scenario {scenario}: "
            f"{SCENARIO_LABELS[scenario]}"
        )

        ax.legend()

        ax.grid(
            alpha=0.3
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                OUT_DIR,
                f"tracking_scenario_{scenario}.png"
            ),
            dpi=200
        )

        plt.close(fig)

    # --------------------------------------------------------
    # 4. Completion rate (% of scenario survived before failure)
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for i, controller in enumerate(
        ["MPC", "PID", "LQR"]
    ):

        sub = df[
            df["controller"] == controller
        ].set_index("scenario")

        values = [
            100.0 * sub["valid_duration_s"].get(s, np.nan) / DURATION_SEC_TOTAL
            for s in SCENARIOS
        ]

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=controller
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [f"S{s}" for s in SCENARIOS]
    )

    ax.set_ylabel(
        "% of Scenario Completed Before Failure"
    )

    ax.set_xlabel(
        "Scenario"
    )

    ax.set_ylim(0, 105)

    ax.set_title(
        "Completion Rate: how much of each 15s run each "
        "controller survived"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUT_DIR,
            "completion_rate_comparison.png"
        ),
        dpi=200
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL CONTROLLER METRICS")
    print("=" * 70)

    print(
        f"Physical hover RPM = {HOVER_RPM:.6f}"
    )

    df = build_metrics_table()

    metrics_path = os.path.join(
        OUT_DIR,
        "metrics_table_final.csv"
    )

    df.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"\nSaved: {metrics_path}"
    )

    # --------------------------------------------------------
    # Percentage comparison
    # --------------------------------------------------------

    comparison = percentage_comparison(df)

    comparison_path = os.path.join(
        OUT_DIR,
        "percentage_improvement_final.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"Saved: {comparison_path}"
    )

    # --------------------------------------------------------
    # Winner summary
    # --------------------------------------------------------

    winners = winner_summary(df)

    winner_path = os.path.join(
        OUT_DIR,
        "best_controller_summary_final.csv"
    )

    winners.to_csv(
        winner_path,
        index=False
    )

    print(
        f"Saved: {winner_path}"
    )

    # --------------------------------------------------------
    # Completion / crash-rate summary
    # --------------------------------------------------------

    completion = completion_summary(df)

    completion_path = os.path.join(
        OUT_DIR,
        "completion_summary_final.csv"
    )

    completion.to_csv(
        completion_path,
        index=False
    )

    print(
        f"Saved: {completion_path}"
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    create_plots(df)

    print(
        "\nAnalysis completed successfully."
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        df[
            [
                "scenario",
                "controller",
                "rmse_tracking_error_m",
                "settling_time_s",
                "control_effort_rms_rpm",
                "max_transient_deviation_m",
                "crash_flag",
                "failure_time_s"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
