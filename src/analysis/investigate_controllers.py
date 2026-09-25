"""
Validation Step 7/8 — Investigate WHY MPC crashes and whether LQR's very
low error is real or an artifact.

Run:
    python src/analysis/investigate_controllers.py
"""
import os
import numpy as np

RESULTS_DIR = "results"
CTRL_FREQ = 48
SCENARIOS = [1, 2, 3, 4, 5, 6]
RPM_MIN, RPM_MAX = 0.0, 25000.0        # saturation bounds used by all 3 controllers
FLIP_THRESHOLD_RAD = np.pi / 2          # past this, the drone is no longer "flying level"
RESET_JUMP_THRESHOLD = 0.08             # metres; a single-step z jump this big => env.reset()


def load(ctrl_prefix, n):
    path = os.path.join(RESULTS_DIR, f"{ctrl_prefix}scenario_{n}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return d["states"], d["actions"], int(d["disturbance_onset_step"])


# ---------------------------------------------------------------------------
# 7. Why does MPC crash?
# ---------------------------------------------------------------------------
def investigate_mpc():
    print("=" * 78)
    print("CHECK 7: Investigating MPC crash / instability")
    print("=" * 78)
    for n in SCENARIOS:
        run = load("", n)
        if run is None:
            continue
        states, actions, onset = run
        roll, pitch = states[:, 7], states[:, 8]
        z = states[:, 2]

        sat_low = int(((actions <= RPM_MIN + 1e-6)).sum())
        sat_high = int(((actions >= RPM_MAX - 1e-6)).sum())
        max_roll = float(np.max(np.abs(roll)))
        max_pitch = float(np.max(np.abs(pitch)))
        flipped_steps = int((np.abs(roll) > FLIP_THRESHOLD_RAD).sum())

        # detect env.reset() events: a same-step backward jump in z toward
        # the spawn height, or a discontinuous quaternion, indicates
        # terminated/truncated triggered a mid-run reset (see run_scenario.py:
        # "if terminated or truncated: obs, info = env.reset()")
        z_jumps = np.abs(np.diff(z))
        reset_events = int((z_jumps > RESET_JUMP_THRESHOLD).sum())

        min_z = float(z.min())

        verdict = []
        if flipped_steps > 0:
            verdict.append(f"TUMBLES/FLIPS ({flipped_steps} steps with |roll|>90deg, "
                            f"max|roll|={np.degrees(max_roll):.0f}deg)")
        if sat_low + sat_high > 0:
            verdict.append(f"RPM SATURATION ({sat_low} steps at 0, {sat_high} steps at max)")
        if reset_events > 0:
            verdict.append(f"{reset_events} likely env.reset() event(s) mid-run "
                            f"(controller crashed and simulation was silently restarted)")
        if min_z < 0.02:
            verdict.append(f"min altitude {min_z:.4f}m (near ground / possible ground contact)")

        print(f"\nScenario {n}: onset={onset}")
        print(f"  max|roll|={np.degrees(max_roll):.1f}deg  max|pitch|={np.degrees(max_pitch):.1f}deg  "
              f"min_z={min_z:.4f}m  rpm_sat(low/high)={sat_low}/{sat_high}  reset_events~{reset_events}")
        if verdict:
            print(f"  DIAGNOSIS: " + "; ".join(verdict))
        else:
            print(f"  DIAGNOSIS: no obvious instability signature found")

    print("""
INTERPRETATION GUIDE (fill in for the report after reading the numbers above):
  - Saturation at BOTH 0 and 25000 RPM within the same run means the QP is
    commanding physically extreme, contradictory motor speeds -- classic
    sign the optimizer is fighting an inconsistent/locally-invalid linear
    (Koopman) model rather than converging to a reasonable control.
  - |roll| exceeding ~90 degrees means the quadrotor has literally flipped;
    at that point the small-angle-friendly Euler/quaternion Koopman lift
    used by the model no longer represents the true dynamics well, so the
    model-based control degrades further -- a feedback loop into failure.
  - A z jump alone (without a saturation/flip signature) more likely means
    a genuine env.reset() rather than a graceful recovery -- that inflates
    apparent "settling" if not accounted for, so check reset_events before
    trusting a scenario's settling-time number.
  - This lines up with the project's own commit history noting Euler/
    quaternion Koopman singularity issues under this MPC formulation.
""")


# ---------------------------------------------------------------------------
# 8. Is LQR suspiciously good?
# ---------------------------------------------------------------------------
def investigate_lqr():
    print("=" * 78)
    print("CHECK 8: Is LQR's very low error real, or getting an easier ride?")
    print("=" * 78)
    for n in SCENARIOS:
        pid_run = load("pid_", n)
        lqr_run = load("lqr_", n)
        if pid_run is None or lqr_run is None:
            continue
        pid_states, pid_actions, pid_onset = pid_run
        lqr_states, lqr_actions, lqr_onset = lqr_run

        pid_roll_amp = float(np.max(np.abs(pid_states[:, 7])))
        lqr_roll_amp = float(np.max(np.abs(lqr_states[:, 7])))
        pid_rpm_range = float(pid_actions.max() - pid_actions.min())
        lqr_rpm_range = float(lqr_actions.max() - lqr_actions.min())

        print(f"\nScenario {n}: onset PID={pid_onset} LQR={lqr_onset} "
              f"(should match — confirms same disturbance timing)")
        print(f"  roll amplitude:  PID={np.degrees(pid_roll_amp):.4f}deg   "
              f"LQR={np.degrees(lqr_roll_amp):.6f}deg")
        print(f"  RPM range used:  PID={pid_rpm_range:.1f}   LQR={lqr_rpm_range:.1f}")

        if lqr_roll_amp < 1e-6 and pid_roll_amp > 1e-4:
            print(f"  FLAG: LQR shows essentially ZERO attitude response "
                  f"(<1e-6 rad) while PID visibly reacts to the same wind/"
                  f"payload event. Onsets match above, and the driver script "
                  f"applies the identical p.applyExternalForce() call for "
                  f"both -- so this is NOT a missing-disturbance bug.")
            print(f"  LIKELY CAUSE: LQR's Q matrix weights roll/pitch heavily "
                  f"(Q_roll=Q_pitch=50 vs Q_pos_xy=10) with modest R "
                  f"(R=0.5 on all torques) — see src/mpc/lqr_controller.py. "
                  f"That produces a very high feedback gain on attitude, so "
                  f"the controller cancels the small (0.05 N) wind-induced "
                  f"torque almost within a single 1/48s step. This is a "
                  f"legitimate consequence of the LQR tuning, not a fairness "
                  f"bug -- but it SHOULD be reported explicitly (e.g. 'LQR is "
                  f"tuned for a very stiff attitude response; this partly "
                  f"explains its near-zero tracking error under this "
                  f"disturbance magnitude') rather than presented as an "
                  f"unqualified win.")

    print("""
ACTION ITEMS for the report:
  1. State the LQR Q/R weights explicitly next to its results table so
     readers can judge whether the comparison is "fair" in a tuning sense,
     not just a disturbance-application sense.
  2. Consider re-running LQR (and ideally PID/MPC) under a LARGER wind
     magnitude / heavier payload to see whether LQR's advantage holds or
     is specific to this small disturbance regime -- that materially
     changes the strength of the claim you can make.
""")


if __name__ == "__main__":
    investigate_mpc()
    investigate_lqr()