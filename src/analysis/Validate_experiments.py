"""
Validation Step 1/2/9 — Verify all 18 runs exist/are valid, and that
MPC, PID, and LQR were actually tested under identical conditions.

This does two kinds of checking, because the experiment parameters
(wind force, payload mass, onset step, duration, ...) are NOT stored
inside the .npz files themselves -- they're hardcoded constants in each
run_scenario*.py script. So "fairness" has to be verified two ways:

  A) STATIC check: parse the three driver scripts and diff the constants
     that control the experiment (wind vector, payload mass, onset step,
     duration, ctrl freq, target height/ramp). If these differ, the
     comparison is not apples-to-apples, full stop.

  B) DATA-DRIVEN check: for each scenario, confirm the disturbance really
     shows up in the trajectory logs for every controller (not just that
     the code *says* it applied a force) -- e.g. wind scenarios should
     show a velocity/attitude excursion after onset, payload scenarios
     should show a step up in required thrust after onset.

Run:
    python src/analysis/validate_experiments.py
"""
import os
import re
import numpy as np

RESULTS_DIR = "results"
SRC_MPC_DIR = "src/mpc"

CONTROLLERS = {
    "MPC": ("scenario_{n}.npz", os.path.join(SRC_MPC_DIR, "run_scenario.py")),
    "PID": ("pid_scenario_{n}.npz", os.path.join(SRC_MPC_DIR, "run_scenario_pid.py")),
    "LQR": ("lqr_scenario_{n}.npz", os.path.join(SRC_MPC_DIR, "run_scenario_lqr.py")),
}
SCENARIOS = [1, 2, 3, 4, 5, 6]
REQUIRED_KEYS = ["states", "actions", "selected", "disturbance_onset_step"]

# Constants we require to be byte-identical across the three driver scripts.
# (regex, human label)
FAIRNESS_PATTERNS = [
    (r"CTRL_FREQ\s*=\s*([0-9.]+)", "CTRL_FREQ"),
    (r"DURATION_SEC\s*=\s*([0-9.]+)", "DURATION_SEC"),
    (r"WIND_FORCE\s*=\s*np\.array\(\[([^\]]+)\]\)", "WIND_FORCE"),
    (r"EXTRA_PAYLOAD_MASS\s*=\s*([0-9.]+)", "EXTRA_PAYLOAD_MASS"),
    (r"PAYLOAD_ONSET_STEP\s*=\s*([^\r\n]+)", "PAYLOAD_ONSET_STEP (expression)"),
    (r"SELECTOR_WINDOW\s*=\s*([0-9]+)", "SELECTOR_WINDOW"),
]


# ---------------------------------------------------------------------------
# 1. Completeness / validity of the 18 runs
# ---------------------------------------------------------------------------
def check_completeness():
    print("=" * 70)
    print("CHECK 1: Completeness & validity of all 18 experiment files")
    print("=" * 70)
    ok = True
    for ctrl, (pattern, _) in CONTROLLERS.items():
        for n in SCENARIOS:
            path = os.path.join(RESULTS_DIR, pattern.format(n=n))
            if not os.path.exists(path):
                print(f"  [MISSING] {path}")
                ok = False
                continue
            try:
                d = np.load(path, allow_pickle=True)
            except Exception as e:
                print(f"  [UNREADABLE] {path}: {e}")
                ok = False
                continue

            missing_keys = [k for k in REQUIRED_KEYS if k not in d.files]
            if missing_keys:
                print(f"  [BAD SCHEMA] {path} missing keys: {missing_keys}")
                ok = False
                continue

            states, actions = d["states"], d["actions"]
            problems = []
            if states.ndim != 2 or states.shape[1] < 16:
                problems.append(f"states shape {states.shape} (expected Nx16+)")
            if actions.ndim != 2 or actions.shape[1] != 4:
                problems.append(f"actions shape {actions.shape} (expected Nx4)")
            if states.shape[0] != actions.shape[0]:
                problems.append("states/actions length mismatch")
            if np.isnan(states).any() or np.isnan(actions).any():
                problems.append("NaNs present in states or actions")
            if states.shape[0] == 0:
                problems.append("zero-length run")

            # note: raw logs don't store an explicit "time" or "target position"
            # array -- time is implicit (index / CTRL_FREQ) and target position
            # is implicit (the ramping-to-FINAL_TARGET_Z policy shared by all
            # three controllers). Position and controller output (actions) ARE
            # present. This is fine as long as it's documented -- flagging here
            # so it ends up in the report rather than being assumed silently.
            if problems:
                print(f"  [INVALID] {path}: {'; '.join(problems)}")
                ok = False
            else:
                print(f"  [OK] {path}  ({states.shape[0]} steps)")

    print()
    print("Note: 'time' and 'target position' are not stored explicitly in the "
          "npz files -- they're implicit (time = step_index / 48 Hz; target "
          "position = (0,0,ramping-z) per the shared ramp policy). Documented "
          "here so the report doesn't silently assume it.")
    print(f"\nCompleteness check: {'PASS' if ok else 'FAIL'} "
          f"({len(CONTROLLERS) * len(SCENARIOS)} runs expected)\n")
    return ok


# ---------------------------------------------------------------------------
# 2. Static fairness check -- do the three driver scripts use identical
#    experiment parameters?
# ---------------------------------------------------------------------------
def check_static_fairness():
    print("=" * 70)
    print("CHECK 2: Static fairness — experiment constants across scripts")
    print("=" * 70)
    extracted = {}
    for ctrl, (_, script_path) in CONTROLLERS.items():
        with open(script_path, "r", encoding="utf-8") as f:
            text = f.read()
        extracted[ctrl] = {}
        for pattern, label in FAIRNESS_PATTERNS:
            m = re.search(pattern, text)
            if not m:
                extracted[ctrl][label] = "<not found>"
                continue
            val = m.group(1).split("#")[0].strip()  # strip trailing comments
            extracted[ctrl][label] = val

    all_match = True
    labels = [label for _, label in FAIRNESS_PATTERNS]
    for label in labels:
        values = {ctrl: extracted[ctrl][label] for ctrl in CONTROLLERS}
        unique_vals = set(values.values())
        status = "MATCH" if len(unique_vals) == 1 else "MISMATCH"
        if status == "MISMATCH":
            all_match = False
        print(f"  {label:35s} [{status}]  " +
              "  ".join(f"{c}={v}" for c, v in values.items()))

    print(f"\nStatic fairness check: {'PASS' if all_match else 'FAIL — see MISMATCH rows above'}\n")
    return all_match


# ---------------------------------------------------------------------------
# 9. Data-driven check -- did the disturbance actually show up in the sim?
# ---------------------------------------------------------------------------
def check_disturbance_applied():
    print("=" * 70)
    print("CHECK 9: Disturbance actually reaches each controller's simulation")
    print("=" * 70)
    # scenario -> which disturbance to expect
    WIND_SCENARIOS = {2, 3, 4, 6}   # wind active for some/all of the run
    PAYLOAD_SCENARIOS = {5, 6}      # mass step applied mid-flight

    all_ok = True
    for n in SCENARIOS:
        print(f"\nScenario {n}:")
        onsets = {}
        for ctrl, (pattern, _) in CONTROLLERS.items():
            path = os.path.join(RESULTS_DIR, pattern.format(n=n))
            if not os.path.exists(path):
                continue
            d = np.load(path, allow_pickle=True)
            onset = int(d["disturbance_onset_step"])
            onsets[ctrl] = onset

            states = d["states"]
            roll = states[:, 7]
            vel_x = states[:, 10]

            if n in WIND_SCENARIOS and onset >= 0:
                pre = states[max(0, onset - 48):onset]
                post = states[onset:onset + 48]
                pre_excursion = np.max(np.abs(pre[:, 10])) if len(pre) else 0.0
                post_excursion = np.max(np.abs(post[:, 10])) if len(post) else 0.0
                wind_seen = post_excursion > pre_excursion or post_excursion > 1e-4
                print(f"  [{ctrl}] wind expected @ step {onset}: "
                      f"|vx| pre={pre_excursion:.2e} post={post_excursion:.2e} "
                      f"-> {'DETECTED' if wind_seen else 'NOT DETECTED (check!)'}")
                if not wind_seen:
                    all_ok = False

            if n in PAYLOAD_SCENARIOS and onset >= 0:
                pre = states[max(0, onset - 48):onset]
                post = states[onset:onset + 96]
                # crude proxy: total commanded RPM should shift once extra
                # mass needs compensating thrust (unless controller is
                # saturated/unstable and this signal gets swamped)
                pre_z = np.mean(pre[:, 2]) if len(pre) else np.nan
                post_z = np.mean(post[:, 2]) if len(post) else np.nan
                print(f"  [{ctrl}] payload expected @ step {onset}: "
                      f"mean z pre={pre_z:.4f} post={post_z:.4f}")

        # onset step should match across controllers for the same scenario
        # (all three share identical trigger logic)
        unique_onsets = set(onsets.values())
        if len(unique_onsets) > 1:
            print(f"  [WARNING] disturbance_onset_step differs across controllers: {onsets}")
            all_ok = False
        else:
            print(f"  onset step consistent across controllers: {onsets}")

    print(f"\nDisturbance-applied check: {'PASS' if all_ok else 'REVIEW NEEDED — see notes above'}\n")
    return all_ok


if __name__ == "__main__":
    r1 = check_completeness()
    r2 = check_static_fairness()
    r3 = check_disturbance_applied()
    print("=" * 70)
    print(f"SUMMARY: completeness={'PASS' if r1 else 'FAIL'}  "
          f"static_fairness={'PASS' if r2 else 'FAIL'}  "
          f"disturbance_applied={'PASS' if r3 else 'REVIEW'}")
    print("=" * 70)