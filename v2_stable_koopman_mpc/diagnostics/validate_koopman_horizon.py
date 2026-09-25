"""
Step 4b: Evaluate the Koopman model at the horizon length that actually
matters for MPC (e.g. 10-20 steps = ~0.2-0.4s), not full 8-second open-loop
rollout. MPC re-solves using a fresh true state every control step, so
what matters is short-horizon prediction accuracy, repeated many times
across an episode -- not how far error compounds over 384 uncorrected steps.
"""

import numpy as np

from build_edmd_dataset import load_episode_raw_state, lift
from validate_koopman import classify_episode

HORIZONS_TO_TEST = [5, 10, 20, 40]  # steps; at 48Hz these are ~0.1s, 0.2s, 0.4s, 0.8s


def rolling_horizon_error(A, B, raw_state, u, horizon):
    """For every valid starting index, predict `horizon` steps ahead using
    the real control sequence, starting from the TRUE state at that index
    (mimicking MPC's fresh-state-every-step behavior), and measure position
    error only at the end of that horizon."""
    lifted = lift(raw_state)
    T = raw_state.shape[0]
    errors = []
    for k in range(0, T - horizon):
        psi = lifted[k]
        for j in range(horizon):
            psi = A @ psi + B @ u[k + j]
        pos_pred = psi[0:3]
        pos_true = raw_state[k + horizon, 0:3]
        errors.append(np.linalg.norm(pos_pred - pos_true))
    return np.array(errors)


if __name__ == "__main__":
    model = np.load("data/koopman_model.npz")
    A, B = model["A"], model["B"]

    edmd_data = np.load("data/edmd_train.npz", allow_pickle=True)
    holdout_files = list(edmd_data["holdout_files"])

    results = {h: {} for h in HORIZONS_TO_TEST}

    for f in holdout_files:
        raw_state, u, meta = load_episode_raw_state(f)
        category = classify_episode(meta)
        for h in HORIZONS_TO_TEST:
            errs = rolling_horizon_error(A, B, raw_state, u, h)
            results[h].setdefault(category, []).append(errs)

    print(f"{'horizon (steps/sec)':<22} {'category':<16} {'mean pos err (m)':>18} {'95th pct err (m)':>18}")
    for h in HORIZONS_TO_TEST:
        for cat, err_lists in results[h].items():
            all_errs = np.concatenate(err_lists)
            print(f"{h} steps (~{h/48:.2f}s){'':<6} {cat:<16} "
                  f"{np.mean(all_errs):>18.4f} {np.percentile(all_errs, 95):>18.4f}")
        print()
