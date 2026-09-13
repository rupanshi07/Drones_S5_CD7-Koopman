"""
Step 4-5: Validate the fitted Koopman model by simulating it OPEN-LOOP
(no ground-truth correction at each step) on held-out episodes the model
never saw during training, using the actual logged control inputs.

This is the real test: one-step error can look great even for a bad model,
since consecutive states barely change at 48Hz. Multi-step open-loop
prediction error reveals whether the model actually captured the dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt

from build_edmd_dataset import load_episode_raw_state, lift


def simulate_koopman_open_loop(A, B, psi0, U_sequence):
    """Roll the lifted linear model forward using only the initial lifted
    state and the real control sequence -- no correction from ground truth."""
    T = U_sequence.shape[0]
    psi_pred = np.zeros((T + 1, len(psi0)))
    psi_pred[0] = psi0
    for k in range(T):
        psi_pred[k + 1] = A @ psi_pred[k] + B @ U_sequence[k]
    return psi_pred


def classify_episode(meta):
    payload = meta["payload_mass"].max() > 0
    wind = np.max(np.abs(meta["wind_force"])) > 0
    if payload and wind:
        return "payload+wind"
    elif payload:
        return "payload only"
    elif wind:
        return "wind only"
    else:
        return "clean"


if __name__ == "__main__":
    model = np.load("data/koopman_model.npz")
    A, B = model["A"], model["B"]

    edmd_data = np.load("data/edmd_train.npz", allow_pickle=True)
    holdout_files = list(edmd_data["holdout_files"])

    results_by_category = {}

    # plot 4 representative examples, one per category if available
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    plotted_categories = set()

    for f in holdout_files:
        raw_state, u, meta = load_episode_raw_state(f)
        lifted = lift(raw_state)
        psi0 = lifted[0]
        psi_pred = simulate_koopman_open_loop(A, B, psi0, u[:-1])

        pos_true = raw_state[:, 0:3]
        pos_pred = psi_pred[:, 0:3]  # position is the first 3 raw-state dims we lifted

        rmse = np.sqrt(np.mean((pos_pred - pos_true) ** 2))
        category = classify_episode(meta)
        results_by_category.setdefault(category, []).append(rmse)

        if category not in plotted_categories and len(plotted_categories) < 4:
            ax = axes[len(plotted_categories)]
            ax.plot(pos_true[:, 0], pos_true[:, 1], label="actual (ground truth)")
            ax.plot(pos_pred[:, 0], pos_pred[:, 1], "--", label="Koopman open-loop prediction")
            ax.set_title(f"{f.split('/')[-1]}: {category}")
            ax.legend(fontsize=8)
            plotted_categories.add(category)

    plt.tight_layout()
    plt.savefig("koopman_validation.png")
    print("Saved koopman_validation.png")
    print()

    print(f"{'category':<16} {'n_episodes':>10} {'mean pos RMSE (m)':>20} {'max pos RMSE (m)':>18}")
    for cat, rmses in results_by_category.items():
        print(f"{cat:<16} {len(rmses):>10} {np.mean(rmses):>20.4f} {np.max(rmses):>18.4f}")
