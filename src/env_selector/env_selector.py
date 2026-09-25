import numpy as np
import sys

CONDITIONS = ["nominal", "windy", "payload"]
precision_values = {}

# Use RPY (indices 7-9) + angular velocity (indices 13-15) as before, PLUS
# a normalized thrust feature (mean commanded RPM, expressed as a fractional
# deviation from hover RPM). Isolated testing found payload and nominal are
# nearly indistinguishable using RPY/angular-velocity alone (payload's
# distance to its own mean: 0.0216, distance to nominal's mean: 0.0223 --
# essentially the same). A heavier drone needs more thrust to hold the same
# attitude, so thrust is a much stronger discriminator for payload
# specifically, and is added here with proper normalization so it doesn't
# dominate the other features by scale (~15000 RPM vs ~0.01-0.04 for
# RPY/angular velocity).
SIG_IDX = list(range(7, 10)) + list(range(13, 16))

nominal_data = np.load("data/nominal/flight_log.npz")
HOVER_RPM = nominal_data["actions"].mean()

for cond in CONDITIONS:
    data = np.load(f"data/{cond}/flight_log.npz")
    states = data["states"][:, :16]
    actions = data["actions"]
    rpy_angvel_sig = states[:, SIG_IDX].mean(axis=0)
    thrust_sig = (actions.mean() - HOVER_RPM) / HOVER_RPM
    signature = np.concatenate([rpy_angvel_sig, [thrust_sig]])
    precision_values[cond] = signature
    print(f"{cond}: signature = {signature}")

np.savez("data/environment_selector_precision.npz",
         nominal=precision_values["nominal"],
         windy=precision_values["windy"],
         payload=precision_values["payload"],
         hover_rpm=HOVER_RPM)
print("Saved precision values to data/environment_selector_precision.npz")

def select_environment(window_states_16, window_actions, precision_dict, hover_rpm):
    rpy_angvel_sig = window_states_16[:, SIG_IDX].mean(axis=0)
    thrust_sig = (window_actions.mean() - hover_rpm) / hover_rpm
    sig = np.concatenate([rpy_angvel_sig, [thrust_sig]])
    dists = {cond: np.linalg.norm(sig - precision_dict[cond]) for cond in precision_dict}
    return min(dists, key=dists.get), dists

if __name__ == "__main__":
    loaded = np.load("data/environment_selector_precision.npz")
    precision_dict = {c: loaded[c] for c in CONDITIONS}
    hover_rpm = float(loaded["hover_rpm"])
    WINDOW = 240

    for cond in CONDITIONS:
        data = np.load(f"data/{cond}/flight_log.npz")
        states = data["states"][:, :16]
        actions = data["actions"]
        start = int(states.shape[0] * 0.75)
        state_window = states[start:start+WINDOW]
        action_window = actions[start:start+WINDOW]
        predicted, dists = select_environment(state_window, action_window, precision_dict, hover_rpm)
        print(f"True: {cond:8s} | Predicted: {predicted:8s} | Distances: {dists}")
