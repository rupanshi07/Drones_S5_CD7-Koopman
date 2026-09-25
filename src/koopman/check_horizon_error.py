import numpy as np
from itertools import combinations

CONDITION = "nominal"
data = np.load(f"data/{CONDITION}/flight_log.npz")
model = np.load(f"data/{CONDITION}/koopman_model.npz")

states_full = data["states"]
actions = data["actions"]
states = states_full[:, :16]

x_mean, x_std = model["x_mean"], model["x_std"]
u_mean, u_std = model["u_mean"], model["u_std"]
A, B = model["A"], model["B"]
cross_pairs = [tuple(p) for p in model["cross_pairs"]]

def lift(x):
    quad = x**2
    cross = np.array([x[i]*x[j] for i,j in cross_pairs])
    return np.concatenate([x, quad, cross])

HORIZON = 10
START = 200

x0 = (states[START] - x_mean) / x_std
psi = lift(x0)

print(f"{'step':5s} {'actual Z':>10s} {'predicted Z':>12s}")
for t in range(HORIZON):
    actual_x = (states[START+t] - x_mean) / x_std
    pred_z = psi[2] * x_std[2] + x_mean[2]
    actual_z = states[START+t, 2]
    print(f"{t:5d} {actual_z:10.4f} {pred_z:12.4f}")

    u_norm = (actions[START+t] - u_mean) / u_std
    psi = A @ psi + B @ u_norm
