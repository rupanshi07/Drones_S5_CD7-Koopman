import numpy as np
import sys
from itertools import combinations

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "nominal"
data = np.load(f"data/{CONDITION}/flight_log.npz")
model = np.load(f"data/{CONDITION}/koopman_model.npz")

states_full = data["states"]
actions = data["actions"]
states = states_full[:, :16]
X, Xp, U = states[:-1], states[1:], actions[:-1]

x_mean, x_std = model["x_mean"], model["x_std"]
u_mean, u_std = model["u_mean"], model["u_std"]
A, B = model["A"], model["B"]

Xn  = (X  - x_mean) / x_std
Xpn = (Xp - x_mean) / x_std
Un  = (U  - u_mean) / u_std

CROSS_IDX = list(range(3, 16))
CROSS_PAIRS = list(combinations(CROSS_IDX, 2))

def lift(x):
    quad = x**2
    cross = np.stack([x[:, i] * x[:, j] for i, j in CROSS_PAIRS], axis=1)
    return np.concatenate([x, quad, cross], axis=1)

Psi_X, Psi_Xp = lift(Xn), lift(Xpn)
Psi_pred = Psi_X @ A.T + Un @ B.T

labels = ["X","Y","Z","Q1","Q2","Q3","Q4","R","P","Y","VX","VY","VZ","WX","WY","WZ"]
err_per_dim = np.sqrt(np.mean((Psi_pred[:, :16] - Psi_Xp[:, :16])**2, axis=0))

print(f"--- {CONDITION}: per-dimension RMSE on ORIGINAL 16 states only ---")
for name, e in zip(labels, err_per_dim):
    print(f"{name:4s}: {e:.4f}")

overall = np.sqrt(np.mean(err_per_dim**2))
print(f"Aggregate (16-dim only): {overall:.4f}")
