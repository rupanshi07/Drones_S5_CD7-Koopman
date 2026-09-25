import numpy as np
import sys
from itertools import combinations

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "nominal"
DATA_PATH = f"data/{CONDITION}/flight_log.npz"
SAVE_DIR = f"data/{CONDITION}"

data = np.load(DATA_PATH)
states_full = data["states"]
actions = data["actions"]
states = states_full[:, :16]

X  = states[:-1]
Xp = states[1:]
U  = actions[:-1]

print(f"Loaded {X.shape[0]} transitions for condition: {CONDITION}")

x_mean = X.mean(axis=0)
x_std  = X.std(axis=0) + 1e-8
u_mean = U.mean(axis=0)
u_std  = U.std(axis=0) + 1e-8

Xn  = (X  - x_mean) / x_std
Xpn = (Xp - x_mean) / x_std
Un  = (U  - u_mean) / u_std

# Cross-term pairs: focus on orientation/angular-velocity coupling
# (Q1-Q4: indices 3-6, R/P/Y: indices 7-9, WX/WY/WZ: indices 13-15)
CROSS_IDX = list(range(3, 16))
CROSS_PAIRS = list(combinations(CROSS_IDX, 2))

def lift(x):
    quad = x**2
    cross = np.stack([x[:, i] * x[:, j] for i, j in CROSS_PAIRS], axis=1)
    return np.concatenate([x, quad, cross], axis=1)

Psi_X  = lift(Xn)
Psi_Xp = lift(Xpn)
print(f"Lifted dimension: {Psi_X.shape[1]}")

Z = np.hstack([Psi_X, Un])
reg = 1e-4
ZtZ = Z.T @ Z + reg * np.eye(Z.shape[1])
ZtY = Z.T @ Psi_Xp
AB_T = np.linalg.solve(ZtZ, ZtY)

L = Psi_X.shape[1]
A = AB_T[:L, :].T
B = AB_T[L:, :].T

Psi_pred = Psi_X @ A.T + Un @ B.T
pred_error = np.linalg.norm(Psi_pred - Psi_Xp, axis=1)
rmse = np.sqrt(np.mean(pred_error**2))
print(f"[{CONDITION}] One-step lifted prediction RMSE (normalized space): {rmse:.6f}")

np.savez(f"{SAVE_DIR}/koopman_model.npz",
         A=A, B=B,
         x_mean=x_mean, x_std=x_std,
         u_mean=u_mean, u_std=u_std,
         cross_pairs=np.array(CROSS_PAIRS))
print(f"Saved Koopman model (A: {A.shape}, B: {B.shape}) to {SAVE_DIR}/koopman_model.npz")
