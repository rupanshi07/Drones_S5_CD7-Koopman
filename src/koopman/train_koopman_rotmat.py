import numpy as np
import sys
sys.path.insert(0, "src/koopman")
from lift_rotmat import lift_rotmat

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "nominal"
data = np.load(f"data/{CONDITION}/flight_log.npz")
states_full = data["states"]
actions = data["actions"]
states = states_full[:, :16]

X, Xp, U = states[:-1], states[1:], actions[:-1]

x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
u_mean, u_std = U.mean(axis=0), U.std(axis=0) + 1e-8
Xn, Xpn, Un = (X - x_mean) / x_std, (Xp - x_mean) / x_std, (U - u_mean) / u_std

# Un-normalize just for the rotation matrix step (rotation matrices must stay
# valid/orthogonal, so we lift from the RAW state, not the normalized one,
# then normalize the lifted output instead)
Psi_X_raw = lift_rotmat(X)
Psi_Xp_raw = lift_rotmat(Xp)

psi_mean = Psi_X_raw.mean(axis=0)
psi_std = Psi_X_raw.std(axis=0) + 1e-8
Psi_X = (Psi_X_raw - psi_mean) / psi_std
Psi_Xp = (Psi_Xp_raw - psi_mean) / psi_std

Z = np.hstack([Psi_X, Un])
reg = 1e-4
ZtZ = Z.T @ Z + reg * np.eye(Z.shape[1])
ZtY = Z.T @ Psi_Xp
AB_T = np.linalg.solve(ZtZ, ZtY)

L = Psi_X.shape[1]
A, B = AB_T[:L, :].T, AB_T[L:, :].T

Psi_pred = Psi_X @ A.T + Un @ B.T
rmse = np.sqrt(np.mean((Psi_pred - Psi_Xp) ** 2))
print(f"[{CONDITION}] Rotation-matrix lifted model | dim={L} | full RMSE (normalized) = {rmse:.4f}")

np.savez(f"data/{CONDITION}/koopman_model_rotmat.npz",
         A=A, B=B, x_mean=x_mean, x_std=x_std, u_mean=u_mean, u_std=u_std,
         psi_mean=psi_mean, psi_std=psi_std)
