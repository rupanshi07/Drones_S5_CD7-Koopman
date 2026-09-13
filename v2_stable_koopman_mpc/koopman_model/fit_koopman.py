"""
Step 3: Fit the Koopman A, B matrices via EDMD least-squares.

Solves: min over A,B of sum_k || Psi(x[k+1]) - A*Psi(x[k]) - B*u[k] ||^2
This is equation (3) in the reference paper -- a single big linear
least-squares problem, solved directly rather than iteratively.

Run after build_edmd_dataset.py has produced data/edmd_train.npz.
"""

import numpy as np

RIDGE_LAMBDA = 1e-4  # reverted: 1e-2 likely over-suppressed the small-magnitude
                      # but physically important tilt-to-acceleration coupling,
                      # since rotmat entries have low variance near hover


def fit_koopman(Psi_k, Psi_kp1, U_k, ridge_lambda=RIDGE_LAMBDA):
    """
    Stack [Psi_k, U_k] as the regressor, solve for [A, B] jointly via
    ridge-regularized least squares (regularization keeps the solution
    well-behaved if the lifted+input dimension is large relative to
    independent information in the data).
    """
    n_lifted = Psi_k.shape[1]
    n_input = U_k.shape[1]

    Z = np.hstack([Psi_k, U_k])          # (N, n_lifted + n_input)
    Y = Psi_kp1                          # (N, n_lifted)

    # Ridge solution: theta = (Z^T Z + lambda*I)^-1 Z^T Y
    ZtZ = Z.T @ Z
    reg = ridge_lambda * np.eye(ZtZ.shape[0])
    theta = np.linalg.solve(ZtZ + reg, Z.T @ Y)  # (n_lifted+n_input, n_lifted)

    A = theta[:n_lifted, :].T   # (n_lifted, n_lifted)
    B = theta[n_lifted:, :].T  # (n_lifted, n_input)
    return A, B


def one_step_prediction_error(A, B, Psi_k, Psi_kp1, U_k):
    pred = Psi_k @ A.T + U_k @ B.T
    err = pred - Psi_kp1
    rmse_per_dim = np.sqrt(np.mean(err ** 2, axis=0))
    return rmse_per_dim


if __name__ == "__main__":
    data = np.load("data/edmd_train.npz", allow_pickle=True)
    Psi_k, Psi_kp1, U_k = data["Psi_k"], data["Psi_kp1"], data["U_k"]

    # RPMs are ~O(10^4) while lifted state entries are ~O(1); this scale
    # mismatch can make the least-squares problem poorly conditioned, so
    # normalize U before fitting (and remember the scale to undo later)
    u_scale = U_k.std(axis=0)
    U_k_norm = U_k / u_scale

    A, B_norm = fit_koopman(Psi_k, Psi_kp1, U_k_norm)
    B = B_norm / u_scale[None, :]  # undo normalization so B applies to raw RPM units

    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"Max |eigenvalue| of A: {np.max(np.abs(np.linalg.eigvals(A))):.4f}  "
          f"(should be close to but not wildly above 1.0 for a stable-ish system)")

    rmse = one_step_prediction_error(A, B, Psi_k, Psi_kp1, U_k)
    print("One-step prediction RMSE per lifted dimension (first 18 = raw state dims):")
    print(np.round(rmse[:18], 5))

    np.savez("data/koopman_model.npz", A=A, B=B, u_scale=u_scale)
    print("Saved data/koopman_model.npz")
