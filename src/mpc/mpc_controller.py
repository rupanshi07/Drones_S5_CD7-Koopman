import numpy as np
import cvxpy as cp
from itertools import combinations

N_STATE = 16
N_ACTION = 4
HORIZON = 6
HOVER_RPM = 14436

_models = {}

def load_model(condition):
    if condition not in _models:
        m = np.load(f"data/{condition}/koopman_model.npz")
        _models[condition] = {
            "A": m["A"], "B": m["B"],
            "x_mean": m["x_mean"], "x_std": m["x_std"],
            "u_mean": m["u_mean"], "u_std": m["u_std"],
            "cross_pairs": [tuple(p) for p in m["cross_pairs"]]
        }
    return _models[condition]

def lift(x, cross_pairs):
    quad = x**2
    cross = np.array([x[i] * x[j] for i, j in cross_pairs])
    return np.concatenate([x, quad, cross])

FINAL_TARGET_Z = 0.3
RAMP_RATE = 0.001  # max height increase per control step - gentle climb, not a jump

STATE_WEIGHT = np.ones(N_STATE)
STATE_WEIGHT[3:7] = 100.0   # quaternion - pushed higher than before
STATE_WEIGHT[7] = 100.0     # roll - pushed higher than before
STATE_WEIGHT[8] = 100.0     # pitch - pushed higher than before
STATE_WEIGHT[2] = 15.0      # height

_last_u_norm = np.zeros(N_ACTION)
_current_target_z = None  # tracks the gradually-ramping height target across calls

def solve_mpc(current_state_raw, condition="nominal"):
    global _last_u_norm, _current_target_z

    if np.any(np.isnan(current_state_raw)) or np.any(np.isinf(current_state_raw)) or np.any(np.abs(current_state_raw) > 1e4):
        model = load_model(condition)
        u0_raw = _last_u_norm * model["u_std"] + model["u_mean"]
        return np.clip(u0_raw, 0, 25000)

    # Initialize the ramp target at the drone's current height on first call
    if _current_target_z is None:
        _current_target_z = current_state_raw[2]

    # Advance the target gradually toward the final goal, never jumping ahead
    if _current_target_z < FINAL_TARGET_Z:
        _current_target_z = min(_current_target_z + RAMP_RATE, FINAL_TARGET_Z)

    target_state_raw = np.zeros(N_STATE)
    target_state_raw[2] = _current_target_z
    target_state_raw[6] = 1.0

    model = load_model(condition)
    A, B = model["A"], model["B"]
    x_mean, x_std = model["x_mean"], model["x_std"]
    u_mean, u_std = model["u_mean"], model["u_std"]
    cross_pairs = model["cross_pairs"]

    x0_norm = (current_state_raw - x_mean) / x_std
    target_norm = (target_state_raw - x_mean) / x_std
    psi0 = lift(x0_norm, cross_pairs)
    target_lifted = lift(target_norm, cross_pairs)

    N_LIFTED = A.shape[0]
    U = cp.Variable((HORIZON, N_ACTION))

    Q_weight = 20.0
    R_weight = 0.1
    DU_weight = 1.0
    U_BOUND = 4.0
    MAX_RATE = 1.0

    cost = 0
    constraints = []
    prev_u = _last_u_norm

    psi_t = psi0
    for t in range(HORIZON):
        psi_next = A @ psi_t + B @ U[t]
        state_error = psi_next[:N_STATE] - target_lifted[:N_STATE]
        weighted_error = cp.multiply(state_error, STATE_WEIGHT)
        cost += Q_weight * cp.sum_squares(weighted_error)
        cost += R_weight * cp.sum_squares(U[t])
        cost += DU_weight * cp.sum_squares(U[t] - prev_u)
        constraints += [U[t] <= U_BOUND, U[t] >= -U_BOUND]
        constraints += [U[t] - prev_u <= MAX_RATE, U[t] - prev_u >= -MAX_RATE]
        prev_u = U[t]
        psi_t = psi_next

    problem = cp.Problem(cp.Minimize(cost), constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000)
    except cp.error.SolverError:
        pass

    if U.value is None or np.any(np.isnan(U.value)):
        u0_raw = _last_u_norm * u_std + u_mean
        return np.clip(u0_raw, 0, 25000)

    u0_norm = U.value[0]
    _last_u_norm = u0_norm
    u0_raw = u0_norm * u_std + u_mean
    return np.clip(u0_raw, 0, 25000)
