import numpy as np
import cvxpy as cp
import sys
from itertools import combinations

model = np.load("data/nominal/koopman_model.npz")
A = model["A"]
B = model["B"]
x_mean, x_std = model["x_mean"], model["x_std"]
u_mean, u_std = model["u_mean"], model["u_std"]
cross_pairs = [tuple(p) for p in model["cross_pairs"]]

N_STATE = 16
N_ACTION = 4
HORIZON = 6

def lift(x):
    quad = x**2
    cross = np.array([x[i] * x[j] for i, j in cross_pairs])
    return np.concatenate([x, quad, cross])

current_state_raw = np.zeros(N_STATE)
current_state_raw[2] = 0.1125
current_state_raw[6] = 1.0

target_state_raw = np.zeros(N_STATE)
target_state_raw[2] = 0.3
target_state_raw[6] = 1.0

x0_norm = (current_state_raw - x_mean) / x_std
target_norm = (target_state_raw - x_mean) / x_std
psi0 = lift(x0_norm)
target_lifted = lift(target_norm)

N_LIFTED = A.shape[0]
Psi = cp.Variable((HORIZON + 1, N_LIFTED))
U = cp.Variable((HORIZON, N_ACTION))

STATE_WEIGHT = np.ones(N_STATE)
STATE_WEIGHT[3:7] = 40.0
STATE_WEIGHT[7] = 40.0
STATE_WEIGHT[8] = 40.0
STATE_WEIGHT[2] = 15.0

cost = 0
constraints = [Psi[0] == psi0]
Q_weight, R_weight, DU_weight, U_BOUND, MAX_RATE = 20.0, 0.1, 1.0, 4.0, 1.0
prev_u = np.zeros(N_ACTION)

for t in range(HORIZON):
    constraints += [Psi[t+1] == A @ Psi[t] + B @ U[t]]
    state_error = Psi[t+1, :N_STATE] - target_lifted[:N_STATE]
    weighted_error = cp.multiply(state_error, STATE_WEIGHT)
    cost += Q_weight * cp.sum_squares(weighted_error)
    cost += R_weight * cp.sum_squares(U[t])
    cost += DU_weight * cp.sum_squares(U[t] - prev_u)
    constraints += [U[t] <= U_BOUND, U[t] >= -U_BOUND]
    constraints += [U[t] - prev_u <= MAX_RATE, U[t] - prev_u >= -MAX_RATE]
    prev_u = U[t]

problem = cp.Problem(cp.Minimize(cost), constraints)

print("Trying CLARABEL with verbose=True to see the real error:")
try:
    problem.solve(solver=cp.CLARABEL, verbose=True)
    print("SOLVED. Status:", problem.status)
except Exception as e:
    print("EXCEPTION TYPE:", type(e))
    print("EXCEPTION MESSAGE:", str(e))
