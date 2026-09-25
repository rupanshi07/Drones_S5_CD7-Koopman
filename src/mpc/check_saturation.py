import numpy as np
import sys
sys.path.insert(0, "src/mpc")
from mpc_controller import solve_mpc, normalize_action, denormalize_action, u_mean, u_std

test_state = np.zeros(16)
test_state[2] = 0.1
test_state[6] = 1.0

action = solve_mpc(test_state)
action_norm = normalize_action(action)
print("Raw action:", action)
print("Normalized action (should be within [-2, 2]):", action_norm)
print("u_mean:", u_mean)
print("u_std:", u_std)
