import numpy as np
import sys
import time
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
sys.path.insert(0, "src/mpc")
from mpc_controller import solve_mpc

RUN_CONDITION = sys.argv[1] if len(sys.argv) > 1 else "nominal"
DURATION_SEC = 10
CTRL_FREQ = 48
NUM_STEPS = DURATION_SEC * CTRL_FREQ

env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=CTRL_FREQ,
    gui=False
)

obs, info = env.reset()

log_states = []
log_actions = []
reset_count = 0

for step in range(NUM_STEPS):
    current_state = obs[0][:16]
    action = solve_mpc(current_state)
    action = action.reshape(1, 4)

    obs, reward, terminated, truncated, info = env.step(action)

    log_states.append(obs[0].copy())
    log_actions.append(action[0].copy())

    if step % 48 == 0:
        z = obs[0][2]
        print(f"t={step/CTRL_FREQ:.1f}s | Z={z:.3f}m | target=0.5m | terminated={terminated} | truncated={truncated}")

    if terminated or truncated:
        reset_count += 1
        print(f"  >>> RESET #{reset_count} at step {step} (t={step/CTRL_FREQ:.2f}s)")
        obs, info = env.reset()

env.close()

log_states = np.array(log_states)
log_actions = np.array(log_actions)
np.savez(f"results/mpc_test_{RUN_CONDITION}.npz", states=log_states, actions=log_actions)
print(f"Saved MPC test run. Total resets: {reset_count}")
