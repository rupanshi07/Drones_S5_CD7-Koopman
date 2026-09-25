import numpy as np
import sys
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
sys.path.insert(0, "src/mpc")
from mpc_controller import solve_mpc

CTRL_FREQ = 48
env = CtrlAviary(
    drone_model=DroneModel.CF2X, num_drones=1,
    physics=Physics.PYB, pyb_freq=240, ctrl_freq=CTRL_FREQ, gui=False
)
obs, info = env.reset()

for step in range(80):
    current_state = obs[0][:16]
    action = solve_mpc(current_state, condition="nominal")
    z = current_state[2]
    roll = current_state[7]
    if step % 5 == 0:
        print(f"step={step} Z={z:.4f} roll={roll:.4f}")
    action = action.reshape(1, 4)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
