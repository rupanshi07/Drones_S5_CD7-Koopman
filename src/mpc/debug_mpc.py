import numpy as np
import sys
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
sys.path.insert(0, "src/mpc")
from mpc_controller import solve_mpc

RUN_CONDITION = sys.argv[1] if len(sys.argv) > 1 else "nominal"
CTRL_FREQ = 48
NUM_STEPS = 20

env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=CTRL_FREQ,
    gui=False
)

obs, info = env.reset()

for step in range(NUM_STEPS):
    current_state = obs[0][:16]
    action = solve_mpc(current_state)
    action = action.reshape(1, 4)

    print(f"step={step} | Z={current_state[2]:.4f} | action={action[0]}")

    obs, reward, terminated, truncated, info = env.step(action)

env.close()
