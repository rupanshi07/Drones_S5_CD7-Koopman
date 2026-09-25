import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p

CTRL_FREQ = 48
env = CtrlAviary(
    drone_model=DroneModel.CF2X, num_drones=1,
    physics=Physics.PYB, pyb_freq=240, ctrl_freq=CTRL_FREQ, gui=False
)
obs, info = env.reset()
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
DRONE_ID = env.DRONE_IDS[0]
base_mass = p.getDynamicsInfo(DRONE_ID, -1, physicsClientId=env.CLIENT)[0]

ONSET_STEP = 100  # apply early so we don't need to wait long
EXTRA_MASS = 0.01
applied = False
target = np.array([0.0, 0.0, 0.3])

for step in range(150):
    if step == ONSET_STEP and not applied:
        print(f"--- Applying mass change at step {step}, base_mass={base_mass:.5f} ---")
        p.changeDynamics(DRONE_ID, -1, mass=base_mass + EXTRA_MASS, physicsClientId=env.CLIENT)
        applied = True
        actual_mass = p.getDynamicsInfo(DRONE_ID, -1, physicsClientId=env.CLIENT)[0]
        print(f"Confirmed new mass: {actual_mass:.5f}")

    state = obs[0]
    rpm, _, _ = ctrl.computeControlFromState(
        control_timestep=1/CTRL_FREQ, state=state,
        target_pos=target, target_rpy=np.array([0,0,0])
    )
    action = rpm.reshape(1,4)
    obs, reward, terminated, truncated, info = env.step(action)

    if ONSET_STEP - 5 <= step <= ONSET_STEP + 20:
        z, roll, pitch = obs[0][2], obs[0][7], obs[0][8]
        print(f"step={step} Z={z:.4f} roll={roll:.4f} pitch={pitch:.4f} rpm_cmd={rpm}")

    if terminated or truncated:
        print(f"RESET at step {step}")
        obs, info = env.reset()

env.close()
