import numpy as np
import os
import sys
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "nominal"
DURATION_SEC = 60
CTRL_FREQ = 48
NUM_STEPS = DURATION_SEC * CTRL_FREQ
SAVE_DIR = f"data/{CONDITION}"
os.makedirs(SAVE_DIR, exist_ok=True)

env = CtrlAviary(
    drone_model=DroneModel.CF2X, num_drones=1,
    physics=Physics.PYB, pyb_freq=240, ctrl_freq=CTRL_FREQ, gui=False
)
obs, info = env.reset()
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

EXTRA_PAYLOAD_MASS = 0.0
WIND_FORCE = np.array([0.0, 0.0, 0.0])
if CONDITION == "payload":
    EXTRA_PAYLOAD_MASS = 0.01
elif CONDITION == "windy":
    WIND_FORCE = np.array([0.05, 0.0, 0.0])

# Payload flights use a gentler height range - isolated diagnostic testing
# found the built-in PID controller loses control and flips when carrying
# 37% extra mass and asked to descend rapidly from ~1m, a maneuver that
# works fine at nominal weight. This reflects a realistic operating
# envelope for an overloaded drone rather than an unrealistic demand.
if CONDITION == "payload":
    TARGET_Z_RANGE = (0.1, 0.6)
else:
    TARGET_Z_RANGE = (0.1, 1.2)

DRONE_ID = env.DRONE_IDS[0]
PAYLOAD_ONSET_STEP = NUM_STEPS // 2
PAYLOAD_APPLIED = False
base_mass = p.getDynamicsInfo(DRONE_ID, -1, physicsClientId=env.CLIENT)[0]

log_states, log_actions = [], []
reset_count = 0
target_hold_steps = 96
current_target = np.array([0.0, 0.0, 0.3])

for step in range(NUM_STEPS):
    if step % target_hold_steps == 0:
        current_target = np.array([
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(*TARGET_Z_RANGE)
        ])

    if CONDITION == "payload" and step == PAYLOAD_ONSET_STEP and not PAYLOAD_APPLIED:
        p.changeDynamics(DRONE_ID, -1, mass=base_mass + EXTRA_PAYLOAD_MASS, physicsClientId=env.CLIENT)
        PAYLOAD_APPLIED = True

    state = obs[0]
    rpm, _, _ = ctrl.computeControlFromState(
        control_timestep=1 / CTRL_FREQ, state=state,
        target_pos=current_target, target_rpy=np.array([0, 0, 0])
    )
    action = rpm.reshape(1, 4)

    obs, reward, terminated, truncated, info = env.step(action)

    if CONDITION == "windy":
        p.applyExternalForce(DRONE_ID, -1, forceObj=WIND_FORCE.tolist(),
                              posObj=[0, 0, 0], flags=p.LINK_FRAME,
                              physicsClientId=env.CLIENT)

    log_states.append(obs[0].copy())
    log_actions.append(action[0].copy())

    if terminated or truncated:
        obs, info = env.reset()
        p.changeDynamics(DRONE_ID, -1, mass=base_mass, physicsClientId=env.CLIENT)
        PAYLOAD_APPLIED = False
        reset_count += 1

env.close()
log_states, log_actions = np.array(log_states), np.array(log_actions)
np.savez(f"{SAVE_DIR}/flight_log.npz", states=log_states, actions=log_actions)

z, roll = log_states[:, 2], log_states[:, 7]
print(f"[{CONDITION}] Saved {log_states.shape[0]} steps, {reset_count} resets")
print(f"Z range: min={z.min():.3f}, max={z.max():.3f}, mean={z.mean():.3f}")
print(f"Roll range: min={roll.min():.3f}, max={roll.max():.3f}, std={roll.std():.3f}")
