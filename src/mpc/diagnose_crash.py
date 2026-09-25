import numpy as np
import sys
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
sys.path.insert(0, "src/mpc")
from mpc_controller import solve_mpc

CTRL_FREQ = 48
DURATION_SEC = 15
NUM_STEPS = DURATION_SEC * CTRL_FREQ

env = CtrlAviary(
    drone_model=DroneModel.CF2X, num_drones=1,
    physics=Physics.PYB, pyb_freq=240, ctrl_freq=CTRL_FREQ, gui=False
)
obs, info = env.reset()

log = []

for step in range(NUM_STEPS):
    current_state = obs[0][:16]
    action = solve_mpc(current_state, condition="nominal")
    action = action.reshape(1, 4)

    z = current_state[2]
    roll = current_state[7]
    pitch = current_state[8]
    q1, q2, q3, q4 = current_state[3], current_state[4], current_state[5], current_state[6]

    log.append([step, z, roll, pitch, q1, q2, q3, q4])

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

log = np.array(log)
np.save("results/crash_diagnostic.npy", log)

# Find the step where Z drops most sharply
z_vals = log[:,1]
dz = np.diff(z_vals)
crash_step = np.argmin(dz)  # most negative change = sharpest drop

print(f"Sharpest height drop detected at step {crash_step} (t={crash_step/CTRL_FREQ:.2f}s)")
print()
print(f"{'step':>5s} {'t(s)':>6s} {'Z':>8s} {'roll':>8s} {'pitch':>8s} {'Q4':>8s}")
start = max(0, crash_step - 10)
end = min(len(log), crash_step + 10)
for row in log[start:end]:
    step, z, roll, pitch, q1, q2, q3, q4 = row
    print(f"{int(step):5d} {step/CTRL_FREQ:6.2f} {z:8.4f} {roll:8.4f} {pitch:8.4f} {q4:8.4f}")
