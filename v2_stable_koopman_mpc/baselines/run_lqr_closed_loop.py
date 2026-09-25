"""
Closed-loop LQR baseline test -- structured identically to
run_mpc_closed_loop.py (same reference trajectory, same simulation setup,
same logging) so the comparison is apples-to-apples.
"""

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from lqr_baseline import LQRController

SIM_FREQ = 240
CTRL_FREQ = 48
AGGR_PHY_STEPS = int(SIM_FREQ / CTRL_FREQ)
EPISODE_LEN_SEC = 8.0
DRONE_MODEL = DroneModel.CF2X


def reference_trajectory(t, amp=0.4, freq=0.15, z0=1.0):
    """Identical to run_mpc_closed_loop.py's reference, for fair comparison."""
    x = amp * np.sin(freq * t)
    y = amp * np.sin(2 * freq * t)
    z = z0
    return np.array([x, y, z])


def run_lqr_episode():
    lqr = LQRController()

    start_pos = reference_trajectory(0.0)

    env = CtrlAviary(
        drone_model=DRONE_MODEL,
        num_drones=1,
        initial_xyzs=np.array([start_pos]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.PYB,
        neighbourhood_radius=10,
        freq=SIM_FREQ,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )
    obs = env.reset()

    n_ctrl_steps = int(EPISODE_LEN_SEC * CTRL_FREQ)

    log = {"t": [], "pos": [], "ref_pos": [], "rpm": []}

    for ctrl_step in range(n_ctrl_steps):
        t = ctrl_step / CTRL_FREQ

        state = obs["0"]["state"]
        pos = state[0:3]
        vel = state[10:13]
        quat = state[3:7]
        ang_vel = state[13:16]
        rotmat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        roll = np.arctan2(rotmat[2, 1], rotmat[2, 2])
        pitch = -np.arcsin(np.clip(rotmat[2, 0], -1, 1))
        yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])

        state12 = np.hstack([pos, vel, [roll, pitch, yaw], ang_vel])
        target_pos = reference_trajectory(t)

        rpm_cmd = lqr.compute(state12, target_pos)
        action = {"0": rpm_cmd}

        obs, _, done, info = env.step(action)

        log["t"].append(t)
        log["pos"].append(pos.copy())
        log["ref_pos"].append(target_pos.copy())
        log["rpm"].append(rpm_cmd.copy())

        if ctrl_step % 20 == 0:
            print(f"t={t:.2f}  pos={np.round(pos,3)}  "
                  f"ref={np.round(target_pos,3)}  rpm={np.round(rpm_cmd,0)}")

    env.close()
    for key in log:
        log[key] = np.array(log[key])
    np.savez("data/lqr_closed_loop_test.npz", **log)
    print("Saved data/lqr_closed_loop_test.npz")

    pos_err = np.linalg.norm(log["pos"] - log["ref_pos"], axis=1)
    print(f"\nTracking RMSE: {np.sqrt(np.mean(pos_err**2)):.4f} m")
    print(f"Max tracking error: {pos_err.max():.4f} m")


if __name__ == "__main__":
    run_lqr_episode()
