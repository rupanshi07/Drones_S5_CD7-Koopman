"""
Step 3: Close the loop -- run Koopman-MPC (instead of PID) in
gym-pybullet-drones, tracking the same kind of reference trajectory used
during data collection. Structurally mirrors collect_koopman_data.py's
simulation loop, but with KoopmanMPC.solve() replacing DSLPIDControl.
"""

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from build_edmd_dataset import lift
from koopman_mpc import KoopmanMPC, RPM_HOVER

SIM_FREQ = 240
CTRL_FREQ = 48
AGGR_PHY_STEPS = int(SIM_FREQ / CTRL_FREQ)
EPISODE_LEN_SEC = 8.0
DRONE_MODEL = DroneModel.CF2X


def reference_trajectory(t, amp=0.4, freq=0.15, z0=1.0):
    """Simple fixed circular-ish reference for this first closed-loop test
    (not randomized episode-to-episode like data collection -- we want a
    known, repeatable trajectory to evaluate MPC tracking quality)."""
    x = amp * np.sin(freq * t)
    y = amp * np.sin(2 * freq * t)
    z = z0
    return np.array([x, y, z])


def run_mpc_episode():
    model = np.load("data/koopman_model.npz")
    A, B = model["A"], model["B"]
    mpc = KoopmanMPC(A, B, Hp=10, Hc=5, pos_weight=120.0, du_weight=1.0, u_weight=1e-3)

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
    dt = 1.0 / CTRL_FREQ

    log = {"t": [], "pos": [], "ref_pos": [], "rpm": []}
    prev_rpm = None

    for ctrl_step in range(n_ctrl_steps):
        t = ctrl_step / CTRL_FREQ

        state = obs["0"]["state"]
        pos = state[0:3]
        vel = state[10:13]
        quat = state[3:7]
        ang_vel = state[13:16]
        rotmat = np.array(p.getMatrixFromQuaternion(quat))

        raw_state = np.hstack([pos, vel, rotmat, ang_vel]).reshape(1, -1)
        psi0 = lift(raw_state)[0]

        # build the Hp-step-ahead reference window MPC needs
        ref_window = np.array([
            reference_trajectory(t + k * dt) for k in range(mpc.Hp)
        ])

        rpm_cmd = mpc.solve(psi0, ref_window, prev_u=prev_rpm)
        prev_rpm = rpm_cmd
        action = {"0": rpm_cmd}

        obs, _, done, info = env.step(action)

        log["t"].append(t)
        log["pos"].append(pos.copy())
        log["ref_pos"].append(reference_trajectory(t))
        log["rpm"].append(rpm_cmd.copy())

        if ctrl_step % 20 == 0:
            print(f"t={t:.2f}  pos={np.round(pos,3)}  "
                  f"ref={np.round(reference_trajectory(t),3)}  "
                  f"rpm={np.round(rpm_cmd,0)}")

    env.close()
    for key in log:
        log[key] = np.array(log[key])
    np.savez("data/mpc_closed_loop_test.npz", **log)
    print("Saved data/mpc_closed_loop_test.npz")

    pos_err = np.linalg.norm(log["pos"] - log["ref_pos"], axis=1)
    print(f"\nTracking RMSE: {np.sqrt(np.mean(pos_err**2)):.4f} m")
    print(f"Max tracking error: {pos_err.max():.4f} m")


if __name__ == "__main__":
    run_mpc_episode()
