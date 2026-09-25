"""
Diagnostic: bypass MPC entirely. Apply a small, KNOWN, fixed differential
RPM pattern directly, and compare:
  (a) what the fitted Koopman model (A, B) PREDICTS should happen, vs
  (b) what the REAL simulator actually does.

If these disagree in DIRECTION (not just magnitude), that points to a
sign/indexing bug in the identified B matrix or the lift() function, rather
than an MPC-specific tuning issue.
"""

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from build_edmd_dataset import lift

SIM_FREQ = 240
CTRL_FREQ = 48
AGGR_PHY_STEPS = int(SIM_FREQ / CTRL_FREQ)
RPM_HOVER = 14468.0
N_TEST_STEPS = 10  # ~0.2s of applying the fixed differential


def run_step_response_test(diff_pattern, label):
    model = np.load("data/koopman_model.npz")
    A, B = model["A"], model["B"]

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
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

    rpm_cmd = RPM_HOVER + np.array(diff_pattern)

    # --- (a) what the model predicts, starting from the true initial state ---
    state0 = obs["0"]["state"]
    pos0, quat0, vel0, ang_vel0 = state0[0:3], state0[3:7], state0[10:13], state0[13:16]
    rotmat0 = np.array(p.getMatrixFromQuaternion(quat0))
    raw_state0 = np.hstack([pos0, vel0, rotmat0, ang_vel0]).reshape(1, -1)
    psi = lift(raw_state0)[0]
    for _ in range(N_TEST_STEPS):
        psi = A @ psi + B @ rpm_cmd
    predicted_pos = psi[0:3]
    predicted_vel = psi[3:6]

    # --- (b) what actually happens in the real simulator ---
    for _ in range(N_TEST_STEPS):
        action = {"0": rpm_cmd}
        obs, _, done, info = env.step(action)
    state_final = obs["0"]["state"]
    actual_pos = state_final[0:3]
    actual_vel = state_final[10:13]

    env.close()

    print(f"\n--- {label} ---")
    print(f"Applied RPM: {np.round(rpm_cmd, 0)}")
    print(f"Model-predicted position after {N_TEST_STEPS} steps: {np.round(predicted_pos, 4)}")
    print(f"Actual simulator position after {N_TEST_STEPS} steps:  {np.round(actual_pos, 4)}")
    print(f"Model-predicted velocity: {np.round(predicted_vel, 4)}")
    print(f"Actual simulator velocity:  {np.round(actual_vel, 4)}")

    for i, axis in enumerate(["x", "y", "z"]):
        pred_sign = np.sign(predicted_pos[i])
        actual_sign = np.sign(actual_pos[i])
        if abs(predicted_pos[i]) > 1e-4 and abs(actual_pos[i]) > 1e-4:
            match = "MATCH" if pred_sign == actual_sign else "*** SIGN MISMATCH ***"
            print(f"  {axis}: predicted sign={pred_sign:+.0f}, actual sign={actual_sign:+.0f}  {match}")


if __name__ == "__main__":
    # motor order for CF2X in this codebase: [0,1,2,3]; try a differential
    # across the first pair vs second pair, and see which real-world
    # direction it produces, then compare to what the model predicts
    run_step_response_test([300, 300, -300, -300], "motors[0,1] up, motors[2,3] down")
    run_step_response_test([300, -300, -300, 300], "motors[0,3] up, motors[1,2] down")
    run_step_response_test([100, 100, 100, 100], "all motors up (should climb, no xy motion)")
