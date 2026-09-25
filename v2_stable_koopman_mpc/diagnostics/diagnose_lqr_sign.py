"""
Verify the LQR mixer's directional conventions against the real simulator,
using the same "apply a known input, compare predicted vs actual direction"
methodology as diagnose_sign.py caught the Koopman sign bug with.

Here "predicted" means: what direction should this virtual command move
the drone in, according to the standard physics convention assumed in
lqr_baseline.py's build_linearized_model()? We then check the real
simulator agrees.
"""

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from lqr_baseline import virtual_to_rpm, RPM_HOVER

SIM_FREQ = 240
CTRL_FREQ = 48
AGGR_PHY_STEPS = int(SIM_FREQ / CTRL_FREQ)
N_TEST_STEPS = 10


def run_virtual_cmd_test(f, taux, tauy, tauz, label, expected_desc):
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

    rpm_cmd = virtual_to_rpm(f, taux, tauy, tauz)

    for _ in range(N_TEST_STEPS):
        action = {"0": rpm_cmd}
        obs, _, done, info = env.step(action)

    state = obs["0"]["state"]
    pos, quat = state[0:3], state[3:7]
    rotmat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    roll = np.degrees(np.arctan2(rotmat[2, 1], rotmat[2, 2]))
    pitch = np.degrees(-np.arcsin(np.clip(rotmat[2, 0], -1, 1)))

    env.close()

    print(f"\n--- {label} ---")
    print(f"Applied RPM: {np.round(rpm_cmd, 0)}")
    print(f"Expected: {expected_desc}")
    print(f"Actual position after {N_TEST_STEPS} steps: {np.round(pos, 4)}")
    print(f"Actual roll={roll:.2f} deg, pitch={pitch:.2f} deg")


if __name__ == "__main__":
    run_virtual_cmd_test(0.002, 0, 0, 0, "thrust up only",
                          "should climb (z increases), no roll/pitch")
    run_virtual_cmd_test(0, 0.000005, 0, 0, "+taux only",
                          "should produce a roll (nonzero roll angle)")
    run_virtual_cmd_test(0, 0, 0.000005, 0, "+tauy only",
                          "should produce a pitch (nonzero pitch angle)")

    print("\n\nCheck: does 'thrust up' actually climb? Does +taux produce")
    print("roll (not pitch)? Does +tauy produce pitch (not roll)? If any")
    print("of these are swapped or backwards, tell me the exact numbers")
    print("and I'll fix the corresponding sign/row in virtual_to_rpm().")
