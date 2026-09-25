"""
Mock test: exercises run_mpc_episode()'s data-flow logic (state extraction,
lifting, MPC solve, action dict shape) using a fake environment that mimics
gym-pybullet-drones' obs/action interface, WITHOUT needing pybullet itself
installed. Not a physics test -- just catches shape/index/API-usage bugs.
"""

import numpy as np
import types
import sys

# --- fake pybullet module (just enough for koopman_mpc/run_mpc_closed_loop) ---
fake_p = types.ModuleType("pybullet")
def getMatrixFromQuaternion(quat):
    x, y, z, w = quat
    return [
        1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y),
    ]
fake_p.getMatrixFromQuaternion = getMatrixFromQuaternion
sys.modules["pybullet"] = fake_p

sys.path.insert(0, "/home/claude")
from build_edmd_dataset import lift  # noqa: E402
from koopman_mpc import KoopmanMPC   # noqa: E402


class FakeCtrlAviary:
    """Mimics just enough of CtrlAviary's interface: dict obs/action, a
    20-element state vector, and a step() that just drifts state slightly
    (not real physics -- purely for exercising the calling code's shapes)."""
    def __init__(self, initial_xyzs, **kwargs):
        self.pos = initial_xyzs[0].copy()
        self.SIM_FREQ = kwargs.get("freq", 240)

    def reset(self):
        state = np.zeros(20)
        state[0:3] = self.pos
        state[3:7] = [0, 0, 0, 1]  # identity quaternion
        return {"0": {"state": state}}

    def step(self, action):
        rpm = action["0"]
        # fake drift: nudge position slightly toward higher-average-rpm side
        self.pos += 0.0001 * (rpm[0] - rpm[2])  # arbitrary, just for shape testing
        state = np.zeros(20)
        state[0:3] = self.pos
        state[3:7] = [0, 0, 0, 1]
        return {"0": {"state": state}}, 0, False, {}

    def close(self):
        pass


def run_mock_episode():
    n = 27
    A = 0.99 * np.eye(n) + 0.0005 * np.random.randn(n, n)
    B = np.zeros((n, 4))
    B[3, :] = [1e-4, 1e-4, -1e-4, -1e-4]
    B[4, :] = [1e-4, -1e-4, -1e-4, 1e-4]
    B[5, :] = [2e-5, 2e-5, 2e-5, 2e-5]

    mpc = KoopmanMPC(A, B, Hp=5, Hc=3, pos_weight=50.0)  # small Hp for a fast test

    def reference_trajectory(t, amp=0.4, freq=0.15, z0=1.0):
        x = amp * np.sin(freq * t)
        y = amp * np.sin(2 * freq * t)
        return np.array([x, y, z0])

    start_pos = reference_trajectory(0.0)
    env = FakeCtrlAviary(initial_xyzs=np.array([start_pos]), freq=240)
    obs = env.reset()

    CTRL_FREQ = 48
    dt = 1.0 / CTRL_FREQ
    n_ctrl_steps = 10  # short test run

    for ctrl_step in range(n_ctrl_steps):
        t = ctrl_step / CTRL_FREQ
        state = obs["0"]["state"]
        pos = state[0:3]
        vel = state[10:13]
        quat = state[3:7]
        ang_vel = state[13:16]
        rotmat = np.array(fake_p.getMatrixFromQuaternion(quat))

        raw_state = np.hstack([pos, vel, rotmat, ang_vel]).reshape(1, -1)
        psi0 = lift(raw_state)[0]

        ref_window = np.array([reference_trajectory(t + k * dt) for k in range(mpc.Hp)])
        rpm_cmd = mpc.solve(psi0, ref_window)
        assert rpm_cmd.shape == (4,), f"BUG: rpm_cmd shape wrong: {rpm_cmd.shape}"

        action = {"0": rpm_cmd}
        obs, _, done, info = env.step(action)
        print(f"t={t:.3f}  pos={np.round(pos,4)}  rpm={np.round(rpm_cmd,1)}")

    print("\nMock episode completed without shape/index errors.")


if __name__ == "__main__":
    run_mock_episode()
