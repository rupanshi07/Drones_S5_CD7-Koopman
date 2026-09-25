"""
Kalman Filter as a state estimator for Koopman-MPC.

Fix vs. previous version: the Kalman filter now provides the FULL filtered
lifted state (27-dim) to MPC, rather than just replacing the 3 position
dimensions. The previous version left 15 other state dimensions unfiltered,
so the filter's correction got swamped. Using the full filtered lifted state
is also more principled: the filter operates in the same space as MPC, so
it can correct noise in velocity, rotation, and angular velocity too.

Also adds a WITH-DISTURBANCE test (payload + wind), since state estimation
matters most when the system is already being stressed.

Three-way comparison:
  (a) perfect  -- ground truth fed to MPC (theoretical upper bound)
  (b) noisy    -- raw noisy measurement fed to MPC (no filtering)
  (c) kalman   -- full filtered lifted state fed to MPC (contribution)
"""

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from build_edmd_dataset import lift
from koopman_mpc import KoopmanMPC, RPM_HOVER

SIM_FREQ   = 240
CTRL_FREQ  = 48
AGGR_PHY_STEPS = int(SIM_FREQ / CTRL_FREQ)
EPISODE_LEN_SEC = 8.0
DRONE_MODEL = DroneModel.CF2X

# Sensor noise: GPS-degraded outdoor scenario.
# At ideal indoor levels (1.5cm pos) the three variants were indistinguishable
# because noise << controller lag. These represent a more challenging scenario.
POS_NOISE_STD  = 0.08    # m     position (degraded GPS, ~8 cm)
VEL_NOISE_STD  = 0.15    # m/s   velocity (IMU integration drift)
ROT_NOISE_STD  = 0.03    #       rotation matrix entries (IMU-derived)
OMEG_NOISE_STD = 0.05    # rad/s angular velocity (~3 deg/s)

Q_SCALE = 1e-4   # process noise: how much we trust model prediction
R_SCALE = 1.0    # measurement noise: matched to sensor std above

# Disturbance schedule (for the WITH-DISTURBANCE test)
PAYLOAD_START_SEC  = 3.0
PAYLOAD_FRAC       = 0.25
WIND_START_SEC     = 3.0
WIND_DURATION_SEC  = 2.0
WIND_FORCE_N       = 0.02
WIND_DIR           = np.array([1.0, 0.0, 0.0])


def reference_trajectory(t, amp=0.4, freq=0.15, z0=1.0):
    return np.array([amp * np.sin(freq * t),
                     amp * np.sin(2 * freq * t),
                     z0])


def add_sensor_noise(raw_state, rng):
    noisy = raw_state.copy()
    noisy[0:3]  += rng.normal(0, POS_NOISE_STD,  3)
    noisy[3:6]  += rng.normal(0, VEL_NOISE_STD,  3)
    noisy[6:15] += rng.normal(0, ROT_NOISE_STD,  9)
    noisy[15:18]+= rng.normal(0, OMEG_NOISE_STD, 3)
    return noisy


class KalmanFilter:
    """
    Discrete Linear Kalman Filter in the 27-dim Koopman lifted space.

    Dynamics model : psi[k+1] = A @ psi[k] + B @ u[k]  (same as MPC's model)
    Measurement model: z[k]   = psi[k] + noise           (identity observation)

    Using the Koopman model as the prediction model is the principled choice:
    our best model of how the lifted state evolves IS the (A, B) pair, so it
    should also be the filter's time-update model.
    """
    def __init__(self, A, B, Q_scale=Q_SCALE, R_scale=R_SCALE):
        self.A = A
        self.B = B
        n = A.shape[0]

        self.Q = Q_scale * np.eye(n)

        # Per-dimension measurement noise, matched to sensor std
        raw_noise_std = np.concatenate([
            np.full(3, POS_NOISE_STD),
            np.full(3, VEL_NOISE_STD),
            np.full(9, ROT_NOISE_STD),
            np.full(3, OMEG_NOISE_STD),
            np.full(n - 18, 0.05),   # cross-terms and bias term
        ])
        self.R = R_scale * np.diag(raw_noise_std ** 2)

        self.psi_est = np.zeros(n)
        self.P = np.eye(n) * 0.1
        self.initialized = False

    def initialize(self, psi0):
        self.psi_est = psi0.copy()
        self.initialized = True

    def update(self, psi_meas, u_prev):
        if not self.initialized:
            self.initialize(psi_meas)
            return self.psi_est

        # --- predict ---
        psi_pred = self.A @ self.psi_est + self.B @ u_prev
        P_pred   = self.A @ self.P @ self.A.T + self.Q

        # --- correct (H = I: identity observation) ---
        S = P_pred + self.R
        K = np.linalg.solve(S.T, P_pred.T).T   # K = P_pred @ S^{-1}
        self.psi_est = psi_pred + K @ (psi_meas - psi_pred)
        self.P = (np.eye(self.A.shape[0]) - K) @ P_pred

        return self.psi_est


def run_episode(state_mode, with_disturbance=False, rng_seed=42):
    model = np.load("data/koopman_model.npz")
    A, B = model["A"], model["B"]
    mpc = KoopmanMPC(A, B, Hp=10, Hc=5,
                     pos_weight=120.0, du_weight=1.0, u_weight=1e-3)
    kf  = KalmanFilter(A, B) if state_mode == "kalman" else None
    rng = np.random.default_rng(rng_seed)

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
        gui=False, record=False, obstacles=False, user_debug_gui=False,
    )
    obs = env.reset()
    client  = env.getPyBulletClient()
    drone_id = env.DRONE_IDS[0]
    nominal_mass = p.getDynamicsInfo(drone_id, -1, physicsClientId=client)[0]

    n_steps = int(EPISODE_LEN_SEC * CTRL_FREQ)
    dt = 1.0 / CTRL_FREQ
    prev_rpm = None
    applied_payload = False

    log = {"t": [], "pos_true": [], "pos_est": [], "ref_pos": [], "rpm": []}

    for step in range(n_steps):
        t = step / CTRL_FREQ

        # --- disturbance injection ---
        if with_disturbance:
            if t >= PAYLOAD_START_SEC and not applied_payload:
                p.changeDynamics(drone_id, -1,
                                  mass=nominal_mass * (1 + PAYLOAD_FRAC),
                                  physicsClientId=client)
                applied_payload = True
            if WIND_START_SEC <= t < WIND_START_SEC + WIND_DURATION_SEC:
                p.applyExternalForce(drone_id, -1,
                                      forceObj=(WIND_FORCE_N * WIND_DIR).tolist(),
                                      posObj=[0, 0, 0], flags=p.LINK_FRAME,
                                      physicsClientId=client)

        # --- state extraction ---
        state = obs["0"]["state"]
        pos_true = state[0:3].copy()
        vel      = state[10:13]
        quat     = state[3:7]
        ang_vel  = state[13:16]
        rotmat   = np.array(p.getMatrixFromQuaternion(quat))
        raw_true = np.hstack([pos_true, vel, rotmat, ang_vel])
        raw_noisy = add_sensor_noise(raw_true, rng)

        # --- choose lifted state for MPC ---
        u_prev = prev_rpm if prev_rpm is not None else np.full(4, RPM_HOVER)
        if state_mode == "perfect":
            psi0 = lift(raw_true.reshape(1, -1))[0]
            pos_est_logged = pos_true.copy()
        elif state_mode == "noisy":
            psi0 = lift(raw_noisy.reshape(1, -1))[0]
            pos_est_logged = raw_noisy[0:3].copy()
        else:  # kalman: use the FULL filtered lifted state
            psi_noisy = lift(raw_noisy.reshape(1, -1))[0]
            psi0 = kf.update(psi_noisy, u_prev)
            pos_est_logged = psi0[0:3].copy()

        ref_window = np.array([reference_trajectory(t + k * dt)
                                for k in range(mpc.Hp)])
        rpm_cmd = mpc.solve(psi0, ref_window, prev_u=prev_rpm)
        prev_rpm = rpm_cmd

        obs, _, done, info = env.step({"0": rpm_cmd})

        log["t"].append(t)
        log["pos_true"].append(pos_true)
        log["pos_est"].append(pos_est_logged)
        log["ref_pos"].append(reference_trajectory(t))
        log["rpm"].append(rpm_cmd)

    env.close()
    for k in log:
        log[k] = np.array(log[k])
    return log


def rmse_and_max(log):
    err = np.linalg.norm(log["pos_true"] - log["ref_pos"], axis=1)
    return np.sqrt(np.mean(err**2)), err.max()


if __name__ == "__main__":
    for scenario, with_dist in [("No disturbance", False),
                                 ("Payload + wind", True)]:
        print(f"\n{'='*55}")
        print(f"  {scenario}")
        print(f"{'='*55}")
        results = {}
        for mode in ["perfect", "noisy", "kalman"]:
            print(f"  Running {mode}...", end=" ", flush=True)
            log = run_episode(mode, with_disturbance=with_dist, rng_seed=42)
            rmse, mx = rmse_and_max(log)
            results[mode] = (rmse, mx)
            tag = scenario.replace(" ", "_").replace("+", "and").lower()
            np.savez(f"data/kalman_{mode}_{tag}.npz", **log)
            print(f"RMSE={rmse:.4f}m  max={mx:.4f}m")

        print(f"\n  {'Mode':<12} {'RMSE (m)':>10} {'Max err (m)':>12}")
        for mode, (rmse, mx) in results.items():
            print(f"  {mode:<12} {rmse:>10.4f} {mx:>12.4f}")
        p_rmse = results["perfect"][0]
        n_rmse = results["noisy"][0]
        k_rmse = results["kalman"][0]
        reduction = 100 * (n_rmse - k_rmse) / (n_rmse - p_rmse + 1e-9)
        print(f"\n  Kalman closes {reduction:.1f}% of the gap between noisy and perfect.")
