"""Classical baselines for the MuJoCo CF2X: cascaded PID and cascaded LQR.

Both share one structure (same as the repo's "two-timescale" idea):
    outer loop : position error  -> desired acceleration -> desired thrust vector -> desired attitude R_d
    inner loop : attitude error  -> body torques
    mixer      : (thrust, torques) -> 4 motor thrusts -> 4 RPMs
They differ only in how the gains are obtained:
    PID : hand-set PD gains + integral action on position
    LQR : gains from the discrete algebraic Riccati equation (no integrator)
Interface (same as the Koopman-MPC):  ctrl.rpm(state_18, ref_xyz) -> 4 RPMs
state_18 = [p(3), v(3), vec(R)(9), omega_world(3)]
"""
import numpy as np
from scipy.linalg import solve_discrete_are


def _vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def dlqr_double_integrator(dt, q_pos, q_vel, r):
    """LQR gain K=[k_pos, k_vel] for  x=[e, edot], x+ = [[1,dt],[0,1]]x + [[dt^2/2],[dt]]u  (u = acceleration)."""
    A = np.array([[1, dt], [0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    P = solve_discrete_are(A, B, np.diag([q_pos, q_vel]), np.array([[r]]))
    K = np.linalg.solve(B.T @ P @ B + r, B.T @ P @ A)
    return K[0]


class CascadedController:
    ARM, KYAW = 0.028, 0.0251                   # arm length (m), yaw-torque / thrust ratio (from the XML)
    I = np.array([1.4e-5, 1.4e-5, 2.17e-5])
    MAX_TILT_ACC = 8.0                          # m/s^2 cap on horizontal acceleration command (~39 deg)
    INT_LIM = np.array([1.0, 1.0, 3.0])         # integrator clip (m*s); z is larger so ~100 % payload can be trimmed out

    def __init__(self, env, kp, kd, ki, k_att, k_rate, name="cascade"):
        self.env, self.name = env, name
        self.kp, self.kd, self.ki = map(np.asarray, (kp, kd, ki))
        self.k_att, self.k_rate = np.asarray(k_att), np.asarray(k_rate)
        l, k = self.ARM, self.KYAW
        M = np.array([[1, 1, 1, 1],
                      [l, l, -l, -l],            # roll  torque: motors 0,1 (y>0, left) up -> +x roll
                      [-l, l, l, -l],           # pitch torque: motors 1,2 (x<0, rear) up -> +y pitch
                      [-k, k, -k, k]])          # yaw   torque: CCW motors 0,2 (-), CW motors 1,3 (+)
        self.Minv = np.linalg.inv(M)
        self.reset()

    def reset(self):
        self.int_e = np.zeros(3)
        self.prev_ref = None
        self.ref_v = np.zeros(3)

    def rpm(self, state, ref):
        env = self.env
        dt = env.dt
        p, v = state[0:3], state[3:6]
        R = state[6:15].reshape(3, 3)
        wb = R.T @ state[15:18]                  # body-frame angular velocity
        ref = np.asarray(ref, dtype=float)

        # reference velocity feed-forward (finite difference, low-passed)
        if self.prev_ref is not None:
            self.ref_v = 0.7 * self.ref_v + 0.3 * (ref - self.prev_ref) / dt
        self.prev_ref = ref.copy()

        # ---- outer loop: desired acceleration ----
        e_p, e_v = ref - p, self.ref_v - v
        self.int_e = np.clip(self.int_e + e_p * dt, -self.INT_LIM, self.INT_LIM)
        a = self.kp * e_p + self.kd * e_v + self.ki * self.int_e
        n_xy = np.linalg.norm(a[:2])
        if n_xy > self.MAX_TILT_ACC:
            a[:2] *= self.MAX_TILT_ACC / n_xy
        f_vec = env.MASS * (a + np.array([0, 0, env.G]))
        f_vec[2] = max(f_vec[2], 0.3 * env.MASS * env.G)
        T = float(f_vec @ R[:, 2])

        # ---- desired attitude (yaw held at 0) ----
        b3 = f_vec / np.linalg.norm(f_vec)
        b2 = np.cross(b3, [1.0, 0.0, 0.0]); b2 /= np.linalg.norm(b2)
        b1 = np.cross(b2, b3)
        Rd = np.column_stack([b1, b2, b3])

        # ---- inner loop: geometric attitude error -> torque ----
        eR = 0.5 * _vee(Rd.T @ R - R.T @ Rd)
        alpha = -self.k_att * eR - self.k_rate * wb          # desired angular acceleration
        tau = self.I * alpha + np.cross(wb, self.I * wb)

        # ---- mixer ----
        f = self.Minv @ np.array([T, *tau])
        f = np.clip(f, 0.0, env.KF * env.MAX_RPM ** 2)
        return np.sqrt(f / env.KF)


def make_pid(env):
    # outer: z fastest, x/y wn ~ 3 rad/s ; inner: wn ~ 18 rad/s (roll/pitch), slower yaw
    return CascadedController(env, kp=[9, 9, 16], kd=[5.4, 5.4, 7.0], ki=[1.0, 1.0, 3.0],
                              k_att=[320, 320, 80], k_rate=[24, 24, 12], name="PID")


def make_lqr(env, integral=True):
    """Per-axis double-integrator LQR gains in the cascade. integral=True adds a small I term so that payload/wind
    offsets are trimmed out (LQR+I); integral=False is the pure state-feedback version (steady-state sag under load)."""
    dt = env.dt
    ki = [0.5, 0.5, 1.5] if integral else [0, 0, 0]
    k_pxy = dlqr_double_integrator(dt, q_pos=10.0, q_vel=2.0, r=1.0)
    k_pz = dlqr_double_integrator(dt, q_pos=40.0, q_vel=4.0, r=1.0)
    k_a = dlqr_double_integrator(dt, q_pos=1000.0, q_vel=10.0, r=1.0)     # roll / pitch
    k_y = dlqr_double_integrator(dt, q_pos=80.0, q_vel=2.0, r=1.0)        # yaw
    c = CascadedController(env,
                           kp=[k_pxy[0], k_pxy[0], k_pz[0]], kd=[k_pxy[1], k_pxy[1], k_pz[1]], ki=ki,
                           k_att=[k_a[0], k_a[0], k_y[0]], k_rate=[k_a[1], k_a[1], k_y[1]], name="LQR")
    return c
