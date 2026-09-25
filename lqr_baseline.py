"""
Cascaded LQR baseline: outer position loop -> inner attitude loop.
Validated mixer (diagnose_lqr_sign.py confirmed correct directions).
Rate-limited attitude AND thrust references to prevent step-input overshoot.
MAX_THRUST_RATE raised to 5.0 N/s (was 0.5, too slow to arrest free-fall).
"""
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

MASS        = 0.027
ARM_LENGTH  = 0.0397
IXX         = 0.000014
IYY         = 0.000014
IZZ         = 0.000022
G           = 9.81
KF          = 3.16e-10
KM          = 7.94e-12
RPM_HOVER   = 14468.0
RPM_MAX     = 21713.0
CTRL_FREQ   = 48
RPM_TO_RADS = 2 * np.pi / 60.0
MAX_TILT_RAD              = np.radians(20.0)
MAX_TILT_RATE_RAD_PER_SEC = np.radians(30.0)
MAX_THRUST_RATE           = 5.0   # N/s (raised from 0.5 -- see diagnose_lqr_verbose.py)


def build_outer_position_model():
    A = np.zeros((6, 6)); B = np.zeros((6, 3))
    A[0,3]=1; A[1,4]=1; A[2,5]=1
    B[3,0]=1; B[4,1]=1; B[5,2]=1
    return A, B


def build_inner_attitude_model():
    A = np.zeros((6, 6)); B = np.zeros((6, 3))
    A[0,3]=1; A[1,4]=1; A[2,5]=1
    B[3,0]=1/IXX; B[4,1]=1/IYY; B[5,2]=1/IZZ
    return A, B


def discretize(A, B, dt):
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(A.shape[0]), 0), dt)
    return Ad, Bd


def virtual_to_rpm(f, taux, tauy, tauz):
    """PWM-based mixer, matching DSLPIDControl.py exactly (CF2X)."""
    PWM2RPM_SCALE = 0.2685
    PWM2RPM_CONST = 4070.3
    MIN_PWM = 20000
    MAX_PWM = 65535
    MIXER_MATRIX = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]])

    thrust_total = MASS * G + f
    thrust_pwm = (np.sqrt(max(thrust_total, 0) / (4 * KF)) - PWM2RPM_CONST) / PWM2RPM_SCALE

    target_torques = np.array([taux, tauy, tauz])
    pwm = thrust_pwm + np.dot(MIXER_MATRIX, target_torques)
    pwm = np.clip(pwm, MIN_PWM, MAX_PWM)
    return PWM2RPM_SCALE * pwm + PWM2RPM_CONST


class LQRController:
    def __init__(self, verbose=False):
        dt = 1.0 / CTRL_FREQ

        Ao, Bo = build_outer_position_model()
        Ado, Bdo = discretize(Ao, Bo, dt)
        Qo = np.diag([20., 20., 30., 2., 2., 3.])
        Ro = np.diag([500., 500., 500.])
        Po = solve_discrete_are(Ado, Bdo, Qo, Ro)
        self.Ko = np.linalg.solve(Ro + Bdo.T@Po@Bdo, Bdo.T@Po@Ado)

        Ai, Bi = build_inner_attitude_model()
        Adi, Bdi = discretize(Ai, Bi, dt)
        Qi = np.diag([50., 50., 50., 20., 20., 5.])
        Ri = np.diag([1e2, 1e2, 1e2])  # raised from 2e9: inner loop was generating ~4e-3 Nm torques
        Pi = solve_discrete_are(Adi, Bdi, Qi, Ri)
        self.Ki = np.linalg.solve(Ri + Bdi.T@Pi@Bdi, Bdi.T@Pi@Adi)

        self._phi_des_prev   = 0.0
        self._theta_des_prev = 0.0
        self._f_cmd_prev     = 0.0
        self._dt             = dt
        self.verbose         = verbose
        self._step           = 0

    def compute(self, state12, target_pos):
        pos, vel           = state12[0:3], state12[3:6]
        phi, theta, psi    = state12[6], state12[7], state12[8]
        rates              = state12[9:12]

        # outer loop
        acc_cmd = -self.Ko @ (np.hstack([pos, vel]) - np.hstack([target_pos, [0,0,0]]))

        # desired tilt (clamped to +-20 deg)
        theta_des_raw = np.arctan(np.clip( acc_cmd[0]/G, -np.tan(MAX_TILT_RAD), np.tan(MAX_TILT_RAD)))
        phi_des_raw   = np.arctan(np.clip(-acc_cmd[1]/G, -np.tan(MAX_TILT_RAD), np.tan(MAX_TILT_RAD)))
        f_cmd_raw     = MASS * acc_cmd[2]

        # rate-limit attitude reference
        max_ang  = MAX_TILT_RATE_RAD_PER_SEC * self._dt
        theta_des = np.clip(theta_des_raw, self._theta_des_prev - max_ang, self._theta_des_prev + max_ang)
        phi_des   = np.clip(phi_des_raw,   self._phi_des_prev   - max_ang, self._phi_des_prev   + max_ang)

        # rate-limit thrust command
        max_f = MAX_THRUST_RATE * self._dt
        f_cmd = np.clip(f_cmd_raw, self._f_cmd_prev - max_f, self._f_cmd_prev + max_f)

        self._theta_des_prev = theta_des
        self._phi_des_prev   = phi_des
        self._f_cmd_prev     = f_cmd

        # inner loop
        torque_cmd = -self.Ki @ (np.hstack([phi, theta, psi, rates])
                                 - np.hstack([phi_des, theta_des, 0., 0, 0, 0]))

        if self.verbose and self._step < 40:
            print(f"  step={self._step:3d} z={pos[2]:.3f} z_err={pos[2]-target_pos[2]:+.3f}  "
                  f"acc_z={acc_cmd[2]:+.4f}  f_raw={f_cmd_raw:+.4f}  "
                  f"f_clamp={f_cmd:+.4f}  theta_des={np.degrees(theta_des):+.2f}deg  "
                  f"vz={vel[2]:+.4f}")
        self._step += 1

        torque_cmd_pwm = torque_cmd * np.array([1/IXX, 1/IYY, 1/IZZ])
        torque_cmd_pwm = np.clip(torque_cmd_pwm, -3200, 3200)
        return virtual_to_rpm(f_cmd, *torque_cmd_pwm)
