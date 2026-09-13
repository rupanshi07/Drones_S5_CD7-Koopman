"""
Step: LQR baseline for comparison against Koopman-MPC and PID.

Standard approach: linearize the nominal 12-state quadrotor dynamics at
hover (small-angle), design LQR gain K via the discrete algebraic Riccati
equation, then convert the LQR's "virtual" commands (thrust + 3 torques)
into actual motor RPMs via the standard X-configuration mixer.

IMPORTANT CAVEAT: kf, km (thrust/torque coefficients) below are the
commonly-published constants for the CF2X model used in this simulator's
asset files. mass, arm length, and inertia are taken directly from your
own printed BaseAviary output (m=0.027, L=0.0397, Ixx=Iyy=0.000014,
Izz=0.000022), which are exact. kf/km should be double-checked against
your installed version's cf2x.urdf if results look physically wrong.

Given how many sign-convention bugs we hit with the Koopman model, this
file includes a standalone mixer sign-check (mirroring diagnose_sign.py)
that MUST be run and confirmed correct before trusting this in the
disturbance comparison.
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

# --- physical constants ---
MASS = 0.027          # kg, from your BaseAviary printout
ARM_LENGTH = 0.0397    # m, from your BaseAviary printout
IXX = 0.000014
IYY = 0.000014
IZZ = 0.000022
G = 9.81

# published CF2X thrust/torque coefficients (VERIFY against your installed
# cf2x.urdf if the sign-check below looks wrong)
KF = 3.16e-10   # N / (rad/s)^2 -- note: RPM must be converted to rad/s
KM = 7.94e-12   # Nm / (rad/s)^2

RPM_HOVER = 14468.0
RPM_MAX = 21713.0
CTRL_FREQ = 48

RPM_TO_RADS = 2 * np.pi / 60.0


def build_linearized_model():
    """12-state model: [x,y,z,vx,vy,vz,phi,theta,psi,p,q,r].
    Control (virtual): [f, taux, tauy, tauz] -- thrust deviation from
    hover (N) and 3 body torques (Nm)."""
    n = 12
    m_ctrl = 4
    A = np.zeros((n, n))
    B = np.zeros((n, m_ctrl))

    # position derivatives = velocity
    A[0, 3] = 1
    A[1, 4] = 1
    A[2, 5] = 1
    # horizontal acceleration from tilt (small-angle)
    A[3, 7] = G    # vx_dot = g * theta (pitch)
    A[4, 6] = -G   # vy_dot = -g * phi (roll)
    # vz_dot from thrust deviation
    B[5, 0] = 1.0 / MASS
    # attitude derivatives = angular rates
    A[6, 9] = 1
    A[7, 10] = 1
    A[8, 11] = 1
    # angular acceleration from torques
    B[9, 1] = 1.0 / IXX
    B[10, 2] = 1.0 / IYY
    B[11, 3] = 1.0 / IZZ

    return A, B


def discretize(A, B, dt):
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(A.shape[0]), 0), dt)
    return Ad, Bd


def build_outer_position_model():
    """Outer loop: [x,y,z,vx,vy,vz] -> virtual [ax_cmd, ay_cmd, az_cmd]
    (desired acceleration). Trivial double-integrator per axis."""
    n = 6
    A = np.zeros((n, n))
    B = np.zeros((n, 3))
    A[0, 3] = 1
    A[1, 4] = 1
    A[2, 5] = 1
    B[3, 0] = 1
    B[4, 1] = 1
    B[5, 2] = 1
    return A, B


def build_inner_attitude_model():
    """Inner loop: [phi,theta,psi,p,q,r] -> torques [taux,tauy,tauz]."""
    n = 6
    A = np.zeros((n, n))
    B = np.zeros((n, 3))
    A[0, 3] = 1
    A[1, 4] = 1
    A[2, 5] = 1
    B[3, 0] = 1.0 / IXX
    B[4, 1] = 1.0 / IYY
    B[5, 2] = 1.0 / IZZ
    return A, B


MAX_TILT_RAD = np.radians(20.0)  # hard safety limit -- never command more than this


class LQRController:
    def __init__(self):
        # outer position loop (slow timescale)
        Ao, Bo = build_outer_position_model()
        Ado, Bdo = discretize(Ao, Bo, 1.0 / CTRL_FREQ)
        Qo = np.diag([20.0, 20.0, 30.0, 2.0, 2.0, 3.0])
        Ro = np.diag([1.0, 1.0, 1.0])
        Po = solve_discrete_are(Ado, Bdo, Qo, Ro)
        self.Ko = np.linalg.solve(Ro + Bdo.T @ Po @ Bdo, Bdo.T @ Po @ Ado)

        # inner attitude loop (fast timescale, tuned much more aggressively
        # since attitude dynamics are much faster than position dynamics)
        Ai, Bi = build_inner_attitude_model()
        Adi, Bdi = discretize(Ai, Bi, 1.0 / CTRL_FREQ)
        Qi = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 5.0])
        Ri = np.diag([2e9, 2e9, 2e9])
        Pi = solve_discrete_are(Adi, Bdi, Qi, Ri)
        self.Ki = np.linalg.solve(Ri + Bdi.T @ Pi @ Bdi, Bdi.T @ Pi @ Adi)

    def compute(self, state12, target_pos):
        pos, vel = state12[0:3], state12[3:6]
        phi, theta, psi = state12[6], state12[7], state12[8]
        rates = state12[9:12]

        # --- outer loop: position error -> desired acceleration ---
        outer_state = np.hstack([pos, vel])
        outer_target = np.hstack([target_pos, [0, 0, 0]])
        acc_cmd = -self.Ko @ (outer_state - outer_target)  # [ax, ay, az]

        # --- convert desired acceleration to desired tilt + thrust ---
        # (standard small-angle inversion, with an explicit safety clamp --
        # this clamp is the key fix: it prevents the controller from ever
        # requesting a tilt angle steep enough to lose vertical thrust)
        theta_des = np.clip(acc_cmd[0] / G, -np.tan(MAX_TILT_RAD), np.tan(MAX_TILT_RAD))
        phi_des = np.clip(-acc_cmd[1] / G, -np.tan(MAX_TILT_RAD), np.tan(MAX_TILT_RAD))
        theta_des = np.arctan(theta_des)
        phi_des = np.arctan(phi_des)
        f_cmd = MASS * acc_cmd[2]

        # --- inner loop: attitude error -> torques ---
        inner_state = np.hstack([phi, theta, psi, rates])
        inner_target = np.hstack([phi_des, theta_des, 0.0, 0, 0, 0])
        torque_cmd = -self.Ki @ (inner_state - inner_target)  # [taux, tauy, tauz]

        return virtual_to_rpm(f_cmd, *torque_cmd)


def virtual_to_rpm(f, taux, tauy, tauz):
    """Convert [thrust deviation, taux, tauy, tauz] to 4 RPM commands,
    linearized around hover. X-configuration mixer (standard form) --
    VERIFY sign convention with the standalone test below before trusting."""
    # thrust and torque as functions of omega^2 (rad/s):
    #   f_total = kf * sum(omega_i^2)
    #   taux    = kf * L/sqrt(2) * (-w0^2 - w1^2 + w2^2 + w3^2)
    #   tauy    = kf * L/sqrt(2) * (-w0^2 + w1^2 + w2^2 - w3^2)
    #   tauz    = km * (-w0^2 + w1^2 - w2^2 + w3^2)
    # invert this linear system for omega_i^2 deviations, given small
    # deviations around hover omega_hover
    omega_hover = RPM_HOVER * RPM_TO_RADS
    f_hover = KF * 4 * omega_hover ** 2
    L_term = ARM_LENGTH / np.sqrt(2)

    mix = np.array([
        [KF, KF, KF, KF],
        [-KF * L_term, -KF * L_term, KF * L_term, KF * L_term],
        [-KF * L_term, KF * L_term, KF * L_term, -KF * L_term],
        [-KM, KM, -KM, KM],
    ])
    rhs = np.array([f, taux, tauy, tauz])
    omega_sq_delta = np.linalg.solve(mix, rhs)

    omega_sq_hover = omega_hover ** 2
    omega = np.sqrt(np.clip(omega_sq_hover + omega_sq_delta, 0, None))
    rpm = omega / RPM_TO_RADS
    return np.clip(rpm, 0, RPM_MAX)


if __name__ == "__main__":
    # --- standalone check: does the mixer respond sensibly? ---
    print("Mixer sign-check (no simulator needed for this part):")
    for label, (f, tx, ty, tz) in [
        ("thrust up only", (0.002, 0, 0, 0)),
        ("+taux only", (0, 0.000005, 0, 0)),
        ("+tauy only", (0, 0, 0.000005, 0)),
        ("+tauz only", (0, 0, 0, 0.000002)),
    ]:
        rpm = virtual_to_rpm(f, tx, ty, tz)
        print(f"  {label:20s} -> rpm={np.round(rpm, 1)}  "
              f"(deviation from hover: {np.round(rpm - RPM_HOVER, 1)})")

    print("\nNext: run diagnose_lqr_sign.py against the real simulator")
    print("to confirm these virtual commands produce motion in the")
    print("expected physical direction, exactly as we did for the")
    print("Koopman model with diagnose_sign.py.")
