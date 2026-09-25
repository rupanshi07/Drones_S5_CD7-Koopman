import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
import pybullet as p

MASS = 0.027
GRAVITY = 9.8
IXX = 1.4e-5
IYY = 1.4e-5
IZZ = 2.2e-5
KF = 3.16e-10
KM = 7.94e-12
L = 0.0397

ALLOC_MATRIX = np.array([
    [KF,     KF,     KF,     KF],
    [-KF*L, -KF*L,   KF*L,   KF*L],
    [-KF*L,  KF*L,   KF*L,  -KF*L],
    [-KM,    KM,    -KM,     KM]
])
ALLOC_INV = np.linalg.inv(ALLOC_MATRIX)

N_STATE = 12
N_INPUT = 4

A = np.zeros((N_STATE, N_STATE))
A[0, 3] = 1.0
A[1, 4] = 1.0
A[2, 5] = 1.0
A[3, 7] = GRAVITY
A[4, 6] = -GRAVITY
A[6, 9] = 1.0
A[7, 10] = 1.0
A[8, 11] = 1.0

B = np.zeros((N_STATE, N_INPUT))
B[5, 0] = 1.0 / MASS
B[9, 1] = 1.0 / IXX
B[10, 2] = 1.0 / IYY
B[11, 3] = 1.0 / IZZ

CTRL_FREQ = 48
DT = 1.0 / CTRL_FREQ
C_dummy = np.eye(N_STATE)
D_dummy = np.zeros((N_STATE, N_INPUT))
Ad, Bd, _, _, _ = cont2discrete((A, B, C_dummy, D_dummy), DT)

Q = np.diag([
    10, 10, 20,
    1, 1, 1,
    50, 50, 10,
    1, 1, 1
])
R = np.diag([0.5, 0.5, 0.5, 0.5])

P = solve_discrete_are(Ad, Bd, Q, R)
K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

FINAL_TARGET_Z = 0.3
RAMP_RATE = 0.001
_current_target_z = None

HOVER_THRUST = MASS * GRAVITY


def _extract_lqr_state(raw_state_16):
    pos = raw_state_16[0:3]
    quat = raw_state_16[3:7]
    vel = raw_state_16[10:13]
    ang_vel = raw_state_16[13:16]
    rpy = p.getEulerFromQuaternion(quat)

    x = np.zeros(N_STATE)
    x[0:3] = pos
    x[3:6] = vel
    x[6:9] = rpy
    x[9:12] = ang_vel
    return x


def solve_lqr(current_state_raw, condition=None):
    global _current_target_z

    if np.any(np.isnan(current_state_raw)) or np.any(np.isinf(current_state_raw)):
        return np.full(4, 14436.0)

    if _current_target_z is None:
        _current_target_z = current_state_raw[2]
    if _current_target_z < FINAL_TARGET_Z:
        _current_target_z = min(_current_target_z + RAMP_RATE, FINAL_TARGET_Z)

    x = _extract_lqr_state(current_state_raw)

    x_target = np.zeros(N_STATE)
    x_target[2] = _current_target_z

    error = x - x_target
    u_delta = -K @ error

    total_thrust = max(HOVER_THRUST + u_delta[0], 0.0)
    tau_roll, tau_pitch, tau_yaw = u_delta[1], u_delta[2], u_delta[3]

    u_vec = np.array([total_thrust, tau_roll, tau_pitch, tau_yaw])
    motor_sq = ALLOC_INV @ u_vec
    motor_sq = np.clip(motor_sq, 0, None)
    rpms = np.sqrt(motor_sq)

    return np.clip(rpms, 0, 25000)
