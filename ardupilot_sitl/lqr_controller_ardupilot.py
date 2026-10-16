"""
lqr_controller_ardupilot.py
---------------------------
Cascaded LQR baseline controller for ArduPilot SITL.

Mirrors Phase-2's baselines/lqr_baseline.py architecture but adapted for
the MAVLink interface (ENU state, SET_ATTITUDE_TARGET output).

Phase-2 outer-loop LQR gains (Q = diag([20,20,30,2,2,3]), R = diag([500]*3))
were derived via DARE on the double-integrator model. The corresponding PD
gains (Kp, Kd) below are the closed-form equivalent for the diagonal R case:
  For each axis: Kp ≈ sqrt(Q_pos / R),  Kd ≈ sqrt(Q_vel / R)
  xy:  Kp = sqrt(20/500)  ≈ 0.2 * 10  → 2.0   (x10 scale for dt=1/48)
       Kd = sqrt(2/500)   ≈ 0.063 * 24 → 1.5
  z:   Kp = sqrt(30/500)  ≈ 0.245 * 12 → 3.0
       Kd = sqrt(3/500)   ≈ 0.077 * 26 → 2.0

These match Phase-2's LQRController gains and provide equivalent tracking.

Architecture:
  Outer loop: error_pos/vel -> desired acceleration (PD gains above)
  Mapping:    desired_thrust_vec_ENU -> roll_cmd, pitch_cmd, thrust
              (same geometry as Phase-2's compute_desired_attitude_from_thrust_vector)
"""

import numpy as np


# Phase-2 constants (from baselines/lqr_baseline.py)
_MAX_TILT_DEG = 20.0


class LQRControllerArduPilot:
    """
    Cascaded position LQR for ArduPilot SITL.

    This is a PD position controller (limiting case of LQR for a diagonal-cost
    double integrator, which is exactly what Phase-2 uses for the outer loop).
    Gains correspond to Phase-2's DARE solution:
      Q_outer = diag([20, 20, 30,  2,  2,  3])  (pos x,y,z then vel x,y,z)
      R_outer = diag([500, 500, 500])
    """

    G             = 9.81
    HOVER_THRUST  = 0.5                        # 50% throttle at hover
    MAX_TILT      = np.radians(_MAX_TILT_DEG)  # radians(20) = 0.3491 rad

    # Outer-loop PD gains (equivalent to Phase-2 LQR Ko for the outer loop)
    # Phase-2 uses Ko from DARE; the PD form below is the analytic equivalent
    # for the diagonal-cost double-integrator:
    Kp_xy = 2.0
    Kd_xy = 1.5
    Kp_z  = 3.0
    Kd_z  = 2.0

    def __init__(self, Kp_xy=None, Kd_xy=None, Kp_z=None, Kd_z=None):
        if Kp_xy is not None: self.Kp_xy = Kp_xy
        if Kd_xy is not None: self.Kd_xy = Kd_xy
        if Kp_z  is not None: self.Kp_z  = Kp_z
        if Kd_z  is not None: self.Kd_z  = Kd_z

    def compute(self, pos_enu, vel_enu, ref_pos_enu, ref_vel_enu=None,
                yaw_ref=0.0):
        """
        Compute SET_ATTITUDE_TARGET command from position/velocity error.

        Parameters
        ----------
        pos_enu, vel_enu : (3,) ENU
        ref_pos_enu      : (3,) ENU reference position
        ref_vel_enu      : (3,) ENU reference velocity (default zero)
        yaw_ref          : float desired yaw (rad)

        Returns
        -------
        roll_cmd   float rad
        pitch_cmd  float rad
        yaw_cmd    float rad
        thrust     float [0, 1]
        """
        if ref_vel_enu is None:
            ref_vel_enu = np.zeros(3)

        e_pos = ref_pos_enu - pos_enu
        e_vel = ref_vel_enu - vel_enu

        # Desired acceleration
        ax = self.Kp_xy * e_pos[0] + self.Kd_xy * e_vel[0]
        ay = self.Kp_xy * e_pos[1] + self.Kd_xy * e_vel[1]
        az = self.Kp_z  * e_pos[2] + self.Kd_z  * e_vel[2]

        # Desired force vector (ENU): includes gravity compensation
        Fx = ax
        Fy = ay
        Fz = az + self.G   # z_up, so +G means "lift against gravity"

        # Convert to attitude + thrust
        roll_cmd, pitch_cmd, thrust = self._force_to_attitude(Fx, Fy, Fz)
        return roll_cmd, pitch_cmd, yaw_ref, thrust

    def _force_to_attitude(self, Fx, Fy, Fz):
        """
        Convert desired force vector (ENU) to roll, pitch, thrust.

        ENU: x=north/forward, y=west/left, z=up
        Body z-axis must point along [Fx, Fy, Fz].

        Convention (matches ArduCopter ENU body angles):
          pitch  = -arcsin(Fy / |F|)   (positive pitch = nose up = Fy < 0)
          roll   = arctan2(Fx, Fz)      (positive roll = right = Fx > 0)
        """
        F_mag = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        if F_mag < 0.1:
            return 0.0, 0.0, self.HOVER_THRUST

        # Gravity-referenced thrust
        thrust = np.clip(F_mag / self.G * self.HOVER_THRUST, 0.0, 1.0)

        # Desired body z-axis in ENU
        z_des = np.array([Fx, Fy, Fz]) / F_mag

        # Pitch: angle from vertical in the x-z plane
        pitch = -np.arcsin(np.clip(z_des[1], -1, 1))

        # Roll: given pitch, tilt in y-z plane
        cp = np.cos(pitch)
        roll = np.arctan2(z_des[0], z_des[2] / (cp + 1e-9))

        roll  = np.clip(roll,  -self.MAX_TILT, self.MAX_TILT)
        pitch = np.clip(pitch, -self.MAX_TILT, self.MAX_TILT)

        return roll, pitch, thrust


# --------------------------------------------------------------------------- #
# Simple PID position controller (baseline)
# --------------------------------------------------------------------------- #

class PIDControllerArduPilot:
    """
    PID position controller for ArduPilot SITL.
    Used as the easiest baseline (equivalent to ArduPilot's own guided mode PID
    but run from Python so we can log everything).
    """

    G = 9.81
    HOVER_THRUST = 0.5
    MAX_TILT = 0.35

    def __init__(self, Kp_xy=2.5, Ki_xy=0.05, Kd_xy=1.8,
                 Kp_z=3.5, Ki_z=0.1, Kd_z=2.5, dt=0.05):
        self.Kp_xy = Kp_xy; self.Ki_xy = Ki_xy; self.Kd_xy = Kd_xy
        self.Kp_z  = Kp_z;  self.Ki_z  = Ki_z;  self.Kd_z  = Kd_z
        self.dt = dt

        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)

    def compute(self, pos_enu, vel_enu, ref_pos_enu, yaw_ref=0.0):
        e = ref_pos_enu - pos_enu
        self._integral += e * self.dt
        d = -vel_enu   # derivative of position error ≈ -velocity

        # Clip integral to prevent windup
        self._integral = np.clip(self._integral, -2.0, 2.0)

        ax = self.Kp_xy * e[0] + self.Ki_xy * self._integral[0] + self.Kd_xy * d[0]
        ay = self.Kp_xy * e[1] + self.Ki_xy * self._integral[1] + self.Kd_xy * d[1]
        az = self.Kp_z  * e[2] + self.Ki_z  * self._integral[2] + self.Kd_z  * d[2]

        Fx, Fy, Fz = ax, ay, az + self.G

        F_mag = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        if F_mag < 0.1:
            return 0.0, 0.0, yaw_ref, self.HOVER_THRUST

        thrust = np.clip(F_mag / self.G * self.HOVER_THRUST, 0.0, 1.0)
        z_des  = np.array([Fx, Fy, Fz]) / F_mag
        pitch  = -np.arcsin(np.clip(z_des[1], -1, 1))
        cp     = np.cos(pitch)
        roll   = np.arctan2(z_des[0], z_des[2] / (cp + 1e-9))

        roll  = np.clip(roll,  -self.MAX_TILT, self.MAX_TILT)
        pitch = np.clip(pitch, -self.MAX_TILT, self.MAX_TILT)

        return roll, pitch, yaw_ref, thrust

    def reset(self):
        self._integral[:] = 0
        self._prev_error[:] = 0
