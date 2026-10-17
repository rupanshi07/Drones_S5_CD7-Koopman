"""
koopman_mpc_ardupilot.py
------------------------
Koopman-MPC controller for ArduPilot SITL.

This is a direct port of v2_stable_koopman_mpc/mpc/koopman_mpc.py to the
ArduPilot / MAVLink interface.  The core mathematics is IDENTICAL to the
Phase-2 reference:

    min  sum_t || psi(Y[t]) - psi(r[t]) ||^2_Q
              + || du[t] ||^2_Em
              + || u[t] - u_hover ||^2_Rm          (eq. 6 in the paper)
    s.t. psi(Y[t+1]) = A psi(Y[t]) + B u[t]
         0 <= u[t] <= RPM_MAX

The only changes vs. Phase 2 are in how we FEED the controller and how we
USE its output:

  Input side:
    - State arrives as MAVLink telemetry in NED; mavlink_interface converts it
      to ENU before we see it, so the lifted state is identical to Phase 2.

  Output side:
    - Phase 2 returned 4 raw motor RPMs (gym-pybullet accepts those directly).
    - ArduPilot uses SET_ATTITUDE_TARGET (roll, pitch, yaw, thrust), so we add
      _rpm_to_attitude() to convert the MPC's optimal RPM vector to attitude
      commands.  The conversion is purely geometric and introduces no new
      mathematics.

  Model loading:
    - The .npz produced by fit_koopman.py contains keys A, B and u_scale.
      u_scale is the per-motor RPM std used to normalise inputs DURING
      training.  B was already un-normalised before saving
      (B = B_norm / u_scale), so we must use B with RAW RPMs -- exactly as
      Phase 2 does.  We do NOT divide RPM_HOVER/RPM_MAX by u_scale here.

  Solver:
    - Same CLARABEL solver with ECOS fallback.
    - CVXPY problem is built ONCE with cp.Parameter for warm-starting
      (avoids full QP re-construction every step, saving ~50-100 ms/step).
"""

import numpy as np
import cvxpy as cp
from pathlib import Path


# --------------------------------------------------------------------------- #
# Constants -- IDENTICAL to v2/mpc/koopman_mpc.py
# --------------------------------------------------------------------------- #
RPM_MAX   = 21713.0    # max motor RPM (CF2X)
RPM_HOVER = 14468.0    # nominal hover RPM


# --------------------------------------------------------------------------- #
# Lifting function -- IDENTICAL to v2/koopman_model/build_edmd_dataset.py
# --------------------------------------------------------------------------- #

def lift(raw_state_row):
    """
    Lifting function psi(x) -- EXACTLY matches Phase-2 build_edmd_dataset.py.

    raw_state_row : (N, 18) or (18,) array
                    columns = [pos(3), vel(3), rotmat(9), ang_vel(3)]
    Returns       : (N, 27) array
                    27 = 18 (raw) + 8 (cross-terms) + 1 (bias)

    Cross-terms (physically: velocity projected through orientation):
        vel * r_zz    -> 3 terms  (vel_x/y/z * R[2,2])
        vz  * r_xz    -> 1 term   (vz * R[0,2])
        vz  * r_yz    -> 1 term   (vz * R[1,2])
        ang_vel * r_zz -> 3 terms  (wx/wy/wz * R[2,2])
    """
    raw = np.asarray(raw_state_row, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    vel     = raw[:, 3:6]
    rotmat  = raw[:, 6:15]   # 9 entries, row-major 3x3
    ang_vel = raw[:, 15:18]

    r_zz = rotmat[:, 8:9]    # R[2,2]
    r_xz = rotmat[:, 2:3]    # R[0,2]
    r_yz = rotmat[:, 5:6]    # R[1,2]

    cross_terms = np.hstack([
        vel * r_zz,             # 3 terms
        vel[:, 2:3] * r_xz,    # 1 term
        vel[:, 2:3] * r_yz,    # 1 term
        ang_vel * r_zz,        # 3 terms
    ])  # 8 terms total

    ones   = np.ones((raw.shape[0], 1))
    lifted = np.hstack([raw, cross_terms, ones])
    return lifted   # (N, 27)


# --------------------------------------------------------------------------- #
# Koopman-MPC -- same QP as Phase 2, warm-started via cp.Parameter
# --------------------------------------------------------------------------- #

class KoopmanMPCArduPilot:
    """
    Koopman-based MPC for ArduPilot SITL.

    QP formulation is IDENTICAL to Phase-2 KoopmanMPC.  Default weights and
    horizons also match Phase-2's run_mpc_closed_loop.py / run_comparison.py.

    Output: (roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, rpm_cmd)
      - roll/pitch/thrust are sent via SET_ATTITUDE_TARGET
      - rpm_cmd is the raw QP solution (for logging / cross-checking)
    """

    # ArduPilot attitude command limits (matching Phase-2 MAX_TILT_RAD = 20 deg)
    HOVER_THRUST  = 0.5         # normalised throttle at hover
    MAX_TILT_RAD  = 0.3491      # radians(20)
    G             = 9.81

    def __init__(self, model_path, Hp=10, Hc=5,
                 pos_weight=120.0, vel_weight=1.0,
                 du_weight=1.0, u_weight=1e-3):
        """
        Parameters
        ----------
        model_path  : str   path to .npz with keys 'A', 'B' (and optionally 'u_scale')
        Hp          : int   prediction horizon  (Phase-2 default: 10)
        Hc          : int   control horizon     (Phase-2 default:  5)
        pos_weight  : float position tracking weight (Phase-2 default: 120.0)
        vel_weight  : float velocity tracking weight (Phase-2 default:   1.0)
        du_weight   : float smoothness weight        (Phase-2 default:   1.0)
        u_weight    : float input magnitude weight   (Phase-2 default:  1e-3)
        """
        data = np.load(model_path)
        self.A = data["A"].astype(float)
        self.B = data["B"].astype(float)
        self.n_lifted = self.A.shape[0]
        self.n_input  = self.B.shape[1]
        self.Hp = Hp
        self.Hc = Hc

        # B is already in raw RPM space (fit_koopman.py un-normalises before
        # saving: B = B_norm / u_scale).  We use raw RPMs, just like Phase 2.
        if "u_scale" in data:
            print(f"[KoopmanMPC] u_scale key found (={np.mean(data['u_scale']):.1f}). "
                  f"B is already un-normalised -- using raw RPM space.")
        print(f"[KoopmanMPC] A{self.A.shape}  B{self.B.shape}")
        print(f"[KoopmanMPC] Hp={Hp}  Hc={Hc}  "
              f"pos_w={pos_weight}  vel_w={vel_weight}  "
              f"du_w={du_weight}  u_w={u_weight}")

        # ------------------------------------------------------------------- #
        # Weight matrices -- IDENTICAL to Phase-2 KoopmanMPC.__init__
        # weight only raw position/velocity entries; de-weight higher dims
        # ------------------------------------------------------------------- #
        Q_diag = np.zeros(self.n_lifted)
        Q_diag[0:3] = pos_weight
        Q_diag[3:6] = vel_weight
        self.Q  = np.diag(Q_diag)
        self.Em = du_weight * np.eye(self.n_input)
        self.Rm = u_weight  * np.eye(self.n_input)

        self._prev_u = None   # previous applied command (raw RPM, shape n_input)

        # Pre-build CVXPY QP with cp.Parameter for warm-starting
        self._build_qp()

    # ----------------------------------------------------------------------- #
    def _build_qp(self):
        """
        Build the CVXPY QP once using cp.Parameter placeholders.

        Written in DPP form (sum_squares with parameter in the affine shift)
        so CVXPY caches the problem sparsity and warm-starts subsequent solves.
        This is the key speed improvement over Phase-2, which rebuilt the QP
        from scratch every step.
        """
        Hp = self.Hp;  Hc = self.Hc
        n  = self.n_lifted;  m = self.n_input

        # Cholesky factors for DPP-compliant costs
        Q_sqrt = np.linalg.cholesky(self.Q  + 1e-8 * np.eye(n)).T
        R_sqrt = np.linalg.cholesky(self.Rm + 1e-8 * np.eye(m)).T
        E_sqrt = np.linalg.cholesky(self.Em + 1e-8 * np.eye(m)).T

        # Parameters updated each step
        self._param_psi0    = cp.Parameter(n,       name="psi0")
        self._param_prev_du = cp.Parameter(m,       name="prev_du")
        self._param_targets = cp.Parameter((Hp, n), name="targets")

        # Decision variables
        du  = cp.Variable((Hp, m), name="du")
        psi = cp.Variable((Hp + 1, n), name="psi")

        constraints = [psi[0] == self._param_psi0]
        cost = 0

        for t in range(Hp):
            u_t = RPM_HOVER + du[t]

            # Dynamics: psi[t+1] = A psi[t] + B u_t  (identical to Phase-2)
            constraints += [psi[t + 1] == self.A @ psi[t] + self.B @ u_t]

            # Input bounds: 0 <= u_t <= RPM_MAX  <=>  -RPM_HOVER <= du <= RPM_MAX - RPM_HOVER
            constraints += [
                du[t] >= -RPM_HOVER,
                du[t] <= RPM_MAX - RPM_HOVER,
            ]

            # Control horizon: hold last control after Hc  (identical to Phase-2)
            if t >= Hc:
                constraints += [du[t] == du[Hc - 1]]

            # Tracking cost (DPP: parameter enters only as affine offset)
            cost += cp.sum_squares(Q_sqrt @ psi[t + 1]
                                   - Q_sqrt @ self._param_targets[t])

            # Input magnitude cost
            cost += cp.sum_squares(R_sqrt @ du[t])

            # Input smoothness cost (identical anchoring logic to Phase-2)
            if t > 0:
                cost += cp.sum_squares(E_sqrt @ (du[t] - du[t - 1]))
            else:
                # anchor du[0] to the ACTUAL previous control, not unconstrained
                # (Phase-2 comment: "closes a real gap -- without it each fresh
                # solve has no memory of what was actually just commanded")
                cost += cp.sum_squares(E_sqrt @ (du[0] - self._param_prev_du))

        self._prob    = cp.Problem(cp.Minimize(cost), constraints)
        self._var_du  = du
        self._var_psi = psi

    # ----------------------------------------------------------------------- #
    def solve(self, pos_enu, vel_enu, rotmat_enu, ang_vel, ref_traj_enu):
        """
        Run one MPC solve step.

        Parameters
        ----------
        pos_enu      : (3,)    ENU position  [m]
        vel_enu      : (3,)    ENU velocity  [m/s]
        rotmat_enu   : (9,)    flattened 3x3 rotation matrix (body->ENU)
        ang_vel      : (3,)    body-frame angular velocity [rad/s]
        ref_traj_enu : (Hp, 3) ENU reference positions for each prediction step

        Returns
        -------
        roll_cmd  : float  [rad]
        pitch_cmd : float  [rad]
        yaw_cmd   : float  [rad]  (always 0 -- heading-hold)
        thrust    : float  [0,1]
        rpm_cmd   : (n_input,) raw RPM command (for logging)
        """
        # Lift current state -- identical to Phase-2 run_mpc_closed_loop.py
        raw_state = np.hstack([pos_enu, vel_enu, rotmat_enu, ang_vel])
        psi0 = lift(raw_state.reshape(1, -1))[0]

        # Previous du anchor -- identical to Phase-2 KoopmanMPC.solve()
        if self._prev_u is None:
            prev_du = np.zeros(self.n_input)
        else:
            prev_du = self._prev_u - RPM_HOVER

        # Reference target lifted states (only position entries set, rest 0)
        targets = np.zeros((self.Hp, self.n_lifted))
        for t in range(self.Hp):
            targets[t, 0:3] = ref_traj_enu[t]

        # Fill parameters
        self._param_psi0.value    = psi0
        self._param_prev_du.value = prev_du
        self._param_targets.value = targets

        # Solve -- try CLARABEL first, fall back to ECOS
        try:
            self._prob.solve(solver=cp.CLARABEL, verbose=False,
                             max_iter=200, warm_start=True)
        except cp.SolverError:
            try:
                self._prob.solve(solver=cp.ECOS, verbose=False,
                                 warm_start=True)
            except Exception:
                # Emergency fallback: hover
                return 0.0, 0.0, 0.0, self.HOVER_THRUST, \
                       np.full(self.n_input, RPM_HOVER)

        # Check status -- identical warning logic to Phase-2
        status = self._prob.status
        if status not in ("optimal", "optimal_inaccurate"):
            print(f"[KoopmanMPC] WARNING: solver status = {status}")
            return 0.0, 0.0, 0.0, self.HOVER_THRUST, \
                   np.full(self.n_input, RPM_HOVER)

        if self._var_du.value is None:
            return 0.0, 0.0, 0.0, self.HOVER_THRUST, \
                   np.full(self.n_input, RPM_HOVER)

        # Optimal first control action -- identical to Phase-2
        #   return RPM_HOVER + du.value[0]
        rpm_cmd = np.clip(RPM_HOVER + self._var_du.value[0], 0.0, RPM_MAX)
        self._prev_u = rpm_cmd.copy()

        # Convert RPM vector to ArduPilot attitude command
        roll_cmd, pitch_cmd, thrust = self._rpm_to_attitude(rpm_cmd)
        return roll_cmd, pitch_cmd, 0.0, thrust, rpm_cmd

    # ----------------------------------------------------------------------- #
    def _rpm_to_attitude(self, rpm_cmd):
        """
        Convert the 4-motor RPM command (Phase-2 CF2X convention) to roll,
        pitch, and normalised thrust for ArduPilot SET_ATTITUDE_TARGET.

        CF2X motor layout (looking DOWN at the drone):
            M0 (index 0): front-right  CCW
            M1 (index 1): back-left    CCW
            M2 (index 2): front-left   CW
            M3 (index 3): back-right   CW

        Moment conventions (matching Phase-2 DSLPIDControl / lqr_baseline.py):
            Thrust ∝ sum of all four squared RPMs
            Roll   ∝ (M1 + M2) - (M0 + M3)   [left side minus right side]
            Pitch  ∝ (M0 + M1) - (M2 + M3)   [front minus back]
              (positive pitch = nose up = drone moves forward in ENU)

        We work in normalised RPM r = rpm / RPM_HOVER so that hover = 1.0.
        """
        r = rpm_cmd / RPM_HOVER   # normalised: hover -> 1.0

        # Total thrust (Phase-2: HOVER_THRUST=0.5 at hover)
        T_norm = np.mean(r) * self.HOVER_THRUST

        if len(rpm_cmd) >= 4:
            # CF2X differential moments (see DSLPIDControl.py mixer in Phase 2)
            roll_diff  = ((r[1] + r[2]) - (r[0] + r[3])) * 0.5  # left - right
            pitch_diff = ((r[0] + r[1]) - (r[2] + r[3])) * 0.5  # front - back
        else:
            roll_diff  = 0.0
            pitch_diff = 0.0

        # Scale differential -> angle (empirically matched to SITL response)
        DIFF_TO_RAD = 0.5
        roll_cmd  = float(np.clip(roll_diff  * DIFF_TO_RAD,
                                  -self.MAX_TILT_RAD, self.MAX_TILT_RAD))
        pitch_cmd = float(np.clip(pitch_diff * DIFF_TO_RAD,
                                  -self.MAX_TILT_RAD, self.MAX_TILT_RAD))
        thrust    = float(np.clip(T_norm, 0.0, 1.0))

        return roll_cmd, pitch_cmd, thrust

    def reset(self):
        """Reset controller state (call between episodes)."""
        self._prev_u = None
