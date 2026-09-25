"""
Step 1-2: Formulate and test the Koopman-MPC controller as a QP, standalone
(no simulator yet) -- confirms the solver converges and produces sane RPM
commands before we close the loop in gym-pybullet-drones.

MPC problem (matches the structure of eq. 6 in the reference paper):
    min sum_t || Psi(Y[t]) - Psi(r[t]) ||^2_Qm + ||du[t]||^2_Em + ||u[t]||^2_Rm
    s.t. Psi(Y[t+1]) = A Psi(Y[t]) + B u[t]
         0 <= u[t] <= RPM_MAX
"""

import numpy as np
import cvxpy as cp

from build_edmd_dataset import lift

RPM_MAX = 21713.0   # from your logged data / CF2X max; adjust if your drone differs
RPM_HOVER = 14468.0  # nominal hover rpm, from your logs -- used as the linearization/reference input


class KoopmanMPC:
    def __init__(self, A, B, u_scale_unused=None, Hp=20, Hc=10,
                 pos_weight=50.0, vel_weight=1.0, du_weight=1.0, u_weight=1e-3):
        self.A = A
        self.B = B
        self.n_lifted = A.shape[0]
        self.n_input = B.shape[1]
        self.Hp = Hp
        self.Hc = Hc

        # weight only the raw position/velocity entries of the lifted state
        # (indices 0:3 = pos, 3:6 = vel in our lift() function); heavily
        # de-weighting the higher observable dimensions keeps the QP
        # focused on what we actually care about tracking
        Q_diag = np.zeros(self.n_lifted)
        Q_diag[0:3] = pos_weight
        Q_diag[3:6] = vel_weight
        self.Q = np.diag(Q_diag)
        self.Em = du_weight * np.eye(self.n_input)
        self.Rm = u_weight * np.eye(self.n_input)

    def solve(self, psi0, ref_traj, prev_u=None):
        """
        psi0: current lifted state, shape (n_lifted,)
        ref_traj: array of raw target positions, shape (Hp, 3) (we only
                  target position here; velocity target left at 0)
        prev_u: the RPM command actually applied on the previous REAL control
                step (not just the previous predicted step within this solve).
                Anchoring du[0] to this closes a real gap: without it, each
                fresh solve has no memory of what was actually just commanded
                a moment ago, so the real applied RPM can jump abruptly
                between consecutive control steps even though each solve's
                internal plan looks smooth in isolation. Defaults to hover.
        Returns: optimal RPM command for the next step, shape (n_input,)
        """
        if prev_u is None:
            prev_du = np.zeros(self.n_input)
        else:
            prev_du = np.asarray(prev_u) - RPM_HOVER

        du = cp.Variable((self.Hp, self.n_input))  # deviation from hover, well-scaled
        psi = cp.Variable((self.Hp + 1, self.n_lifted))

        constraints = [psi[0] == psi0]
        cost = 0

        for t in range(self.Hp):
            u_t = RPM_HOVER + du[t]
            constraints += [psi[t + 1] == self.A @ psi[t] + self.B @ u_t]
            constraints += [du[t] >= -RPM_HOVER, du[t] <= RPM_MAX - RPM_HOVER]
            if t >= self.Hc:
                constraints += [du[t] == du[self.Hc - 1]]

            target = np.zeros(self.n_lifted)
            target[0:3] = ref_traj[t]
            err = psi[t + 1] - target
            cost += cp.quad_form(err, self.Q)
            cost += cp.quad_form(du[t], self.Rm)
            if t > 0:
                cost += cp.quad_form(du[t] - du[t - 1], self.Em)
            else:
                # anchor the first predicted action to what was actually
                # applied last real control step, not left unconstrained
                cost += cp.quad_form(du[0] - prev_du, self.Em)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.CLARABEL)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            print(f"WARNING: solver status = {problem.status}")
            return np.full(self.n_input, RPM_HOVER)

        return RPM_HOVER + du.value[0]


if __name__ == "__main__":
    # --- standalone sanity test: no simulator, just check the QP behaves ---
    model = np.load("data/koopman_model.npz")
    A, B = model["A"], model["B"]

    # fake a hovering initial state: pos=(0,0,1), vel=0, rotmat=identity, ang_vel=0
    raw_state0 = np.zeros((1, 18))
    raw_state0[0, 0:3] = [0.0, 0.0, 1.0]           # pos
    raw_state0[0, 6:15] = np.eye(3).flatten()       # identity rotation
    psi0 = lift(raw_state0)[0]

    print("Sweeping position weight and target distance to check responsiveness:\n")
    for pos_weight in [50.0, 500.0, 5000.0]:
        for target_offset in [0.3, 1.0]:
            mpc = KoopmanMPC(A, B, Hp=20, Hc=10, pos_weight=pos_weight)
            ref_traj = np.tile([target_offset, 0.0, 1.0], (mpc.Hp, 1))
            rpm_cmd = mpc.solve(psi0, ref_traj)
            spread = rpm_cmd.max() - rpm_cmd.min()
            print(f"pos_weight={pos_weight:>7.0f}  target_x={target_offset:>4.1f}m  "
                  f"rpm={np.round(rpm_cmd, 1)}  spread={spread:.1f}")
    print()
    print("(Expect: spread should grow noticeably as pos_weight or target_offset")
    print(" increases. If spread stays tiny (<10 rpm) even at pos_weight=5000 and")
    print(" target=1.0m, something is likely wrong in the weighting/formulation.)")
