"""Test the project's authentic base Koopman MPC model in MuJoCo."""
import os
import sys
import numpy as np

# Load base model from v2_stable_koopman_mpc/data/koopman_model.npz
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "data", "koopman_model.npz")
model_data = np.load(model_path)
A, B = model_data["A"], model_data["B"]
print(f"Loaded authentic Koopman model from {model_path}")
print(f"  A shape: {A.shape}, B shape: {B.shape}")

# Import base lifting function and base KoopmanMPC
sys.path.insert(0, os.path.join(base_dir, "koopman_model"))
from build_edmd_dataset import lift as base_lift

sys.path.insert(0, os.path.join(base_dir, "mpc"))
import cvxpy as cp
from mujoco_quad_env import MujocoQuadEnv

class AuthenticKoopmanController:
    def __init__(self, env, Hp=10, Hc=5, pos_weight=150.0, vel_weight=5.0, du_weight=1.0, u_weight=1e-3):
        self.env = env
        self.A = A
        self.B = B
        self.Hp = Hp
        self.Hc = Hc
        self.n_lifted = A.shape[0]
        self.n_input = B.shape[1]
        self.hover_rpm = env.HOVER_RPM
        self.max_rpm = env.MAX_RPM

        # State cost matrix
        Q_diag = np.zeros(self.n_lifted)
        Q_diag[0:3] = pos_weight
        Q_diag[3:6] = vel_weight
        self.Q = np.diag(Q_diag)
        self.Em = du_weight * np.eye(self.n_input)
        self.Rm = u_weight * np.eye(self.n_input)
        self.reset()

    def reset(self):
        self.prev_u = None

    def rpm(self, state, ref):
        """Map (state_18, ref) -> 4 RPM commands."""
        # state is 18-dim: pos(3), vel(3), rotmat(9), ang_vel(3)
        raw_state = np.asarray(state).reshape(1, -1)
        psi0 = base_lift(raw_state)[0]

        # Build reference trajectory across prediction horizon Hp
        ref = np.asarray(ref, dtype=float)
        ref_traj = np.tile(ref, (self.Hp, 1))

        if self.prev_u is None:
            prev_du = np.zeros(self.n_input)
        else:
            prev_du = np.asarray(self.prev_u) - self.hover_rpm

        du = cp.Variable((self.Hp, self.n_input))
        psi = cp.Variable((self.Hp + 1, self.n_lifted))

        constraints = [psi[0] == psi0]
        cost = 0

        for t in range(self.Hp):
            u_t = self.hover_rpm + du[t]
            constraints += [psi[t + 1] == self.A @ psi[t] + self.B @ u_t]
            constraints += [du[t] >= -self.hover_rpm, du[t] <= self.max_rpm - self.hover_rpm]
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
                cost += cp.quad_form(du[0] - prev_du, self.Em)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.CLARABEL)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            u_cmd = np.full(self.n_input, self.hover_rpm)
        else:
            u_cmd = self.hover_rpm + du.value[0]

        self.prev_u = u_cmd
        return u_cmd

if __name__ == "__main__":
    env = MujocoQuadEnv(flat=True)
    ctrl = AuthenticKoopmanController(env)
    r0 = np.array([0.0, 0.0, 1.0])
    s = env.reset(r0)
    print("Starting test flight in MuJoCo...")
    errs = []
    for step in range(int(6.0 / env.dt)):
        t = step * env.dt
        # Simple figure-8 / smooth circular reference
        ref = np.array([0.4 * np.sin(0.3 * t), 0.4 * np.sin(0.6 * t), 1.0])
        cmd = ctrl.rpm(s, ref)
        s = env.step(cmd)
        err = np.linalg.norm(s[:3] - ref)
        errs.append(err)
        if step % 48 == 0:
            print(f"t={t:.2f}s  pos={np.round(s[:3], 3)}  ref={np.round(ref, 3)}  rpm={np.round(cmd, 0)}  err={err:.3f}m")
        if env.touching_terrain() or env.flipped():
            print(f"FAILED at t={t:.2f}s")
            break
    print(f"Flight complete! Mean tracking error = {np.mean(errs):.3f} m, Max = {np.max(errs):.3f} m")
