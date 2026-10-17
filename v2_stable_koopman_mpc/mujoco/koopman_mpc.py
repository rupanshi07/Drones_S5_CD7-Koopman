"""Koopman-MPC for the MuJoCo CF2X (re-implements the repo's Phase-2 pipeline, trained on MuJoCo data).

 * lifting     : psi(x) = [x(18), v*Rzz(3), vz*Rxz, vz*Ryz, w*Rzz(3), 1]   (27-dim, rotation-matrix based)
 * model       : psi_{k+1} = A psi_k + B u_k,  u = (rpm - hover)/hover       (EDMD, ridge least squares)
 * controller  : receding-horizon QP (Hp=10, Hc=5), CLARABEL via cvxpy, anchored to the last applied input

Train once (about a minute or two):   python koopman_mpc.py     -> writes koopman_mujoco_model.npz
"""
import os
import sys
import time
import numpy as np
import cvxpy as cp
from mujoco_quad_env import MujocoQuadEnv
from controllers import make_pid

BASE_MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "koopman_model.npz")
MODEL_FILE = BASE_MODEL_FILE if os.path.exists(BASE_MODEL_FILE) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "koopman_mujoco_model.npz")
NPSI, NU = 27, 4


# ------------------------------------------------------------------ lifting
def lift(x):
    x = np.asarray(x)
    v, R, w = x[3:6], x[6:15], x[15:18]
    rzz, rxz, ryz = R[8], R[2], R[5]
    return np.concatenate([x, v * rzz, [v[2] * rxz, v[2] * ryz], w * rzz, [1.0]])


# ------------------------------------------------------------------ data collection
def collect(n_episodes=200, T=12.0, seed=0, ou_sigma=0.03, verbose=True):
    """PID tracks a random sinusoid; an Ornstein-Uhlenbeck signal on the RPMs gives independent excitation.
    Random payload + wind per episode."""
    rng = np.random.default_rng(seed)
    env = MujocoQuadEnv(flat=True)
    pid = make_pid(env)
    H = env.HOVER_RPM
    X0, X1, U = [], [], []
    kept = 0
    for ep in range(n_episodes):
        ax, ay = rng.uniform(0.3, 1.0, 2)
        wx, wy, wz = rng.uniform(0.05, 0.4, 3)
        px, py, pz = rng.uniform(0, 2 * np.pi, 3)
        z0 = rng.uniform(0.8, 2.2)
        pay, t_pay = rng.uniform(0, 0.6), rng.uniform(1, 6)
        wnd, t_w, d_w = rng.uniform(0, 0.10), rng.uniform(1, 6), rng.uniform(1, 3)
        wdir = rng.uniform(0, 2 * np.pi)
        ref0 = np.array([ax * np.sin(px), ay * np.sin(py), z0 + 0.15 * np.sin(pz)])
        s = env.reset(ref0)
        pid.reset()
        ou = np.zeros(4)
        ep_x0, ep_x1, ep_u, ok = [], [], [], True
        for k in range(int(T / env.dt)):
            t = k * env.dt
            ref = np.array([ax * np.sin(wx * t + px), ay * np.sin(wy * t + py), z0 + 0.15 * np.sin(wz * t + pz)])
            if abs(t - t_pay) < env.dt / 2:
                env.set_payload(pay)
            on = t_w <= t < t_w + d_w
            env.set_wind(wnd * np.cos(wdir) if on else 0.0, wnd * np.sin(wdir) if on else 0.0)
            ou += -0.2 * ou + ou_sigma * rng.standard_normal(4)
            rpm = np.clip(pid.rpm(s, ref) + H * ou, 0.0, env.MAX_RPM)
            s1 = env.step(rpm)
            ep_x0.append(lift(s)); ep_x1.append(lift(s1)); ep_u.append((rpm - H) / H)
            if s1[14] < 0.85 or np.linalg.norm(s1[:3] - ref) > 1.5:     # crashed / diverged -> drop episode
                ok = False; break
            s = s1
        if ok:
            X0 += ep_x0; X1 += ep_x1; U += ep_u; kept += 1
        if verbose and (ep + 1) % 50 == 0:
            print(f"  collected {ep+1}/{n_episodes} episodes ({kept} kept, {len(U)} samples)")
    env.set_wind(0, 0)
    return np.array(X0), np.array(X1), np.array(U)


# ------------------------------------------------------------------ EDMD fit
def fit(X0, X1, U, lam=1e-6):
    Z = np.hstack([X0, U])
    sd = Z.std(axis=0)
    sd[sd < 1e-9] = 1.0
    sd[NPSI - 1] = 1.0                                   # constant "1" column: leave as is
    Zs = Z / sd
    G = Zs.T @ Zs + lam * len(Zs) * np.eye(Z.shape[1])
    theta = np.linalg.solve(G, Zs.T @ X1) / sd[:, None]  # (27+4) x 27
    A, B = theta[:NPSI].T, theta[NPSI:].T
    pred = X0 @ A.T + U @ B.T
    rmse = np.sqrt(((pred - X1) ** 2).mean(axis=0))
    return A, B, rmse


def train_and_save(n_episodes=200):
    t0 = time.time()
    print("collecting data ...")
    X0, X1, U = collect(n_episodes)
    A, B, rmse = fit(X0, X1, U)
    np.savez(MODEL_FILE, A=A, B=B)
    print(f"fit on {len(U)} samples in {time.time()-t0:.0f}s | 1-step RMSE  pos {rmse[:3].max():.2e}  vel {rmse[3:6].max():.2e}  "
          f"R {rmse[6:15].max():.2e}  omega {rmse[15:18].max():.2e}")
    print("saved", MODEL_FILE)
    return A, B


def check_signs(env, A, B):
    """Direct sign test (like the repo's diagnose_sign.py): known motor differential -> model vs simulator."""
    H = env.HOVER_RPM
    print("sign check  (velocity after 6 control steps, model vs MuJoCo)")
    for name, d in [("roll  (motors 2,3 up)", [0, 0, .04, .04]), ("pitch (motors 1,2 up)", [0, .04, .04, 0])]:
        s = env.reset((0, 0, 1.5)); psi = lift(s)
        for _ in range(6):
            s = env.step(H * (1 + np.array(d)))
            psi = A @ psi + B @ np.array(d)
        print(f"  {name}:  model vx,vy = {psi[3]:+.3f},{psi[4]:+.3f}   sim vx,vy = {s[3]:+.3f},{s[4]:+.3f}")


# ------------------------------------------------------------------ MPC
class KoopmanMPC:
    name = "Koopman-MPC"

    def __init__(self, env, model_file=MODEL_FILE, Hp=10, Hc=5,
                 pos_weight=(700.0, 700.0, 800.0), vel_weight=(20.0, 20.0, 50.0),
                 tilt_weight=80.0, rate_weight=3.0, du_weight=3e-6, u_weight=1e-6):
        m = np.load(model_file)
        self.A = m["A"]
        self.B = m["B"].copy()
        self.env = env
        self.Hp, self.Hc = Hp, Hc
        self.n_lifted = self.A.shape[0]
        self.n_input = self.B.shape[1]
        self.hover_rpm = env.HOVER_RPM
        self.max_rpm = env.MAX_RPM

        # State tracking cost weighting
        Q_diag = np.zeros(self.n_lifted)
        Q_diag[0:3] = pos_weight
        Q_diag[3:6] = vel_weight
        # Tilt penalty on off-diagonal rotation elements R[0,2], R[1,2], R[2,0], R[2,1]
        Q_diag[[8, 11, 12, 13]] = tilt_weight
        # Angular velocity damping on wx, wy, wz
        Q_diag[15:18] = rate_weight
        self.Q = np.diag(Q_diag)
        self.Em = du_weight * np.eye(self.n_input)
        self.Rm = u_weight * np.eye(self.n_input)
        self.reset()

    def reset(self):
        self.prev_u = None
        self.int_ez = 0.0
        self.prev_ref = None

    def rpm(self, state, ref):
        """Map (state_18, ref) -> 4 motor RPM commands.
        ref can be a 3D target position [x, y, z] or an (Hp, 3) trajectory window."""
        psi0 = lift(state)
        dt = self.env.dt
        ref_arr = np.asarray(ref, dtype=float)

        if ref_arr.ndim == 1:
            ref_pos = ref_arr[:3]
            if self.prev_ref is not None:
                v_ref = np.clip((ref_pos - self.prev_ref) / dt, -3.0, 3.0)
            else:
                v_ref = np.zeros(3)
            self.prev_ref = ref_pos.copy()

            # Extrapolate future reference along horizon, terrain-aware if env supports ground_height
            ref_pos_seq = np.zeros((self.Hp, 3))
            ref_vel_seq = np.tile(v_ref, (self.Hp, 1))
            cur_gh = self.env.ground_height(ref_pos[0], ref_pos[1]) if hasattr(self.env, "ground_height") else 0.0
            cur_agl = max(ref_pos[2] - cur_gh, 0.5)

            for t_idx in range(self.Hp):
                px = ref_pos[0] + (t_idx + 1) * dt * v_ref[0]
                py = ref_pos[1] + (t_idx + 1) * dt * v_ref[1]
                if hasattr(self.env, "ground_height"):
                    gh_ahead = self.env.ground_height(px, py)
                    pz = max(ref_pos[2] + (t_idx + 1) * dt * v_ref[2], gh_ahead + cur_agl)
                else:
                    pz = ref_pos[2] + (t_idx + 1) * dt * v_ref[2]
                ref_pos_seq[t_idx] = [px, py, pz]
        else:
            ref_pos = ref_arr[0, :3]
            ref_pos_seq = ref_arr[:self.Hp, :3]
            if ref_arr.shape[1] >= 6:
                ref_vel_seq = ref_arr[:self.Hp, 3:6]
            else:
                ref_vel_seq = np.zeros((self.Hp, 3))
            self.prev_ref = ref_pos.copy()

        # Altitude disturbance rejection (payload & vertical draft) via adaptive collective baseline
        ez = ref_pos[2] - state[2]
        vz = state[5]
        v_ref_z = float(ref_vel_seq[0, 2]) if ref_vel_seq.shape[1] >= 3 else 0.0
        evz = v_ref_z - vz
        self.int_ez = np.clip(self.int_ez + ez * dt, -0.6, 0.6)

        az_cmd = 9.8 + 12.0 * ez + 5.0 * evz + 4.0 * self.int_ez
        thrust_cmd = self.env.MASS * max(az_cmd, 2.0)
        self.hover_rpm = float(np.sqrt(thrust_cmd / (4 * self.env.KF)))

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
            constraints += [du[t] >= -6000.0, du[t] <= (self.max_rpm - self.hover_rpm)]
            if t >= self.Hc:
                constraints += [du[t] == du[self.Hc - 1]]

            target = np.zeros(self.n_lifted)
            target[0:3] = ref_pos_seq[t]
            target[3:6] = ref_vel_seq[t]
            # Desired upright orientation (R = Identity) and constant bias 1.0
            target[6] = 1.0
            target[10] = 1.0
            target[14] = 1.0
            target[26] = 1.0

            err = psi[t + 1] - target
            cost += cp.quad_form(err, self.Q)
            cost += cp.quad_form(du[t], self.Rm)
            if t > 0:
                cost += cp.quad_form(du[t] - du[t - 1], self.Em)
            else:
                cost += cp.quad_form(du[0] - prev_du, self.Em)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            pass

        if problem.status not in ("optimal", "optimal_inaccurate") or du.value is None:
            u_cmd = self.prev_u if self.prev_u is not None else np.full(self.n_input, self.hover_rpm)
        else:
            u_cmd = self.hover_rpm + du.value[0]

        self.prev_u = u_cmd
        return u_cmd


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    A, B = train_and_save(n)
    check_signs(MujocoQuadEnv(flat=True), A, B)