"""MuJoCo CF2X over terrain with every controller from the project: PID, LQR, Koopman-MPC.

  python run_sim.py --gui                      # Koopman-MPC in the viewer
  python run_sim.py --ctrl all --gui           # PID, then LQR, then Koopman-MPC, one after another
  python run_sim.py --ctrl all                 # headless, prints a table and saves comparison.png
  python run_sim.py --ctrl all --payload 0.25 --wind 0.02     # +25 % mass at t=3 s, wind gust 3-5 s
Options: --ctrl pid|lqr|koopman|all  --flat  --agl 1.5  --amp 2.5  --T 12  --seed 0  --retrain  --no-plot
"""
import sys
import time
import numpy as np
from mujoco_quad_env import MujocoQuadEnv
from controllers import make_pid, make_lqr


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


GUI, FLAT, RETRAIN, NOPLOT = "--gui" in sys.argv, "--flat" in sys.argv, "--retrain" in sys.argv, "--no-plot" in sys.argv
CTRL = arg("--ctrl", "koopman", str).lower()
TERRAIN = "flat" if FLAT else arg("--terrain", "rolling", str).lower()
AGL, AMP, T_END, SEED = arg("--agl", 1.5), arg("--amp", 2.0), arg("--T", 12.0), arg("--seed", 0, int)
PAYLOAD, WIND = arg("--payload", 0.0), arg("--wind", 0.0)
PARAMS = (AMP, AMP, 0.30, 0.25, 0.0, 1.0)                  # ax, ay, wx, wy, phase_x, phase_y
COLORS = {"PID": (1.0, 0.55, 0.0, 0.35), "LQR": (0.6, 0.25, 1.0, 0.35), "Koopman-MPC": (1.0, 0.1, 0.1, 0.35)}


def reference(t, z_prev, env):
    """Sinusoidal x/y path; altitude = terrain height + AGL, rate-limited."""
    ax, ay, wx, wy, px, py = PARAMS
    x, y = ax * np.sin(wx * t + px), ay * np.sin(wy * t + py)
    z = z_prev + np.clip(env.ground_height(x, y) + AGL - z_prev, -0.45 * env.dt, 0.45 * env.dt)
    return np.array([x, y, z]), z


def make_controller(name, env):
    if name == "pid":
        return make_pid(env)
    if name == "lqr":
        return make_lqr(env)
    from koopman_mpc import KoopmanMPC, train_and_save
    if RETRAIN:
        train_and_save()
    return KoopmanMPC(env)


def add_segment(scn, p, q, width, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p, q)
    scn.ngeom += 1


def run(env, ctrl, label):
    import mujoco
    ax, ay, _, _, px, py = PARAMS
    x0, y0 = ax * np.sin(px), ay * np.sin(py)
    r0 = np.array([x0, y0, env.ground_height(x0, y0) + AGL])
    s = env.reset(r0)
    ctrl.reset()
    env.set_ref_marker(r0)
    halo = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "halo")
    if halo >= 0:
        env.model.geom_rgba[halo] = COLORS[label]
    z_ref, log, status = r0[2], [], "ok"

    viewer = None
    if GUI:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.body_id
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.65, -20, 135
        viewer.user_scn.ngeom = 0
        print(f"[{label}] viewer open (close the window to skip)")
    prev_p, prev_r = s[:3].copy(), r0.copy()

    for k in range(int(T_END / env.dt)):
        t0, t = time.time(), k * env.dt
        ref, z_ref = reference(t, z_ref, env)
        if PAYLOAD and abs(t - 3.0) < env.dt / 2:
            env.set_payload(PAYLOAD)
        if WIND:
            env.set_wind(WIND if 3.0 <= t < 5.0 else 0.0, 0.0)
        env.set_ref_marker(ref)
        rpm = ctrl.rpm(s, ref)
        s = env.step(rpm)
        log.append(np.r_[t, s[:3], ref, env.agl(), env.ground_height(s[0], s[1])])
        if env.touching_terrain() or env.flipped():
            status = f"FAILED t={t:.2f}s (terrain contact or flip)"
        if viewer:
            if k % 3 == 0:
                add_segment(viewer.user_scn, prev_p, s[:3], 0.012, COLORS[label][:3] + (1.0,))
                add_segment(viewer.user_scn, prev_r, ref, 0.006, (0.1, 0.9, 0.2, 0.8))
                prev_p, prev_r = s[:3].copy(), ref.copy()
            viewer.sync()
            if not viewer.is_running():
                status = "window closed"
                break
            time.sleep(max(0.0, env.dt - (time.time() - t0)))
        if status.startswith("FAILED"):
            break
    if viewer:
        if status == "ok":
            time.sleep(1.0)
        viewer.close()
    env.set_wind(0, 0)
    return np.array(log), status


def summarize(label, log, status):
    err = np.linalg.norm(log[:, 1:4] - log[:, 4:7], axis=1)
    return dict(label=label, status=status, steps=len(log), rmse=float(np.sqrt((err ** 2).mean())),
                maxerr=float(err.max()), minagl=float(log[:, 7].min()), err=err)


def main():
    names = ["pid", "lqr", "koopman"] if CTRL == "all" else [CTRL]
    labels = {"pid": "PID", "lqr": "LQR", "koopman": "Koopman-MPC"}
    env = MujocoQuadEnv(terrain_seed=SEED, terrain=TERRAIN)
    print(f"terrain: {TERRAIN} (seed {SEED}) | AGL {AGL} m | path amplitude {AMP} m | "
          f"payload {PAYLOAD:+.0%} @3s | wind {WIND} N @3-5s")
    results, logs = [], {}
    for n in names:
        ctrl = make_controller(n, env)
        log, status = run(env, ctrl, labels[n])
        logs[labels[n]] = log
        r = summarize(labels[n], log, status)
        results.append(r)
        np.savez(f"mujoco_run_{n}.npz", log=log)
        if status == "window closed" and len(names) > 1:
            print("window closed -> stopping"); break
    print(f"\n{'controller':<13}{'status':<10}{'RMSE (m)':>10}{'max err (m)':>13}{'min AGL (m)':>13}")
    for r in results:
        print(f"{r['label']:<13}{r['status'][:9]:<10}{r['rmse']:>10.3f}{r['maxerr']:>13.3f}{r['minagl']:>13.3f}")
    if not NOPLOT:
        plot(logs, results)


def plot(logs, results):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    any_log = next(iter(logs.values()))
    ax[0, 0].plot(any_log[:, 4], any_log[:, 5], "g--", lw=2, label="reference")
    for lab, lg in logs.items():
        c = COLORS[lab][:3]
        ax[0, 0].plot(lg[:, 1], lg[:, 2], color=c, label=lab)
        ax[0, 1].plot(lg[:, 0], lg[:, 3], color=c, label=lab)
        ax[1, 0].plot(lg[:, 0], np.linalg.norm(lg[:, 1:4] - lg[:, 4:7], axis=1), color=c, label=lab)
        ax[1, 1].plot(lg[:, 0], lg[:, 7], color=c, label=lab)
    ax[0, 1].plot(any_log[:, 0], any_log[:, 6], "g--", lw=2, label="z reference")
    ax[0, 1].plot(any_log[:, 0], any_log[:, 8], color="saddlebrown", lw=1.5, label="ground under drone")
    for a, t, xl, yl in [(ax[0, 0], "Top-down path", "x (m)", "y (m)"), (ax[0, 1], "Altitude follows the terrain", "t (s)", "z (m)"),
                         (ax[1, 0], "3-D tracking error", "t (s)", "m"), (ax[1, 1], "Height above ground (AGL)", "t (s)", "m")]:
        a.set_title(t); a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[0, 0].set_aspect("equal")
    fig.suptitle("CF2X in MuJoCo over terrain: " + ", ".join(f"{r['label']} RMSE {r['rmse']:.3f} m" for r in results))
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=130)
    print("saved comparison.png")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()