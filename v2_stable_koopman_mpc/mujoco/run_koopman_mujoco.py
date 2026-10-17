"""Closed-loop Koopman-MPC in MuJoCo over terrain.
Directly uses the project's authentic Phase-2 Koopman-MPC model (v2_stable_koopman_mpc/data/koopman_model.npz).

Usage:
  python run_koopman_mujoco.py                      # Headless run, default rolling terrain
  python run_koopman_mujoco.py --gui                # Live 3-D MuJoCo viewer (close-up drone tracking)
  python run_koopman_mujoco.py --terrain mountain   # Test on mountain peaks
  python run_koopman_mujoco.py --terrain canyon --gui
  python run_koopman_mujoco.py --terrain terraced --agl 1.2
  python run_koopman_mujoco.py --terrain flat       # Flat arena run

Options:
  --terrain flat|gentle|rolling|rugged|mountain|terraced|canyon|obstacles
  --agl <float>     Clearance above terrain in meters (default: 1.5)
  --amp <float>     X/Y reference path amplitude (default: 2.0)
  --T <float>       Episode duration in seconds (default: 12.0)
  --seed <int>      Terrain random seed (default: 0)
  --gui             Open passive 3D MuJoCo viewer
"""
import os
import sys
import time
import numpy as np
import cvxpy as cp
import mujoco

from mujoco_quad_env import MujocoQuadEnv
from terrain import PRESETS, list_terrains

# Determine repository root
MUJOCO_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(MUJOCO_DIR)
MODEL_FILE = os.path.join(V2_DIR, "data", "koopman_model.npz")

# CLI argument parser helper
def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

GUI = "--gui" in sys.argv
FLAT = "--flat" in sys.argv
TERRAIN = "flat" if FLAT else arg("--terrain", "rolling", str).lower()
AGL = arg("--agl", 1.5)
AMP = arg("--amp", 2.0)
T_END = arg("--T", 12.0)
SEED = arg("--seed", 0, int)
PAYLOAD = arg("--payload", 0.0)             # Extra mass fraction, e.g. 0.25 = +25%
WIND = arg("--wind", 0.0)                   # Wind force in N, e.g. 0.02
WIND_DIR = arg("--wind-dir", 0.0)           # Direction in degrees (0 = +x, 90 = +y)
PARAMS = (AMP, AMP, 0.30, 0.25, 0.0, 1.0)


# ======================================================================
# Authentic Phase-2 Lifting Function
# ======================================================================
def lift(x):
    """27-dimensional rotation-matrix-based lifting matching Phase-2 EDMD."""
    x = np.asarray(x, dtype=float)
    v = x[3:6]
    R = x[6:15]
    w = x[15:18]
    rzz, rxz, ryz = R[8], R[2], R[5]
    cross = np.concatenate([
        v * rzz,
        [v[2] * rxz, v[2] * ryz],
        w * rzz,
    ])
    return np.concatenate([x, cross, [1.0]])


from koopman_mpc import KoopmanMPC, lift

AuthenticKoopmanMPC = KoopmanMPC


# ======================================================================
# Reference Trajectory with Safe Terrain Tracking
# ======================================================================
def reference(t, z_prev, env, params):
    """Sinusoidal x/y trajectory with terrain-following altitude, rate-limited."""
    ax, ay, wx, wy, px, py = params
    x, y = ax * np.sin(wx * t + px), ay * np.sin(wy * t + py)
    g_h = env.ground_height(x, y)
    z_goal = g_h + AGL
    # Rate limit altitude changes gently (up to 0.8 m/s climb/descent)
    z = z_prev + np.clip(z_goal - z_prev, -0.80 * env.dt, 0.80 * env.dt)
    return np.array([x, y, z]), z


def add_segment(scn, p, q, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p, q)
    scn.ngeom += 1


def main():
    print("=" * 70)
    print(" Authentic Koopman-MPC Closed-Loop Flight in MuJoCo")
    print("=" * 70)
    print(f"Terrain: {TERRAIN} (seed {SEED}) | AGL: {AGL}m | Path Amp: {AMP}m | T: {T_END}s")
    if PAYLOAD > 0:
        print(f"Disturbance: Payload {PAYLOAD:+.0%} mass injected at t=3.0s")
    if WIND > 0:
        print(f"Disturbance: Crosswind {WIND}N ({WIND_DIR:.0f} deg) active at t=3.0s - 6.0s")
    print(f"Base model loaded from: {MODEL_FILE}")

    env = MujocoQuadEnv(terrain=TERRAIN, terrain_seed=SEED)
    ctrl = AuthenticKoopmanMPC(env)

    # Spawn directly at reference start point
    x0 = PARAMS[0] * np.sin(PARAMS[4])
    y0 = PARAMS[1] * np.sin(PARAMS[5])
    start_z = env.ground_height(x0, y0) + AGL
    r0 = np.array([x0, y0, start_z])
    s = env.reset(r0)
    z_ref = r0[2]
    log = []
    prev_p, prev_r = s[:3].copy(), r0.copy()
    payload_injected = False
    wind_ended = False

    viewer = None
    if GUI:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.body_id
        viewer.cam.distance = 0.65       # Close-up follow cam for realistic drone view
        viewer.cam.elevation = -22
        viewer.cam.azimuth = 135
        viewer.user_scn.ngeom = 0
        print("\n[Viewer Active] Close-up tracking camera enabled. Close window to exit.\n")

    steps = int(T_END / env.dt)
    status = "SUCCESS"

    for k in range(steps):
        t0 = time.time()
        t = k * env.dt
        ref, z_ref = reference(t, z_ref, env, PARAMS)
        env.set_ref_marker(ref)

        if PAYLOAD > 0 and t >= 3.0:
            env.set_payload(PAYLOAD)
            if not payload_injected:
                print(f"\n>>> [t={t:4.1f}s] DISTURBANCE INJECTED: {PAYLOAD:+.0%} Payload mass dropped onto drone!")
                payload_injected = True
        if WIND > 0 and 3.0 <= t < 6.0:
            rad = np.radians(WIND_DIR)
            env.set_wind(WIND * np.cos(rad), WIND * np.sin(rad))
        else:
            env.set_wind(0.0, 0.0)
            if WIND > 0 and t >= 6.0 and not wind_ended:
                print(f"\n>>> [t={t:4.1f}s] WIND DISTURBANCE ENDED: Controller stabilized back to calm state.")
                wind_ended = True

        # Build Hp-step prediction window with lookahead for safe terrain clearance
        ref_window = np.zeros((ctrl.Hp, 6))
        ax, ay, wx, wy, px, py = PARAMS
        for i in range(ctrl.Hp):
            tf = t + (i + 1) * env.dt
            fx = ax * np.sin(wx * tf + px)
            fy = ay * np.sin(wy * tf + py)
            fvx = ax * wx * np.cos(wx * tf + px)
            fvy = ay * wy * np.cos(wy * tf + py)
            fz = env.ground_height(fx, fy) + AGL
            ref_window[i] = [fx, fy, fz, fvx, fvy, 0.0]

        rpm = ctrl.rpm(s, ref_window)
        s = env.step(rpm)

        agl = env.agl()
        err = np.linalg.norm(s[:3] - ref)
        log.append(np.r_[t, s[:3], ref, agl, env.ground_height(s[0], s[1])])

        if k % 48 == 0 or (k == int(3.5 / env.dt)):
            tilt_deg = float(np.degrees(np.arccos(np.clip(s[14], -1.0, 1.0))))
            state_str = "HOLDING PAYLOAD" if (PAYLOAD > 0 and t >= 3.0) else "NOMINAL"
            if WIND > 0 and 3.0 <= t < 6.0:
                state_str += " + FIGHTING WIND"
            print(f"t={t:4.1f}s | pos=[{s[0]:+5.2f}, {s[1]:+5.2f}, {s[2]:5.2f}] | AGL={agl:4.2f}m | tilt={tilt_deg:4.1f}° | RPM={int(np.mean(rpm))} | err={err:5.3f}m | [{state_str}]")

        if GUI and viewer:
            if k % 3 == 0:
                add_segment(viewer.user_scn, prev_p, s[:3], 0.008, (1.0, 0.25, 0.1, 0.9))     # red drone trail
                add_segment(viewer.user_scn, prev_r, ref, 0.005, (0.1, 0.9, 0.3, 0.8))        # green reference trail
                prev_p, prev_r = s[:3].copy(), ref.copy()
            if WIND > 0 and 3.0 <= t < 6.0:
                rad = np.radians(WIND_DIR)
                w_vec = np.array([np.cos(rad), np.sin(rad), 0.0]) * 0.30
                add_segment(viewer.user_scn, s[:3], s[:3] + w_vec, 0.007, (0.0, 0.85, 1.0, 0.95)) # cyan wind force vector
            viewer.sync()
            if not viewer.is_running():
                status = "WINDOW_CLOSED"
                break
            time.sleep(max(0.0, env.dt - (time.time() - t0)))

        if env.touching_terrain() or env.flipped():
            status = f"FAILED (t={t:.2f}s, terrain contact or flip)"
            break

    if GUI and viewer and viewer.is_running():
        time.sleep(0.5)
        viewer.close()

    log = np.array(log)
    pos_err = np.linalg.norm(log[:, 1:4] - log[:, 4:7], axis=1)
    rmse = np.sqrt(np.mean(pos_err**2))
    max_err = np.max(pos_err)
    min_agl = np.min(log[:, 7])

    print("\n" + "-" * 70)
    print(f"Status:   {status}")
    print(f"Steps:    {len(log)} / {steps}")
    print(f"RMSE:     {rmse:.3f} m")
    print(f"Max Err:  {max_err:.3f} m")
    print(f"Min AGL:  {min_agl:.3f} m")
    print("-" * 70)

    out_file = os.path.join(MUJOCO_DIR, "mujoco_run_koopman.npz")
    np.savez(out_file, log=log)
    print(f"Saved flight log to: {out_file}\n")


if __name__ == "__main__":
    main()