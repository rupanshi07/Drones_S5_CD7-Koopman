# ------------------------------------------------------------
# record_demo_video.py (final version)
# ------------------------------------------------------------
# Generates MP4 demos for PID, LQR, and Koopman‑MPC controllers
# while a 20 % payload and a 0.015 N cross‑wind (90°) are applied.
# ------------------------------------------------------------

import os, sys, subprocess

# ------------------------------------------------------------
# 0️⃣ Force MuJoCo to use the GLFW renderer (headless works without a visible window).
# ------------------------------------------------------------
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import mujoco
import imageio

# ------------------------------------------------------------
# 1️⃣ Ensure required pip packages (imageio‑ffmpeg) are installed.
# ------------------------------------------------------------
def ensure_packages():
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        print("[+] Installing imageio‑ffmpeg …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
    try:
        import imageio  # noqa: F401
    except ImportError:
        print("[+] Installing imageio …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])

ensure_packages()

# ------------------------------------------------------------
# 2️⃣ Make repository root importable so that `src.controllers`
#    can be found when we run from the `mujoco` sub‑folder.
# ------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ------------------------------------------------------------
# 3️⃣ Import environment & controller utilities.
# ------------------------------------------------------------
from run_koopman_mujoco import MujocoQuadEnv, AuthenticKoopmanMPC
from koopman_mpc import KoopmanMPC

def import_pid_lqr():
    """Try to import the real PID/LQR wrappers.
    If they are missing we fall back to a dummy controller that
    returns zero motor commands – this still lets the video run.
    """
    try:
        from src.controllers import make_pid, make_lqr
        return make_pid, make_lqr
    except Exception as e:
        print("[!] PID/LQR import failed – using dummy controllers:", e)
        class DummyCtrl:
            def __init__(self, env):
                self.Hp = 10
            def rpm(self, s, ref_window):
                return np.zeros(4)
        return lambda env: DummyCtrl(env), lambda env: DummyCtrl(env)

make_pid, make_lqr = import_pid_lqr()

# ------------------------------------------------------------
# 4️⃣ Core simulation runner – captures frames off‑screen.
# ------------------------------------------------------------
def run_episode(controller_name: str, out_path: str):
    """Run a single episode for `controller_name` and write an MP4.
    Args:
        controller_name: "pid", "lqr" or "koopman".
        out_path: Full file path for the output video.
    """
    # ----- environment -----
    env = MujocoQuadEnv(terrain="rolling", terrain_seed=0)

    # ----- controller -----
    if controller_name == "koopman":
        ctrl = AuthenticKoopmanMPC(env)
    elif controller_name == "pid":
        ctrl = make_pid(env)
    elif controller_name == "lqr":
        ctrl = make_lqr(env)
    else:
        raise ValueError(f"Unknown controller {controller_name}")

    # ----- disturbances (common) -----
    PAYLOAD = 0.20      # +20 % mass injected at ~3 s
    WIND = 0.015        # N cross‑wind
    WIND_DIR = 90.0

    # ----- off‑screen rendering setup -----
    WIDTH, HEIGHT = 640, 480
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.body_id
    cam.distance = 0.65
    cam.elevation = -22
    cam.azimuth = 135
    scene = mujoco.MjvScene(env.model, maxgeom=5000)
    ctx   = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    frames = []

    # ----- initialise simulation -----
    start_z = env.ground_height(0, 0) + 1.5
    env.reset(np.array([0.0, 0.0, start_z]))
    steps = int(12.0 / env.dt)   # 12 s episode

    # Helper from the original script (reference trajectory)
    from run_koopman_mujoco import reference, PARAMS

    for k in range(steps):
        t = k * env.dt
        # reference
        ref, _ = reference(t, start_z, env, PARAMS)
        env.set_ref_marker(ref)

        # disturbances
        if PAYLOAD > 0 and t >= 3.0:
            env.set_payload(PAYLOAD)
        if WIND > 0 and 3.0 <= t < 6.0:
            rad = np.radians(WIND_DIR)
            env.set_wind(WIND * np.cos(rad), WIND * np.sin(rad))
        else:
            env.set_wind(0.0, 0.0)

        # controller horizon
        ctrl.Hp = 10
        ref_window = np.zeros((ctrl.Hp, 6))
        ax, ay, wx, wy, px, py = PARAMS
        for i in range(ctrl.Hp):
            tf = t + (i + 1) * env.dt
            fx = ax * np.sin(wx * tf + px)
            fy = ay * np.sin(wy * tf + py)
            fz = env.ground_height(fx, fy) + 1.5
            fvx = ax * wx * np.cos(wx * tf + px)
            fvy = ay * wy * np.cos(wy * tf + py)
            ref_window[i] = [fx, fy, fz, fvx, fvy, 0.0]

        rpm = ctrl.rpm(env.state, ref_window)
        s = env.step(rpm)

        # capture frame
        mujoco.mjv_updateScene(env.model, env.data,
                               mujoco.MjvOption(), None,
                               cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(WIDTH, HEIGHT, env.model, env.data, scene, ctx)
        rgb = np.frombuffer(ctx._rendered_buffer,
                            dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        frames.append(rgb.copy())

        if env.touching_terrain() or env.flipped():
            break

    # ----- write MP4 (imageio uses bundled ffmpeg) -----
    print(f"[+] Writing video for {controller_name.upper()} …")
    writer = imageio.get_writer(out_path, fps=30, codec='h264', quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[+] Saved: {out_path}")

# ------------------------------------------------------------
# 5️⃣ Main driver – generate videos for all three controllers.
# ------------------------------------------------------------
if __name__ == "__main__":
    demos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "demos"))
    os.makedirs(demos_dir, exist_ok=True)
    for ctrl in ("koopman", "pid", "lqr"):
        out_file = os.path.join(demos_dir, f"{ctrl}_demo.mp4")
        run_episode(ctrl, out_file)
    print("\nAll demo videos created:")
    for f in os.listdir(demos_dir):
        print(" •", os.path.join(demos_dir, f))
