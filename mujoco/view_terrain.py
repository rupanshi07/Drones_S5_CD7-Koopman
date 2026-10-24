"""Open the MuJoCo viewer.   python view_terrain.py [hover|fall] [--seed N]
(macOS: run with `mjpython view_terrain.py` instead of `python`.)"""
import sys, time
import mujoco, mujoco.viewer
from mujoco_quad_env import MujocoQuadEnv

mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "hover"
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
terrain = sys.argv[sys.argv.index("--terrain") + 1].lower() if "--terrain" in sys.argv else ("flat" if "--flat" in sys.argv else "rolling")
env = MujocoQuadEnv(terrain_seed=seed, terrain=terrain)
env.reset((0, 0, 1.0) if mode == "hover" else (2.0, 2.0, env.ground_height(2.0, 2.0) + 1.0))

with mujoco.viewer.launch_passive(env.model, env.data) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid, v.cam.distance, v.cam.elevation, v.cam.azimuth = env.body_id, 0.65, -20, 135
    while v.is_running():
        t0 = time.time()
        env.step([env.HOVER_RPM if mode == "hover" else 0.0] * 4)
        v.sync()
        time.sleep(max(0.0, env.dt - (time.time() - t0)))          # real-time pacing
