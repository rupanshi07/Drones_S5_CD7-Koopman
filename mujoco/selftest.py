"""Run this first. Headless checks: hover, motor-sign, terrain lookup vs. physics engine, collision."""
import numpy as np, mujoco
from mujoco_quad_env import MujocoQuadEnv
from terrain import max_slope_deg

env = MujocoQuadEnv(terrain_seed=0)
H = env.HOVER_RPM
print(f"hover RPM {H:.0f}, max RPM {env.MAX_RPM:.0f}, control dt {env.dt:.4f}s, max terrain slope {max_slope_deg(env.model):.1f} deg")

# 1. hover on the flat spawn pad
env.reset((0, 0, 1.0))
for _ in range(48 * 3):
    s = env.step([H] * 4)
print(f"[hover 3s]  z={s[2]:.4f}  |v|={np.linalg.norm(s[3:6]):.5f}  R_zz={s[3+3+8]:.5f}   (want z~1, v~0, R_zz~1)")

# 2. motor sign checks
def spin(delta):
    env.reset((0, 0, 3.0))
    for _ in range(10):
        s = env.step(np.array([H] * 4) + delta)
    return s[15:18]
w = spin(np.array([600, 600, 0, 0.]));  print(f"[roll  test] motors 0,1 up (left) -> omega={np.round(w,3)}  (want wx > 0)")
w = spin(np.array([0, 600, 600, 0.]));  print(f"[pitch test] motors 1,2 up (rear) -> omega={np.round(w,3)}  (want wy > 0)")
w = spin(np.array([0, 600, 0, 600.]));  print(f"[yaw   test] motors 1,3 up (CW)   -> omega={np.round(w,3)}  (want wz > 0)")

# 3. terrain lookup vs the engine's own ray cast
worst = 0.0
for x, y in np.random.default_rng(1).uniform(-5.5, 5.5, (20, 2)):
    pnt = np.array([x, y, 5.0]); vec = np.array([0, 0, -1.0]); gid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(env.model, env.data, pnt, vec, None, 1, -1, gid)
    worst = max(worst, abs((5.0 - dist) - env.ground_height(x, y)))
print(f"[terrain]   max |lookup - ray| = {worst*1000:.2f} mm   (want < ~5 mm)")

# 4. collision: motors off, drop onto a hill
x, y = 3.0, 3.0
env.reset((x, y, env.ground_height(x, y) + 0.6))
t = 0
while not env.touching_terrain() and t < 96:
    env.step([0] * 4); t += 1
print(f"[collision] terrain contact after {t*env.dt:.2f}s (ground there {env.ground_height(x,y):.2f} m)")
