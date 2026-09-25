import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sys

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "1"
CONTROLLER = sys.argv[2] if len(sys.argv) > 2 else "pid"  # pid, lqr, or scenario (mpc)

prefix = "scenario" if CONTROLLER == "mpc" else f"{CONTROLLER}_scenario"
data = np.load(f"results/{prefix}_{SCENARIO}.npz")
states = data["states"]

x, y, z = states[:, 0], states[:, 1], states[:, 2]
roll, pitch, yaw = states[:, 7], states[:, 8], states[:, 9]

CTRL_FREQ = 48
STEP = 4  # sample every 4th frame to keep the animation smooth but not huge

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(x.min()-0.2, x.max()+0.2)
ax.set_ylim(y.min()-0.2, y.max()+0.2)
ax.set_zlim(0, max(z.max()+0.1, 0.4))
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"{CONTROLLER.upper()} - Scenario {SCENARIO} Flight Path")

path_line, = ax.plot([], [], [], "b-", linewidth=1, alpha=0.5)
drone_point, = ax.plot([], [], [], "ro", markersize=10)
arm_lines = [ax.plot([], [], [], "k-", linewidth=2)[0] for _ in range(2)]

ARM_LEN = 0.05

def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

def update(frame):
    i = frame * STEP
    if i >= len(x):
        i = len(x) - 1
    path_line.set_data(x[:i], y[:i])
    path_line.set_3d_properties(z[:i])
    drone_point.set_data([x[i]], [y[i]])
    drone_point.set_3d_properties([z[i]])

    R = rotation_matrix(roll[i], pitch[i], yaw[i])
    arm1 = R @ np.array([ARM_LEN, 0, 0])
    arm2 = R @ np.array([0, ARM_LEN, 0])
    center = np.array([x[i], y[i], z[i]])
    p1a, p1b = center - arm1, center + arm1
    p2a, p2b = center - arm2, center + arm2
    arm_lines[0].set_data([p1a[0], p1b[0]], [p1a[1], p1b[1]])
    arm_lines[0].set_3d_properties([p1a[2], p1b[2]])
    arm_lines[1].set_data([p2a[0], p2b[0]], [p2a[1], p2b[1]])
    arm_lines[1].set_3d_properties([p2a[2], p2b[2]])

    ax.set_title(f"{CONTROLLER.upper()} - Scenario {SCENARIO}  |  t = {i/CTRL_FREQ:.2f}s")
    return path_line, drone_point, *arm_lines

n_frames = len(x) // STEP
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)

out_path = f"results/plots_updated/flight_animation_{CONTROLLER}_scenario_{SCENARIO}.gif"
ani.save(out_path, writer="pillow", fps=25)
print(f"Saved animation to {out_path}")
