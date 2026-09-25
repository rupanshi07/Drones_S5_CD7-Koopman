import numpy as np
import matplotlib.pyplot as plt
import sys
import os

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "1"
CTRL_FREQ = 48

mpc_data = np.load(f"results/scenario_{SCENARIO}.npz")
pid_data = np.load(f"results/pid_scenario_{SCENARIO}.npz")
lqr_data = np.load(f"results/lqr_scenario_{SCENARIO}.npz")

mpc_z = mpc_data["states"][:, 2]
pid_z = pid_data["states"][:, 2]
lqr_z = lqr_data["states"][:, 2]

t_mpc = np.arange(len(mpc_z)) / CTRL_FREQ
t_pid = np.arange(len(pid_z)) / CTRL_FREQ
t_lqr = np.arange(len(lqr_z)) / CTRL_FREQ

target_z = 0.3

plt.figure(figsize=(10, 6))
plt.axhline(y=target_z, color="red", linestyle="--", label="Target Z")
plt.plot(t_mpc, mpc_z, color="blue", label="MPC Z-pos")
plt.plot(t_pid, pid_z, color="orange", label="PID Z-pos")
plt.plot(t_lqr, lqr_z, color="green", label="LQR Z-pos")
plt.xlabel("Time (s)")
plt.ylabel("Z Position (m)")
plt.title(f"Scenario {SCENARIO} Z-Tracking Performance (updated)")
plt.legend()
plt.grid(True)

os.makedirs("results/plots_updated", exist_ok=True)
plt.savefig(f"results/plots_updated/scenario_{SCENARIO}_z_tracking.png", dpi=150)
print(f"Saved results/plots_updated/scenario_{SCENARIO}_z_tracking.png")
