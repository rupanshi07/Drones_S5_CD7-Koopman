import numpy as np
import matplotlib.pyplot as plt

episodes_to_check = {
    11: "payload only (40%)",
    5: "wind only (strong)",
    1: "payload + wind (toughest)",
    18: "clean baseline",
}

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for ax, (ep, label) in zip(axes, episodes_to_check.items()):
    d = np.load(f"data/raw/episode_{ep:04d}.npz")
    ax.plot(d["pos"][:, 0], d["pos"][:, 1], label="actual")
    ax.plot(d["ref_pos"][:, 0], d["ref_pos"][:, 1], "--", label="reference")
    ax.set_title(f"ep {ep}: {label}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("check_spread.png")
print("saved check_spread.png")

print()
print(f"{'episode':>8} {'label':<28} {'max|roll|':>10} {'max|pitch|':>11} {'max_rpm':>9} {'min_rpm':>9} {'final_z':>9}")
for ep, label in episodes_to_check.items():
    d = np.load(f"data/raw/episode_{ep:04d}.npz")
    R = d["rotmat"].reshape(-1, 3, 3)
    roll = np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    pitch = np.degrees(-np.arcsin(np.clip(R[:, 2, 0], -1, 1)))
    print(f"{ep:>8} {label:<28} {np.max(np.abs(roll)):>10.2f} {np.max(np.abs(pitch)):>11.2f} "
          f"{d['rpm'].max():>9.0f} {d['rpm'].min():>9.0f} {d['pos'][-1,2]:>9.3f}")
