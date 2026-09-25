import numpy as np

d = np.load('data/raw/episode_0000.npz')
R = d['rotmat'].reshape(-1, 3, 3)
roll = np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2]))
pitch = np.degrees(-np.arcsin(np.clip(R[:, 2, 0], -1, 1)))

for i in range(0, len(d['t']), 10):
    rpm_str = np.array2string(np.round(d['rpm'][i], 0))
    print(f"t={d['t'][i]:.2f}  roll={roll[i]:7.2f}  pitch={pitch[i]:7.2f}  rpm={rpm_str}")
