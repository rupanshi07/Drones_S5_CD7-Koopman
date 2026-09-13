import glob
import numpy as np

files = sorted(glob.glob("data/raw/episode_*.npz"))
print(f"Scanning {len(files)} episodes for instability...\n")

bad_episodes = []
for f in files:
    d = np.load(f)
    pos = d["pos"]
    rotmat = d["rotmat"].reshape(-1, 3, 3)
    roll = np.degrees(np.arctan2(rotmat[:, 2, 1], rotmat[:, 2, 2]))
    pitch = np.degrees(-np.arcsin(np.clip(rotmat[:, 2, 0], -1, 1)))

    max_roll = np.max(np.abs(roll))
    max_pitch = np.max(np.abs(pitch))
    min_z = pos[:, 2].min()
    max_xy = np.max(np.abs(pos[:, :2]))

    # flag anything that looks like a crash or serious instability
    is_bad = (max_roll > 30) or (max_pitch > 30) or (min_z < 0.05) or (max_xy > 3.0)
    if is_bad:
        bad_episodes.append((f, max_roll, max_pitch, min_z, max_xy))

print(f"Found {len(bad_episodes)} problematic episodes out of {len(files)}:\n")
for f, mr, mp, mz, mxy in bad_episodes:
    print(f"  {f}: max_roll={mr:.1f} max_pitch={mp:.1f} min_z={mz:.3f} max_xy={mxy:.2f}")

print(f"\n{len(bad_episodes)}/{len(files)} episodes ({100*len(bad_episodes)/len(files):.1f}%) are problematic.")
