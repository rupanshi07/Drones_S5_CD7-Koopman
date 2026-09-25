"""
Step 1-2: Build the raw state/input dataset from all episodes, and define
the lifting (observable) function for EDMD.

State we track per timestep (13-dim, before lifting):
    pos (3) + vel (3) + rotmat (9) ... wait, that's 15. Let's be precise:
    pos(3), vel(3), rotmat flattened (9), ang_vel(3)  => 18-dim raw state
Input: 4 motor RPMs.

Run this after collect_koopman_data.py has produced data/raw/episode_*.npz.
"""

import glob
import numpy as np

DATA_DIR = "data/raw"
N_HOLDOUT_EPISODES = 40  # last N episodes reserved for validation, not training


def load_episode_raw_state(path):
    d = np.load(path)
    # raw state: pos(3) vel(3) rotmat(9) ang_vel(3) = 18-dim
    raw_state = np.hstack([d["pos"], d["vel"], d["rotmat"], d["ang_vel"]])
    u = d["rpm"]
    meta = {
        "payload_mass": d["payload_mass"],
        "wind_force": d["wind_force"],
    }
    return raw_state, u, meta


def lift(raw_state):
    """
    Lifting function psi(x) -> higher-dim observable space.
    raw_state: (N, 18) array, columns = [pos(3), vel(3), rotmat(9), ang_vel(3)]

    Keep this modest and interpretable at first: the raw state itself,
    plus a handful of quadratic cross-terms known to matter physically
    for a quadrotor (velocity coupled with orientation determines how
    thrust translates into world-frame acceleration).
    """
    pos = raw_state[:, 0:3]
    vel = raw_state[:, 3:6]
    rotmat = raw_state[:, 6:15]  # 9 entries, row-major 3x3
    ang_vel = raw_state[:, 15:18]

    # quadratic cross terms: how each velocity component projects through
    # the current orientation (this is physically what determines
    # world-frame acceleration from body-frame thrust)
    r_zz = rotmat[:, 8:9]  # R[2,2], the "how upright" term
    r_xz = rotmat[:, 2:3]  # R[0,2]
    r_yz = rotmat[:, 5:6]  # R[1,2]
    cross_terms = np.hstack([
        vel * r_zz,           # 3 terms
        vel[:, 2:3] * r_xz,   # vz * R[0,2]
        vel[:, 2:3] * r_yz,   # vz * R[1,2]
        ang_vel * r_zz,       # 3 terms
    ])

    ones = np.ones((raw_state.shape[0], 1))  # bias/constant term

    lifted = np.hstack([raw_state, cross_terms, ones])
    return lifted


def is_episode_healthy(raw_state):
    """Reject episodes that crashed or went badly unstable -- these
    contaminate the EDMD regression far more than they inform it."""
    pos = raw_state[:, 0:3]
    rotmat = raw_state[:, 6:15].reshape(-1, 3, 3)
    roll = np.degrees(np.arctan2(rotmat[:, 2, 1], rotmat[:, 2, 2]))
    pitch = np.degrees(-np.arcsin(np.clip(rotmat[:, 2, 0], -1, 1)))
    if np.max(np.abs(roll)) > 30 or np.max(np.abs(pitch)) > 30:
        return False
    if pos[:, 2].min() < 0.05:
        return False
    if np.max(np.abs(pos[:, :2])) > 3.0:
        return False
    return True


def build_dataset(episode_files):
    """Returns stacked (Psi_k, Psi_k+1, U_k) across all given episodes,
    automatically skipping any episode that crashed or went unstable."""
    Psi_k_list, Psi_kp1_list, U_k_list = [], [], []
    n_skipped = 0
    for f in episode_files:
        raw_state, u, _ = load_episode_raw_state(f)
        if not is_episode_healthy(raw_state):
            n_skipped += 1
            continue
        lifted = lift(raw_state)
        Psi_k_list.append(lifted[:-1])
        Psi_kp1_list.append(lifted[1:])
        U_k_list.append(u[:-1])
    if n_skipped:
        print(f"Skipped {n_skipped}/{len(episode_files)} unstable/crashed episodes.")
    Psi_k = np.vstack(Psi_k_list)
    Psi_kp1 = np.vstack(Psi_kp1_list)
    U_k = np.vstack(U_k_list)
    return Psi_k, Psi_kp1, U_k


if __name__ == "__main__":
    all_files = sorted(glob.glob(f"{DATA_DIR}/episode_*.npz"))
    print(f"Found {len(all_files)} episode files.")

    train_files = all_files[:-N_HOLDOUT_EPISODES]
    holdout_files = all_files[-N_HOLDOUT_EPISODES:]
    print(f"Training on {len(train_files)} episodes, holding out {len(holdout_files)}.")

    Psi_k, Psi_kp1, U_k = build_dataset(train_files)
    print(f"Lifted state dimension: {Psi_k.shape[1]}")
    print(f"Total training samples (timesteps across all episodes): {Psi_k.shape[0]}")
    print(f"Control input dimension: {U_k.shape[1]}")

    np.savez(
        "data/edmd_train.npz",
        Psi_k=Psi_k, Psi_kp1=Psi_kp1, U_k=U_k,
        train_files=train_files, holdout_files=holdout_files,
    )
    print("Saved data/edmd_train.npz")
