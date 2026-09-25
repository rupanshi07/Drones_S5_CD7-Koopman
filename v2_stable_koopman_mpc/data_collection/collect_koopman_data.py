"""
Data collection for Koopman/EDMD modeling of a disturbed quadrotor.

Matches the API of the utiasDSL gym-pybullet-drones v1.0.0 tag specifically
(confirmed against that version's examples/fly.py):
  - obs/action are DICTS keyed by str(drone_index), each obs[str(j)]["state"]
    is the 20-element state vector: pos(3) quat(4) rpy(3) vel(3) ang_vel(3) rpm(4)
  - env.step(action) returns the OLD gym 4-tuple: (obs, reward, done, info)
  - env.reset() returns just obs (no info dict)
  - control frequency is handled manually via CTRL_EVERY_N_STEPS, not a
    ctrl_freq constructor argument
  - CtrlAviary takes: drone_model, num_drones, initial_xyzs, initial_rpys,
    physics, neighbourhood_radius, freq (=sim freq), aggregate_phy_steps,
    gui, record, obstacles, user_debug_gui

ASSUMPTION TO VERIFY: this script accesses the drone's pybullet body id via
`env.DRONE_IDS[0]` to apply payload/wind disturbances directly with pybullet
calls. If this attribute name is wrong for your checked-out version, run:
    Select-String -Path .\\gym_pybullet_drones\\envs\\BaseAviary.py -Pattern "DRONE_IDS|loadURDF"
and tell me what you find so I can correct it.
"""

import os
import csv
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR = "data/raw"
N_EPISODES = 2
EPISODE_LEN_SEC = 8.0
SIM_FREQ = 240          # Hz, physics
CTRL_FREQ = 48           # Hz, control/logging rate
AGGR_PHY_STEPS = int(SIM_FREQ / CTRL_FREQ)
DRONE_MODEL = DroneModel.CF2X

# Disturbance ranges
PAYLOAD_MASS_FRACS = [0.0, 0.10, 0.25, 0.40]
WIND_FORCE_RANGE_N = (0.0, 0.03)
WIND_ON_PROB = 0.6
PAYLOAD_ON_PROB = 0.6

RNG = np.random.default_rng(42)


def random_smooth_reference(t, freqs, phases, amp=0.4, z0=1.0):
    x = amp[0] * np.sin(freqs[0] * t + phases[0])
    y = amp[1] * np.sin(freqs[1] * t + phases[1])
    z = z0 + 0.15 * np.sin(freqs[2] * t + phases[2])
    return np.array([x, y, z])


def generate_excitation_signal(n_steps, dt, theta=1.5, sigma=0.03, seed_rng=None):
    """Ornstein-Uhlenbeck process: smooth, correlated wandering noise
    (as a fraction of mean RPM) rather than independent per-step jumps.
    theta controls how fast it reverts to zero; sigma controls magnitude."""
    r = seed_rng if seed_rng is not None else RNG
    n = np.zeros((n_steps, 4))
    for k in range(1, n_steps):
        n[k] = n[k - 1] + theta * (0 - n[k - 1]) * dt + sigma * np.sqrt(dt) * r.normal(size=4)
    return n


def quat_to_rotmat(quat_xyzw):
    r = p.getMatrixFromQuaternion(quat_xyzw)
    return np.array(r).reshape(3, 3)


def run_episode(ep_idx):
    # generate the reference's randomness FIRST, so we can spawn the drone
    # exactly on the trajectory's starting point (avoids a sudden step input
    # at t=0, which was causing the PID to command an aggressive initial
    # tilt and destabilize -- see fly.py, which does the same thing)
    # widened ranges vs. the original dataset: amplitude now varies per
    # episode across a much larger span (0.3 to 1.0m, was fixed at 0.4m),
    # and frequency range widened too -- this elicits genuinely bigger,
    # sharper, more varied PID corrections, teaching the Koopman model
    # what larger control differentials actually do (the original dataset
    # only ever saw gentle, small-amplitude tracking, so the model never
    # learned to generalize to the aggressive regime an MPC would want)
    freqs = RNG.uniform(0.05, 0.4, size=3)
    phases = RNG.uniform(0, 2 * np.pi, size=3)
    amp = RNG.uniform(0.3, 1.0, size=2)  # separate random amplitude for x, y
    start_pos = random_smooth_reference(0.0, freqs=freqs, phases=phases, amp=amp)

    env = CtrlAviary(
        drone_model=DRONE_MODEL,
        num_drones=1,
        initial_xyzs=np.array([start_pos]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.PYB,
        neighbourhood_radius=10,
        freq=SIM_FREQ,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )
    ctrl = DSLPIDControl(drone_model=DRONE_MODEL)
    client = env.getPyBulletClient()
    drone_id = env.DRONE_IDS[0]  # <-- verify this attribute name, see docstring
    nominal_mass = p.getDynamicsInfo(drone_id, -1, physicsClientId=client)[0]

    obs = env.reset()

    n_ctrl_steps = int(EPISODE_LEN_SEC * CTRL_FREQ)

    # --- randomize this episode's disturbance schedule ---
    excitation = generate_excitation_signal(n_ctrl_steps, dt=1.0 / CTRL_FREQ)
    payload_on = RNG.random() < PAYLOAD_ON_PROB
    wind_on = RNG.random() < WIND_ON_PROB
    payload_frac = RNG.choice(PAYLOAD_MASS_FRACS) if payload_on else 0.0
    payload_start_ctrl_step = RNG.integers(n_ctrl_steps // 4, n_ctrl_steps // 2) if payload_on else None
    wind_force = RNG.uniform(*WIND_FORCE_RANGE_N) if wind_on else 0.0
    wind_dir = RNG.normal(size=3)
    wind_dir[2] *= 0.2
    wind_dir = wind_dir / (np.linalg.norm(wind_dir) + 1e-8)
    wind_start_ctrl_step = RNG.integers(n_ctrl_steps // 4, 3 * n_ctrl_steps // 4) if wind_on else None
    wind_duration_ctrl_steps = RNG.integers(1, 3 * CTRL_FREQ) if wind_on else 0

    log = {
        "t": [], "pos": [], "vel": [], "rotmat": [], "ang_vel": [],
        "rpm": [], "ref_pos": [], "payload_mass": [], "wind_force": [],
    }

    applied_payload = False
    action = {"0": np.array([0.0, 0.0, 0.0, 0.0])}
    ctrl_step = -1

    total_phys_steps = int(EPISODE_LEN_SEC * SIM_FREQ)

    for i in range(0, total_phys_steps, AGGR_PHY_STEPS):
        # this loop iterates once per control step, since env.step()
        # internally advances AGGR_PHY_STEPS physics steps at a time
        ctrl_step += 1
        t = i / SIM_FREQ

        # --- disturbance injection (applied once per control step) ---
        current_payload_mass = 0.0
        if payload_on and ctrl_step >= payload_start_ctrl_step:
            if not applied_payload:
                p.changeDynamics(
                    drone_id, -1,
                    mass=nominal_mass * (1.0 + payload_frac),
                    physicsClientId=client,
                )
                applied_payload = True
            current_payload_mass = nominal_mass * payload_frac

        current_wind_vec = np.zeros(3)
        if wind_on and wind_start_ctrl_step <= ctrl_step < wind_start_ctrl_step + wind_duration_ctrl_steps:
            current_wind_vec = wind_force * wind_dir
            p.applyExternalForce(
                drone_id, -1,
                forceObj=current_wind_vec.tolist(),
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=client,
            )

        # --- reference tracking + excitation ---
        target_pos = random_smooth_reference(t, freqs=freqs, phases=phases, amp=amp)
        state = obs["0"]["state"]
        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        ang_vel = state[13:16]

        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=AGGR_PHY_STEPS / SIM_FREQ,
            state=state,
            target_pos=target_pos,
            target_rpy=np.zeros(3),
        )
        rpm_excited = rpm * (1.0 + excitation[ctrl_step])
        rpm_excited = np.clip(rpm_excited, 0, None)
        action = {"0": rpm_excited}

        obs, _, done, info = env.step(action)

        # --- log (state used above is the pre-step state at time t) ---
        log["t"].append(t)
        log["pos"].append(pos.copy())
        log["vel"].append(vel.copy())
        log["rotmat"].append(quat_to_rotmat(quat).flatten())
        log["ang_vel"].append(ang_vel.copy())
        log["rpm"].append(rpm_excited.copy())
        log["ref_pos"].append(target_pos.copy())
        log["payload_mass"].append(current_payload_mass)
        log["wind_force"].append(current_wind_vec.copy())

    env.close()

    for key in log:
        log[key] = np.array(log[key])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"episode_{ep_idx:04d}.npz")
    np.savez(out_path, **log)

    return {
        "episode": ep_idx,
        "file": out_path,
        "payload_on": payload_on,
        "payload_frac": payload_frac,
        "wind_on": wind_on,
        "wind_force_N": wind_force,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    rows = []
    for i in range(N_EPISODES):
        print(f"Running episode {i+1}/{N_EPISODES} ...")
        rows.append(run_episode(i))

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
