r"""
Side-by-side obstacle-course demo: PID vs LQR vs Koopman-MPC, flying a shared
reference trajectory that has been bent around a set of static box obstacles
using a repulsive potential field, run simultaneously in one PyBullet GUI
window.

Obstacle avoidance is implemented ONCE, at the reference-trajectory level: a
repulsive force pushes the nominal (straight-line) reference away from each
obstacle's position whenever the reference passes within an influence radius.
All three controllers then simply track this already-bent reference -- none
of PID/LQR/Koopman-MPC's internal control logic is changed or obstacle-aware.
This keeps the controller comparison itself honest: any difference in how
closely each drone follows the bent path is purely a tracking-quality
difference, not a difference in avoidance intelligence.

Run from v2_stable_koopman_mpc/ with PYTHONPATH set to its subfolders, e.g.:

    cd v2_stable_koopman_mpc
    $env:PYTHONPATH = "$PWD\koopman_model;$PWD\mpc;$PWD\baselines"
    python ..\demo_flight_obstacles.py
"""
import time

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from build_edmd_dataset import lift
from koopman_mpc import KoopmanMPC
from lqr_baseline import LQRController

SIM_FREQ = 240
CTRL_FREQ = 48
AGGR_PHY_STEPS = int(SIM_FREQ / CTRL_FREQ)
EPISODE_LEN_SEC = 8.0
DRONE_MODEL = DroneModel.CF2X

LANE_OFFSET = {"pid": -0.8, "lqr": 0.0, "koopman_mpc": 0.8}
CONTROLLER_ORDER = ["pid", "lqr", "koopman_mpc"]
CONTROLLER_LABEL = {"pid": "PID", "lqr": "LQR", "koopman_mpc": "Koopman-MPC"}
PATH_COLOR = {"pid": [0, 1, 0], "lqr": [1, 0.6, 0], "koopman_mpc": [0, 0.6, 1]}

# --- static obstacles: box pillars, defined in the SAME local frame each lane
# flies in (i.e. relative to that lane's own Y-offset, so all three lanes see
# an identical obstacle layout relative to their own flight path) ---
OBSTACLES_LOCAL = [
    {"pos": np.array([0.3, 0.55, 1.0]), "half_extents": [0.05, 0.05, 0.5]},
    {"pos": np.array([0.55, 0.15, 1.0]), "half_extents": [0.05, 0.05, 0.5]},
]
DRONE_SAFETY_RADIUS = 0.15      # m -- approx. drone footprint + margin
OBSTACLE_CLEARANCE_DIST = 0.30  # m -- extra stand-off beyond the safety radius


def nominal_trajectory(t, amp=0.7, freq=0.15, z0=1.0):
    x = amp * np.sin(freq * t)
    y = amp * np.sin(2 * freq * t)
    return np.array([x, y, z0])


def bend_around_obstacles(target):
    """For each obstacle currently too close to the candidate target, push the
    target directly away from that obstacle's center, radially, until it
    clears (obstacle radius + drone safety radius + clearance distance).

    This is evaluated fresh from scratch on every call -- no precomputed time
    windows -- so it is correct by construction even when the reference
    trajectory is periodic and revisits nearby spatial points at different
    times, which broke an earlier precomputed-window version of this
    function (windows computed from a single pass over t could overlap or
    span nearly the whole flight for an oscillating path)."""
    adjusted = target.copy()
    for obs in OBSTACLES_LOCAL:
        obs_radius = max(obs["half_extents"][0], obs["half_extents"][1])
        min_clear = obs_radius + DRONE_SAFETY_RADIUS + OBSTACLE_CLEARANCE_DIST

        delta = adjusted[:2] - obs["pos"][:2]
        dist = np.linalg.norm(delta)
        if dist < min_clear:
            direction = delta / dist if dist > 1e-6 else np.array([1.0, 0.0])
            adjusted[0] = obs["pos"][0] + direction[0] * min_clear
            adjusted[1] = obs["pos"][1] + direction[1] * min_clear
    return adjusted


_SMOOTH_DT = 0.01                 # s -- resolution used to pre-smooth the bent path
_SMOOTH_ALPHA = 0.06               # exponential-smoothing factor (lower = smoother,
                                    # slower to react; tuned to remove a verified
                                    # 0.87 m single-step jump in the raw clamped path)
_SMOOTHED_PATH_CACHE = {}


def _smoothed_local_trajectory(episode_len_sec):
    """Pre-compute an exponentially-smoothed version of the LOCAL (lane_offset=0)
    bent trajectory once, at high time resolution, and cache it. The raw
    per-step obstacle clamp in bend_around_obstacles() is correct in terms of
    final clearance but can jump discontinuously as the clamp direction
    flips; smoothing the resulting path removes that jump while leaving the
    clamp logic itself untouched. Caching avoids recomputing this every call
    across all three lanes and every control step."""
    key = episode_len_sec
    if key in _SMOOTHED_PATH_CACHE:
        return _SMOOTHED_PATH_CACHE[key]

    ts = np.arange(0.0, episode_len_sec + 10 * _SMOOTH_DT, _SMOOTH_DT)
    raw = np.array([bend_around_obstacles(nominal_trajectory(tt)) for tt in ts])
    smoothed = np.zeros_like(raw)
    smoothed[0] = raw[0]
    for i in range(1, len(raw)):
        smoothed[i] = _SMOOTH_ALPHA * raw[i] + (1 - _SMOOTH_ALPHA) * smoothed[i - 1]

    _SMOOTHED_PATH_CACHE[key] = (ts, smoothed)
    return ts, smoothed


def reference_trajectory(t, lane_offset):
    """Nominal trajectory, bent around obstacles and smoothed to remove
    discontinuities, then shifted into this lane's local frame."""
    ts, smoothed = _smoothed_local_trajectory(EPISODE_LEN_SEC)
    idx = int(np.clip(round(t / _SMOOTH_DT), 0, len(ts) - 1))
    local = smoothed[idx]
    return local + np.array([0.0, lane_offset, 0.0])


def spawn_obstacles(client):
    for obs in OBSTACLES_LOCAL:
        for lane_offset in LANE_OFFSET.values():
            world_pos = obs["pos"] + np.array([0.0, lane_offset, 0.0])
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=obs["half_extents"],
                                                physicsClientId=client)
            vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=obs["half_extents"],
                                             rgbaColor=[0.6, 0.15, 0.15, 1.0],
                                             physicsClientId=client)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape,
                               baseVisualShapeIndex=vis_shape,
                               basePosition=world_pos.tolist(), physicsClientId=client)


def main():
    print("Obstacle-course demo | PID vs LQR vs Koopman-MPC")
    print("Lanes (left to right, -Y to +Y): PID | LQR | Koopman-MPC")
    print("Obstacles are static box pillars; the reference path bends around "
          "them via a repulsive field BEFORE any controller sees it.")

    start_pos = {c: reference_trajectory(0.0, LANE_OFFSET[c]) for c in CONTROLLER_ORDER}
    initial_xyzs = np.array([start_pos[c] for c in CONTROLLER_ORDER])
    initial_rpys = np.zeros((3, 3))

    env = CtrlAviary(
        drone_model=DRONE_MODEL,
        num_drones=3,
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys,
        physics=Physics.PYB,
        neighbourhood_radius=10,
        freq=SIM_FREQ,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=True,
        record=False,
        obstacles=False,  # we spawn our own obstacles below, not the env's default set
        user_debug_gui=False,
    )
    obs = env.reset()
    client = env.getPyBulletClient()
    drone_ids = {c: env.DRONE_IDS[i] for i, c in enumerate(CONTROLLER_ORDER)}

    p.resetDebugVisualizerCamera(
        cameraDistance=2.6, cameraYaw=0, cameraPitch=-55,
        cameraTargetPosition=[0, 0, 1], physicsClientId=client,
    )

    spawn_obstacles(client)

    # draw each lane's (already-bent) reference path so the detour is visible
    for c in CONTROLLER_ORDER:
        pts = [reference_trajectory(tt, LANE_OFFSET[c])
               for tt in np.arange(0, EPISODE_LEN_SEC, 0.05)]
        for a, b in zip(pts[:-1], pts[1:]):
            p.addUserDebugLine(a.tolist(), b.tolist(), PATH_COLOR[c],
                                lineWidth=1.5, physicsClientId=client)

    ctrls = {
        "pid": DSLPIDControl(drone_model=DRONE_MODEL),
        "lqr": LQRController(),
    }
    model = np.load("data/koopman_model.npz")
    A, B = model["A"], model["B"]
    ctrls["koopman_mpc"] = KoopmanMPC(A, B, Hp=10, Hc=5, pos_weight=120.0)
    prev_rpm_mpc = None

    n_ctrl_steps = int(EPISODE_LEN_SEC * CTRL_FREQ)
    label_ids = {}
    collision_flagged = {c: False for c in CONTROLLER_ORDER}

    wall_clock_start = time.time()
    for ctrl_step in range(n_ctrl_steps):
        t = ctrl_step / CTRL_FREQ

        rpm_cmds = {}
        for c in CONTROLLER_ORDER:
            state = obs[str(CONTROLLER_ORDER.index(c))]["state"]
            pos = state[0:3]
            target = reference_trajectory(t, LANE_OFFSET[c])

            if c == "pid":
                rpm_cmd, _, _ = ctrls["pid"].computeControlFromState(
                    control_timestep=AGGR_PHY_STEPS / SIM_FREQ,
                    state=state, target_pos=target, target_rpy=np.zeros(3),
                )
            elif c == "lqr":
                vel = state[10:13]
                quat = state[3:7]
                rotmat_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
                roll = np.arctan2(rotmat_mat[2, 1], rotmat_mat[2, 2])
                pitch = -np.arcsin(np.clip(rotmat_mat[2, 0], -1, 1))
                yaw = np.arctan2(rotmat_mat[1, 0], rotmat_mat[0, 0])
                ang_vel = state[13:16]
                state12 = np.hstack([pos, vel, [roll, pitch, yaw], ang_vel])
                rpm_cmd = ctrls["lqr"].compute(state12, target)
            else:  # koopman_mpc
                vel = state[10:13]
                quat = state[3:7]
                ang_vel = state[13:16]
                rotmat = np.array(p.getMatrixFromQuaternion(quat))
                raw_state = np.hstack([pos, vel, rotmat, ang_vel]).reshape(1, -1)
                psi0 = lift(raw_state)[0]
                ref_window = np.array([
                    reference_trajectory(t + k / CTRL_FREQ, LANE_OFFSET[c])
                    for k in range(ctrls["koopman_mpc"].Hp)
                ])
                rpm_cmd = ctrls["koopman_mpc"].solve(psi0, ref_window, prev_u=prev_rpm_mpc)
                prev_rpm_mpc = rpm_cmd

            rpm_cmds[str(CONTROLLER_ORDER.index(c))] = rpm_cmd

            # ground-truth collision check via PyBullet's own contact-point query,
            # rather than our own distance heuristic
            if not collision_flagged[c]:
                contacts = p.getContactPoints(bodyA=drone_ids[c], physicsClientId=client)
                for contact in contacts:
                    if contact[2] != 0:  # bodyB id 0 would be the ground plane; anything else is an obstacle
                        print(f"  [t={t:.2f}] {CONTROLLER_LABEL[c]} COLLIDED with an obstacle "
                              f"(body id {contact[2]})")
                        collision_flagged[c] = True
                        break

        obs, _, done, info = env.step(rpm_cmds)

        if ctrl_step % int(CTRL_FREQ * 0.5) == 0:
            print(f"t={t:.2f}  " + "  ".join(
                f"{c}:pos={np.round(obs[str(CONTROLLER_ORDER.index(c))]['state'][0:3], 3)}"
                for c in CONTROLLER_ORDER
            ))
            for c in CONTROLLER_ORDER:
                drone_pos, _ = p.getBasePositionAndOrientation(drone_ids[c], physicsClientId=client)
                if c in label_ids:
                    p.removeUserDebugItem(label_ids[c])
                label_ids[c] = p.addUserDebugText(
                    CONTROLLER_LABEL[c],
                    [drone_pos[0], drone_pos[1], drone_pos[2] + 0.15],
                    textColorRGB=[1, 1, 1], textSize=1.2,
                    physicsClientId=client,
                )

        target_wall_time = wall_clock_start + t * 1.15
        sleep_time = target_wall_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("Flight finished -- holding window open for 5s before closing...")
    time.sleep(5.0)
    env.close()
    print("Demo complete.")


if __name__ == "__main__":
    main()
