r"""
Side-by-side live demo: PID vs LQR vs Koopman-MPC, same reference trajectory,
same (optional) disturbance schedule, run simultaneously in one PyBullet GUI
window so they can be watched and compared directly.

Run from v2_stable_koopman_mpc/ with PYTHONPATH set to its subfolders, e.g.:

    cd v2_stable_koopman_mpc
    $env:PYTHONPATH = "$PWD\koopman_model;$PWD\mpc;$PWD\baselines"
    python ..\demo_flight_conditions.py                  # clean trajectory
    python ..\demo_flight_conditions.py payload           # with payload disturbance
    python ..\demo_flight_conditions.py wind              # with wind disturbance
    python ..\demo_flight_conditions.py payload+wind      # both

Drones are spawned side by side (offset along Y) purely for visual separation
in the GUI -- each still tracks the SAME reference trajectory in its own local
frame, so a direct visual comparison of tracking quality is meaningful.
"""
import sys
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
RPM_HOVER = 14468.0

# visual-only separation between the three drones in the GUI (meters, along Y)
LANE_OFFSET = {"pid": -0.8, "lqr": 0.0, "koopman_mpc": 0.8}
CONTROLLER_ORDER = ["pid", "lqr", "koopman_mpc"]
CONTROLLER_LABEL = {"pid": "PID", "lqr": "LQR", "koopman_mpc": "Koopman-MPC"}

PAYLOAD_START_SEC = 3.0
PAYLOAD_FRAC = 0.25
WIND_START_SEC = 3.0
WIND_DURATION_SEC = 2.0
WIND_FORCE_N = 0.02
WIND_DIR = np.array([1.0, 0.0, 0.0])


def reference_trajectory(t, amp=0.7, freq=0.15, z0=1.0):
    x = amp * np.sin(freq * t)
    y = amp * np.sin(2 * freq * t)
    return np.array([x, y, z0])


def parse_condition(argv):
    cond_name = argv[1] if len(argv) > 1 else "none"
    valid = {"none": {"payload": False, "wind": False},
              "payload": {"payload": True, "wind": False},
              "wind": {"payload": False, "wind": True},
              "payload+wind": {"payload": True, "wind": True}}
    if cond_name not in valid:
        print(f"Unknown condition '{cond_name}'. Choose from: {list(valid)}")
        sys.exit(1)
    return cond_name, valid[cond_name]


def main():
    cond_name, cond = parse_condition(sys.argv)
    print(f"Running side-by-side demo | condition: {cond_name}")
    print("Lanes (left to right, -Y to +Y): PID | LQR | Koopman-MPC")

    start_ref = reference_trajectory(0.0)
    initial_xyzs = np.array([
        start_ref + np.array([0.0, LANE_OFFSET[c], 0.0]) for c in CONTROLLER_ORDER
    ])
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
        obstacles=False,
        user_debug_gui=False,
    )
    obs = env.reset()
    client = env.getPyBulletClient()
    p.resetDebugVisualizerCamera(
        cameraDistance=2.2, cameraYaw=0, cameraPitch=-60,
        cameraTargetPosition=[0, 0, 1], physicsClientId=client,
    )

    # draw each lane's reference path as a thin line so it's visually obvious
    # what each drone is SUPPOSED to be tracking, not just where it ends up
    path_colors = {"pid": [0, 1, 0], "lqr": [1, 0.6, 0], "koopman_mpc": [0, 0.6, 1]}
    for c in CONTROLLER_ORDER:
        pts = [reference_trajectory(tt) + np.array([0.0, LANE_OFFSET[c], 0.0])
               for tt in np.arange(0, EPISODE_LEN_SEC, 0.1)]
        for a, b in zip(pts[:-1], pts[1:]):
            p.addUserDebugLine(a.tolist(), b.tolist(), path_colors[c],
                                lineWidth=1.5, physicsClientId=client)
    drone_ids = {c: env.DRONE_IDS[i] for i, c in enumerate(CONTROLLER_ORDER)}
    nominal_mass = {c: p.getDynamicsInfo(drone_ids[c], -1, physicsClientId=client)[0]
                     for c in CONTROLLER_ORDER}

    # --- controller instances, one per lane ---
    ctrls = {
        "pid": DSLPIDControl(drone_model=DRONE_MODEL),
        "lqr": LQRController(),
    }
    model = np.load("data/koopman_model.npz")
    A, B = model["A"], model["B"]
    ctrls["koopman_mpc"] = KoopmanMPC(A, B, Hp=10, Hc=5, pos_weight=120.0)
    prev_rpm_mpc = None

    applied_payload = {c: False for c in CONTROLLER_ORDER}
    n_ctrl_steps = int(EPISODE_LEN_SEC * CTRL_FREQ)

    # --- GUI text labels above each drone, refreshed periodically ---
    label_ids = {}

    wall_clock_start = time.time()
    for ctrl_step in range(n_ctrl_steps):
        t = ctrl_step / CTRL_FREQ

        rpm_cmds = {}
        for c in CONTROLLER_ORDER:
            drone_id = drone_ids[c]

            # --- disturbance injection, identical schedule for every lane ---
            if cond["payload"] and t >= PAYLOAD_START_SEC and not applied_payload[c]:
                p.changeDynamics(drone_id, -1,
                                  mass=nominal_mass[c] * (1 + PAYLOAD_FRAC),
                                  physicsClientId=client)
                applied_payload[c] = True
            if cond["wind"] and WIND_START_SEC <= t < WIND_START_SEC + WIND_DURATION_SEC:
                p.applyExternalForce(drone_id, -1,
                                      forceObj=(WIND_FORCE_N * WIND_DIR).tolist(),
                                      posObj=[0, 0, 0], flags=p.LINK_FRAME,
                                      physicsClientId=client)

            state = obs[str(CONTROLLER_ORDER.index(c))]["state"]
            pos = state[0:3]
            lane_target = reference_trajectory(t) + np.array([0.0, LANE_OFFSET[c], 0.0])

            if c == "pid":
                rpm_cmd, _, _ = ctrls["pid"].computeControlFromState(
                    control_timestep=AGGR_PHY_STEPS / SIM_FREQ,
                    state=state, target_pos=lane_target, target_rpy=np.zeros(3),
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
                rpm_cmd = ctrls["lqr"].compute(state12, lane_target)
            else:  # koopman_mpc
                vel = state[10:13]
                quat = state[3:7]
                ang_vel = state[13:16]
                rotmat = np.array(p.getMatrixFromQuaternion(quat))
                raw_state = np.hstack([pos, vel, rotmat, ang_vel]).reshape(1, -1)
                psi0 = lift(raw_state)[0]
                ref_window = np.array([
                    reference_trajectory(t + k / CTRL_FREQ) + np.array([0.0, LANE_OFFSET[c], 0.0])
                    for k in range(ctrls["koopman_mpc"].Hp)
                ])
                rpm_cmd = ctrls["koopman_mpc"].solve(psi0, ref_window, prev_u=prev_rpm_mpc)
                prev_rpm_mpc = rpm_cmd

            rpm_cmds[str(CONTROLLER_ORDER.index(c))] = rpm_cmd

        obs, _, done, info = env.step(rpm_cmds)

        if ctrl_step % int(CTRL_FREQ * 0.5) == 0:
            print(f"t={t:.2f}  " + "  ".join(
                f"{c}:pos={np.round(obs[str(CONTROLLER_ORDER.index(c))]['state'][0:3], 3)}"
                for c in CONTROLLER_ORDER
            ))

        # refresh overhead labels every ~0.5s so the GUI stays legible
        if ctrl_step % int(CTRL_FREQ * 0.5) == 0:
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

        # pace slightly slower than strict real time (1.15x) so the GUI render
        # thread has headroom and never falls behind the physics stepping
        target_wall_time = wall_clock_start + t * 1.15
        sleep_time = target_wall_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("Flight finished -- holding window open for 5s so the GUI can catch up "
          "and the final state is visible before closing...")
    time.sleep(5.0)
    env.close()
    print("Demo complete.")


if __name__ == "__main__":
    main()
