"""
Final comparison: PID vs Koopman-MPC across a fixed disturbance test matrix.

Each controller is run through the SAME reference trajectory and SAME
disturbance schedule (payload attach, wind gust, both, or neither) so the
comparison is fair. This produces the core results table for the project.
"""

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from build_edmd_dataset import lift
from koopman_mpc import KoopmanMPC

SIM_FREQ = 240
CTRL_FREQ = 48
AGGR_PHY_STEPS = int(SIM_FREQ / CTRL_FREQ)
EPISODE_LEN_SEC = 8.0
DRONE_MODEL = DroneModel.CF2X

# --- fixed disturbance test matrix (same for every controller) ---
DISTURBANCE_CONDITIONS = {
    "none": {"payload": False, "wind": False},
    "payload": {"payload": True, "wind": False},
    "wind": {"payload": False, "wind": True},
    "payload+wind": {"payload": True, "wind": True},
}
PAYLOAD_START_SEC = 3.0
PAYLOAD_FRAC = 0.25
WIND_START_SEC = 3.0
WIND_DURATION_SEC = 2.0
WIND_FORCE_N = 0.02
WIND_DIR = np.array([1.0, 0.0, 0.0])  # fixed direction for repeatability


def reference_trajectory(t, amp=0.4, freq=0.15, z0=1.0):
    x = amp * np.sin(freq * t)
    y = amp * np.sin(2 * freq * t)
    return np.array([x, y, z0])


def run_episode(controller_type, condition_name):
    cond = DISTURBANCE_CONDITIONS[condition_name]
    start_pos = reference_trajectory(0.0)

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
    obs = env.reset()
    client = env.getPyBulletClient()
    drone_id = env.DRONE_IDS[0]
    nominal_mass = p.getDynamicsInfo(drone_id, -1, physicsClientId=client)[0]

    if controller_type == "pid":
        ctrl = DSLPIDControl(drone_model=DRONE_MODEL)
    elif controller_type in ("koopman_mpc", "koopman_mpc_adaptive"):
        model = np.load("data/koopman_model.npz")
        A, B = model["A"], model["B"]
        ctrl = KoopmanMPC(A, B, Hp=10, Hc=5, pos_weight=120.0)
        if controller_type == "koopman_mpc_adaptive":
            # a second, more aggressive controller instance used only when
            # tracking error spikes (a proxy for "something disturbed us"),
            # with a hard cap on how long aggressive mode can run -- our
            # standalone test showed this weight set stays stable for at
            # least ~2s of continuous use before instability starts to build,
            # so we cap aggressive mode well under that
            ctrl_aggressive = KoopmanMPC(A, B, Hp=10, Hc=5, pos_weight=250.0,
                                          du_weight=0.3, u_weight=1e-4)
    else:
        raise ValueError(controller_type)

    ERROR_TRIGGER_THRESHOLD = 0.55  # m; well above the ~0.42m peak lag seen
                                     # in baseline (undisturbed) tracking, so
                                     # this only fires for genuine anomalies
    MAX_AGGRESSIVE_DURATION = 1.5   # s; hard safety cap
    COOLDOWN_DURATION = 2.0         # s; after aggressive mode ends, must wait
                                     # this long before it can retrigger --
                                     # prevents the retrigger loop we saw
    aggressive_until = -1.0
    cooldown_until = -1.0

    n_ctrl_steps = int(EPISODE_LEN_SEC * CTRL_FREQ)
    applied_payload = False
    prev_rpm = None

    log = {"t": [], "pos": [], "ref_pos": [], "rpm": []}

    for ctrl_step in range(n_ctrl_steps):
        t = ctrl_step / CTRL_FREQ

        # --- disturbance injection (identical schedule for every controller) ---
        if cond["payload"] and t >= PAYLOAD_START_SEC and not applied_payload:
            p.changeDynamics(drone_id, -1, mass=nominal_mass * (1 + PAYLOAD_FRAC),
                              physicsClientId=client)
            applied_payload = True

        if cond["wind"] and WIND_START_SEC <= t < WIND_START_SEC + WIND_DURATION_SEC:
            p.applyExternalForce(drone_id, -1,
                                  forceObj=(WIND_FORCE_N * WIND_DIR).tolist(),
                                  posObj=[0, 0, 0], flags=p.LINK_FRAME,
                                  physicsClientId=client)

        # --- control computation ---
        state = obs["0"]["state"]
        pos = state[0:3]
        target_pos = reference_trajectory(t)

        if controller_type == "pid":
            rpm_cmd, _, _ = ctrl.computeControlFromState(
                control_timestep=AGGR_PHY_STEPS / SIM_FREQ,
                state=state, target_pos=target_pos, target_rpy=np.zeros(3),
            )
        else:  # koopman_mpc or koopman_mpc_adaptive
            vel = state[10:13]
            quat = state[3:7]
            ang_vel = state[13:16]
            rotmat = np.array(p.getMatrixFromQuaternion(quat))
            raw_state = np.hstack([pos, vel, rotmat, ang_vel]).reshape(1, -1)
            psi0 = lift(raw_state)[0]

            active_ctrl = ctrl
            if controller_type == "koopman_mpc_adaptive":
                current_err = np.linalg.norm(pos - target_pos)
                if (current_err > ERROR_TRIGGER_THRESHOLD
                        and t > aggressive_until and t > cooldown_until):
                    aggressive_until = t + MAX_AGGRESSIVE_DURATION
                    cooldown_until = aggressive_until + COOLDOWN_DURATION
                if t < aggressive_until:
                    active_ctrl = ctrl_aggressive

            ref_window = np.array([reference_trajectory(t + k / CTRL_FREQ)
                                    for k in range(active_ctrl.Hp)])
            rpm_cmd = active_ctrl.solve(psi0, ref_window, prev_u=prev_rpm)
            prev_rpm = rpm_cmd

        obs, _, done, info = env.step({"0": rpm_cmd})

        log["t"].append(t)
        log["pos"].append(pos.copy())
        log["ref_pos"].append(target_pos.copy())
        log["rpm"].append(np.array(rpm_cmd).copy())

    env.close()
    for key in log:
        log[key] = np.array(log[key])
    return log


def compute_metrics(log):
    t = log["t"]
    pos_err = np.linalg.norm(log["pos"] - log["ref_pos"], axis=1)

    overall_rmse = np.sqrt(np.mean(pos_err ** 2))
    max_err = pos_err.max()

    # recovery time: after disturbance ends (max of payload/wind windows),
    # how long until error drops back under a threshold and stays there
    disturbance_end = max(PAYLOAD_START_SEC, WIND_START_SEC + WIND_DURATION_SEC)
    post_mask = t >= disturbance_end
    recovery_time = None
    threshold = 0.15
    if post_mask.any():
        post_err = pos_err[post_mask]
        post_t = t[post_mask]
        below = post_err < threshold
        for i in range(len(below)):
            if below[i:].all():
                recovery_time = post_t[i] - disturbance_end
                break

    # control effort: mean squared RPM deviation from hover
    control_effort = np.mean((log["rpm"] - 14468.0) ** 2)

    return {
        "rmse": overall_rmse,
        "max_err": max_err,
        "recovery_time": recovery_time,
        "control_effort": control_effort,
    }


if __name__ == "__main__":
    results = {}
    for controller in ["pid", "koopman_mpc"]:
        for condition in DISTURBANCE_CONDITIONS:
            print(f"Running {controller} / {condition} ...")
            log = run_episode(controller, condition)
            metrics = compute_metrics(log)
            results[(controller, condition)] = metrics
            np.savez(f"data/comparison_{controller}_{condition}.npz", **log)

    print("\n\n=== COMPARISON RESULTS ===")
    print(f"{'controller':<14} {'condition':<14} {'RMSE (m)':>10} {'max err (m)':>12} "
          f"{'recovery (s)':>13} {'ctrl effort':>16}")
    for (controller, condition), m in results.items():
        rec = f"{m['recovery_time']:.2f}" if m['recovery_time'] is not None else "N/A"
        print(f"{controller:<14} {condition:<14} {m['rmse']:>10.4f} {m['max_err']:>12.4f} "
              f"{rec:>13} {m['control_effort']:>16.6f}")

    print("\n\n=== RELATIVE DEGRADATION (the core question this project asks) ===")
    print("How much worse does each controller get, percentage-wise, going from")
    print("clean conditions to the worst disturbance? This matters more than raw")
    print("tracking tightness, since PID and Koopman-MPC may simply be tuned to")
    print("different absolute aggressiveness.\n")
    print(f"{'controller':<14} {'clean RMSE':>12} {'worst RMSE':>12} {'degradation':>14}")
    for controller in ["pid", "koopman_mpc"]:
        clean_rmse = results[(controller, "none")]["rmse"]
        worst_rmse = results[(controller, "payload+wind")]["rmse"]
        degradation_pct = 100 * (worst_rmse - clean_rmse) / clean_rmse
        print(f"{controller:<14} {clean_rmse:>12.4f} {worst_rmse:>12.4f} {degradation_pct:>13.1f}%")
