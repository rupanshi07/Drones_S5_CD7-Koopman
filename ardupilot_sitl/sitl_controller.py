"""
sitl_controller.py
------------------
Main Koopman-MPC / LQR / PID simulation loop for ArduPilot SITL.

This is the ACTUAL SIMULATION -- it connects to a live ArduCopter SITL
process (via Mission Planner on Windows), reads real MAVLink telemetry,
and sends real SET_ATTITUDE_TARGET commands back to the simulated drone.

Controller parameters are kept IDENTICAL to Phase 2 (v2_stable_koopman_mpc):
  - Koopman-MPC: Hp=10, Hc=5, pos_weight=120.0, vel_weight=1.0,
                 du_weight=1.0, u_weight=1e-3
  - Disturbance schedule: payload +25% at 30% of episode, wind as SIM params
  - Reference trajectory: same Lissajous x=0.4*sin(0.15t), y=0.4*sin(0.3t)

Usage:
  python sitl_controller.py --controller koopman [--scenario clean|wind|payload|combined]
  python sitl_controller.py --controller lqr     [--scenario ...]
  python sitl_controller.py --controller pid      [--scenario ...]

Options:
  --host HOST        SITL TCP host (default: 127.0.0.1)
  --port PORT        SITL TCP port (default: 5760)
  --duration SEC     Episode duration in seconds (default: 15)
  --ctrl-freq HZ     Control loop frequency (default: 20)
  --model PATH       Path to koopman_model.npz (default: data/koopman_model.npz)
  --scenario NAME    Disturbance scenario: clean, wind, payload, combined
  --no-gui           Don't open live plot window (recommended for first run)
  --save PATH        Override result save path

Requirements:
  - ArduCopter SITL running (Mission Planner -> Simulation -> Run SITL)
  - Drone armed and in GUIDED mode (run setup_vehicle.py first)
"""

import argparse
import time
import sys
import os
import numpy as np
import threading
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Our modules
from mavlink_interface import MAVLinkInterface
from koopman_mpc_ardupilot import KoopmanMPCArduPilot
from lqr_controller_ardupilot import LQRControllerArduPilot, PIDControllerArduPilot
from disturbance_injector import DisturbanceInjector, SCENARIOS


# =========================================================================== #
# Reference trajectory (same as Phase 2 closed-loop test)
# =========================================================================== #

def reference_trajectory(t, amp=0.4, freq=0.15, z0=1.0):
    """
    Lissajous-like reference in ENU.
    x = amp * sin(freq * t)
    y = amp * sin(2 * freq * t)
    z = z0 (constant altitude)
    """
    x = amp * np.sin(freq * t)
    y = amp * np.sin(2 * freq * t)
    z = z0
    return np.array([x, y, z])


def reference_velocity(t, amp=0.4, freq=0.15):
    """Analytical derivative of reference_trajectory (x, y only; z=const)."""
    vx = amp * freq * np.cos(freq * t)
    vy = amp * 2 * freq * np.cos(2 * freq * t)
    vz = 0.0
    return np.array([vx, vy, vz])


# =========================================================================== #
# Live plot (optional, can be disabled with --no-gui)
# =========================================================================== #

class LivePlot:
    def __init__(self, duration):
        matplotlib.use("TkAgg")   # interactive backend
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.suptitle("ArduPilot SITL – Koopman-MPC Real-Time Tracking",
                           fontsize=12, fontweight='bold')
        self._duration = duration
        self._t   = []
        self._pos = []
        self._ref = []
        self._err = []
        self._lock = threading.Lock()
        self._closed = False

        labels = ["X (ENU) [m]", "Y (ENU) [m]", "Z (ENU) [m]"]
        for ax, label in zip(self.axes, labels):
            ax.set_ylabel(label)
            ax.set_xlim(0, duration)
            ax.grid(True, alpha=0.4)
        self.axes[-1].set_xlabel("Time [s]")

    def update(self, t, pos, ref):
        with self._lock:
            self._t.append(t)
            self._pos.append(pos.copy())
            self._ref.append(ref.copy())
            self._err.append(np.linalg.norm(pos - ref))

    def refresh(self):
        """Called from main thread to refresh the plot."""
        if self._closed:
            return
        with self._lock:
            if len(self._t) < 2:
                return
            t   = np.array(self._t)
            pos = np.array(self._pos)
            ref = np.array(self._ref)

        for i, ax in enumerate(self.axes):
            ax.cla()
            ax.plot(t, pos[:, i], 'b-', linewidth=1.5, label='Actual')
            ax.plot(t, ref[:, i], 'r--', linewidth=1.5, label='Reference')
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.4)
        self.axes[-1].set_xlabel("Time [s]")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        self._closed = True
        try:
            if plt.fignum_exists(self.fig.number):
                plt.close(self.fig)
        except Exception:
            pass


# =========================================================================== #
# Main simulation loop
# =========================================================================== #

def run_simulation(args):
    # ----------------------------------------------------------------------- #
    # Controller and scenario setup
    # ----------------------------------------------------------------------- #
    scenario_cfg = SCENARIOS.get(args.scenario, SCENARIOS["clean"])
    print(f"\n{'='*60}")
    print(f"  Controller : {args.controller.upper()}")
    print(f"  Scenario   : {args.scenario} -- {scenario_cfg['description']}")
    print(f"  Duration   : {args.duration} s  |  Freq: {args.ctrl_freq} Hz")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------------------- #
    # MAVLink connection
    # ----------------------------------------------------------------------- #
    mav = MAVLinkInterface(host=args.host, port=args.port)
    try:
        mav.connect(timeout=20)
    except ConnectionError as e:
        print(f"\nERROR: {e}")
        print("Please make sure SITL is running in Mission Planner.")
        sys.exit(1)

    # Ensure mode is GUIDED
    try:
        mav.set_mode("GUIDED")
    except Exception as e:
        print(f"[Warning] Mode set error: {e}")

    # Check arming status
    if not mav.is_armed:
        print("\nDrone is not armed via script. Attempting to arm ...")
        try:
            mav.arm(force=True)
            time.sleep(1.0)
        except Exception as e:
            print(f"[Warning] Arming command encountered: {e}")
            print(">> If drone is already armed and flying in Mission Planner, continuing...")
    else:
        print("\nDrone is already ARMED! Proceeding with controller execution...")

    # ----------------------------------------------------------------------- #
    # Controller
    # ----------------------------------------------------------------------- #
    ctrl_dt = 1.0 / args.ctrl_freq
    model_path = Path(args.model)

    if args.controller == "koopman":
        if not model_path.exists():
            print(f"\nERROR: Koopman model not found at {model_path}")
            print("Copy data/koopman_model.npz from Phase 2 to ardupilot_sitl/data/")
            print("Or run: python copy_model.py")
            sys.exit(1)
        # Parameters IDENTICAL to Phase-2 run_comparison.py / run_mpc_closed_loop.py
        ctrl = KoopmanMPCArduPilot(
            str(model_path), Hp=10, Hc=5,
            pos_weight=120.0, vel_weight=1.0,
            du_weight=1.0, u_weight=1e-3
        )
        ctrl_name = "Koopman-MPC"
    elif args.controller == "lqr":
        ctrl = LQRControllerArduPilot()
        ctrl_name = "LQR"
    else:  # pid
        ctrl = PIDControllerArduPilot(dt=ctrl_dt)
        ctrl_name = "PID"

    # ----------------------------------------------------------------------- #
    # Disturbance injector
    # ----------------------------------------------------------------------- #
    dist = DisturbanceInjector(mav)
    if scenario_cfg["wind_speed"] > 0:
        dist.set_wind(scenario_cfg["wind_speed"], scenario_cfg["wind_dir"])

    # ----------------------------------------------------------------------- #
    # Live plot
    # ----------------------------------------------------------------------- #
    live_plot = None
    if not args.no_gui:
        try:
            plt.ion()
            live_plot = LivePlot(args.duration)
            plt.show(block=False)
        except Exception as e:
            print(f"[Warning] Live plot init failed: {e}. Continuing without GUI.")
            live_plot = None

    # ----------------------------------------------------------------------- #
    # Flight log
    # ----------------------------------------------------------------------- #
    log = {
        "t":        [],
        "pos":      [],
        "vel":      [],
        "ref_pos":  [],
        "roll":     [],
        "pitch":    [],
        "thrust":   [],
        "pos_err":  [],
        "disturbance": [],
    }

    # ----------------------------------------------------------------------- #
    # Main control loop
    # ----------------------------------------------------------------------- #
    print(f"\n[Sim] Starting {ctrl_name} simulation ...")
    print(f"[Sim] Watch the drone in Mission Planner!\n")

    n_steps = int(args.duration * args.ctrl_freq)
    t_start = time.time()
    payload_applied = False

    try:
        for step in range(n_steps):
            t_loop_start = time.time()
            t = step * ctrl_dt

            # --- Disturbance injection (time-based) ---
            if scenario_cfg["payload_onset_frac"] is not None:
                onset_t = args.duration * scenario_cfg["payload_onset_frac"]
                if t >= onset_t and not payload_applied:
                    dist.apply_payload(scenario_cfg["payload_frac"])
                    payload_applied = True
                    print(f"[Sim] t={t:.1f}s: PAYLOAD APPLIED "
                          f"(+{scenario_cfg['payload_frac']*100:.0f}% mass)")

            # --- Read state ---
            pos, vel, rotmat, ang_vel = mav.get_state_enu()

            # --- Reference ---
            ref_pos = reference_trajectory(t)
            ref_vel = reference_velocity(t)

            # --- Compute control ---
            if args.controller == "koopman":
                # Build Hp-step reference window
                ref_window = np.array([
                    reference_trajectory(t + k * ctrl_dt) for k in range(ctrl.Hp)
                ])
                roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, _rpm = ctrl.solve(
                    pos, vel, rotmat, ang_vel, ref_window
                )
            elif args.controller == "lqr":
                roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = ctrl.compute(
                    pos, vel, ref_pos, ref_vel
                )
            else:  # pid
                roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = ctrl.compute(
                    pos, vel, ref_pos
                )

            # Payload compensation: boost thrust slightly to fight extra weight
            if dist.payload_active:
                thrust_cmd = np.clip(thrust_cmd * (1.0 + dist.payload_frac * 0.4),
                                     0.0, 1.0)

            # --- Send command ---
            mav.send_attitude_target(
                thrust=thrust_cmd,
                roll=roll_cmd,
                pitch=pitch_cmd,
                yaw=yaw_cmd
            )

            # --- Log ---
            pos_err = np.linalg.norm(pos - ref_pos)
            log["t"].append(t)
            log["pos"].append(pos.copy())
            log["vel"].append(vel.copy())
            log["ref_pos"].append(ref_pos.copy())
            log["roll"].append(roll_cmd)
            log["pitch"].append(pitch_cmd)
            log["thrust"].append(thrust_cmd)
            log["pos_err"].append(pos_err)
            log["disturbance"].append(dist.get_state_dict())

            # --- Live plot refresh (every 10 steps to reduce GUI overhead) ---
            if live_plot and step % 10 == 0:
                live_plot.update(t, pos, ref_pos)
                live_plot.refresh()

            # --- Console status ---
            if step % 20 == 0 or step < 5:
                dist_str = ""
                if dist.wind_active:
                    dist_str += f" [WIND {dist.wind_speed:.1f}m/s]"
                if dist.payload_active:
                    dist_str += f" [PAYLOAD +{dist.payload_frac*100:.0f}%]"
                print(f"t={t:5.1f}s  pos={pos.round(3)}  ref={ref_pos.round(3)}"
                      f"  err={pos_err:.3f}m  T={thrust_cmd:.2f}{dist_str}")

            # --- Timing ---
            elapsed = time.time() - t_loop_start
            sleep_t = ctrl_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            elif step % 50 == 0:
                print(f"  [WARN] Control loop overran by {-sleep_t*1000:.1f}ms at t={t:.1f}s")

    except KeyboardInterrupt:
        print("\n[Sim] Interrupted by user.")

    # ----------------------------------------------------------------------- #
    # Post-flight: safe hover then disarm
    # ----------------------------------------------------------------------- #
    print("\n[Sim] Episode complete. Hovering for 2 seconds ...")
    dist.clear_wind()
    t_end = time.time()
    while time.time() - t_end < 2.0:
        mav.send_attitude_target(thrust=0.5, roll=0.0, pitch=0.0, yaw=0.0)
        time.sleep(0.05)

    print("[Sim] Disarming ...")
    mav.disarm()
    time.sleep(0.5)

    # ----------------------------------------------------------------------- #
    # Save results
    # ----------------------------------------------------------------------- #
    for key in ["t", "pos", "vel", "ref_pos", "roll", "pitch", "thrust", "pos_err"]:
        log[key] = np.array(log[key])

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.save) if args.save else \
                out_dir / f"{args.controller}_{args.scenario}_{ts}.npz"

    # Remove non-array key before saving
    log_save = {k: v for k, v in log.items() if k != "disturbance"}
    log_save["controller"] = np.array([args.controller])
    log_save["scenario"]   = np.array([args.scenario])
    np.savez(str(save_path), **log_save)
    print(f"\n[Sim] Results saved to {save_path}")

    # ----------------------------------------------------------------------- #
    # Summary metrics
    # ----------------------------------------------------------------------- #
    pos_err = log["pos_err"]
    rmse = np.sqrt(np.mean(pos_err**2))
    max_err = pos_err.max()
    print(f"\n{'='*50}")
    print(f"  Controller: {ctrl_name}  |  Scenario: {args.scenario}")
    print(f"  Tracking RMSE : {rmse:.4f} m")
    print(f"  Max error     : {max_err:.4f} m")
    print(f"  Mean thrust   : {log['thrust'].mean():.3f}")
    print(f"{'='*50}\n")

    mav.close()

    if live_plot:
        live_plot.close()

    return {"rmse": rmse, "max_err": max_err, "log_path": str(save_path)}


# =========================================================================== #
# Entry point
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Koopman-MPC / LQR / PID simulation on ArduPilot SITL (Windows)"
    )
    parser.add_argument("--controller", choices=["koopman", "lqr", "pid"],
                        default="koopman", help="Controller to run")
    parser.add_argument("--scenario",
                        choices=["clean", "wind", "payload", "combined"],
                        default="clean", help="Disturbance scenario")
    parser.add_argument("--host", default="127.0.0.1",
                        help="ArduPilot SITL TCP host")
    parser.add_argument("--port", type=int, default=5760,
                        help="ArduPilot SITL TCP port")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Episode duration in seconds")
    parser.add_argument("--ctrl-freq", type=float, default=20.0,
                        help="Control loop frequency in Hz")
    parser.add_argument("--model", default="data/koopman_model.npz",
                        help="Path to Koopman model .npz")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable live plot window")
    parser.add_argument("--save", default=None,
                        help="Override result save path")
    args = parser.parse_args()
    run_simulation(args)
