"""
setup_vehicle.py
----------------
Setup script for ArduCopter SITL before running the Koopman-MPC simulation.

Run this ONCE after starting SITL in Mission Planner and before running
sitl_controller.py.

Usage:
  python setup_vehicle.py [--host 127.0.0.1] [--port 5760]
"""

import argparse
import time
import sys

from mavlink_interface import MAVLinkInterface


def setup_vehicle(host="127.0.0.1", port=5760, use_udp=False):
    print("=" * 60)
    print("   ArduPilot SITL Setup - Koopman-MPC Project")
    print("=" * 60)
    print()
    if use_udp:
        print(f"Connecting via UDP to {host}:{port} (Mission Planner forwarded port)")
    else:
        print("Make sure Mission Planner SITL is running!")
        print("(Simulation tab -> Multirotor -> Run SITL)")
    print()

    mav = MAVLinkInterface(host=host, port=port, use_udp=use_udp)

    try:
        mav.connect(timeout=30)
    except ConnectionError as e:
        print(f"\nERROR: {e}")
        print("\nPlease start ArduCopter SITL in Mission Planner first.")
        sys.exit(1)

    print(f"\nConnected! target_sys={mav.conn.target_system}  "
          f"Mode: {mav.mode}, Armed: {mav.is_armed}")

    # ----------------------------------------------------------------------- #
    # Configure parameters for simulation
    # ----------------------------------------------------------------------- #
    print("\n[Setup] Configuring ArduCopter parameters ...")

    params = {
        # Bypass pre-arm safety checks for SITL
        "ARMING_CHECK": 0,
        "ARMING_SKIPCHK": 1,
        "FS_THR_ENABLE": 0,
        "FS_BATT_ENABLE": 0,
        "FS_GCS_ENABLE": 0,
        "FS_CRASH_CHECK": 0,

        # GUIDED mode settings (bit 3 = allow SET_ATTITUDE_TARGET in GUIDED)
        "GUID_OPTIONS": 8,

        # Attitude controller rates (fast response for 20Hz commands)
        "ATC_ACCEL_R_MAX": 110000,
        "ATC_ACCEL_P_MAX": 110000,
        "ATC_ACCEL_Y_MAX": 27000,
        "ATC_RATE_R_MAX":  220,
        "ATC_RATE_P_MAX":  220,

        # Position controller
        "WPNAV_SPEED": 500,
        "WPNAV_ACCEL": 250,

        # Hover throttle calibration (0.5 = 50% throttle at hover)
        "MOT_THST_HOVER": 0.50,
        "MOT_SPIN_MIN": 0.15,
        "MOT_SPIN_ARM": 0.10,

        # Wind simulation (calm default; changed at runtime by disturbance_injector)
        "SIM_WIND_SPD":  0.0,
        "SIM_WIND_DIR":  0.0,
        "SIM_WIND_TURB": 0.0,

        # RC override time (allows external control for longer)
        "RC_OVERRIDE_TIME": 10,
    }

    for param_name, value in params.items():
        try:
            mav.set_param(param_name, value)
            time.sleep(0.1)  # delay so each param is applied
        except Exception as e:
            print(f"  Note: Parameter {param_name} skipped: {e}")

    print("[Setup] Parameters configured.")
    print("[Setup] Waiting 2s for params to take effect ...")
    time.sleep(2.0)

    # ----------------------------------------------------------------------- #
    # Mode switch & Arming Sequence
    # ----------------------------------------------------------------------- #
    # Arm in STABILIZE first (easiest mode to arm in SITL)
    print("\n[Setup] Switching to STABILIZE for arming ...")
    try:
        mav.set_mode("STABILIZE")
        time.sleep(1.0)
    except Exception:
        pass

    print("\n[Setup] Arming vehicle ...")
    success = mav.arm(force=True)
    if not success:
        print("[Setup] Retrying arm with force=True ...")
        time.sleep(1.0)
        success = mav.arm(force=True)

    if not success:
        print("\nERROR: Could not arm. Check ArduPilot pre-arm messages in Mission Planner.")
        mav.close()
        sys.exit(1)

    print(f"  Armed: {mav.is_armed}")
    time.sleep(0.5)

    # Switch to GUIDED after arming (preferred for external SET_ATTITUDE_TARGET)
    # Try GUIDED first (works when EKF has a position estimate via baro)
    print("\n[Setup] Setting mode to GUIDED ...")
    guided_set = False
    for mode_name in ["GUIDED", "GUIDED_NOGPS"]:
        try:
            mav.set_mode(mode_name)
            time.sleep(1.0)
            current = mav.mode
            if "GUIDED" in current.upper():
                print(f"  Mode set to: {current}")
                guided_set = True
                break
        except Exception as e:
            print(f"  {mode_name} failed: {e}")

    if not guided_set:
        print("[Setup] WARNING: Could not set GUIDED mode. "
              "Ensure EKF is healthy in Mission Planner.")

    # ----------------------------------------------------------------------- #
    # Initial Takeoff / Hover
    # ----------------------------------------------------------------------- #
    print("\n[Setup] Sending takeoff thrust to get airborne ...")
    # Ramp up from 0.5 (hover) to 0.58 briefly, then settle
    t_start = time.time()
    while time.time() - t_start < 3.0:
        mav.send_attitude_target(thrust=0.58, roll=0.0, pitch=0.0, yaw=0.0)
        time.sleep(0.05)

    print("  Settling at hover thrust=0.5 for 2 seconds ...")
    t_start = time.time()
    while time.time() - t_start < 2.0:
        mav.send_attitude_target(thrust=0.50, roll=0.0, pitch=0.0, yaw=0.0)
        time.sleep(0.05)

    try:
        pos, vel, rotmat, ang_vel = mav.get_state_enu()
        print(f"\n  Current position (ENU): {pos.round(3)} m")
        print(f"  Altitude: {pos[2]:.2f} m")
    except Exception:
        pass

    print("\n[Setup] Vehicle setup complete! Ready for controllers.")
    print("  -> Now run:  python sitl_controller.py --controller koopman")
    mav.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup ArduPilot SITL for Koopman-MPC")
    parser.add_argument("--host", default="127.0.0.1", help="SITL host")
    parser.add_argument("--port", type=int, default=5760, help="SITL port")
    parser.add_argument("--udp", action="store_true",
                        help="Connect via UDP (use after adding UDP output in "
                             "Mission Planner Ctrl+F -> MavLink -> port 14551)")
    args = parser.parse_args()
    setup_vehicle(args.host, args.port, use_udp=args.udp)