"""
test_controllers.py
-------------------
Unit test / dry-run of all controller modules WITHOUT needing SITL running.

Verifies:
  1. mavlink_interface frame conversion math
  2. KoopmanMPCArduPilot lifting function
  3. LQR and PID controller math
  4. Disturbance injector state management

This can be run at any time to confirm the code is correct before connecting
to a real SITL instance.

Usage:
  python test_controllers.py
"""

import numpy as np
import sys
from pathlib import Path


def test_frame_conversions():
    print("[Test] Frame conversion utilities ...")
    from mavlink_interface import (ned_to_enu, enu_to_ned, quat_to_rotmat_wxyz,
                                   ned_rotmat_to_enu, euler_to_quat_wxyz,
                                   rotmat_to_euler, compute_desired_attitude_from_thrust_vector)

    # NED <-> ENU roundtrip
    v_ned = np.array([1.0, 2.0, -3.0])
    v_enu = ned_to_enu(v_ned)
    v_ned2 = enu_to_ned(v_enu)
    assert np.allclose(v_ned, v_ned2), "NED->ENU->NED roundtrip FAILED"
    print("  NED<->ENU roundtrip: OK")

    # Quaternion -> rotation matrix (identity quaternion = identity matrix)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    R = quat_to_rotmat_wxyz(q_id)
    assert np.allclose(R, np.eye(3)), "Identity quaternion -> identity matrix FAILED"
    print("  Identity quaternion -> R_identity: OK")

    # Euler -> quaternion (zero angles)
    q_zero = euler_to_quat_wxyz(0, 0, 0)
    assert np.allclose(q_zero, [1, 0, 0, 0]), "Zero euler -> identity quat FAILED"
    print("  Zero Euler -> identity quaternion: OK")

    # Attitude from thrust vector (vertical = hover)
    roll, pitch, thrust = compute_desired_attitude_from_thrust_vector([0, 0, 9.81])
    assert abs(roll) < 0.01 and abs(pitch) < 0.01, \
        f"Vertical thrust should give ~zero tilt, got roll={roll:.3f}, pitch={pitch:.3f}"
    assert 0.3 < thrust < 0.7, f"Hover thrust should be ~0.5, got {thrust:.3f}"
    print("  Vertical thrust -> near-zero tilt: OK")

    print("[Test] Frame conversions: ALL PASSED\n")


def test_lifting_function():
    print("[Test] Koopman lifting function ...")
    from koopman_mpc_ardupilot import lift

    # Create a dummy raw state row: [pos(3), vel(3), rotmat(9), ang_vel(3)] = 18-dim
    raw = np.random.randn(5, 18)  # 5 samples
    lifted = lift(raw)
    print(f"  lift({raw.shape}) -> {lifted.shape}")

    # Shape must be consistent
    lifted_single = lift(raw[0:1])
    assert lifted_single.shape[1] == lifted.shape[1], "Inconsistent lift dimension"
    print(f"  Lift dimension: {lifted.shape[1]}")
    print("[Test] Lifting function: OK\n")


def test_koopman_mpc_no_model():
    print("[Test] KoopmanMPCArduPilot (without model file) ...")
    print("  Skipping (no model file -- run copy_model.py first)")
    print("  This test will run once you have data/koopman_model.npz\n")


def test_koopman_mpc_with_model():
    print("[Test] KoopmanMPCArduPilot (with model file, Phase-2 params) ...")
    from koopman_mpc_ardupilot import KoopmanMPCArduPilot, RPM_HOVER, RPM_MAX

    model_path = Path("data/koopman_model.npz")
    if not model_path.exists():
        print("  Model not found -- skipping (run copy_model.py first)")
        return

    # Use Phase-2 identical parameters (same as run_comparison.py / run_mpc_closed_loop.py)
    ctrl = KoopmanMPCArduPilot(str(model_path), Hp=10, Hc=5,
                                pos_weight=120.0, vel_weight=1.0,
                                du_weight=1.0, u_weight=1e-3)
    pos     = np.array([0.0, 0.0, 1.0])
    vel     = np.zeros(3)
    rotmat  = np.eye(3).flatten()
    ang_vel = np.zeros(3)
    # Build Hp=10 step reference window (same as sitl_controller.py)
    ref_traj = np.tile(np.array([0.1, 0.0, 1.0]), (10, 1))

    roll, pitch, yaw, thrust, rpm_cmd = ctrl.solve(pos, vel, rotmat, ang_vel, ref_traj)
    print(f"  Koopman solve output: roll={roll:.3f}, pitch={pitch:.3f},"
          f" yaw={yaw:.3f}, thrust={thrust:.3f}")
    print(f"  RPM cmd: {np.round(rpm_cmd, 0)}  (hover={RPM_HOVER:.0f})")
    assert 0.0 <= thrust <= 1.0, "Thrust out of range!"
    assert abs(roll)  < 1.0, "Roll out of reasonable range!"
    assert abs(pitch) < 1.0, "Pitch out of reasonable range!"
    assert len(rpm_cmd) == 4, "Expected 4-motor RPM command"
    assert np.all(rpm_cmd >= 0), "Negative RPM!"
    assert np.all(rpm_cmd <= RPM_MAX * 1.01), "RPM above max!"
    print("[Test] KoopmanMPC solve: OK\n")


def test_lqr_controller():
    print("[Test] LQR controller ...")
    from lqr_controller_ardupilot import LQRControllerArduPilot

    ctrl = LQRControllerArduPilot()
    pos = np.array([0.0, 0.0, 0.8])
    vel = np.zeros(3)
    ref = np.array([0.1, 0.1, 1.0])

    roll, pitch, yaw, thrust = ctrl.compute(pos, vel, ref)
    print(f"  LQR output: roll={roll:.3f}, pitch={pitch:.3f},"
          f" yaw={yaw:.3f}, thrust={thrust:.3f}")
    assert 0.0 <= thrust <= 1.0, "Thrust out of range!"
    assert abs(roll)  <= ctrl.MAX_TILT + 0.01, "Roll exceeds max tilt!"
    assert abs(pitch) <= ctrl.MAX_TILT + 0.01, "Pitch exceeds max tilt!"
    print("[Test] LQR: OK\n")


def test_pid_controller():
    print("[Test] PID controller ...")
    from lqr_controller_ardupilot import PIDControllerArduPilot

    ctrl = PIDControllerArduPilot(dt=0.05)
    pos = np.array([0.0, 0.0, 0.8])
    vel = np.zeros(3)
    ref = np.array([0.1, 0.1, 1.0])

    # Run 5 steps
    for i in range(5):
        roll, pitch, yaw, thrust = ctrl.compute(pos, vel, ref)

    print(f"  PID output: roll={roll:.3f}, pitch={pitch:.3f},"
          f" yaw={yaw:.3f}, thrust={thrust:.3f}")
    assert 0.0 <= thrust <= 1.0, "Thrust out of range!"
    print("[Test] PID: OK\n")


def test_disturbance_injector():
    print("[Test] DisturbanceInjector (no connection needed) ...")
    from disturbance_injector import DisturbanceInjector, SCENARIOS

    # Mock MAVLink interface (no actual connection)
    class MockMAV:
        def set_param(self, name, value):
            pass

    inj = DisturbanceInjector(MockMAV())
    inj.apply_payload(0.25)
    assert inj.payload_active
    assert abs(inj.effective_gravity - 9.81 * 1.25) < 0.01
    inj.clear_payload()
    assert not inj.payload_active

    # Check all scenarios are defined
    for name, cfg in SCENARIOS.items():
        assert "wind_speed" in cfg, f"Missing wind_speed in scenario {name}"
        assert "payload_frac" in cfg, f"Missing payload_frac in scenario {name}"

    print(f"  Scenarios defined: {list(SCENARIOS.keys())}")
    print("[Test] DisturbanceInjector: OK\n")


def main():
    print("=" * 60)
    print("  ArduPilot SITL Module Tests (no SITL connection needed)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    tests = [
        test_frame_conversions,
        test_lifting_function,
        test_lqr_controller,
        test_pid_controller,
        test_disturbance_injector,
        test_koopman_mpc_with_model,  # only runs if model exists
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}\n")
            failed += 1
        except ImportError as e:
            print(f"  IMPORT ERROR: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ALL TESTS PASSED - Ready to run SITL simulation!")
    else:
        print("  Some tests failed. Fix errors before connecting to SITL.")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
