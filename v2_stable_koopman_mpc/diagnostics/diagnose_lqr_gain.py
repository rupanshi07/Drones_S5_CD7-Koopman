"""
Diagnostic: inspect the cascaded LQR's gain magnitudes and the virtual
commands / RPMs it produces for a realistic position error.

NOTE: this script was originally written against an earlier FLAT 12-state
LQR design (a single gain matrix `self.K`). That design was abandoned --
it destabilized the platform regardless of weight tuning -- and replaced
by the cascaded outer-position / inner-attitude design in lqr_baseline.py,
which exposes `self.Ko` and `self.Ki` instead. This script has been
updated to match; before the fix it raised AttributeError on `lqr.K`.

CAVEAT: the cascaded LQR baseline itself is NOT working. It does not
stabilize in closed loop (see run_lqr_closed_loop.py, which crashes
within ~1-2 s in every configuration tried). This script is retained as
a debugging artifact only; do not treat its output as evidence of a
functioning baseline.
"""

import numpy as np
from lqr_baseline import LQRController, virtual_to_rpm, RPM_HOVER, MAX_TILT_RAD, G

lqr = LQRController()

print("Cascaded LQR gains:")
print(f"  Outer (position -> desired accel) Ko shape: {lqr.Ko.shape}")
print(f"    row norms [ax, ay, az]: {np.round(np.linalg.norm(lqr.Ko, axis=1), 6)}")
print(f"  Inner (attitude -> torques)       Ki shape: {lqr.Ki.shape}")
print(f"    row norms [taux, tauy, tauz]: {np.round(np.linalg.norm(lqr.Ki, axis=1), 6)}")

# realistic scenario: already hovering at z=1.0, target 0.3 m away in x
state12 = np.zeros(12)
state12[2] = 1.0
target_pos = np.array([0.3, 0.0, 1.0])

# --- outer loop in isolation: what tilt does a 0.3 m error demand? ---
outer_state = np.hstack([state12[0:3], state12[3:6]])
outer_target = np.hstack([target_pos, [0, 0, 0]])
acc_cmd = -lqr.Ko @ (outer_state - outer_target)
theta_des_unclamped = np.arctan(acc_cmd[0] / G)
theta_des = np.arctan(np.clip(acc_cmd[0] / G, -np.tan(MAX_TILT_RAD), np.tan(MAX_TILT_RAD)))

print(f"\nFor a 0.3 m x-position error (hovering at z=1.0, target x=0.3):")
print(f"  outer-loop accel command [ax, ay, az] = {np.round(acc_cmd, 6)}")
print(f"  implied pitch target: {np.degrees(theta_des_unclamped):.2f} deg "
      f"(unclamped) -> {np.degrees(theta_des):.2f} deg (after clamp)")

# --- full cascade: what RPM does the whole controller actually command? ---
rpm = lqr.compute(state12, target_pos)
print(f"  final commanded RPM: {np.round(rpm, 1)}")
print(f"  deviation from hover: {np.round(rpm - RPM_HOVER, 1)}")
print(f"  (for scale: diagnose_lqr_sign.py used a ~450 rpm deviation to")
print(f"   produce a clean, isolated ~15 deg tilt)")
