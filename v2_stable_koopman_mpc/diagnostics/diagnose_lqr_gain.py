import numpy as np
from lqr_baseline import LQRController, virtual_to_rpm

lqr = LQRController()

print("LQR gain matrix K shape:", lqr.K.shape)
print("K row norms (one per virtual input: f, taux, tauy, tauz):")
print(np.round(np.linalg.norm(lqr.K, axis=1), 6))

# realistic scenario: already hovering at z=1.0, target 0.3m away in x
state12 = np.zeros(12)
state12[2] = 1.0  # already at hover altitude
target_pos = np.array([0.3, 0.0, 1.0])
target = np.zeros(12)
target[0:3] = target_pos
error = state12 - target

virtual_cmd = -lqr.K @ error
print(f"\nFor a 0.3m x-position error (state at origin, target at x=0.3):")
print(f"  virtual command [f, taux, tauy, tauz] = {virtual_cmd}")
print(f"  (compare to the ~5e-6 Nm torque scale that worked cleanly in the sign-check)")

rpm = virtual_to_rpm(*virtual_cmd)
print(f"  resulting RPM: {np.round(rpm, 1)}")

# --- check implied tilt angle for a larger, more realistic error ---
target_pos2 = np.array([0.4, 0.4, 1.0])
target2 = np.zeros(12)
target2[0:3] = target_pos2
error2 = state12 - target2
virtual_cmd2 = -lqr.K @ error2
tauy2 = virtual_cmd2[2]
# rough estimate: at equilibrium tilt, tauy needed to HOLD an angle theta
# scales with how far state is from target; here we just check the immediate
# commanded torque isn't wildly larger than what produced a reasonable
# (already-validated) ~15 degree tilt in our sign-check test
print(f"\nFor a larger (0.4, 0.4)m position error:")
print(f"  virtual command = {np.round(virtual_cmd2, 8)}")
print(f"  (our validated sign-check used tauy=5e-6 for a 15-degree tilt;")
print(f"   compare this tauy to that scale to gauge how aggressive this is)")
