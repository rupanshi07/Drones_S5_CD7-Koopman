# Gazebo Simulation (work in progress)

Goal: reproduce the PID / LQR baseline experiments in Gazebo as a second simulator.
Status: environment set up, a quadcopter flies from raw motor commands, and the drone's pose can be read live.
Controllers are NOT ported yet.

## Environment
- Windows 11 + WSL2, Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Sim 8.15.0 (Harmonic), installed from the official Gazebo apt repository
  (`ros-jazzy-ros-gz` alone did not provide the `gz` command)

## Done
- [x] WSL2 + Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic installed (see `scripts/setup_ubuntu.sh`)
- [x] Gazebo GUI opens under WSL2 (`gz sim shapes.sdf`)
- [x] Example world `quadcopter.sdf` (X3 UAV) loads and flies
- [x] Found the command and state topics (below)
- [x] Drove the four motors from the terminal (`scripts/motor_test.sh`)
- [x] Read the live drone pose (`scripts/read_pose.sh`)
- [x] Read the drone's physical constants from the model file

## Interface (X3 UAV in quadcopter.sdf)
| Topic | Role | PyBullet equivalent |
|---|---|---|
| `/X3/gazebo/command/motor_speed` (`gz.msgs.Actuators`, 4 subscribers = 4 motors) | motor commands | `action` |
| `/world/quadcopter/pose/info` | pose of every entity in the world | `obs` |
| `/world/quadcopter/dynamic_pose/info` | pose of moving entities | (not used yet) |
| `/world/quadcopter/clock` | simulation clock | (not used yet) |

Note: a single `gz topic -p` message did nothing (likely lost before the subscriber connected).
Sending the message repeatedly made the drone fly.

## Reading the drone state
`/world/quadcopter/pose/info` publishes one message containing every entity (ground_plane, x3,
base link, rotors, visuals, sun). The header stamp is simulation time.
- The drone body is the entry named `x3` (lowercase, id 8): `position {x y z}` and
  `orientation {x y z w}` (quaternion).
- Entries printing an empty `position { }` mean 0, because zero fields are omitted from the text output.
- The message has NO velocity. Velocity must be derived from successive poses, or another topic must be found.
- The pose is in the world frame; the drone rests at z about 0.055 m at the start.

## Findings from the pose test
- After the 1000 rad/s motor test the drone was at z about 15.65 m, x and y about 0, orientation about level.
- Consistency check: at the 800 rad/s limit, thrust is about 4 x 8.54858e-06 x 800^2 = 21.9 N against a weight
  of about 14.9 N, so net acceleration is about 4.6 m/s^2. Over the 2.64 s of simulation time in that message that gives
  about 16 m, close to the observed 15.65 m. (Own calculation, not a measured fit.)
- After restarting `gz sim -r quadcopter.sdf` the drone rests at z about 0.055 m, level.
  Restart the simulation to reset the drone after a test.
- Not yet checked: whether the drone stays at rest with no commands.

## X3 constants (from the world and model files)
Model: https://fuel.gazebosim.org/1.0/OpenRobotics/models/X3 UAV/4

| Quantity | Value |
|---|---|
| Total mass | 1.52 kg (base 1.5 + 4 x 0.005 rotors) |
| Ixx, Iyy, Izz | 0.0347563, 0.07, 0.0977 kg m^2 (base link) |
| motorConstant (kf) | 8.54858e-06 (thrust = kf * omega^2) |
| momentConstant | 0.016 |
| maxRotVelocity | 800 rad/s |
| Rotor positions (x, y) m | r0 (0.13, -0.22), r1 (-0.13, 0.20), r2 (0.13, 0.22), r3 (-0.13, -0.20) |
| Estimated hover speed | about 660 rad/s per motor (sqrt(m g / (4 kf)), g = 9.8) |

## Not done yet
- [ ] Python node that reads pose and publishes motor speeds (closed loop)
- [ ] PID and LQR ported with the X3 constants
- [ ] Wind and payload disturbances in Gazebo
- [ ] The six scenarios and metrics in the same `.npz` format
- [ ] Koopman-MPC (needs new data collection and retraining in Gazebo)

## Unverified (do not rely on these yet)
- Yaw signs of the motor allocation matrix, and how motor indices map to the rotors in the model file
- The exact meaning of momentConstant (assumed torque = 0.016 x thrust)
- The ~660 rad/s hover speed (computed from the model file, not checked in simulation)
- Python package names for gz-transport (`python3-gz-transport13`, `gz.msgs10`)

## Notes
- The X3 is much larger than the Crazyflie used in the PyBullet experiments (1.52 kg vs 0.027 kg),
  so Gazebo results will not be numerically comparable to PyBullet results, and LQR weights will need retuning.
- The model cache (`~/.gz`) is not included in this repo.


## Update: closed-loop height hold

Scripts (all in `scripts/`): `read_pose.py` (pose feed), `motor_step_test.py` (thrust check), `hover_pid.py` (height PID).
Log from the last hover run: `results/hover_pid_log.csv`.

| Test | Result |
|---|---|
| Pose feed (`read_pose.py`) | drone at rest stays at z = 0.055 m with no commands; sim time runs a little slower than wall time, so controllers use the sim clock from the pose header |
| Motor step, 700 rad/s (`motor_step_test.py`) | predicted 1.223 m/s^2, measured 1.083 m/s^2; the gap is about 1.3% in thrust terms (rough fit, one run) |
| Height PID (`hover_pid.py`), target 0.3 m, ramp 0.048 m/s, 48 Hz | overshoot +0.026 m, z = 0.306 m at 12 s, RMSE (last 5 s) 0.0157 m, max tilt 0.03 deg |

Gains (first guess, not tuned): Kp 6, Kd 5, Ki 1.5. Thrust is converted to motor speed with omega = sqrt(F / (4 kf)) and sent equally to all four motors.

Verified now: hover speed is about 660 rad/s (the motors held 660.0-660.1 rad/s with near-zero vertical speed);
the Python bindings are `python3-gz-transport13` with `gz.transport13` and `gz.msgs10`.

Still unverified: yaw signs and motor-to-rotor mapping; the meaning of momentConstant.

Next: attitude control (needed once wind or payload create tilt), then LQR, then wind and payload disturbances, then the scenarios.
Note: this test controls height only, with equal motor speeds, so it cannot resist sideways forces or an off-centre load.


## Update: payload disturbance

World: `worlds/quadcopter_wrench.sdf` is a copy of Gazebo's `quadcopter.sdf` with the `ApplyLinkWrench` system added.
The original `quadcopter.sdf` has no wrench topics, so payload and wind runs MUST use this world:
`gz sim -r worlds/quadcopter_wrench.sdf` (run from the `gazebo_sim` folder). Force topic: `/world/quadcopter/wrench/persistent` (`gz.msgs.EntityWrench`, entity "x3", type MODEL).

Script: `scripts/hover_pid_payload.py <newtons>`. Height PID, target 0.3 m, 48 Hz, ramp 0.048 m/s.
Gains (first guess, not tuned): Kp 15, Kd 8, Ki 6, integral clamp 8 N. A persistent downward force is applied at t = 7.5 s of simulation time (mid-flight, as in the PyBullet scenarios).

| Run | Result |
|---|---|
| 0 N (baseline) | peak z = 0.320 m at about 6 s, back to 0.300 m by about 14 s, motors 660 rad/s, tilt <= 0.02 deg |
| 2 N at 7.5 s | dips to about 0.196 m (0.104 m below target), settles within 2 cm in 4.71 s after onset, motors settle at 702.9 rad/s (predicted 702.9), z = 0.296 m at 15 s and still converging, tilt <= 0.07 deg |

Logs: `results/hover_pid_payload_0N_log.csv`, `results/hover_pid_payload_2N_log.csv` (columns: t, z_ref, z, vz, omega, tilt_deg, force_applied).

Simplification: the payload is a constant downward force in the world frame on the model's main link. Mass and inertia do not change, so this is not identical to the mass change used in PyBullet.
2 N is about 13% of the drone's weight; PyBullet's 0.01 kg on a 0.027 kg drone was about 37%.
Single run per condition.

Verified now: the wrench plugin acts on the drone (entity "x3", type MODEL); mass and thrust constant confirmed under load.
Still unverified: yaw signs and motor-to-rotor mapping; the meaning of momentConstant.
Not done: wind, attitude control (equal motor speeds cannot resist a sideways force), LQR, the six scenarios.


## Update: motor mapping and yaw signs (bump test)

Script: `scripts/motor_bump_test.py <motor>`. It holds 0.3 m with the height PID for 10 s of sim time, then adds +60 rad/s to ONE motor for 0.3 s and reads the change in roll, pitch and yaw.

| Motor | Position (x, y) m | d_roll deg | d_pitch deg | d_yaw deg |
|---|---|---|---|---|
| 0 | (+0.13, -0.22) | -12.26, -10.60 | -3.68, -3.18 | -0.060, -0.091 |
| 1 | (-0.13, +0.20) | +10.79 | +3.56 | -0.094 |
| 2 | (+0.13, +0.22) | +11.27 | -3.38 | +0.080 |
| 3 | (-0.13, -0.20) | -9.38 | +3.09 | +0.115 |

Roll and pitch match a prediction of 0.5 * (dF * arm / I) * t^2 with dF = 0.708 N (roll 10.5-11.6, pitch 3.4 deg), within about 10%. This checks the inertias, rotor positions and kf.
Yaw signs: motors 0 and 1 negative, motors 2 and 3 positive (consistent with the model file's spin directions).
Yaw strength: with momentConstant = 0.016 I predict about 0.30 deg; measured about 0.09 deg on average. The effective coefficient is about 0.005 (rough: yaw changes are under 0.1 deg and motor 0's two runs differ by 40%).

Allocation (s_i = omega_i^2):
F = kf (s0+s1+s2+s3); tau_roll = kf (-0.22 s0 + 0.20 s1 + 0.22 s2 - 0.20 s3);
tau_pitch = kf 0.13 (-s0 + s1 - s2 + s3); tau_yaw = c kf (-s0 - s1 + s2 + s3), c about 0.005.

Now verified: motor numbering, yaw signs, inertias, rotor positions. Still unverified: yaw coefficient (rough only).

CONFIRMED (bump test): motors 0..3 at (x,y) = (+.13,-.22), (-.13,+.20), (+.13,+.22), (-.13,-.20); yaw sign motors 0,1 = -, motors 2,3 = +; roll/pitch authority matches inertias within 10%; yaw coefficient c about 0.005 (documented 0.016, rough).
ALLOCATION: F = kf(s0+s1+s2+s3); tau_roll = kf(-.22 s0 + .20 s1 + .22 s2 - .20 s3); tau_pitch = kf*.13(-s0+s1-s2+s3); tau_yaw = c*kf(-s0-s1+s2+s3); s_i = omega_i^2.
SCRIPTS added: motor_bump_test.py (committed on gazebo-lqr), lqr_hover.py <wind_N> <payload_N> [onset_s] (needs worlds/quadcopter_wrench.sdf, python3-numpy, python3-scipy).
GIT LESSON: merge a PR only after the last push, or open a new PR for later commits.


## Update: LQR hover with payload and wind

Script: `scripts/lqr_hover.py <wind_N> <payload_N> [onset_s]` (needs `worlds/quadcopter_wrench.sdf`; uses system python3 with numpy and scipy).
12-state LQR (position, velocity, roll/pitch/yaw, rates) with the same A and B structure as the PyBullet LQR, discretised at 48 Hz, X3 constants,
motor allocation from the bump test. Weights by Bryson's rule (first guess): Q = 1/(max acceptable)^2, R = 1/(max input)^2. Velocity and rates come from filtered finite differences of the pose. No integral action.

| Run (disturbance at t = 7.5 s) | Result |
|---|---|
| none | z = 0.300 m, x = y = 0.000, tilt 0.00 deg, motors 660.0 rad/s |
| payload 2 N (down) | z dips to 0.245 m and stays (sag 0.055 m, about 36 N/m effective stiffness); motors 702.9 rad/s (predicted 702.9) |
| wind 2.8 N (+x) | x settles at +0.446 m, tilt 10.65 deg (physics: atan(2.8/14.9) = 10.65), motors 665.8 rad/s (physics: 665.8), z = 0.293 m, y about 0 |

Offsets under disturbance are expected: there is no integral term (the PyBullet LQR also sagged under payload).
Wind 2.8 N is a rough scaling of PyBullet's 0.05 N on 0.027 kg (about 19% of weight), not an exact equivalence.
One deterministic run per condition. Wind and payload together not yet tested.
Logs: `results/lqr_wind*_payload*_log.csv`.

RESULTS (Gazebo, X3, LQR no integral, disturbance at 7.5 s): none -> z 0.300, x=y=0; payload 2 N -> sag 0.055 m, motors 702.9 (predicted 702.9); wind 2.8 N -> x +0.446 m, tilt 10.65 deg (physics 10.65), motors 665.8 (physics 665.8). LQR and PID logs in gazebo_sim/results/*.csv.
SCRIPTS added: lqr_hover.py, pid_hover.py (both <wind_N> <payload_N> [onset_s]; PID has position/attitude/altitude loops with integral action), run in worlds/quadcopter_wrench.sdf.
NEXT: PID results; scenario runner for the six scenarios (wind from start in 2,3,6; wind from mid in 4; payload mid in 5,6) saving .npz like PyBullet (states, actions, disturbance_onset_step); push; merge PR once after last push.


## Update: PID with position, attitude and altitude loops

Script: `scripts/pid_hover.py <wind_N> <payload_N> [onset_s]` (run in `worlds/quadcopter_wrench.sdf`).
Loops: x/y PID -> desired tilt (small angle), attitude PD -> torques (wn 10 rad/s, zeta 0.9), altitude PID (thrust divided by cos of tilt). Gains are a first guess.
Integral action on x, y, z.

| Run (disturbance at t = 7.5 s) | Result |
|---|---|
| none | z = 0.300 m, x = y = 0 |
| payload 2 N | min z 0.195 m, z = 0.296 m at 15 s; motors 702.9 rad/s (predicted 702.9) |
| wind 2.8 N | max x 0.370 m, x = -0.009 m at 15 s, tilt 10.6 deg, motors 665.7 (physics 665.8) |
| wind + payload | max x 0.310 m, x = -0.008 m, z = 0.295 m, tilt 9.4 deg, motors 707.6 (physics 707.7, tilt 9.41) |

LQR (no integral term) under the same disturbances stays offset (payload sag 0.055 m, wind offset +0.446 m); PID returns to the reference.
The PID dip under payload is deeper because its height gain is softer (15 N/m vs about 36 N/m effective for the LQR); the lasting offset is the key difference.
Single deterministic runs, untuned gains. Logs: `results/pid_wind*_payload*_log.csv`.