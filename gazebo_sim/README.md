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