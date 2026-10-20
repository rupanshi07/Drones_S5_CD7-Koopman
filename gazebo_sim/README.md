# Gazebo Simulation: PID and LQR baselines on a quadcopter

Goal: reproduce the PyBullet PID / LQR baseline experiments (same six disturbance scenarios) in a second simulator, Gazebo.
Status: PID and LQR run through all six scenarios in Gazebo, and every motor speed matches a hand calculation from the physics.
Koopman-MPC is NOT ported to Gazebo.

## 1. Environment
- Windows 11 + WSL2, Ubuntu 24.04 (commands below run in Ubuntu; git runs in Windows PowerShell)
- ROS 2 Jazzy (installed but only Gazebo's own Python API is used by the scripts)
- Gazebo Sim 8.15.0 (Harmonic), installed with `gz-harmonic` from the official Gazebo apt repository
  (`ros-jazzy-ros-gz` alone did not provide the `gz` command)
- Python: `python3-gz-transport13` (imports `gz.transport13`, `gz.msgs10`), plus `python3-numpy` and `python3-scipy` from apt
- Setup record: `scripts/setup_ubuntu.sh`

## 2. Drone and world
- Drone: X3 UAV from Gazebo Fuel (https://fuel.gazebosim.org/1.0/OpenRobotics/models/X3 UAV/4), loaded by `quadcopter.sdf`.
- It is much larger than the Crazyflie used in PyBullet (1.52 kg vs 0.027 kg), so Gazebo numbers are NOT numerically comparable to PyBullet numbers.
- `worlds/quadcopter_wrench.sdf` is a copy of Gazebo's `quadcopter.sdf` with the `ApplyLinkWrench` system added.
  The original world has no wrench topics, so all disturbance runs must use this copy.

### Constants (read from the world and model files)
| Quantity | Value |
|---|---|
| Total mass | 1.52 kg (base 1.5 + 4 x 0.005 rotors) |
| Ixx, Iyy, Izz | 0.0347563, 0.07, 0.0977 kg m^2 (base link) |
| motorConstant kf | 8.54858e-06 (thrust = kf * omega^2) |
| momentConstant | 0.016 (documented); measured effective value about 0.005 (rough, see section 5) |
| Max motor speed | 800 rad/s |
| Rotor positions (x, y) m | r0 (+0.13, -0.22), r1 (-0.13, +0.20), r2 (+0.13, +0.22), r3 (-0.13, -0.20) |
| Hover speed | 660 rad/s per motor (sqrt(m g / (4 kf)), g = 9.8); confirmed in simulation (motors held 660.0) |

### Motor allocation (verified by the bump test, section 5)
With s_i = omega_i^2:
```
F        = kf (s0 + s1 + s2 + s3)
tau_roll = kf (-0.22 s0 + 0.20 s1 + 0.22 s2 - 0.20 s3)
tau_pitch= kf * 0.13 (-s0 + s1 - s2 + s3)
tau_yaw  = c * kf (-s0 - s1 + s2 + s3),   c about 0.005
```

## 3. Interface (Gazebo transport)
| Topic | Role | PyBullet equivalent |
|---|---|---|
| `/X3/gazebo/command/motor_speed` (`gz.msgs.Actuators`, rad/s, 4 subscribers = 4 motors) | motor commands | `action` |
| `/world/quadcopter/pose/info` (`gz.msgs.Pose_V`) | pose of every entity | `obs` |
| `/world/quadcopter/wrench/persistent` (`gz.msgs.EntityWrench`) | external force on the drone | `p.applyExternalForce` |

Notes learned the hard way:
- The drone is the pose entry named `x3` (lowercase). Zero fields are omitted from the text output (an empty `position { }` means 0).
- The pose message has NO velocity. Velocity and angular rates are derived by filtered finite differences of the pose.
- Use the simulation clock from the pose header, not Python's wall clock (the simulation runs slightly slower than real time).
- The control loop runs once per pose message, about 30 Hz (about 442 steps in 15 s), NOT 48 Hz. Designs and ramps use dt = 1/48 s, so results are not at the PyBullet control rate.
- A single `gz topic -p` motor message gets lost (publisher/subscriber not yet connected). Publish repeatedly.
- Persistent wrench messages ADD to each other. To apply wind then wind + payload, send only the increment at each step.
- Force messages target model `x3` (type MODEL); forces are in world coordinates.

## 4. Scenarios (same definitions as PyBullet)
| # | Condition |
|---|---|
| 1 | Nominal |
| 2 | Wind from t = 0 (selector off in the Koopman-MPC ablation) |
| 3 | Wind from t = 0 (selector on) |
| 4 | Nominal, then wind from t = 7.5 s |
| 5 | Payload from t = 7.5 s |
| 6 | Wind from t = 0 plus payload from t = 7.5 s |

Disturbance sizes: wind 2.8 N along +x (about 19% of weight, a rough scaling of PyBullet's 0.05 N on 0.027 kg); payload 2 N downward
(about 13% of weight; PyBullet's was about 37%). t = 7.5 s is mid-flight, as in PyBullet (step 360 of 720).
Payload is a constant world-frame force on the main link: mass and inertia do not change, so it is not identical to the PyBullet mass change.
PID and LQR do not use the environment selector, so scenarios 2 and 3 are identical for each controller by design.
Target: height 0.3 m, reached with a 0.048 m/s ramp (same ramp rate as PyBullet), x = y = 0.

## 5. Tests, in the order they were done
| Test (script) | Result |
|---|---|
| Pose feed (`read_pose.py`) | Drone at rest stays at z = 0.055 m with no commands |
| First motor test (1000 rad/s) | Motors clip at 800 rad/s; drone climbed to about 15.65 m (consistent with a 4.6 m/s^2 climb over 2.6 s) |
| Thrust check (`motor_step_test.py 700`) | Predicted 1.223 m/s^2, measured 1.083 m/s^2; the gap is about 1.3% in thrust terms (rough fit, single run) |
| Height PID (`hover_pid.py`, Kp 6 Kd 5 Ki 1.5) | Overshoot +0.026 m, z = 0.306 m at 12 s, RMSE (last 5 s) 0.0157 m, tilt <= 0.03 deg |
| Payload on height PID (`hover_pid_payload.py`, Kp 15 Kd 8 Ki 6) | 2 N at 7.5 s: dips to 0.196 m, settles within 2 cm in 4.71 s after onset, motors 702.9 rad/s (predicted 702.9) |

The first payload attempt had no effect because the run used the original world without the wrench system; reruns used `worlds/quadcopter_wrench.sdf`.

### Motor bump test (`motor_bump_test.py <motor>`)
Holds 0.3 m with the height PID for 10 s of simulation time, adds +60 rad/s to ONE motor for 0.3 s, and reads the change in roll, pitch and yaw.

| Motor | Position (x, y) m | d_roll deg | d_pitch deg | d_yaw deg |
|---|---|---|---|---|
| 0 | (+0.13, -0.22) | -12.26, -10.60 | -3.68, -3.18 | -0.060, -0.091 |
| 1 | (-0.13, +0.20) | +10.79 | +3.56 | -0.094 |
| 2 | (+0.13, +0.22) | +11.27 | -3.38 | +0.080 |
| 3 | (-0.13, -0.20) | -9.38 | +3.09 | +0.115 |

- Roll and pitch match the prediction 0.5 * (dF * arm / I) * t^2 with dF = 0.708 N (roll 10.5 to 11.6 deg, pitch 3.4 deg) within about 10%. This checks the inertias, rotor positions and kf.
- Yaw signs: motors 0 and 1 negative, motors 2 and 3 positive (consistent with the model file's spin directions).
- Yaw strength: momentConstant = 0.016 predicts about 0.30 deg; measured about 0.09 deg on average, so the effective coefficient is about 0.005.
  This is rough (yaw changes are under 0.1 deg and motor 0's two runs differ by 40%). Yaw barely matters in these scenarios.

## 6. Controllers
### LQR (`lqr_hover.py`, `run_scenario_gazebo.py lqr`)
- 12 states (x y z, vx vy vz, roll pitch yaw, p q r), 4 inputs (thrust, three torques); same A and B structure as the PyBullet LQR with the X3 constants
  (1/m = 0.658, 1/Ixx = 28.8, 1/Iyy = 14.3, 1/Izz = 10.2), discretised at dt = 1/48 s, discrete Riccati solution.
- Weights by Bryson's rule (first guess): Q = 1 / (max acceptable)^2 (0.1 m position, 0.5 m/s velocity, 0.2 rad roll and pitch, 0.5 rad yaw, 1 rad/s rates), R = 1 / (max input)^2 (4 N, 0.2, 0.2, 0.02 N m).
- No integral action (as in the PyBullet LQR).

### PID (`pid_hover.py`, `run_scenario_gazebo.py pid`)
- Position PID on x and y gives a desired tilt (small angle, limited to 20 deg): Kp 4, Kd 3.6, Ki 2, integral clamp 3.
- Attitude PD gives torques (natural frequency 10 rad/s, damping 0.9; yaw 3 rad/s, damping 1.0).
- Altitude PID: Kp 15, Kd 8, Ki 6, integral clamp 8 N, thrust divided by cos(tilt).
- Integral action on x, y and z. Gains are a first guess, not tuned.

## 7. Results: six scenarios, PID and LQR (`run_all_gazebo.sh`)
| Scenario | PID | LQR |
|---|---|---|
| 1 | z 0.300, x 0.000, motors 660.0 rad/s (physics 660.0) | z 0.300, x 0.000, motors 660.0 |
| 2 and 3 | max x 0.366, final x 0.000, z 0.300, motors 665.7 (physics 665.8) | final x +0.446 (max 0.476), z 0.293, motors 665.8 |
| 4 | max x 0.372, final x -0.009, z 0.299, motors 663.7 (physics 665.8; tail still settling) | final x +0.446, z 0.293, motors 665.8 |
| 5 | final z 0.296 (min 0.195), motors 702.9 (physics 702.9) | final z 0.245 (sag 0.055 m), motors 702.9 |
| 6 | final x +0.001, final z 0.296, motors 707.5 (physics 707.7) | final x +0.394, final z 0.239, motors 707.7 |

- Physics values: 660.0 (none), 665.8 (wind 2.8 N: thrust sqrt(14.9^2 + 2.8^2)), 702.9 (payload 2 N), 707.7 (wind + payload).
  All 12 runs match within 2 rad/s after the scenario 6 fix.
- Scenario 6, first run: motors 718.0 (PID) and 721.5 (LQR) against 707.7. Cause: persistent wrench messages add, so wind was counted twice
  (physics for 5.6 N wind + 2 N payload = 721.5, which the LQR matched). Fixed by sending only the increment at each step.
- The PID returns to the reference under wind and payload (it has integral action). The LQR keeps lasting offsets (payload sag 0.055 m, wind offset +0.446 m), as in PyBullet.
- The LQR wind offset in scenario 6 (+0.394 m) is smaller than in wind-only (+0.446 m) because the larger thrust needs a smaller tilt (9.4 vs 10.65 deg):
  0.446 x tan 9.41 / tan 10.65 = 0.393.
- Wind-only LQR tilt 10.65 deg equals atan(2.8 / 14.9); the PID's wind + payload tilt of 9.4 deg matches the physics.
- Earlier single-controller runs gave the same numbers: `lqr_hover.py` (no disturbance, payload 2 N, wind 2.8 N) and `pid_hover.py` (all four combinations).

## 8. How to run (Ubuntu)
Start Gazebo with the wrench world, then run a script in a second Ubuntu window. Restart Gazebo between runs so the drone starts on the ground.
```
gz sim -r /mnt/c/S5/Drones/koopman-mpc-drone/gazebo_sim/worlds/quadcopter_wrench.sdf
python3 /mnt/c/S5/Drones/koopman-mpc-drone/gazebo_sim/scripts/pid_hover.py 2.8 2
python3 /mnt/c/S5/Drones/koopman-mpc-drone/gazebo_sim/scripts/lqr_hover.py 0 2
```
Arguments for `pid_hover.py` and `lqr_hover.py`: wind force N (+x), payload force N (down), optional onset time in seconds (default 7.5).

All 12 scenario runs (headless Gazebo, takes about 6 to 10 minutes):
```
bash /mnt/c/S5/Drones/koopman-mpc-drone/gazebo_sim/scripts/run_all_gazebo.sh
```
One run: `python3 scripts/run_scenario_gazebo.py <pid|lqr> <1-6>` with a Gazebo server running.

## 9. Files
- `scripts/setup_ubuntu.sh`: record of the install commands
- `scripts/read_pose.py`, `read_pose.sh`: print the drone pose
- `scripts/motor_test.sh`, `motor_step_test.py`: raw motor command and thrust check
- `scripts/hover_pid.py`, `hover_pid_payload.py`: height-only PID tests
- `scripts/motor_bump_test.py`: motor numbering, yaw signs and torque authority
- `scripts/lqr_hover.py`, `pid_hover.py`: hover with optional wind and payload
- `scripts/run_scenario_gazebo.py`, `run_all_gazebo.sh`: the six scenarios for both controllers
- `worlds/quadcopter_wrench.sdf`: world with the wrench system
- `results/*.csv`: time series from the hover tests; `results/gazebo_<pid|lqr>_scenario_<n>.npz`: scenario runs
  (arrays: `states` with columns x y z vx vy vz roll pitch yaw p q r, `actions` in rad/s, `t`, `disturbance_flag`, `disturbance_onset_step`)

## 10. Limitations and not done
- Koopman-MPC is not ported (it would need new data collection and retraining in Gazebo).
- The `.npz` files use 12 state columns and rad/s, while the PyBullet analysis scripts expect a 16-column state and RPM (rad/s to RPM is a factor of 9.55), so an adapter is needed to include these runs in the comparison plots.
- Control loop is about 30 Hz (pose message rate), not 48 Hz.
- Single deterministic run per condition; gains and LQR weights are first guesses.
- Wind 2.8 N is a rough scaling of the PyBullet wind; the payload is a force, not a mass change.
- Yaw coefficient (about 0.005 vs documented 0.016) is only a rough estimate.
- The X3 differs from the Crazyflie, so Gazebo and PyBullet numbers should be compared qualitatively (offsets, recovery), not numerically.
- The Gazebo model cache (`~/.gz`) is not included in this repository.