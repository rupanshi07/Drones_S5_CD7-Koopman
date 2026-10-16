   <img width="509" height="115" alt="amrita_logo" src="https://github.com/user-attachments/assets/ddc16d63-4a54-47f6-b606-2eae07f12ee3" />

# ArduPilot SITL Integration — Koopman Model Predictive Control

This folder contains the **ArduPilot Software-in-the-Loop (SITL) integration** for testing a Koopman-based Model Predictive Controller (MPC) for a simulated multirotor. It is intended to connect the controller to an ArduPilot SITL vehicle through MAVLink, run controlled simulation scenarios, and record tracking performance.

## Contents

- [Project overview](#project-overview)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the simulation](#run-the-simulation)
- [Scenarios](#scenarios)
- [Results and validation](#results-and-validation)
- [Troubleshooting](#troubleshooting)
- [Safety and limitations](#safety-and-limitations)
- [Future work](#future-work)
- [References](#references)

## Project overview

The purpose of this folder is to evaluate a Koopman-MPC controller in a repeatable ArduPilot SITL environment before considering any physical flight tests.

The controller uses a learned Koopman representation to predict system behaviour over a finite horizon and optimize control inputs. ArduPilot SITL provides the simulated vehicle, while MAVLink carries telemetry and commands between the controller and the simulator.

### Goals

- Connect a Python controller to ArduPilot SITL using MAVLink.
- Run a baseline, or **clean**, trajectory-tracking scenario.
- Evaluate the controller under additional conditions such as payload changes or wind disturbances, where supported by the local scripts.
- Record position-tracking error and control behaviour.
- Compare simulation results with the standalone Koopman-MPC reference implementation.

## How it works

```text
ArduPilot SITL / Mission Planner
             |
           MAVLink
             |
       Python SITL bridge
             |
       State estimation
             |
       Koopman-MPC controller
             |
       Control commands
             |
       ArduPilot SITL vehicle
             |
       Telemetry and logs
```

## Repository structure

The file names below describe the roles of commonly used files in this integration. Your local version may contain additional files or slightly different names.

| File or folder | Purpose |
|---|---|
| `bridge.py` | MAVLink bridge or communication helper, if included in this version |
| `setup_vehicle.py` | Vehicle connection and initial setup, if included |
| `sitl_controller.py` | Runs the controller against the SITL vehicle, if included |
| `koopman_model.npz` | Saved Koopman model arrays and associated scaling values, if included |
| `results` | Simulation logs and metrics, if generated |

## Requirements

- Windows, Linux, or another environment supported by the installed ArduPilot SITL and Python dependencies
- ArduPilot SITL
- Mission Planner (optional, for a graphical Windows workflow)
- Python 3
- The Python packages required by the scripts in this folder
- A compatible Koopman model file, if the controller loads one from disk

Common Python dependencies used by this type of workflow include:

```bash
python -m pip install numpy scipy matplotlib pymavlink cvxpy clarabel
```

Install only the packages required by the scripts in your copy. If a solver or package is unavailable for your Python version, consult that package's installation documentation.

## Setup

### 1. Open a terminal in this folder

On Windows Command Prompt, navigate to the directory containing this README and the SITL scripts. For example:

```bat
cd /d "D:\path\to\your\repository\ardupilot_sitl"
```

Replace the example path with the actual location on your computer.

### 2. Install dependencies

```bat
python -m pip install numpy scipy matplotlib pymavlink cvxpy clarabel
```

### 3. Check script options

Run the following for each script that exists in your folder:

```bat
python sitl_controller.py --help
python setup_vehicle.py --help
python bridge.py --help
```

If a script does not support `--help`, inspect its argument parser and configuration at the top of the file. Confirm the connection port, controller frequency, scenario name, and duration before launching the simulation.

### 4. Start ArduPilot SITL

Start the appropriate ArduPilot SITL vehicle using your installed workflow. If you use Mission Planner, start its simulation and confirm that the vehicle is responding before starting the Python controller.

**Important:** The MAVLink connection endpoint must match between SITL, Mission Planner, any bridge process, and the controller. Do not copy a port number from another setup without checking the configuration.

## Run the simulation

The commands below are **templates**, not guaranteed commands for every version. Confirm supported options with `python sitl_controller.py --help` and adjust them to match your scripts.

Example controller invocation, if these options are supported:

```bat
python sitl_controller.py --controller koopman --scenario clean --ctrl-freq 48 --duration 15
```

If the controller requires an explicit port, specify the port configured in your SITL/MAVLink setup. If your script uses different argument names, use the names shown by its help output.

### Recommended test order

1. **Connection check:** Confirm MAVLink connection and telemetry before enabling control.
2. **Short clean test:** Run a short baseline simulation with no added disturbances.
3. **Inspect logs:** Check actual position, reference position, tracking error, flight mode, arming state, and control outputs.
4. **Repeat the baseline:** Ensure the result is reasonably repeatable.
5. **Disturbance tests:** Only after the baseline behaves correctly, run payload or wind scenarios supported by the code.

## Scenarios

Use only scenario names that are actually implemented in your local controller.

| Scenario | Intended purpose |
|---|---|
| `clean` | Baseline tracking without an added disturbance |
| `payload` | Evaluate the effect of a payload or mass change, if implemented |
| `wind` | Evaluate response to a wind disturbance, if implemented |
| `combined` | Evaluate multiple disturbances together, if implemented |

Example template:

```bat
python sitl_controller.py --controller koopman --scenario clean --duration 15
```

Replace the scenario and arguments with the supported values listed by your script.

## Results and validation
<img width="1917" height="1133" alt="Screenshot 2026-10-17 032706" src="https://github.com/user-attachments/assets/b1c0e550-6816-4093-86cc-3a60df33fec4" />
<img width="1917" height="1136" alt="Screenshot 2026-10-17 015808" src="https://github.com/user-attachments/assets/2127bd4a-8aeb-48ed-b4cb-c894b3ec2c23" />
<img width="1916" height="1140" alt="Screenshot 2026-10-17 032734" src="https://github.com/user-attachments/assets/11121352-77ff-4f59-a775-2962d23ba17a" />

Record each test with the controller settings, scenario, duration, and output metrics.

| Scenario | Position RMSE (m) | Maximum error (m) | Mean thrust | Notes |
|---|---:|---:|---:|---|
| Clean | 62.9789 m | 92.8462 m | 0.500 | Run and validate first |
| Payload | 301.0049 m | 332.3632 m | 0.531 | Run after baseline validation |
| Wind | 201.1515 m | 225.4062 m | 0.500 | Run after baseline validation |
| Combined | 421.8613 m | 450.6540 m | 0.531 | Run after individual tests |

### What to check

- **Position RMSE:** Lower is better, but the acceptable value depends on the reference trajectory and test conditions.
- **Maximum position error:** Helps identify large tracking failures that an average metric may hide.
- **Reference vs. actual trajectory:** Plot each position axis on the same time base.
- **Coordinate frames:** Verify that the controller and MAVLink interface agree on NED/ENU world axes and body-frame conventions.
- **Units and scaling:** Confirm metres vs. centimetres, radians vs. degrees, thrust scaling, and actuator-command mapping.
- **Vehicle state:** Check whether the vehicle armed, entered the intended mode, remained airborne, and avoided disarming unexpectedly.
- **Model compatibility:** Ensure the controller loads the intended model file and applies its scaling exactly once.

Suggested repository layout:

## Troubleshooting

### The controller cannot connect

- Confirm that SITL is running.
- Check the configured MAVLink endpoint and port.
- Ensure that the bridge and controller are using the expected endpoint.
- Check that another process is not already using the same port.

### The vehicle does not arm

- Read the SITL and controller messages for the exact rejection reason.
- Check the current flight mode and ArduPilot pre-arm checks.
- Do not disable arming checks indiscriminately; resolve the underlying issue first.

### The vehicle moves away from the reference

- Stop the test safely and return to a short clean scenario.
- Plot reference and actual positions together.
- Verify world/body coordinate conversions and axis signs.
- Check the sign and units of the control commands.
- Confirm model input scaling is not applied twice.
- Verify that the controller frequency and prediction horizon match the configuration.

### Tracking error is unexpectedly large

- Confirm the RMSE calculation uses matching timestamps and coordinate frames.
- Check for invalid, missing, or stale telemetry.
- Inspect whether the vehicle disarmed or left the intended mode during the run.
- Validate the baseline before adding wind or payload disturbances.

## Safety and limitations

- This repository is for **simulation and research**; it does not establish that the controller is safe for real flight.
- Do not connect this experimental controller to a physical aircraft without a separate safety review, appropriate failsafes, supervised testing, and validated command limits.
- Results depend on the learned model, simulator settings, coordinate conventions, controller configuration, and solver behaviour.
- Keep generated logs and plots so that results can be reproduced and independently checked.

## Future work

- Validate baseline trajectory tracking and document quantitative metrics.
- Add plots comparing reference and actual trajectories.
- Evaluate robustness to payload and wind changes after baseline validation.
- Document model training, state lifting, input scaling, and coordinate-frame conventions.
- Compare Koopman-MPC performance with a baseline controller under identical conditions.


## References

1. **ArduPilot Documentation — SITL Simulator**  
   Official documentation for Software-in-the-Loop simulation of ArduPilot vehicles.  
   https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html

2. **ArduPilot Documentation — Using SITL for Testing**  
   Guidance on using SITL to test autopilot behaviour in a simulated environment.  
   https://ardupilot.org/dev/docs/using-sitl-for-ardupilot-testing.html

3. **ArduPilot — Official Source Code Repository**  
   Source code for ArduPilot, including autopilot software and simulation-related components.  
   https://github.com/ArduPilot/ardupilot

4. **MAVLink Developer Guide**  
   Protocol documentation for communication between the flight controller, companion computer, and ground control station.  
   https://mavlink.io/en/

5. **Mission Planner — Official Documentation**  
   Documentation for Mission Planner, the ground control station used to monitor and interact with ArduPilot vehicles.  
   https://ardupilot.org/planner/
