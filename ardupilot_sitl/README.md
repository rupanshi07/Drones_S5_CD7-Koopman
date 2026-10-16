# ArduPilot SITL – Koopman-MPC Simulation (Windows, No WSL)

**Phase 3: Real ArduPilot Simulation — Windows Native**

This folder contains a **complete, runnable ArduPilot SITL simulation** of the Koopman-MPC
controller developed in Phases 1 and 2, running on **native Windows** (no WSL, no Linux).

---

## What This Is

The Phase 1 and 2 work validated the Koopman-MPC controller inside `gym-pybullet-drones`.
This phase moves the same Koopman-MPC controller into a real ArduPilot SITL loop:

- **ArduCopter SITL** runs via **Mission Planner** (Windows, no WSL needed)
- **pymavlink** connects Python to the simulated flight controller over MAVLink (TCP)
- The Koopman-MPC controller reads vehicle state and sends `SET_ATTITUDE_TARGET` + thrust
- **Three controllers compared**: Koopman-MPC, LQR baseline, PID baseline
- All scenarios from Phase 2 replicated: clean, wind, payload, combined

## Architecture

```
Mission Planner (Windows)
  |-- ArduCopter SITL (TCP :5760) <--> Mission Planner GCS (3D view)
         |
         | MAVLink TCP (pymavlink)
         v
  Python Control Loop (this folder)
     sitl_controller.py
       - Reads: pos, vel, attitude, ang_vel via MAVLink
       - Lifts state with rotation-matrix lifting (Phase 2 method)
       - Solves Koopman-MPC QP (cvxpy + CLARABEL)
       - Sends: SET_ATTITUDE_TARGET to ArduCopter
```

---

## Prerequisites

### 1. Install Mission Planner (free, Windows native)
Download from: https://ardupilot.org/planner/docs/mission-planner-installation.html

### 2. Python packages
```
pip install pymavlink cvxpy numpy scipy matplotlib
```

### 3. Optional: MAVProxy (already installed)
Used to multiplex the SITL connection.

---

## How to Run

### Step 1 - Start ArduCopter SITL via Mission Planner
1. Open **Mission Planner**
2. Go to **Simulation** tab (top menu)
3. Select **Multirotor** (Copter)
4. Click **Run SITL** - downloads and starts ArduCopter.exe locally
5. Leave Mission Planner open (connects automatically on port 5760)

### Step 2 - Set up the vehicle
```bash
python setup_vehicle.py
```
Arms the drone, sets GUIDED_NOGPS mode, configures parameters.

### Step 3 - Run the Koopman-MPC simulation
```bash
python sitl_controller.py --controller koopman
python sitl_controller.py --controller koopman --scenario wind
python sitl_controller.py --controller koopman --scenario payload
python sitl_controller.py --controller koopman --scenario combined
python sitl_controller.py --controller lqr
python sitl_controller.py --controller pid
```

### Step 4 - Full comparison (all controllers x all scenarios)
```bash
python run_comparison.py
```

### Step 5 - Plot results
```bash
python plot_results.py
```

---

## Folder Structure

| File | Purpose |
|---|---|
| `sitl_controller.py` | Main simulation loop - Koopman-MPC / LQR / PID via MAVLink |
| `koopman_mpc_ardupilot.py` | Koopman-MPC controller (lifted state, QP) |
| `lqr_controller_ardupilot.py` | LQR cascade controller for ArduPilot |
| `mavlink_interface.py` | MAVLink connection, state reader, command sender |
| `setup_vehicle.py` | Arms drone, sets mode and parameters |
| `run_comparison.py` | Automated comparison across all scenarios |
| `plot_results.py` | Tracking error plots, comparison tables |
| `disturbance_injector.py` | Payload and wind disturbance injection |
| `params/copter_params.parm` | ArduCopter parameter file for SITL |
| `data/` | Trained Koopman model (from Phase 2) |
| `results/` | Flight logs (.npz) and plots |
| `logs/` | MAVLink telemetry logs |

---

## Key Design Decisions

### Why SET_ATTITUDE_TARGET, not direct motor RPM?
ArduPilot does not expose individual motor RPMs to external scripts via standard MAVLink.
SET_ATTITUDE_TARGET lets us command roll/pitch/yaw rate + collective thrust, and ArduPilot's
inner attitude PID handles motor mixing. Our Koopman-MPC runs as the **outer position loop**;
ArduPilot stabilises attitude. This matches how real companion-computer controllers work.

### Koopman Model
The trained Koopman model from Phase 2 (v2_stable_koopman_mpc/data/koopman_model.npz)
is used directly. State representation is translated from gym-pybullet ENU to ArduPilot NED.

### Frame Conventions
- ArduPilot: NED frame, attitude in quaternion
- Phase 2 model: ENU-like gym-pybullet frame
- mavlink_interface.py handles frame conversion transparently
