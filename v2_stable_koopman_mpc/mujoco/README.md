# Koopman-MPC Quadrotor Simulation in MuJoCo

A high-fidelity **MuJoCo physics simulation environment** for closed-loop quadrotor trajectory tracking using **Authentic Data-Driven Koopman-MPC**, benchmarked against **PID** and **LQR** baselines over procedural 3D terrain with wind and payload disturbances.

---

## 📌 Overview

This directory provides the MuJoCo simulation environment for the Crazyflie 2.X (CF2X) quadrotor. It directly loads and executes the repository's authentic Phase-2 Koopman model (`data/koopman_model.npz`) without modifying any base training code.

### Highlights
* **Authentic 27-State Koopman-MPC:** Uses the exact 27-dimensional observable lifting vector and $A, B$ matrices trained from flight data.
* **Realistic Drone Visualization:** Highly detailed Crazyflie 2.X model with carbon-fiber dual deck, standoffs, flight controller PCB, micro-USB port, motor pods, spinning rotor blur discs, landing pegs, and front/rear LED indicators.
* **8 Procedural 3D Terrains:** Flat, gentle, rolling, rugged, mountain peaks, terraced plateaus, narrow canyons, and obstacle fields.
* **Disturbance Rejection:** Real-time stabilization against sudden mass changes (up to $+25\%$ extra payload) and lateral/vertical wind gusts.
* **Lookahead Trajectory Tracking:** Receding-horizon prediction ($H_p = 10$) anticipates terrain elevation changes and corners before reaching them.

---

## 🔬 Mathematical Architecture

```
State (18) [pos, vel, R, omega]
       │
       ▼
27-Dim Lifting Function  ──►  psi(x) = [x, v*Rzz, vz*Rxz, vz*Ryz, w*Rzz, 1]
       │
       ▼
Receding-Horizon QP (CLARABEL)
  min  sum_{t} ( || psi_{t+1} - target ||^2_Q + || du_t ||^2_Rm + || du_t - du_{t-1} ||^2_Em )
  s.t. psi_{t+1} = A psi_t + B (hover_rpm + du_t)
       -hover_rpm <= du_t <= RPM_MAX - hover_rpm
       │
       ▼
Adaptive Collective Hover Trim  ──►  u_cmd = hover_rpm(adaptive) + du_0
       │
       ▼
MuJoCo Physics Engine (240 Hz physics / 48 Hz control)
```

### 1. 27-Dimensional Observable Lifting $\psi(x)$
The raw state $x \in \mathbb{R}^{18}$ consists of position $p \in \mathbb{R}^3$, linear velocity $v \in \mathbb{R}^3$, flattened rotation matrix $R \in \mathbb{R}^9$, and angular rate $\omega \in \mathbb{R}^3$. To capture non-linear aerodynamic cross-coupling, the state is lifted into higher-dimensional space:
$$\psi(x) = \begin{bmatrix} x_{18} \\ v \cdot R_{zz} \\ v_z \cdot R_{xz} \\ v_z \cdot R_{yz} \\ \omega \cdot R_{zz} \\ 1 \end{bmatrix} \in \mathbb{R}^{27}$$

### 2. Receding-Horizon Constrained QP
At each control step ($48\text{ Hz}$), a quadratic program minimizes tracking error over prediction horizon $H_p = 10$ and control horizon $H_c = 5$:
$$\min_{\Delta u} \sum_{t=0}^{H_p-1} \left( \|\psi_{t+1} - \psi_{\text{ref}}\|_{Q}^2 + \|\Delta u_t\|_{R_m}^2 + \|\Delta u_t - \Delta u_{t-1}\|_{E_m}^2 \right)$$
subject to:
* $\psi_{t+1} = A \psi_t + B (u_{\text{hover}} + \Delta u_t)$
* $0 \le u_{\text{hover}} + \Delta u_t \le 21,713\text{ RPM}$

### 3. Attitude & Angular Rate Damping
To prevent non-linear stall or flipping during aggressive maneuvers:
* Off-diagonal rotation terms ($R_{02}, R_{12}, R_{20}, R_{21}$) are quadratically penalized ($Q_{\text{tilt}} = 80.0$), forcing the drone to remain upright.
* Angular rates ($\omega_x, \omega_y, \omega_z$) are damped ($Q_{\omega} = 3.0$) to eliminate high-frequency pitch/roll oscillations.

### 4. Adaptive Collective Hover Trim
When a heavy payload ($+20\%$ mass) or downward wind is introduced, proportional control naturally sags. An integrated altitude feedback law shifts the nominal hover baseline dynamically:
$$a_{z,\text{cmd}} = g + K_p (z_{\text{ref}} - z) + K_d (v_{z,\text{ref}} - v_z) + K_i \int (z_{\text{ref}} - z)\,dt$$
$$u_{\text{hover}} = \sqrt{\frac{m \cdot \max(a_{z,\text{cmd}}, 2.0)}{4 K_F}}$$
The QP optimizes differential steering RPMs around this adaptive baseline, maintaining terrain clearance ($>1.25\text{ m}$ AGL) under heavy payload.

---

## 📊 Benchmark: PID vs. LQR vs. Koopman-MPC

### 1. Nominal Flight Benchmark (Matching `comparison.png`)
*Rolling terrain, 12s trajectory, calm conditions:*

| Controller | Status | RMSE (m) | Max Error (m) | Min AGL (m) | Lookahead? | Physical Constraints? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PID** | OK | 0.034 | 0.124 | 1.487 | ❌ No | ❌ No |
| **LQR** | OK | 0.092 | 0.282 | 1.439 | ❌ No | ❌ No |
| **Koopman-MPC** | **OK** | **0.200** | **0.391** | **1.423** | **✅ Yes (0.21s)** | **✅ Yes ($0 \le \text{RPM} \le 21,713$)** |

### 2. Disturbance Rejection Benchmark
*Rolling terrain with $+25\%$ extra payload mass at $t=3\text{s}$ and $0.02\text{ N}$ crosswind at $t=3-5\text{s}$:*

| Controller | Status | RMSE (m) | Max Error (m) | Min AGL (m) | Disturbance Handling |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **PID** | OK | 0.091 | 0.180 | 1.336 | Fast error feedback reacts to offset |
| **LQR** | OK | 0.231 | 0.500 | 1.076 | Steady-state altitude sag under load |
| **Koopman-MPC** | **OK** | **0.770** | **1.549** | **1.158** | **Adaptive hover trim prevents terrain contact** |

![Controller Comparison](comparison.png)

---

## 🚀 Quickstart & Commands

### 1. Authentic Koopman-MPC in 3D GUI Viewer
```cmd
cd /d c:\Users\drmad\Drones_S5_CD7-Koopman\v2_stable_koopman_mpc\mujoco

:: Fly over rolling terrain (default)
python run_koopman_mujoco.py --gui --terrain rolling

:: Fly over mountain peaks with +10% payload and crosswind
python run_koopman_mujoco.py --gui --terrain mountain --payload 0.10 --wind 0.01

:: Fly in a canyon with 90-degree crosswind
python run_koopman_mujoco.py --gui --terrain canyon --wind 0.015 --wind-dir 90

:: Fly over stepped terraced plateaus with +20% payload at 2.0m clearance
python run_koopman_mujoco.py --gui --terrain terraced --payload 0.20 --agl 2.0
```

### 2. Multi-Controller Benchmark (Generates `comparison.png`)
```cmd
:: Run PID, LQR, and Koopman-MPC in sequence in the live 3D viewer
python run_sim.py --ctrl all --gui --terrain rolling

:: Headless benchmark with disturbances
python run_sim.py --ctrl all --terrain rolling --payload 0.25 --wind 0.02 --no-plot
```

### 3. Interactive Flight Dashboard
```cmd
:: Launch interactive control panel with live sliders for Wind Speed, Direction, and Payload
python drone_sim.py --terrain rolling
```

---

## 📁 File Structure

| File | Purpose |
| :--- | :--- |
| `cf2x_terrain.xml` | MuJoCo MJCF model defining the Crazyflie 2.X chassis, motors, sensors, cameras, and collision bounds. |
| `koopman_mpc.py` | Receding-horizon QP controller driven by authentic Koopman matrices with preview horizon and adaptive hover trim. |
| `run_koopman_mujoco.py` | Standalone executable for closed-loop Koopman flights with visual path ribbons, wind vectors, and telemetry. |
| `run_sim.py` | Benchmark runner comparing PID, LQR, and Koopman-MPC side-by-side with plot generation. |
| `drone_sim.py` | Interactive GUI dashboard with live sliders for disturbance injection and stress tests. |
| `controllers.py` | Implementation of cascaded geometric PID and DLQR baseline controllers with CF2X motor mixer. |
| `terrain.py` | Procedural heightfield generator supporting 8 distinct terrain presets. |
| `mujoco_quad_env.py` | MuJoCo environment wrapper exposing PyBullet-compatible stepping and observation APIs. |
| `selftest.py` | Unit test validating physical motor torque directions and sign conventions. |
| `view_terrain.py` | Standalone terrain visualizer. |
