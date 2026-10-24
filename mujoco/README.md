<p align="center">
  <img src="amrita_vishwa_vidyapeetham_logo.png" alt="Amrita Vishwa Vidyapeetham logo" width="100%">
</p>

# Koopman-MPC Quadrotor Simulation in MuJoCo

<p align="center">
  <img src="mujoco_simulation_screenshot.png" alt="MuJoCo simulation viewer showing the quadrotor and visualized flight paths" width="90%">
</p>

<p align="center"><em>Interactive MuJoCo simulation: the quadrotor is shown with visual trajectory/reference markers in the scene.</em></p>

> **How to read this README:** Start with the project overview and simulation workflow, then use the file guide and controller explanations to understand how the code fits together. The comparison gallery later in this document contains the saved experiments; each figure is labeled by terrain, payload fraction, and wind magnitude.

## What this project does

This project studies how a small quadrotor can follow a moving 3-D target while flying over terrain whose height changes. The simulation uses **MuJoCo** to calculate the drone's motion from its model, motor thrust, gravity, and contact with the ground. Python code supplies the flight reference, reads the drone's state, computes motor commands, and records the result.

The central controller is **Koopman Model Predictive Control (Koopman-MPC)**. Instead of predicting the drone directly with a fully nonlinear model, the project represents its state using a set of features called *lifted observables*. A linear model is fitted to state-transition data in this feature space. At every control step, the MPC controller uses that model to predict future behavior and chooses motor commands intended to reduce tracking error while keeping commands within limits. Only the first action is applied; the state is measured again and the optimization is solved again.

The project also includes two classical comparison controllers:

- **PID:** a cascaded proportional–integral–derivative controller. Position errors are converted into desired motion, then an attitude/rate controller and a motor mixer produce four motor RPM commands.
- **LQR:** a cascaded controller whose feedback gains are obtained from a discrete-time linear-quadratic regulator design for simplified double-integrator models. The implementation also includes a small integral term to reduce steady offsets under load.
- **Koopman-MPC:** a predictive controller using the fitted lifted-state model and a constrained quadratic optimization problem.

The point of the comparison is not to assume that one controller always wins. It is to observe how each controller follows the same reference when the terrain, added mass, or externally applied wind force changes. The plots show the reference path, each controller's path, altitude behavior, position error, and clearance above the terrain.

### Simulation loop at a glance

```text
Terrain + XML drone model
          |
          v
MuJoCo state: position, velocity, attitude, angular velocity
          |
          v
Reference generator: x/y path + terrain height + desired clearance
          |
          v
Controller: PID, LQR, or Koopman-MPC
          |
          v
Four motor RPM commands (clipped to the allowed range)
          |
          v
MuJoCo advances physics for several substeps
          |
          +------> record state, tracking error and altitude clearance
          |
          +------> repeat until the episode ends or a failure is detected
```

The XML model defines the drone and terrain collision geometry. `mujoco_quad_env.py` is the Python adapter between that model and the controllers. `terrain.py` creates terrain height fields. `run_koopman_mujoco.py` runs the Koopman-MPC controller, `run_sim.py` runs the requested controller benchmark, and `drone_sim.py` launches the interactive dashboard.

## Project files and how they work together

All paths below are relative to the top-level `mujoco/` directory:

| File | Role in the complete workflow |
|---|---|
| `cf2x_terrain.xml` | Main MuJoCo scene: quadrotor body, rotors/actuators, terrain height field, reference marker, and visual elements. |
| `mjmodel.xml` | Additional MuJoCo XML asset in this directory. |
| `mujoco_quad_env.py` | Loads the XML, configures physics and control rates, exposes the 18-value state, applies motor thrust, updates the terrain, injects payload/wind, and computes altitude above ground. |
| `terrain.py` | Generates terrain height fields and provides terrain-height and terrain-preset helpers. |
| `koopman_mpc.py` | Defines the lifting function, gathers training transitions, fits the lifted linear model, saves/loads model matrices, and implements the Koopman-MPC optimizer. |
| `controllers.py` | Implements the cascaded PID and LQR baselines and maps total thrust/torque requests to four motor RPMs. |
| `run_koopman_mujoco.py` | Runs a single Koopman-MPC flight, draws the target and flight trails in the viewer, applies optional disturbances, logs metrics, and saves `mujoco_run_koopman.npz`. |
| `run_sim.py` | Runs PID, LQR, Koopman-MPC, or all three sequentially. It prints RMSE, maximum error and minimum AGL, and saves `comparison.png`. |
| `run_pid_mujoco.py`, `run_lqr_mujoco.py` | Convenience launchers for the individual baseline controllers. |
| `drone_sim.py` | Interactive flight dashboard, including terrain and live disturbance controls. |
| `selftest.py`, `test_koopman_base_in_mujoco.py` | `selftest.py` checks the local model/motor setup. `test_koopman_base_in_mujoco.py` tests the base Koopman-MPC matrices using `../data/koopman_model.npz`. |
| `run_all_terrain.py` | Runs controller experiments across terrain presets. |
| `generate_wind_payload_analysis.py` | Analyzes existing run logs for wind/payload effects. |
| `analysis/run_all_comparisons.py` | Runs and stores one controller-comparison figure. |
| `analysis/run_batch_comparisons.py` | Runs a grid of terrain, payload and wind configurations and stores each resulting PNG in `analysis/comparisons/`. |
| `analysis/generate_summary_plots.py` | Summarizes existing CSV logs and creates metric plots/report output. |
| `analysis/generate_plots.py` | Produces plots from a summary CSV. |
| `analysis/run_pipeline.py` | Optional plotting/report pipeline for an existing summary CSV. |
| `analysis/comparisons/` | Contains the 36 saved four-panel comparison figures discussed below. |
| `comparison.png`, `comparison_nominal.png`, `stress_summary.png` | Additional figures already saved in the directory. |

## The simulation model

### Physics and timing

The environment uses the Crazyflie-style quadrotor model defined by the XML and sets gravity to \(g=9.8\,\mathrm{m/s^2}\). The defaults in `MujocoQuadEnv` are 240 Hz for physics and 48 Hz for control. Thus the physics step is

\[
\Delta t_{\mathrm{physics}}=\frac{1}{240}\ \mathrm{s},
\]

and each control command is held for five physics substeps, giving

\[
\Delta t_{\mathrm{control}}=\frac{5}{240}=\frac{1}{48}\ \mathrm{s}.
\]

This separation lets MuJoCo integrate the physical motion at a finer time step than the controller update.

### State vector

The environment returns an 18-dimensional state:

\[
x_k =
\begin{bmatrix}
p_k\\
v_k\\
\operatorname{vec}(R_k)\\
\omega_k
\end{bmatrix}
\in\mathbb{R}^{18}.
\]

Here \(p_k=[x_k,y_k,z_k]^\mathsf{T}\) is position, \(v_k=[v_x,v_y,v_z]^\mathsf{T}\) is linear velocity, \(R_k\in\mathbb{R}^{3\times3}\) is the body-to-world rotation matrix flattened in row-major order, and \(\omega_k\in\mathbb{R}^3\) is angular velocity represented in world coordinates by the environment adapter. The flattened rotation matrix has nine values, so \(3+3+9+3=18\).

MuJoCo internally stores the free-joint orientation as a quaternion. The environment converts that orientation to a rotation matrix for the controller's state vector.

### Motor model and action

The environment accepts four rotor RPM values. The thrust applied by rotor \(i\) is modeled as

\[
T_i=k_f\,\mathrm{RPM}_i^2,
\]

where \(k_f\) is the thrust coefficient defined by the environment. The sum of the four thrusts supports the drone's weight; differences between rotor thrusts create roll, pitch and yaw moments. The environment clips commands to

\[
0\leq \mathrm{RPM}_i\leq \mathrm{RPM}_{\max}.
\]

The hover RPM used by the environment is calculated from the nominal mass and gravity:

\[
\mathrm{RPM}_{\mathrm{hover}}
=\sqrt{\frac{mg}{4k_f}}.
\]

The implementation uses \(m=0.027\,\mathrm{kg}\), \(g=9.8\,\mathrm{m/s^2}\), and \(k_f=3.16\times10^{-10}\) in its RPM-based thrust formula (\(\mathrm{RPM}_{\mathrm{hover}} \approx 14,468\)). The exact values and units are matched to the XML actuator configuration.

### Terrain and altitude above ground

`terrain.py` creates the height field used by MuJoCo. The terrain height at horizontal location \((x,y)\) is \(h_{\mathrm{ground}}(x,y)\). The reference trajectory follows a sinusoidal horizontal path and sets its desired altitude to the local ground height plus a target clearance:

\[
z_{\mathrm{ref}}(t)=h_{\mathrm{ground}}\bigl(x_{\mathrm{ref}}(t),y_{\mathrm{ref}}(t)\bigr)+h_{\mathrm{target}}.
\]

The code rate-limits changes to the altitude reference so that a sudden change in terrain height does not cause an instantaneous jump in the target. The drone's altitude above ground level is

\[
h_{\mathrm{AGL}}(t)=z(t)-h_{\mathrm{ground}}\bigl(x(t),y(t)\bigr).
\]

AGL is a clearance measure, not the same thing as world-frame altitude \(z\). A drone can have constant world altitude but changing AGL when the terrain rises or falls beneath it.

Terrain presets in `terrain.py` include `flat`, `gentle`, `rolling`, `rugged`, `mountain`, `terraced`, `canyon`, and `obstacles`. The 36 saved comparison images use four of these presets: canyon, flat, mountain and rolling.

### Payload and wind disturbances

A payload fraction of `0.20` adds 20% of the nominal drone mass. The environment changes the drone body's mass but leaves its inertia unchanged; this is the specific simplified disturbance model implemented here.

For `run_koopman_mujoco.py`, `--wind` is a horizontal force in newtons, and `--wind-dir` gives its direction in degrees: \(0^\circ\) points along \(+x\), and \(90^\circ\) points along \(+y\). The runner applies the configured force from \(t=3\) s until \(t=6\) s. By contrast, `run_sim.py` applies its wind force along \(+x\) from \(t=3\) s until \(t=5\) s and has no `--wind-dir` option. Use the right runner when interpreting or recreating a particular experiment.

The dashboard has its own wind model and interactive controls; its wind settings should not be assumed identical to the constant-force disturbance in the two command-line runners.

## The three controllers

### PID controller

The PID baseline uses a cascaded architecture. An outer position loop computes a desired acceleration from position error, velocity error and an accumulated integral term. That acceleration is converted into a desired thrust vector and attitude. An inner attitude/rate loop computes desired torques, and the mixer converts total thrust and torques into four motor thrusts and then RPM commands.

A typical one-axis PID relation is

\[
a_{\mathrm{cmd}}=K_p e_p+K_d e_v+K_i\int e_p\,dt,
\]

where \(e_p=p_{\mathrm{ref}}-p\) and \(e_v=v_{\mathrm{ref}}-v\). The code filters the reference-velocity estimate and clips the integrator and horizontal acceleration to avoid unbounded commands.

### LQR controller

The LQR baseline uses a simplified discrete double-integrator model for each axis. For the state \(q_k=[e_k,\dot e_k]^\mathsf{T}\), the code uses

\[
q_{k+1}=Aq_k+Ba_k,\qquad
A=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix},\quad
B=\begin{bmatrix}\tfrac{1}{2}\Delta t^2\\\Delta t\end{bmatrix}.
\]

The gain \(K\) is obtained from the discrete algebraic Riccati equation for a quadratic state/input cost. Feedback is applied in the cascaded controller structure, with a small integral term enabled by default to help trim load-related offsets. Therefore, the implemented baseline is an LQR-based cascade with integral correction, not a claim that every part of the nonlinear drone is globally linear.

### Koopman-MPC controller

#### Lifting the state

The Koopman approach maps the 18-dimensional state into 27 features. The implementation in `koopman_mpc.py` constructs

\[
\psi(x)=
\begin{bmatrix}
x\\
v_xR_{zz}\\
v_yR_{zz}\\
v_zR_{zz}\\
v_zR_{xz}\\
v_zR_{yz}\\
\omega_xR_{zz}\\
\omega_yR_{zz}\\
\omega_zR_{zz}\\
1
\end{bmatrix}
\in\mathbb{R}^{27}.
\]

The extra features encode selected interactions between velocity/angular velocity and orientation. The constant feature is a bias term. This feature map does not make the underlying physical system exactly linear; it gives regression a richer set of signals from which to approximate its transitions.

#### Learning the linear predictor

The training routine runs PID-controlled trajectories on flat terrain while varying reference motion, motor excitation, payload and wind. It collects pairs of consecutive lifted states and the associated normalized motor input. Episodes that crash or diverge are discarded. The fitted predictor is

\[
\psi(x_{k+1})\approx A\psi(x_k)+Bu_k,
\]

where \(A\in\mathbb{R}^{27\times27}\), \(B\in\mathbb{R}^{27\times4}\), and the training input is

\[
u_k=\frac{\mathrm{RPM}_k-\mathrm{RPM}_{\mathrm{hover}}}
{\mathrm{RPM}_{\mathrm{hover}}}.
\]

The matrices are estimated with a ridge-regularized least-squares fit. In generic notation, if each row of \(Z\) contains the current lifted state and normalized input, and each row of \(Y\) contains the next lifted state, the fit solves a regularized regression problem of the form

\[
\Theta^\star=\arg\min_\Theta
\left\|Y-Z\Theta\right\|_F^2+\lambda\|\Theta\|_F^2.
\]

The resulting matrices are stored in `data/koopman_model.npz` and `koopman_mujoco_model.npz` and loaded automatically when the controller starts. If you wish to retrain the local model from scratch, run `python koopman_mpc.py` from this directory to collect new simulated transitions and re-fit the matrices.

#### Receding-horizon MPC

At each update, the controller starts from the current lifted state and predicts a sequence of future lifted states. It minimizes a quadratic cost that penalizes position error, linear velocity, selected tilt-related rotation entries, angular velocity and changes in motor commands. In the implementation, the state-cost matrix \(Q\) is diagonal, with nonzero weights on the position entries \(0{:}3\), velocity entries \(3{:}6\), selected rotation entries \(8,11,12,13\), and angular velocity entries \(15{:}18\).

The optimization uses a prediction horizon \(H_p=10\) and a control horizon \(H_c=5\). Beyond the control horizon, the input increments are held at the final optimized increment. The solver is CVXPY with CLARABEL. Only the first action from the optimized sequence is applied; then the next state is observed and the optimization repeats.

**Important code note on input scaling:** In `koopman_mpc.py`, the training data uses the normalized input \(u=(\mathrm{RPM}-\mathrm{RPM}_{\mathrm{hover}})/\mathrm{RPM}_{\mathrm{hover}}\), while the MPC prediction constraint inside the receding-horizon optimization passes `hover_rpm + du[t]` into the linear model. These operate on different scales. In practice, the controller also includes an altitude compensation calculation in `KoopmanMPC.rpm`: it updates a collective hover baseline using vertical position/velocity error and a bounded integral term. This helps the controller respond to mass and vertical disturbances, complementing the lifted model and MPC optimization.

## Installation

### Requirements

Use Python 3.10 or 3.11 in a virtual environment. The core simulation and plotting workflow uses:

- `numpy` — numerical arrays and linear algebra.
- `mujoco` — rigid-body simulation, actuators, collision/contact calculation, and 3-D viewer.
- `cvxpy` — expresses the MPC optimization problem.
- `clarabel` — optimization solver used by the Koopman-MPC controller.
- `scipy` — discrete algebraic Riccati equation used by the LQR baseline.
- `pandas` — analysis of saved CSV logs, where applicable.
- `matplotlib` — the four-panel controller comparison and summary plots.

### Windows (PowerShell / Command Prompt)

Run from the repository root:

```powershell
cd mujoco
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy mujoco cvxpy clarabel scipy pandas matplotlib
```

If PowerShell blocks script execution, use Command Prompt:

```cmd
cd mujoco
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install numpy mujoco cvxpy clarabel scipy pandas matplotlib
```

### Linux or macOS

```bash
cd mujoco
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy mujoco cvxpy clarabel scipy pandas matplotlib
```

### Verify imports

```bash
python -c "import numpy, mujoco, cvxpy, clarabel, scipy, pandas, matplotlib; print('All required imports succeeded')"
python selftest.py
```

The self-test checks model and motor setup.

## Run the simulations

**Run every command from the `mujoco/` directory** after activating the environment. The pre-trained model matrices are loaded automatically from `../data/koopman_model.npz` or local `koopman_mujoco_model.npz`.

### 1. Live 3-D flight of Koopman-MPC over rolling terrain

```bash
python run_koopman_mujoco.py --gui --terrain rolling
```

This opens the MuJoCo viewer and runs the Koopman-MPC controller on rolling terrain. The reference moves in the horizontal plane along a sinusoidal path. The reference altitude is based on the local terrain height plus the default 1.5 m target clearance. The viewer shows the drone and reference marker; the runner draws a red trail for the actual flight and a green trail for the reference. The terminal prints position, AGL, tilt, mean RPM and tracking error. The episode runs for 12 seconds by default and saves the log to `mujoco_run_koopman.npz`.

### 2. Test disturbance rejection (+20% payload mass and 90-degree crosswind)

```bash
python run_koopman_mujoco.py --gui --terrain rolling --payload 0.20 --wind 0.015 --wind-dir 90
```

This runs the same terrain-following task, but adds 20% to the drone's nominal body mass at \(t=3\) s and applies a horizontal force of \(0.015\) N along \(+y\) from \(t=3\) s to \(t=6\) s. The cyan line in the viewer indicates the wind-force direction during the gust. Watch whether the actual trajectory separates from the green reference trail at disturbance onset and whether it returns toward the reference after the gust ends.

### 3. Test on steep mountain peaks

```bash
python run_koopman_mujoco.py --gui --terrain mountain
```

This selects the mountain terrain preset. The reference altitude follows the local terrain height plus the desired clearance, with its climb/descent rate limited by the runner. Observe the drone as the reference crosses high and low terrain areas. The important checks are whether it follows the reference, maintains positive clearance, avoids terrain contact, and stays upright.

### 4. Run the 3-controller benchmark (PID vs LQR vs Koopman-MPC, generates `comparison.png`)

```bash
python run_sim.py --ctrl all --gui --terrain rolling
```

The script runs PID, LQR and Koopman-MPC sequentially on the same rolling-terrain setup. With `--gui`, a viewer is opened for each controller run; close a viewer when that run is finished so the next controller can be tested. At the end, the terminal prints each controller's status, position-error RMSE, maximum position error and minimum AGL. The script saves `comparison.png` in the current directory.

The figure has four panels:

1. **Top-down path:** horizontal \(x\)-\(y\) reference and each controller's actual path. Curves closer to the reference indicate better horizontal tracking.
2. **Altitude follows the terrain:** actual \(z\), reference \(z\), and ground height under the drone over time. This panel shows how world altitude changes as the terrain changes.
3. **3-D tracking error:** Euclidean distance between actual and reference positions at each time step.
4. **Height above ground (AGL):** actual clearance over the terrain directly below the drone. Lower values mean the drone is closer to the surface.

### 5. Launch interactive flight dashboard with real-time sliders

```bash
python drone_sim.py --terrain rolling
```

This opens the interactive dashboard. It is intended for hands-on exploration of the drone, terrain and disturbance settings using the controls provided by the application. Select or adjust the available controls, start the flight, and observe the resulting state and trajectory.

### Other useful options

`run_koopman_mujoco.py` supports `--terrain`, `--agl`, `--amp`, `--T`, `--seed`, `--payload`, `--wind`, and `--wind-dir`. For example:

```bash
python run_koopman_mujoco.py --gui --terrain canyon --agl 1.2 --T 12
python run_koopman_mujoco.py --gui --terrain terraced --agl 1.2
python run_koopman_mujoco.py --gui --terrain flat
```

The recognized terrain names are `flat`, `gentle`, `rolling`, `rugged`, `mountain`, `terraced`, `canyon`, and `obstacles`.

To run the controller benchmark without opening the viewer, omit `--gui`:

```bash
python run_sim.py --ctrl all --terrain rolling
```

To optionally retrain the model before the benchmark:

```bash
python run_sim.py --ctrl all --retrain --terrain rolling
```

## How the benchmark metrics are calculated

Let the actual position be \(p_k\) and the reference position be \(p_k^{\mathrm{ref}}\). The instantaneous 3-D tracking error is

\[
e_k=\left\|p_k-p_k^{\mathrm{ref}}\right\|_2
=\sqrt{(x_k-x_k^{\mathrm{ref}})^2+(y_k-y_k^{\mathrm{ref}})^2+(z_k-z_k^{\mathrm{ref}})^2}.
\]

The position-error RMSE over \(N\) logged samples is

\[
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{k=1}^{N}e_k^2}.
\]

RMSE measures typical error over the full run; the maximum error shows the largest observed deviation. Minimum AGL is

\[
\min_k h_{\mathrm{AGL},k},
\]

the smallest logged distance between the drone's reference point and the terrain height directly below it. Interpret all three together: a low RMSE does not rule out a brief large error, and minimum AGL says nothing about horizontal tracking by itself.

## The 36 saved comparison figures

### How each figure was generated

`analysis/run_batch_comparisons.py` runs `run_sim.py --ctrl all --gui` for every combination of four terrain presets, three payload values and three wind values. It moves each generated `comparison.png` into `analysis/comparisons/` and gives it a filename that encodes the configuration.

The 36 figures form a \(4\times3\times3\) grid:

- **Terrain:** canyon, flat, mountain, rolling.
- **Payload fraction:** `0.0` (nominal mass), `0.2` (+20% mass), `0.4` (+40% mass).
- **Wind force:** `0.0`, `0.015`, `0.03` N. In the `run_sim.py` batch workflow, the wind is applied along \(+x\) between \(t=3\) s and \(t=5\) s.

The panels show horizontal path tracking, altitude relative to the changing terrain, 3-D tracking error and AGL. Note: `comparison_mountain_payload0.0_wind0.0.png` contains PID and LQR traces for that specific run.

- On **flat** terrain, ground height is constant, so altitude and AGL are closely aligned.
- On **rolling** terrain, the reference altitude smoothly follows hills and valleys.
- On **mountain** terrain, steeper slope changes test vertical authority and clearance limits.
- In a **canyon**, sudden transitions near elevated corridor walls test tracking precision.
- Increasing **payload** tests vertical tracking and baseline thrust trim.
- Increasing **wind** evaluates lateral disturbance rejection during the active gust window.

### Canyon terrain

<details>
<summary><strong>comparison_canyon_payload0.0_wind0.0.png</strong> — Nominal mass baseline; wind 0.0 N</summary>

![comparison_canyon_payload0.0_wind0.0.png](analysis/comparisons/comparison_canyon_payload0.0_wind0.0.png)

**What is happening in this experiment:** The drone tracks a path above the canyon height field. The reference altitude follows the ground-height profile, so the altitude panel shows how the target changes near the corridor and raised sides.

**Payload condition:** No added payload is applied (nominal-mass baseline).

**Wind condition:** No wind force is injected in this run.

**How to interpret the panels:**
1. **Top-down path:** compare the reference line with PID, LQR and Koopman-MPC paths.
2. **Altitude:** compare actual world-frame height with the reference altitude and the ground-height trace.
3. **3-D tracking error:** identify the time and size of error peaks.
4. **AGL:** inspect the lowest point of each trace to assess clearance.

**Result from the saved figure title (3-D position RMSE):** PID = 0.034 m, LQR = 0.092 m, Koopman-MPC = 0.201 m.

</details>

<details>
<summary><strong>comparison_canyon_payload0.0_wind0.015.png</strong> — Nominal mass baseline; wind 0.015 N</summary>

![comparison_canyon_payload0.0_wind0.015.png](analysis/comparisons/comparison_canyon_payload0.0_wind0.015.png)

**What is happening in this experiment:** Canyon tracking with nominal mass under a 0.015 N horizontal wind gust applied from \(t = 3\) s to \(t = 5\) s.

**Result from the saved figure title (3-D position RMSE):** PID = 0.047 m, LQR = 0.130 m, Koopman-MPC = 0.348 m.

</details>

<details>
<summary><strong>comparison_canyon_payload0.0_wind0.03.png</strong> — Nominal mass baseline; wind 0.03 N</summary>

![comparison_canyon_payload0.0_wind0.03.png](analysis/comparisons/comparison_canyon_payload0.0_wind0.03.png)

**What is happening in this experiment:** Canyon tracking under strong 0.03 N crosswind gust during \(t = 3\) to \(5\) s.

**Result from the saved figure title (3-D position RMSE):** PID = 0.065 m, LQR = 0.187 m, Koopman-MPC = 0.548 m.

</details>

<details>
<summary><strong>comparison_canyon_payload0.2_wind0.0.png</strong> — +20% payload mass; wind 0.0 N</summary>

![comparison_canyon_payload0.2_wind0.0.png](analysis/comparisons/comparison_canyon_payload0.2_wind0.0.png)

**What is happening in this experiment:** Canyon tracking with 20% mass increase injected at \(t = 3\) s.

**Result from the saved figure title (3-D position RMSE):** PID = 0.067 m, LQR = 0.166 m, Koopman-MPC = 0.213 m.

</details>

<details>
<summary><strong>comparison_canyon_payload0.2_wind0.015.png</strong> — +20% payload mass; wind 0.015 N</summary>

![comparison_canyon_payload0.2_wind0.015.png](analysis/comparisons/comparison_canyon_payload0.2_wind0.015.png)

**What is happening in this experiment:** Canyon tracking with simultaneous 20% payload and 0.015 N wind gust.

**Result from the saved figure title (3-D position RMSE):** PID = 0.075 m, LQR = 0.192 m, Koopman-MPC = 0.329 m.

</details>

<details>
<summary><strong>comparison_canyon_payload0.2_wind0.03.png</strong> — +20% payload mass; wind 0.03 N</summary>

![comparison_canyon_payload0.2_wind0.03.png](analysis/comparisons/comparison_canyon_payload0.2_wind0.03.png)

**What is happening in this experiment:** Canyon tracking with +20% mass and maximum 0.03 N wind gust.

**Result from the saved figure title (3-D position RMSE):** PID = 0.088 m, LQR = 0.237 m, Koopman-MPC = 0.486 m.

</details>

<details>
<summary><strong>comparison_canyon_payload0.4_wind0.0.png</strong> — +40% payload mass; wind 0.0 N</summary>

![comparison_canyon_payload0.4_wind0.0.png](analysis/comparisons/comparison_canyon_payload0.4_wind0.0.png)

**What is happening in this experiment:** Heavy +40% cargo load injected at \(t = 3\) s in canyon corridor.

**Result from the saved figure title (3-D position RMSE):** PID = 0.121 m, LQR = 0.292 m, Koopman-MPC = 0.254 m.

</details>

<details>
<summary><strong>comparison_canyon_payload0.4_wind0.015.png</strong> — +40% payload mass; wind 0.015 N</summary>

![comparison_canyon_payload0.4_wind0.015.png](analysis/comparisons/comparison_canyon_payload0.4_wind0.015.png)

**What is happening in this experiment:** +40% payload disturbance combined with 0.015 N crosswind.

**Result from the saved figure title (3-D position RMSE):** PID = 0.126 m, LQR = 0.309 m, Koopman-MPC = 0.340 m.

</details>

<details>
<summary><strong>comparison_canyon_payload0.4_wind0.03.png</strong> — +40% payload mass; wind 0.03 N</summary>

![comparison_canyon_payload0.4_wind0.03.png](analysis/comparisons/comparison_canyon_payload0.4_wind0.03.png)

**What is happening in this experiment:** Maximum disturbance scenario in canyon: +40% mass and 0.03 N wind gust.

**Result from the saved figure title (3-D position RMSE):** PID = 0.134 m, LQR = 0.341 m, Koopman-MPC = 0.462 m.

</details>

### Flat terrain

<details>
<summary><strong>comparison_flat_payload0.0_wind0.0.png</strong> — Nominal mass baseline; wind 0.0 N</summary>

![comparison_flat_payload0.0_wind0.0.png](analysis/comparisons/comparison_flat_payload0.0_wind0.0.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.034 m, LQR = 0.092 m, Koopman-MPC = 0.202 m.

</details>

<details>
<summary><strong>comparison_flat_payload0.0_wind0.015.png</strong> — Nominal mass baseline; wind 0.015 N</summary>

![comparison_flat_payload0.0_wind0.015.png](analysis/comparisons/comparison_flat_payload0.0_wind0.015.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.047 m, LQR = 0.130 m, Koopman-MPC = 0.350 m.

</details>

<details>
<summary><strong>comparison_flat_payload0.0_wind0.03.png</strong> — Nominal mass baseline; wind 0.03 N</summary>

![comparison_flat_payload0.0_wind0.03.png](analysis/comparisons/comparison_flat_payload0.0_wind0.03.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.065 m, LQR = 0.187 m, Koopman-MPC = 0.548 m.

</details>

<details>
<summary><strong>comparison_flat_payload0.2_wind0.0.png</strong> — +20% payload mass; wind 0.0 N</summary>

![comparison_flat_payload0.2_wind0.0.png](analysis/comparisons/comparison_flat_payload0.2_wind0.0.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.068 m, LQR = 0.166 m, Koopman-MPC = 0.214 m.

</details>

<details>
<summary><strong>comparison_flat_payload0.2_wind0.015.png</strong> — +20% payload mass; wind 0.015 N</summary>

![comparison_flat_payload0.2_wind0.015.png](analysis/comparisons/comparison_flat_payload0.2_wind0.015.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.075 m, LQR = 0.193 m, Koopman-MPC = 0.330 m.

</details>

<details>
<summary><strong>comparison_flat_payload0.2_wind0.03.png</strong> — +20% payload mass; wind 0.03 N</summary>

![comparison_flat_payload0.2_wind0.03.png](analysis/comparisons/comparison_flat_payload0.2_wind0.03.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.088 m, LQR = 0.238 m, Koopman-MPC = 0.487 m.

</details>

<details>
<summary><strong>comparison_flat_payload0.4_wind0.0.png</strong> — +40% payload mass; wind 0.0 N</summary>

![comparison_flat_payload0.4_wind0.0.png](analysis/comparisons/comparison_flat_payload0.4_wind0.0.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.121 m, LQR = 0.293 m, Koopman-MPC = 0.255 m.

</details>

<details>
<summary><strong>comparison_flat_payload0.4_wind0.015.png</strong> — +40% payload mass; wind 0.015 N</summary>

![comparison_flat_payload0.4_wind0.015.png](analysis/comparisons/comparison_flat_payload0.4_wind0.015.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.126 m, LQR = 0.310 m, Koopman-MPC = 0.341 m.

</details>

<details>
<summary><strong>comparison_flat_payload0.4_wind0.03.png</strong> — +40% payload mass; wind 0.03 N</summary>

![comparison_flat_payload0.4_wind0.03.png](analysis/comparisons/comparison_flat_payload0.4_wind0.03.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.135 m, LQR = 0.342 m, Koopman-MPC = 0.463 m.

</details>

### Mountain terrain

<details>
<summary><strong>comparison_mountain_payload0.0_wind0.0.png</strong> — Nominal mass baseline; wind 0.0 N</summary>

![comparison_mountain_payload0.0_wind0.0.png](analysis/comparisons/comparison_mountain_payload0.0_wind0.0.png)

**What is happening in this experiment:** Steep peak tracking. Note: this particular logged figure shows PID and LQR baseline traces.

**Result from the saved figure title (3-D position RMSE):** PID = 0.039 m, LQR = 0.121 m.

</details>

<details>
<summary><strong>comparison_mountain_payload0.0_wind0.015.png</strong> — Nominal mass baseline; wind 0.015 N</summary>

![comparison_mountain_payload0.0_wind0.015.png](analysis/comparisons/comparison_mountain_payload0.0_wind0.015.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.050 m, LQR = 0.136 m, Koopman-MPC = 0.350 m.

</details>

<details>
<summary><strong>comparison_mountain_payload0.0_wind0.03.png</strong> — Nominal mass baseline; wind 0.03 N</summary>

![comparison_mountain_payload0.0_wind0.03.png](analysis/comparisons/comparison_mountain_payload0.0_wind0.03.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.067 m, LQR = 0.191 m, Koopman-MPC = 0.555 m.

</details>

<details>
<summary><strong>comparison_mountain_payload0.2_wind0.0.png</strong> — +20% payload mass; wind 0.0 N</summary>

![comparison_mountain_payload0.2_wind0.0.png](analysis/comparisons/comparison_mountain_payload0.2_wind0.0.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.074 m, LQR = 0.176 m, Koopman-MPC = 0.213 m.

</details>

<details>
<summary><strong>comparison_mountain_payload0.2_wind0.015.png</strong> — +20% payload mass; wind 0.015 N</summary>

![comparison_mountain_payload0.2_wind0.015.png](analysis/comparisons/comparison_mountain_payload0.2_wind0.015.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.081 m, LQR = 0.201 m, Koopman-MPC = 0.330 m.

</details>

<details>
<summary><strong>comparison_mountain_payload0.2_wind0.03.png</strong> — +20% payload mass; wind 0.03 N</summary>

![comparison_mountain_payload0.2_wind0.03.png](analysis/comparisons/comparison_mountain_payload0.2_wind0.03.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.093 m, LQR = 0.245 m, Koopman-MPC = 0.492 m.

</details>

<details>
<summary><strong>comparison_mountain_payload0.4_wind0.0.png</strong> — +40% payload mass; wind 0.0 N</summary>

![comparison_mountain_payload0.4_wind0.0.png](analysis/comparisons/comparison_mountain_payload0.4_wind0.0.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.127 m, LQR = 0.302 m, Koopman-MPC = 0.256 m.

</details>

<details>
<summary><strong>comparison_mountain_payload0.4_wind0.015.png</strong> — +40% payload mass; wind 0.015 N</summary>

![comparison_mountain_payload0.4_wind0.015.png](analysis/comparisons/comparison_mountain_payload0.4_wind0.015.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.132 m, LQR = 0.318 m, Koopman-MPC = 0.343 m.

</details>

<details>
<summary><strong>comparison_mountain_payload0.4_wind0.03.png</strong> — +40% payload mass; wind 0.03 N</summary>

![comparison_mountain_payload0.4_wind0.03.png](analysis/comparisons/comparison_mountain_payload0.4_wind0.03.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.140 m, LQR = 0.350 m, Koopman-MPC = 0.468 m.

</details>

### Rolling terrain

<details>
<summary><strong>comparison_rolling_payload0.0_wind0.0.png</strong> — Nominal mass baseline; wind 0.0 N</summary>

![comparison_rolling_payload0.0_wind0.0.png](analysis/comparisons/comparison_rolling_payload0.0_wind0.0.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.034 m, LQR = 0.092 m, Koopman-MPC = 0.200 m.

</details>

<details>
<summary><strong>comparison_rolling_payload0.0_wind0.015.png</strong> — Nominal mass baseline; wind 0.015 N</summary>

![comparison_rolling_payload0.0_wind0.015.png](analysis/comparisons/comparison_rolling_payload0.0_wind0.015.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.047 m, LQR = 0.131 m, Koopman-MPC = 0.350 m.

</details>

<details>
<summary><strong>comparison_rolling_payload0.0_wind0.03.png</strong> — Nominal mass baseline; wind 0.03 N</summary>

![comparison_rolling_payload0.0_wind0.03.png](analysis/comparisons/comparison_rolling_payload0.0_wind0.03.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.065 m, LQR = 0.187 m, Koopman-MPC = 0.552 m.

</details>

<details>
<summary><strong>comparison_rolling_payload0.2_wind0.0.png</strong> — +20% payload mass; wind 0.0 N</summary>

![comparison_rolling_payload0.2_wind0.0.png](analysis/comparisons/comparison_rolling_payload0.2_wind0.0.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.068 m, LQR = 0.166 m, Koopman-MPC = 0.213 m.

</details>

<details>
<summary><strong>comparison_rolling_payload0.2_wind0.015.png</strong> — +20% payload mass; wind 0.015 N</summary>

![comparison_rolling_payload0.2_wind0.015.png](analysis/comparisons/comparison_rolling_payload0.2_wind0.015.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.075 m, LQR = 0.192 m, Koopman-MPC = 0.330 m.

</details>

<details>
<summary><strong>comparison_rolling_payload0.2_wind0.03.png</strong> — +20% payload mass; wind 0.03 N</summary>

![comparison_rolling_payload0.2_wind0.03.png](analysis/comparisons/comparison_rolling_payload0.2_wind0.03.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.088 m, LQR = 0.237 m, Koopman-MPC = 0.490 m.

</details>

<details>
<summary><strong>comparison_rolling_payload0.4_wind0.0.png</strong> — +40% payload mass; wind 0.0 N</summary>

![comparison_rolling_payload0.4_wind0.0.png](analysis/comparisons/comparison_rolling_payload0.4_wind0.0.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.121 m, LQR = 0.291 m, Koopman-MPC = 0.254 m.

</details>

<details>
<summary><strong>comparison_rolling_payload0.4_wind0.015.png</strong> — +40% payload mass; wind 0.015 N</summary>

![comparison_rolling_payload0.4_wind0.015.png](analysis/comparisons/comparison_rolling_payload0.4_wind0.015.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.126 m, LQR = 0.309 m, Koopman-MPC = 0.341 m.

</details>

<details>
<summary><strong>comparison_rolling_payload0.4_wind0.03.png</strong> — +40% payload mass; wind 0.03 N</summary>

![comparison_rolling_payload0.4_wind0.03.png](analysis/comparisons/comparison_rolling_payload0.4_wind0.03.png)

**Result from the saved figure title (3-D position RMSE):** PID = 0.135 m, LQR = 0.341 m, Koopman-MPC = 0.465 m.

</details>

## Generating the 36 figures again

From the `mujoco/` directory:

```bash
python analysis/run_batch_comparisons.py --terrains rolling flat mountain canyon --payloads 0.0 0.2 0.4 --winds 0.0 0.015 0.03
```

This launches the batch comparison runner across every parameter combination and writes the resulting figures to `analysis/comparisons/`.

## References

1. Koopman, B. O. (1931). “Hamiltonian Systems and Transformations in Hilbert Space.” *Proceedings of the National Academy of Sciences*, 17(5), 315–318. https://doi.org/10.1073/pnas.17.5.315
2. Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015). “A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition.” *Journal of Nonlinear Science*, 25, 1307–1346. https://doi.org/10.1007/s00332-015-9258-5
3. Korda, M., & Mezić, I. (2018). “Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control.” *Automatica*, 93, 149–160. https://doi.org/10.1016/j.automatica.2018.03.046
4. Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2022). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing. https://sites.engineering.ucsb.edu/~jbraw/mpc/
5. Todorov, E., Erez, T., & Tassa, Y. (2012). “MuJoCo: A physics engine for model-based control.” *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 5026–5033. https://doi.org/10.1109/IROS.2012.6386109
6. Google DeepMind. *MuJoCo Documentation*. https://mujoco.readthedocs.io/
7. CVXPY Developers. *CVXPY Documentation*. https://www.cvxpy.org/
8. Clarabel Contributors. *Clarabel Conic Interior Point Solver Documentation*. https://clarabel.org/
