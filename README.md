![Amrita Vishwa Vidyapeetham](assets/amrita_logo.jpeg)

# Koopman-Operator-Based Model Predictive Control for Payload-Carrying and Wind-Disturbed Quadrotors

### A Two-Phase Comparative Study Against Classical PID and LQR Control

**Group: CD7 | Course: Drones, Semester 5**

---

## Team Members

| Name             | Roll Number      | Email                                     |
| ---------------- | ---------------- | ------------------------------------------ |
| Rupanshi Sangwan | CB.SC.U4AIE24262 | <cb.sc.u4aie24262@cb.amrita.students.edu> |
| Devana Madhavan  | CB.SC.U4AIE24213 | <cb.sc.u4aie24213@cb.amrita.students.edu> |
| Indraneel R      | CB.SC.U4AIE24323 | <cb.sc.u4aie24323@cb.amrita.students.edu> |
| Adithya U K      | CB.SC.U4AIE24302 | <cb.sc.u4aie24302@cb.amrita.students.edu> |

---

## Repository Contents

| Folder / File               | Description                                                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `data/`                      | Phase 1 logged flight data (`nominal`, `windy`, `payload`) and trained Koopman models      |
| `src/koopman/`                | Phase 1 data collection and EDMD Koopman model training                                   |
| `src/mpc/`                    | Phase 1 MPC controller, LQR controller, and scenario runners                              |
| `src/env_selector/`           | Phase 1 environment (nominal/windy/payload) classifier                                    |
| `src/analysis/`               | Phase 1 metrics computation and validation scripts                                        |
| `results/`                    | Phase 1 `.npz` flight logs, tracking plots, and metrics CSVs for MPC, PID, and LQR         |
| `results/analysis/`           | Phase 1 final metrics tables and analysis notes                                           |
| `results/plots_updated/`      | Phase 1 corrected Z-tracking comparison plots (all 3 controllers, all 6 scenarios)         |
| `v2_stable_koopman_mpc/`      | **Phase 2**: from-scratch reimplementation using rotation-matrix lifting                   |
| `v2_stable_koopman_mpc/data_collection/` | Phase 2 flight data collection with independent excitation and disturbance injection |
| `v2_stable_koopman_mpc/koopman_model/`   | Phase 2 EDMD dataset construction and Koopman model fitting                          |
| `v2_stable_koopman_mpc/mpc/`             | Phase 2 Koopman-MPC controller (QP formulation) and closed-loop test runner          |
| `v2_stable_koopman_mpc/baselines/`       | Phase 2 cascaded LQR baseline                                                        |
| `v2_stable_koopman_mpc/comparison/`      | Phase 2 PID vs. Koopman-MPC disturbance comparison harness                           |
| `v2_stable_koopman_mpc/diagnostics/`     | Phase 2 sign-consistency checks, horizon validation, crashed-episode filtering       |
| `Matlab.mlx`                  | Supplementary Phase 1 MATLAB notebook                                                      |
| `README.md`                   | This file — full two-phase project report                                                 |

---

## Abstract

Model Predictive Control (MPC) built on a Koopman-operator linearization of nonlinear
quadrotor dynamics has been proposed in recent literature as a way to obtain fast,
real-time-capable control without deriving an exact first-principles model of the
system. **Phase 1** of this project replicates the Koopman-based control approach of
Oh et al. (IEEE Access, 2024), including their Extended Dynamic Mode Decomposition
(EDMD) system identification and environment-selector architecture, in the
`gym-pybullet-drones` simulation environment, and extends it with systematic payload
and wind disturbance testing plus a genuine comparison against classical PID and LQR
baselines. Phase 1 identifies and diagnoses a persistent orientation instability in
the Koopman-MPC controller: the drone tips and loses attitude control within
approximately half a second of flight, independent of disturbance type, isolated to
the Euler-angle/quaternion-based Koopman lifting function used, consistent with known
singularity issues described in SE(3)-structured Koopman-MPC literature.

**Phase 2** independently re-implements the Koopman-MPC pipeline from scratch using a
**rotation-matrix-based** lifting function (avoiding Euler angles and quaternions
entirely), following exactly the fix Phase 1's own literature review predicted would
be needed. This resolves the specific failure Phase 1 diagnosed: the Phase 2 controller no
longer loses attitude control, holding roll and pitch within roughly a degree for
the full flight in every condition tested, where Phase 1's controller tipped past
90° within half a second. Under undisturbed and wind-only conditions it tracks for
the full flight and stays at altitude. **It does not, however, survive every
condition:** under a mid-flight payload increase it holds attitude but loses
altitude and settles to the ground (min. altitude ≈0.01 m, below the 0.05 m
threshold this project's own episode-screening uses to classify a run as failed).
The attitude-loss failure mode is eliminated; an altitude-authority failure mode
under payload remains. In the course of this work,
two further, previously undiagnosed issues were found and fixed: a **closed-loop
system identification bias** (the training data's control input was dominated by a
feedback controller's corrective response rather than independent excitation,
producing a Koopman model with the wrong sign on the thrust-differential-to-motion
coupling), and a **cross-step consistency gap** in the MPC formulation itself (no
anchoring between consecutive real control steps, allowing the controller's actual
commanded input to jump unpredictably even when each individual solve looked smooth
in isolation). With both fixed and validated, Phase 2's Koopman-MPC is stable but
still does not outperform PID on tracking accuracy or disturbance degradation
(RMSE 0.34 m vs. PID's 0.04 m; 135.8% vs. 37.8% relative degradation under combined
payload and wind). This is reported honestly, with control-effort evidence showing
the controller is not failing to detect disturbance but failing to react to it with
sufficient authority — a conservatism required for stability given the model's
validity region, not a bug. Concrete next steps to close this remaining gap are
identified.

---

## 1. Introduction

Multirotor drones are increasingly deployed for applications — package delivery,
inspection, agriculture — where payload mass and wind conditions vary unpredictably
during flight. Classical model-based controllers (PID, LQR) are simple and robust but
are typically tuned around a fixed nominal model of the vehicle; when the true
dynamics deviate substantially (e.g., due to a sudden mass change), performance can
degrade. Data-driven control offers an alternative: rather than deriving the
dynamics analytically, a model is *learned* directly from input/output flight data.

The Koopman operator provides a principled way to do this: it represents a nonlinear
dynamical system as a (possibly infinite-dimensional) *linear* operator acting on a
space of observable functions of the state. In practice, a finite-dimensional
approximation is obtained via Extended Dynamic Mode Decomposition (EDMD), and the
resulting linear model is used inside a standard, convex Model Predictive Control
(MPC) formulation, avoiding the nonlinear optimization that a first-principles model
would otherwise require.

**Base paper.** This project replicates and extends Oh, Lee, and Moon (2024),
*"Koopman-Based Control System for Quadrotors in Noisy Environments"* (IEEE Access),
which trains separate Koopman models for nominal and windy flight conditions, uses a
nearest-mean *environment selector* to pick the active model in real time, and drives
an MPC controller with the selected model. The base paper validates its approach
under hover-level attitude tracking and windy conditions only; it does not test
payload disturbances, and it does not compare against a classical control baseline.

**This project's contributions, across both phases:**

1. **Payload disturbance testing** (Phase 1) — a mid-flight mass addition not present
   in the base paper, tested independently and in combination with wind.
2. **A genuine classical-control baseline** (Phase 1) — PID (via `gym-pybullet-drones`'
   built-in `DSLPIDControl`) and a discrete-time LQR controller, evaluated on the same
   disturbance scenarios as Koopman-MPC, with matched targets and a crash-aware
   metrics pipeline.
3. **Root-cause diagnosis of a Koopman-MPC instability** (Phase 1) — systematic
   isolation testing showing the failure is independent of disturbance type or target
   height, connecting it to a known representational issue in the SE(3) Koopman-MPC
   literature.
4. **Independent confirmation and partial resolution of that diagnosis** (Phase 2) —
   a from-scratch rotation-matrix-lifted Koopman-MPC pipeline that eliminates the
   attitude-loss failure entirely (roll/pitch within ≈1° for full-length flights,
   no crash-truncated runs), while surfacing a separate, milder altitude-authority
   limitation under payload that is characterized rather than hidden.
5. **Discovery and correction of two further issues** (Phase 2) — a closed-loop
   system identification bias affecting the sign-correctness of the learned dynamics,
   and a cross-step consistency gap in the MPC formulation — both are general
   findings applicable beyond this specific project.
6. **An honest final comparison, with mechanism, not just numbers** (Phase 2) —
   Koopman-MPC remains behind PID even once stabilized, and control-effort data is
   used to explain *why*, rather than treating the result as a simple pass/fail.

---

# Phase 1: Initial Replication (Euler-Angle / Quaternion Lifting)

## 2. Methodology

The Phase 1 pipeline consists of six stages.

### 2.1 Flight Data Collection

Flight data is collected in `gym-pybullet-drones` (CF2X quadrotor model, PyBullet
physics) under three conditions:

- **Nominal** — no external disturbance.
- **Windy** — a constant horizontal force disturbance.
- **Payload** — additional mass introduced mid-flight.

A PID-stabilized controller (`DSLPIDControl`) flies to randomized position/height
setpoints during data collection. This was found necessary empirically: driving the
four motors with independent random RPM noise (a natural first choice for exciting
system dynamics) caused the drone to tumble (roll standard deviation
σ_φ ≈ 0.73 rad, full ±π excursions) rather than fly — data collected this way is
unusable for identifying *flight* dynamics. Flying to randomized setpoints under PID
stabilization keeps the drone upright (σ_φ ≈ 0.08 rad) while still covering a wide
height range for the Koopman model to learn from.

At each timestep the logged state vector is

$$
\mathbf{x} = \begin{bmatrix} x & y & z & q_1 & q_2 & q_3 & q_4 & \phi & \theta & \psi & v_x & v_y & v_z & \omega_x & \omega_y & \omega_z \end{bmatrix}^\top \in \mathbb{R}^{16}
$$

(position, quaternion, roll/pitch/yaw, linear velocity, angular velocity), alongside
the 4-dimensional motor RPM command $\mathbf{u} \in \mathbb{R}^4$.

> **Note (added after Phase 2):** this state vector carries *both* a quaternion and
> the corresponding Euler angles for the same orientation — a redundant, and as
> Phase 1's own findings show, singularity-prone representation. Phase 2 (§6) drops
> both entirely in favor of the rotation matrix.

### 2.2 Koopman Model Training (EDMD)

Let $\mathbf{x}[k]$ be the state at discrete time step $k$. The Koopman operator
$\mathcal{K}$ acts on observable functions $\psi$ of the state such that

$$
\mathcal{K}\,\psi(\mathbf{x}[k]) = \psi(\mathbf{x}[k+1])
$$

In its finite-dimensional, control-affine EDMD approximation, a *lifting function*
$\psi(\mathbf{x})$ maps the raw state into a higher-dimensional space in which the
dynamics are approximately linear:

$$
\psi(\mathbf{x}[k+1]) \approx A\,\psi(\mathbf{x}[k]) + B\,\mathbf{u}[k]
$$

The lifting function used in Phase 1 is

$$
\psi(\mathbf{x}) = \Big[\ \mathbf{x},\ \ \mathbf{x}^{\circ 2},\ \ \{x_i x_j\}_{(i,j)\in P}\ \Big]
$$

where $\mathbf{x}^{\circ 2}$ denotes the elementwise square and $P$ is the set of
pairwise cross-terms among the orientation and angular-velocity dimensions (indices
3–15). Cross-terms were found necessary empirically: elementwise squares alone fit
translational states well but fit rotational states poorly, consistent with the fact
that rigid-body rotational dynamics (Euler's equation, below) are inherently coupled
across axes.

$(A, B)$ are obtained via regularized least squares on normalized data:

$$
(A, B) = \arg\min_{A,B} \sum_{k} \left\lVert \psi(\mathbf{x}[k+1]) - A\,\psi(\mathbf{x}[k]) - B\,\mathbf{u}[k] \right\rVert_2^2 + \lambda \lVert [A\ B] \rVert_F^2
$$

with $\lambda = 10^{-4}$. Separate $(A,B)$ pairs are trained independently for the
nominal, windy, and payload conditions, matching the base paper's approach of one
Koopman model per environment.

**Underlying physics (for reference).** Discretely, the translational dynamics
follow Newton's second law and the rotational dynamics follow Euler's rotation
equation:

$$
m\,\dot{\mathbf{v}} = R(\mathbf{q})\,[0,0,T]^\top - m g\,\hat{\mathbf{z}} + \mathbf{F}_{\text{dist}}
$$

$$
I\,\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times (I\,\boldsymbol{\omega})
$$

where $T$ is total thrust, $R(\mathbf{q})$ is the rotation matrix corresponding to
quaternion $\mathbf{q}$, $I$ is the inertia tensor, and $\mathbf{F}_{\text{dist}}$
represents an external disturbance force (e.g. wind). The Koopman/EDMD model does not
use this equation directly — it is learned from data — but it explains why
orientation/angular-velocity cross-terms are needed in the lifting function: the
$\boldsymbol{\omega} \times (I\boldsymbol{\omega})$ term is itself a bilinear coupling
between angular-velocity components.

### 2.3 MPC Controller

At each control step, given the current state $\mathbf{x}_0$ and a target state
$\mathbf{x}_{\text{ref}}$, the controller solves a receding-horizon quadratic program
over the lifted dynamics. The lifted trajectory is *substituted directly* into the
cost (rather than carried as a free optimization variable subject to equality
constraints) — an early formulation using free lifted-state variables was found to be
numerically fragile, causing the QP solver to fail on nearly every call in practice:

$$
\min_{\mathbf{u}_{0:H-1}} \ \sum_{t=0}^{H-1} \Big[\, Q \left\lVert W \odot \big(\psi_{t+1} - \psi_{\text{ref}}\big) \right\rVert_2^2 + R \lVert \mathbf{u}_t \rVert_2^2 + R_\Delta \lVert \mathbf{u}_t - \mathbf{u}_{t-1} \rVert_2^2 \Big]
$$

$$
\text{s.t.} \quad \psi_{t+1} = A\,\psi_t + B\,\mathbf{u}_t, \qquad |\mathbf{u}_t| \le U_{\max}, \qquad |\mathbf{u}_t - \mathbf{u}_{t-1}| \le \Delta U_{\max}
$$

where $W$ is a per-dimension weight vector that penalizes orientation error
(quaternion, roll, pitch) more heavily than other states — added after empirically
observing that a uniform weight allowed the controller to sacrifice orientation
stability for marginal height-tracking gains, leading to the drone tipping over.
$H = 6$ is the prediction horizon; the QP is solved with `cvxpy`/OSQP at each control
step and only the first control action $\mathbf{u}_0$ is applied (receding-horizon
control), matching the base paper's MPC formulation in structure.

### 2.4 Environment Selector

Following the base paper's Algorithm 1, a nearest-mean classifier identifies the
active environment. A precision vector $\boldsymbol{\lambda}_c$ is computed per
condition $c \in \{\text{nominal}, \text{windy}, \text{payload}\}$ as the mean of the
roll/pitch/yaw and angular-velocity dimensions over that condition's training data.
At runtime, over a rolling 5-second window (240 steps at 48 Hz, matching the base
paper), the selector computes

$$
\hat{c} = \arg\min_{c} \left\lVert \bar{\mathbf{y}}_{\text{window}} - \boldsymbol{\lambda}_c \right\rVert_2
$$

and switches the active Koopman model $(A_{\hat c}, B_{\hat c})$ accordingly.

### 2.5 Disturbance Scenario Testing

Six scenarios are tested, extending the base paper's four:

| # | Scenario                                | Selector           |
| - | ---------------------------------------- | ------------------- |
| 1 | Nominal environment                     | On                  |
| 2 | Windy environment                       | **Off** (ablation)  |
| 3 | Windy environment                       | On                  |
| 4 | Nominal → windy transition (mid-flight) | On                  |
| 5 | Payload added mid-flight                | On                  |
| 6 | Payload + wind combined                 | On                  |

Scenarios 5 and 6 (payload) are not present in the base paper.

### 2.6 PID / LQR Baseline

A PID baseline uses `gym-pybullet-drones`' built-in `DSLPIDControl`. An LQR baseline
linearizes the height-channel dynamics about hover,

$$
\begin{bmatrix} z \\ v_z \end{bmatrix}_{k+1} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} z \\ v_z \end{bmatrix}_k + \begin{bmatrix} 0 \\ \Delta t / m \end{bmatrix} \delta T_k
$$

and solves the discrete algebraic Riccati equation for the optimal gain $K$
minimizing $\sum_k \mathbf{x}_k^\top Q \mathbf{x}_k + R\,\delta T_k^2$. Both baselines
are run through the identical six scenarios, targeting the identical height setpoint
as the Koopman-MPC controller.

## 3. Phase 1 Results

### 3.1 Final Comparison Metrics

Computed with crash-aware truncation (metrics are computed only up to the first
failure event — ground contact, a roll/pitch flip beyond 90°, or a simulator reset —
so that post-crash artifacts do not silently corrupt the reported numbers):

| Scenario | Controller | RMSE (m)  | Settling Time (s) | Control Effort (RMS RPM) | Max Transient Dev. (m) | Crashed | Failure Time (s) |
| -------- | ---------- | --------- | ------------------ | ------------------------- | ------------------------ | ------- | ------------------ |
| 1        | MPC        | 0.106     | –                   | 3199.6                    | 0.187                    | **Yes** | 0.50                |
| 1        | PID        | 0.062     | 4.04                | 14.8                       | 0.187                    | No      | –                   |
| 1        | LQR        | **0.060** | 3.92                | 18.5                       | 0.187                    | No      | –                   |
| 2        | MPC        | 0.118     | –                   | 3321.8                    | 0.187                    | **Yes** | 0.42                |
| 2        | PID        | 0.062     | 4.00                | 951.9                      | 0.187                    | No      | –                   |
| 2        | LQR        | **0.060** | 3.92                | 19.5                       | 0.187                    | No      | –                   |
| 3        | MPC        | 0.118     | –                   | 3321.8                    | 0.187                    | **Yes** | 0.42                |
| 3        | PID        | 0.062     | 4.00                | 951.9                      | 0.187                    | No      | –                   |
| 3        | LQR        | **0.060** | 3.92                | 19.5                       | 0.187                    | No      | –                   |
| 4        | MPC        | 0.106     | –                   | 3199.6                    | –                         | **Yes** | 0.50                |
| 4        | PID        | 0.062     | 0.00                | 608.0                      | 0.003                    | No      | –                   |
| 4        | LQR        | **0.060** | 0.00                | 18.9                       | 0.0001                   | No      | –                   |
| 5        | MPC        | 0.106     | –                   | 3199.6                    | –                         | **Yes** | 0.50                |
| 5        | PID        | 0.079     | –                   | 1749.6                    | 0.073                    | No      | –                   |
| 5        | LQR        | **0.062** | –                   | 1748.4                    | 0.027                    | No      | –                   |
| 6        | MPC        | 0.118     | –                   | 3321.8                    | 0.187                    | **Yes** | 0.42                |
| 6        | PID        | 0.079     | –                   | 1853.0                    | 0.187                    | No      | –                   |
| 6        | LQR        | **0.062** | –                   | 1746.1                    | 0.187                    | No      | –                   |

**Bold** marks the best (lowest) value per scenario per metric. Across every metric,
in every scenario, PID or LQR wins; Koopman-MPC does not win a single metric in any
scenario, and does not complete any scenario (see §3.2).

### 3.2 Completion Rate

Koopman-MPC completes only 5–7% of every 15-second run before failing; PID and LQR
complete 100% of every scenario.

*(Figure: `results/analysis/completion_rate_comparison.png`)*

### 3.3 Z-Tracking Comparison (All Controllers, All Scenarios)

*(Figures: `results/plots_updated/scenario_1_z_tracking.png` through `scenario_6_z_tracking.png`)*

PID and LQR both climb smoothly to the 0.3 m target and hold steady, including
sensible recovery after mid-flight payload addition (scenario 5/6: PID settles ≈0.23
m, LQR ≈0.27 m post-payload, both closer to target than a naively re-tuned
controller would achieve without adaptation). Koopman-MPC climbs briefly, then tips
past 90° roll and lands, in every scenario.

### 3.4 Isolated Hover Instability (Root-Cause Diagnostic)

To determine whether the Koopman-MPC failure was specific to disturbance conditions
or aggressive height targets, an isolated test was run: the controller was asked to
hold its own *starting* height (zero commanded climb) with *zero* disturbance
applied. The instability persisted — roll grew unbounded even under this
minimal-demand condition, ruling out disturbance response or target aggressiveness as
the cause.

*(Figure: `results/plots/notebook_hover_instability.png`)*

This isolates the failure to the controller/model itself — specifically, to the
Euler-angle/quaternion-based orientation representation used in the Koopman lifting
function — rather than to any disturbance-testing methodology in this project.

## 4. Phase 1 Discussion

The base paper's Koopman-MPC approach, faithfully replicated here (same state
representation, same EDMD lifting structure, same environment-selector logic), is
reproducible, and its data-driven modeling pipeline (§2.1–§2.2) achieves low one-step
prediction error on position, velocity, and orientation states (normalized RMSE
≈ 0.19–0.27 across conditions). The failure occurs specifically at the *control*
stage: the linear MPC controller, operating on this learned model, cannot maintain
attitude stability over a sustained flight, independent of disturbance type.

This is consistent with a known, published limitation of Euler-angle/quaternion-based
Koopman lifting functions: Narayanan et al. (SE(3) Koopman-MPC, IFAC-PapersOnLine,
2023) specifically avoid this representation, instead lifting with observables built
from the rotation matrix $R \in SO(3)$ directly (e.g. $R\boldsymbol{\omega}$,
$R\boldsymbol{\omega}^{\circ 2}$), citing exactly the kind of orientation singularity
this project's diagnostics independently reproduce. **Phase 2 (below) carries out
that migration and reports the result.**

## 5. Verification and Reproducibility Notes (Phase 1)

In the interest of transparency, two significant bugs were found and corrected during
Phase 1's own internal verification process, prior to reporting final results:

1. **Target-height mismatch.** `mpc_controller.py`'s target was found to be
   mismatched (0.1125 m, the drone's own starting height) against the 0.3 m target
   used by the PID/LQR baselines and the metrics pipeline, invalidating any
   RMSE/tracking comparison computed before the fix. Corrected by unifying the
   target to 0.3 m across all three controllers and the metrics script, and
   regenerating all 18 scenario result files.
2. **Payload data corruption.** Payload-condition training data was found to have
   roll instability (σ_φ ≈ 1.5–2.1 rad) traced to repeated calls to PyBullet's
   `changeDynamics()` destabilizing the simulator, and separately to the randomized
   height range (0.1–1.2 m) being unachievable for a 37%-overloaded drone. Fixed by
   calling `changeDynamics()` exactly once per mass change and using a
   condition-specific, achievable height range (0.1–0.6 m) for payload data
   collection; payload data, Koopman model, and environment-selector precision
   values were all regenerated.
3. **Environment selector unreliable on payload.** The environment selector's
   RPY/angular-velocity-only signature was found, through wider empirical testing
   across 12 non-overlapping windows per condition, to be statistically unreliable
   for distinguishing the payload condition from nominal (payload's distance to its
   own precision mean: 0.0216; distance to nominal's precision mean: 0.0223 —
   effectively indistinguishable). Corrected by adding a normalized thrust feature
   (mean commanded RPM as a fractional deviation from hover) to the signature.
   Post-fix, payload's distance to its own mean (0.092) is clearly smaller than its
   distance to nominal (0.098) or windy (0.114) across all 12 test windows. This fix
   does not alter the headline Phase 1 results, since Koopman-MPC crashes at
   t≈0.42–0.50s in every scenario, before the selector's first 5-second
   classification window ever executes.

All reported Phase 1 results reflect the corrected pipeline. Git tags
`before-target-and-payload-fix` and `corrected-final-results` mark the exact commits
before and after these corrections for full reproducibility.

---

# Phase 2: Rotation-Matrix Lifting — Resolution and Extended Investigation

Phase 1's §4 identified rotation-matrix-based lifting as the literature-predicted
fix for the orientation instability, and §5.1 began a preliminary investigation
using the *existing* (calm, PID-stabilized) Phase 1 flight data, finding the test
inconclusive because that data contained almost no extreme-tilt examples.

Phase 2 completes this investigation as a genuinely independent, from-scratch
reimplementation: new flight data, a new lifting function, a new MPC formulation,
and a new disturbance-comparison harness. Phase 2 is built against the `v1.0.0`
tag of `gym-pybullet-drones` in a single-drone configuration; the two phases'
pipelines are not drop-in compatible and should be run in separate environments.
All Phase 2 code is in `v2_stable_koopman_mpc/`.

## 6. Methodology (Phase 2)

### 6.1 State representation: rotation matrix, not quaternion or Euler angle

The raw state logged and used for Koopman fitting is

$$
\mathbf{x} = \big[\ \mathbf{p},\ \ \mathbf{v},\ \ \text{vec}(R),\ \ \boldsymbol{\omega}\ \big] \in \mathbb{R}^{18}
$$

where $\mathbf{p}, \mathbf{v} \in \mathbb{R}^3$ are position and velocity,
$\text{vec}(R) \in \mathbb{R}^9$ is the flattened $3\times3$ rotation matrix (read
directly from the simulator's quaternion via `pybullet.getMatrixFromQuaternion`, but
never itself stored or fit as a quaternion or Euler angle), and
$\boldsymbol{\omega} \in \mathbb{R}^3$ is angular velocity. **No Euler angle and no
quaternion component appears anywhere in this state vector** — this is the
representational change Phase 1 §4/§5.1/§5.2 identified as necessary.

The lifting function extends this with a modest set of physically-motivated
cross-terms:

$$
\psi(\mathbf{x}) = \big[\ \mathbf{x},\ \ \mathbf{v} \odot R_{zz},\ \ v_z R_{xz},\ \ v_z R_{yz},\ \ \boldsymbol{\omega} \odot R_{zz},\ \ 1\ \big] \in \mathbb{R}^{27}
$$

where $R_{zz}, R_{xz}, R_{yz}$ are individual entries of the rotation matrix. This
lifting was deliberately kept small (27 dimensions, vs. Phase 1's much larger
polynomial expansion) so that a numerical bug in the EDMD fit or the MPC formulation
would be easy to isolate and debug — a design choice that proved necessary, given
the two additional issues found and fixed below (§6.3, §6.4).

### 6.2 Flight data collection and disturbance injection

Training flight data (200 episodes) is generated in `gym-pybullet-drones` (CF2X
model) using the built-in `DSLPIDControl` PID controller tracking a per-episode
randomized sinusoidal reference trajectory:

$$
\mathbf{r}(t) = \big[\ a_x \sin(\omega_x t + \phi_x),\ \ a_y \sin(\omega_y t + \phi_y),\ \ z_0 + 0.15\sin(\omega_z t + \phi_z)\ \big]
$$

$$
a_x, a_y \sim U(0.3,\ 1.0)\ \text{m}, \qquad \omega_x, \omega_y, \omega_z \sim U(0.05,\ 0.4)\ \text{rad/s}, \qquad \phi_\bullet \sim U(0,\ 2\pi), \qquad z_0 = 1.0\ \text{m}
$$

Note that the *training* reference oscillates in altitude as well as in the
horizontal plane, whereas the *evaluation* reference used in §7
(`run_mpc_closed_loop.py`, `run_comparison.py`) holds altitude constant at
$z_0$ — the training set deliberately covers a wider range of motion than the
evaluation trajectory exercises.

The drone is spawned exactly at $\mathbf{r}(0)$ each episode (rather than a fixed
point), which was found necessary to avoid a large, discontinuous initial position
error that otherwise destabilized flight before the reference trajectory itself even
began (see §6.5 for further discussion of destabilizing factors found during this
phase).

Two disturbances are injected during training data collection, at randomized
severity and timing per episode, so the resulting Koopman model generalizes across
disturbed and undisturbed flight rather than requiring a separate model per
condition (unlike Phase 1's per-environment approach, §2.2): a mid-flight payload
mass increase (0–40% of nominal mass, applied via `pybullet.changeDynamics`), and a
horizontal wind force (0–0.03 N, applied via `pybullet.applyExternalForce` for a
1–3 second window). Any episode exhibiting excessive roll/pitch, altitude collapse,
or excessive drift is automatically excluded from the training set (final exclusion
rate: 1.0–1.5%, `diagnostics/scan_bad_episodes.py`).

### 6.3 A previously undiagnosed issue: closed-loop system identification bias

An early version of the Phase 2 model, fit on this data with only a very small
independent excitation signal superimposed on the PID's own commands, was found —
via a direct controlled test (`diagnostics/diagnose_sign.py`: apply a known motor
differential, compare the model's predicted direction of motion against the real
simulator's actual direction) — to have **the wrong sign** on the coupling between
differential motor thrust and horizontal acceleration.

The cause is a classic closed-loop identification bias: since the training data's
control input was overwhelmingly a *feedback response to error* rather than an
independent probe of the dynamics, the EDMD regression partially learned the
*controller's* corrective behavior rather than the *plant's* causal dynamics. The fix
was to substantially strengthen the independent excitation signal — an
Ornstein–Uhlenbeck process superimposed on the commanded RPM — to a level
($\sigma \approx 0.03$, tuned empirically) large enough to dominate the PID's own
correlated corrections without itself destabilizing flight (a stronger setting,
$\sigma \approx 0.05$, was found to cause **55% of training episodes to crash**,
which would have silently corrupted the training set had it not been checked
directly). After this fix, `diagnose_sign.py` confirms correct-sign predictions on
all three translational axes.

This finding generalizes beyond this project: **any Koopman/EDMD model trained on
data generated primarily by a feedback controller carries this same risk**, and
should be checked with a direct sign/direction test before being trusted in a
model-based controller, rather than assumed correct from prediction-error metrics
alone (a model with the wrong sign can still show low one-step prediction error near
equilibrium, where the required control differential is small).

### 6.4 A previously undiagnosed issue: MPC cross-step consistency

The QP's control-rate penalty,

$$
\left\lVert \mathbf{u}_t - \mathbf{u}_{t-1} \right\rVert_2^2, \qquad t = 1, \dots, H_p - 1
$$

penalizes rate-of-change *within* a single receding-horizon solve, but has no
reference to what was *actually applied* on the previous real control step — each
fresh solve starts with no memory of it. This means the real applied command,
$\mathbf{u}_0$, could in principle jump between two consecutive real-time control
steps even though each individual solve's internal plan looks smooth in isolation.

The fix anchors the very first predicted action to the last actually-applied
command, $\mathbf{u}_{-1}^{\text{real}}$, carried across solves (not just within one):

$$
\text{cost} \mathrel{+}= \left\lVert \mathbf{u}_0 - \mathbf{u}_{-1}^{\text{real}} \right\rVert_2^2 \cdot R_\Delta
$$

This measurably improved closed-loop stability margin: a weight configuration that
previously caused an outright crash within ~1 second was, with this fix in place,
degraded only to a bounded (non-crashing) tracking error over the full flight — a
genuine, if partial, improvement (see §6.6).

### 6.5 QP formulation and numerical conditioning

The QP formulation follows the same receding-horizon structure as Phase 1 (§2.3),
adapted to this phase's lifted state and reformulated around the *deviation* from
hover RPM, $\delta\mathbf{u} = \mathbf{u} - \mathbf{u}_{\text{hover}}$, rather than
absolute RPM:

$$
\min_{\delta\mathbf{u}_{0:H_p-1}} \ \sum_{t=0}^{H_p-1} \Big[\, \left\lVert \psi_{t+1} - \psi_{\text{ref},t} \right\rVert^2_{Q_m} + \left\lVert \delta\mathbf{u}_t \right\rVert^2_{R_m} + \left\lVert \delta\mathbf{u}_t - \delta\mathbf{u}_{t-1} \right\rVert^2_{E_m} \Big]
$$

$$
\text{s.t.} \quad \psi_{t+1} = A\,\psi_t + B\,(\mathbf{u}_{\text{hover}} + \delta\mathbf{u}_t), \qquad \delta\mathbf{u}_t \in [-\mathbf{u}_{\text{hover}},\ \text{RPM}_{\max} - \mathbf{u}_{\text{hover}}]
$$

with control horizon $H_c = 5 < H_p = 10$ (control frozen beyond $H_c$). This
reformulation was necessary for numerical reasons: absolute RPM values are $O(10^4)$
while lifted-state entries are $O(1)$, and mixing these magnitudes directly in one
QP caused the OSQP solver to fail to converge; the deviation reformulation, together
with switching to the CLARABEL solver, resolved this. Final validated weights:
$Q_m$ position weight 120, $E_m$ rate weight 1.0, $R_m$ effort weight $10^{-3}$.

### 6.6 Baseline: cascaded LQR

**This baseline is incomplete and is reported as a negative result.** An LQR
baseline independent of Phase 1's (§2.6) was attempted for Phase 2 and never
reached a stable closed loop; the Phase 2 comparison in §7 is therefore
**PID vs. Koopman-MPC only**, with no Phase 2 LQR column.

Six substantively different configurations were tried, all of which lost altitude
and reached the ground within roughly 0.5–2 seconds:

1. Flat 12-state LQR (position + attitude in a single gain matrix), at three
   different $Q$/$R$ weightings spanning four orders of magnitude in control cost.
2. A **cascaded** redesign — outer position-tracking LQR producing a desired tilt
   angle, **explicitly clamped to ±20°**, tracked by a separate faster inner
   attitude LQR — mirroring the two-timescale structure that makes PID work on this
   platform.
3. The cascaded design with a substantially softened inner loop, to reduce
   overshoot past the clamped tilt target.

The cascade and the tilt clamp did *not* rescue it, so the failure is not simply
"missing a cascade". Two things were verified and are not the cause: the
virtual-command-to-RPM mixer is directionally correct (`diagnose_lqr_sign.py`
confirms thrust-up climbs with zero roll/pitch coupling, $+\tau_x$ produces pure
roll, $+\tau_y$ produces pure pitch), and the gain magnitudes were brought into a
sane range (`diagnose_lqr_gain.py`). The most likely remaining cause, not
investigated further, is that the inner loop overshoots the clamped tilt reference
during the fast initial transient — a rate-limited (ramped) attitude reference and
anti-windup would be the next things to try.

One finding does survive from this attempt, and is worth carrying back to Phase 1's
future-work discussion: **direct full-state LQR is not a drop-in baseline for a
low-inertia quadrotor**. Phase 1's LQR (§2.6) succeeded because it controls only
the decoupled *height* channel about hover, where the linearization is valid;
extending LQR to full position-and-attitude control on this platform proved
substantially harder than that result suggests.

## 7. Phase 2 Results

### 7.1 Final comparison metrics

Same reference trajectory and disturbance schedule (payload at t=3s, wind gust
t=3–5s) applied identically to both controllers:

| Controller  | Condition       | RMSE (m) | Max err (m) | Control effort |
| ----------- | --------------- | -------- | ----------- | --------------- |
| PID         | none            | 0.0381   | 0.0583      | 58.7            |
| PID         | payload         | 0.0529   | 0.0680      | 1,836,034       |
| PID         | wind            | 0.0375   | 0.0583      | 96.7            |
| PID         | payload+wind    | 0.0525   | 0.0680      | 1,835,657       |
| Koopman-MPC | none            | 0.3394   | 0.4560      | 0.0006          |
| Koopman-MPC | payload         | 0.8117   | 1.1027      | 0.0009          |
| Koopman-MPC | wind            | 0.4071   | 0.8211      | 0.0006          |
| Koopman-MPC | payload+wind    | 0.8003   | 1.0865      | 0.0008          |

**Relative degradation** (clean → payload+wind): PID **37.8%**, Koopman-MPC
**135.8%**.

**Attitude stability — the Phase 1 failure mode — is fully resolved.** Across every
condition above, Koopman-MPC holds roll and pitch within ≈1°, versus Phase 1's
tip-past-90°-in-0.5 s. The RMSE figures above are computed over the full 8-second
flight in all four conditions, with no crash-truncation needed (contrast Phase 1's
§3.1, where every MPC row is truncated at t≈0.42–0.50 s).

**Altitude authority under payload is not resolved.** The two payload rows above
are dominated by a loss of altitude: the controller holds the drone level but sinks
from 1.0 m to ≈0.01 m and remains on the ground for the rest of the flight. By the
0.05 m minimum-altitude criterion this project uses to screen training episodes
(`diagnostics/scan_bad_episodes.py`), those two runs would be classified as
failures. Reducing the payload from 25% to 15% of nominal mass does *not*
meaningfully change this — the drone still descends to ≈0.01 m — so it is a
control-authority limit (§7.3), not a threshold effect of one particular
disturbance magnitude.

Koopman-MPC does not beat PID on any tracking metric in any condition.

### 7.2 Short-horizon prediction validation

Since MPC only ever needs accurate prediction over its own horizon (10 steps ≈0.2s
here), not indefinitely, the fitted Koopman model was validated at the horizon
lengths that actually matter, using real held-out control sequences
(`diagnostics/validate_koopman_horizon.py`), rather than only via full-episode
open-loop rollout (which diverges for any nonlinear system approximated linearly,
and is not informative about closed-loop MPC performance):

| Horizon | Condition       | Mean pos. error (m) | 95th pct. error (m) |
| ------- | --------------- | -------------------- | ---------------------- |
| 5 steps (~0.10s)  | clean         | 0.0002 | 0.0005 |
| 5 steps (~0.10s)  | payload+wind  | 0.0033 | 0.0032 |
| 20 steps (~0.42s) | clean         | 0.0020 | 0.0043 |
| 20 steps (~0.42s) | payload+wind  | 0.0194 | 0.0315 |

The model itself is accurate at the timescales MPC actually operates at, even under
combined disturbance — the remaining gap to PID (§7.3) is a control-authority
question, not a model-accuracy question.

## 7.3 Why Koopman-MPC still underperforms PID

The control-effort column in §7.1 is the key evidence. Koopman-MPC's control effort
is essentially flat (≈0.0006–0.0009) regardless of disturbance severity, while PID's
scales by four orders of magnitude under payload (58.7 → 1,836,034). **Koopman-MPC
is not failing to detect the disturbance — §7.2 shows its predictions remain
accurate — it is failing to react to it with meaningful authority.**

This was investigated directly, not assumed. Increasing the QP's tracking weight or
reducing its effort penalty (to make the controller react harder) was tried at
multiple settings: below a certain gain, tracking stayed stable but sluggish; above
it, the controller reliably diverged or crashed over a multi-second horizon, even
with the §6.4 anchor in place. This is consistent with a general property of linear
data-driven models: prediction accuracy is trustworthy near the training
distribution and degrades as the commanded control differential grows, so driving
the model harder eventually pushes it outside its valid region regardless of QP
tuning.

Two further, more targeted attempts to close this gap were made and are reported
honestly as **negative results**, since they revealed real structural limits rather
than simply needing more tuning:

- **Adaptive weight-switching** (a more aggressive controller engaged only when
  tracking error spikes, with a hard time cap and cooldown): **no net benefit.** At
  a 0.15 m trigger threshold it made things substantially *worse* (clean-condition
  RMSE 0.29 → 1.71 m), because the controller's own baseline tracking lag already
  exceeds that threshold, so the aggressive mode engaged almost permanently and its
  overshoot re-triggered itself. Raising the threshold to 0.55 m (above the baseline
  lag) removed the false triggering but left the outcome numerically identical to
  the non-adaptive controller to four decimal places under payload — i.e. any
  threshold sensitive enough to catch the real disturbance also catches ordinary
  tracking lag, and any threshold conservative enough to avoid that is too
  conservative to help.
- **Integral action**, tried two ways: biasing the MPC's reference target by an
  accumulated-error term was diluted by the QP's own effort weighting and had
  negligible effect; a state-blind feedforward RPM offset added directly (bypassing
  the QP) caused an immediate crash, because it has no awareness of what the QP is
  simultaneously doing with attitude — the two mechanisms fight each other.
  Correctly implementing offset-free MPC requires augmenting the QP's own state and
  cost with the integral term (§8, item 1), not bolting it on externally.

### 7.4 Richer training data as a partial improvement

Retraining on a wider per-episode reference amplitude range (§6.2: 0.3–1.0 m, vs. an
earlier, narrower 0.4 m fixed amplitude) measurably improved both prediction
accuracy (in the §6.3 sign test, predicted lateral velocity $-0.086$ m/s vs. actual
$-0.112$ m/s, a ~23% magnitude underestimate with the correct sign, against a
pre-fix model that was both wrong-signed and ~5x off in magnitude) and closed-loop
degradation (172.6% → 135.8%), and
eliminated a previously-observed crash at an aggressive MPC weight setting — evidence
that the model's remaining limitation is at least partly addressable by training-data
coverage, not purely an architectural ceiling, though it did not close the full gap
to PID.

## 8. Future Work

Following directly from §7.3's evidence:

1. **Offset-free MPC via integral-state augmentation.** Add an explicit
   integral-of-error state into the QP's own state vector and cost function, so
   steady-state lag is corrected *within* the optimization on equal footing with its
   other objectives, rather than via either of the two external bolt-on attempts in
   §7.3 (both of which failed for different, now-understood reasons).
2. **Training data spanning much larger tilt angles, specifically.** §7.4 widened
   position-trajectory amplitude, but not the range of tilt angles the model was
   trained on. Following Phase 1's own §5.1/§5.2, deliberately collecting data that
   spans much larger roll/pitch angles and re-evaluating prediction accuracy
   specifically in that regime remains undone, and is likely necessary before a
   more aggressive MPC gain can be trusted.
3. **A proper cascaded MPC structure**, mirroring §6.6's finding for LQR: an outer
   position-tracking MPC producing a desired attitude, and a separate, faster inner
   attitude-tracking MPC or LQR, rather than one QP handling both timescales at once.
4. **Nonlinear MPC**, if the above is still insufficient — a larger undertaking,
   since it gives up the convexity that makes the current QP fast and reliable to
   solve, but would remove the "valid only near the training distribution"
   constraint that §7.3 identifies as the binding limitation.

---

## 9. Conclusion

This project, across two phases, replicates and then extends a Koopman-operator-based
MPC control system for quadrotors (Oh et al., 2024). Phase 1 faithfully reproduces
the base paper's approach, extends it with payload disturbance testing and a genuine
classical-control comparison, and — under a rigorously verified, fair experimental
setup — finds the replicated Koopman-MPC controller reliably unstable, tipping over
within approximately half a second regardless of disturbance type, while PID and LQR
complete every scenario. Isolated testing traces this to the controller's
Euler-angle/quaternion-based orientation representation, consistent with published
SE(3) Koopman-MPC literature, and identifies rotation-matrix-based lifting as the
specific next step.

Phase 2 independently implements that step, from scratch, and confirms the
diagnosis: with rotation-matrix lifting, the attitude instability disappears
entirely — roll and pitch stay within about a degree for full-length flights, in
every condition tested, and no run is crash-truncated. The resolution is specific
rather than total: under a mid-flight payload increase the Phase 2 controller keeps
the drone level but cannot hold altitude, sinking to the ground, so one failure mode
has been exchanged for a milder and better-understood one rather than eliminated
outright. In doing so, this
phase also finds and corrects two further, general issues — a closed-loop system
identification bias and an MPC cross-step consistency gap — both worth checking for
in any similar data-driven control pipeline. The now-stable Koopman-MPC still does
not outperform PID, and this is reported honestly, with direct control-effort
evidence explaining the mechanism: the controller is conservatively tuned to remain
within its model's valid region, at the cost of tracking authority PID does not need
to sacrifice. Concrete, evidence-grounded next steps are identified to close this
remaining gap.

---

## 10. References

1. Y. Oh, M. H. Lee, and J. Moon, "Koopman-Based Control System for Quadrotors in
   Noisy Environments," *IEEE Access*, vol. 12, pp. 71675–71684, 2024.
   <https://doi.org/10.1109/ACCESS.2024.3403104>
2. S. Narayanan et al., "SE(3) Koopman-MPC: Data-driven Learning and Control of
   Quadrotor UAVs," *IFAC-PapersOnLine*, 2023.
3. M. O. Williams, I. G. Kevrekidis, and C. W. Rowley, "A Data–Driven Approximation
   of the Koopman Operator: Extending Dynamic Mode Decomposition," *Journal of
   Nonlinear Science*, vol. 25, no. 6, pp. 1307–1346, 2015.

---

## Appendix: Running Phase 2

Built against the `v1.0.0` tag of `gym-pybullet-drones`, Python 3.11, and `cvxpy`
with the CLARABEL solver.

```
python v2_stable_koopman_mpc/data_collection/collect_koopman_data.py
python v2_stable_koopman_mpc/koopman_model/build_edmd_dataset.py
python v2_stable_koopman_mpc/koopman_model/fit_koopman.py
python v2_stable_koopman_mpc/diagnostics/diagnose_sign.py          # verify sign-correctness first
python v2_stable_koopman_mpc/diagnostics/validate_koopman_horizon.py
python v2_stable_koopman_mpc/mpc/run_mpc_closed_loop.py
python v2_stable_koopman_mpc/comparison/run_comparison.py          # final PID vs Koopman-MPC comparison
```

`baselines/run_lqr_closed_loop.py` and `diagnostics/diagnose_lqr_gain.py` are
retained for completeness but are **not** part of the reported results: the Phase 2
LQR baseline does not stabilize (§6.6), and `run_lqr_closed_loop.py` will crash
within a couple of seconds of simulated flight.

For the live demonstrations (GUI window, paced to real time):

```
python demo_flight_gym_pybullet.py                      # single Koopman-MPC flight
python demo_flight_conditions.py koopman_mpc pid        # all 4 disturbance conditions, both controllers
```

---

*Amrita Vishwa Vidyapeetham — Group CD7 — Drones, Semester 5*
