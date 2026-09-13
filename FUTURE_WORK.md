# Future Work: Rotation-Matrix Lifting, Excitation Fixes, and a Stable Koopman-MPC

This document reports work carried out **after** the results in the main
README, directly following up on the limitations and future-work items
identified there (§4, §5.1, §5.2). It should be read as a continuation of
that report, not a replacement for it — the original findings (Koopman-MPC
crashing within ~0.5s due to an Euler-angle/quaternion-based lifting
function) are confirmed as the correct diagnosis, and this work independently
arrives at and validates the fix the original report predicted but did not
complete.

All code for this phase lives in `v2_stable_koopman_mpc/`, built on a
different (older, single-drone) version of `gym-pybullet-drones` than the
original pipeline. It is a from-scratch reimplementation, not a patch to the
original `src/` code, since the state representation, lifting function, and
control-loop structure all needed to change together.

---

## Summary of outcome

**The instability described in §3.3/§3.4 of the main README is resolved.**
The new Koopman-MPC controller completes full-length (8 second) flights with
no crashes, across a wide reference trajectory range (0.3–1.0 m amplitude),
under payload and wind disturbance. This directly confirms the hypothesis in
§4: the crash was caused by the Euler-angle/quaternion lifting
representation, not by anything about the disturbance scenarios or MPC
formulation in general.

**However, a new, honest limitation replaces the old one**: the now-stable
Koopman-MPC controller does not outperform PID. It tracks less tightly
(RMSE ≈0.34 m vs. PID's ≈0.04 m) and degrades more under combined
payload+wind disturbance (135.8% vs. PID's 37.8%). Section 4 below explains
why, with supporting evidence, and section 5 lays out what would be needed to
close this remaining gap.

---

## 1. What was already correctly diagnosed in the main README

The original report's diagnosis (§4, §5.1) was right on the mechanism:
Euler-angle/quaternion-based lifting produces an orientation representation
with a singularity that the linear Koopman model cannot represent well near
extreme tilt, and the report correctly identified rotation-matrix-based
lifting (citing Narayanan et al., SE(3) Koopman-MPC) as the literature-backed
fix. §5.1's preliminary rotation-matrix investigation found inconclusive
results specifically because the training data (collected under calm,
PID-stabilized flight) never covered extreme tilt angles, so the improved
lifting function was never actually tested under the conditions that
mattered.

## 2. What this phase did differently

### 2.1 Rotation-matrix lifting, from the ground up

The new pipeline's lifting function uses the flattened 3×3 rotation matrix
directly (from `pybullet.getMatrixFromQuaternion`), never converting to or
from Euler angles or storing quaternions as lifted-state entries:

```
raw_state = [pos(3), vel(3), rotmat(9), ang_vel(3)]   # 18-dim
lifted     = [raw_state, vel⊙rotmat cross-terms, ang_vel⊙rotmat cross-terms, bias]  # 27-dim
```

This is structurally the same idea §5.1 proposed, arrived at independently
in this phase before that section of the original README was cross-referenced.

### 2.2 A second, previously-undiagnosed root cause: closed-loop identification bias

Rotation-matrix lifting alone was not sufficient. A second, distinct bug was
found during this phase: the training data's control inputs were generated
almost entirely by a PID controller *reacting to* position error, with only
a very small independent excitation signal on top. Fitting EDMD on this data
produced a model with **the wrong sign** on the coupling between
differential motor thrust and horizontal acceleration — confirmed directly
by commanding a known motor differential and comparing the model's predicted
direction of motion against the real simulator's actual direction (see
`diagnostics/diagnose_sign.py`).

This is a classic closed-loop system identification bias: when the training
input is mostly a feedback response to error rather than an independent
probe, the regression can learn the *controller's* corrective behavior
instead of the *plant's* true causal dynamics. The fix was to substantially
increase the independent excitation signal (an Ornstein-Uhlenbeck process on
commanded RPM, tuned to a level that measurably perturbs the dynamics
without itself causing instability — found by iterative testing to be
`sigma≈0.03`, with `sigma≈0.05` already causing 55% of training episodes to
crash and contaminate the dataset). After this fix, the model's predicted
direction of motion matched the real simulator's on all three axes.

This finding is worth carrying back to the main report's methodology
section: **the same closed-loop identification risk applies to the original
`src/koopman/` data collection**, since it also flew PID-stabilized
trajectories. It was not diagnosed in the original work because the
controller crashed before this particular failure mode had a chance to
matter for tracking (a wrong-signed model would degrade tracking accuracy,
which is a smaller effect than the orientation-singularity crash that
dominated the original results).

### 2.3 A genuine MPC formulation gap: no cross-step consistency anchor

The QP formulation's control-rate penalty (`‖u[t] − u[t-1]‖²`) only applies
*within* a single receding-horizon solve — it has no memory of what was
*actually applied* on the previous real control step. This means two
consecutive real-time solves could, in principle, jump to very different
"optimal" first actions even though each solve's internal plan looks smooth
in isolation. Anchoring the first predicted action to the last actually-applied
command (`prev_u`, see `mpc/koopman_mpc.py`) closed this gap and measurably
improved stability margin at higher-gain configurations (a previously-crashing
weight configuration, tested with anchoring, degraded to a bounded tracking
error rather than a crash).

### 2.4 Richer training data (wider maneuver amplitude/frequency)

The original training trajectories used a fixed 0.4 m amplitude. This phase
widened it to a per-episode-randomized 0.3–1.0 m range (with automatic
filtering of any resulting crashed/unstable episodes — final rate 1.0%,
`diagnostics/scan_bad_episodes.py`), on the hypothesis that a model trained
only on gentle motion cannot be expected to generalize when MPC asks it to
reason about larger control differentials. This measurably improved
degradation under disturbance (172.6% → 135.8%) and eliminated a
previously-observed crash at an aggressive MPC weight setting, though it did
not close the full gap to PID (see §4).

### 2.5 Systematic validation methodology

Before trusting the model in closed-loop control, three independent checks
were run and are included in `diagnostics/`:

- **Sign-consistency test** (`diagnose_sign.py`): apply a known motor
  differential, compare predicted vs. actual direction of motion on all
  three axes.
- **Short-horizon rolling validation** (`validate_koopman_horizon.py`):
  measure prediction error at 5/10/20/40-step horizons using real control
  sequences from held-out episodes — this is the horizon length that
  actually matters for MPC, as opposed to full-episode open-loop rollout
  (which diverges for any nonlinear system approximated linearly, and is
  not informative about closed-loop MPC performance).
- **Crashed-episode filtering** (`scan_bad_episodes.py`): automatically
  exclude any training episode with excessive roll/pitch, altitude
  collapse, or excessive drift, so a small number of unlucky random
  disturbance combinations cannot silently corrupt the regression.

## 3. Final comparison result (honest numbers)

Same six-scenario structure as the original report's §3.1, condensed to the
four conditions actually re-run in this phase (no disturbance, payload only,
wind only, payload+wind), using the same reference trajectory for both
controllers:

| Controller | Condition | RMSE (m) | Max err (m) | Control effort |
|---|---|---|---|---|
| PID | none | 0.0381 | 0.0583 | 58.7 |
| PID | payload | 0.0529 | 0.0680 | 1,836,034 |
| PID | wind | 0.0375 | 0.0583 | 96.7 |
| PID | payload+wind | 0.0525 | 0.0680 | 1,835,657 |
| Koopman-MPC | none | 0.3394 | 0.4560 | 0.0006 |
| Koopman-MPC | payload | 0.8117 | 1.1027 | 0.0009 |
| Koopman-MPC | wind | 0.4071 | 0.8211 | 0.0006 |
| Koopman-MPC | payload+wind | 0.8003 | 1.0865 | 0.0008 |

Relative degradation (clean → payload+wind): **PID 37.8%, Koopman-MPC
135.8%**.

Unlike the original report, Koopman-MPC **completes 100% of runs** (no
crashes) — this is the headline improvement. It does not, however, beat PID
on any tracking metric.

## 4. Why Koopman-MPC still underperforms PID (with evidence, not speculation)

The control-effort column above is the key piece of evidence. Koopman-MPC's
control effort is essentially flat and tiny (≈0.0006–0.0009) regardless of
disturbance severity, while PID's scales by four orders of magnitude when
carrying extra payload (58.7 → 1,836,034). **Koopman-MPC is not failing to
detect the disturbance — the model's predictions are accurate (sub-cm error
at 5–20 step horizons, `diagnostics/validate_koopman_horizon.py`) — it is
failing to react to it with meaningful authority.**

This was investigated directly. Increasing the QP's position-tracking weight
or lowering its control-effort penalty (to make the controller react harder)
was tried at multiple settings. Below a certain gain, tracking stayed stable
but sluggish; above it, the controller reliably diverged or crashed over a
multi-second horizon, even with the cross-step anchor from §2.3 in place.
This is consistent with a fundamental property of linear data-driven models:
prediction accuracy is trustworthy near the training distribution and
degrades as the commanded control differential grows, so driving the model
harder eventually pushes it outside its valid region regardless of how the
QP's weights are tuned.

Two further, more targeted attempts to close this gap were made and are
reported honestly as **negative results**:

- **Adaptive weight-switching** (trigger a more aggressive controller only
  when tracking error spikes, with a hard time cap and cooldown): reduced
  but did not eliminate the gap, and required careful threshold tuning to
  avoid a self-sustaining bad loop where the aggressive mode's own overshoot
  re-triggered itself.
- **Integral action** (to address the specific "steady, growing lag"
  failure mode observed in every stable configuration): tried two ways.
  Biasing the MPC's reference target by an accumulated-error term was
  diluted by the QP's own effort weighting and had negligible effect.
  Adding a state-blind feedforward RPM offset directly (bypassing the QP)
  caused an immediate crash, because it has no awareness of what the QP is
  simultaneously trying to do with attitude — the two mechanisms fight each
  other. Properly implementing offset-free / integral-augmented MPC (folding
  the integral term into the QP's own state and cost, rather than bolting
  it on outside) is a known, standard technique but requires reformulating
  the optimization itself; this was not completed in this phase.

## 5. Concrete next steps

These follow directly from §4's evidence, in priority order:

1. **Offset-free MPC via integral-state augmentation.** Add an explicit
   integral-of-error state to the QP's own state vector and cost function
   (rather than the two外部 bolt-on attempts in §4, both of which failed for
   different reasons), so steady-state lag is corrected from *within* the
   optimization, competing on equal footing with the QP's other objectives
   rather than being diluted or fighting them externally.
2. **A genuinely richer lifting function for large-tilt regimes.** Even
   after widening the training trajectory amplitude (§2.4), the model still
   degrades when MPC is driven aggressively. Following up on the original
   README's own §5.1/§5.2, deliberately collecting training data that spans
   much larger tilt angles (not just larger position amplitude) and
   re-evaluating rotation-matrix prediction accuracy specifically in that
   regime remains undone, and is likely necessary before a more aggressive
   MPC gain can be trusted.
3. **A proper cascaded MPC structure**, mirroring why PID succeeds: an
   outer position-tracking MPC producing a desired attitude, and a separate,
   faster inner attitude-tracking MPC (or LQR) — rather than one QP handling
   both timescales at once. This mirrors the fix that was needed for this
   phase's own LQR baseline (see `baselines/lqr_baseline.py`, which required
   an explicit outer/inner cascade with a hard tilt-angle clamp before it
   would stabilize at all) and is a plausible explanation for part of the
   remaining PID gap.
4. **Nonlinear MPC**, if the above still isn't sufficient — a larger
   undertaking, since it gives up the convexity that makes the current QP
   fast and reliable to solve, but would remove the fundamental "valid only
   near the training distribution" constraint that appears to be the
   binding limitation identified in §4.

---

*This document supersedes the crash-based conclusion in the main README's
§3–§6 with respect to controller completion rate only. The main report's
diagnosis of the root cause (§4) and its literature-grounded proposed fix
(§5.1–§5.2) are both confirmed correct by this phase's results.*
