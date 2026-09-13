# v2: Stable Koopman-MPC (Rotation-Matrix Lifting)

This folder contains a from-scratch reimplementation built after the results
in the main repository README. See **`../FUTURE_WORK.md`** for the full
writeup of what changed, why, and the honest final results.

**Quick summary:** the original pipeline's Koopman-MPC crashed within ~0.5s
in every scenario due to Euler-angle/quaternion-based lifting. This folder's
pipeline uses rotation-matrix-based lifting instead (as the main README's
own §5.1/§5.2 predicted would help), fixes a separately-discovered
closed-loop system identification bias, and adds a cross-step consistency
fix to the MPC formulation. The result: **Koopman-MPC now completes every
scenario with no crashes** — but still does not outperform PID on tracking
accuracy or disturbance degradation. `FUTURE_WORK.md` explains why, with
evidence, and lays out concrete next steps.

## Folder structure

| Folder | Contents |
|---|---|
| `data_collection/` | Flight data collection with PID + independent excitation, payload/wind disturbance injection |
| `koopman_model/` | Builds the EDMD training dataset and fits the Koopman (A, B) matrices |
| `mpc/` | The Koopman-MPC controller (QP formulation) and closed-loop test runner |
| `baselines/` | LQR baseline (cascaded outer-position/inner-attitude design) |
| `comparison/` | Runs PID vs. Koopman-MPC through a fixed disturbance test matrix |
| `diagnostics/` | Sign-consistency checks, short-horizon validation, crashed-episode filtering, and other verification scripts used throughout development |

## Requirements

Built against the `v1.0.0` tag of `gym-pybullet-drones`
(https://github.com/utiasDSL/gym-pybullet-drones), Python 3.11, and `cvxpy`
with the CLARABEL solver. This is an older API than whatever version the
original `src/` pipeline may have used — the two pipelines are not
drop-in compatible.

## Suggested run order

```
python data_collection/collect_koopman_data.py
python koopman_model/build_edmd_dataset.py
python koopman_model/fit_koopman.py
python diagnostics/diagnose_sign.py          # verify model sign-correctness before trusting it
python diagnostics/validate_koopman_horizon.py
python mpc/run_mpc_closed_loop.py
python baselines/run_lqr_closed_loop.py
python comparison/run_comparison.py          # final PID vs Koopman-MPC comparison
```
