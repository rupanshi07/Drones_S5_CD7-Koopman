# Analysis outputs — read this first

## Which files are authoritative

Two versions of the metrics pipeline were run: `src/analysis/compute_metrics.py`
(v1) and `src/analysis/Compute_metrics_v2.py` (v2 / "final"). **Only the
`*_final.*` files and `completion_summary_final.csv` should be used or
cited.** v1 does not detect controller crashes, so it computes RMSE and
other metrics over the *entire* 15s trace even for runs where the drone
flipped and the simulator reset mid-run — this silently mixes real flight
data with post-crash reset artifacts and understates how bad the MPC
crashes actually are. v2 detects the first failure (ground contact, a
>90° roll/pitch flip, or a `env.reset()` discontinuity) and truncates all
metrics at that point.

The non-`_final` files (`metrics_table.csv`, `percentage_improvement.csv`,
`best_controller_summary.csv`, `metrics_summary.png`, and the
`settling_plot_scenario_*.png` files) are kept for reference only — do not
use their numbers in the final writeup.

## Correction applied — target-height mismatch (resolved)

An earlier version of this analysis was computed while `FINAL_TARGET_Z` in
`mpc_controller.py` and `Compute_metrics_v2.py` was mismatched at 0.1125 m
(the drone's own starting height) instead of the intended 0.3 m target that
PID and LQR were correctly targeting. This made PID and LQR's task trivially
easy in the earlier numbers and understated MPC's tracking error relative to
its actual assigned task. **This has since been fixed**: all three
controllers are now evaluated against the same 0.3 m target, and all 18
scenario files, the payload Koopman model, and this metrics table have been
regenerated accordingly. A separate data-corruption bug in payload flight
data collection (repeated `p.changeDynamics()` calls destabilizing the
simulator even at zero added mass) was also found and fixed in the same
pass — see the project README for full details.

**The corrected numbers strengthen the headline result**, they do not
change it: with a fair target, Koopman-MPC now loses on tracking RMSE as
well as completion rate, not completion rate alone.

## Why so many cells are blank — this is expected, not a bug

**Koopman-MPC crashes (flips) in all 6 scenarios**, now consistently around
t ≈ 0.42–0.50s into a 15s run (faster than the ≈0.75–0.98s observed under
the earlier, incorrect 0.1125 m target — consistent with the drone reaching
the correct, higher 0.3 m target and losing orientation control shortly
after arrival, confirmed via isolated diagnostic testing). See
`completion_summary_final.csv`: MPC completes only a few percent of every
scenario before failing; PID and LQR complete 100% of every scenario.

This has two direct consequences visible in the tables:

1. **`percentage_improvement_final.csv` has an empty `pct_improvement`
   column everywhere.** This is deliberate: the script refuses to state a
   percentage improvement over a controller that crashed, since "PID beat
   a crashed run by X%" isn't a meaningful number. Because MPC crashes in
   every scenario, this column is empty in every row.

2. **`settling_time_s` and `max_transient_deviation_m` are blank for MPC
   in scenarios 4, 5, and 6.** The wind transition (scenario 4) and
   payload drop (scenarios 5/6) are both scripted to trigger at t=7.5s
   (`NUM_STEPS // 2`), but MPC has already crashed by t<1s in every case.
   The disturbance these scenarios are supposed to test never actually
   reaches MPC; the controller is already down. So these scenarios never
   test MPC's disturbance rejection — only its ability to survive to t=7.5s
   at all, which it doesn't.

## Scenario 2 vs. Scenario 3 — identical PID/LQR rows are expected

Scenarios 2 and 3 are a wind ablation defined only in terms of the
Koopman-MPC environment selector (off vs. on) — the physical wind
disturbance is identical in both. PID and LQR don't use the selector at
all, so their scenario 2 and scenario 3 result files are byte-for-byte
identical by design. The scenario 2 vs. 3 comparison is only meaningful
for the MPC row.

## Headline result

Across every metric and every scenario, PID or LQR wins
(`best_controller_summary_final.csv`) — Koopman-MPC does not win a single
metric in any scenario, including tracking RMSE now that the target
mismatch is fixed, and does not survive any scenario to completion.
`completion_rate_comparison.png` is the clearest single figure for this.

## Update following direct isolation testing

Further isolated testing (holding a fixed hover target with zero commanded
climb and zero disturbance applied) confirmed that the MPC instability is
not specific to reaching disturbance trigger times or climbing to target
height. The controller becomes unstable even during plain, undisturbed
hovering, indicating a fundamental limitation in the current Koopman
lifting function's orientation representation (Euler angles and
quaternions), consistent with known singularity issues described in
related published work (Narayanan et al., SE(3) Koopman-MPC,
IFAC-PapersOnLine, 2023). This is a more precise characterization than
"MPC does not survive long enough to reach the disturbance" — MPC does not
survive independent of any disturbance at all.
