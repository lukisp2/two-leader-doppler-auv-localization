# V35 closed-loop stress and evidence-qualified bias-model switch — frozen protocol

Date frozen: 2026-07-16  
Status: development-only closed-loop stress campaign; no RL training, planner
retuning, reserved qualification or final evaluation

## Questions

1. Which previously diagnosed measurement-model mismatches propagate through
   the complete frozen V24 stack (`V19 + V22 + audited gate + PID`) into false
   TRACK decisions or task failure?
2. Can the unchanged V31/V32 causal evidence rule, evaluated once at 300 s,
   safely select the V30 two-link profiled-bias model and recover terminal or
   Tail80 performance under matched constant Doppler bias?

V33 did not authorize a nuisance-aware acquisition score.  V35 therefore
keeps the V22 planner unchanged.  It tests only the measurement-model switch.

## Data boundary and seed registry

- Engineering smoke uses only the already opened V24 seeds `49566` and
  `49591`.  Smoke has no scientific meaning.
- The full development campaign uses candidate fresh seeds `48100..48199`,
  exactly 100 paired scenarios.  Before opening them, a filesystem/metadata
  audit must find no previous scenario result with an `episode_seed` in this
  interval.  Failure aborts the campaign.
- Reserved qualification seeds `49900..49999` and final seeds
  `50000..50999` are rejected by the runner and remain closed.
- Simulator truth is never passed to the stress channel, estimator, planner,
  model-evidence calculation, gate or PID.  Truth is opened only after an
  action for scoring.

## Frozen environment and common random numbers

Use the same full-difficulty environment metadata as V24, including 220
actions of 2 s, one Doppler row per second, the V19 publication-size global
search (`4096` candidates, two sweeps, 48 refinements), raw history,
uniform-radius support, the frozen V22 action bank and the frozen V24 audited
gate thresholds.

For a scenario, every condition and arm starts from the same episode seed and
uses the same frozen exogenous noise tape.  A separate deterministic stress
tape is keyed only by episode index and condition, never by arm or result.
Stress families are separate and are not stacked.

## Frozen stress families

The primary arm is evaluated in all eight conditions:

1. `nominal` — unchanged online channels;
2. `doppler_common_bias_p003` — add `[+0.03,+0.03] m/s`;
3. `doppler_differential_bias_003` — add `[+0.03,-0.03] m/s`;
4. `doppler_scale_102` — multiply measured Doppler by `1.02`;
5. `colored_noise_rho09_sd003` — independent per-link stationary AR(1),
   `rho=0.9`, marginal standard deviation `0.03 m/s`;
6. `dropout_iid10` — omit a complete two-link measurement row with
   probability `0.10`;
7. `broadcast_delay_2s` — give estimator, planner and PID the latest leader
   position/velocity broadcast at least 2 s old; and
8. `dead_reckoning_scale_101` — multiply measured follower velocity and its
   integrated displacement by `1.01`; planner/PID receive the correspondingly
   scaled measured speed.

All constants match V28.  Physical truth and leader/follower dynamics remain
unchanged.  The legacy PF is outside the controller information path and is
allowed to receive the original nominal measurement; V35 records and uses the
stressed deployable channel explicitly.

## Frozen arms

### `v24_nominal_model`

The unchanged V24 estimator, V22 planner, audited gate and PID.  This arm is
run in all eight conditions (800 full-campaign runs).

### `v35_bias_gate_300`

Run only for nominal, common bias, differential bias and colored noise (400
full-campaign runs).  It is bitwise/controller-state identical to
`v24_nominal_model` through the post-update state at 300 s.  At 300 s exactly
once:

1. evaluate the unchanged V32 general-checkpoint evidence rule with the
   unchanged V31 constants and a 225-s training / 75-s held-out split;
2. if evidence fails, retain the nominal V24 model forever;
3. if evidence passes, declare an evidence-qualified measurement-model
   switch, force `TRACK -> ACQUIRE` if necessary, reset release debounce, and
   activate the V30 two-link variable-projected estimator;
4. construct causal profiled global evidence from a retrospective 240-s
   profiled solve and two differently seeded 300-s profiled global
   replications.  The 240-s estimate predicts `(240,300]`, providing 60
   held-out rows and global-stability evidence without future data;
5. after activation use profiled local refinement at ordinary decisions and
   two profiled global replications at the unchanged V21 refresh times
   `360,420,440 s`; residual checks subtract each fitted link bias;
6. reuse the unchanged V24 audit thresholds and three-action release debounce,
   and reuse the unchanged V22 planner.  No V33 score or new planner weight is
   allowed.

The model switch itself is not a successful ACQUIRE-to-TRACK transition.  A
new TRACK period begins only after the profiled evidence passes the complete
audited gate.

## Stress-channel implementation contract

The V35 online recorder must be append-only code; V24 and V30--V32 sources are
not edited.  It intercepts deployable measurements before constructing the
V19 history.  A truth-free online environment view supplies the same delayed
leader broadcast and/or scaled follower speed to both V22 and PID.  Thus the
broadcast-delay condition cannot silently use current leader truth for
planning or formation control.

Dropout removes rows from the V19 history but does not alter time or dead
reckoning.  Every persisted episode records hashes of the nominal noise tape,
stress tape and resulting online history.

## Frozen endpoints

Primary task endpoints, using the existing exact V24 semantics:

- terminal joint success at 440 s: formation error `<8 m` and localization
  error `<7 m`;
- Tail80 success: at least 80% joint-success occupancy over the final 50
  actions;
- all ACQUIRE-to-TRACK transition, TRACK-start and TRACK-end localization
  errors, unsafe at `>=7 m` or non-finite.

Secondary endpoints:

- ever-lock, first/re-lock times, unlock count and time to sustained joint
  success;
- Dwell15, terminal localization/formation median, p95 and maximum;
- tail occupancy, action effort, estimator/planner/model-evidence runtime;
- nominal local-radius coverage as a descriptive diagnostic only;
- for the bias-switch arm: activation, evidence components, fitted biases,
  position jump, post-300 unsafe actions and time from switch to re-lock and
  restored joint success.

Rates receive Wilson intervals.  Binary paired arm differences use exact
discordance/McNemar reporting; continuous paired effects use 50,000
deterministic percentile-bootstrap resamples with PCG64 seed `35048100`.
Stress families are reported separately, never pooled into a reassuring
average.  Inferential p-values across families, if shown, use Holm correction.

## Frozen integrity gate

The full result is valid only if all hold:

- 1200/1200 contracted runs and fixed 220-action traces complete;
- source closure and an independent raw-trace audit pass;
- reserved/final ranges remain untouched;
- conditions/arms pair on initial state and exogenous/stress tapes;
- paired arms are identical through the 300-s treatment point;
- every reported V24 release satisfies all six added audit checks;
- no non-finite control action or required estimator output; and
- maximum combined estimator, planner and evidence decision runtime is below
  2 s.

Smoke may return only `SMOKE_PASS` or `SMOKE_FAIL`.

## Frozen V24 interpretation gates

Nominal V24 support requires terminal success at least 95/100, Tail80 at
least 90/100, ever-lock at least 95/100, zero unsafe transition/TRACK-start/
TRACK-end events and zero audit-release violations.

Each stressed family is independently labelled `SUPPORTED_STRESS_FAMILY`
only if terminal success is at least 90/100, Tail80 at least 85/100 and there
are zero unsafe transition/TRACK-start/TRACK-end events.  A family failure is
reported as a boundary of the model; it does not authorize threshold tuning
on these seeds and does not by itself invalidate the nominal method.

## Frozen profiled-switch retention gate

`RETAIN_EVIDENCE_QUALIFIED_PROFILED_BIAS_SWITCH` requires all:

1. activation at 300 s is at least 95/100 in each true-bias family, at most
   5/100 nominally and at most 10/100 under colored noise;
2. at 440 s localization success is at least 95/100 and localization p95 is
   at most 3 m in each true-bias family;
3. for each bias family, the lower 95% paired-bootstrap endpoint for terminal
   localization-error improvement versus V24 exceeds 1 m;
4. in each bias family at least one of terminal or Tail80 improves by at
   least five percentage points, and neither endpoint regresses by more than
   two points;
5. nominal terminal and Tail80 lose at most two successes each, nominal p95
   localization rises by at most 0.5 m, colored terminal/Tail80 lose at most
   five successes each and colored p95 rises by at most 1 m;
6. no unsafe TRACK event occurs after a profiled re-lock; and
7. the integrity/runtime gate passes.

Failure yields `DO_NOT_RETAIN_PROFILED_SWITCH_FOR_MAIN_METHOD`.  If position
improves but task endpoints do not, the switch may be reported only as an
estimator/model-selection ablation.

## Execution order and stopping rule

1. Freeze this protocol.
2. Implement append-only V35 code, independent audit and tests.
3. Run smoke on `49566,49591`; fix implementation defects only, never
   scientific thresholds.
4. Verify `48100..48199` are fresh.
5. Run all 1200 arms sequentially under `caffeinate`, with atomic episode
   checkpoints, resumability, persistent console log and PID record.
6. Do not start concurrently with another heavy campaign.
7. Independently audit and report the frozen decision.  Do not open
   `49900..50999` and do not retune after viewing V35.

