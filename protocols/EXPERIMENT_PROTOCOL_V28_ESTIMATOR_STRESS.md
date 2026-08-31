# V28 estimator stress and calibration — frozen development protocol

Date frozen: 2026-07-15  
Status: development-only model-mismatch stress campaign; no controller tuning,
no RL training, no reserved/final evaluation

## Motivation

V27 established three facts on nominal recorded trajectories:

1. full Doppler history and continuous local refinement are necessary;
2. a corrected 16384-particle Liu-West filter is accurate but slower than
   nonlinear least squares; and
3. six deterministic full-history NLS starts match the expensive global/local
   point estimate almost exactly after 240 s.

The remaining reason to retain global coverage and multiple modes is
robustness to ambiguity and model mismatch.  V28 freezes that question before
any stress outcome is observed.

## Data and boundary

- Source histories are the same already opened V19 development archive,
  scenarios `45000..45099`.
- Estimator inputs are produced only by deterministic transformations of the
  V19 `online_inputs.npz` allowlist.
- Truth labels remain separate and are opened only after every unscored
  condition/arm/checkpoint output for an episode has been persisted.
- Smoke uses episode indices `0` and `73` only and has no scientific meaning.
- Reserved `49900..49999` and final `50000..50999` remain untouched.

V28 diagnoses the model before the one-shot reserved closed-loop campaign.  It
is not publication validation and does not authorize the final holdout.

## Frozen arms

At causal checkpoints `120` and `440 s`, every stress history is evaluated by:

1. `global_full`: the selected V27 full-history two-sweep/48-start global-local
   estimator;
2. `local_nls6`: six deterministic axis starts with full-history damped
   Gauss-Newton refinement; and
3. `pf_lw_16384`: the corrected uniform-radius Liu-West particle filter.

All arm settings, support radii and measurement sigma remain identical to V27.
The stress generator never receives simulator truth.

## Frozen stress families

Each condition starts from the nominal recorded history; stresses are not
stacked.  The condition is applied once to the full 440-s history and causal
prefixes are then taken, preserving nested inputs.

1. `nominal`: byte-equivalent online history.
2. `doppler_common_bias_p003`: add `+0.03 m/s` to both leader range-rate
   measurements.
3. `doppler_differential_bias_003`: add `+0.03 m/s` to leader 1 and
   `-0.03 m/s` to leader 2.
4. `doppler_scale_102`: multiply both measured range rates by `1.02`.
5. `colored_noise_rho09_sd003`: add independent stationary AR(1) noise per
   leader with `rho=0.9` and marginal standard deviation `0.03 m/s`.
6. `dropout_iid10`: remove 10% of measurement rows using a fixed experiment
   RNG keyed only by episode index; rows at 1, 120 and 440 s are retained.
7. `broadcast_delay_2s`: replace leader position and velocity at every row by
   the most recent broadcast two seconds earlier (first rows reuse the first
   available broadcast).
8. `broadcast_offset_2m`: add fixed position offsets `[2,-1,0.5] m` and
   `[-2,1,-0.5] m` to leaders 1 and 2 respectively.
9. `dead_reckoning_scale_101`: multiply dead-reckoned displacement and measured
   follower velocity by `1.01`.
10. `dead_reckoning_drift_001`: add the fixed velocity bias
    `[0.0087287156,-0.0043643578,0.0021821789] m/s` (norm `0.01 m/s`) to
    measured follower velocity and its time integral to displacement.

The AR(1) and dropout generators use PCG64 seed
`28000000 + 1009*episode_index + condition_code`.  Their outputs are shared by
all estimator arms.  No result-dependent stress strength or exclusion is
allowed.

## Smoke

Smoke evaluates all ten conditions, three arms and both checkpoints on the two
fixed smoke episodes.  It uses the V27 reduced search settings and 2048
particles for the PF arm.  It passes only if all 120 cells complete, stress
hashes pair exactly across arms, deterministic repeats match apart from
runtime, truth does not enter unscored payloads, the independent audit
recomputes every error and closed seed ranges remain untouched.

## Metrics

For each condition/arm/checkpoint:

- success count at endpoint error `<=7 m`;
- mean, median, p95 and maximum endpoint error;
- estimator runtime;
- nominal-radius empirical coverage when defined;
- `q95(error/radius)` as a descriptive development inflation factor; and
- paired mean error difference and 95% percentile-bootstrap interval against
  `global_full` from 50,000 PCG64 resamples, seed `28045000`.

Calibration factors estimated here are diagnostic and cannot be reused as
independently validated confidence claims.

## Frozen development decisions

### Robustness gate

`global_full` may proceed to the reserved closed-loop qualification only if:

- integrity and independent audit pass;
- nominal regression is 100/100 at 120 and 440 s;
- every stressed family has at least 80/100 success at 120 s and 90/100 at
  440 s;
- no arm output required by the contract is non-finite; and
- nearest-rank p99 runtime at 440 s is below 2 s in every condition.

Passing gives `PROCEED_TO_RESERVED_CLOSED_LOOP_QUALIFICATION`.  Failure gives
`MODEL_EXTENSION_REQUIRED_BEFORE_CLOSED_LOOP` and names each failing family.

### Global-search simplification gate

Global search has material point-estimation headroom only when at least one
stress family satisfies both:

- `global_full` 440-s success exceeds `local_nls6` by at least 5 percentage
  points; and
- the paired 95% interval for `error(local_nls6)-error(global_full)` has lower
  endpoint above `0.5 m`.

If no condition passes both checks, the point estimator is simplified to
NLS-6 for the next controller study.  Global/retained-mode solves may still be
kept as a separate ambiguity and gate-evidence module, but not claimed as
necessary for nominal point accuracy.

### Calibration decision

Any arm/condition with defined-radius coverage below 90% at 440 s is marked
`REQUIRES_CALIBRATION_OR_MODEL_EXTENSION`.  Coverage is reported per family;
stresses are never pooled into a single reassuring average.

## Next stage

After V28, freeze either the unchanged or model-extended estimator.  Then write
one complete reserved-range protocol that simultaneously covers acquisition
baselines, V23/V24 gate comparison, supported estimator simplification and the
selected closed-loop stress families.  Only after that one-shot qualification
and a frozen analysis plan may `50000..50999` be opened once.
