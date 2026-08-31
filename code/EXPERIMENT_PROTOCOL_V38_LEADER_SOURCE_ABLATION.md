# V38 Paired Doppler-Reference and Active-Acquisition Ablation

## Scientific question

Does a follower localize and track more reliably when it uses two Doppler
references rather than one, and does belief-conditioned active acquisition
improve the result beyond a fixed excitation manoeuvre?

This experiment is a component ablation. It does not change the formation task,
the audited ACQUIRE-to-TRACK gate, the post-lock PID controller, the plant, the
noise model, or the fixed 440 s horizon.

## Frozen 3 x 2 design

Every episode seed is evaluated in all six cells:

| Doppler reference(s) used for inference | Fixed S-turn acquisition | Belief-conditioned active acquisition |
|---|---:|---:|
| leader 1 only | yes | yes |
| leader 2 only | yes | yes |
| both leaders | yes | yes |

All six cells use the same episode seed and the same exogenous-noise tape. Arms
are executed sequentially.

## Information boundary

The source mask is applied before the estimator and planner.

- `leader1_only` exposes only leader 1 position, velocity, and Doppler.
- `leader2_only` exposes only leader 2 position, velocity, and Doppler.
- `both_leaders` exposes both columns.
- The likelihood, residual Jacobian, Hessian, covariance, local and forward
  residual RMSE, alternative-mode evidence, and active-planner prediction all
  operate on the masked history.
- An excluded link is removed, not assigned a small or zero numerical weight.
  Consequently every RMSE denominator is `time rows x active sources`.
- Both physical leader broadcasts remain common to all six arms for three
  purposes only: the reset-time centroid of the frozen \([120,350]\)-m prior
  shell, the identical two-leader velocity reference used by the fixed S-turn
  and active candidate-action anchor, and the unchanged formation target during
  TRACK. These common quantities do not contain Doppler measurements.
- Active-planner information prediction, pairwise discrimination, likelihood,
  Jacobians, residuals, and gate evidence use only the admitted
  leader-state/Doppler column(s).
- The legacy particle filter and simulator truth are excluded from the
  controller. Truth is used after each action for scoring only.

## Frozen method

- Globalized causal Doppler-history estimator: same schedule and numerical
  settings as the publication method.
- Global refresh times: 30, 60, 90, 120, 180, 240, 300, 360, 420, and 440 s.
- Independent primary and confirmation searches: unchanged deterministic seeds.
- Audited ACQUIRE-to-TRACK gate: unchanged V24 thresholds, debounce, hysteresis,
  and reacquisition.
- Active acquisition: unchanged V22 action bank, horizon, and utility weights,
  evaluated only for active sources.
- Fixed acquisition: the same deterministic two-leader-referenced S-turn in
  every source arm until the audited gate releases TRACK.
- Active candidate bank: anchored to that same common S-turn in every source
  arm; only the information score changes with the admitted Doppler references.
- TRACK: the same PID and the same two-leader desired formation in every cell.

## Cohorts

- Development campaign: 100 paired seeds, 48400--48499.
- Full-settings smoke: two paired seeds, 48598 and 48599.
- Reserved/final range 49900--50999 remains closed.
- The full development campaign must not start until the full-settings smoke and
  independent causal audit pass.

The smoke uses the publication estimator settings: 4096 coarse candidates,
two sweeps, and 48 local starts. Reduced settings are permitted only as
explicit debug overrides and cannot pass the publication integrity gate.

## Primary endpoints

For each cell:

1. terminal joint success: formation error below 8 m and localization error
   below 7 m;
2. Tail80 joint success over the final 50 actions;
3. localization error and success below 7 m at the 60-s checkpoint;
4. terminal localization error;
5. terminal formation error;
6. unsafe gate transitions and unsafe TRACK actions.

The 60-s checkpoint is a common-trajectory source-count comparison only in the
three fixed-S-turn arms. Their commanded actions and physical histories must be
identical through this checkpoint, and neither `phase_track` nor the
post-update gate state may be active at 60 s. Later terminal outcomes are
system-level effects because source-dependent release times intentionally cause
the trajectories to diverge.

Robustness endpoints:

- median, p90, p95, and maximum terminal errors;
- maximum localization error over the episode;
- maximum localization error during TRACK;
- false-confidence actions (nominal local radius below 7 m while actual
  localization error is at least 7 m);
- dwell-15 success, ever-lock rate, first TRACK time, action effort, and
  maximum combined estimator/planner decision runtime.

## Prespecified paired contrasts

1. active acquisition minus fixed S-turn within each of the three source masks;
2. both leaders minus leader 1 only within each acquisition policy;
3. both leaders minus leader 2 only within each acquisition policy.

The fixed-policy both-minus-single contrasts at 60 s isolate the added Doppler
reference on a common trajectory. The active-policy terminal contrasts measure
the complete closed-loop consequence of source-dependent estimation, planning,
release, and subsequent formation tracking.

Binary endpoints report paired discordances, success-rate differences, and an
exact two-sided McNemar test. Continuous endpoint differences report the paired
mean, median, and a deterministic 20,000-replicate bootstrap 95% interval.
Positive error improvement means that the right-hand method has lower error.

## Integrity gates

A campaign is valid only when:

- all six arms exist for every seed;
- all traces contain the fixed action horizon;
- the six arms share the seed, noise-tape hash, and mission support;
- saved masks match the declared source arm;
- active scalar counts equal rows times active-source count;
- the fixed S-turn arms never invoke the belief planner;
- fixed-arm actions are identical through 60 s, and no fixed arm has released
  TRACK by that checkpoint;
- audited release violations are zero;
- maximum combined online decision runtime is below 2 s;
- source hashes remain unchanged throughout execution;
- the reserved/final range is untouched;
- the development seed range was unused before campaign freeze.

## Claim gate for the 100-seed development campaign

The result supports the narrow statement that two Doppler references plus
active acquisition improve reliable localization and tracking only if all
conditions below hold:

1. the campaign integrity gate passes;
2. the both-leader active cell has at least 90% terminal success, at least 85%
   Tail80 success, p95 terminal localization error below 7 m, and zero unsafe
   transitions or unsafe TRACK actions;
3. within the both-leader source mask, active acquisition improves terminal
   success by at least 5 percentage points over fixed S-turn and has a positive
   paired mean localization-error improvement;
4. within the active policy, the both-leader cell improves terminal success by
   at least 5 percentage points over each one-leader cell and has a positive
   paired mean terminal localization-error improvement against each;
5. on the identical fixed trajectory at 60 s, the both-link cell improves the
   localization success rate by at least 5 percentage points over each
   single-link cell and has a positive paired mean error improvement against
   each.

Failure is reported as a result. It does not authorize threshold tuning, seed
replacement, or a new algorithmic version.

## Files and audit

- Core: `uuv_v38_leader_source_ablation.py`
- Runner: `run_v38_leader_source_ablation.py`
- Tests: `tests/test_uuv_v38_leader_source_ablation.py`
- Independent audit: `audit_v38_leader_source_ablation.py`

The audit reconstructs the masked history from each trace, recomputes all task
and gate safety scores, verifies scalar denominators, validates the six-arm
pairing, and checks frozen source hashes without trusting runner decisions.
