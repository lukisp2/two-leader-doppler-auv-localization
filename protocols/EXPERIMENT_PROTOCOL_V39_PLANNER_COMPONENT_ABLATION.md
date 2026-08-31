# V39 Paired Planner-Component Ablation

## Status and scientific question

This protocol is frozen before any V39 development result is inspected.  It
tests which component of the two-reference active-acquisition loop produces a
material closed-loop benefit:

1. pairwise prediction separation between retained position hypotheses;
2. retaining multiple hypotheses in the worst-case information objective; and
3. belief-informed action selection rather than an uninformed feasible action.

This is not a reinforcement-learning experiment.  The estimator, audited
ACQUIRE-to-TRACK gate, post-lock PID controller, plant, measurement model,
noise model, two admitted Doppler links, action bank, and 440 s horizon are
identical in all arms.

## Frozen four-arm intervention

Every episode seed is evaluated sequentially in all four arms:

| Arm | Retained hypotheses | Pair weight | Acquisition action selection |
|---|---:|---:|---|
| `full_active` | at most 16 | 0.25 | frozen belief-conditioned utility |
| `no_pair_term` | at most 16 | 0 | otherwise identical utility |
| `best_hypothesis_only` | exactly the current best hypothesis | 0.25, but pair term is identically zero | otherwise identical utility |
| `random_feasible` | not used by action selector | not used | uniform draw from the same feasible candidate bank |

The first three arms have identical physical actions until the first planner
decision.  `full_active` versus `no_pair_term` changes only the pairwise
separation weight.  `no_pair_term` versus `best_hypothesis_only` isolates the
worst-case local-information calculation over retained hypotheses.  The
prespecified headline retained-hypothesis contrast is `full_active` versus
`best_hypothesis_only`, as it tests the complete multimodal planner against its
single-hypothesis reduction.

The random-feasible comparator is chosen instead of another hand-designed
trajectory because it uses exactly the frozen candidate bank and decision
rate, but contains no localization belief, information matrix, residual,
covariance, competing mode, gate evidence, particle-filter state, reward, or
truth.  Before the first estimator solution it uses the same deterministic
S-turn fallback as the active arms.  Thereafter it draws uniformly from the
candidate bank using a dedicated counterfactual policy RNG derived only from
the episode seed and a frozen domain separator.  The policy RNG is independent
of the environment noise tape.  Candidate indices, candidate counts, RNG seed,
and a policy-tape hash are saved and independently reconstructed.

## Information and truth boundary

- Both leader-state and Doppler columns are admitted in every arm.
- Estimation, gate evidence, and TRACK use the frozen two-link method.
- The active planners receive only deployable Doppler history, dead reckoning,
  leader broadcasts, follower kinematics, and estimator state.
- The random-feasible selector receives only the current feasible action bank
  and its private policy RNG.
- Simulator truth is read only after an action has completed and is used only
  for scoring.
- The legacy particle filter is excluded from all controller decisions.

## Frozen numerical method

- Estimator: globalized causal Doppler-history estimator with 4096 coarse
  candidates, two sweeps, 48 local starts, raw likelihood, and uniform-radius
  support sampling.
- Global refresh times: 30, 60, 90, 120, 180, 240, 300, 360, 420, and 440 s.
- Gate: frozen audited release, debounce, hysteresis, and reacquisition rules.
- Active action bank, 30 s prediction horizon, information utility, energy
  penalty, and action-change penalty: frozen publication settings.
- TRACK: identical two-leader formation reference and PID in all arms.
- Horizon: exactly 220 actions (440 s), without early success termination.

## Cohorts

- Full-settings smoke: seeds 48998 and 48999.
- Development campaign: 100 fresh paired seeds, 48800--48899.
- Reserved/final range 49900--50999 remains closed.

The development campaign must not begin until unit tests, the two-seed
full-settings smoke, and the independent causal audit pass.  Reduced estimator
settings are debug-only and cannot pass the publication integrity gate.

## Endpoints

Primary binary endpoints:

1. terminal joint success: formation error below 8 m and localization error
   below 7 m;
2. Tail80 joint success over the last 50 actions;
3. unsafe release transitions;
4. unsafe TRACK actions at action start and action end.

Primary descriptive continuous endpoints:

- first TRACK-action time, with no-lock episodes assigned 442 s for paired
  analysis;
- terminal localization and formation errors;
- tail joint occupancy;
- mean squared action;
- maximum localization error, maximum TRACK localization error, and
  false-confidence action count;
- maximum combined estimator/planner decision runtime.

Binary endpoints report success-rate differences, paired discordances, and an
exact two-sided McNemar test.  Continuous paired differences report mean,
median, and a deterministic 20,000-replicate bootstrap 95% interval.  The
continuous intervals are descriptive and do not replace the material-effect
gates below.

## Prespecified material-effect decisions

All success-rate differences are `full_active` minus comparator.

### Pairwise prediction-separation term

Report `MATERIAL_PAIR_TERM_EFFECT` only when:

1. campaign integrity passes;
2. `full_active` has no more unsafe transitions or unsafe TRACK actions than
   `no_pair_term`; and
3. `full_active` improves terminal success or Tail80 success by at least
   5 percentage points over `no_pair_term`.

Otherwise report `NO_MATERIAL_PAIR_TERM_EFFECT_DETECTED`.  This wording is not
an equivalence or non-inferiority claim.

### Retained planner hypotheses

Report `MATERIAL_RETAINED_HYPOTHESIS_EFFECT` only when the same integrity and
safety conditions hold and `full_active` improves terminal or Tail80 success
by at least 5 percentage points over `best_hypothesis_only`.

Otherwise report `NO_MATERIAL_RETAINED_HYPOTHESIS_EFFECT_DETECTED`.  The
secondary `no_pair_term` versus `best_hypothesis_only` contrast is reported to
separate worst-case information planning from pairwise separation, but it does
not replace the prespecified headline contrast.

### Belief-informed selection

Report `MATERIAL_INFORMED_SELECTION_EFFECT` only when the same integrity and
safety conditions hold and `full_active` improves terminal or Tail80 success
by at least 5 percentage points over `random_feasible`.

Otherwise report `NO_MATERIAL_INFORMED_SELECTION_EFFECT_DETECTED`.

Irrespective of component decisions, the full arm is operationally acceptable
only if terminal success is at least 90%, Tail80 success is at least 85%, p95
terminal localization error is below 7 m, and all unsafe counts are zero.

## Integrity gates

A campaign is valid only when:

- all four arms exist for every seed and share the same exogenous-noise tape
  hash and mission support;
- every trace contains exactly 220 actions;
- both source columns are active in every trace and every residual denominator
  equals time rows times two sources;
- the first three arms have identical actions and physical histories before
  the first planner decision;
- `full_active` and `no_pair_term` retain identical hypothesis sets at every
  paired pre-divergence decision;
- `best_hypothesis_only` never reports more than one hypothesis;
- `random_feasible` never invokes the belief planner, every selected index is
  within the saved candidate count, and the complete index stream matches the
  independently reconstructed policy RNG;
- audited release violations are zero;
- maximum combined online decision runtime is below 2 s;
- the loaded source closure is unchanged during execution;
- publication numerical settings are used;
- the development seed block was unused at freeze; and
- the reserved/final range remains untouched.

Failure is a reportable result.  It does not authorize threshold tuning, arm
replacement, seed replacement, or a new algorithmic version.

## Files

- Core: `uuv_v39_planner_component_ablation.py`
- Runner: `run_v39_planner_component_ablation.py`
- Independent audit: `audit_v39_planner_component_ablation.py`
- Tests: `tests/test_uuv_v39_planner_component_ablation.py`

