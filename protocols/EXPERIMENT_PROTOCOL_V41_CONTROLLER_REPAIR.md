# Frozen protocol: delay-aware post-TRACK controller repair

## Scientific question

The experiment tests one narrow causal hypothesis: the loss of formation-tracking
success under the frozen low-order plant is caused primarily by using a
delay-unaware post-TRACK controller. It does **not** modify or retune the
full-history estimator, active-acquisition planner, audited ACQUIRE-to-TRACK
gate, formation definition, sensor model, dynamic plant, current process, or
success criteria.

The treatment is a deterministic delay-aware formation tracker. The reference
is the previously published stateless proportional tracker. Both receive the
same causal online inputs. The treatment has no access to simulator truth.

## Frozen paired design

Each episode is run sequentially through four arms:

| Current condition | Post-TRACK controller |
|---|---|
| no current | reference proportional tracker |
| no current | delay-aware tracker |
| visible horizontal current | reference proportional tracker |
| visible horizontal current | delay-aware tracker |

Every arm uses:

- both leader broadcasts;
- the deterministic belief-based active-acquisition policy before TRACK;
- the causal full-history multimodal estimator;
- the audited evidence-qualified gate;
- the low-order dynamic plant with one action interval of command delay;
- surge, yaw-rate, and pitch-rate time constants of 5 s, 2 s, and 3 s;
- the same fixed 440 s horizon and 2 s action interval;
- the same sensor-noise tape and exogenous current tape within a seed.

The arm order pairs the reference and treatment within each current condition
before advancing to the next current condition. Seeds are processed one at a
time. No arms or seeds run concurrently.

## Controller freeze

The delay-aware controller is frozen in
`delay_aware_formation_tracker.py`. Its configuration is serialized by
`uuv_v41_controller_repair.condition_contract()` and copied into the immutable
campaign contract. It uses the known command delay and the frozen first-order
plant constants, propagates the delayed command queue, advances the desired
formation with the mean leader velocity, and applies a causal reference
governor, bumpless ACQUIRE-to-TRACK transfer, and per-action command slew
limits.

The controller is initialized with the last command requested during ACQUIRE.
It is reset after a loss of TRACK. It receives only the estimated follower
position, desired formation position, leader velocities, measured follower
speed/yaw/pitch, time, and its own requested-command history. Truth is used only
after an action has completed for scoring.

Before any qualification seed was opened, an integration smoke test showed
that an initially conservative outer loop suppressed command saturation but
also removed the reference controller's ability to catch the moving formation
after a late lock. Version 1.1 therefore preserves the reference controller's
published formation-error gains (0.055 along-track, 0.065 cross-track, and
0.055 vertical per second) and correction authority. Its horizontal vector cap
is $\sqrt{2}\,1.65$ m/s, the Euclidean envelope of the reference controller's
independent 1.65 m/s along/cross caps, and its vertical cap is 1.20 m/s. The
Smith predictor, inner-loop pole placement, reference governor, six-action
bumpless transfer, and command slew limits were not changed. This amendment
was fixed using only the previously opened smoke seeds 49566 and 49591; the
qualification range 51000--51099 remained unopened.

## Seed policy

- Development and integration smoke seeds: 49566 and 49591. These seeds were
  already open before this protocol.
- Fresh qualification seeds: 51000--51099, exactly 100 episodes.
- Sealed final range: 50000--50999. This range remains unopened and is rejected
  by the runner.

Before a qualification run starts, the runner scans existing experiment
artifacts for references to the selected qualification seeds. Any finding
aborts the campaign. Once any qualification result is generated, controller
retuning is prohibited. A negative result is retained as the result of this
frozen test.

## Estimator settings

Publication settings are:

- 4096 coarse candidates;
- two coarse sweeps;
- 48 local starts;
- raw gate mode;
- uniform-radius candidate distribution.

The command-line estimator overrides exist only for smoke and diagnostic
execution. A qualification result is publication-eligible only when all three
publication values are used.

## Primary endpoints

For each current condition, the primary endpoints are:

1. terminal joint success, requiring terminal formation error below 8 m and
   terminal localization error below 7 m;
2. Tail80 joint success, requiring the joint condition during at least 80% of
   the frozen tail window.

The primary paired effects are treatment-minus-reference risk differences on
the same 100 seeds. McNemar's exact two-sided test is reported descriptively.

## Safety and mechanism endpoints

The campaign also reports:

- evidence-qualified TRACK transition counts and all truth-audited unsafe
  transition/start/end counts;
- terminal localization and formation error distributions;
- first TRACK time and lock rate;
- mean squared normalized action;
- requested-versus-delivered action RMS;
- the fraction of TRACK actions for which any normalized command channel has
  magnitude at least 0.98;
- channel-specific saturation fractions;
- post-TRACK action total variation over consecutive TRACK actions;
- post-TRACK action-curvature RMS, defined from the second finite difference of
  normalized commands over three consecutive TRACK actions;
- maximum combined estimator/planner decision time.

The saturation fraction measures command clipping pressure. The action-
curvature RMS is the frozen oscillation/chatter endpoint. These definitions are
computed from saved traces, not from controller-internal diagnostics.

## Pre-TRACK identity check

Within each seed and current condition, reference and treatment must have:

- the same initial truth state;
- identical sensor-noise and current-tape hashes;
- the same first TRACK action index;
- bitwise-identical deterministic traces before the first TRACK action,
  excluding wall-clock runtime fields.

Any violation invalidates and stops the campaign. Thus a measured treatment
effect cannot be attributed to a changed estimator, planner, gate, noise draw,
or pre-TRACK maneuver.

For smoke execution only, each V41 reference arm is also run directly through
the frozen V40 implementation. Their common trace fields must be bitwise equal;
only nondeterministic planner wall-clock timing is excluded.

## A priori acceptance criteria

The repair is accepted only if the complete 100-seed qualification is valid and
**every** condition below holds separately under no current and visible current:

1. delay-aware terminal joint success is at least 90%;
2. delay-aware Tail80 joint success is at least 85%;
3. the paired terminal-success gain is at least +20 percentage points;
4. the paired Tail80-success gain is at least +40 percentage points;
5. there are zero truth-audited unsafe TRACK transitions, starts, or ends in
   either arm;
6. treatment terminal-localization median and p95 are each no more than 0.50 m
   above the paired reference-arm aggregate;
7. at least 90% of pairs provide post-TRACK saturation and oscillation metrics;
8. the paired mean treatment-minus-reference post-TRACK saturation fraction is
   strictly below zero;
9. the paired mean treatment-minus-reference action-curvature RMS is strictly
   below zero.

These thresholds are joint gates, not quantities to optimize after seeing the
qualification data. Failure of any condition is reported without a new
controller version or threshold change.

## Statistical reporting

The runner stores episode-level values and reports:

- count, mean, median, p90, p95, and maximum for continuous endpoints;
- counts and rates for binary endpoints;
- paired treatment-minus-reference differences;
- deterministic 20,000-replicate paired bootstrap intervals for mean
  continuous differences;
- McNemar exact tests for paired binary endpoints.

Current conditions are reported separately. They are not pooled into a single
headline average.

## Reproducibility and interruption handling

The runner writes one atomic JSON summary and one atomic compressed NPZ trace
after each arm. It additionally maintains:

- `control/campaign_contract.json`, an immutable contract containing source and
  metadata hashes;
- `control/source_snapshot/`, a copy of the complete local source closure;
- `control/progress.json`, including completed runs, elapsed time, ETA, and
  estimated UTC finish time;
- `control/campaign.log`, a flushed event log;
- `control/campaign.lock` and `control/runner.pid`, preventing concurrent
  execution;
- `control/crash_state.json` after an exception or interruption.

Resume mode loads and validates both members of every completed JSON/NPZ pair.
It refuses a changed contract or changed source closure. A missing member of an
artifact pair is treated as corruption and stops the run.

The terminal displays one `tqdm` progress bar for the 400 qualification arms,
with the current seed, controller/current arm, elapsed time, rate, and updated
ETA. The runner itself does not require interactive supervision.

## Frozen commands

Smoke test:

```bash
python run_v41_controller_repair.py --smoke
```

Full qualification:

```bash
python run_v41_controller_repair.py
```

Resume after interruption:

```bash
python run_v41_controller_repair.py --resume
```

Partial slices are supported with `--episode-start` and `--episodes`, but a
partial slice is never eligible for the acceptance decision.
