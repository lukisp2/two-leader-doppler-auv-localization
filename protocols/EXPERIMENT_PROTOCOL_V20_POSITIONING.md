# V20 engineering protocol: positioning bottleneck ablation

## Question

V20 asks whether the failed V18 reinforcement-learning result was primarily
caused by the particle filter, and whether a good position estimate leaves a
meaningful control problem for RL.  V20 contains no RL training and loads no
policy.

This is an engineering development experiment.  Seeds `50000..50999` remain
sealed and untouched.

## Paired design

Development episodes use seeds `46000..46099`.  For each seed, all arms use
the same initial scenario and immutable exogenous-noise tape:

1. `pf_pid`: legacy PF mean followed by the restored position-tracking PID;
2. `batch_pid`: causal V19 full-history batch estimate followed by the same PID;
3. `oracle_pid`: simulator-truth position followed by the same PID.  This is a
   diagnostic ceiling and is never called deployable.

All arms execute the same position-independent acquisition manoeuvre for the
first 60 actions (`0..120 s`).  The first PID action is action 61.  Pairing is
checked by requiring identical initial states, first 60 plant actions, true
trajectories through the acquisition interval, noise-tape hashes and tape
cursors.

The direct action path is forced with `v11_controller_id="pid_track"`.  The
external PID receives only an explicit allowlist: estimated follower position,
formation target derived from leader broadcasts, leader speed/course and
measured follower speed/yaw/pitch.  It does not receive the environment info
dictionary, reward, PF planner or PF uncertainty.

## Causal batch estimator

The estimator records:

- measured follower velocity at every `0.1 s` propagation substep;
- raw Doppler, leader positions and leader velocities at every available
  `1 s` measurement;
- the complete measurement prefix available at the decision boundary.

PF gate factors are replaced by ones and the estimator uses `gate_mode="raw"`.
Global V19 multi-start solves run at nominal `120, 240, 360, 440 s`; local
full-history warm starts run at the intervening 2-second action boundaries.

Because the frozen simulator accumulates floating-point time, only 119 Doppler
samples are available at the first nominal 120-second decision.  This is kept
causal.  The estimated initial position is propagated to the exact decision
time using the current accumulated dead reckoning, not merely the displacement
at the last Doppler timestamp.

The shadow PF is still executed by the frozen simulator, but `batch_pid` never
reads it when selecting an action.  Environment reward/success fields are not
used because they remain tied to the PF.

## Endpoints

Externally recomputed primary endpoints are:

- terminal joint success: true formation error `<8 m` and position-source
  localization error `<7 m`;
- joint success for all final 15 actions;
- at least 80% joint-success occupancy in the final 50 actions;
- time to a joint lock sustained through 440 s;
- mean and maximum true formation error after 120 s;
- mean localization error after 120 s;
- mean squared direct action.

The main positioning contrasts are `batch_pid - pf_pid` and
`oracle_pid - batch_pid` on the paired episodes.

## Interpretation fixed before results

- `batch ≈ oracle >> PF`: PF was a major bottleneck.  The old V18 result does
  not reject RL in a correctly observed problem, but any new RL must be trained
  from scratch on the batch-observation contract.
- `batch ≈ oracle`, with both already solving formation tracking: improved
  positioning works, but there is little mission-success headroom left for RL.
- `oracle >> batch`: online positioning/certification is still the bottleneck.
- `oracle ≈ PF` with weak control results: positioning was not the principal
  cause; the controller/action architecture is the next target.

Only if deterministic control leaves a repeatable gap after batch positioning
should V20 proceed to a new three-seed RL training campaign.  A frozen V18
policy swap is at most an out-of-distribution diagnostic because its
normalization, PF-health features, guard and reward contract all change.
