# V22 protocol: belief-conditioned active Doppler acquisition

## Question

V22 asks whether the long lock times of a fixed, position-independent S-turn
can be reduced by deterministic control that uses the causal measurement
history and V21 belief, while keeping the identical truth-free lock and the
identical post-lock PID.

This is the required baseline before any new reinforcement-learning run.  V22
contains no learned policy and no training.

Development confirmation uses seeds `48000..48099`.  Final seeds
`50000..50999` remain sealed.

## Paired arms

1. `s_turn_causal`: V21 fixed S-turn until the V21 gate releases, then PID;
2. `belief_fim_causal`: the active planner below until the same V21 gate
   releases, then the same PID.

Both arms use the same initialization and immutable exogenous-noise tape.  No
pairing of plant trajectories is expected after 30 s because active sensing
is the treatment.

## Causal belief used by the planner

Before the first global solve at 30 s, the active arm uses the same S-turn.
Afterwards its belief support contains:

- the current full-information best mode;
- retained modes from both independent global searches;
- covariance-axis support points around the current mode;
- current dead-reckoned displacement;
- the complete raw Doppler/leader history.

Support points closer than 1 m are merged.  At most 16 hypotheses are used.
Simulator truth, PF state, reward and external localization error are absent.

## Receding-horizon action search

At every 2 s action boundary before lock, the planner evaluates 45 feasible
increment commands plus the fixed S-turn command.  The grid is

- speed increment: `{-1, 0, +1}`;
- yaw increment: `{-1, -0.5, 0, +0.5, +1}`;
- pitch increment: `{-1, 0, +1}`.

The simulator's measured onboard speed/yaw/pitch and actuator limits convert
each command to a feasible candidate velocity.  That velocity is propagated
for a 30 s look-ahead with 1 s Doppler samples and broadcast leader motion.

For every belief hypothesis, the planner adds the predicted Doppler Jacobians
to the full-history information matrix and computes its local 95% worst-axis
radius.  For every separated pair it also predicts the accumulated Doppler
separation in noise-normalized chi-square units.

The frozen utility is

`worst_radius_reduction + 0.25*log1p(min_pair_chi2)
 - 0.05*action_energy - 0.05*change_from_previous_action`.

If only one hypothesis is present, the pair term is zero.  Ties are resolved
by the stable candidate order.  The selected normalized command is applied
directly; the optimization never reads truth.

## Endpoints and decision

Safety remains primary:

- false-lock episodes/actions;
- localization error at first release;
- maximum estimator/planner runtime versus the 2 s deadline.

Efficacy endpoints are first release time, truth-ready time, p50/p95 lock
delay, terminal/dwell/tail joint success, acquisition action energy and
formation excursion.

V22 supports a material active-control effect if, with zero false locks, the
active arm improves median time to lock by at least 30 s or 20% and does not
reduce terminal success by more than two percentage points.  This development
criterion is not final publication inference.

Only after this deterministic baseline is measured may an RL policy be
trained.  A future RL policy must consume the same belief/history contract,
use the same V21 gate and PID after lock, and demonstrate an increment over
this deterministic planner across at least three independent training seeds.
