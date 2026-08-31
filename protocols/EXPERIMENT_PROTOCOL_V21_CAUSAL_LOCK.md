# V21 protocol: causal acquisition-to-tracking lock

## Scientific question

V21 tests whether the V19 full-information Doppler estimator can decide,
without simulator truth and without the legacy particle filter, when its
position estimate is safe to release to the restored tracking PID.

V21 contains no reinforcement-learning training.  Its purpose is to create
the deployable estimator/lock contract required before comparing a
deterministic information planner with a history-conditioned RL policy.

The final validation seeds `50000..50999` remain sealed.  Thresholds below
were selected from the already accessed V19/V20 development evidence.  A new
development confirmation block uses seeds `47000..47099`.

## Frozen estimator input boundary

The estimator and gate may use only:

- measured follower velocity integrated as dead reckoning;
- raw Doppler measurements;
- broadcast leader positions and velocities;
- the public initialization shell;
- their own past estimates, modes, residuals and covariance surrogates.

They may not use follower truth, localization error, reward, the particle
filter state/covariance, or any environment success field.  Truth is opened
only by the external diagnostic layer after an action has been chosen.

The estimator remains a full-history nonlinear least-squares/maximum-
likelihood estimator for the unknown initial position `p0`.  It is not called
MHE in V21.

## Global and local estimation

Estimation starts at 30 s.  Independent global multi-start searches are run
at nominal times

`30, 60, 90, 120, 180, 240, 300, 360, 420, 440 s`

and after a lost lock.  Each search uses the frozen V19 radial support and two
independently shifted Halton sweeps.  A second global search with an
independent deterministic shift confirms the selected mode.  Between global
searches the selected mode is refined on the complete causal history at every
2 s action boundary.

At a global checkpoint, the previous global solution predicts all new
measurements collected since that checkpoint.  This forward residual is not
refitted on those measurements.

## Prespecified release gate

Release is forbidden before 120 s.  At an action boundary all of the following
must hold:

1. the current local solve converged, has Hessian rank 3 and a valid local
   covariance surrogate;
2. local `r95 <= 7 m`;
3. full-history and recent-20-sample residual RMSE are each `<= 0.08 m/s`;
4. the latest global evidence is no older than 65 s;
5. both independent global searches converged with rank 3 and valid local
   covariance;
6. their best initial-position modes agree within 3 m;
7. the best global initial-position estimate changed by at most 5 m from the
   preceding global checkpoint;
8. forward-prediction RMSE is `<= 0.08 m/s` on at least 20 new scalar-time
   Doppler samples;
9. every reported mode at least 7 m from the winner is rejected by
   `delta chi2 >= 11.345` (99% reference value for three parameters);
10. the last three local initial-position estimates lie within 2 m of the
    newest estimate.

The complete release predicate must pass on three consecutive 2 s action
boundaries.  The controller then enters `TRACK` on the next action.  No
threshold is adapted after truth scoring.

## Hysteresis and fail-closed behaviour

While locked, health uses relaxed thresholds: local `r95 <= 10 m`, full and
recent residual RMSE `<= 0.10 m/s`, rank 3, valid covariance, global search
agreement within 7 m, and no unresolved separated mode below the 95%
`delta chi2 = 7.815` reference.  Three consecutive unhealthy action
boundaries return the controller to `ACQUIRE` and request an immediate global
refresh.  A scheduled global contradiction also counts as unhealthy.

After loss, the original three-consecutive-pass release rule is required.

## Paired development design

For every seed the causal arm and the fixed-120-s V20 arm receive the same
initial state and immutable exogenous-noise tape.  Both use the identical
position-independent acquisition controller until their respective release.
After release they use the same restored PID and differ only in the estimator
release rule.

Primary V21 gate endpoints are:

- episode with any false locked action (`true localization error >= 7 m`);
- localization error at first released action;
- first release time, reacquisition count and time spent locked;
- missed lock and delay relative to an offline truth-ready diagnostic;
- terminal, dwell-15 and tail-80 joint task success;
- p50/p95/max estimator and gate runtime.

The truth-ready diagnostic is the first time at which the causal batch error
is below 7 m for 15 consecutive actions.  It is used only after the run and
never enters the gate.

## Development decision

The gate passes the V21 engineering screen only if the new 100-episode block
has:

- zero episodes with a false locked action;
- at least 95/100 episodes reaching lock;
- at least 95/100 terminal joint successes;
- no source-boundary or pairing violation;
- solver runtime below the 2 s action interval in every episode.

This is not final publication evidence.  Passing V21 permits implementation
of fair estimator baselines and deterministic belief-space information MPC.
RL is justified only if that deterministic planner leaves a reproducible gap
in time-to-lock, formation excursion or control effort.

## Conditional RL hypothesis

The later RL problem is not “maximize a local FIM”.  Its observation must be
a causal belief/history representation: retained mode positions and weights,
mode separation, global and recent residuals, covariance/FIM summaries,
leader geometry, dead-reckoned motion and gate state.  Its primary objective
is time to certified lock, with penalties for formation excursion and effort.

The learned policy must be compared with the same estimator under fixed
S-turn, greedy FIM and receding-horizon multi-mode information planning.  Its
increment is scored relative to the deterministic floor across at least three
training seeds.  If actor removal does not reduce performance, no RL claim is
made.
