# Resubmission paired policy--plant--current campaign --- frozen protocol

Date frozen: 2026-08-31, before opening qualification scenarios
49900--49999.

## Scientific purpose

Reviewer 1 questioned whether the reported benefit of active acquisition
survives actuator response, command latency, and environmental disturbances.
This campaign answers that question with a paired 2 x 2 x 2
factorial experiment. It compares the submitted belief-driven acquisition
policy with the submitted fixed S-turn baseline under both the submitted
rate-limited kinematics and a deliberately modest low-order execution model,
with and without a horizontally varying current.

The experiment is a robustness qualification, not a new controller and not a
vehicle-specific hydrodynamic validation. The estimator, retained hypotheses,
belief-information planner, evidence gate, post-lock PID, action bank,
measurement model, thresholds, and solve schedule are frozen. No algorithm
component receives the plant time constants or the current as an additional
input.

## Data boundary and pairing

- Engineering smoke uses only the previously opened seeds 49566 and 49591.
- Qualification uses the previously sealed seeds 49900--49999: exactly 100
  scenarios and all eight arms for every scenario (800 runs).
- The final holdout 50000--50999 remains sealed. The runner must reject it.
- Within a seed, all eight arms use exactly the same initial state, leader
  histories, sensor-noise tape, and stochastic-current tape. Policy, execution
  model, and the current on/off switch are the only factorial changes.
- The stochastic-current generator uses a dedicated, documented random-number
  namespace. It cannot consume from or alter the sensor-noise stream.
- Arms are run sequentially. Results are checkpointed atomically at scenario
  boundaries and carry source, protocol, configuration, and tape hashes.

## Frozen 2 x 2 x 2 factors

### Policy

1. `fixed_s_turn`: the previously specified open-loop S-turn acquisition
   maneuver followed by the same evidence gate and post-lock PID.
2. `belief_active`: the frozen multi-hypothesis belief-information acquisition
   policy followed by the same evidence gate and post-lock PID.

The comparison therefore isolates acquisition motion. The estimator, gate,
and tracker are identical between policies.

### Command execution

1. `kinematic`: the submitted rate-limited kinematic follower.
2. `low_order_dynamic`: one action interval (2 s) of command delay followed by
   first-order execution states for commanded surge acceleration `a_c`, yaw
   rate `r_c`, and pitch rate `q_c`:

   `tau_a da/dt + a = a_c`, `tau_r dr/dt + r = r_c`, and
   `tau_q dq/dt + q = q_c`.

   The frozen constants are `tau_a = 5 s`, `tau_r = 2 s`, and
   `tau_q = 3 s`. The executed states are integrated at the
   unchanged 0.1-s plant step using the exact stable first-order update. The
   existing speed, pitch, acceleration, and angular-rate limits remain in
   force.

This low-order response captures finite actuation bandwidth and transport/
execution latency without pretending to be a calibrated six-degree-of-freedom
model. It intentionally adds no condition-specific feedforward or gain change.

### Horizontal current

1. `none`: zero water-current velocity.
2. `bottom_track_visible`: a horizontal current consisting of a seed-fixed
   steady vector of magnitude 0.30 m/s plus a stationary
   two-dimensional Gauss--Markov component. Its two components are independent,
   each with marginal standard deviation 0.05 m/s and correlation time 120 s.
   At the 0.1-s integration step,

   \[
   \boldsymbol\xi_{n+1}=\alpha\boldsymbol\xi_n+
   \sigma_c\sqrt{1-\alpha^2}\,\boldsymbol\varepsilon_n,
   \qquad
   \alpha=\exp(-\Delta t/120),
   \]

   where \(\boldsymbol\varepsilon_n\sim\mathcal N(\boldsymbol0,I_2)\) and
   \(\boldsymbol\xi_0\sim\mathcal N(\boldsymbol0,\sigma_c^2I_2)\). The steady
   direction and the complete innovation tape are deterministic functions of
   the scenario seed.

Follower position and Doppler truth use velocity over ground (body-relative
velocity plus current). The simulated bottom-track DVL also measures velocity
over ground, with the unchanged measurement noise, and dead reckoning
integrates that measured ground velocity. Thus this factor tests whether the
motion planner and tracker tolerate drift of the executed trajectory; it does
not introduce an undisclosed navigation-frame bias. The current is not passed
separately to the planner, gate, or PID.

## Frozen common numerical settings

- 440-s fixed horizon, 220 decisions, 2-s action interval;
- 1-s two-link Doppler measurements and 0.1-s plant integration;
- full causal history, retained competing modes, two shifted global sweeps,
  4096 broad candidates, and 48 local starts;
- frozen raw evidence gate, support convention, belief-information action
  score, action bank, global-refresh schedule, and post-lock PID;
- no training, retuning, threshold changes, additional starts, policy-specific
  gains, or condition-specific exception paths.

## Endpoints and paired analysis

For every arm, report terminal joint success, Tail80, Dwell15, ever-lock,
first-TRACK time, terminal localization error, terminal formation error,
truth-invalid TRACK starts and ends, action effort, maximum online decision
runtime, requested-to-executed command mismatch, and command-response lag.
Continuous errors are reported by median, p95, and maximum; binary endpoints
are reported as counts out of 100.

The primary comparisons are paired `belief_active` minus `fixed_s_turn`
contrasts within each of the four execution--current cells. Binary discordance
is reported directly and with exact McNemar tests. Continuous paired
differences receive deterministic percentile-bootstrap 95% intervals from
50,000 PCG64 resamples. Secondary pre-specified contrasts quantify, separately
for each policy, the dynamic-execution effect, current effect, and their
interaction. A difference-in-differences reports whether either nonideality
materially changes the active-policy advantage. Cellwise results remain
visible; an aggregate mean cannot conceal a failing condition.

The bootstrap seed is 40049900 plus a stable integer index of the endpoint and
contrast. Exact counts, effect sizes, and intervals are primary; isolated
nominal p-values are not used to select a favorable narrative.

## Pre-specified interpretation

1. **Nominal replication.** The `belief_active / kinematic / none` cell must
   attain at least 95/100 terminal successes, 90/100 Tail80 successes, 95/100
   ever-lock episodes, and zero truth-invalid TRACK events. Failure invalidates
   the robustness claim and is reported rather than tuned away.
2. **Absolute robustness screen.** A non-nominal active-policy cell is labelled
   `ROBUST_WITHIN_SCREEN` only with at least 90/100 terminal successes, 85/100
   Tail80 successes, and zero truth-invalid TRACK starts and ends.
3. **Policy comparison.** Active acquisition is described as outperforming the
   fixed maneuver in a cell only when the paired effect has the favorable sign
   for both terminal success and Tail80, with the complete discordant counts
   reported. Statistical uncertainty must be stated even when a point estimate
   is favorable.
4. **Scope.** Passing this screen supports robustness to the explicitly tested
   lag, delay, and bottom-track-visible current. It does not establish robustness
   to unmodelled DVL bias, six-degree-of-freedom hydrodynamics, saturation beyond
   the stated limits, acoustic multipath, or sea-trial conditions.

No result from seeds 49900--49999 may trigger a scientific-constant change.
Only implementation defects that violate this written protocol may be fixed;
the defect, patch, affected outputs, and rerun must be logged. A scientifically
motivated redesign would require a new version and new unopened seeds and
cannot be presented as this frozen qualification.

## Validity checks and execution

The campaign is valid only if all 800 runs complete at 220 decisions, every
seed has all eight unique arms, all paired hashes close, required values are
finite, the two kinematic/no-current smoke traces match the unmodified engine,
and the final holdout remains untouched.

Execution order:

1. freeze this protocol, source, tests, arm order, and all hashes;
2. run unit tests and the full eight-arm smoke on seeds 49566 and 49591;
3. correct protocol-violating implementation defects only, then refreeze;
4. launch the 800 qualification runs sequentially in the background with a
   persistent log, PID file, atomic checkpoints, and resumability;
5. do not supervise continuously and do not launch another heavy campaign in
   parallel.
