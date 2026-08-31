# V19 observability and estimator benchmark protocol

Date frozen: 2026-07-14  
Status: development diagnostic; no RL training; final evaluation forbidden

## Purpose

V19 is not a new learned controller.  It is a fail-closed diagnostic that
separates structural Doppler observability, estimator support loss, and
closed-loop control.  The benchmark asks two questions before any further
controller training is allowed:

1. Is the initial follower position identifiable from the Doppler history of
   an already recorded trajectory when the assumed motion model is exact?
2. If it is identifiable, can a causal, multi-start batch estimator recover
   it from the recorded noisy measurements and dead-reckoning history?

The primary replay trajectory is the model-independent
`deterministic_greedy_grid` controller from the completed V18.1 campaign.  It
is used because it is bit-identical across the three V18 model-seed
evaluations and does not depend on a learned policy.

## Data boundary

- Development scenarios: `45000..45099`, already accessed by V18.1.
- Primary source evaluation: model-seed directory `28001`; only the
  model-independent deterministic-greedy traces are consumed.
- Sealed final scenarios: `50000..50999`.
- V19 code must reject every attempt to load or generate a sealed-final seed.
- Results on `45000..45099` are diagnostic and may guide implementation.  They
  are not a fresh validation set for a future positive method.
- A new development/validation block may be selected only after the estimator
  and all gates below have been frozen.  The final block remains sealed until
  that future protocol passes.

No SAC model is loaded and no training process is started by this benchmark.

## Exact replay contract

For each scenario, V19 loads the saved exogenous-noise tape and the saved
applied three-channel plant actions.  It resets the frozen V18 environment at
the recorded episode seed, feeds the applied actions through the direct plant
path, and captures every one-second particle-filter measurement call.
Before replay it reconstructs the complete environment configuration from the
source campaign metadata and verifies the frozen source hashes.  The only
overrides are the direct applied-action path and disabled generation of a new
noise tape.  The tape content digest, trace identity and final tape cursors
(`4400` dead-reckoning substeps, `440` Doppler samples) must agree with the
frozen artifacts.

The online estimator input consists only of:

- measurement time;
- dead-reckoned follower displacement from the unknown initial position;
- measured follower velocity;
- the two broadcast leader positions and velocities;
- the two measured Doppler range rates;
- the historical PF soft-gate factors, retained only as an ablation.

The known centre of the initialization shell is reconstructed from the first
online leader broadcast as `mean(pL(t1) - vL(t1)*t1)`.  Its radii are the
frozen initialization contract, `120..350 m`.  The search-design seed is a
fixed solver constant determined only by problem type and prefix; it never
depends on the episode seed.

Simulator truth is stored in a separate diagnostic object and is never passed
to the noisy batch estimator.  Exact replay is accepted only if all 220 saved
action endpoints reproduce the recorded true follower position, PF mean, and
PF covariance within the frozen numerical tolerance.  The captured time grids
must contain exactly 220 action endpoints at `2..440 s` and 440 measurements
at `1..440 s`.

## Estimation problems

### Structural noiseless problem

The true follower displacement and velocity are used to synthesize noiseless
Doppler measurements under the exact measurement model.  The estimator is
given only this synthetic history and the initial support shell.  This is an
oracle diagnostic of global identifiability, not a deployable estimator.

### Recorded noisy problem

The estimator uses the captured measured velocity, integrated
dead-reckoning displacement, and recorded noisy Doppler measurements.  The
only optimized state in the first V19 benchmark is the initial follower
position `p0 in R^3`.  The current position at a prefix endpoint is
`p0 + dead_reckoned_displacement`.

Both problems are solved at the causal prefixes `30, 60, 120, 240, 440 s`.
The default solver uses deterministic shell coverage, coarse likelihood
ranking, damped Gauss--Newton refinement, and spatial clustering of converged
solutions.  Its frozen default is two independently shifted 4096-point Halton
shell sweeps, 48 spatially separated local starts, 5 m start separation, 1 m
mode clustering and a 7 m competing-mode separation.  Several separated
minima are retained.  Absence of a found alternative is only a numerical
coverage screen, never proof of global uniqueness.

The Hessian covariance is only a local surrogate.  It is invalidated when the
Hessian has rank below three, condition number above `1e10`, or the optimum is
on the support-shell boundary.  An invalid surrogate has no finite confidence
radius and can never support a confidence claim.

## Required outputs

For every scenario and prefix, the benchmark records:

- best initial-position and current-position error, for diagnostics only;
- best residual RMSE;
- number and locations of separated modes;
- distance and likelihood/SSE gap to the best separated alternative;
- local Hessian eigenvalues and local covariance surrogate;
- full-prefix residuals, explicitly not described as held-out validation;
- current PF endpoint error and covariance diagnostics from the frozen trace;
- initial PF nearest-particle distance and mass/count within 7 m and 20 m;
- leader-course difference, speed difference, and leader separation at the
  prefix endpoint.

All configuration, source hashes, input hashes, commands, and interpreter
details are written to a campaign manifest.

The estimator stage reloads only `online_inputs.npz`, derives the public shell
centre, uses the fixed shell radii and solver seed, and writes unscored modes.
It does not read capture metadata, episode seed or truth labels.  Truth labels
are opened only by the later scoring stage.  No localization certificate is
defined or tested in this first benchmark; certificate calibration is a
separate, prespecified validation problem.

## Development decision gates

These gates decide the next research step; they do not authorize final
evaluation.

### Structural observability

- `GO/SCREEN`: the best noiseless solution is within 1 m of truth at 440 s in
  all 100 scenarios and the frozen multi-sweep search finds no alternative at
  least 7 m away with residual RMSE within `1e-6 m/s` of the best solution.
- `STOP/GEOMETRY`: an alternative at least 7 m away remains measurement-
  equivalent.  The sensing geometry, prescribed maneuver, or sensor set must
  change before estimator work continues.

This is a finite numerical screen.  Any later identifiability claim requires a
separate coverage-sensitivity run with more candidates and additional shifts.

### Estimator feasibility

- `GO/SCREEN`: the noisy estimate is within 7 m at 240 s in at least 95% of
  scenarios, is within 7 m at both 240 and 440 s in at least 95%, and is within
  7 m at 440 s in at least 99%.  The solver-only nearest-rank p99 runtime at
  440 s must be below the two-second action interval on the target machine.
- `STOP/ESTIMATOR`: structural observability passes but the noisy estimator
  gate fails.  Replace or extend the estimator; do not train a controller.

Runtime excludes replay, archive I/O and truth scoring.  The first completed
solver call at 30 s is treated as warm-up.  Because prefixes are evaluated in
order, every 440 s runtime used by the gate is post-warm-up; none of those 100
values is discarded.  Nearest-rank p99 is the sorted value at rank
`ceil(0.99*n)`.  The 120/240/440 s checkpoint errors also provide a
time-to-lock diagnostic: the earliest checkpoint after which all later
checkpoints remain below 7 m.

The 95%/99% targets cannot be established from only 100 scenarios as safety
claims.  They are engineering screens.  A later frozen validation requires at
least 1000 independent scenarios, prespecified paired intervals and a separately
frozen confidence-certificate definition.

### Controller opportunity

This gate is deferred until the estimator gates pass.  A 2x2 comparison will
cross the verified estimator/oracle state with current greedy/long-horizon
information MPC.  RL is considered only if the oracle/MPC comparison exposes
a material control opportunity that deterministic methods do not fill.

## Interpretation tree

1. Noiseless multi-start batch estimation fails: change sensing or geometry.
2. Noiseless succeeds but noisy batch estimation fails: improve the
   estimator/model and acquisition maneuver.
3. Estimator succeeds and deterministic MPC is sufficient: omit RL.
4. Estimator and MPC succeed and a learned approximation is needed for
   runtime: train only after a separate protocol is frozen.

## Non-claims

V19 development results do not establish field safety, do not validate the
existing article's RL-superiority claim, and do not authorize access to the
sealed final scenarios.  FIM, PF covariance, and ESS remain diagnostic
quantities; none is accepted alone as a global localization certificate.
