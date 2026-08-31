# V27 publication estimator baselines — frozen development protocol

Date frozen: 2026-07-15  
Status: development-only baseline and component-ablation campaign; no RL training;
reserved and final evaluation ranges remain closed

## Scientific purpose

V26 retained the deployable one-step greedy belief-FIM planner and found no
material teacher headroom for RL.  V27 therefore stops planner tuning and asks
the reviewer-facing question that must be answered before robustness or a final
holdout:

> Does the causal full-history global/local estimator remain preferable after
> comparison with a corrected particle implementation, a Gaussian recursive
> estimator, local nonlinear least squares, and explicit history/search
> ablations on exactly the same Doppler records?

This stage evaluates estimators only.  Every arm receives the same already
recorded online history.  Controller trajectories, measurements and truth are
not regenerated per arm.  Consequently, estimator effects cannot be confused
with different acquisition actions.

## Data and seed boundary

- Development source: the frozen V19 replay archive for scenarios
  `45000..45099`.  These scenarios were already used for development and are
  not fresh publication validation.
- Smoke episodes are fixed to episode indices `0` and `73`; smoke is a
  mechanics and integrity check only.
- Seeds `49900..49999` remain reserved for a later, separately frozen
  qualification campaign.
- Seeds `50000..50999` remain the sealed final holdout.
- V27 must reject every source or requested seed in `49900..50999`.
- No V27 result authorizes opening either closed range.

For each source episode, `online_inputs.npz` is loaded through the V19 strict
allowlist.  Simulator truth stays in the separate `truth_labels.npz` archive
and is opened only after all unscored estimator outputs for that episode have
been written.  The estimator API receives no truth object, episode seed,
historical truth error or future measurement.

The public initialization support is reconstructed exactly as in V19: the
initial leader centroid derived from the first broadcast and the shell radius
`120..350 m` stored in the capture metadata.  All algorithms use raw Doppler
measurements with nominal standard deviation `0.05 m/s`.

## Common causal checkpoints

The full campaign evaluates all arms at `30, 60, 120, 240, 440 s`.  A prefix
contains only measurements at or before its endpoint.  The reported current
position is evaluated against truth at the same endpoint.  Runtime is
estimator-only wall-clock time measured with `perf_counter`; file I/O and truth
scoring are excluded.

## Frozen estimator arms

### 1. `legacy_pf`

The historical V18.1 PF mean and covariance are read from the frozen source
trace at the exact checkpoint.  Only `t_s`, `pF_hat_{x,y,z}` and the six PF
covariance entries are admitted by the loader.  The source trace hash must
match V19 capture metadata.  This arm is historical and has no comparable
runtime measurement.

### 2. `pf_lw_4096`

A corrected static-parameter particle filter over initial position uses 4096
deterministic uniform-radius shell particles.  It applies sequential Gaussian
likelihood updates, systematic resampling below ESS `0.5 N`, and a Liu-West
kernel with `h=0.12`, `a=sqrt(1-h^2)`.  Resampling randomness uses a fixed
algorithm-design seed independent of episode identity.  The estimate is the
weighted particle mean and covariance.

### 3. `pf_lw_16384`

Identical to `pf_lw_4096`, except for 16384 particles.  The pair tests particle
scaling without changing the prior or update law.

### 4. `ekf_static`

A static-state EKF estimates initial position.  Its prior mean is the public
shell centre and its isotropic covariance is the exact second moment of the
uniform-radius shell, `E[r^2]/3 I`.  It uses the analytic Doppler Jacobian,
`R=0.05^2 I`, zero process noise and the Joseph covariance update.  It is not
projected toward truth or onto a selected shell direction.

### 5. `local_nls6`

Six damped Gauss-Newton refinements start at the positive and negative
Cartesian axes at the shell midpoint radius.  The lowest residual solution is
reported.  There is no coarse global search and no outcome-dependent restart.

### 6. `coarse_only_8192`

The exact two shifted 4096-point Halton shell sweeps of the primary arm are
scored, but no local refinement is performed.  The minimum-SSE particle is
reported.  This isolates the value of local refinement.

### 7. `global_window60`

The complete primary global/local algorithm is run on only the last 60 seconds
of the causal prefix.  Dead-reckoned displacement remains referenced to the
unknown initial position, so this changes information history but not state
parameterization.  Search coverage and local refinement are identical to the
primary arm.

### 8. `global_full`

The frozen selected estimator uses the complete causal prefix, two shifted
4096-point uniform-radius Halton sweeps, 48 spatially separated local starts,
damped Gauss-Newton refinement, one-metre mode clustering and up to 12 retained
modes.  Search seeds are fixed by arm and checkpoint, never by scenario or
outcome.

## Smoke contract

Smoke runs the same code paths on episodes `0` and `73` at `30, 120, 440 s`.
To keep mechanics fast, it uses 512 shell candidates, one sweep and eight local
starts for global arms, and 512/2048 particles for the two PF arms.  These
reduced numerical settings are serialized in the smoke contract and their
efficacy is never pooled with the full campaign.

Smoke passes only if:

- all eight arms finish at all prescribed episode-prefix cells;
- outputs are finite where the arm contract requires them;
- the online-input and source-trace hashes match the V19 metadata;
- unscored files contain no truth/error/success fields;
- a repeated deterministic cell is bitwise identical apart from runtime;
- independent audit recomputes every scored error; and
- no reserved/final source or artifact exists.

A smoke failure permits repair of a code defect.  It does not permit changing
the full scientific arms after their outcomes are observed.

## Required metrics

For every arm, scenario and checkpoint, report:

- endpoint and initial-position error (initial error is omitted for the direct
  historical PF arm when it cannot be reconstructed faithfully);
- success at `<7 m` and at `<=7 m`;
- residual RMSE where a p0 representation exists;
- nominal local/PF/EKF 95% radius when finite;
- empirical coverage of that nominal radius;
- runtime, particle count, minimum ESS/resampling count or mode/search counts
  as applicable; and
- exact input, trace, source and contract hashes.

Aggregate tables include success counts, mean/median/p95/max error,
mean/median/p95/max runtime, coverage, radius quantiles and paired per-scenario
differences against `global_full`.  Intervals are percentile bootstrap 95%
intervals from 50,000 paired resamples using PCG64 seed `27045000`.

## Development decision

The V27 primary estimator is eligible to proceed to frozen closed-loop
component tests and stress design only if:

1. campaign and independent-audit integrity both pass;
2. `global_full` completes all 500 cells without non-finite estimates;
3. `global_full` is within `7 m` in at least 95/100 scenarios at 120 s, at
   least 95/100 jointly at 240 and 440 s, and at least 99/100 at 440 s;
4. nearest-rank p99 `global_full` runtime at 440 s is below 2 s; and
5. its 440-s success rate is not lower than the best corrected non-truth
   comparator by more than one percentage point.

Passing gives `PROCEED_TO_CLOSED_LOOP_STRESS_DESIGN`, not a publication claim.
Failure gives `REVISE_ESTIMATOR_BEFORE_STRESS`.  Individual ablation effects
are reported honestly; a component is claimed only when its prespecified
paired interval excludes zero in the favourable direction and its practical
effect is non-trivial.  No component is retained merely because it was part of
the original design.

## Sequence after V27

If the development gate passes:

1. freeze the simplest supported estimator configuration;
2. run closed-loop acquisition baselines and gate-component ablations through
   the common V24/V26 execution path;
3. freeze one-at-a-time stress families for Doppler bias/scale, coloured noise,
   dropout, broadcast delay/error, dead-reckoning bias/scale and leader
   geometry/course changes;
4. calibrate reported uncertainty on data independent of parameter tuning;
5. write and freeze the complete final analysis plan; and only then
6. execute the sealed `50000..50999` holdout once.

RL remains out of scope unless a later stress-specific offline teacher first
shows material headroom that the deployable deterministic planner cannot
realize.
