# V34 arrival-cost MHE-60 publication baseline — frozen protocol

Date frozen: 2026-07-16  
Status: append-only development baseline; frozen before any V34 outcome was
observed; no RL training; V27 remains unchanged; reserved and final seed ranges
remain closed

## Scientific question

V27 showed that a global/local solve on only the last 60 seconds deteriorates
late in the episode.  That arm discards old factors and is therefore a
truncated-history nonlinear least-squares ablation, not a conventional
moving-horizon estimator.  V34 asks the stronger reviewer-facing question:

> Can a standard single-hypothesis, 60-s moving-horizon estimator preserve the
> useful old information through a Gaussian arrival cost and match the frozen
> full-history estimator on exactly the same causal Doppler records?

The arm is named `mhe60_arrival_fej`.  It is also the reduced form of a
fixed-lag factor graph.  Because the frozen V27 model treats dead-reckoned
displacement as known, all follower positions satisfy

`p_F(t_k) = p_F,0 + Delta_p_DR(t_k)`.

Eliminating these equality-constrained trajectory nodes leaves only the static
three-dimensional state `p_F,0`.  V34 must not introduce process noise merely
to make the graph look higher-dimensional.

## Data, truth boundary, and closed ranges

- Source histories are the already opened V19 development records for
  scenarios `45000..45099`.
- Smoke episodes are fixed to indices `0` and `73`.
- The full V34 campaign uses all 100 development episodes.
- Checkpoints are fixed to `30, 60, 120, 240, 440 s` in smoke and full runs.
- Seeds `49900..49999` remain reserved and untouched.
- Seeds `50000..50999` remain the sealed final holdout and untouched.
- The runner and independent auditor must reject every source seed in
  `49900..50999`.
- The estimator receives `online_inputs.npz`, the public shell bounds and
  algorithm constants only.  It does not receive truth, episode seed, past
  truth error, success labels or future measurements.
- All five unscored checkpoint outputs for an episode must be persisted before
  that episode's `truth_labels.npz` is opened.

V34 reuses the exact V27 raw Doppler model, nominal standard deviation
`0.05 m/s`, initial leader centroid inferred from the first broadcast, and
hard initialization support `120..350 m` recorded in capture metadata.

## Frozen estimator

### State and objective

At update `n`, with a horizon of exactly 60 one-hertz samples, V34 minimizes

```text
0.5 * (x' Lambda_a x - 2 xi_a' x + kappa_a)
+ 0.5 * sum_{k=max(1,n-59)}^n ||z_k - h_k(x)||^2
```

subject to the same hard shell support as V27.  A common factor of
`1 / 0.05^2` is omitted from the optimization because it does not change the
minimizer.  The window contains `n-59,...,n`; the sample at `n-60` is not
included.

The arrival factor is initialized to zero.  When sample `j=n-60` leaves the
window, its two-link residual `r_j` and analytic Jacobian `J_j` are evaluated
at the previous estimate `x_bar`.  With

`d_j = r_j(x_bar) - J_j x_bar`,

the fixed-first-estimate-Jacobian update is

```text
Lambda_a <- Lambda_a + J_j' J_j
xi_a     <- xi_a - J_j' d_j
kappa_a  <- kappa_a + d_j' d_j.
```

Every measurement therefore appears exactly once: either as an exact nonlinear
factor inside the active window or as one linearized contribution to the
arrival cost.  Old factors are never relinearized and never read by the
optimizer after marginalization.

### Initialization and recurrence

- At `30 s`, use the exact full V27 global initializer: two shifted 4096-point
  uniform-radius Halton sweeps, 48 spatially separated starts, damped
  Gauss--Newton refinement, the same fixed design seed `27001 + 30`, and the
  same shell projection.  Only the lowest-residual mode initializes MHE; no
  parallel hypotheses are retained by this conventional baseline.
- From `31` through `440 s`, perform one warm-started projected
  Levenberg--Marquardt/Gauss--Newton update per one-second sample.
- Maximum iterations `80`, initial damping `1e-3`, gradient tolerance `1e-10`
  and step tolerance `1e-8 m` are inherited unchanged from V19/V27.
- There is no outcome-dependent restart or global rescue after 30 s.  A finite
  iterate at the iteration limit is reported with `converged=false`; a
  non-finite result is a campaign failure.

### Explicit exclusions

V34 has no Gaussian initialization penalty, process noise, fading/forgetting,
Huber loss, PF soft gate, nuisance parameter, truth-derived covariance,
adaptive horizon or post-outcome tuning.  Adding any of these requires a new
pre-frozen experiment and is forbidden in V34.

## Uncertainty and runtime

At each checkpoint the local information matrix is the sum of the arrival
matrix and the exact window Gauss--Newton matrix.  Its covariance uses the same
V27 variance floor:

```text
s2 = max(approximate_full_SSE / (2*N - 3), 0.05^2)
P  = s2 * inverse(Lambda_a + J_window' J_window).
```

The covariance is invalidated at a shell boundary, for rank below three or
condition number above `1e10`.  The resulting radius is a nominal local
Hessian radius, not a calibrated 95-percent guarantee.

Report both:

- latency of the update that produced the checkpoint;
- cumulative estimator CPU time from the 30-s initialization through the
  checkpoint; and
- nearest-rank p99 and maximum latency over all online updates completed so
  far.

File I/O, full-prefix diagnostic residual evaluation, truth scoring and audit
time are excluded from estimator runtime.

## Smoke contract

Smoke uses the same code path on episodes `0` and `73`, all five checkpoints,
and a reduced initializer of one 512-point Halton sweep and eight local starts.
These numerical smoke settings are serialized and never pooled with scientific
results.

Smoke passes only if:

1. all 10 unscored and scored cells exist and are finite;
2. arrival/window counts equal the causal prefix without double counting;
3. unscored outputs contain no truth/error/success fields;
4. an entire repeated episode is deterministic apart from runtime fields;
5. unit tests pass;
6. independent audit recomputes every score and validates source/snapshot
   hashes; and
7. no source or artifact intersects `49900..50999`.

A code defect may be repaired only before scientific full-run outcomes are
seen and without changing the frozen method or decision thresholds.  A
methodological defect stops V34 before the full run.

## Full campaign metrics

For each checkpoint report endpoint and initial-position error, success below
and at `7 m`, median/p95/max error, nominal-radius coverage, false-confidence
count (`radius < 7 m` while endpoint error is `> 7 m`), convergence, local
rank/condition, active-window and marginalized counts, checkpoint latency,
cumulative CPU time, all-update p99/max latency, and full-prefix residual RMSE.

All comparisons with V27 `global_full` are paired by episode and checkpoint.
The frozen V27 result files and their hashes are included in the V34 campaign
contract before execution.  Mean paired excess error is

`MHE endpoint error - global_full endpoint error`.

Its percentile-bootstrap 95% interval uses 50,000 paired resamples and PCG64
seed `34045000`.

## Frozen development decision

The full result is `MHE60_COMPETITIVE_NOMINAL_BASELINE` only if all of the
following hold:

1. campaign integrity, deterministic tests and independent audit pass;
2. all 500 V34 cells contain finite estimates;
3. V34 has at least 99/100 successes at each of `120, 240, 440 s`;
4. at each of those checkpoints it loses no more than one success relative to
   the frozen V27 `global_full` arm;
5. the upper 95% bootstrap bound for mean paired excess error at `440 s` is at
   most `0.5 m`;
6. nearest-rank p99 latency over all post-initialization online updates is less
   than the one-second measurement interval; and
7. false confidence occurs in at most one episode at each of
   `120, 240, 440 s`.

Otherwise the result is `MHE60_NOT_NONINFERIOR`.  Either outcome remains a
required publication baseline and does not authorize tuning, opening a closed
seed range, a final publication claim or removal of an unfavourable result.

