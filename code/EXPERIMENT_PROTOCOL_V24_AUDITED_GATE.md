# V24 audited causal gate — frozen development protocol

## Question

Does repairing the V23 release gate so that the current local Doppler
solution is explicitly consistent with both global optimizer replications,
and both global modes satisfy explicit nominal Hessian-radius and residual
limits, preserve useful lock/terminal performance without unsafe TRACK
transitions?

This is a gate audit, not RL training and not a final evaluation.  The final
seed range `50000..50999` remains sealed.

## Frozen paired campaign

- Development seeds: `49500..49599` (100 paired scenarios).
- Engineering smoke seeds: `49480..49499`; smoke results cannot support the
  scientific decision.
- Both arms use the same environment metadata, initial seed, exogenous noise
  tape, V19 estimator, V22 belief-FIM acquisition planner and PID tracker.
- `v23_early_reference`: frozen V22 runner with the V23 early gate.
- `v24_audited_gate`: identical stack with only the audited gate substituted.
- Fixed horizon and full estimator settings for dev100: 4096 coarse
  candidates, 2 sweeps, 48 local starts, raw gate mode and uniform-radius
  candidates.

## Prespecified gate

The unchanged V23-early base release uses minimum time 60 s, primary versus
optimizer-replication agreement at most 2 m, global stability at most 3 m,
alternative-mode delta chi-square at least 13.82, and three consecutive
complete passes.  All other V21 base thresholds remain unchanged.

V24 adds all of the following at release:

- current local initial-position solution to primary global mode <= 2 m;
- current local solution to confirmation optimizer replication <= 2 m;
- nominal local Hessian radius of each global best mode <= 7 m;
- full residual RMSE of each global best mode <= 0.08 m/s.

The corresponding hold limits are 7 m, 10 m and 0.10 m/s.  Ordinary hold
failures retain the V21 three-action hysteresis.  A material local-to-global
hold disagreement causes immediate ACQUIRE re-entry and requests a global
refresh before another TRACK action.  These checks receive no simulator truth,
legacy PF state, reward or task-success signal.

The optimizer replication uses a different deterministic multistart seed but
the same measurements.  It is therefore described as optimizer replication,
not statistically independent evidence.  Likewise, `local_radius95_m` is a
nominal local Hessian/FIM radius, not claimed to be an empirically calibrated
95% coverage interval.

## Exact causal scoring

For trace row `k`, a gate transition is
`gate_locked_after_update[k] and not gate_locked_after_update[k-1]`; its error
is the post-update localization error in row `k`.  A TRACK action in row `k`
starts from the localization state in row `k-1` and ends in row `k`.
Transitions, TRACK starts and TRACK ends are scored separately, for every
reacquisition.  A localization error is unsafe at `>= 7 m` (or if non-finite).

Pairing is verified through the last action before the first phase divergence,
not through an arbitrary fixed time.  Initial conditions, tape hash, noise
cursors, actions, truth and estimates must agree before treatment.

## Prespecified development decision

`SUPPORT_V24_AUDITED_GATE` requires all of:

- audited ever-lock rate >= 0.95;
- audited terminal joint-success rate >= 0.95;
- zero unsafe transition, TRACK-start and TRACK-end events;
- audited terminal rate no more than 0.02 below the paired reference;
- maximum estimator plus planner decision runtime < 2 s;
- median paired V24 minus V23 first-transition delay <= 30 s;
- no release missing the six added audit checks and no pre-treatment mismatch.

All thresholds are rates or paired statistics and therefore remain valid for
non-100 diagnostic runs.  A smoke or incomplete campaign returns
`SMOKE_ONLY`/`INCOMPLETE`, never a supportive scientific label.

## Reporting

Report arm-wise rates and exact error distributions; paired transition-delay
median/p95/max and faster/tied/later counts; terminal both/reference-only/
V24-only/neither counts; compute runtime; source hashes; and the untouched
sealed range.  V24 is a development gate audit.  A supportive result is not a
final generalization claim and does not authorize opening final seeds.
