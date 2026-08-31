# Frozen V18 protocol: causal PF-resampling guard

Protocol frozen before V18 training on 2026-07-14.  This document authorizes
one implementation, one three-seed training campaign and one development
evaluation boundary.  Results may not be used to change the hypothesis,
mechanism, thresholds, seeds, models or gates below.

## 1. Scientific question and single change from V17

V17 bounded the learned increment but still allowed its deterministic greedy
reference to request gain 1.0.  The V17 development traces indicate that, in
some Tail80 failures, a large request near particle-filter degeneracy was
followed by resampling/roughening and a sharp increase in estimated position
uncertainty.  This is a causal interpretation of the traces, not a proven
mechanism, because V17 did not log every resampling decision directly.

V18 tests one hypothesis only:

> Capping the complete information-ray gain while the online particle filter
> is at immediate or recent resampling risk will remove the isolated Tail80
> regressions without losing V17's terminal-success, information and formation
> benefits over PID+EXC.

At decision state `t`, let `M_{t-1}` be the minimum pre-resampling effective
sample-size fraction (`ESS/N`) observed across every PF measurement update in
the immediately preceding completed action. If that action contained no PF
measurement (including reset), use the ESS fraction of the current particle
weights. Non-finite or malformed ESS evidence fails closed as zero. Let `R_k`
indicate whether any PF resampling occurred during completed action interval
`k`, and let `g_ref`, `g_safe` and `e` retain their exact V17 meanings. The
frozen trigger is

    guard_t = (M_{t-1} <= 0.50) OR any(R_k for k in t-15, ..., t-1).

The inclusive ESS threshold is exactly `0.50`.  The history is exactly the 15
previous completed action intervals.  When the guard is active,

    g_ref18     = min(g_ref, 0.50)
    g_ceiling18 = min(g_safe, g_ref18 + 0.25, 0.50)
    g_request18 = g_ref18 + e (g_ceiling18 - g_ref18).

Thus the deterministic reference, actor ceiling, requested gain and any
certified applied gain are all at most `0.50` while guarded.  When the guard is
inactive, the reference, ceiling, interpolation, certificate and fallback are
exactly V17.  The one-grid-step maximum learned increment remains `0.25`.

No plant, sensor, PF update, resampling algorithm, roughening rule, noise tape,
observation width/history, reward, greedy search, four-second predictor,
half-open 30 s FIM window, action certificate, curriculum, success definition
or baseline controller changes are authorized.  No truth state or future
noise enters the policy or guard.

## 2. Frozen causal timing

The guard is evaluated before composing action `t`, using only cached online
ESS evidence from the completed interval `t-1`, the current PF weights when
that interval had no measurement, and resampling outcomes already produced by
action intervals `t-15` through `t-1`. Resampling caused during the action
currently being selected is unknown and cannot affect that same action.

A resampling during interval `t` first activates the history branch at
decision `t+1` and remains in its 15-interval window through decision `t+15`.
At episode reset the resampling history is empty. The preceding-action minimum
pre-resampling ESS, previous-interval resampling and post-action outcome must
be logged separately. Any current/future resampling leakage, loss of an early
resampling when a later update in the same action did not resample, off-by-one
history or mismatch between the logged trigger and the applied cap invalidates
that seed.

## 3. Frozen implementation and provenance

- Candidate version: `v18_resampling_guard_1.0`.
- Candidate variant: `resampling_risk_guard_v18`.
- Run-manifest schema: exactly `8`.
- Frozen source inventory: exactly 51 regular files, copied to a hash-bound,
  read-only `execution_source` tree before preflight.
- Output root:
  `experiments_v18_resampling_guard_numba_3x8env`.
- Approved interpreter: `.venv-v13-numba/bin/python`.
- Preflight/integration cache seed: `28000` only.  It is not a training seed,
  and no training-seed cache may exist before the detached worker starts.
- One immutable campaign contract binds interpreter hash, source hashes,
  commands, seeds, scientific constants and output layout.

The public launcher may perform only a bounded runtime probe and the frozen
V18 tests, then it starts one detached process group under macOS `caffeinate`.
After the startup handshake there is no watchdog or periodic monitoring.
There is no resume, overwrite, retry, warm start or automatic recovery.

## 4. Frozen training campaign

Three independent SAC models are trained from scratch and strictly in order:

1. `28001`;
2. `28002`;
3. `28003`.

A later seed starts only after the preceding process exits zero and its schema-8
manifest, exact 51-file snapshot and required model hashes validate.  Failure
closes the queue and leaves later seed directories absent.

Every seed uses the unchanged V17 matrix:

- 1,000,000 requested transitions and 1,000,008 saved transitions;
- 8 subprocess environments, train frequency 3 and 8 gradient steps;
- PF1024, Numba PF and predictors, CPU;
- four-frame history and observation dimension 392;
- `512,512,512` ReLU actor/critic, batch 256;
- learning starts at 50,000 transitions;
- fixed entropy coefficient `0.002`, target entropy `auto`;
- curriculum fraction `0.60`, replay reset disabled;
- action interval 2 s and FIM window 30 s;
- internal evaluation every 31,248 callback calls, 20 episodes;
- checkpoint every 62,499 callback calls;
- TensorBoard information every 24,984 transitions;
- diagnostic trace every 249,984 transitions.

Only `final_model.zip` is eligible for reported evaluation.  Internal best
models and checkpoints cannot be selected retrospectively.

## 5. Frozen evaluation boundary and estimands

The V17 development range `42000..42099` influenced V18 and is not unseen V18
evidence.  If all three training runs validate, each final model is evaluated
separately and deterministically on the paired development scenarios
`43000..43099`, exactly 100 episodes per controller.  Controllers, initial
states and noise tapes must align by episode.

The two prespecified comparisons are:

- `rl_vs_pid_track_exc` (`PID+EXC`);
- `rl_vs_deterministic_greedy_grid` (the frozen V15 full-safe greedy grid).

Differences are always candidate minus baseline.  Reported intervals are
paired two-sided 95% percentile bootstrap intervals with 5,000 resamples and
the evaluator's frozen deterministic bootstrap seeds.  No episode may be
dropped.  All 100 pairs and all required finite fields must be present.

Terminal formation is the continuous
`formation_error_ratio_terminal` metric, not an ambiguously named binary
formation-success field.  Tail80 uses the already frozen last-80-second
success definition.  Terminal information is
`fim_online_win_eig_min_terminal`.

## 6. Per-seed advancement gates

Each of `28001`, `28002` and `28003` must pass every gate independently.
Pooling, majority vote or selecting the best seed cannot conceal a failure.

Against PID+EXC:

1. the lower paired 95% CI for terminal-success difference is strictly above
   zero;
2. the lower paired 95% CI for Tail80-success difference is strictly above
   zero;
3. the lower paired 95% CI for terminal FIM minimum-eigenvalue difference is
   strictly above zero;
4. the upper paired 95% CI for terminal formation-error-ratio difference is
   at most `+0.08`.

Against deterministic greedy grid, V18 is a prespecified non-inferiority test,
not a claim of RL superiority:

1. lower paired 95% CIs for terminal and Tail80 success differences are each
   at least `-0.05`;
2. the lower paired 95% CI for terminal FIM difference is at least `-5e-5`;
3. the upper paired 95% CI for terminal formation-error-ratio difference is at
   most `+0.08`.

Mechanism and integrity gates:

1. at least one guarded and one unguarded RL decision occur on the 100-episode
   development set, so both branches are exercised;
2. the trigger equals exactly the inclusive ESS/history formula on every RL
   decision, with the causal timing in section 2;
3. whenever guarded, reference, ceiling, requested and applied gain are all
   at most `0.50`; whenever unguarded, V18 bounds equal the logged V17
   counterfactual bounds;
4. ceiling minus reference never exceeds `0.25`, requested gain lies within
   the bounds, and the unchanged certificate/fallback invariants hold;
5. scenario, noise-tape, pairing, range and trace alignment violations are
   zero; NaN and non-finite counts are zero;
6. plant-action saturation step fraction is at most 1%.

The mechanism gates establish that the intended intervention actually ran;
they do not substitute for the paired outcome gates.

## 7. Sealed final boundary

Scenarios `50000..50999` remain sealed.  This protocol does not authorize
opening them.  A final evaluation requires all three per-seed development
gates above to pass and a separate explicit decision after the development
report is complete.  Any exploratory access makes that range ineligible as an
unseen final test.

## 8. Runtime estimate

The completed V17 sequential campaign took approximately 5 h 05 min on this
host with the same 3-by-1M, 8-environment matrix.  The operational V18 window
is therefore frozen as 5--7 h from detached launch.  This is an estimate, not
an early-stopping or timeout rule.
