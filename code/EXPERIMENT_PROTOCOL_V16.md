# EXPERIMENT PROTOCOL V16 — temporal greedy-floor escalation, 1M screening

Status: frozen before the only authorized v16 screening launch.

## 1. Hypothesis and scope

The completed v15 development evaluation established that the learned gain was
genuinely state-dependent and improved terminal, Tail80, and dwell success over
PID+EXC.  It also exposed a temporal failure: the learned accepted gain fell in
the final 100 s even though the deterministic safe greedy-grid gain rose.  V15
therefore had better time-mean information than the greedy controller but worse
terminal information and terminal uncertainty.

V16 tests one bounded hypothesis: explicit remaining-horizon and FIM-expiry
context, combined with a deterministic safe greedy lower bound and learned
late escalation, can correct that timing error.  This is a one-million-
transition architecture screen, not a confirmatory paper run.  No final
evaluation seed is authorized by this protocol.

## 2. Controller and online boundary

The plant, particle filter, measured online state, PID+EXC inner controller,
v14 information direction, v15 four-second predictor, retained half-open 30 s
FIM window, and complete fail-closed v15 action certificate are unchanged.

The policy emits one scalar escalation `e` in `[0,1]`.  The v15 deterministic
greedy-grid gain `g_ref` is the lowest-gain maximizer of strictly positive v15
utility among exact full-safe gains `{0.25, 0.50, 0.75, 1.00}`, with zero as the
fallback. Let `g_safe` be the largest full-safe grid gain. The continuous gain
sent to the full certificate is

    g = g_ref + e (g_safe - g_ref).

The certificate still evaluates this exact continuous action. If a continuous
intermediate escalation unexpectedly fails, the exact greedy reference is
re-certified and applied; if there is no positive reference or that
re-certification fails, the exact PID+EXC action is applied. Reference fallback
has a small actor-dependent training cost. Fixed-gain ablations in the
evaluator retain exact fixed-gain semantics and do not pass through the RL
escalation transform.

Simulator truth, oracle FIM/error, actual localization error, and future
exogenous-noise samples remain forbidden policy, reference, certificate, and
reward inputs.

## 3. Observation and reward

The inherited v15 base observation has 88 values.  V16 adds exactly ten
online-only values:

1. normalized episode progress;
2. normalized remaining time;
3. final-100-s active flag;
4. continuous final-100-s phase;
5. raw-PF-uncertainty deficit relative to the online success threshold;
6. terminal urgency (tail phase times uncertainty deficit);
7. scaled zero-gain post-window FIM minimum eigenvalue;
8. signed imminent FIM-expiry feature;
9. deterministic greedy reference gain;
10. maximum exact full-safe grid gain.

The base dimension is exactly 98 and four-frame history gives policy dimension
392.

An accepted action receives the inherited online reward plus one bounded v16
tradeoff.  The tradeoff keeps the v15 information benefit, adds an online late-
uncertainty urgency benefit, applies a predicted-formation barrier only above
65% of the online formation tolerance, and retains a smaller authority cost
that is discounted under late uncertainty urgency.  The frozen constants are:

- information weight and scale inherited from v15: `0.75`, `5e-5`;
- urgency weight: `0.45`;
- formation barrier start: `0.65` of tolerance;
- formation barrier weight: `0.60`;
- authority weight: `0.08`;
- maximum urgency discount of authority cost: `0.75`.
- rejected-escalation fallback penalty weight: `0.05`.

## 4. Frozen screening matrix

There is exactly one authorized run:

- phase: `screening`;
- version: `v16_temporal_escalation_1.0`;
- variant: `temporal_greedy_escalation_v16`;
- seed: `26001`;
- requested transitions: `1,000,000`;
- expected saved SB3 transitions with 24 environments: `1,000,008`;
- SAC trained from scratch, with no resume or warm start;
- CPU, 24 subprocess environments, PF1024, compiled Numba PF and predictors;
- observation history 4, exact dimension 392;
- `512,512,512` ReLU network;
- batch 256, train frequency one vector step, eight gradient steps;
- learning starts at 50,000 transitions;
- fixed entropy coefficient `0.002`;
- curriculum fraction `0.60`, giving approximately 600k ramp transitions and
  400k full-hard transitions;
- replay reset disabled;
- action interval 2 s and online FIM window 30 s.

Callbacks retain their native units:

- evaluation every 10,416 callback calls = 249,984 transitions;
- checkpoint every 20,833 callback calls = 499,992 transitions;
- TensorBoard information every 24,984 model transitions;
- diagnostic trace every 249,984 model transitions;
- 20 internal evaluation episodes.

The exact output is

    experiments_v16_temporal_escalation_numba_24env/
      screening/temporal_greedy_escalation_v16/seed_26001/

## 5. Execution and absence of monitoring

The launcher validates the completed v15 source hashes, copies an exact v16 to
v8 allowlist into a read-only execution tree, runs the v16 runtime/Numba probe
and the frozen preflight tests, binds the command and hashes in a contract, and
starts one detached process under `/usr/bin/caffeinate -dimsu`.

The launcher waits only for the initial running manifest and then returns.
There is no periodic external monitor, watchdog, automatic retry, restart,
resume, checkpoint selection, or early stopping.  Internal fixed evaluations
and checkpoints are training callbacks and do not authorize intervention.

Runtime duration is estimated from the completed v15 5M run and its identical
expensive predictor/PF path.  The planning estimate is approximately 2.5 h
after the detached start; the actual completion time may vary with CPU load and
the four internal evaluations.

## 6. Interpretation after completion

The screening model must not be evaluated on final seeds `50000..50999`.
Development evaluation must compare v16 RL with PID, PID+EXC, exact fixed gains,
the unchanged v15 deterministic greedy-grid controller, and v15 RL on paired
development scenarios.  Advancement requires, at minimum, no formation-
guardrail violation and evidence that v16 no longer reduces gain in the final
100 s.  One training seed does not estimate training variance.
