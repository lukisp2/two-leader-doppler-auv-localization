# EXPERIMENT PROTOCOL V17 — bounded temporal escalation, three-seed screen

Status: frozen before any V17 training transition was generated.

## 1. Motivation and single scientific change

The completed V16 development analysis showed that the temporal greedy-floor
controller is useful, but that unrestricted learned escalation from the greedy
reference to the largest fully safe gain can spend more formation authority
than necessary.  This was most visible for training seed `26004`.  V17 tests
one hypothesis only: retaining the V16 greedy lower bound while limiting the
learned increment to one interval of the already frozen V15 gain grid will
preserve the late information benefit and reduce unnecessary escalation.

Let `g_ref` be the unchanged V16 deterministic greedy reference, `g_safe` the
unchanged largest gain passing the complete V15 certificate on the frozen
grid, and `e` the SAC output in `[0,1]`.  V17 defines

    g_ceiling   = min(g_safe, g_ref + 0.25)
    g_requested = g_ref + e (g_ceiling - g_ref).

The exact continuous requested action is passed through the unchanged full
certificate.  The V16 reference fallback remains unchanged.  The plant,
particle filter, observation history, greedy rule, temporal features, reward,
four-second predictor, retained half-open 30 s FIM window, certificate and
direct/fixed-gain baselines are otherwise identical to V16.  The last of the
98 base observation values now exposes the actionable `g_ceiling`; the raw
`g_safe` remains a diagnostic only.  No truth or future-noise input is added.

## 2. Frozen training campaign

Three independent SAC models are trained from scratch and strictly one after
another in this order:

1. seed `27001`;
2. seed `27002`;
3. seed `27003`.

Every seed uses:

- `1,000,000` requested transitions with 8 subprocess environments, producing
  `1,000,008` saved transitions because complete 24-transition rollout blocks
  are retained (`8` environments times train frequency `3`);
- PF1024 and compiled Numba PF/predictors on CPU;
- four-frame history, policy observation dimension 392;
- `512,512,512` ReLU policy and critic networks;
- batch 256, train frequency 3 vector calls, 8 gradient steps;
- learning starts at 50,000 transitions;
- fixed entropy coefficient `0.002` and target entropy `auto`;
- curriculum fraction `0.60`, with replay reset disabled;
- action interval 2 s and FIM window 30 s;
- internal evaluation every 31,248 callback calls, 20 episodes;
- checkpoint every 62,499 callback calls;
- TensorBoard information every 24,984 model transitions;
- diagnostic trace every 249,984 model transitions.

Each seed has separate model, log, TensorBoard and cache directories below
`experiments_v17_bounded_escalation_numba_3x8env/replication/`.  There is no
warm start, resume, overwrite, automatic retry, checkpoint selection, early
stopping or concurrent V17 training.  A failed seed stops the queue and the
next seed is not started.  Advancement requires a completed run manifest and
hashable `final_model.zip`, `last_model.zip` and `vecnormalize.pkl`.

The launcher freezes a read-only source tree and a hash-bound campaign
contract, performs a bounded preflight and integration smoke test, then starts
one detached sequential worker under macOS `caffeinate`.  After the startup
handshake it performs no external monitoring.

## 3. Evaluation boundary frozen before training

Development scenarios `41000..41099` influenced the V17 design and therefore
must not be used as unseen V17 evidence.  If all three runs complete, the next
development evaluation is paired on the previously unseen scenarios
`42000..42099`.  Only each run's `final_model.zip` is eligible.  Results remain
separate by training seed; pooling may summarize them but cannot conceal a
failed seed.

For each V17 training seed, advancement requires all of the following on the
paired development set:

- the upper paired 95% CI for the terminal formation-success difference versus
  PID+EXC is no greater than `+0.08`;
- lower paired 95% CIs for terminal success and Tail80 success versus PID+EXC
  are strictly positive;
- the lower paired 95% CI for terminal FIM minimum eigenvalue versus PID+EXC
  is strictly positive;
- the lower 95% CI of applied gain in the final 100 s before termination is
  strictly positive;
- zero scenario-alignment, pairing, range, NaN or non-finite violations;
- actuator saturation is at most 1%.

The `+0.08` formation threshold is a new conservative V17 development
criterion declared here; it is not retrospectively attributed to an older
protocol.  The final scenarios `50000..50999` remain sealed and are not
authorized by this protocol.

## 4. Runtime estimate

The completed solo V16 seed `26002` took 2 h 56 min 49 s with the same 8-env
training matrix.  Three sequential seeds therefore have a measured-base
estimate of about 8 h 50 min.  Allowing for preflight, internal evaluations and
ordinary load variation, the operational estimate is 9–10 h, with a cautious
window of 9–11 h from detached launch.
