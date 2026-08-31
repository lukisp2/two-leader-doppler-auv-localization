# EXPERIMENT PROTOCOL V14 — exploratory dense information-ray pilot

Status: frozen before the first v14 pilot.

## 1. Scope and interpretation

V14 is a new exploratory controller version motivated by the corrected-Numba
v13 dev100 `NO_GO`. V13 retained the PID+EXC reference but did not demonstrate
the prespecified positive improvement in terminal online FIM. Its learned gain
was low and its learned entropy coefficient collapsed. V14 is intended to test
whether a stronger certified ray, early planner exposure, a same-belief
counterfactual certificate, a bounded dense online information bonus, and a
fixed entropy coefficient make the information action identifiable during a
1M-step pilot.

This protocol does not alter or reinterpret any v13 endpoint or result. All
v13 files, models, evaluations, reports, gates and final seeds remain frozen.
The v14 pilot is exploratory and cannot convert the v13 `NO_GO` into a `GO`.

## 2. Frozen controller and reward contract

The inner stabilizer remains direct `pid_track_exc`. The learned action remains
a scalar gain in `[0,1]` on the certified non-negative online information ray.
The plant action remains three-dimensional. Gain zero returns the inherited
PID+EXC action and inherited reward exactly, without an added term or an extra
clip operation.

The frozen v14 direction scale is:

    S14 = diag(0.30, 0.50, 0.40)

This increases the v13 authority while retaining channel-specific limits. The
inherited v13 certificate, tracking gate, information gate, online-only
boundary and non-negative alignment invariants remain in force. Before a
candidate reaches the plant, v14 evaluates the actual unclipped composite
action and the exact PID+EXC action from the same pre-step online PF belief.
The candidate must be finite, within plant bounds without clipping, improve
the predicted minimum FIM eigenvalue by more than `1e-8`, remain within the
online formation tolerance, and degrade predicted formation error by no more
than `0.25 * tolerance`. Any rejection applies exact PID+EXC. V14 sets the
online planner curriculum start difficulty to `0.0` and ramp width to `0.10`
so that the ray is available before SAC begins learning.

When the counterfactual candidate is accepted, the once-only additional reward
is derived only from its pre-step online predicted improvement:

    delta_eig = candidate_predicted_eig_min - inner_predicted_eig_min
    r_dense = 0.75 * tanh(delta_eig / 5e-5)
    reward_v14 = reward_v13 + r_dense

Otherwise `r_dense` is exactly zero. The term is finite, non-negative and
bounded by `0.75`. The prediction pair is evaluated from one unchanged online
belief before the plant step. Truth positions, truth errors, oracle FIM/CRLB
and future information are forbidden reward or action inputs.

The manifest action contract identifies at least:

- environment version `v14_info_ray_1.0`;
- variant `info_ray_gain_v14`;
- policy action dimension 1 and bounds `[0,1]`;
- plant action dimension 3 and bounds `[-1,1]^3`;
- inner controller `pid_track_exc`;
- direction scale `[0.30,0.50,0.40]`;
- same-belief counterfactual minimum eigenvalue threshold `1e-8` and maximum
  predicted-error degradation ratio `0.25`;
- fixed dense reward weight `0.75` and eigenvalue scale `5e-5`;
- planner curriculum start `0.0` and ramp `0.10`.

## 3. Frozen 1M training matrix

There is exactly one scientific v14 run in this protocol:

- phase: `pilot`;
- variant: `info_ray_gain_v14`;
- seed: `24001`;
- requested transitions: `1,000,000`;
- expected SB3 saved transitions with 24 vector environments: `1,000,008`;
- new SAC policy trained from scratch;
- CPU and 24 vector environments;
- PF1024 with effective Numba and warm-up;
- online information planner explicitly using Numba;
- four-frame observation history;
- `512,512,512` ReLU network;
- action interval 2 s and FIM window 30 s;
- batch 256, train frequency 1 and one gradient step;
- learning starts at 50,000 transitions;
- fixed entropy coefficient `0.02`; target entropy argument `auto` is recorded
  but inactive for a fixed coefficient;
- 90% curriculum ramp and replay reset at difficulty 0.80;
- the v13 pilot evaluation, checkpoint, TensorBoard and trace frequencies.

The exact output is:

    experiments_v14_info_ray_gain_numba_24env/
      pilot/info_ray_gain_v14/seed_24001/

Resume, retry, overwrite, alternate seed, shortened budget, device override and
additional v14 scientific runs are not authorized. A failure consumes this
one-shot pilot. A new attempt requires a new explicit protocol/version/seed.

## 4. Frozen sources and runtime

Before launch, the runner copies an exact allowlist into
`control/pilot/execution_source`. The allowlist includes the v14 core,
evaluator, runner, protocol and all three v14 tests, plus the full inherited
v13/v12/v11/v10/v8 dependency and regression chain. The evaluator is therefore
frozen before training rather than written after observing the result.

Every copied file is a regular non-symlink file, its raw-byte SHA-256 is bound
in `source_manifest.json`, the inventory admits no extra or missing files, and
the copied tree is made read-only. Inherited sources must match the source map
of the completed corrected-Numba v13 pilot. Training imports v14 from this
copied tree, not from the mutable project tree. The inner model snapshot and
manifest must reproduce the launch source map.

The approved interpreter is `.venv-v13-numba/bin/python`. The preflight pins
Python 3.11.9, NumPy 2.3.3, Torch 2.10.0, Stable-Baselines3 2.7.1, Gymnasium
1.2.3, Cloudpickle 3.1.2, Numba 0.66.0 and llvmlite 0.48.0. It requires the
approved module origins, effective compiled Numba PF and planner dispatchers,
and deterministic Numba/reference equivalence. `NUMBA_DISABLE_JIT` and
unapproved backend/variant overrides are removed.

## 5. One-shot detached execution

The public runner exposes only `plan`, `launch` and read-only `status`. It has
no scientific override options. Launch performs bounded source, runtime and
test preflight, creates immutable hash bindings and commits a retained
`one_shot.lock`. Once the output root is committed, all later launches are
forbidden even if preflight, process creation or training fails.

The worker is started under `/usr/bin/caffeinate -dimsu` in a new session with
stdin disconnected and stdout/stderr written exclusively to the pilot console.
It calls the frozen v14 `main` in the same process, avoiding an independently
orphanable trainer subprocess. The public launcher waits only for a bounded
startup receipt and then returns. There is no periodic external monitor,
watchdog, selective stopping rule, automatic retry or resume.

`caffeinate` prevents ordinary macOS idle sleep while the process lives. It
does not make training survive reboot, power loss or a hard process kill.

## 6. Completion evidence and later evaluation

Completion is not inferred from process exit alone. The inner
`v14_run_manifest.json` has schema 4 and must report `completed`, the frozen
version/variant/seed/action contract and the exact arguments. SHA-256 values of
`final_model.zip`, `last_model.zip` and `vecnormalize.pkl` must match the files.
The final SAC archive must reopen and contain exactly 1,000,008 saved
transitions. The outer manifest records the single completed attempt and binds
the inner validation.

The evaluator and its comparison contract are frozen before training. Reported
development evaluation uses exactly seeds `41000..41099`; final evaluation
reserves exactly `50000..50999`. In addition to the learned controller and the
inherited direct comparators, the identical v14 certificate must be exercised
with fixed scalar gains `0.5` and `1.0` as mandatory mechanistic ablations.

Freezing these ranges and controllers does not make the 1M training pilot
confirmatory and does not authorize either evaluation now. The decision to run
development/final evaluation and the criteria used to interpret it will be a
separate, explicit post-pilot decision. No post-pilot GO/NO-GO criterion is
specified by this training protocol.
