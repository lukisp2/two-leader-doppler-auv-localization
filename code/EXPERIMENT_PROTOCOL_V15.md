# EXPERIMENT PROTOCOL V15 — contextual certified gain, 5M training

Status: frozen before the only v15 training launch.

## 1. Scope and interpretation

V15 is an exploratory successor to the completed v14 one-million-transition
pilot.  V14 improved success over PID+EXC, but its deterministic learned gain
was nearly constant and did not outperform the mandatory fixed-gain ablations.
V15 tests the concrete hypothesis that the actor lacked an explicit online
description of information opportunity and the post-window FIM consequence of
its gain.  It is not a reinterpretation of v11--v14 and does not modify their
sources, artifacts, evaluations, or conclusions.

There is exactly one authorized v15 training run.  A single training seed does
not estimate training variance.  Development results already inspected while
designing v15 remain exploratory; an untouched final seed panel is required
before a confirmatory paper claim.

## 2. Frozen controller, observation, and information-time contract

The inner stabilizer remains direct `pid_track_exc`.  The policy still emits a
single scalar gain in `[0,1]` on the certified non-negative v14 information
ray.  The plant receives three bounded channels.  A rejected certificate or a
zero gain remains the exact inner-controller action and neutral inherited
reward branch.  Simulator truth, oracle error/FIM, and future exogenous noise
are forbidden policy, certificate, and reward inputs.

V15 retains the v14 channel scale `diag(0.30, 0.50, 0.40)` and adds an
online-only opportunity grid at gains `0.25, 0.50, 0.75, 1.00`.  The base
observation has exactly 88 values: the inherited 67, nine contextual values,
and twelve grid values.  Four-frame history therefore gives an exact policy
observation dimension of 352.  Grid values are computed from the current
online PF belief and measured kinematics only.

The predictor horizon is exactly `H = 4.0 s` with `dt = 1.0 s`, hence four
future prediction steps.  The online reporting window is exactly `T = 30.0 s`.
At decision time `t`, retained measured online FIM is formed from the half-open
interval

    (t + H - T, t]

using the exact cutoff `t + H - T`; predicted four-second increments are then
added to compare the exact PID+EXC command and candidate command at `t+H`.
The inherited v10/v14 runtime configuration uses a four-step, one-second
information planner, and v15 explicitly freezes the consistent value `4.0 s`.

The runtime must prove Python/Numba equivalence for `fim_increment`,
`pred_err`, and `post_window_eigmin`, compile `_v15_predict_increment_numba`,
verify `BASE_OBS_DIM == 88`, verify `H=4`, `T=30`, and verify the public retained
cutoff helper on a real environment.

## 3. Frozen 5M training matrix

There is one run only:

- phase: `training`;
- version: `v15_contextual_gain_1.0`;
- variant: `contextual_info_ray_gain_v15`;
- seed: `25001`;
- requested transitions: `5,000,000`;
- expected saved SB3 transitions with 24 environments: `5,000,016`;
- new SAC policy trained from scratch, with no resume or warm start;
- CPU, 24 subprocess environments, PF1024, compiled Numba PF and planners;
- observation history 4, exact dimension 352;
- `512,512,512` ReLU network;
- batch 256, train frequency one vector step, eight gradient steps;
- update/transition ratio `8/24 = 1/3`;
- learning starts at 50,000 transitions;
- fixed entropy coefficient `0.002`;
- `target_entropy=auto` is recorded but inert for fixed entropy;
- the lower coefficient relative to v14 is deliberate because unsafe/masked
  actor commands collapse to the inner action, whose normalized actor midpoint
  would otherwise be over-rewarded by entropy;
- curriculum fraction `0.60`, i.e. 125,000 steps per environment to full hard,
  approximately 3M global transitions of ramp and 2M at full difficulty;
- replay reset threshold `0.0`, meaning reset is disabled; the rolling buffer
  forgets earlier curriculum data without an abrupt small-buffer update burst;
- action interval 2 s and online FIM window 30 s.

The exact callback arguments are intentionally expressed in their native
units:

- `eval_freq=10416` callback calls = 249,984 global transitions;
- `eval_episodes=20`;
- `save_freq=20833` callback calls = 499,992 global transitions;
- `tb_info_freq=24984` model timesteps;
- `trace_freq=249984` model timesteps.

`EvalCallback` and `CheckpointCallback` count callback calls, while the custom
TensorBoard and trace callbacks compare `model.num_timesteps` directly.  These
units must not be divided by 24 a second time.  There are 20 scheduled internal
evaluations through step 4,999,680 and ten scheduled checkpoints through step
4,999,920.  Internal evaluation and checkpointing are fixed training callbacks;
they do not authorize early stopping, external polling, or checkpoint-based
intervention.

The exact output is:

    experiments_v15_contextual_gain_numba_24env/
      training/contextual_info_ray_gain_v15/seed_25001/

## 4. Frozen sources, runtime, and completion evidence

Before launch, the runner copies an exact allowlist containing v15 core,
evaluator, runner, protocol, and tests plus the complete frozen v14--v8 parent
chain.  Every entry must be a regular non-symlink file.  The copied tree is
made read-only, bound by SHA-256, and used as the only execution `PYTHONPATH`.
The mutable project tree must match the completed v14 source map for every
parent file.  The v15 inner source snapshot and manifest must reproduce the
launch map exactly.

The approved interpreter remains `.venv-v13-numba/bin/python`.  Runtime pins
Python 3.11.9, NumPy 2.3.3, Torch 2.10.0, Stable-Baselines3 2.7.1, Gymnasium
1.2.3, Cloudpickle 3.1.2, Numba 0.66.0, and llvmlite 0.48.0.  The launcher
requires at least 6 GiB free before committing the run.

Completion requires schema-5 `v15_run_manifest.json` with `completed` status,
the exact arguments and dimensions, matching SHA-256 values for
`final_model.zip`, `last_model.zip`, and `vecnormalize.pkl`, and a reopened SAC
archive containing exactly 5,000,016 transitions.  Checkpoints do not contain
the replay buffer and do not authorize resume.

## 5. One-shot detached execution and absence of monitoring

The public runner exposes read-only `plan` and `status` plus one mutating
`launch`.  It has no seed, budget, device, entropy, curriculum, resume, retry,
or overwrite switches.  Launch performs the bounded source, runtime, Numba,
test, and disk preflight; commits a retained one-shot lock; then starts one
worker under `/usr/bin/caffeinate -dimsu` in a new session with stdin detached
and stdout/stderr written exclusively to the run console.

The launcher waits at most 15 seconds for the worker-started receipt and then
returns.  There is no periodic external monitor, watchdog, automatic restart,
retry, resume, selective stopping rule, or two-hour inspection.  `caffeinate`
prevents ordinary idle sleep only; it does not survive reboot, power loss, or a
hard process kill.
