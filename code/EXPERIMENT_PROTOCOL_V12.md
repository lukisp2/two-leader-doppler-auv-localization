# EXPERIMENT PROTOCOL V12 — PID+exc inner loop with RL information residual

Status: frozen before the first scientific 1M-sample v12 pilot.

## 1. Motivation and scope

The v11 dev100 experiment showed a repeatable trade-off:

- full_online RL found the online success corridor in 99% of scenarios and
  improved the online FIM/CRLB surrogate;
- restored pid_track_exc achieved higher terminal and tail stability;
- direct RL therefore acquired information well but did not retain formation
  as reliably as the classical inner loop.

V12 changes exactly one scientific mechanism: the action architecture. The
v11 observation, reward, estimator, success definitions, dynamics, RNG streams,
fixed horizon and noise-tape protocol remain unchanged. PF calibration changes
are explicitly out of scope for this pilot.

The v11 source files and their trained models remain immutable. V12 is
implemented only in new files and starts training from scratch.

## 2. Controller architecture

The SAC policy outputs a normalized residual:

    r_t = clip(policy(o_t), -1, 1)

The restored online-only inner action is:

    a_pid_t = pid_track_exc(online_snapshot_t)

The estimated formation-error ratio is:

    q_t = ||pF_des_t - pF_hat_t|| / tol_pos_t

The tracking-safety gate is:

    g_track = 1 - smoothstep((q_t - 0.50) / (1.00 - 0.50))

Consequently, RL has full tracking authority only for q <= 0.50, loses it
smoothly inside the success corridor, and has exactly zero authority for
q >= 1.00.

The information-need gate is:

    g_info = max(info_plan_gate, clip((PF_std_raw - tol_std) / tol_std, 0, 1))

The per-axis normalized residual scale is [0.15, 0.30, 0.25] for speed, yaw
and pitch.

The plant action is:

    a_t = clip(a_pid_t + g_track * g_info * scale * r_t, -1, 1)

At action_dt=2 s, the maximum residual contribution corresponds to:

- speed increment: 0.06 m/s per action step;
- yaw increment: 6 degrees per action step;
- pitch increment: 3.5 degrees per action step.

The observation's historical action slots retain the v11 meaning of the action
actually applied to the simulated vehicle. Trace artifacts additionally store
the raw residual, PID action, both gates, effective residual, pre-clipped action,
final action and per-axis clipping flags.

The v11 full-online reward is evaluated on the applied composite action and is
otherwise unchanged. No residual-specific reward is introduced in the first
pilot.

## 3. Online-only boundary

The inner controller receives an explicit allowlist:

- PF estimate and desired formation position;
- follower speed, yaw and pitch only from explicit dead-reckoning fields;
- leader speeds and headings available through the cooperative link;
- online PF uncertainty and online tolerances;
- cached online information-planner action and gate.

Truth positions, truth follower kinematics, oracle FIM/CRLB, truth errors,
diagnostic consistency metrics and legacy oracle aliases are never forwarded.
Missing required online fields fail closed.

The composer is deterministic and consumes no RNG. The fallback excitation in
the restored PID uses only the within-episode step_count.

## 4. Per-controller action mode in evaluation

- learned candidate rl: residual input, internal pid_track_exc enabled;
- pid_track, pid_track_exc, planner_only and random: direct action, internal
  hybrid composition disabled.

This rule prevents accidental PID + PID composition for the reference
baseline. All controllers still share the same physical initial state and one
saved exogenous noise tape per episode.

## 5. Training matrix

Hardware-calibrated settings remain those of v11:

- SAC, 24 vector environments;
- 512 x 512 x 512 ReLU network;
- 1 gradient step per vector call;
- action interval 2 s;
- PF 1024 particles with Numba;
- four-frame observation history;
- 90% curriculum ramp and replay reset at difficulty 0.80.

Frozen runs:

1. Pilot: variant hybrid_pid_exc_rl, seed 20001, 1,000,000 samples.
2. Production (only after the pilot gate): seeds 21001–21005,
   25,000,000 samples each.

`--execute pilot` accepts only seed 20001 and exactly 1,000,000 samples. A
shortened `--pilot-timesteps` value is permitted only for a clearly marked dry
preview. Smoke training is non-scientific, must use a separate temporary output
tree, and must never create or occupy the frozen `pilot/.../seed_20001`
directory.

Every run uses a new isolated output directory. Resume is disabled. The run
manifest uses schema 2 and must contain:

- exact model and VecNormalize SHA-256;
- all source SHA-256 values;
- full environment configuration;
- the complete hybrid action contract;
- observation and action dimensions;
- source snapshot.

A subprocess return code of zero is not sufficient for completion. The outer
runner marks a run completed only after `models/v12_run_manifest.json` has
schema 2, status `completed`, the expected environment version, variant and
seed, and matching SHA-256 values for both `final_model.zip` and
`vecnormalize.pkl`. Missing or inconsistent inner artifacts make the outer run
failed.

## 6. Evaluation

Development:

- 100 fixed scenarios, seeds 40000–40099;
- controllers in frozen order: rl, pid_track, pid_track_exc, planner_only,
  random;
- deterministic SAC;
- difficulty 1.0, 220 steps, 440 s;
- shared saved noise tapes;
- paired bootstrap with at least 5000 resamples.

The restored v11 `full_online` learned policy is a mandatory historical
comparator. It is evaluated with the frozen v11 evaluator on the same scenario
seeds and equivalent saved exogenous noise tapes. Initial-state and tape
SHA-256 values must match the corresponding v12 episodes before comparisons
are interpreted.

Confirmatory stability endpoints:

- terminal success;
- tail80 success over the last 100 s.

The single prespecified primary information endpoint is
`fim_online_win_eig_min_terminal`; higher is better. The confirmatory pilot
contrast is hybrid RL minus `pid_track_exc` on paired dev100 scenarios.

Supporting endpoints, which do not independently pass or fail the pilot gate,
are 30 s dwell success, corridor occupancy, maximum outage, terminal and
trajectory formation error, localization RMSE, raw PF spread, coverage, NEES,
the remaining online FIM/CRLB summaries, residual authority, saturation rate
and applied control effort. They remain mandatory to report.

The gate is conjunctive: both stability non-inferiority tests and the one
primary information-superiority test must pass. Therefore no endpoint may
compensate for failure of another. The single primary information endpoint is
not selected from a family after seeing results. Supporting endpoints are
exploratory; if any family of them is promoted to confirmatory language, Holm
adjustment is applied within that explicitly declared family. This rule is
recorded as `conjunctive_gate_secondary_holm`.

Return is not evidence of superiority across architectures.

The final 1000-scenario set, seeds 50000–50999, remains untouched until the
architecture and all training seeds are frozen.

## 7. Pilot go/no-go gate

All pilot contrasts below use paired differences `hybrid RL - pid_track_exc`
and 95% paired percentile-bootstrap confidence intervals with at least 5000
resamples. The pilot may advance to multi-seed training only if:

1. all contract/regression tests pass and v11 archive hashes remain unchanged;
2. no NaN/Inf or action-contract violation occurs;
3. zero residual reproduces the restored pid_track_exc trajectory;
4. the lower 95% CI bound for terminal-success risk difference is at least
   -0.05;
5. the lower 95% CI bound for tail80-success risk difference is at least
   -0.05;
6. the lower 95% CI bound for the primary information endpoint
   `fim_online_win_eig_min_terminal` is strictly greater than zero;
7. the frozen v11 `full_online` comparison is complete on the same dev100
   scenarios and matching initial-state/noise-tape hashes;
8. PF consistency pathologies and every supporting stability endpoint are
   reported, not hidden by terminal-only metrics.

Passing the gate must be recorded in
`<output-root>/v12_go_no_go_decision.json`. Production execution is fail-closed
if this file is missing, malformed, says `passed=false`, references a different
pilot model, or does not hash-match its pilot model and dev100 report. The
auditable schema is:

```json
{
  "schema_version": 1,
  "decision_version": "v12_go_no_go_1.0",
  "environment_version": "v12_hybrid_1.0",
  "variant": "hybrid_pid_exc_rl",
  "passed": true,
  "recorded_at_utc": "YYYY-MM-DDTHH:MM:SS+00:00",
  "approved_by": "responsible reviewer",
  "pilot": {
    "seed": 20001,
    "model_relative_path": "pilot/hybrid_pid_exc_rl/seed_20001/models/final_model.zip",
    "model_sha256": "<64 lowercase hexadecimal characters>"
  },
  "development_evaluation": {
    "mode": "dev",
    "episodes": 100,
    "seed_first": 40000,
    "seed_last": 40099,
    "comparators": ["pid_track_exc", "v11_full_online"],
    "report_relative_path": "decisions/dev100_gate_report.json",
    "report_sha256": "<64 lowercase hexadecimal characters>"
  },
  "criteria": {
    "candidate_controller": "rl",
    "reference_controller": "pid_track_exc",
    "confidence_level": 0.95,
    "paired_bootstrap_samples": 5000,
    "terminal_paired_ci_lower": -0.04,
    "tail80_paired_ci_lower": -0.03,
    "primary_information_endpoint": "fim_online_win_eig_min_terminal",
    "primary_information_direction": "higher",
    "primary_information_paired_ci_lower": 0.001,
    "multiplicity_rule": "conjunctive_gate_secondary_holm"
  }
}
```

The numerical CI values above are illustrative; the decision file contains the
actual dev100 results. Relative artifact paths must remain inside the output
root.

Failure of the gate leads to a new explicitly named v12 ablation. It does not
authorize tuning on the final holdout.

## 8. Five-seed production and final reporting rule

After a valid pilot decision, all five frozen production seeds are trained. No
"best" seed is selected from dev100. Every completed seed is evaluated and
reported individually, and all five frozen model SHA-256 values are retained.

All five models advance to the untouched final1000 evaluation. The primary
cross-seed estimate gives each training seed equal weight and is accompanied by
a hierarchical bootstrap that resamples training seeds and then matched
scenarios within seed. Per-seed paired estimates and confidence intervals are
reported beside the aggregate, so training instability cannot be hidden by
pooling scenario rows. A deployment-model selection rule, if later needed,
must be specified in a separate protocol and cannot use final1000 outcomes.
