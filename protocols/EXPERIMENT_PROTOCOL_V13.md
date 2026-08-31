# EXPERIMENT PROTOCOL V13 — certified non-negative information-ray gain

Status: frozen before the first scientific v13 pilot.

## 1. Motivation and scientific scope

V12 restored `pid_track_exc` as the inner stabilizing controller and let SAC
add a bounded three-axis residual. On paired dev100 scenarios, v12 retained
stability relative to `pid_track_exc` but failed the prespecified information
gate:

- terminal success was 82% for v12 and 80% for `pid_track_exc`;
- tail80 success was 83% and 78%, respectively;
- terminal online minimum FIM eigenvalue was `3.31e-5` for v12 and `7.61e-5`
  for `pid_track_exc`;
- the residual/planner cosine was -0.191 on the inspected hard trace and 56.3%
  of those residuals had a negative dot product with the planner direction.

V13 changes the learned action geometry and adds an online eligibility
certificate for the resulting ray. The learned controller may choose only a
non-negative gain on a certified online information-planner ray. It cannot
command a component opposite or orthogonal to that ray. Dynamics, sensing,
PF, reward, observation contents, PID+exc implementation, online planner,
success definitions, RNG streams, fixed horizon, curriculum and noise-tape
pairing remain those of the frozen v12 experiment.

The v12 and earlier source files, models and reports are immutable. V13 is
implemented in new files and all v13 learned policies train from scratch.

## 2. Frozen controller contract

The SAC policy action is a scalar gain:

    lambda_t in [0, 1]

The policy action space has dimension one. The plant action retains dimension
three and the normalized channel order:

    [speed increment, yaw increment, pitch increment]

Let the valid online planner action be `p_t`, and let

    S = diag(0.15, 0.30, 0.25)
    d_t = S p_t

The planner vector is not normalized. This preserves the v12 per-axis maximum
authority and preserves the planner's own amplitude. The v12 gates are
unchanged:

    g_t = g_track_t g_info_t
    r_t = g_t lambda_t d_t
    a_t = clip(a_pid_track_exc_t + r_t, -1, 1)

The ray is enabled only by an online certificate. It requires a finite,
non-zero planner action inside `[-1,1]^3`, an active planner, planner gate at
least `info_planner_gate_min`, best-versus-zero score margin greater than
`1e-6`, positive predicted trace reduction, and predicted formation error no
greater than the online position tolerance. Every certificate input is derived
online and every failure bit is logged.

If the certificate fails, `r_t` is exactly zero. If either authority gate is
zero, `r_t` is exactly zero. At estimated formation-error ratio `q >= 1`,
`g_track` and the learned contribution are exactly zero. Gain zero must
reproduce `pid_track_exc` bit for bit under the same initial state and
exogenous noise tape.

The historical action fields in the observation continue to contain the
three-dimensional action actually applied to the plant. They do not contain
the scalar gain. The v11 full-online reward is evaluated on the composite
plant action and is otherwise unchanged.

The manifest action contract contains at least:

- `policy_action_dim: 1`;
- `plant_action_dim: 3`;
- `action_contract.policy_action_low: [0.0]`;
- `action_contract.policy_action_high: [1.0]`;
- `action_contract.plant_action_dim: 3`.

SAC target entropy remains configured as `auto`; because policy dimension is
one, its resolved target differs from the three-axis v12 policy. This fact and
the entropy coefficient history are mandatory training diagnostics. No v12
model, replay buffer or optimizer state may initialize v13.

## 3. Online-only boundary

The inner controller and ray composer receive only the explicit online
allowlist established in v12: PF estimate, desired formation, measured/dead-
reckoned follower kinematics, communicated leader motion, online PF spread,
online tolerances, online planner action and online planner gate. Truth
positions, truth errors, oracle FIM/CRLB, truth follower kinematics and legacy
oracle aliases are forbidden. Missing required data fails closed.

The composer is deterministic and consumes no RNG. Baseline controllers are
applied directly; the evaluator must never compose PID+exc twice.

## 4. Logging and alignment invariants

Every learned-policy step records:

- scalar policy gain;
- planner action and scaled ray `d_t`;
- planner-certificate mask/reasons and ray-eligible flag;
- tracking, information and combined gates;
- inner PID+exc action;
- intended residual before plant clipping;
- pre-clipped and applied plant actions;
- realized delta `applied - inner` and per-channel clipping flags;
- dot product, cosine and effective coefficient along the ray;
- orthogonal residual norm and alignment-violation flag.

Alignment is evaluated on every dev100 step, not on a selected training trace.
Zero-ray and zero-gain rows are reported separately rather than assigned an
arbitrary cosine. On eligible non-zero pre-clipping rows:

    dot(r_t, d_t) >= 0
    ||r_t - dot(r_t,d_t)/dot(d_t,d_t) d_t|| / max(||r_t||, eps) <= 1e-6

The total alignment-violation count must be zero. Post-clipping realized
alignment and saturation remain separate diagnostics: clipping can reduce or
rotate the realized increment even when the intended residual is a valid ray.

Both unconditional and eligible-step-conditional gain/activity summaries are
reported: mean, standard deviation, p05, median, p95, zero fraction, high-gain
fraction, non-zero authority fraction, planner availability and gain temporal
variation. These are stratified by information gate and formation-error ratio.

## 5. Smoke and frozen training matrix

The non-scientific end-to-end smoke is fixed at 4,096 samples, seed 21999, two
vector environments, PF64, one-frame history and a `64,64` network. It checks
training updates, serialization, manifest validation, model reload, evaluation
and action traces. Its performance is never interpreted.

Smoke uses a separate temporary/output root and must never create any path
inside `experiments_v13_info_ray_24env`. The runner refuses such a request.

Scientific settings match v12:

- SAC and 24 vector environments;
- `512,512,512` ReLU network;
- PF1024 with Numba and warm-up;
- four-frame observation history;
- action interval 2 s and FIM window 30 s;
- batch 256, one gradient step per vector call and learning starts at 100;
- entropy coefficient `auto_0.2`, target entropy `auto`;
- 90% curriculum ramp and replay reset at difficulty 0.80.

Frozen scientific runs:

1. Pilot: variant `info_ray_gain`, seed 22001, 1,000,000 samples.
2. Production, only after a valid GO: seeds 23001–23005, 25,000,000 samples
   each.

Resume and overwrite are disabled. `--execute pilot` accepts only seed 22001
and exactly 1,000,000 samples. A shortened pilot budget is a dry preview only;
the executable short check is the isolated smoke phase.

The inner `v13_run_manifest.json` uses schema 3 and status `completed`. After a
subprocess returns zero, the runner independently validates version
`v13_info_ray_1.0`, variant, seed, scalar/plant action dimensions, gain bounds,
and SHA-256 values of `final_model.zip` and `vecnormalize.pkl`. Return code zero
without a valid inner manifest marks the outer run failed.

## 6. Development evaluation and comparator policy

The v12 development scenarios 40000–40099 informed the v13 design and are not
reused for a confirmatory decision. The frozen v13 dev100 set is:

- 100 episodes, seeds 41000–41099;
- difficulty 1.0, 220 steps and 440 s;
- deterministic learned policies;
- one saved exogenous noise tape per scenario;
- at least 5,000 paired bootstrap resamples.

Every candidate episode is paired by initial-state and noise-tape hashes with:

1. direct `pid_track_exc`, the confirmatory reference;
2. the frozen v12 1M `hybrid_pid_exc_rl` model;
3. the frozen v11 `full_online` model.

The frozen models use their own matching VecNormalize artifacts and versioned
evaluators. Cross-version results are interpreted only after all initial-state
and exogenous-tape hashes match. PID+exc is the confirmatory reference. V12 is
the mechanistic ablation comparator. V11 is the historical high-information,
lower-retention Pareto point; v13 is not required to match its FIM to pass the
pilot.

The final 1,000 scenarios, seeds 50000–50999, remain untouched until the v13
architecture and all five production training seeds are frozen.

## 7. Confirmatory go/no-go gate

All confirmatory controller contrasts are paired differences `v13 RL -
pid_track_exc`. The gate is conjunctive; no endpoint compensates for another.
V13 advances only if:

1. all action-contract, online-boundary and regression tests pass, and v12 and
   earlier hashes remain unchanged;
2. no NaN, Inf, manifest, action-range or pairing violation occurs;
3. zero gain reproduces direct `pid_track_exc` bit for bit;
4. the lower paired 95% CI for terminal-success difference is at least -0.05;
5. the lower paired 95% CI for tail80-success difference is at least -0.05;
6. the lower paired 95% CI for
   `fim_online_win_eig_min_terminal` is strictly greater than zero;
7. the upper paired 95% CI for terminal estimated formation-error ratio
   difference is no greater than +0.10;
8. alignment-violation count over all v13 dev100 rows is zero;
9. applied-action saturation step fraction is no greater than 0.01;
10. PID+exc, frozen v12 and frozen v11 comparisons are all present with
    matching scenario/tape hashes;
11. all mandatory supporting and PF-consistency metrics are reported.

The continuous formation-error guardrail was added before the v13 pilot
because v12's binary success rates hid a large terminal mean error increase.
The primary information endpoint is retained unchanged to avoid outcome
switching after the v12 result.

All conditions are required, so the confirmatory gate uses an intersection-
union/conjunctive rule. Supporting families promoted to confirmatory language
require prespecified Holm adjustment.

## 8. Auditable decision file

Production is fail-closed unless
`<output-root>/v13_go_no_go_decision.json` exists and validates. Required
schema:

```json
{
  "schema_version": 1,
  "decision_version": "v13_go_no_go_1.0",
  "environment_version": "v13_info_ray_1.0",
  "variant": "info_ray_gain",
  "passed": true,
  "recorded_at_utc": "YYYY-MM-DDTHH:MM:SS+00:00",
  "approved_by": "responsible reviewer",
  "pilot": {
    "seed": 22001,
    "model_relative_path": "pilot/info_ray_gain/seed_22001/models/final_model.zip",
    "model_sha256": "<64 lowercase hexadecimal characters>"
  },
  "development_evaluation": {
    "mode": "dev",
    "episodes": 100,
    "seed_first": 41000,
    "seed_last": 41099,
    "comparators": [
      "pid_track_exc",
      "v12_hybrid_pid_exc_rl",
      "v11_full_online"
    ],
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
    "primary_information_paired_ci_lower": 0.000001,
    "formation_error_ratio_endpoint": "formation_error_ratio_terminal",
    "terminal_formation_error_ratio_paired_ci_upper": 0.08,
    "alignment_violation_count": 0,
    "action_saturation_step_fraction": 0.005,
    "multiplicity_rule": "conjunctive_gate_secondary_holm"
  }
}
```

Numerical CI values in the example are illustrative. The recorded file must
contain actual dev100 outcomes and exact SHA-256 values. Relative paths may not
escape the output root. A missing, malformed, false, stale or hash-mismatched
decision prevents every 25M run.

## 9. Mandatory supporting results and interpretation limits

The report includes dwell success, corridor occupancy, maximum outage,
trajectory and terminal formation error, localization RMSE, raw PF spread,
95% coverage, NEES, time-mean FIM, CRLB trace, planner score/margin, gain
authority, control RMS/integral, clipping and saturation. Per-episode values,
paired differences and uncertainty intervals remain available; aggregate means
alone are insufficient.

PF inconsistency is reported explicitly and cannot be used to waive a failed
information gate. A positive control-space dot product is a structural action
property, not proof of increased physical information. Only the observed,
paired online FIM endpoint can satisfy the information criterion.

The one-dimensional policy has known limitations: it can collapse to zero
gain, cannot choose an alternative direction when the planner is myopic or
noisy, mixes gain with planner-vector magnitude, may double existing PID+exc
excitation, and may be distorted by actuator clipping. Gate activity,
entropy, planner availability, realized alignment and saturation therefore
remain mandatory diagnostics, not optional plots.

## 10. Production and final reporting

After a valid GO, all five frozen 25M seeds run. No seed is selected using
dev100. Every seed and SHA-256 is retained and all five models advance to the
untouched final1000 evaluation. The primary aggregate gives each training seed
equal weight and uses a hierarchical bootstrap over training seeds and paired
scenarios. Per-seed estimates remain visible so training instability cannot be
hidden by pooling.
