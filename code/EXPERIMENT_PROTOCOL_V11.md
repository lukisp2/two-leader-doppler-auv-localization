# V11 online-only experimental protocol

Status: implementation/pilot protocol, frozen before production training.

## 1. Scientific contract

The primary method is `v11_online`, not the historical privileged v10 policy.

- Actor observation, guard, planner, reward, and primary success use quantities available online.
- Ground-truth follower position and oracle FIM are logging-only diagnostics.
- Historical reward/actor metrics and the predictive planner share the same online Doppler Jacobian, soft gate, and noise convention. They are deliberately distinct surrogates: reward uses the historical mean-geometry window `(t-T,t]`, while the planner propagates PF support points over a short future horizon.
- `pf_std_max_raw = sqrt(lambda_max(P_PF))` is reported separately from the explicitly named `sigma_eff_hybrid_diagnostic`.
- The regularized inverse of the windowed FIM is called an online conditional-geometry CRLB surrogate, not unconditional PF posterior uncertainty.
- Episodes always run for 440 s. Reaching a corridor does not terminate an episode.

## 2. Frozen endpoint definitions

Hard-regime thresholds are 8 m estimated formation error and 7 m raw PF largest-eigenvalue standard deviation.

Primary endpoints, in order:

1. Terminal online success at 440 s: both thresholds are satisfied at the final action step.
2. Continuous 30 s dwell success: both thresholds are satisfied for at least 15 consecutive 2 s action steps.
3. Final-100-s occupancy success: both thresholds are satisfied for at least 80% of the final 50 action steps.

Secondary endpoints:

- time fraction inside the online corridor;
- final and trajectory RMSE of `||p_hat - p_true||` (evaluation-only truth diagnostic);
- raw PF largest-eigenvalue standard deviation;
- NEES and empirical 95% ellipsoid coverage;
- online windowed FIM minimum eigenvalue and conditional-geometry CRLB-surrogate trace;
- information-outage duration, final formation error, return, and control effort.

`ever success` may be logged only as a diagnostic and may not be used as the headline success rate.

## 3. Randomness and test split

Every episode has independent named PCG64 streams for scenario, Doppler sensor, dead reckoning, particle filter, and controller. Final paired evaluation uses one saved exogenous-noise tape per scenario. All controllers receive the same scenario, Doppler errors, and dead-reckoning errors; PF and controller-internal draws never advance those streams.

- Development/pilot seed: `20001`.
- Production training seeds: `21001, 21002, 21003, 21004, 21005`.
- Development evaluation seeds: `40000..40999`; these may be used for pilot go/no-go and tuning reports.
- Frozen final evaluation scenario seeds: `50000..50999` (1000 scenarios).
- Final evaluation seeds must not be used for tuning, checkpoint selection, or debugging.

The test set is distinct from the old v10 training seed and old evaluation range `42..1041`.

## 4. Training matrix

All production policies use the same architecture, action limits, curriculum, environment horizon, optimizer settings, CPU device, and nominal 25 million environment steps unless the pilot demonstrates a documented numerical failure.

The resource-calibrated vectorization is 24 environments with one SAC gradient step per vector step. Its update/sample ratio `1/24 = 0.0417` closely matches the historical 128-environment run with five gradient steps, `5/128 = 0.0391`, while avoiding memory/swap exhaustion on the 10-core, 24-GiB development Mac. A 128-environment startup attempted on 2026-07-10 was stopped before learning began after memory pressure exceeded the safe limit; its interrupted manifest is retained.

Required runs:

| Variant | Independent training seeds | Purpose |
|---|---:|---|
| full online-only | 5 | primary method |
| no FIM/CRLB reward | 3 | isolate explicit information shaping |
| no planner | 3 | isolate planner guidance |
| no guard | 3 | isolate guarded optimization |
| tracking-only SAC | 3 | architecture-matched learned baseline |

The old v10 privileged checkpoint is an oracle/legacy reference only and is not a deployable main method.

## 5. Non-learning baselines

- restored `pid_track`;
- restored `pid_track_exc`, with raw `std_max_eff` in v11 (because conservative inflation is disabled);
- new `planner_only`, clearly labelled as a v11 baseline;
- random bounded actions as a negative control.

The exact restored-source provenance is recorded in `BASELINE_PROVENANCE_V11.md`.

## 6. Statistical analysis

- Scenario-level comparisons are paired by frozen scenario ID and exogenous-noise tape.
- Binary endpoints: paired risk difference with bootstrap confidence interval and McNemar test.
- Continuous endpoints: paired median/mean differences with bootstrap confidence intervals; Wilcoxon is secondary.
- Multiplicity: Holm correction within each declared endpoint family.
- Training randomness is represented by independent checkpoints. Scenario repetitions from one checkpoint must not be presented as independent training replications.
- Report per-training-seed results and a hierarchical aggregate over training seed and scenario.

## 7. Go/no-go gates

Before any 25M run:

1. all unit and contract tests pass;
2. a short CLI smoke training saves a loadable model and VecNormalize state;
3. one pilot policy reaches the hard regime without NaN/Inf and improves at least one terminal/dwell endpoint over tracking-only SAC;
4. reward-invariance tests prove that changing truth position or oracle FIM does not change actor observation or reward;
5. paired-noise tests prove that extra PF draws and different controller actions do not change later exogenous samples.

If a gate fails, fix the implementation or objective before scaling compute. Do not tune on final evaluation seeds.

## 8. Reproducibility artifacts

Each production run must save:

- source commit/diff and `V10_ARCHIVE_MANIFEST.json`;
- complete dataclass/CLI configuration and package versions;
- model, normalization state, checkpoints, and training curves;
- all Python/NumPy/PyTorch/SB3 seeds;
- final evaluation trace summaries and the shared noise-tape digest;
- executable commands used for training, evaluation, tables, and figures.

Exact interrupted-run resume is disabled until replay-buffer and complete environment/curriculum state restoration are implemented and tested. A checkpoint may be evaluated, but it must not be presented as an exact continuation of a production run.

## 9. Manuscript rule

Do not update the abstract's numerical claims until the frozen 1000-scenario evaluation is complete. The manuscript must distinguish raw PF covariance, actual localization error, the hybrid diagnostic, and the conditional-geometry CRLB surrogate in every table and figure.
