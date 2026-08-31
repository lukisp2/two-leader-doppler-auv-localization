from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import run_v24_audited_gate as runner24
import uuv_v19_observability as v19
import uuv_v21_causal_lock as v21
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24


def _mode(
    position=(0.0, 0.0, 0.0),
    *,
    radius95=2.0,
    rmse=0.03,
) -> v19.BatchMode:
    return v19.BatchMode(
        initial_position_m=np.asarray(position, dtype=np.float64),
        residual_sse_mps2=0.1,
        residual_rmse_mps=float(rmse),
        iterations=4,
        converged=True,
        hessian_eigenvalues=np.asarray([1.0, 2.0, 3.0]),
        hessian_rank=3,
        hessian_condition_number=3.0,
        local_covariance_m2=np.eye(3),
        local_covariance_valid=True,
        local_radius95_m=float(radius95),
    )


def _estimate(
    position=(0.0, 0.0, 0.0),
    *,
    radius95=2.0,
    rmse=0.03,
) -> v19.BatchEstimate:
    return v19.BatchEstimate(
        modes=(_mode(position, radius95=radius95, rmse=rmse),),
        runtime_s=0.01,
        coarse_best_rmse_mps=float(rmse),
        candidate_count=128,
        refined_start_count=4,
        clustered_mode_count=1,
        alternative_mode_index=None,
        alternative_distance_m=None,
        alternative_delta_sse_mps2=None,
        alternative_delta_chi2=None,
    )


def _evidence(
    *,
    time_s=60.0,
    primary=(0.0, 0.0, 0.0),
    confirmation=(0.1, 0.0, 0.0),
    primary_radius=2.0,
    confirmation_radius=2.0,
    primary_rmse=0.03,
    confirmation_rmse=0.03,
) -> v21.GlobalLockEvidence:
    primary_estimate = _estimate(
        primary, radius95=primary_radius, rmse=primary_rmse
    )
    confirmation_estimate = _estimate(
        confirmation,
        radius95=confirmation_radius,
        rmse=confirmation_rmse,
    )
    return v21.GlobalLockEvidence(
        time_s=float(time_s),
        primary=primary_estimate,
        confirmation=confirmation_estimate,
        search_agreement_m=float(
            np.linalg.norm(
                primary_estimate.best.initial_position_m
                - confirmation_estimate.best.initial_position_m
            )
        ),
        stability_from_previous_m=0.5,
        forward_prediction_rmse_mps=0.03,
        forward_prediction_sample_count=30,
    )


_DEFAULT_EVIDENCE = object()


def _evaluate(
    gate: v24.AuditedCausalLockGate,
    now_s: float,
    *,
    position=(0.0, 0.0, 0.0),
    radius95=2.0,
    evidence: v21.GlobalLockEvidence | None | object = _DEFAULT_EVIDENCE,
) -> bool:
    mode = _mode(position, radius95=radius95)
    return gate.evaluate(
        now_s=now_s,
        mode=mode,
        initial_position_m=position,
        recent_residual_rmse_mps=0.03,
        global_evidence=(
            _evidence() if evidence is _DEFAULT_EVIDENCE else evidence
        ),
    )


def test_release_requires_three_complete_audited_passes() -> None:
    gate = v24.AuditedCausalLockGate(v24.AuditedLockConfig())
    for now in (60.0, 62.0, 64.0, 66.0):
        assert not _evaluate(gate, now)
    assert _evaluate(gate, 68.0)
    assert gate.state.lock_count == 1
    assert gate.last_release_checks["local_primary_agreement"]
    assert gate.last_release_checks["confirmation_radius95"]


@pytest.mark.parametrize(
    ("evidence", "failed"),
    [
        (_evidence(primary=(2.1, 0.0, 0.0), confirmation=(0.2, 0.0, 0.0)), "local_primary_agreement"),
        (_evidence(primary=(0.2, 0.0, 0.0), confirmation=(2.1, 0.0, 0.0)), "local_confirmation_agreement"),
        (_evidence(primary_radius=7.01), "primary_radius95"),
        (_evidence(confirmation_radius=7.01), "confirmation_radius95"),
        (_evidence(primary_rmse=0.081), "primary_rmse"),
        (_evidence(confirmation_rmse=0.081), "confirmation_rmse"),
    ],
)
def test_each_added_release_check_can_block_lock(
    evidence: v21.GlobalLockEvidence, failed: str
) -> None:
    gate = v24.AuditedCausalLockGate(v24.AuditedLockConfig())
    for now in (60.0, 62.0, 64.0, 66.0):
        assert not _evaluate(gate, now, evidence=evidence)
    assert failed in gate.state.last_failed_checks


def test_stale_global_evidence_blocks_release() -> None:
    gate = v24.AuditedCausalLockGate(v24.AuditedLockConfig())
    evidence = _evidence(time_s=0.0)
    for now in (66.0, 68.0, 70.0):
        assert not _evaluate(gate, now, evidence=evidence)
    assert "global_fresh" in gate.state.last_failed_checks


def test_material_hold_disagreement_unlocks_immediately_and_forces_refresh() -> None:
    gate = v24.AuditedCausalLockGate(v24.AuditedLockConfig())
    for now in (60.0, 62.0, 64.0, 66.0, 68.0):
        _evaluate(gate, now)
    assert gate.is_locked
    assert not _evaluate(
        gate,
        70.0,
        position=(8.0, 0.0, 0.0),
        evidence=_evidence(time_s=60.0),
    )
    assert gate.state.unlock_count == 1
    assert gate.consume_force_global_refresh()


def test_ordinary_hold_failure_retains_three_action_hysteresis() -> None:
    gate = v24.AuditedCausalLockGate(v24.AuditedLockConfig())
    for now in (60.0, 62.0, 64.0, 66.0, 68.0):
        _evaluate(gate, now)
    assert _evaluate(gate, 70.0, radius95=11.0)
    assert _evaluate(gate, 72.0, radius95=11.0)
    assert not _evaluate(gate, 74.0, radius95=11.0)


def test_missing_hold_evidence_uses_hysteresis_not_material_unlock() -> None:
    gate = v24.AuditedCausalLockGate(v24.AuditedLockConfig())
    for now in (60.0, 62.0, 64.0, 66.0, 68.0):
        _evaluate(gate, now)
    assert gate.is_locked
    assert _evaluate(gate, 70.0, evidence=None)
    assert _evaluate(gate, 72.0, evidence=None)
    assert not _evaluate(gate, 74.0, evidence=None)


def test_gate_rejects_position_that_is_not_the_supplied_mode() -> None:
    gate = v24.AuditedCausalLockGate(v24.AuditedLockConfig())
    with pytest.raises(ValueError, match="differs"):
        gate.evaluate(
            now_s=60.0,
            mode=_mode((0.0, 0.0, 0.0)),
            initial_position_m=(0.0, 0.0, 0.1),
            recent_residual_rmse_mps=0.03,
            global_evidence=_evidence(),
        )


def test_exact_trace_score_uses_transition_track_start_and_track_end_indices() -> None:
    trace = {
        "gate_locked_after_update": np.asarray([0, 1, 1, 0, 1, 1]),
        "phase_track": np.asarray([0, 0, 1, 1, 0, 1]),
        "localization_error_m": np.asarray([1.0, 2.0, 3.0, 8.0, 6.0, 7.0]),
        "time_s": np.asarray([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        "action_start_time_s": np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
    }
    score = v24.exact_trace_score(trace)
    assert score["transition_count"] == 2
    assert score["first_transition_time_s"] == 4.0
    assert score["first_transition_error_m"] == 2.0
    assert score["first_locked_action_start_error_m"] == 2.0
    assert score["first_locked_action_end_error_m"] == 3.0
    assert score["false_transition_count"] == 0
    assert score["false_locked_action_start_count"] == 0
    assert score["false_locked_action_end_count"] == 2


def test_exact_seven_and_nonfinite_are_unsafe() -> None:
    trace = {
        "gate_locked_after_update": np.asarray([1, 1]),
        "phase_track": np.asarray([0, 1]),
        "localization_error_m": np.asarray([7.0, np.nan]),
        "time_s": np.asarray([2.0, 4.0]),
        "action_start_time_s": np.asarray([0.0, 2.0]),
    }
    score = v24.exact_trace_score(trace)
    assert score["false_transition_count"] == 1
    assert score["false_locked_action_start_count"] == 1
    assert score["false_locked_action_end_count"] == 1


def _decision_metrics(rate: float, episodes: int = 4) -> dict:
    return {
        "episodes": episodes,
        "ever_locked_rate": rate,
        "terminal_joint_success_rate": rate,
        "unsafe_transition_count": 0,
        "unsafe_track_start_count": 0,
        "unsafe_track_end_count": 0,
        "maximum_combined_decision_runtime_s": 0.5,
        "audit_release_violation_count": 0,
    }


def test_decision_uses_rates_not_absolute_counts() -> None:
    paired = {"v24_minus_v23_transition_delay_s": {"p50": 0.0}}
    by_arm = {
        runner24.ARM_NAMES[0]: _decision_metrics(1.0),
        runner24.ARM_NAMES[1]: _decision_metrics(0.75),
    }
    decision, checks = runner24._decision(
        by_arm,
        paired,
        complete_design=True,
        smoke=False,
        pairing_mismatch_count=0,
    )
    assert decision == "NO_SUPPORT_V24_AUDITED_GATE"
    assert not checks["ever_lock_rate"]
    by_arm[runner24.ARM_NAMES[1]] = _decision_metrics(1.0)
    decision, checks = runner24._decision(
        by_arm,
        paired,
        complete_design=True,
        smoke=False,
        pairing_mismatch_count=0,
    )
    assert decision == "SUPPORT_V24_AUDITED_GATE"
    assert all(checks.values())


def test_manifest_contains_transitive_runner_dependencies() -> None:
    manifest = runner24._loaded_local_sources(runner24.Path(runner24.__file__).parent)
    for required in (
        "run_v20_positioning_ablation.py",
        "run_v21_causal_lock.py",
        "run_v22_active_acquisition.py",
        "uuv_v24_audited_gate.py",
    ):
        assert required in manifest


def test_runner_refuses_sealed_final_seed() -> None:
    args = runner24._parse(["--seed-start", "50000", "--episodes", "1"])
    with pytest.raises(PermissionError):
        runner24._settings(args)


def test_resume_identity_validator_rejects_wrong_seed() -> None:
    trace = {
        name: np.zeros(3)
        for name in (
            "time_s",
            "phase_track",
            "gate_locked_after_update",
            "localization_error_m",
        )
    }
    outcome = v22.ArmOutcome(
        summary={
            "version": v24.VERSION,
            "arm": runner24.ARM_NAMES[0],
            "episode_index": 0,
            "episode_seed": 123,
            "noise_tape_sha256": "abc",
            "action_count": 3,
        },
        trace=trace,
    )
    with pytest.raises(RuntimeError, match="episode_seed"):
        runner24._validate_outcome_identity(
            outcome,
            arm=runner24.ARM_NAMES[0],
            episode_index=0,
            seed=124,
            tape_sha256="abc",
            expected_actions=3,
        )
