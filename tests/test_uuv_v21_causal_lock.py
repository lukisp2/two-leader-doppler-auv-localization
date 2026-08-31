from __future__ import annotations

import numpy as np

import uuv_v19_observability as v19
import uuv_v21_causal_lock as v21


def _mode(
    position=(10.0, 20.0, 30.0),
    *,
    radius95=2.0,
    rmse=0.03,
    valid=True,
    rank=3,
    converged=True,
    sse=0.1,
) -> v19.BatchMode:
    return v19.BatchMode(
        initial_position_m=np.asarray(position, dtype=np.float64),
        residual_sse_mps2=float(sse),
        residual_rmse_mps=float(rmse),
        iterations=4,
        converged=bool(converged),
        hessian_eigenvalues=np.asarray([1.0, 2.0, 3.0]),
        hessian_rank=int(rank),
        hessian_condition_number=3.0,
        local_covariance_m2=np.eye(3),
        local_covariance_valid=bool(valid),
        local_radius95_m=float(radius95),
    )


def _estimate(
    position=(10.0, 20.0, 30.0),
    *,
    alternative_delta_chi2=None,
) -> v19.BatchEstimate:
    modes = [_mode(position)]
    alternative_index = None
    alternative_distance = None
    alternative_sse = None
    if alternative_delta_chi2 is not None:
        alternative_index = 1
        alternative_distance = 20.0
        alternative_sse = float(alternative_delta_chi2) * 0.05**2
        modes.append(
            _mode(
                (30.0, 20.0, 30.0),
                sse=0.1 + alternative_sse,
            )
        )
    return v19.BatchEstimate(
        modes=tuple(modes),
        runtime_s=0.01,
        coarse_best_rmse_mps=0.03,
        candidate_count=128,
        refined_start_count=4,
        clustered_mode_count=len(modes),
        alternative_mode_index=alternative_index,
        alternative_distance_m=alternative_distance,
        alternative_delta_sse_mps2=alternative_sse,
        alternative_delta_chi2=alternative_delta_chi2,
    )


def _evidence(
    *,
    time_s=120.0,
    alternative_delta_chi2=None,
) -> v21.GlobalLockEvidence:
    return v21.GlobalLockEvidence(
        time_s=float(time_s),
        primary=_estimate(alternative_delta_chi2=alternative_delta_chi2),
        confirmation=_estimate((10.1, 20.0, 30.0)),
        search_agreement_m=0.1,
        stability_from_previous_m=0.5,
        forward_prediction_rmse_mps=0.03,
        forward_prediction_sample_count=30,
    )


def _evaluate(
    gate: v21.CausalLockGate,
    now_s: float,
    *,
    mode: v19.BatchMode | None = None,
    evidence: v21.GlobalLockEvidence | None = None,
) -> bool:
    return gate.evaluate(
        now_s=float(now_s),
        mode=_mode() if mode is None else mode,
        initial_position_m=[10.0 + 0.01 * float(now_s), 20.0, 30.0],
        recent_residual_rmse_mps=0.03,
        global_evidence=_evidence() if evidence is None else evidence,
    )


def test_gate_cannot_release_before_minimum_time() -> None:
    gate = v21.CausalLockGate(v21.CausalLockConfig())
    for now in (110.0, 112.0, 114.0, 116.0, 118.0):
        assert not _evaluate(gate, now)
    assert gate.state.lock_count == 0
    assert "minimum_time" in gate.state.last_failed_checks


def test_gate_requires_three_complete_consecutive_passes() -> None:
    gate = v21.CausalLockGate(v21.CausalLockConfig())
    for now in (114.0, 116.0, 118.0):
        assert not _evaluate(gate, now)
    assert not _evaluate(gate, 120.0)
    assert not _evaluate(gate, 122.0)
    assert _evaluate(gate, 124.0)
    assert gate.is_locked
    assert gate.state.first_lock_time_s == 124.0
    assert gate.state.lock_count == 1


def test_unresolved_separated_mode_blocks_release() -> None:
    gate = v21.CausalLockGate(v21.CausalLockConfig())
    ambiguous = _evidence(alternative_delta_chi2=2.0)
    for now in (120.0, 122.0, 124.0, 126.0, 128.0):
        assert not _evaluate(gate, now, evidence=ambiguous)
    assert "primary_mode_clear" in gate.state.last_failed_checks


def test_invalid_covariance_and_rank_block_release() -> None:
    for mode in (_mode(valid=False), _mode(rank=2), _mode(converged=False)):
        gate = v21.CausalLockGate(v21.CausalLockConfig())
        for now in (120.0, 122.0, 124.0, 126.0, 128.0):
            assert not _evaluate(gate, now, mode=mode)
        assert not gate.is_locked


def test_three_unhealthy_actions_drop_lock_and_force_global_refresh() -> None:
    gate = v21.CausalLockGate(v21.CausalLockConfig())
    for now in (114.0, 116.0, 118.0, 120.0, 122.0, 124.0):
        _evaluate(gate, now)
    assert gate.is_locked
    bad = _mode(radius95=11.0)
    assert _evaluate(gate, 126.0, mode=bad)
    assert _evaluate(gate, 128.0, mode=bad)
    assert not _evaluate(gate, 130.0, mode=bad)
    assert not gate.is_locked
    assert gate.state.unlock_count == 1
    assert gate.consume_force_global_refresh()
    assert not gate.consume_force_global_refresh()
