from types import SimpleNamespace

import numpy as np
import pytest

import uuv_v19_observability as v19
import uuv_v20_positioning_ablation as v20


class _OnlineOnlyFakeEnv:
    def __init__(self) -> None:
        self.vL1 = np.asarray([2.0, 0.2, 0.0])
        self.vL2 = np.asarray([1.8, -0.1, 0.0])
        self._v11_speed_meas = 1.7
        self._v11_yaw_meas = 5.0
        self._v11_pitch_meas = -2.0
        self.leader1_speed = float(np.linalg.norm(self.vL1))
        self.leader2_speed = float(np.linalg.norm(self.vL2))
        self.yaw_L1 = 5.7
        self.yaw_L2 = 357.0
        self.step_count = 60
        self.cfg = SimpleNamespace(
            f_min_speed=0.2,
            f_max_speed=5.0,
            pitch_min_deg=-45.0,
            pitch_max_deg=45.0,
            rl_speed_delta_per_step=0.4,
            rl_yaw_per_step_deg=20.0,
            rl_pitch_per_step_deg=14.0,
            leader_speed_max=3.0,
        )

    @property
    def pF(self):  # pragma: no cover - failure message is the assertion
        raise AssertionError("online controller attempted to read truth")

    @property
    def pf(self):  # pragma: no cover - failure message is the assertion
        raise AssertionError("online controller attempted to read the PF")

    def _formation_desired(self):
        return np.zeros(3), np.asarray([10.0, 20.0, -30.0])


def _history(last_time_s: float = 119.0) -> v19.OnlineDopplerHistory:
    return v19.OnlineDopplerHistory(
        t_s=np.asarray([last_time_s]),
        dead_reckoned_displacement_m=np.asarray([[0.5, 0.5, 0.5]]),
        leader_position_m=np.asarray([[[0.0, -50.0, -30.0], [0.0, 50.0, -70.0]]]),
        leader_velocity_mps=np.asarray([[[2.0, 0.0, 0.0], [1.8, 0.2, 0.0]]]),
        follower_velocity_measured_mps=np.asarray([[1.0, 0.0, 0.0]]),
        doppler_measured_mps=np.asarray([[0.1, -0.1]]),
        historical_pf_gate_factor=np.asarray([[1.0, 1.0]]),
    )


def _mode(position) -> v19.BatchMode:
    return v19.BatchMode(
        initial_position_m=np.asarray(position, dtype=np.float64),
        residual_sse_mps2=0.01,
        residual_rmse_mps=0.02,
        iterations=3,
        converged=True,
        hessian_eigenvalues=np.asarray([1.0, 2.0, 3.0]),
        hessian_rank=3,
        hessian_condition_number=3.0,
        local_covariance_m2=np.eye(3),
        local_covariance_valid=True,
        local_radius95_m=2.8,
    )


def test_acquisition_and_pid_use_no_pf_or_truth() -> None:
    env = _OnlineOnlyFakeEnv()
    acquisition = v20.position_independent_acquisition_action(env, 60.0)
    pid = v20.pid_action_for_position(env, np.asarray([1.0, 2.0, 3.0]))
    for action in (acquisition, pid):
        assert action.shape == (3,)
        assert np.all(np.isfinite(action))
        assert np.all(np.abs(action) <= 1.0)


def test_batch_boundary_is_causal_and_uses_current_dead_reckoning(monkeypatch) -> None:
    captured = {}
    estimate = v19.BatchEstimate(
        modes=(_mode([10.0, 20.0, 30.0]),),
        runtime_s=0.0,
        coarse_best_rmse_mps=0.02,
        candidate_count=64,
        refined_start_count=2,
        clustered_mode_count=1,
        alternative_mode_index=None,
        alternative_distance_m=None,
        alternative_delta_sse_mps2=None,
        alternative_delta_chi2=None,
    )

    def fake_estimate(history, center, radius_min, radius_max, config, *, candidate_seed):
        captured["last_measurement_s"] = float(history.t_s[-1])
        captured["candidate_seed"] = int(candidate_seed)
        return estimate

    monkeypatch.setattr(v20.v19, "estimate_initial_position_multistart", fake_estimate)
    estimator = v20.StreamingBatchEstimator(
        config=v19.BatchEstimatorConfig(coarse_candidates=64, local_starts=2),
        support_radius_min_m=120.0,
        support_radius_max_m=350.0,
        episode_index=99,
    )
    estimator.update(
        _history(119.0),
        decision_time_s=120.0,
        current_dead_reckoning_m=[1.0, 2.0, 3.0],
    )
    assert captured == {"last_measurement_s": 119.0, "candidate_seed": 20_121}
    np.testing.assert_allclose(estimator.state.endpoint_position_m, [11.0, 22.0, 33.0])


def test_batch_rejects_a_future_measurement() -> None:
    estimator = v20.StreamingBatchEstimator(
        config=v19.BatchEstimatorConfig(coarse_candidates=64, local_starts=2),
        support_radius_min_m=120.0,
        support_radius_max_m=350.0,
        episode_index=0,
    )
    with pytest.raises(RuntimeError, match="future Doppler"):
        estimator.update(
            _history(121.0),
            decision_time_s=120.0,
            current_dead_reckoning_m=[0.0, 0.0, 0.0],
        )


def test_sealed_final_seed_is_still_refused() -> None:
    with pytest.raises(PermissionError):
        v19.assert_seed_is_not_sealed_final(50_000)
