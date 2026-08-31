import unittest

import numpy as np

from uuv_v18_resampling_guard import UUV3DConfig
import uuv_v19_observability as v19
import uuv_v21_causal_lock as v21
import uuv_v22_active_acquisition as v22
import uuv_v38_leader_source_ablation as v38


def _full_history(count=40):
    t = np.arange(1, count + 1, dtype=np.float64)
    dr = np.column_stack((1.1 * t, 0.2 * np.sin(t / 7.0), -0.03 * t))
    leader_position = np.empty((count, 2, 3), dtype=np.float64)
    leader_velocity = np.empty((count, 2, 3), dtype=np.float64)
    leader_velocity[:, 0, :] = np.asarray([1.8, 0.2, 0.0])
    leader_velocity[:, 1, :] = np.asarray([1.4, -0.3, 0.0])
    leader_position[:, 0, :] = (
        np.asarray([0.0, 50.0, -30.0])
        + t[:, None] * leader_velocity[:, 0, :]
    )
    leader_position[:, 1, :] = (
        np.asarray([0.0, -50.0, -75.0])
        + t[:, None] * leader_velocity[:, 1, :]
    )
    follower_velocity = np.tile(
        np.asarray([1.1, 0.0, -0.03]),
        (count, 1),
    )
    provisional = v19.OnlineDopplerHistory(
        t_s=t,
        dead_reckoned_displacement_m=dr,
        leader_position_m=leader_position,
        leader_velocity_mps=leader_velocity,
        follower_velocity_measured_mps=follower_velocity,
        doppler_measured_mps=np.zeros((count, 2)),
        historical_pf_gate_factor=np.ones((count, 2)),
    )
    truth_p0 = np.asarray([-170.0, 80.0, -115.0])
    doppler = v19.predict_doppler(truth_p0, provisional)
    return v19.OnlineDopplerHistory(
        t_s=t,
        dead_reckoned_displacement_m=dr,
        leader_position_m=leader_position,
        leader_velocity_mps=leader_velocity,
        follower_velocity_measured_mps=follower_velocity,
        doppler_measured_mps=doppler,
        historical_pf_gate_factor=np.ones((count, 2)),
    )


class _PlannerEnv:
    def __init__(self):
        self.cfg = UUV3DConfig()
        self._v11_speed_meas = 1.6
        self._v11_yaw_meas = 12.0
        self._v11_pitch_meas = -1.0
        self.pL1 = np.asarray([50.0, 55.0, -30.0])
        self.pL2 = np.asarray([45.0, -60.0, -75.0])
        self.vL1 = np.asarray([1.8, 0.2, 0.0])
        self.vL2 = np.asarray([1.4, -0.3, 0.0])


class _PlannerEstimator:
    def __init__(self, history):
        self.estimator_config = v19.BatchEstimatorConfig(
            coarse_candidates=64,
            coarse_sweeps=1,
            local_starts=2,
        )
        self.support = v38.MissionSupport(
            center_m=np.asarray([0.0, 0.0, -52.5]),
            radius_min_m=100.0,
            radius_max_m=365.0,
        )
        p0 = np.asarray([-170.0, 80.0, -115.0])
        residual, jacobian = v38.residual_and_jacobian(
            p0,
            history,
            self.estimator_config,
        )
        hessian = jacobian.T @ jacobian
        covariance = np.linalg.inv(hessian) * 0.05**2
        mode = v19.BatchMode(
            initial_position_m=p0,
            residual_sse_mps2=float(residual @ residual),
            residual_rmse_mps=float(np.sqrt(np.mean(residual * residual))),
            iterations=1,
            converged=True,
            hessian_eigenvalues=np.linalg.eigvalsh(hessian),
            hessian_rank=3,
            hessian_condition_number=float(np.linalg.cond(hessian)),
            local_covariance_m2=covariance,
            local_covariance_valid=True,
            local_radius95_m=2.0,
        )
        self.state = v21.CausalEstimatorState(
            initial_position_m=p0.copy(),
            endpoint_position_m=(
                p0 + history.dead_reckoned_displacement_m[-1]
            ),
            latest_mode=mode,
        )

    @property
    def has_solution(self):
        return True


class V38LeaderSourceTests(unittest.TestCase):
    def test_contract_is_paired_three_by_two(self):
        self.assertEqual(len(v38.SOURCE_NAMES), 3)
        self.assertEqual(len(v38.POLICIES), 2)
        self.assertEqual(len(v38.arm_pairs()), 6)
        self.assertEqual(len({v38.arm_name(*arm) for arm in v38.arm_pairs()}), 6)

    def test_reserved_and_final_seeds_are_rejected(self):
        v38.assert_seed_allowed(48_400)
        v38.assert_seed_allowed(48_599)
        for seed in (49_900, 50_000, 50_999):
            with self.assertRaises(PermissionError):
                v38.assert_seed_allowed(seed)

    def test_mission_support_matches_frozen_two_leader_prior(self):
        cfg = UUV3DConfig()
        leader1 = np.asarray([0.0, 50.0, -31.0])
        leader2 = np.asarray([0.0, -50.0, -74.0])
        support = v38.mission_support_from_initial_leaders(
            cfg,
            leader1,
            leader2,
        )
        np.testing.assert_array_equal(
            support.center_m,
            0.5 * (leader1 + leader2),
        )
        self.assertEqual(support.radius_min_m, float(cfg.start_rho_min))
        self.assertEqual(support.radius_max_m, float(cfg.start_rho_max))

    def test_both_source_math_matches_frozen_v19(self):
        full = _full_history()
        masked = v38.MaskedDopplerHistory.from_full(full, (True, True))
        point = np.asarray([-165.0, 75.0, -110.0])
        config = v19.BatchEstimatorConfig(
            coarse_candidates=64,
            coarse_sweeps=1,
            local_starts=2,
        )
        np.testing.assert_allclose(
            v38.predict_doppler(point, masked),
            v19.predict_doppler(point, full),
            rtol=0.0,
            atol=1e-12,
        )
        residual_masked, jacobian_masked = v38.residual_and_jacobian(
            point,
            masked,
            config,
        )
        residual_full, jacobian_full = v19.residual_and_jacobian(
            point,
            full,
            config,
        )
        np.testing.assert_allclose(residual_masked, residual_full, atol=1e-12)
        np.testing.assert_allclose(jacobian_masked, jacobian_full, atol=1e-12)

    def test_excluded_link_cannot_change_likelihood_or_jacobian(self):
        full = _full_history()
        changed = v19.OnlineDopplerHistory(
            t_s=full.t_s,
            dead_reckoned_displacement_m=full.dead_reckoned_displacement_m,
            leader_position_m=full.leader_position_m.copy(),
            leader_velocity_mps=full.leader_velocity_mps.copy(),
            follower_velocity_measured_mps=full.follower_velocity_measured_mps,
            doppler_measured_mps=full.doppler_measured_mps.copy(),
            historical_pf_gate_factor=full.historical_pf_gate_factor,
        )
        changed.leader_position_m[:, 1, :] += 1e6
        changed.leader_velocity_mps[:, 1, :] -= 1e4
        changed.doppler_measured_mps[:, 1] += 1e5
        left = v38.MaskedDopplerHistory.from_full(full, (True, False))
        right = v38.MaskedDopplerHistory.from_full(changed, (True, False))
        self.assertEqual(left.content_sha256(), right.content_sha256())
        point = np.asarray([-165.0, 75.0, -110.0])
        config = v19.BatchEstimatorConfig(
            coarse_candidates=64,
            coarse_sweeps=1,
            local_starts=2,
        )
        np.testing.assert_array_equal(
            v38.predict_doppler(point, left),
            v38.predict_doppler(point, right),
        )
        for left_value, right_value in zip(
            v38.residual_and_jacobian(point, left, config),
            v38.residual_and_jacobian(point, right, config),
        ):
            np.testing.assert_array_equal(left_value, right_value)

    def test_rmse_denominator_counts_only_active_scalars(self):
        full = _full_history(count=30)
        l1 = v38.MaskedDopplerHistory.from_full(full, (True, False))
        both = v38.MaskedDopplerHistory.from_full(full, (True, True))
        self.assertEqual(l1.scalar_measurement_count, 30)
        self.assertEqual(both.scalar_measurement_count, 60)
        config = v19.BatchEstimatorConfig(
            coarse_candidates=64,
            coarse_sweeps=1,
            local_starts=2,
        )
        residual, jacobian = v38.residual_and_jacobian(
            np.asarray([-160.0, 70.0, -105.0]),
            l1,
            config,
        )
        self.assertEqual(residual.shape, (30,))
        self.assertEqual(jacobian.shape, (30, 3))

    def test_common_s_turn_is_source_independent(self):
        env = _PlannerEnv()
        expected = v38.common_s_turn_action(env, 80.0)
        for _ in v38.SOURCE_NAMES:
            np.testing.assert_array_equal(
                v38.common_s_turn_action(env, 80.0),
                expected,
            )

    def test_excluded_leader_position_cannot_change_active_prediction(self):
        full = _full_history()
        history = v38.MaskedDopplerHistory.from_full(full, (True, False))
        env_left = _PlannerEnv()
        env_right = _PlannerEnv()
        env_right.pL2 += np.asarray([1e6, -2e6, 3e6])
        estimator = _PlannerEstimator(history)
        config = v22.ActivePlannerConfig(maximum_hypotheses=4)
        planner_left = v38.MaskedBeliefFIMPlanner(config, (True, False))
        planner_right = v38.MaskedBeliefFIMPlanner(config, (True, False))
        decision_left = planner_left.action(
            env_left,
            history=history,
            current_dead_reckoning_m=history.dead_reckoned_displacement_m[-1],
            estimator=estimator,
            time_s=80.0,
        )
        decision_right = planner_right.action(
            env_right,
            history=history,
            current_dead_reckoning_m=history.dead_reckoned_displacement_m[-1],
            estimator=estimator,
            time_s=80.0,
        )
        np.testing.assert_array_equal(decision_left.action, decision_right.action)
        self.assertAlmostEqual(decision_left.utility, decision_right.utility)
        self.assertAlmostEqual(
            decision_left.minimum_pair_chi2,
            decision_right.minimum_pair_chi2,
        )


if __name__ == "__main__":
    unittest.main()
