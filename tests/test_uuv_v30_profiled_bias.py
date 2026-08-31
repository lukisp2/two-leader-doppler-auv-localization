from __future__ import annotations

import inspect
import unittest

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27
import uuv_v30_profiled_bias as v30
from tests.test_uuv_v27_publication_baselines import synthetic_history


def with_bias(
    history: v19.OnlineDopplerHistory, bias_mps: np.ndarray
) -> v19.OnlineDopplerHistory:
    return v19.OnlineDopplerHistory(
        t_s=history.t_s.copy(),
        dead_reckoned_displacement_m=history.dead_reckoned_displacement_m.copy(),
        leader_position_m=history.leader_position_m.copy(),
        leader_velocity_mps=history.leader_velocity_mps.copy(),
        follower_velocity_measured_mps=history.follower_velocity_measured_mps.copy(),
        doppler_measured_mps=(
            history.doppler_measured_mps
            + np.asarray(bias_mps, dtype=np.float64)[None, :]
        ),
        historical_pf_gate_factor=history.historical_pf_gate_factor.copy(),
    )


class V30ProfiledBiasTests(unittest.TestCase):
    def setUp(self):
        self.history, self.truth_position, _ = synthetic_history(120)
        self.bias = np.array([0.03, -0.03], dtype=np.float64)
        self.biased = with_bias(self.history, self.bias)
        self.config = v27.EvaluatorConfig(
            coarse_candidates=512,
            coarse_sweeps=2,
            local_starts=16,
            maximum_modes=8,
            pf_particles_small=256,
            pf_particles_large=512,
        )

    def test_estimator_api_has_no_truth_parameter(self):
        parameters = inspect.signature(v30.evaluate_arm).parameters
        self.assertNotIn("truth", parameters)
        self.assertNotIn("episode_seed", parameters)

    def test_variable_projection_recovers_exact_bias_at_truth(self):
        residual, _, estimated = v30.profiled_residual_and_jacobian(
            self.truth_position, self.biased, self.config.batch_config()
        )
        np.testing.assert_allclose(estimated, self.bias, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1e-12)

    def test_profiled_jacobian_matches_finite_difference(self):
        _, jacobian, _ = v30.profiled_residual_and_jacobian(
            self.truth_position, self.biased, self.config.batch_config()
        )
        step = 1e-5
        for axis in range(3):
            positive = self.truth_position.copy()
            negative = self.truth_position.copy()
            positive[axis] += step
            negative[axis] -= step
            residual_positive, _, _ = v30.profiled_residual_and_jacobian(
                positive, self.biased, self.config.batch_config()
            )
            residual_negative, _, _ = v30.profiled_residual_and_jacobian(
                negative, self.biased, self.config.batch_config()
            )
            numerical = (residual_positive - residual_negative) / (2.0 * step)
            np.testing.assert_allclose(
                jacobian[:, axis], numerical, rtol=2e-6, atol=2e-8
            )

    def test_profiled_global_recovers_position_and_bias(self):
        output = v30.evaluate_arm(
            v30.PROFILED_GLOBAL,
            self.biased,
            120.0,
            350.0,
            self.config,
        )
        self.assertLess(
            float(np.linalg.norm(output.initial_position_m - self.truth_position)),
            1e-5,
        )
        np.testing.assert_allclose(output.link_bias_mps, self.bias, atol=1e-10)
        self.assertIsNotNone(output.covariance_m2)
        self.assertIsNotNone(output.nominal_radius95_m)

    def test_profiled_arms_are_deterministic_apart_from_runtime(self):
        for arm in (v30.PROFILED_LOCAL6, v30.PROFILED_GLOBAL):
            with self.subTest(arm=arm):
                first = v30.evaluate_arm(
                    arm, self.biased, 120.0, 350.0, self.config
                )
                second = v30.evaluate_arm(
                    arm, self.biased, 120.0, 350.0, self.config
                )
                self.assertEqual(
                    v30.deterministic_payload(first),
                    v30.deterministic_payload(second),
                )


if __name__ == "__main__":
    unittest.main()
