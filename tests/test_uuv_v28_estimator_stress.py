from __future__ import annotations

import inspect
import unittest
from unittest import mock

import numpy as np

import run_v28_estimator_stress as runner28
import uuv_v28_estimator_stress as v28
from tests.test_uuv_v27_publication_baselines import synthetic_history


class V28StressTransformTests(unittest.TestCase):
    def setUp(self):
        self.history, _, _ = synthetic_history(440)

    def test_stress_api_has_no_truth_parameter(self):
        parameters = inspect.signature(v28.apply_stress).parameters
        self.assertNotIn("truth", parameters)
        self.assertNotIn("episode_seed", parameters)

    def test_resume_contract_environment_excludes_process_id(self):
        with mock.patch.object(
            runner28.runner27,
            "_runtime_environment",
            side_effect=(
                {"python": "3.x", "platform": "test", "pid": 101},
                {"python": "3.x", "platform": "test", "pid": 202},
            ),
        ):
            first = runner28._stable_runtime_environment()
            second = runner28._stable_runtime_environment()
        self.assertEqual(first, second)
        self.assertNotIn("pid", first)

    def test_all_conditions_are_deterministic_valid_and_distinct(self):
        nominal_hash = v28.history_sha256(self.history)
        for condition in v28.CONDITIONS:
            with self.subTest(condition=condition):
                first = v28.apply_stress(self.history, condition, 12)
                second = v28.apply_stress(self.history, condition, 12)
                self.assertEqual(v28.history_sha256(first), v28.history_sha256(second))
                self.assertGreater(first.measurement_count, 2)
                if condition == v28.NOMINAL:
                    self.assertEqual(v28.history_sha256(first), nominal_hash)
                else:
                    self.assertNotEqual(v28.history_sha256(first), nominal_hash)

    def test_doppler_bias_and_scale_are_exact(self):
        common = v28.apply_stress(self.history, v28.DOPPLER_COMMON_BIAS, 0)
        differential = v28.apply_stress(
            self.history, v28.DOPPLER_DIFFERENTIAL_BIAS, 0
        )
        scale = v28.apply_stress(self.history, v28.DOPPLER_SCALE, 0)
        np.testing.assert_allclose(
            common.doppler_measured_mps - self.history.doppler_measured_mps,
            0.03,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            differential.doppler_measured_mps - self.history.doppler_measured_mps,
            np.broadcast_to(
                np.array([0.03, -0.03])[None, :],
                self.history.doppler_measured_mps.shape,
            ),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            scale.doppler_measured_mps,
            1.02 * self.history.doppler_measured_mps,
        )

    def test_colored_noise_is_paired_by_episode_index(self):
        same_a = v28.apply_stress(self.history, v28.COLORED_NOISE, 5)
        same_b = v28.apply_stress(self.history, v28.COLORED_NOISE, 5)
        other = v28.apply_stress(self.history, v28.COLORED_NOISE, 6)
        np.testing.assert_array_equal(
            same_a.doppler_measured_mps, same_b.doppler_measured_mps
        )
        self.assertFalse(
            np.array_equal(same_a.doppler_measured_mps, other.doppler_measured_mps)
        )

    def test_dropout_is_nested_and_preserves_scoring_checkpoints(self):
        dropped = v28.apply_stress(self.history, v28.DROPOUT, 7)
        self.assertLess(dropped.measurement_count, self.history.measurement_count)
        for checkpoint in (1.0, 120.0, 440.0):
            self.assertTrue(
                np.any(np.isclose(dropped.t_s, checkpoint, rtol=0.0, atol=1e-12))
            )
        prefix = dropped.prefix(120.0)
        np.testing.assert_array_equal(prefix.t_s, dropped.t_s[dropped.t_s <= 120.0])

    def test_broadcast_delay_and_offset_are_exact(self):
        delayed = v28.apply_stress(self.history, v28.BROADCAST_DELAY, 0)
        offset = v28.apply_stress(self.history, v28.BROADCAST_OFFSET, 0)
        np.testing.assert_array_equal(
            delayed.leader_position_m[2], self.history.leader_position_m[0]
        )
        np.testing.assert_array_equal(
            delayed.leader_velocity_mps[2], self.history.leader_velocity_mps[0]
        )
        np.testing.assert_allclose(
            offset.leader_position_m - self.history.leader_position_m,
            np.broadcast_to(
                v28.BROADCAST_OFFSETS_M[None, :, :],
                self.history.leader_position_m.shape,
            ),
        )

    def test_dead_reckoning_scale_and_drift_are_consistent(self):
        scaled = v28.apply_stress(self.history, v28.DR_SCALE, 0)
        drifted = v28.apply_stress(self.history, v28.DR_DRIFT, 0)
        np.testing.assert_allclose(
            scaled.dead_reckoned_displacement_m,
            1.01 * self.history.dead_reckoned_displacement_m,
        )
        np.testing.assert_allclose(
            drifted.dead_reckoned_displacement_m
            - self.history.dead_reckoned_displacement_m,
            self.history.t_s[:, None] * v28.DR_DRIFT_VELOCITY_MPS[None, :],
        )
        np.testing.assert_allclose(
            drifted.follower_velocity_measured_mps
            - self.history.follower_velocity_measured_mps,
            np.broadcast_to(
                v28.DR_DRIFT_VELOCITY_MPS[None, :],
                self.history.follower_velocity_measured_mps.shape,
            ),
        )


if __name__ == "__main__":
    unittest.main()
