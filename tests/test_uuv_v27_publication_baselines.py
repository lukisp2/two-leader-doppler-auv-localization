from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27


def synthetic_history(count: int = 80):
    t_s = np.arange(1, count + 1, dtype=np.float64)
    displacement = np.column_stack(
        (
            0.35 * t_s + 3.0 * np.sin(t_s / 13.0),
            -0.20 * t_s + 2.0 * np.cos(t_s / 17.0),
            0.08 * t_s + np.sin(t_s / 9.0),
        )
    )
    follower_velocity = np.column_stack(
        (
            0.35 + (3.0 / 13.0) * np.cos(t_s / 13.0),
            -0.20 - (2.0 / 17.0) * np.sin(t_s / 17.0),
            0.08 + (1.0 / 9.0) * np.cos(t_s / 9.0),
        )
    )
    leader_1_position = np.column_stack(
        (
            70.0 + 0.80 * t_s + 25.0 * np.sin(t_s / 11.0),
            -60.0 + 0.25 * t_s + 18.0 * np.cos(t_s / 15.0),
            20.0 + 12.0 * np.sin(t_s / 19.0),
        )
    )
    leader_2_position = np.column_stack(
        (
            -80.0 + 0.15 * t_s + 20.0 * np.cos(t_s / 9.0),
            75.0 - 0.55 * t_s + 22.0 * np.sin(t_s / 14.0),
            -25.0 + 15.0 * np.cos(t_s / 17.0),
        )
    )
    leader_1_velocity = np.column_stack(
        (
            0.80 + (25.0 / 11.0) * np.cos(t_s / 11.0),
            0.25 - (18.0 / 15.0) * np.sin(t_s / 15.0),
            (12.0 / 19.0) * np.cos(t_s / 19.0),
        )
    )
    leader_2_velocity = np.column_stack(
        (
            0.15 - (20.0 / 9.0) * np.sin(t_s / 9.0),
            -0.55 + (22.0 / 14.0) * np.cos(t_s / 14.0),
            -(15.0 / 17.0) * np.sin(t_s / 17.0),
        )
    )
    leader_position = np.stack((leader_1_position, leader_2_position), axis=1)
    leader_velocity = np.stack((leader_1_velocity, leader_2_velocity), axis=1)
    truth_position = np.array([100.0, -140.0, 80.0], dtype=np.float64)
    provisional = v19.OnlineDopplerHistory(
        t_s=t_s,
        dead_reckoned_displacement_m=displacement,
        leader_position_m=leader_position,
        leader_velocity_mps=leader_velocity,
        follower_velocity_measured_mps=follower_velocity,
        doppler_measured_mps=np.zeros((count, 2), dtype=np.float64),
        historical_pf_gate_factor=np.ones((count, 2), dtype=np.float64),
    )
    history = v19.OnlineDopplerHistory(
        t_s=t_s,
        dead_reckoned_displacement_m=displacement,
        leader_position_m=leader_position,
        leader_velocity_mps=leader_velocity,
        follower_velocity_measured_mps=follower_velocity,
        doppler_measured_mps=v19.predict_doppler(truth_position, provisional),
        historical_pf_gate_factor=np.ones((count, 2), dtype=np.float64),
    )
    truth = v19.ReplayTruthDiagnostics(
        initial_follower_position_m=truth_position,
        initial_leader_centroid_m=v19.initial_leader_centroid_from_history(history),
        true_displacement_m=displacement,
        follower_velocity_true_mps=follower_velocity,
    )
    return history, truth_position, truth


class V27BoundaryTests(unittest.TestCase):
    def test_reserved_and_final_seed_guard(self):
        for seed in (49_900, 49_999, 50_000, 50_999):
            with self.subTest(seed=seed), self.assertRaises(PermissionError):
                v27.assert_v27_development_seed(seed)
        for seed in (45_000, 49_899, 51_000):
            with self.subTest(seed=seed):
                v27.assert_v27_development_seed(seed)

    def test_estimator_api_cannot_receive_truth_or_episode_seed(self):
        parameters = inspect.signature(v27.evaluate_arm).parameters
        self.assertNotIn("truth", parameters)
        self.assertNotIn("episode_seed", parameters)
        self.assertNotIn("seed", parameters)

    def test_window_keeps_absolute_dead_reckoning_reference(self):
        history, _, _ = synthetic_history(80)
        window = v27.last_window(history, 20.0)
        self.assertEqual(window.measurement_count, 20)
        self.assertEqual(float(window.t_s[0]), 61.0)
        np.testing.assert_array_equal(
            window.dead_reckoned_displacement_m[-1],
            history.dead_reckoned_displacement_m[-1],
        )


class V27EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.history, self.truth_position, self.truth = synthetic_history()
        self.config = v27.EvaluatorConfig(
            coarse_candidates=512,
            coarse_sweeps=2,
            local_starts=16,
            maximum_modes=8,
            pf_particles_small=256,
            pf_particles_large=512,
        )

    def test_full_global_recovers_identifiable_noiseless_position(self):
        result = v27.evaluate_arm(
            v27.GLOBAL_FULL,
            self.history,
            120.0,
            350.0,
            self.config,
        )
        self.assertLess(
            float(np.linalg.norm(result.initial_position_m - self.truth_position)),
            1e-5,
        )
        self.assertLess(float(result.residual_rmse_mps), 1e-10)

    def test_all_new_online_estimators_are_finite_and_deterministic(self):
        arms = (
            v27.PF_LW_4096,
            v27.PF_LW_16384,
            v27.EKF_STATIC,
            v27.LOCAL_NLS6,
            v27.COARSE_ONLY_8192,
            v27.GLOBAL_WINDOW60,
            v27.GLOBAL_FULL,
        )
        for arm in arms:
            with self.subTest(arm=arm):
                first = v27.evaluate_arm(
                    arm, self.history, 120.0, 350.0, self.config
                )
                second = v27.evaluate_arm(
                    arm, self.history, 120.0, 350.0, self.config
                )
                self.assertTrue(np.all(np.isfinite(first.current_position_m)))
                self.assertEqual(
                    v27.deterministic_payload(first),
                    v27.deterministic_payload(second),
                )

    def test_scoring_is_separate_and_exact(self):
        result = v27.evaluate_arm(
            v27.GLOBAL_FULL,
            self.history,
            120.0,
            350.0,
            self.config,
        )
        payload = result.to_unscored_dict()
        self.assertFalse(any("truth" in key.lower() for key in payload))
        self.assertFalse(any("error" in key.lower() for key in payload))
        score = v27.score_output(result, self.truth, self.history)
        self.assertLess(score["endpoint_position_error_m"], 1e-5)
        self.assertTrue(score["success_le_7m"])

    def test_legacy_pf_loader_uses_checkpoint_and_covariance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trace.npz"
            times = np.array([30.0, 60.0], dtype=np.float64)
            values = {
                "t_s": times,
                "pF_hat_x": np.array([1.0, 2.0]),
                "pF_hat_y": np.array([3.0, 4.0]),
                "pF_hat_z": np.array([5.0, 6.0]),
                "pf_cov_xx": np.array([1.0, 4.0]),
                "pf_cov_xy": np.zeros(2),
                "pf_cov_xz": np.zeros(2),
                "pf_cov_yy": np.array([1.0, 9.0]),
                "pf_cov_yz": np.zeros(2),
                "pf_cov_zz": np.array([1.0, 16.0]),
                # A forbidden-looking source field exists in the archive, but
                # the strict loader never reads or returns it.
                "pF_true_x": np.array([99.0, 99.0]),
            }
            np.savez(path, **values)
            result = v27.load_legacy_pf_output(
                path, v27.sha256_file(path), 60.0
            )
            np.testing.assert_array_equal(
                result.current_position_m, np.array([2.0, 4.0, 6.0])
            )
            self.assertIsNone(result.initial_position_m)
            self.assertAlmostEqual(
                float(result.nominal_radius95_m),
                v19.CHI2_3_95_SQRT * 4.0,
            )


if __name__ == "__main__":
    unittest.main()
