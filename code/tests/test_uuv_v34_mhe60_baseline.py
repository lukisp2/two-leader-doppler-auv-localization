from __future__ import annotations

import inspect
import unittest

import numpy as np

import uuv_v19_observability as v19
import uuv_v34_mhe60_baseline as v34


def synthetic_history(count: int = 440):
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
    initial_position = np.array([100.0, -140.0, 80.0], dtype=np.float64)
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
        doppler_measured_mps=v19.predict_doppler(initial_position, provisional),
        historical_pf_gate_factor=np.ones((count, 2), dtype=np.float64),
    )
    truth = v19.ReplayTruthDiagnostics(
        initial_follower_position_m=initial_position,
        initial_leader_centroid_m=v19.initial_leader_centroid_from_history(history),
        true_displacement_m=displacement,
        follower_velocity_true_mps=follower_velocity,
    )
    return history, initial_position, truth


def forbidden_unscored_keys(value):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if any(token in str(key).lower() for token in ("truth", "error", "success")):
                found.append(str(key))
            found.extend(forbidden_unscored_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(forbidden_unscored_keys(item))
    return found


class V34BoundaryTests(unittest.TestCase):
    def test_closed_seed_guard(self):
        for seed in (49_900, 49_999, 50_000, 50_999):
            with self.subTest(seed=seed), self.assertRaises(PermissionError):
                v34.assert_v34_development_seed(seed)
        v34.assert_v34_development_seed(45_000)

    def test_estimator_api_has_no_truth_or_episode_identity(self):
        parameters = inspect.signature(v34.evaluate_mhe_history).parameters
        for forbidden in ("truth", "episode_seed", "episode_index", "seed"):
            self.assertNotIn(forbidden, parameters)

    def test_horizon_and_iteration_contract_cannot_change(self):
        with self.assertRaises(ValueError):
            v34.MHEConfig(horizon_samples=59)
        with self.assertRaises(ValueError):
            v34.MHEConfig(maximum_iterations=79)


class V34ArrivalTests(unittest.TestCase):
    def test_canonical_arrival_equals_stacked_linearized_factor_cost(self):
        rng = np.random.default_rng(34)
        arrival = v34.ArrivalCost.zero()
        stacked = []
        for _ in range(7):
            residual = rng.normal(size=2)
            jacobian = rng.normal(size=(2, 3))
            linearization = rng.normal(size=3)
            offset = residual - jacobian @ linearization
            stacked.append((jacobian, offset))
            arrival.add_linearized_factor(residual, jacobian, linearization)
        for _ in range(5):
            candidate = rng.normal(size=3)
            direct = sum(
                float((jacobian @ candidate + offset) @ (jacobian @ candidate + offset))
                for jacobian, offset in stacked
            )
            self.assertAlmostEqual(arrival.quadratic_value(candidate), direct, places=11)
        self.assertEqual(arrival.sample_count, 7)


class V34EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history, cls.initial_position, cls.truth = synthetic_history()
        cls.config = v34.MHEConfig(
            coarse_candidates=512,
            coarse_sweeps=1,
            local_starts=8,
            maximum_modes=8,
        )
        cls.outputs = v34.evaluate_mhe_history(
            cls.history, 120.0, 350.0, cls.config
        )

    def test_noiseless_history_is_recovered(self):
        for output in self.outputs:
            with self.subTest(checkpoint=output.checkpoint_s):
                self.assertLess(
                    float(np.linalg.norm(output.initial_position_m - self.initial_position)),
                    1e-4,
                )

    def test_arrival_and_window_counts_have_no_gap_or_overlap(self):
        for output in self.outputs:
            checkpoint = int(round(output.checkpoint_s))
            diagnostics = output.diagnostics
            self.assertEqual(
                diagnostics["window_sample_count"], min(checkpoint, 60)
            )
            self.assertEqual(
                diagnostics["marginalized_sample_count"], max(checkpoint - 60, 0)
            )
            self.assertEqual(diagnostics["total_sample_count"], checkpoint)

    def test_outputs_are_deterministic_apart_from_runtime(self):
        repeated = v34.evaluate_mhe_history(
            self.history, 120.0, 350.0, self.config
        )
        self.assertEqual(
            [v34.deterministic_payload(output) for output in self.outputs],
            [v34.deterministic_payload(output) for output in repeated],
        )

    def test_unscored_payload_has_no_scoring_fields(self):
        for output in self.outputs:
            self.assertEqual(forbidden_unscored_keys(output.to_unscored_dict()), [])

    def test_scoring_is_separate(self):
        score = v34.score_output(self.outputs[-1], self.truth, self.history)
        self.assertLess(score["endpoint_position_error_m"], 1e-4)
        self.assertTrue(score["success_le_7m"])


if __name__ == "__main__":
    unittest.main()

