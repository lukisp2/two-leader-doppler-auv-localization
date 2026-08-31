from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import uuv_v19_observability as v19


ONLINE_ARCHIVE_FIELDS = {
    "t_s",
    "dead_reckoned_displacement_m",
    "leader_position_m",
    "leader_velocity_mps",
    "follower_velocity_measured_mps",
    "doppler_measured_mps",
    "historical_pf_gate_factor",
    "schema_version",
}


def synthetic_noiseless_history():
    """Return an identifiable, exactly noiseless two-leader batch problem."""

    count = 80
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
    truth = np.array([100.0, -140.0, 80.0], dtype=np.float64)
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
        doppler_measured_mps=v19.predict_doppler(truth, provisional),
        historical_pf_gate_factor=np.ones((count, 2), dtype=np.float64),
    )
    return history, truth


def replay_capture_fixture() -> v19.ReplayCapture:
    history, truth_position = synthetic_noiseless_history()
    truth = v19.ReplayTruthDiagnostics(
        initial_follower_position_m=truth_position,
        initial_leader_centroid_m=np.zeros(3, dtype=np.float64),
        true_displacement_m=history.dead_reckoned_displacement_m.copy(),
        follower_velocity_true_mps=history.follower_velocity_measured_mps.copy(),
    )
    return v19.ReplayCapture(
        online=history,
        truth=truth,
        episode_index=0,
        episode_seed=45_000,
        controller=v19.PRIMARY_CONTROLLER,
        support_radius_min_m=120.0,
        support_radius_max_m=350.0,
        initial_pf_metrics={"nearest_particle_distance_m": 12.0},
        frozen_pf_endpoint={"localization_error_diagnostic_m": 20.0},
        scenario_metrics={"duration_s": 80.0},
        replay_integrity={"passed": True},
        provenance={"fixture": True},
    )


class V19MathematicsTests(unittest.TestCase):
    def test_sealed_final_seed_guard_includes_both_boundaries(self):
        for seed in (50_000, 50_001, 50_999):
            with self.subTest(seed=seed), self.assertRaises(PermissionError):
                v19.assert_seed_is_not_sealed_final(seed)
        for seed in (49_999, 51_000, 45_000, 45_099):
            with self.subTest(seed=seed):
                v19.assert_seed_is_not_sealed_final(seed)

    def test_analytic_residual_jacobian_matches_central_difference(self):
        rng = np.random.default_rng(1207)
        count = 11
        t_s = np.arange(1, count + 1, dtype=np.float64)
        history = v19.OnlineDopplerHistory(
            t_s=t_s,
            dead_reckoned_displacement_m=rng.normal(size=(count, 3)),
            leader_position_m=(
                rng.normal(size=(count, 2, 3)) * 12.0
                + np.array([50.0, -30.0, 20.0])[None, None, :]
            ),
            leader_velocity_mps=rng.normal(size=(count, 2, 3)),
            follower_velocity_measured_mps=rng.normal(size=(count, 3)),
            doppler_measured_mps=rng.normal(size=(count, 2)),
            historical_pf_gate_factor=rng.uniform(0.2, 1.0, size=(count, 2)),
        )
        config = v19.BatchEstimatorConfig(
            coarse_candidates=64,
            local_starts=2,
            maximum_modes=2,
            gate_mode="historical_soft",
        )
        position = np.array([3.0, -8.0, 5.0], dtype=np.float64)
        _, analytic = v19.residual_and_jacobian(position, history, config)
        epsilon = 1e-6
        numerical_columns = []
        for axis in range(3):
            perturbation = np.zeros(3, dtype=np.float64)
            perturbation[axis] = epsilon
            residual_plus, _ = v19.residual_and_jacobian(
                position + perturbation, history, config
            )
            residual_minus, _ = v19.residual_and_jacobian(
                position - perturbation, history, config
            )
            numerical_columns.append(
                (residual_plus - residual_minus) / (2.0 * epsilon)
            )
        numerical = np.column_stack(numerical_columns)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=2e-9)

    def test_shell_design_is_deterministic_bounded_and_uniform_in_radius(self):
        center = np.array([8.0, -2.0, 4.0], dtype=np.float64)
        candidates = v19.deterministic_shell_candidates(
            center,
            120.0,
            350.0,
            4096,
            19001,
            radial_distribution="uniform_radius",
        )
        repeated = v19.deterministic_shell_candidates(
            center,
            120.0,
            350.0,
            4096,
            19001,
            radial_distribution="uniform_radius",
        )
        changed_seed = v19.deterministic_shell_candidates(
            center,
            120.0,
            350.0,
            4096,
            19002,
            radial_distribution="uniform_radius",
        )
        np.testing.assert_array_equal(candidates, repeated)
        self.assertFalse(np.array_equal(candidates, changed_seed))
        radii = np.linalg.norm(candidates - center[None, :], axis=1)
        self.assertGreaterEqual(float(np.min(radii)), 120.0)
        self.assertLessEqual(float(np.max(radii)), 350.0)
        self.assertAlmostEqual(float(np.mean(radii)), 235.0, delta=0.12)
        quartiles = np.quantile(radii, [0.25, 0.50, 0.75])
        np.testing.assert_allclose(
            quartiles,
            np.array([177.5, 235.0, 292.5]),
            atol=0.15,
            rtol=0.0,
        )

    def test_multistart_recovers_identifiable_noiseless_position(self):
        history, truth = synthetic_noiseless_history()
        config = v19.BatchEstimatorConfig(
            coarse_candidates=1024,
            coarse_sweeps=2,
            local_starts=20,
            maximum_modes=8,
            maximum_iterations=100,
        )
        estimate = v19.estimate_initial_position_multistart(
            history,
            np.zeros(3, dtype=np.float64),
            120.0,
            350.0,
            config,
            candidate_seed=1919,
        )
        self.assertTrue(estimate.best.converged)
        self.assertLess(
            float(np.linalg.norm(estimate.best.initial_position_m - truth)), 1e-6
        )
        self.assertLess(estimate.best.residual_rmse_mps, 1e-12)
        self.assertEqual(estimate.candidate_count, 2048)
        self.assertEqual(estimate.refined_start_count, 20)

    def test_rank_deficient_hessian_never_reports_finite_confidence_radius(self):
        count = 10
        history = v19.OnlineDopplerHistory(
            t_s=np.arange(1, count + 1, dtype=np.float64),
            dead_reckoned_displacement_m=np.zeros((count, 3), dtype=np.float64),
            leader_position_m=np.broadcast_to(
                np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]),
                (count, 2, 3),
            ).copy(),
            leader_velocity_mps=np.zeros((count, 2, 3), dtype=np.float64),
            follower_velocity_measured_mps=np.zeros((count, 3), dtype=np.float64),
            doppler_measured_mps=np.zeros((count, 2), dtype=np.float64),
            historical_pf_gate_factor=np.ones((count, 2), dtype=np.float64),
        )
        config = v19.BatchEstimatorConfig(
            coarse_candidates=64,
            local_starts=2,
            maximum_modes=2,
        )
        mode = v19.refine_damped_gauss_newton(
            np.array([200.0, 0.0, 0.0]),
            history,
            np.zeros(3, dtype=np.float64),
            120.0,
            350.0,
            config,
        )
        self.assertLess(mode.hessian_rank, 3)
        self.assertFalse(mode.local_covariance_valid)
        self.assertTrue(np.isinf(mode.local_radius95_m))
        self.assertIsNone(mode.to_dict()["local_radius95_m"])
        json.dumps(mode.to_dict(), allow_nan=False)


class V19ArchiveBoundaryTests(unittest.TestCase):
    def test_online_archive_has_exact_allowlist_and_rejects_extra_fields(self):
        history, _ = synthetic_noiseless_history()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            online_path = root / "online_inputs.npz"
            v19.save_online_history(online_path, history)
            with np.load(online_path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), ONLINE_ARCHIVE_FIELDS)
                for name in archive.files:
                    lowered = name.lower()
                    self.assertNotIn("truth", lowered)
                    self.assertNotIn("seed", lowered)
                    self.assertNotIn("noise", lowered)
            loaded = v19.load_online_history(online_path)
            np.testing.assert_array_equal(
                loaded.doppler_measured_mps, history.doppler_measured_mps
            )

            forbidden_path = root / "forbidden_online_inputs.npz"
            with np.load(online_path, allow_pickle=False) as archive:
                copied = {name: archive[name].copy() for name in archive.files}
            np.savez_compressed(
                forbidden_path,
                **copied,
                diagnostic_truth_position_m=np.zeros(3, dtype=np.float64),
            )
            with self.assertRaisesRegex(ValueError, "forbidden fields"):
                v19.load_online_history(forbidden_path)

    def test_capture_digest_detects_online_archive_modification(self):
        capture = replay_capture_fixture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "episode"
            paths = v19.save_replay_capture(root, capture)
            metadata = json.loads(
                Path(paths["metadata"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["online_inputs_sha256"],
                v19.sha256_file(Path(paths["online_inputs"])),
            )
            self.assertEqual(
                metadata["truth_labels_sha256"],
                v19.sha256_file(Path(paths["truth_labels"])),
            )
            round_trip = v19.load_replay_capture(root)
            self.assertEqual(round_trip.episode_seed, 45_000)
            np.testing.assert_array_equal(
                round_trip.online.t_s, capture.online.t_s
            )

            modified = v19.OnlineDopplerHistory(
                t_s=capture.online.t_s,
                dead_reckoned_displacement_m=(
                    capture.online.dead_reckoned_displacement_m
                ),
                leader_position_m=capture.online.leader_position_m,
                leader_velocity_mps=capture.online.leader_velocity_mps,
                follower_velocity_measured_mps=(
                    capture.online.follower_velocity_measured_mps
                ),
                doppler_measured_mps=(
                    capture.online.doppler_measured_mps + 0.01
                ),
                historical_pf_gate_factor=(
                    capture.online.historical_pf_gate_factor
                ),
            )
            v19.save_online_history(Path(paths["online_inputs"]), modified)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                v19.load_replay_capture(root)


class V19FrozenReplayIntegrationTests(unittest.TestCase):
    def test_first_available_v181_episode_replays_bit_exactly(self):
        repository = Path(__file__).resolve().parents[1]
        archive_root = Path(
            os.environ.get("UUV_ARCHIVE_ROOT", repository / "data" / "raw")
        ).expanduser().resolve()
        evaluation_directory = (
            archive_root
            / "estimator_benchmark"
            / "v18_replay_fixture"
        )
        trace = (
            evaluation_directory
            / "traces_npz"
            / v19.PRIMARY_CONTROLLER
            / "episode_0000_seed_45000.npz"
        )
        tape = (
            evaluation_directory
            / "noise_tapes"
            / "episode_0000_seed_45000.npz"
        )
        if not trace.is_file() or not tape.is_file():
            self.skipTest(
                "DOI-archived V18.1 replay fixture is not present under "
                f"{archive_root}"
            )

        capture = v19.capture_v181_episode(
            evaluation_directory,
            episode_index=0,
            episode_seed=45_000,
        )
        self.assertTrue(capture.replay_integrity["passed"])
        self.assertEqual(capture.replay_integrity["action_count"], 220)
        self.assertEqual(capture.replay_integrity["measurement_count"], 440)
        self.assertEqual(capture.online.measurement_count, 440)
        np.testing.assert_allclose(
            capture.online.t_s,
            np.arange(1.0, 441.0, dtype=np.float64),
            rtol=0.0,
            atol=1e-10,
        )
        maximum_difference = capture.replay_integrity[
            "maximum_absolute_difference"
        ]
        self.assertEqual(set(maximum_difference), {
            "pF_true_x",
            "pF_true_y",
            "pF_true_z",
            "pF_hat_x",
            "pF_hat_y",
            "pF_hat_z",
            "pf_cov_xx",
            "pf_cov_xy",
            "pf_cov_xz",
            "pf_cov_yy",
            "pf_cov_yz",
            "pf_cov_zz",
        })
        self.assertLessEqual(
            max(float(value) for value in maximum_difference.values()),
            v19.REPLAY_TOLERANCE,
        )
        self.assertTrue(np.all(np.isfinite(capture.online.doppler_measured_mps)))
        self.assertTrue(np.all(np.isfinite(capture.truth.true_displacement_m)))
        self.assertEqual(
            capture.replay_integrity["noise_cursor_state"],
            {"substep_index": 4400, "measurement_index": 440},
        )
        np.testing.assert_allclose(
            v19.initial_leader_centroid_from_history(capture.online),
            capture.truth.initial_leader_centroid_m,
            rtol=0.0,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
