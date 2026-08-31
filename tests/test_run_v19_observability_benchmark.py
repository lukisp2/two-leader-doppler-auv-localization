from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_v19_observability_benchmark as benchmark


def decision_rows(
    count: int = 100,
    *,
    noisy_240_failures=(),
    noisy_440_failures=(),
    structural_failures=(),
    runtimes_s=None,
):
    """Build the minimal internally consistent rows consumed by _decision."""

    noisy_240_failures = set(noisy_240_failures)
    noisy_440_failures = set(noisy_440_failures)
    structural_failures = set(structural_failures)
    if runtimes_s is None:
        runtimes_s = [0.5] * count
    if len(runtimes_s) != count:
        raise ValueError("one runtime is required per scenario")

    rows = []
    for episode_index in range(count):
        pass_240 = episode_index not in noisy_240_failures
        pass_440 = episode_index not in noisy_440_failures
        rows.extend(
            (
                {
                    "episode_index": episode_index,
                    "problem": "structural_noiseless",
                    "prefix_s": 440.0,
                    "structural_gate_pass": (
                        episode_index not in structural_failures
                    ),
                },
                {
                    "episode_index": episode_index,
                    "problem": "recorded_noisy",
                    "prefix_s": 240.0,
                    "noisy_7m_gate_pass": pass_240,
                    # This is the value produced by _annotate_episode_lock:
                    # lock at 240 s is sustained only if both endpoints pass.
                    "sustained_noisy_7m_lock_through_440": (
                        pass_240 and pass_440
                    ),
                },
                {
                    "episode_index": episode_index,
                    "problem": "recorded_noisy",
                    "prefix_s": 440.0,
                    "noisy_7m_gate_pass": pass_440,
                    "runtime_s": float(runtimes_s[episode_index]),
                },
            )
        )
    return rows


class NearestRankPercentileTests(unittest.TestCase):
    def test_nearest_rank_uses_ceil_rank_without_interpolation(self):
        values = list(range(1, 101))
        self.assertEqual(benchmark._nearest_rank_percentile(values, 95), 95.0)
        self.assertEqual(benchmark._nearest_rank_percentile(values, 99), 99.0)
        self.assertEqual(benchmark._nearest_rank_percentile(values, 100), 100.0)

    def test_nearest_rank_ignores_nonfinite_and_empty_values(self):
        values = [None, math.nan, math.inf, -math.inf, 1.0, 2.0, 3.0, 4.0, 100.0]
        self.assertEqual(benchmark._nearest_rank_percentile(values, 50), 3.0)
        self.assertEqual(benchmark._nearest_rank_percentile(values, 99), 100.0)
        self.assertIsNone(
            benchmark._nearest_rank_percentile([None, math.nan, math.inf], 99)
        )


class DevelopmentDecisionTests(unittest.TestCase):
    def test_complete_dev100_passes_all_prespecified_thresholds(self):
        # Exactly 95 pass at 240 s and jointly. Exactly 99 pass at 440 s.
        # A single slow outlier remains above nearest-rank p99 for n=100.
        runtimes = [0.5] * 99 + [20.0]
        decision = benchmark._decision(
            decision_rows(
                noisy_240_failures=range(5),
                noisy_440_failures={0},
                runtimes_s=runtimes,
            ),
            440.0,
        )

        self.assertEqual(decision["decision_code"], "PASS_ENGINEERING_SCREEN_ONLY")
        self.assertTrue(decision["engineering_screen_eligible"])
        self.assertTrue(decision["structural_observability"]["go"])
        screen = decision["recorded_noisy_estimator_screen"]
        self.assertTrue(screen["go"])
        self.assertEqual(screen["success_240s_count"], 95)
        self.assertEqual(screen["required_success_240s_count"], 95)
        self.assertEqual(screen["joint_success_240s_and_440s_count"], 95)
        self.assertEqual(
            screen["required_joint_success_240s_and_440s_count"], 95
        )
        self.assertEqual(screen["success_count"], 99)
        self.assertEqual(screen["required_success_count"], 99)
        self.assertEqual(screen["runtime_p99_nearest_rank_s"], 0.5)
        self.assertFalse(decision["training_authorized"])
        self.assertFalse(decision["final_authorized"])

    def test_each_estimator_screen_can_stop_complete_dev100(self):
        cases = {
            "below_95_percent_at_240s": decision_rows(
                noisy_240_failures=range(6), noisy_440_failures={0}
            ),
            "below_95_percent_joint": decision_rows(
                noisy_240_failures=range(5), noisy_440_failures={5}
            ),
            "below_99_percent_at_440s": decision_rows(
                noisy_240_failures=range(5), noisy_440_failures={0, 1}
            ),
            # The gate is deliberately strict: p99 equal to 2 s does not pass.
            "runtime_p99_at_limit": decision_rows(
                noisy_240_failures=range(5),
                noisy_440_failures={0},
                runtimes_s=[0.5] * 98 + [2.0, 2.0],
            ),
        }
        for label, rows in cases.items():
            with self.subTest(label=label):
                decision = benchmark._decision(rows, 440.0)
                self.assertEqual(decision["decision_code"], "STOP_ESTIMATOR")
                self.assertFalse(decision["recorded_noisy_estimator_screen"]["go"])

    def test_structural_failure_stops_before_estimator_work(self):
        decision = benchmark._decision(
            decision_rows(structural_failures={99}), 440.0
        )
        self.assertEqual(decision["decision_code"], "STOP_GEOMETRY")
        self.assertFalse(decision["structural_observability"]["go"])

    def test_smoke_or_partial_run_reports_observation_but_no_decision(self):
        decision = benchmark._decision(decision_rows(count=2), 440.0)
        self.assertEqual(
            decision["decision_code"], "SMOKE_OR_PARTIAL_COMPLETE_NO_DECISION"
        )
        self.assertFalse(decision["engineering_screen_eligible"])
        self.assertTrue(decision["structural_observability"]["observed_pass"])
        self.assertIsNone(decision["structural_observability"]["go"])
        self.assertTrue(
            decision["recorded_noisy_estimator_screen"]["observed_pass"]
        )
        self.assertIsNone(decision["recorded_noisy_estimator_screen"]["go"])
        self.assertFalse(decision["training_authorized"])
        self.assertFalse(decision["final_authorized"])


class SearchSeedAndSettingsTests(unittest.TestCase):
    def test_search_seed_mapping_depends_only_on_problem_and_prefix(self):
        prefixes = (30.0, 60.0, 120.0, 240.0, 440.0)
        mapping = benchmark._search_seed_map(prefixes)
        contract = {
            "episode_seeds": list(range(45_000, 45_100)),
            "search_seed_policy": {
                "independent_of_episode_seed": True,
                "mapping": mapping,
            },
        }

        self.assertEqual(
            benchmark._candidate_seed(contract, "recorded_noisy", 30.0), 19_001
        )
        self.assertEqual(
            benchmark._candidate_seed(contract, "structural_noiseless", 30.0),
            19_002,
        )
        self.assertEqual(
            benchmark._candidate_seed(contract, "recorded_noisy", 440.0), 19_009
        )
        self.assertEqual(
            benchmark._candidate_seed(contract, "structural_noiseless", 440.0),
            19_010,
        )
        first_episode_result = benchmark._candidate_seed(
            contract, "recorded_noisy", 240.0
        )
        contract["episode_seeds"] = list(range(50_000, 50_100))
        self.assertEqual(
            benchmark._candidate_seed(contract, "recorded_noisy", 240.0),
            first_episode_result,
        )

    def test_resolved_settings_enforce_episode_bounds_and_final_seed_guard(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            output = root / "output"

            valid = benchmark._parse_args(
                [
                    "--evaluation-dir",
                    str(evaluation),
                    "--output-dir",
                    str(output),
                    "--episode-start",
                    "99",
                    "--episodes",
                    "1",
                ]
            )
            settings = benchmark._resolved_settings(valid)
            self.assertEqual(settings["episode_indices"], (99,))
            self.assertEqual(settings["episode_seeds"], (45_099,))

            for start, count in ((-1, 1), (99, 2), (0, 0)):
                with self.subTest(start=start, count=count):
                    invalid = benchmark._parse_args(
                        [
                            "--evaluation-dir",
                            str(evaluation),
                            "--output-dir",
                            str(output),
                            "--episode-start",
                            str(start),
                            "--episodes",
                            str(count),
                        ]
                    )
                    with self.assertRaises(ValueError):
                        benchmark._resolved_settings(invalid)

            final_seed_args = benchmark._parse_args(
                [
                    "--evaluation-dir",
                    str(evaluation),
                    "--output-dir",
                    str(output),
                    "--episodes",
                    "1",
                ]
            )
            with mock.patch.object(benchmark.v19, "V181_DEV_SEED_START", 50_000):
                with self.assertRaises(PermissionError):
                    benchmark._resolved_settings(final_seed_args)

    def test_smoke_defaults_to_two_development_episodes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            args = benchmark._parse_args(
                [
                    "--evaluation-dir",
                    str(evaluation),
                    "--output-dir",
                    str(root / "output"),
                    "--smoke",
                ]
            )
            settings = benchmark._resolved_settings(args)
            self.assertEqual(settings["episode_indices"], (0, 1))
            self.assertEqual(settings["episode_seeds"], (45_000, 45_001))
            self.assertEqual(settings["config"].coarse_candidates, 512)
            self.assertEqual(settings["config"].local_starts, 12)


if __name__ == "__main__":
    unittest.main()
