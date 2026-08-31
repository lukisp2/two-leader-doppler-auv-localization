import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from uuv_v18_resampling_guard import UUV3DConfig
import run_v39_planner_component_ablation as runner
import uuv_v39_planner_component_ablation as v39
import uuv_v38_leader_source_ablation as v38


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
        self.pF = np.asarray([-200.0, 100.0, -120.0])


def _minimal_history():
    count = 3
    return v38.MaskedDopplerHistory(
        source_mask=(True, True),
        source_indices=(0, 1),
        t_s=np.asarray([1.0, 2.0, 3.0]),
        dead_reckoned_displacement_m=np.zeros((count, 3)),
        leader_position_m=np.zeros((count, 2, 3)),
        leader_velocity_mps=np.zeros((count, 2, 3)),
        follower_velocity_measured_mps=np.zeros((count, 3)),
        doppler_measured_mps=np.zeros((count, 2)),
        historical_pf_gate_factor=np.ones((count, 2)),
    )


def _estimator_with_competing_modes():
    current = np.asarray([-170.0, 80.0, -115.0])
    mode = SimpleNamespace(
        initial_position_m=current.copy(),
        local_covariance_valid=False,
        local_covariance_m2=np.full((3, 3), np.nan),
    )
    primary = SimpleNamespace(
        modes=[
            SimpleNamespace(initial_position_m=current + [20.0, 0.0, 0.0]),
            SimpleNamespace(initial_position_m=current + [0.0, 20.0, 0.0]),
        ]
    )
    confirmation = SimpleNamespace(
        modes=[
            SimpleNamespace(initial_position_m=current + [0.0, 0.0, 20.0])
        ]
    )
    state = SimpleNamespace(
        initial_position_m=current,
        latest_mode=mode,
        latest_global_evidence=SimpleNamespace(
            primary=primary,
            confirmation=confirmation,
        ),
    )
    support = v38.MissionSupport(
        center_m=np.asarray([0.0, 0.0, -52.5]),
        radius_min_m=120.0,
        radius_max_m=350.0,
    )
    return SimpleNamespace(state=state, support=support)


class V39ComponentAblationTests(unittest.TestCase):
    def _args(self, output, **values):
        defaults = {
            "output_dir": Path(output),
            "smoke": False,
            "resume": False,
            "episodes": None,
            "episode_start": 0,
            "coarse_candidates": None,
            "coarse_sweeps": None,
            "local_starts": None,
            "progress_every": 1,
        }
        defaults.update(values)
        return argparse.Namespace(**defaults)

    def test_four_arm_contract_is_unique(self):
        self.assertEqual(len(v39.ARM_NAMES), 4)
        self.assertEqual(len(set(v39.ARM_NAMES)), 4)
        self.assertEqual(
            set(v39.ACTIVE_ARMS),
            {v39.ARM_FULL, v39.ARM_NO_PAIR, v39.ARM_BEST_ONLY},
        )

    def test_planner_interventions_change_only_frozen_targets(self):
        full = v39.planner_config_for_arm(v39.ARM_FULL)
        no_pair = v39.planner_config_for_arm(v39.ARM_NO_PAIR)
        best = v39.planner_config_for_arm(v39.ARM_BEST_ONLY)
        self.assertIsNotNone(full)
        self.assertIsNotNone(no_pair)
        self.assertIsNotNone(best)
        full_values = full.to_dict()
        no_pair_values = no_pair.to_dict()
        best_values = best.to_dict()
        differing_pair = {
            key
            for key in full_values
            if full_values[key] != no_pair_values[key]
        }
        differing_best = {
            key
            for key in full_values
            if full_values[key] != best_values[key]
        }
        self.assertEqual(differing_pair, {"pair_weight"})
        self.assertEqual(no_pair.pair_weight, 0.0)
        self.assertEqual(differing_best, {"maximum_hypotheses"})
        self.assertEqual(best.maximum_hypotheses, 1)
        self.assertIsNone(v39.planner_config_for_arm(v39.ARM_RANDOM))

    def test_zero_pair_weight_is_legal_but_negative_is_rejected(self):
        self.assertEqual(v39.PlannerConfig(pair_weight=0.0).pair_weight, 0.0)
        with self.assertRaises(ValueError):
            v39.PlannerConfig(pair_weight=-1e-9)

    def test_best_only_retains_current_best_and_full_sets_match(self):
        history = _minimal_history()
        estimator = _estimator_with_competing_modes()
        full = v39.AuditedBeliefPlanner(
            v39.planner_config_for_arm(v39.ARM_FULL),
            v39.SOURCE_MASK,
        )
        no_pair = v39.AuditedBeliefPlanner(
            v39.planner_config_for_arm(v39.ARM_NO_PAIR),
            v39.SOURCE_MASK,
        )
        best = v39.AuditedBeliefPlanner(
            v39.planner_config_for_arm(v39.ARM_BEST_ONLY),
            v39.SOURCE_MASK,
        )
        full_values = full._hypotheses(estimator, history)
        no_pair_values = no_pair._hypotheses(estimator, history)
        best_values = best._hypotheses(estimator, history)
        np.testing.assert_array_equal(full_values, no_pair_values)
        self.assertEqual(
            full.last_hypothesis_sha256,
            no_pair.last_hypothesis_sha256,
        )
        self.assertGreater(full_values.shape[0], 1)
        self.assertEqual(best_values.shape, (1, 3))
        np.testing.assert_array_equal(
            best_values[0],
            estimator.state.initial_position_m,
        )

    def test_random_policy_is_reproducible_and_truth_independent(self):
        left_env = _PlannerEnv()
        right_env = _PlannerEnv()
        right_env.pF += np.asarray([1e6, -2e6, 3e6])
        left = v39.RandomFeasiblePlanner(48_998)
        right = v39.RandomFeasiblePlanner(48_998)
        left_actions = []
        right_actions = []
        for time_s in (30.0, 32.0, 34.0, 36.0, 38.0):
            left_actions.append(left.action(left_env, time_s=time_s).action)
            right_actions.append(right.action(right_env, time_s=time_s).action)
        np.testing.assert_array_equal(left_actions, right_actions)
        self.assertEqual(left.candidate_indices, right.candidate_indices)
        self.assertEqual(left.tape_sha256(), right.tape_sha256())
        self.assertTrue(
            all(
                0 <= index < count
                for index, count in zip(
                    left.candidate_indices,
                    left.candidate_counts,
                )
            )
        )

    def test_reserved_range_and_frozen_development_block(self):
        v39.assert_seed_allowed(48_800)
        v39.assert_seed_allowed(48_899)
        for seed in (49_900, 50_000, 50_999):
            with self.assertRaises(PermissionError):
                v39.assert_seed_allowed(seed)
        with tempfile.TemporaryDirectory() as directory:
            settings = runner._settings(self._args(directory))
        self.assertEqual(settings["seeds"], list(range(48_800, 48_900)))
        self.assertTrue(settings["publication_settings"])

    def test_smoke_uses_two_full_settings_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = runner._settings(
                self._args(directory, smoke=True)
            )
        self.assertEqual(settings["seeds"], [48_998, 48_999])
        self.assertEqual(settings["episodes"], 2)
        self.assertTrue(settings["publication_settings"])

    def test_freshness_audit_detects_prior_selected_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "experiments_old"
            old.mkdir()
            (old / "row_seed_48807.json").write_text(
                json.dumps({"episode_seed": 48_807}),
                encoding="utf-8",
            )
            output = root / "experiments_new"
            output.mkdir()
            audit = runner._fresh_seed_audit(
                root,
                output,
                list(range(48_800, 48_900)),
            )
        self.assertFalse(audit["fresh"])
        self.assertEqual(audit["finding_count"], 1)
        self.assertEqual(audit["findings"][0]["seeds"], [48_807])

    def test_material_gate_uses_five_percentage_points_and_safety(self):
        contrast = {
            "terminal_success_rate_difference_full_minus_comparator": 0.04,
            "tail80_success_rate_difference_full_minus_comparator": 0.05,
        }
        full = {
            "unsafe_transition_count": 0,
            "unsafe_track_start_count": 0,
            "unsafe_track_end_count": 0,
        }
        comparator = dict(full)
        decision = runner._material_decision(
            contrast,
            full,
            comparator,
            positive="YES",
            negative="NO",
            integrity_valid=True,
        )
        self.assertEqual(decision, "YES")
        full["unsafe_track_end_count"] = 1
        decision = runner._material_decision(
            contrast,
            full,
            comparator,
            positive="YES",
            negative="NO",
            integrity_valid=True,
        )
        self.assertEqual(decision, "NO")


if __name__ == "__main__":
    unittest.main()
