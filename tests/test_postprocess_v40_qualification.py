from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(MODULE_ROOT))

import postprocess_v40_qualification as post  # noqa: E402


MINI_SPEC = post.CampaignSpec(
    seeds=(49_900, 49_901),
    horizon_actions=4,
    action_interval_s=2.0,
    plant_interval_s=0.1,
    nominal_terminal_min=2,
    nominal_tail80_min=2,
    nominal_lock_min=2,
    robust_terminal_min=1,
    robust_tail80_min=1,
)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def environment_config(horizon: int) -> Dict[str, Any]:
    return {
        "max_steps": horizon,
        "action_dt": 2.0,
        "sub_dt": 0.1,
        "rl_speed_delta_per_step": 0.4,
        "rl_yaw_per_step_deg": 20.0,
        "rl_pitch_per_step_deg": 14.0,
        "max_yaw_rate_deg_s": 60.0,
        "max_pitch_rate_deg_s": 45.0,
    }


def exact_gate(
    gate_after: np.ndarray,
    phase: np.ndarray,
    localization: np.ndarray,
    time_s: np.ndarray,
    action_start_s: np.ndarray,
) -> Dict[str, Any]:
    previous = np.concatenate([np.asarray([False]), gate_after[:-1]])
    transitions = np.flatnonzero(gate_after & ~previous)
    phase_indices = np.flatnonzero(phase)
    transition_errors = localization[transitions]
    start_errors = localization[phase_indices - 1] if phase_indices.size else np.asarray([])
    end_errors = localization[phase_indices] if phase_indices.size else np.asarray([])

    def first(values: np.ndarray) -> Any:
        return None if not values.size else float(values[0])

    return {
        "transition_count": int(transitions.size),
        "first_transition_time_s": first(time_s[transitions]),
        "false_transition_count": int(np.sum(transition_errors >= 7.0)),
        "locked_action_count": int(phase_indices.size),
        "first_locked_action_time_s": (
            None if not phase_indices.size else float(action_start_s[phase_indices[0]])
        ),
        "false_locked_action_start_count": int(np.sum(start_errors >= 7.0)),
        "false_locked_action_end_count": int(np.sum(end_errors >= 7.0)),
    }


def requested_actions(horizon: int, seed: int, policy: str) -> np.ndarray:
    sign = 1.0 if policy == post.POLICY_ACTIVE else -1.0
    base = np.asarray(
        [
            [0.5, 0.2 * sign, -0.1],
            [0.2, -0.4 * sign, 0.3],
            [-0.1, 0.5 * sign, 0.2],
            [0.0, -0.2 * sign, -0.3],
        ],
        dtype=np.float64,
    )
    return np.roll(base[:horizon], seed % horizon, axis=0)


def make_record_and_trace(
    *,
    seed: int,
    index: int,
    arm: Mapping[str, str],
    spec: post.CampaignSpec,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    policy = arm["policy"]
    dynamics = arm["dynamics"]
    current = arm["current"]
    horizon = spec.horizon_actions
    time_s = np.arange(1, horizon + 1, dtype=np.float64) * spec.action_interval_s
    action_start_s = time_s - spec.action_interval_s

    # Active nominal always passes.  Other cells deliberately contain both
    # favorable and unfavorable outcomes so paired effects and interactions
    # are nontrivial in the synthetic fixture.
    success = policy == post.POLICY_ACTIVE
    if dynamics == post.DYNAMICS_LOW_ORDER and current == post.CURRENT_VISIBLE and seed == 49_901:
        success = False
    localization_value = 4.0 if success else 9.0
    localization = np.full(horizon, localization_value, dtype=np.float64)
    formation = np.full(horizon, 2.0, dtype=np.float64)
    joint = (localization < 7.0) & (formation < 8.0)

    locked = policy == post.POLICY_ACTIVE or seed == 49_900
    gate_after = np.zeros(horizon, dtype=bool)
    phase = np.zeros(horizon, dtype=bool)
    if locked:
        gate_after[1:] = True
        phase[2:] = True
    exact = exact_gate(gate_after, phase, localization, time_s, action_start_s)

    requested = requested_actions(horizon, seed, policy)
    if dynamics == post.DYNAMICS_LOW_ORDER:
        delayed = np.vstack([np.zeros((1, 3)), requested[:-1]])
    else:
        delayed = requested.copy()
    scales = np.asarray([0.2, 10.0, 7.0])
    delayed_physical = delayed * scales
    if dynamics == post.DYNAMICS_LOW_ORDER:
        executed = post._expected_executed_state(
            delayed_physical,
            action_interval_s=spec.action_interval_s,
        )
    elif current == post.CURRENT_NONE:
        executed = np.zeros_like(delayed_physical)
    else:
        executed = delayed_physical.copy()

    water = np.zeros((horizon, 3), dtype=np.float64)
    if current == post.CURRENT_VISIBLE:
        water[:, 0] = 0.3
    body = np.tile(np.asarray([1.0, 0.1, 0.0]), (horizon, 1))
    ground = body + water
    current_rms = float(np.sqrt(np.mean(np.sum(water * water, axis=1))))
    delay_rms = float(np.sqrt(np.mean(np.square(requested - delayed))))
    speed = np.linalg.norm(ground, axis=1)

    audit_checks = np.ones(horizon, dtype=bool)
    trace: Dict[str, np.ndarray] = {
        "time_s": time_s,
        "action_start_time_s": action_start_s,
        "phase_track": phase.astype(float),
        "gate_locked_after_update": gate_after.astype(float),
        "action_speed": requested[:, 0],
        "action_yaw": requested[:, 1],
        "action_pitch": requested[:, 2],
        "formation_error_truth_m": formation,
        "localization_error_m": localization,
        "audit_release_checks_pass": audit_checks.astype(float),
        "plant_requested_action": requested,
        "plant_delivered_action": delayed,
        "plant_executed_rate_state": executed,
        "water_current_mps": water,
        "body_velocity_through_water_mps": body,
        "ground_velocity_mps": ground,
    }
    noise_hash = digest_text(f"noise-{seed}")
    current_hash = digest_text(f"current-{seed}")
    record: Dict[str, Any] = {
        "version": post.EXPERIMENT_VERSION,
        "episode_index": index,
        "episode_seed": seed,
        "arm": arm["name"],
        "policy_name": policy,
        "dynamics_name": dynamics,
        "current_name": current,
        "source_name": "both_leaders",
        "source_mask": [True, True],
        "active_source_count": 2,
        "action_count": horizon,
        "terminal_joint_success": bool(joint[-1]),
        "tail80_joint_success": bool(float(np.mean(joint)) >= 0.8),
        "dwell15_joint_success": bool(np.all(joint)),
        "tail50_joint_occupancy": float(np.mean(joint)),
        "terminal_localization_error_m": float(localization[-1]),
        "terminal_formation_error_m": float(formation[-1]),
        "mean_squared_action": float(np.mean(np.sum(requested * requested, axis=1))),
        "maximum_combined_decision_runtime_s": 0.2 + 0.01 * (seed - 49_900),
        "noise_tape_sha256": noise_hash,
        "current_tape_sha256": current_hash,
        "initial_truth_m": [100.0 + seed, -25.0, -40.0],
        "mission_support": {
            "center_m": [0.0, 0.0, -30.0],
            "radius_min_m": 50.0,
            "radius_max_m": 500.0,
        },
        "gate": {
            "ever_locked": bool(locked),
            "first_track_action_time_s": exact["first_locked_action_time_s"],
            "audit_release_violation_count": 0,
            "exact": exact,
        },
        "plant": {
            "parameters": dict(post.PLANT_PARAMETER_LEVELS[dynamics]),
            "bottom_track_current_visible": current == post.CURRENT_VISIBLE,
            "planner_uses_execution_model": False,
            "planner_uses_current_model": False,
            "requested_delivered_action_rms": delay_rms,
            "current_speed_rms_mps": current_rms,
            "minimum_ground_speed_mps": float(np.min(speed)),
            "ground_speed_below_0p05_action_count": int(np.sum(speed < 0.05)),
        },
    }
    return record, trace


def build_campaign(root: Path, spec: post.CampaignSpec = MINI_SPEC) -> Path:
    campaign = root / "campaign"
    snapshot = campaign / "control" / "source_snapshot"
    source_hashes: Dict[str, str] = {}
    for name in post.SOURCE_NAMES:
        content = f"synthetic frozen source: {name}\n"
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        source_hashes[name] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    metadata = root / "metadata.json"
    write_json(metadata, {"environment_config": environment_config(spec.horizon_actions)})
    contract = {
        "schema_version": 1,
        "runner_version": post.RUNNER_VERSION,
        "experiment_version": post.EXPERIMENT_VERSION,
        "created_at_utc": "2026-08-31T00:00:00Z",
        "purpose": "synthetic unit-test fixture",
        "smoke": False,
        "episodes": len(spec.seeds),
        "episode_start": 0,
        "seeds": list(spec.seeds),
        "arms": list(post.EXPECTED_ARMS),
        "expected_runs": len(spec.seeds) * len(post.EXPECTED_ARMS),
        "publication_settings": True,
        "estimator_config": {
            "coarse_candidates": 4096,
            "coarse_sweeps": 2,
            "local_starts": 48,
            "gate_mode": "raw",
            "candidate_radial_distribution": "uniform_radius",
        },
        "factorial_contract": post._expected_factorial_contract(spec),
        "fixed_horizon_actions": spec.horizon_actions,
        "action_interval_s": spec.action_interval_s,
        "plant_interval_s": spec.plant_interval_s,
        "environment_metadata_path": str(metadata),
        "environment_metadata_sha256": post._sha256(metadata),
        "fresh_qualification_audit": {
            "range": [49_900, 49_999],
            "fresh": True,
            "finding_count": 0,
            "findings": [],
        },
        "sealed_final_range": list(post.SEALED_FINAL_RANGE),
        "source_sha256": source_hashes,
        "source_manifest_sha256": hashlib.sha256(
            post._canonical(source_hashes).encode("utf-8")
        ).hexdigest(),
        "package_versions": {"python": "synthetic", "numpy": np.__version__},
        "retuning_allowed": False,
    }
    write_json(campaign / "control" / "campaign_contract.json", contract)

    for index, seed in enumerate(spec.seeds):
        for arm in post.EXPECTED_ARMS:
            record, trace = make_record_and_trace(
                seed=seed,
                index=index,
                arm=arm,
                spec=spec,
            )
            result_path = (
                campaign
                / "episode_results"
                / arm["name"]
                / f"episode_{index:04d}_seed_{seed}.json"
            )
            trace_path = (
                campaign
                / "traces_npz"
                / arm["name"]
                / f"episode_{index:04d}_seed_{seed}.npz"
            )
            write_json(result_path, record)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace_path, **trace)

    # Deliberately false generated summaries prove that the postprocessor does
    # not trust or read them.
    write_json(campaign / "campaign_summary.json", {"integrity_valid": False, "fake": True})
    write_json(campaign / "decision.json", {"decision": "FAKE"})
    return campaign


class ReadOnlyPostprocessorTests(unittest.TestCase):
    def test_released_metadata_projection_matches_embedded_hashes(self) -> None:
        repository_root = MODULE_ROOT.parent
        metadata = (
            repository_root
            / "code"
            / "experiments_v18_1_guard_ablation_dev_3seed"
            / "evaluations"
            / "dev100_seed_28001_range_45000_45099"
            / "metadata.json"
        )
        self.assertEqual(
            post._sha256(metadata),
            post.PUBLIC_ENVIRONMENT_METADATA_SHA256,
        )
        document = json.loads(metadata.read_text(encoding="utf-8"))
        config_hash = hashlib.sha256(
            post._canonical(document["environment_config"]).encode("utf-8")
        ).hexdigest()
        self.assertEqual(config_hash, post.FROZEN_ENVIRONMENT_CONFIG_SHA256)

    def test_complete_fixture_recomputes_results_and_writes_compact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = build_campaign(root)
            result = post.analyze_campaign(
                campaign,
                spec=MINI_SPEC,
                bootstrap_replicates=100,
            )
            self.assertTrue(result["integrity_valid"])
            self.assertFalse(result["integrity_checks"]["campaign_summary_files_read"])
            self.assertEqual(len(result["by_arm"]), 8)
            contrast = result["paired_main_effects"][
                "active_minus_fixed_within_execution_current"
            ]["kinematic__no_current"]
            self.assertGreater(contrast["terminal_joint_success"]["risk_difference"], 0.0)
            self.assertIn("ever_locked", contrast)
            self.assertIn("tail50_mean_localization_error_m_treatment_minus_reference", contrast)
            self.assertIn("maximum_combined_decision_runtime_s_treatment_minus_reference", contrast)
            interactions = result["factorial_interactions_and_difference_in_differences"]
            self.assertIn("three_way_policy_x_dynamics_x_current", interactions)
            self.assertIn(
                "first_TRACK_action_time_s",
                interactions["policy_x_current_within_dynamics"]["kinematic"],
            )

            dynamic_name = post.arm_name(
                post.POLICY_ACTIVE,
                post.DYNAMICS_LOW_ORDER,
                post.CURRENT_NONE,
            )
            dynamic = result["command_execution_diagnostics"]["by_arm"][dynamic_name]
            self.assertGreater(
                dynamic["requested_to_delayed_command"]["normalized_RMS_all_channels"],
                0.0,
            )
            self.assertTrue(dynamic["first_order_response"]["applicable"])
            self.assertGreater(
                dynamic["first_order_response"][
                    "delayed_command_to_executed_state_RMS_by_channel"
                ][0],
                0.0,
            )
            kinematic_name = post.arm_name(
                post.POLICY_ACTIVE,
                post.DYNAMICS_KINEMATIC,
                post.CURRENT_NONE,
            )
            kinematic = result["command_execution_diagnostics"]["by_arm"][kinematic_name]
            self.assertFalse(kinematic["first_order_response"]["applicable"])
            self.assertTrue(
                kinematic["kinematic_physical_rate_reconstruction"][
                    "physical_rate_commands_reconstructed"
                ]
            )

            artifact_dir = root / "analysis"
            files = post.write_publication_artifacts(
                result,
                artifact_dir,
                campaign_directory=campaign,
            )
            self.assertEqual(
                set(files),
                {
                    "full_json",
                    "publication_json",
                    "arms_csv",
                    "interaction_plot_csv",
                    "episode_rows_csv",
                },
            )
            with Path(files["arms_csv"]).open(encoding="utf-8", newline="") as handle:
                arm_rows = list(csv.DictReader(handle))
            self.assertEqual(len(arm_rows), 8)
            with Path(files["episode_rows_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                episode_rows = list(csv.DictReader(handle))
            self.assertEqual(len(episode_rows), 16)
            with Path(files["publication_json"]).open(encoding="utf-8") as handle:
                compact = json.load(handle)
            self.assertEqual(len(compact["arms"]), 8)
            self.assertIn("paired_main_effects", compact)
            full_text = Path(files["full_json"]).read_text(encoding="utf-8")
            self.assertNotIn(str(root.resolve()), full_text)
            with self.assertRaises(post.ValidationError):
                post.write_publication_artifacts(
                    result,
                    campaign / "derived",
                    campaign_directory=campaign,
                )

    def test_missing_first_track_is_allowed_only_for_no_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = build_campaign(Path(temporary))
            # The untouched fixture already contains valid no-lock records with
            # null first-TRACK time.
            post.analyze_campaign(campaign, spec=MINI_SPEC, bootstrap_replicates=10)

            arm = post.arm_name(
                post.POLICY_ACTIVE,
                post.DYNAMICS_KINEMATIC,
                post.CURRENT_NONE,
            )
            path = campaign / "episode_results" / arm / "episode_0000_seed_49900.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["gate"]["first_track_action_time_s"] = None
            record["gate"]["exact"]["first_locked_action_time_s"] = None
            write_json(path, record)
            with self.assertRaisesRegex(post.ValidationError, "lacks first TRACK-action time"):
                post.analyze_campaign(campaign, spec=MINI_SPEC, bootstrap_replicates=10)

    def test_source_snapshot_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = build_campaign(Path(temporary))
            path = campaign / "control" / "source_snapshot" / post.SOURCE_NAMES[0]
            path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(post.ValidationError, "source snapshot hash mismatch"):
                post.analyze_campaign(campaign, spec=MINI_SPEC, bootstrap_replicates=10)

    def test_nonfinite_required_outcome_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = build_campaign(Path(temporary))
            arm = post.EXPECTED_ARM_NAMES[0]
            path = campaign / "episode_results" / arm / "episode_0000_seed_49900.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["terminal_localization_error_m"] = None
            write_json(path, record)
            with self.assertRaisesRegex(post.ValidationError, "terminal_localization_error_m is not numeric"):
                post.analyze_campaign(campaign, spec=MINI_SPEC, bootstrap_replicates=10)

    def test_contract_must_contain_exact_seed_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = build_campaign(Path(temporary))
            path = campaign / "control" / "campaign_contract.json"
            contract = json.loads(path.read_text(encoding="utf-8"))
            contract["seeds"] = [49_900, 49_902]
            write_json(path, contract)
            with self.assertRaisesRegex(post.ValidationError, "contract field seeds differs"):
                post.analyze_campaign(campaign, spec=MINI_SPEC, bootstrap_replicates=10)

    def test_approved_public_metadata_projection_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = build_campaign(Path(temporary))
            contract_path = campaign / "control" / "campaign_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            metadata = Path(contract["environment_metadata_path"])
            metadata_document = json.loads(metadata.read_text(encoding="utf-8"))
            config_hash = hashlib.sha256(
                post._canonical(metadata_document["environment_config"]).encode("utf-8")
            ).hexdigest()
            contract["environment_metadata_sha256"] = "synthetic-private-original"
            write_json(contract_path, contract)
            with (
                mock.patch.object(
                    post,
                    "ORIGINAL_ENVIRONMENT_METADATA_SHA256",
                    "synthetic-private-original",
                ),
                mock.patch.object(
                    post,
                    "PUBLIC_ENVIRONMENT_METADATA_SHA256",
                    post._sha256(metadata),
                ),
                mock.patch.object(
                    post,
                    "FROZEN_ENVIRONMENT_CONFIG_SHA256",
                    config_hash,
                ),
            ):
                result = post.analyze_campaign(
                    campaign,
                    spec=MINI_SPEC,
                    bootstrap_replicates=10,
                )
            status = result["integrity_checks"]["environment_metadata"]
            self.assertEqual(
                status["verification_mode"],
                "approved_path_sanitized_public_projection",
            )
            self.assertFalse(status["hash_matches_contract"])
            self.assertTrue(
                status["public_projection_preserves_frozen_environment_config"]
            )

    def test_paired_newcombe_method10_reference_values(self) -> None:
        observed = post.paired_newcombe_method10_interval(20, 12, 2, 16)
        for value, expected in zip(observed, (0.2, 0.0562, 0.3292)):
            self.assertAlmostEqual(value, expected, delta=5.0e-5)
        boundary = post.paired_newcombe_method10_interval(96, 4, 0, 0)
        for value, expected in zip(
            boundary,
            (0.04, -0.0042808501, 0.0983707144),
        ):
            self.assertAlmostEqual(value, expected, delta=5.0e-10)


if __name__ == "__main__":
    unittest.main()
