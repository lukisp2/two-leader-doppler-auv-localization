#!/usr/bin/env python3
"""Read-only, summary-independent public postprocessor for frozen V40.

The program reads the immutable campaign contract, per-arm JSON records, and
NPZ traces.  It deliberately does not read ``campaign_summary.json``,
``decision.json``, or ``episode_arm_summary.csv``.  All reported metrics,
contrasts, interactions, and qualification screens are reconstructed from the
episode records after structural and numerical validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ANALYSIS_VERSION = "v40_readonly_postprocessor_1.0"
RUNNER_VERSION = "v40_factorial_runner_1.0"
EXPERIMENT_VERSION = "v40_dynamic_current_factorial_1.0"

POLICY_FIXED = "fixed_s_turn"
POLICY_ACTIVE = "belief_active"
POLICIES = (POLICY_FIXED, POLICY_ACTIVE)
DYNAMICS_KINEMATIC = "kinematic"
DYNAMICS_LOW_ORDER = "low_order_dynamic"
DYNAMICS = (DYNAMICS_KINEMATIC, DYNAMICS_LOW_ORDER)
CURRENT_NONE = "no_current"
CURRENT_VISIBLE = "bottom_track_visible"
CURRENTS = (CURRENT_NONE, CURRENT_VISIBLE)

QUALIFICATION_SEEDS = tuple(range(49_900, 50_000))
SEALED_FINAL_RANGE = (50_000, 50_999)
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 40_049_900

# The frozen private metadata included execution paths that cannot be released.
# The public projection preserves ``environment_config`` byte-for-value while
# removing those paths; both file hashes and the canonical configuration hash
# are recorded in docs/SOURCE_PROVENANCE.md.
ORIGINAL_ENVIRONMENT_METADATA_SHA256 = "b486d280e57661f4b526b926e0f8e32a5373b99f585de09e3354d87b3342dfe7"
PUBLIC_ENVIRONMENT_METADATA_SHA256 = "33aff4c6794c9eab2d92bd4c9aa2f3ff3b89b4f3fc121904df40124f584b626f"
FROZEN_ENVIRONMENT_CONFIG_SHA256 = "74e83dae26f198604803d50f2da3c14453118e9418e41dc3316684391eb7e211"

TERMINAL_LOCALIZATION_GATE_M = 7.0
TERMINAL_FORMATION_GATE_M = 8.0
TAIL_WINDOW_ACTIONS = 50
DWELL_ACTIONS = 15

PLANT_PARAMETER_LEVELS: Mapping[str, Mapping[str, Any]] = {
    DYNAMICS_KINEMATIC: {
        "command_delay_actions": 0,
        "command_delay_s": 0.0,
        "surge_acceleration_time_constant_s": 0.0,
        "yaw_rate_time_constant_s": 0.0,
        "pitch_rate_time_constant_s": 0.0,
    },
    DYNAMICS_LOW_ORDER: {
        "command_delay_actions": 1,
        "command_delay_s": 2.0,
        "surge_acceleration_time_constant_s": 5.0,
        "yaw_rate_time_constant_s": 2.0,
        "pitch_rate_time_constant_s": 3.0,
    },
}

SOURCE_NAMES = (
    "EXPERIMENT_PROTOCOL_V40_DYNAMIC_PLANT_STRESS.md",
    "uuv_v40_dynamic_plant_stress.py",
    "run_v40_dynamic_plant_stress.py",
    "audit_v40_dynamic_plant_stress.py",
    "tests/test_uuv_v40_dynamic_plant_stress.py",
    "uuv_v38_leader_source_ablation.py",
    "uuv_v24_audited_gate.py",
    "uuv_v22_active_acquisition.py",
    "uuv_v21_causal_lock.py",
    "uuv_v20_positioning_ablation.py",
    "uuv_v19_observability.py",
    "uuv_v18_resampling_guard.py",
    "uuv_v11_online.py",
    "uuv_v11_rng.py",
)

TRACE_VECTOR_FIELDS = (
    "plant_requested_action",
    "plant_delivered_action",
    "plant_executed_rate_state",
    "water_current_mps",
    "body_velocity_through_water_mps",
    "ground_velocity_mps",
)

BINARY_ENDPOINTS = (
    "terminal_joint_success",
    "tail80_joint_success",
    "ever_locked",
)
CONTINUOUS_ENDPOINTS = (
    "terminal_localization_error_m",
    "terminal_formation_error_m",
    "tail50_mean_localization_error_m",
    "tail50_mean_formation_error_m",
    "maximum_combined_decision_runtime_s",
    "unsafe_track_start_count",
    "unsafe_track_end_count",
)


class ValidationError(RuntimeError):
    """Raised when a campaign violates the frozen V40 contract."""


def arm_name(policy: str, dynamics: str, current: str) -> str:
    return f"{policy}__{dynamics}__{current}"


def arm_dictionary(policy: str, dynamics: str, current: str) -> Dict[str, str]:
    return {
        "name": arm_name(policy, dynamics, current),
        "policy": policy,
        "dynamics": dynamics,
        "current": current,
    }


EXPECTED_ARMS: Tuple[Dict[str, str], ...] = tuple(
    arm_dictionary(policy, dynamics, current)
    for policy in POLICIES
    for dynamics in DYNAMICS
    for current in CURRENTS
)
EXPECTED_ARM_NAMES = tuple(arm["name"] for arm in EXPECTED_ARMS)
ARM_LOOKUP = {arm["name"]: arm for arm in EXPECTED_ARMS}


@dataclass(frozen=True)
class CampaignSpec:
    """Structural expectations; the CLI always uses QUALIFICATION_SPEC."""

    seeds: Tuple[int, ...]
    horizon_actions: int
    action_interval_s: float
    plant_interval_s: float
    nominal_terminal_min: int
    nominal_tail80_min: int
    nominal_lock_min: int
    robust_terminal_min: int
    robust_tail80_min: int


QUALIFICATION_SPEC = CampaignSpec(
    seeds=QUALIFICATION_SEEDS,
    horizon_actions=220,
    action_interval_s=2.0,
    plant_interval_s=0.1,
    nominal_terminal_min=95,
    nominal_tail80_min=90,
    nominal_lock_min=95,
    robust_terminal_min=90,
    robust_tail80_min=85,
)


@dataclass
class TraceDiagnostics:
    seed: int
    arm: str
    requested_normalized: np.ndarray
    delayed_normalized: np.ndarray
    requested_physical: np.ndarray
    delayed_physical: np.ndarray
    executed_physical: np.ndarray
    response_applicable: bool
    kinematic_trace_semantics: Optional[str]


def _fail(message: str) -> None:
    raise ValidationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is not numeric") from exc
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result >= 0.0, f"{label} is negative")
    return result


def _strict_bool(value: Any, label: str) -> bool:
    _require(isinstance(value, bool), f"{label} is not a JSON boolean")
    return bool(value)


def _nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} is not an integer",
    )
    _require(value >= 0, f"{label} is negative")
    return int(value)


def _close(left: Any, right: Any, label: str, tolerance: float = 1e-12) -> None:
    a = _finite(left, label)
    b = _finite(right, f"expected {label}")
    _require(math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance), f"{label} differs: {a} != {b}")


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == (length,), f"{label} has shape {array.shape}, expected {(length,)}")
    _require(np.all(np.isfinite(array)), f"{label} contains non-finite values")
    return array


def _validate_source_snapshot(root: Path, contract: Mapping[str, Any]) -> Dict[str, Any]:
    sources = contract.get("source_sha256")
    _require(isinstance(sources, dict), "contract source_sha256 is not an object")
    _require(set(sources) == set(SOURCE_NAMES) and len(sources) == len(SOURCE_NAMES), "contract source list differs from the frozen runner")
    hash_pattern = re.compile(r"^[0-9a-f]{64}$")
    snapshot = root / "control" / "source_snapshot"
    verified: Dict[str, str] = {}
    for name in SOURCE_NAMES:
        expected = sources[name]
        _require(isinstance(expected, str) and bool(hash_pattern.fullmatch(expected)), f"invalid source digest for {name}")
        path = snapshot / name
        _require(path.is_file(), f"missing source snapshot: {name}")
        actual = _sha256(path)
        _require(actual == expected, f"source snapshot hash mismatch: {name}")
        verified[name] = actual
    manifest = hashlib.sha256(_canonical(sources).encode("utf-8")).hexdigest()
    _require(manifest == contract.get("source_manifest_sha256"), "source manifest hash mismatch")
    return {
        "source_file_count": len(verified),
        "source_manifest_sha256": manifest,
        "all_source_snapshot_hashes_match": True,
    }


def _metadata_candidates(
    root: Path,
    contract: Mapping[str, Any],
    override: Optional[Path],
) -> List[Path]:
    candidates: List[Path] = []
    if override is not None:
        candidates.append(override.expanduser().resolve())
    candidates.extend(
        [
            root / "control" / "environment_metadata.json",
            root / "environment_metadata.json",
        ]
    )
    recorded = contract.get("environment_metadata_path")
    if isinstance(recorded, str) and recorded:
        candidates.append(Path(recorded).expanduser())
    unique: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_environment_config(
    root: Path,
    contract: Mapping[str, Any],
    override: Optional[Path],
    spec: CampaignSpec,
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    metadata_path = next(
        (path for path in _metadata_candidates(root, contract, override) if path.is_file()),
        None,
    )
    _require(
        metadata_path is not None,
        "frozen environment metadata is unavailable; pass --environment-metadata",
    )
    assert metadata_path is not None
    actual_hash = _sha256(metadata_path)
    expected_hash = contract.get("environment_metadata_sha256")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("cannot read environment metadata") from exc
    config = metadata.get("environment_config")
    _require(isinstance(config, dict), "environment metadata lacks environment_config")
    config_hash = hashlib.sha256(_canonical(config).encode("utf-8")).hexdigest()
    exact_file = actual_hash == expected_hash
    approved_public_projection = bool(
        expected_hash == ORIGINAL_ENVIRONMENT_METADATA_SHA256
        and actual_hash == PUBLIC_ENVIRONMENT_METADATA_SHA256
        and config_hash == FROZEN_ENVIRONMENT_CONFIG_SHA256
    )
    _require(
        exact_file or approved_public_projection,
        "environment metadata is neither the contract-exact file nor the approved public projection",
    )
    required = (
        "max_steps",
        "action_dt",
        "sub_dt",
        "rl_speed_delta_per_step",
        "rl_yaw_per_step_deg",
        "rl_pitch_per_step_deg",
        "max_yaw_rate_deg_s",
        "max_pitch_rate_deg_s",
    )
    for key in required:
        _require(key in config, f"environment config lacks {key}")
        _finite(config[key], f"environment_config.{key}")
    _require(int(config["max_steps"]) == spec.horizon_actions, "metadata max_steps differs from contract")
    _close(config["action_dt"], spec.action_interval_s, "metadata action_dt")
    _close(config["sub_dt"], spec.plant_interval_s, "metadata sub_dt")
    return config, {
        "file_name": metadata_path.name,
        "sha256": actual_hash,
        "environment_config_sha256": config_hash,
        "verification_mode": (
            "contract_exact_file"
            if exact_file
            else "approved_path_sanitized_public_projection"
        ),
        "hash_matches_contract": exact_file,
        "public_projection_preserves_frozen_environment_config": approved_public_projection,
    }


def _expected_factorial_contract(spec: CampaignSpec) -> Dict[str, Any]:
    return {
        "version": EXPERIMENT_VERSION,
        "design": "paired_2x2x2",
        "policies": list(POLICIES),
        "dynamics": list(DYNAMICS),
        "currents": list(CURRENTS),
        "arms": list(EXPECTED_ARMS),
        "plant_parameters": {
            key: dict(value) for key, value in PLANT_PARAMETER_LEVELS.items()
        },
        "current_parameters": {
            "steady_horizontal_speed_mps": 0.30,
            "direction": "uniform_per_episode_from_frozen_current_stream",
            "gauss_markov_component_std_mps": 0.05,
            "gauss_markov_time_constant_s": 120.0,
            "vertical_component_mps": 0.0,
            "measurement_convention": "bottom_track_velocity_over_ground",
        },
        "qualification_seeds": [49_900, 49_999],
        "runs": 100 * len(EXPECTED_ARMS),
        "retuning_allowed": False,
        "sealed_final_range": list(SEALED_FINAL_RANGE),
    }


def validate_contract(
    root: Path,
    *,
    metadata_override: Optional[Path] = None,
    spec: CampaignSpec = QUALIFICATION_SPEC,
) -> Tuple[Dict[str, Any], Mapping[str, Any], Dict[str, Any]]:
    contract_path = root / "control" / "campaign_contract.json"
    _require(contract_path.is_file(), "missing control/campaign_contract.json")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("cannot read campaign contract") from exc
    _require(isinstance(contract, dict), "campaign contract is not an object")
    exact_values = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "smoke": False,
        "episodes": len(spec.seeds),
        "episode_start": 0,
        "seeds": list(spec.seeds),
        "arms": list(EXPECTED_ARMS),
        "expected_runs": len(spec.seeds) * len(EXPECTED_ARMS),
        "publication_settings": True,
        "fixed_horizon_actions": spec.horizon_actions,
        "sealed_final_range": list(SEALED_FINAL_RANGE),
        "retuning_allowed": False,
    }
    for key, expected in exact_values.items():
        _require(contract.get(key) == expected, f"contract field {key} differs from the frozen qualification")
    _close(contract.get("action_interval_s"), spec.action_interval_s, "contract action_interval_s")
    _close(contract.get("plant_interval_s"), spec.plant_interval_s, "contract plant_interval_s")
    _require(
        contract.get("factorial_contract") == _expected_factorial_contract(spec),
        "factorial contract differs from frozen V40",
    )
    estimator = contract.get("estimator_config")
    _require(isinstance(estimator, dict), "contract estimator_config is absent")
    expected_estimator = {
        "coarse_candidates": 4096,
        "coarse_sweeps": 2,
        "local_starts": 48,
        "gate_mode": "raw",
        "candidate_radial_distribution": "uniform_radius",
    }
    for key, expected in expected_estimator.items():
        _require(estimator.get(key) == expected, f"estimator setting {key} differs from publication setting")
    freshness = contract.get("fresh_qualification_audit")
    _require(isinstance(freshness, dict), "fresh qualification audit is absent")
    _require(freshness.get("range") == [49_900, 49_999], "freshness range differs")
    _require(freshness.get("fresh") is True, "qualification range was not fresh when opened")
    _require(freshness.get("finding_count") == 0, "freshness audit recorded findings")
    source_status = _validate_source_snapshot(root, contract)
    environment_config, metadata_status = _load_environment_config(
        root, contract, metadata_override, spec
    )
    return contract, environment_config, {
        "contract_exact": True,
        **source_status,
        "environment_metadata": metadata_status,
    }


def _physical_scales(config: Mapping[str, Any]) -> np.ndarray:
    action_dt = _finite(config["action_dt"], "environment action_dt")
    return np.asarray(
        [
            _finite(config["rl_speed_delta_per_step"], "rl speed delta") / action_dt,
            min(
                _finite(config["max_yaw_rate_deg_s"], "max yaw rate"),
                _finite(config["rl_yaw_per_step_deg"], "rl yaw per step") / action_dt,
            ),
            min(
                _finite(config["max_pitch_rate_deg_s"], "max pitch rate"),
                _finite(config["rl_pitch_per_step_deg"], "rl pitch per step") / action_dt,
            ),
        ],
        dtype=np.float64,
    )


def _optional_finite(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    return _finite(value, label)


def _validate_summary_record(
    record: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_index: int,
    expected_arm: str,
    spec: CampaignSpec,
) -> Dict[str, Any]:
    prefix = f"seed {expected_seed}, arm {expected_arm}"
    arm = ARM_LOOKUP[expected_arm]
    _require(record.get("version") == EXPERIMENT_VERSION, f"{prefix}: wrong result version")
    _require(record.get("episode_seed") == expected_seed, f"{prefix}: record seed differs from filename")
    _require(record.get("episode_index") == expected_index, f"{prefix}: episode index differs")
    _require(record.get("arm") == expected_arm, f"{prefix}: arm differs from path")
    _require(record.get("policy_name") == arm["policy"], f"{prefix}: policy factor differs")
    _require(record.get("dynamics_name") == arm["dynamics"], f"{prefix}: dynamics factor differs")
    _require(record.get("current_name") == arm["current"], f"{prefix}: current factor differs")
    _require(record.get("source_name") == "both_leaders", f"{prefix}: not a two-link arm")
    _require(record.get("source_mask") == [True, True], f"{prefix}: source mask is not two-link")
    _require(record.get("active_source_count") == 2, f"{prefix}: active source count differs")
    _require(record.get("action_count") == spec.horizon_actions, f"{prefix}: incomplete fixed horizon")

    booleans = {
        key: _strict_bool(record.get(key), f"{prefix}.{key}")
        for key in (
            "terminal_joint_success",
            "tail80_joint_success",
            "dwell15_joint_success",
        )
    }
    numeric = {
        "tail50_joint_occupancy": _finite(record.get("tail50_joint_occupancy"), f"{prefix}.tail50_joint_occupancy"),
        "terminal_localization_error_m": _nonnegative(record.get("terminal_localization_error_m"), f"{prefix}.terminal_localization_error_m"),
        "terminal_formation_error_m": _nonnegative(record.get("terminal_formation_error_m"), f"{prefix}.terminal_formation_error_m"),
        "mean_squared_action": _nonnegative(record.get("mean_squared_action"), f"{prefix}.mean_squared_action"),
        "maximum_combined_decision_runtime_s": _nonnegative(record.get("maximum_combined_decision_runtime_s"), f"{prefix}.maximum_combined_decision_runtime_s"),
    }
    _require(0.0 <= numeric["tail50_joint_occupancy"] <= 1.0, f"{prefix}: tail occupancy outside [0,1]")

    gate = record.get("gate")
    _require(isinstance(gate, dict), f"{prefix}: gate record missing")
    exact = gate.get("exact")
    _require(isinstance(exact, dict), f"{prefix}: exact gate record missing")
    ever_locked = _strict_bool(gate.get("ever_locked"), f"{prefix}.gate.ever_locked")
    counts = {
        "unsafe_transition_count": _nonnegative_int(exact.get("false_transition_count"), f"{prefix}.false_transition_count"),
        "unsafe_track_start_count": _nonnegative_int(exact.get("false_locked_action_start_count"), f"{prefix}.false_locked_action_start_count"),
        "unsafe_track_end_count": _nonnegative_int(exact.get("false_locked_action_end_count"), f"{prefix}.false_locked_action_end_count"),
        "audit_release_violation_count": _nonnegative_int(gate.get("audit_release_violation_count"), f"{prefix}.audit_release_violation_count"),
    }
    transition_count = _nonnegative_int(exact.get("transition_count"), f"{prefix}.transition_count")
    locked_action_count = _nonnegative_int(exact.get("locked_action_count"), f"{prefix}.locked_action_count")
    _require(ever_locked == (transition_count > 0), f"{prefix}: ever_locked disagrees with transition_count")
    first_transition = _optional_finite(exact.get("first_transition_time_s"), f"{prefix}.first_transition_time_s")
    first_track = _optional_finite(exact.get("first_locked_action_time_s"), f"{prefix}.first_locked_action_time_s")
    gate_first_track = _optional_finite(gate.get("first_track_action_time_s"), f"{prefix}.gate.first_track_action_time_s")
    if ever_locked:
        _require(first_transition is not None, f"{prefix}: locked episode lacks first transition time")
        _require(first_track is not None, f"{prefix}: locked episode lacks first TRACK-action time")
        _require(gate_first_track is not None, f"{prefix}: locked episode lacks gate first-TRACK time")
        _close(first_track, gate_first_track, f"{prefix} first-TRACK fields")
        _require(locked_action_count > 0, f"{prefix}: locked episode has no TRACK action")
    else:
        _require(first_transition is None, f"{prefix}: no-lock episode has a transition time")
        _require(first_track is None and gate_first_track is None, f"{prefix}: no-lock episode has a first-TRACK time")
        _require(locked_action_count == 0, f"{prefix}: no-lock episode has TRACK actions")

    plant = record.get("plant")
    _require(isinstance(plant, dict), f"{prefix}: plant record missing")
    _require(plant.get("parameters") == PLANT_PARAMETER_LEVELS[arm["dynamics"]], f"{prefix}: plant parameters differ")
    _require(
        plant.get("bottom_track_current_visible") is (arm["current"] == CURRENT_VISIBLE),
        f"{prefix}: current visibility flag differs",
    )
    _require(plant.get("planner_uses_execution_model") is False, f"{prefix}: planner execution-model flag differs")
    _require(plant.get("planner_uses_current_model") is False, f"{prefix}: planner current-model flag differs")
    plant_numeric = {
        "requested_delivered_action_rms": _nonnegative(plant.get("requested_delivered_action_rms"), f"{prefix}.requested_delivered_action_rms"),
        "current_speed_rms_mps": _nonnegative(plant.get("current_speed_rms_mps"), f"{prefix}.current_speed_rms_mps"),
        "minimum_ground_speed_mps": _nonnegative(plant.get("minimum_ground_speed_mps"), f"{prefix}.minimum_ground_speed_mps"),
    }
    low_speed_count = _nonnegative_int(plant.get("ground_speed_below_0p05_action_count"), f"{prefix}.ground_speed_below_0p05_action_count")
    _require(low_speed_count <= spec.horizon_actions, f"{prefix}: low-speed count exceeds horizon")

    hash_pattern = re.compile(r"^[0-9a-f]{64}$")
    for key in ("noise_tape_sha256", "current_tape_sha256"):
        value = record.get(key)
        _require(isinstance(value, str) and bool(hash_pattern.fullmatch(value)), f"{prefix}: invalid {key}")
    _finite_vector(record.get("initial_truth_m"), 3, f"{prefix}.initial_truth_m")
    support = record.get("mission_support")
    _require(isinstance(support, dict), f"{prefix}: mission support missing")
    _finite_vector(support.get("center_m"), 3, f"{prefix}.mission_support.center_m")
    radius_min = _nonnegative(support.get("radius_min_m"), f"{prefix}.mission_support.radius_min_m")
    radius_max = _nonnegative(support.get("radius_max_m"), f"{prefix}.mission_support.radius_max_m")
    _require(radius_max > radius_min, f"{prefix}: mission support radii invalid")

    return {
        "episode_index": expected_index,
        "episode_seed": expected_seed,
        "arm": expected_arm,
        "policy": arm["policy"],
        "dynamics": arm["dynamics"],
        "current": arm["current"],
        **booleans,
        **numeric,
        "ever_locked": ever_locked,
        "first_transition_time_s": first_transition,
        "first_track_time_s": first_track,
        **counts,
        **plant_numeric,
        "ground_speed_below_0p05_action_count": low_speed_count,
        "action_count": spec.horizon_actions,
        "noise_tape_sha256": record["noise_tape_sha256"],
        "current_tape_sha256": record["current_tape_sha256"],
    }


def _exact_gate_from_trace(trace: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    gate_after = np.asarray(trace["gate_locked_after_update"], dtype=bool)
    phase = np.asarray(trace["phase_track"], dtype=bool)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    times = np.asarray(trace["time_s"], dtype=np.float64)
    action_times = np.asarray(trace["action_start_time_s"], dtype=np.float64)
    previous = np.concatenate([np.asarray([False]), gate_after[:-1]])
    transitions = np.flatnonzero(gate_after & ~previous)
    phase_indices = np.flatnonzero(phase)
    _require(not (phase_indices.size and np.any(phase_indices == 0)), "TRACK action occurs without a preceding state")
    transition_errors = localization[transitions]
    start_errors = localization[phase_indices - 1] if phase_indices.size else np.asarray([], dtype=np.float64)
    end_errors = localization[phase_indices]

    def optional_first(values: np.ndarray) -> Optional[float]:
        if values.size == 0 or not math.isfinite(float(values[0])):
            return None
        return float(values[0])

    transition_unsafe = (~np.isfinite(transition_errors)) | (transition_errors >= TERMINAL_LOCALIZATION_GATE_M)
    start_unsafe = (~np.isfinite(start_errors)) | (start_errors >= TERMINAL_LOCALIZATION_GATE_M)
    end_unsafe = (~np.isfinite(end_errors)) | (end_errors >= TERMINAL_LOCALIZATION_GATE_M)
    return {
        "transition_count": int(transitions.size),
        "first_transition_time_s": optional_first(times[transitions]),
        "false_transition_count": int(np.sum(transition_unsafe)),
        "locked_action_count": int(phase_indices.size),
        "first_locked_action_time_s": None if not phase_indices.size else float(action_times[phase_indices[0]]),
        "false_locked_action_start_count": int(np.sum(start_unsafe)),
        "false_locked_action_end_count": int(np.sum(end_unsafe)),
    }


def _expected_executed_state(
    delayed_physical: np.ndarray,
    *,
    action_interval_s: float,
) -> np.ndarray:
    taus = np.asarray([5.0, 2.0, 3.0], dtype=np.float64)
    gains = 1.0 - np.exp(-float(action_interval_s) / taus)
    state = np.zeros(3, dtype=np.float64)
    values = np.empty_like(delayed_physical, dtype=np.float64)
    for index, command in enumerate(delayed_physical):
        state = state + gains * (command - state)
        values[index] = state
    return values


def _validate_trace(
    path: Path,
    record: Mapping[str, Any],
    row: Dict[str, Any],
    *,
    config: Mapping[str, Any],
    spec: CampaignSpec,
) -> TraceDiagnostics:
    prefix = f"seed {row['episode_seed']}, arm {row['arm']}"
    _require(path.is_file(), f"{prefix}: missing NPZ trace")
    try:
        with np.load(path, allow_pickle=False) as archive:
            trace = {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ValidationError(f"{prefix}: cannot load NPZ trace") from exc
    required_scalar = (
        "time_s",
        "action_start_time_s",
        "phase_track",
        "gate_locked_after_update",
        "action_speed",
        "action_yaw",
        "action_pitch",
        "formation_error_truth_m",
        "localization_error_m",
        "audit_release_checks_pass",
    )
    for key in required_scalar:
        _require(key in trace, f"{prefix}: trace lacks {key}")
        _require(trace[key].shape == (spec.horizon_actions,), f"{prefix}: {key} has wrong shape")
    _require(np.all(np.isfinite(trace["time_s"])), f"{prefix}: time_s is non-finite")
    _require(np.all(np.isfinite(trace["action_start_time_s"])), f"{prefix}: action_start_time_s is non-finite")
    _require(np.all(np.isfinite(trace["formation_error_truth_m"])), f"{prefix}: formation trace is non-finite")
    for key in TRACE_VECTOR_FIELDS:
        _require(key in trace, f"{prefix}: trace lacks {key}")
        value = np.asarray(trace[key], dtype=np.float64)
        _require(value.shape == (spec.horizon_actions, 3), f"{prefix}: {key} has wrong shape")
        _require(np.all(np.isfinite(value)), f"{prefix}: {key} contains non-finite values")

    requested = np.asarray(trace["plant_requested_action"], dtype=np.float64)
    delayed = np.asarray(trace["plant_delivered_action"], dtype=np.float64)
    executed = np.asarray(trace["plant_executed_rate_state"], dtype=np.float64)
    actions = np.column_stack(
        [trace["action_speed"], trace["action_yaw"], trace["action_pitch"]]
    ).astype(np.float64)
    _require(np.array_equal(requested, actions), f"{prefix}: requested plant actions differ from decision trace")
    _require(np.all(np.abs(requested) <= 1.0 + 1e-12), f"{prefix}: normalized action outside [-1,1]")

    dynamics = str(row["dynamics"])
    if dynamics == DYNAMICS_KINEMATIC:
        _require(np.array_equal(delayed, requested), f"{prefix}: kinematic command path has an unexpected delay")
    else:
        expected_delayed = np.vstack([np.zeros((1, 3)), requested[:-1]])
        _require(np.array_equal(delayed, expected_delayed), f"{prefix}: delayed command is not the frozen one-action shift")

    scales = _physical_scales(config)
    requested_physical = requested * scales
    delayed_physical = delayed * scales
    response_applicable = dynamics == DYNAMICS_LOW_ORDER
    kinematic_semantics: Optional[str] = None
    if response_applicable:
        expected_state = _expected_executed_state(
            delayed_physical,
            action_interval_s=spec.action_interval_s,
        )
        _require(np.allclose(executed, expected_state, rtol=0.0, atol=2e-12), f"{prefix}: first-order executed state differs from frozen recurrence")
        executed_physical = executed
    else:
        executed_physical = delayed_physical.copy()
        if np.allclose(executed, delayed_physical, rtol=0.0, atol=2e-12):
            kinematic_semantics = "physical_rate_state_recorded"
        elif row["current"] == CURRENT_NONE and np.array_equal(executed, np.zeros_like(executed)):
            kinematic_semantics = "compatibility_placeholder_zero_reconstructed_from_action"
        else:
            _fail(f"{prefix}: uninterpretable kinematic executed-rate trace")

    water = np.asarray(trace["water_current_mps"], dtype=np.float64)
    body = np.asarray(trace["body_velocity_through_water_mps"], dtype=np.float64)
    ground = np.asarray(trace["ground_velocity_mps"], dtype=np.float64)
    if row["current"] == CURRENT_NONE:
        _require(np.array_equal(water, np.zeros_like(water)), f"{prefix}: no-current arm has nonzero current")
    _require(np.allclose(ground, body + water, rtol=0.0, atol=2e-12), f"{prefix}: ground velocity is not body velocity plus current")

    recomputed_delay_rms = float(np.sqrt(np.mean(np.square(requested - delayed))))
    _close(
        recomputed_delay_rms,
        row["requested_delivered_action_rms"],
        f"{prefix} requested-to-delayed RMS",
        tolerance=2e-12,
    )
    recomputed_current_rms = float(np.sqrt(np.mean(np.sum(np.square(water), axis=1))))
    _close(recomputed_current_rms, row["current_speed_rms_mps"], f"{prefix} current RMS", tolerance=2e-12)
    speeds = np.linalg.norm(ground, axis=1)
    _close(float(np.min(speeds)), row["minimum_ground_speed_mps"], f"{prefix} minimum ground speed", tolerance=2e-12)
    _require(int(np.sum(speeds < 0.05)) == row["ground_speed_below_0p05_action_count"], f"{prefix}: low-ground-speed count differs")

    formation = np.asarray(trace["formation_error_truth_m"], dtype=np.float64)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    _require(math.isfinite(float(localization[-1])), f"{prefix}: terminal localization error is not finite")
    _close(float(formation[-1]), row["terminal_formation_error_m"], f"{prefix} terminal formation error", tolerance=2e-10)
    _close(float(localization[-1]), row["terminal_localization_error_m"], f"{prefix} terminal localization error", tolerance=2e-10)
    joint = (formation < TERMINAL_FORMATION_GATE_M) & (
        np.isfinite(localization) & (localization < TERMINAL_LOCALIZATION_GATE_M)
    )
    tail = joint[-min(TAIL_WINDOW_ACTIONS, spec.horizon_actions) :]
    dwell = joint[-min(DWELL_ACTIONS, spec.horizon_actions) :]
    tail_localization = localization[-min(TAIL_WINDOW_ACTIONS, spec.horizon_actions) :]
    tail_formation = formation[-min(TAIL_WINDOW_ACTIONS, spec.horizon_actions) :]
    _require(np.all(np.isfinite(tail_localization)), f"{prefix}: tail localization errors are not finite")
    _require(np.all(np.isfinite(tail_formation)), f"{prefix}: tail formation errors are not finite")
    row["tail50_mean_localization_error_m"] = float(np.mean(tail_localization))
    row["tail50_mean_formation_error_m"] = float(np.mean(tail_formation))
    terminal = bool(joint[-1])
    tail_occupancy = float(np.mean(tail))
    _require(terminal == row["terminal_joint_success"], f"{prefix}: terminal success differs from trace")
    _close(tail_occupancy, row["tail50_joint_occupancy"], f"{prefix} tail occupancy", tolerance=2e-12)
    _require(bool(tail_occupancy >= 0.8) == row["tail80_joint_success"], f"{prefix}: Tail80 differs from trace")
    _require(bool(np.all(dwell)) == row["dwell15_joint_success"], f"{prefix}: Dwell15 differs from trace")

    exact_trace = _exact_gate_from_trace(trace)
    exact_saved = record["gate"]["exact"]
    for key, recomputed in exact_trace.items():
        saved = exact_saved.get(key)
        if isinstance(recomputed, float):
            _close(recomputed, saved, f"{prefix}.{key}", tolerance=2e-12)
        else:
            _require(saved == recomputed, f"{prefix}: exact gate field {key} differs from trace")
    gate_after = np.asarray(trace["gate_locked_after_update"], dtype=bool)
    previous = np.concatenate([np.asarray([False]), gate_after[:-1]])
    audit_violations = int(
        np.sum(
            (gate_after & ~previous)
            & ~np.asarray(trace["audit_release_checks_pass"], dtype=bool)
        )
    )
    _require(audit_violations == row["audit_release_violation_count"], f"{prefix}: audit-release violation count differs")

    return TraceDiagnostics(
        seed=int(row["episode_seed"]),
        arm=str(row["arm"]),
        requested_normalized=requested,
        delayed_normalized=delayed,
        requested_physical=requested_physical,
        delayed_physical=delayed_physical,
        executed_physical=executed_physical,
        response_applicable=response_applicable,
        kinematic_trace_semantics=kinematic_semantics,
    )


def load_and_validate_records(
    root: Path,
    *,
    config: Mapping[str, Any],
    spec: CampaignSpec = QUALIFICATION_SPEC,
) -> Tuple[List[Dict[str, Any]], List[TraceDiagnostics], Dict[str, Any]]:
    result_root = root / "episode_results"
    trace_root = root / "traces_npz"
    _require(result_root.is_dir(), "missing episode_results directory")
    _require(trace_root.is_dir(), "missing traces_npz directory")
    result_files = sorted(result_root.rglob("*.json"))
    trace_files = sorted(trace_root.rglob("*.npz"))
    expected_runs = len(spec.seeds) * len(EXPECTED_ARMS)
    _require(len(result_files) == expected_runs, f"expected {expected_runs} JSON records, found {len(result_files)}")
    _require(len(trace_files) == expected_runs, f"expected {expected_runs} NPZ traces, found {len(trace_files)}")

    file_pattern = re.compile(r"^episode_(\d{4})_seed_(\d+)\.json$")
    seen: set[Tuple[int, str]] = set()
    rows: List[Dict[str, Any]] = []
    diagnostics: List[TraceDiagnostics] = []
    raw_records: Dict[Tuple[int, str], Mapping[str, Any]] = {}
    for path in result_files:
        match = file_pattern.fullmatch(path.name)
        _require(match is not None, f"unexpected result filename: {path.name}")
        episode_index = int(match.group(1))
        seed = int(match.group(2))
        _require(seed in spec.seeds, f"unexpected seed in result: {seed}")
        expected_index = spec.seeds.index(seed)
        _require(episode_index == expected_index, f"seed {seed}: filename episode index differs")
        arm = path.parent.name
        _require(arm in ARM_LOOKUP, f"unknown arm directory: {arm}")
        _require(path.parent.parent == result_root, f"unexpected nested result path: {path}")
        key = (seed, arm)
        _require(key not in seen, f"duplicate result record: {key}")
        seen.add(key)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(f"cannot read result record: {path}") from exc
        _require(isinstance(record, dict), f"result record is not an object: {path}")
        row = _validate_summary_record(
            record,
            expected_seed=seed,
            expected_index=expected_index,
            expected_arm=arm,
            spec=spec,
        )
        trace_path = trace_root / arm / path.name.replace(".json", ".npz")
        trace_diagnostic = _validate_trace(
            trace_path,
            record,
            row,
            config=config,
            spec=spec,
        )
        rows.append(row)
        diagnostics.append(trace_diagnostic)
        raw_records[key] = record

    expected_keys = {(seed, arm) for seed in spec.seeds for arm in EXPECTED_ARM_NAMES}
    _require(seen == expected_keys, "campaign lacks the exact seed-by-arm Cartesian product")
    expected_trace_paths = {
        trace_root / arm / f"episode_{spec.seeds.index(seed):04d}_seed_{seed}.npz"
        for seed, arm in expected_keys
    }
    _require(set(trace_files) == expected_trace_paths, "trace files do not match the exact seed-by-arm Cartesian product")

    pairing_fields = (
        "episode_index",
        "noise_tape_sha256",
        "current_tape_sha256",
        "initial_truth_m",
        "mission_support",
    )
    for seed in spec.seeds:
        records = [raw_records[(seed, arm)] for arm in EXPECTED_ARM_NAMES]
        for field in pairing_fields:
            reference = _canonical(records[0].get(field))
            _require(
                all(_canonical(record.get(field)) == reference for record in records[1:]),
                f"seed {seed}: paired arms differ in {field}",
            )

    return rows, diagnostics, {
        "json_record_count": len(rows),
        "npz_trace_count": len(diagnostics),
        "seed_count": len(spec.seeds),
        "exact_seed_support": True,
        "all_eight_arms_per_seed": True,
        "paired_initial_state_and_tapes": True,
        "finite_required_outcomes": True,
        "first_track_missing_only_for_no_lock": True,
        "trace_outcomes_recomputed": True,
    }


def _distribution(values: Iterable[Optional[float]]) -> Optional[Dict[str, float]]:
    array = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if not array.size:
        return None
    _require(np.all(np.isfinite(array)), "distribution received non-finite values")
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "episodes": len(rows),
        "terminal_success_count": sum(bool(row["terminal_joint_success"]) for row in rows),
        "tail80_success_count": sum(bool(row["tail80_joint_success"]) for row in rows),
        "dwell15_success_count": sum(bool(row["dwell15_joint_success"]) for row in rows),
        "ever_lock_count": sum(bool(row["ever_locked"]) for row in rows),
        "unsafe_transition_count": sum(int(row["unsafe_transition_count"]) for row in rows),
        "unsafe_track_start_count": sum(int(row["unsafe_track_start_count"]) for row in rows),
        "unsafe_track_end_count": sum(int(row["unsafe_track_end_count"]) for row in rows),
        "audit_release_violation_count": sum(int(row["audit_release_violation_count"]) for row in rows),
        "first_evidence_transition_time_s": _distribution(row["first_transition_time_s"] for row in rows),
        "first_TRACK_action_time_s": _distribution(row["first_track_time_s"] for row in rows),
        "terminal_localization_error_m": _distribution(row["terminal_localization_error_m"] for row in rows),
        "terminal_formation_error_m": _distribution(row["terminal_formation_error_m"] for row in rows),
        "tail50_mean_localization_error_m": _distribution(row["tail50_mean_localization_error_m"] for row in rows),
        "tail50_mean_formation_error_m": _distribution(row["tail50_mean_formation_error_m"] for row in rows),
        "tail50_joint_occupancy": _distribution(row["tail50_joint_occupancy"] for row in rows),
        "mean_squared_action": _distribution(row["mean_squared_action"] for row in rows),
        "maximum_combined_decision_runtime_s": _distribution(row["maximum_combined_decision_runtime_s"] for row in rows),
        "maximum_combined_decision_runtime_s_across_episodes": max(float(row["maximum_combined_decision_runtime_s"]) for row in rows),
        "decision_deadline_miss_episode_count": sum(float(row["maximum_combined_decision_runtime_s"]) >= 2.0 for row in rows),
    }


def _bootstrap(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int,
) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, "empty bootstrap input")
    _require(np.all(np.isfinite(array)), "non-finite bootstrap input")
    _require(replicates > 0, "bootstrap replicate count must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        end = min(start + 1000, replicates)
        indices = rng.integers(0, array.size, size=(end - start, array.size))
        draws[start:end] = np.mean(array[indices], axis=1)
    return {
        "pairs": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "bootstrap_mean_95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "replicates": int(replicates),
        "seed": int(seed),
    }


def _mcnemar_exact(reference_only: int, treatment_only: int) -> Optional[float]:
    total = int(reference_only) + int(treatment_only)
    if total == 0:
        return None
    lower = min(int(reference_only), int(treatment_only))
    probability = sum(math.comb(total, k) for k in range(lower + 1)) / (2.0 ** total)
    return float(min(1.0, 2.0 * probability))


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    _require(total > 0, "Wilson interval requires a positive sample size")
    _require(0 <= successes <= total, "Wilson successes are outside support")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_newcombe_method10_interval(
    both_success: int,
    treatment_only: int,
    reference_only: int,
    neither_success: int,
    *,
    z: float = 1.959963984540054,
) -> Tuple[float, float, float]:
    """Newcombe method-10 interval for treatment minus reference."""

    cells = (both_success, treatment_only, reference_only, neither_success)
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in cells),
        "paired cells must be integer counts",
    )
    _require(all(value >= 0 for value in cells), "paired cells must be nonnegative")
    total = sum(cells)
    _require(total > 0, "paired interval requires at least one pair")
    treatment_success = both_success + treatment_only
    reference_success = both_success + reference_only
    treatment_rate = treatment_success / total
    reference_rate = reference_success / total
    difference = treatment_rate - reference_rate
    treatment_lower, treatment_upper = _wilson_interval(treatment_success, total, z)
    reference_lower, reference_upper = _wilson_interval(reference_success, total, z)

    phi_denominator = math.sqrt(
        (both_success + treatment_only)
        * (reference_only + neither_success)
        * (both_success + reference_only)
        * (treatment_only + neither_success)
    )
    phi_numerator = both_success * neither_success - treatment_only * reference_only
    if phi_numerator > 0:
        phi_numerator = max(phi_numerator - total / 2.0, 0.0)
    phi = phi_numerator / phi_denominator if phi_denominator > 0.0 else 0.0
    phi = min(1.0, max(-1.0, phi))

    treatment_lower_half = treatment_rate - treatment_lower
    treatment_upper_half = treatment_upper - treatment_rate
    reference_lower_half = reference_rate - reference_lower
    reference_upper_half = reference_upper - reference_rate
    lower = difference - math.sqrt(
        max(
            0.0,
            treatment_lower_half**2
            - 2.0 * phi * treatment_lower_half * reference_upper_half
            + reference_upper_half**2,
        )
    )
    upper = difference + math.sqrt(
        max(
            0.0,
            treatment_upper_half**2
            - 2.0 * phi * treatment_upper_half * reference_lower_half
            + reference_lower_half**2,
        )
    )
    return difference, max(-1.0, lower), min(1.0, upper)


def _rows_by_seed(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    result = {int(row["episode_seed"]): row for row in rows}
    _require(len(result) == len(rows), "paired contrast contains duplicate seeds")
    return result


def _paired_contrast(
    reference_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    *,
    seed_offset: int,
    bootstrap_replicates: int,
) -> Dict[str, Any]:
    reference = _rows_by_seed(reference_rows)
    treatment = _rows_by_seed(treatment_rows)
    _require(set(reference) == set(treatment), "paired contrast has unequal seed support")
    seeds = sorted(reference)
    result: Dict[str, Any] = {
        "pairs": len(seeds),
        "direction": "treatment_minus_reference",
    }
    for endpoint in BINARY_ENDPOINTS:
        reference_only = 0
        treatment_only = 0
        concordant_success = 0
        concordant_failure = 0
        differences: List[float] = []
        for seed in seeds:
            left = bool(reference[seed][endpoint])
            right = bool(treatment[seed][endpoint])
            reference_only += int(left and not right)
            treatment_only += int(right and not left)
            concordant_success += int(left and right)
            concordant_failure += int(not left and not right)
            differences.append(float(right) - float(left))
        _, interval_lower, interval_upper = paired_newcombe_method10_interval(
            concordant_success,
            treatment_only,
            reference_only,
            concordant_failure,
        )
        result[endpoint] = {
            "reference_only": reference_only,
            "treatment_only": treatment_only,
            "concordant_success": concordant_success,
            "concordant_failure": concordant_failure,
            "risk_difference": float(np.mean(differences)),
            "paired_newcombe_method10_95": [interval_lower, interval_upper],
            "mcnemar_exact_two_sided_p": _mcnemar_exact(reference_only, treatment_only),
        }
    for endpoint_index, endpoint in enumerate(CONTINUOUS_ENDPOINTS):
        differences = [
            float(treatment[seed][endpoint]) - float(reference[seed][endpoint])
            for seed in seeds
        ]
        result[f"{endpoint}_treatment_minus_reference"] = _bootstrap(
            differences,
            seed=BOOTSTRAP_SEED + seed_offset * 10 + endpoint_index,
            replicates=bootstrap_replicates,
        )
    both_track = [
        seed
        for seed in seeds
        if reference[seed]["first_track_time_s"] is not None
        and treatment[seed]["first_track_time_s"] is not None
    ]
    result["first_TRACK_action_time_s_conditional_both_locked"] = {
        "pairs": len(both_track),
        "reference_missing": sum(reference[seed]["first_track_time_s"] is None for seed in seeds),
        "treatment_missing": sum(treatment[seed]["first_track_time_s"] is None for seed in seeds),
        "selection_warning": "time contrast is conditional on both arms reaching TRACK",
        "treatment_minus_reference": (
            None
            if not both_track
            else _bootstrap(
                [
                    float(treatment[seed]["first_track_time_s"])
                    - float(reference[seed]["first_track_time_s"])
                    for seed in both_track
                ],
                seed=BOOTSTRAP_SEED + seed_offset * 10 + 9,
                replicates=bootstrap_replicates,
            )
        ),
    }
    return result


def _cell_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str], List[Mapping[str, Any]]]:
    cells: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for policy in POLICIES:
        for dynamics in DYNAMICS:
            for current in CURRENTS:
                cells[(policy, dynamics, current)] = [
                    row
                    for row in rows
                    if row["policy"] == policy
                    and row["dynamics"] == dynamics
                    and row["current"] == current
                ]
    return cells


def factorial_main_effect_contrasts(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
) -> Dict[str, Any]:
    cells = _cell_rows(rows)
    active_vs_fixed: Dict[str, Any] = {}
    dynamic_vs_kinematic: Dict[str, Any] = {}
    current_vs_none: Dict[str, Any] = {}
    offset = 0
    for dynamics in DYNAMICS:
        for current in CURRENTS:
            name = f"{dynamics}__{current}"
            active_vs_fixed[name] = _paired_contrast(
                cells[(POLICY_FIXED, dynamics, current)],
                cells[(POLICY_ACTIVE, dynamics, current)],
                seed_offset=offset,
                bootstrap_replicates=bootstrap_replicates,
            )
            offset += 1
    for policy in POLICIES:
        for current in CURRENTS:
            name = f"{policy}__{current}"
            dynamic_vs_kinematic[name] = _paired_contrast(
                cells[(policy, DYNAMICS_KINEMATIC, current)],
                cells[(policy, DYNAMICS_LOW_ORDER, current)],
                seed_offset=offset,
                bootstrap_replicates=bootstrap_replicates,
            )
            offset += 1
    for policy in POLICIES:
        for dynamics in DYNAMICS:
            name = f"{policy}__{dynamics}"
            current_vs_none[name] = _paired_contrast(
                cells[(policy, dynamics, CURRENT_NONE)],
                cells[(policy, dynamics, CURRENT_VISIBLE)],
                seed_offset=offset,
                bootstrap_replicates=bootstrap_replicates,
            )
            offset += 1
    return {
        "treatment_minus_reference_convention": True,
        "active_minus_fixed_within_execution_current": active_vs_fixed,
        "dynamic_minus_kinematic_within_policy_current": dynamic_vs_kinematic,
        "current_minus_none_within_policy_execution": current_vs_none,
    }


def _interaction(
    cells: Mapping[Tuple[str, str, str], Sequence[Mapping[str, Any]]],
    terms: Sequence[Tuple[float, Tuple[str, str, str]]],
    *,
    formula: str,
    seed_offset: int,
    bootstrap_replicates: int,
) -> Dict[str, Any]:
    maps = [(coefficient, _rows_by_seed(cells[cell])) for coefficient, cell in terms]
    support = set(maps[0][1])
    _require(all(set(values) == support for _, values in maps[1:]), "interaction has unequal seed support")
    seeds = sorted(support)
    endpoints: Dict[str, Any] = {}
    for endpoint_index, endpoint in enumerate(BINARY_ENDPOINTS + CONTINUOUS_ENDPOINTS):
        differences = [
            sum(
                coefficient * float(values[seed][endpoint])
                for coefficient, values in maps
            )
            for seed in seeds
        ]
        endpoints[endpoint] = _bootstrap(
            differences,
            seed=BOOTSTRAP_SEED + seed_offset * 10 + endpoint_index,
            replicates=bootstrap_replicates,
        )
    eligible_track_seeds = [
        seed
        for seed in seeds
        if all(values[seed]["first_track_time_s"] is not None for _, values in maps)
    ]
    first_track: Dict[str, Any] = {
        "pairs": len(eligible_track_seeds),
        "total_seed_support": len(seeds),
        "selection_warning": "first-TRACK interaction is conditional on every contributing cell reaching TRACK",
        "difference_in_differences": None,
    }
    if eligible_track_seeds:
        first_track["difference_in_differences"] = _bootstrap(
            [
                sum(
                    coefficient * float(values[seed]["first_track_time_s"])
                    for coefficient, values in maps
                )
                for seed in eligible_track_seeds
            ],
            seed=BOOTSTRAP_SEED + 1_000_000 + seed_offset * 100 + 99,
            replicates=bootstrap_replicates,
        )
    return {
        "pairs": len(seeds),
        "formula": formula,
        "interpretation": "difference_in_differences; zero means no interaction on the endpoint scale",
        "endpoints": endpoints,
        "first_TRACK_action_time_s": first_track,
    }


def factorial_interactions(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
) -> Dict[str, Any]:
    cells = _cell_rows(rows)
    result: Dict[str, Any] = {
        "dynamics_x_current_within_policy": {},
        "policy_x_dynamics_within_current": {},
        "policy_x_current_within_dynamics": {},
    }
    offset = 100
    for policy in POLICIES:
        result["dynamics_x_current_within_policy"][policy] = _interaction(
            cells,
            [
                (+1.0, (policy, DYNAMICS_LOW_ORDER, CURRENT_VISIBLE)),
                (-1.0, (policy, DYNAMICS_KINEMATIC, CURRENT_VISIBLE)),
                (-1.0, (policy, DYNAMICS_LOW_ORDER, CURRENT_NONE)),
                (+1.0, (policy, DYNAMICS_KINEMATIC, CURRENT_NONE)),
            ],
            formula="(dynamic-current - kinematic-current) - (dynamic-none - kinematic-none)",
            seed_offset=offset,
            bootstrap_replicates=bootstrap_replicates,
        )
        offset += 1
    for current in CURRENTS:
        result["policy_x_dynamics_within_current"][current] = _interaction(
            cells,
            [
                (+1.0, (POLICY_ACTIVE, DYNAMICS_LOW_ORDER, current)),
                (-1.0, (POLICY_FIXED, DYNAMICS_LOW_ORDER, current)),
                (-1.0, (POLICY_ACTIVE, DYNAMICS_KINEMATIC, current)),
                (+1.0, (POLICY_FIXED, DYNAMICS_KINEMATIC, current)),
            ],
            formula="active-minus-fixed under dynamic execution minus active-minus-fixed under kinematics",
            seed_offset=offset,
            bootstrap_replicates=bootstrap_replicates,
        )
        offset += 1
    for dynamics in DYNAMICS:
        result["policy_x_current_within_dynamics"][dynamics] = _interaction(
            cells,
            [
                (+1.0, (POLICY_ACTIVE, dynamics, CURRENT_VISIBLE)),
                (-1.0, (POLICY_FIXED, dynamics, CURRENT_VISIBLE)),
                (-1.0, (POLICY_ACTIVE, dynamics, CURRENT_NONE)),
                (+1.0, (POLICY_FIXED, dynamics, CURRENT_NONE)),
            ],
            formula="active-minus-fixed with current minus active-minus-fixed without current",
            seed_offset=offset,
            bootstrap_replicates=bootstrap_replicates,
        )
        offset += 1
    result["three_way_policy_x_dynamics_x_current"] = _interaction(
        cells,
        [
            (+1.0, (POLICY_ACTIVE, DYNAMICS_LOW_ORDER, CURRENT_VISIBLE)),
            (-1.0, (POLICY_FIXED, DYNAMICS_LOW_ORDER, CURRENT_VISIBLE)),
            (-1.0, (POLICY_ACTIVE, DYNAMICS_KINEMATIC, CURRENT_VISIBLE)),
            (+1.0, (POLICY_FIXED, DYNAMICS_KINEMATIC, CURRENT_VISIBLE)),
            (-1.0, (POLICY_ACTIVE, DYNAMICS_LOW_ORDER, CURRENT_NONE)),
            (+1.0, (POLICY_FIXED, DYNAMICS_LOW_ORDER, CURRENT_NONE)),
            (+1.0, (POLICY_ACTIVE, DYNAMICS_KINEMATIC, CURRENT_NONE)),
            (-1.0, (POLICY_FIXED, DYNAMICS_KINEMATIC, CURRENT_NONE)),
        ],
        formula="policy-by-dynamics interaction with current minus the same interaction without current",
        seed_offset=offset,
        bootstrap_replicates=bootstrap_replicates,
    )
    offset += 1
    result["active_advantage_joint_nonideality_minus_nominal"] = _interaction(
        cells,
        [
            (+1.0, (POLICY_ACTIVE, DYNAMICS_LOW_ORDER, CURRENT_VISIBLE)),
            (-1.0, (POLICY_FIXED, DYNAMICS_LOW_ORDER, CURRENT_VISIBLE)),
            (-1.0, (POLICY_ACTIVE, DYNAMICS_KINEMATIC, CURRENT_NONE)),
            (+1.0, (POLICY_FIXED, DYNAMICS_KINEMATIC, CURRENT_NONE)),
        ],
        formula="active-minus-fixed under joint nonideality minus active-minus-fixed nominally",
        seed_offset=offset,
        bootstrap_replicates=bootstrap_replicates,
    )
    return result


def _rms_by_channel(values: np.ndarray) -> List[float]:
    return np.sqrt(np.mean(np.square(values), axis=0)).astype(float).tolist()


def _best_lag_by_channel(
    diagnostics: Sequence[TraceDiagnostics],
    *,
    input_field: str,
    output_field: str,
    maximum_lag_actions: int = 8,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for channel in range(3):
        candidates: List[Tuple[float, int, int]] = []
        for lag in range(maximum_lag_actions + 1):
            squared_error = 0.0
            sample_count = 0
            for diagnostic in diagnostics:
                input_values = np.asarray(getattr(diagnostic, input_field))[:, channel]
                output_values = np.asarray(getattr(diagnostic, output_field))[:, channel]
                if lag == 0:
                    left = output_values
                    right = input_values
                else:
                    left = output_values[lag:]
                    right = input_values[:-lag]
                squared_error += float(np.sum(np.square(left - right)))
                sample_count += int(left.size)
            rms = math.sqrt(squared_error / max(sample_count, 1))
            candidates.append((rms, lag, sample_count))
        best = min(candidates, key=lambda item: (item[0], item[1]))
        result.append(
            {
                "channel": ("surge_acceleration", "yaw_rate", "pitch_rate")[channel],
                "best_lag_actions": best[1],
                "best_lag_s": float(best[1] * 2.0),
                "rms_at_best_lag": best[0],
                "samples": best[2],
            }
        )
    return result


def aggregate_command_execution(
    diagnostics: Sequence[TraceDiagnostics],
) -> Dict[str, Any]:
    by_arm: Dict[str, Any] = {}
    units = ["m s^-2", "deg s^-1", "deg s^-1"]
    for arm in EXPECTED_ARM_NAMES:
        selected = [diagnostic for diagnostic in diagnostics if diagnostic.arm == arm]
        _require(bool(selected), f"no trace diagnostics for {arm}")
        requested_normalized = np.concatenate([value.requested_normalized for value in selected], axis=0)
        delayed_normalized = np.concatenate([value.delayed_normalized for value in selected], axis=0)
        requested_physical = np.concatenate([value.requested_physical for value in selected], axis=0)
        delayed_physical = np.concatenate([value.delayed_physical for value in selected], axis=0)
        executed_physical = np.concatenate([value.executed_physical for value in selected], axis=0)
        response_applicable = all(value.response_applicable for value in selected)
        _require(response_applicable or not any(value.response_applicable for value in selected), f"mixed execution semantics in {arm}")
        requested_to_delayed = {
            "meaning": "transport/command-delay mismatch only; it excludes first-order response",
            "normalized_RMS_all_channels": float(np.sqrt(np.mean(np.square(requested_normalized - delayed_normalized)))),
            "normalized_RMS_by_channel": _rms_by_channel(requested_normalized - delayed_normalized),
            "physical_RMS_by_channel": _rms_by_channel(requested_physical - delayed_physical),
            "physical_channel_units": units,
            "estimated_delay": _best_lag_by_channel(
                selected,
                input_field="requested_physical",
                output_field="delayed_physical",
            ),
        }
        if response_applicable:
            response = {
                "applicable": True,
                "meaning": "delayed physical rate command minus end-of-action first-order executed state",
                "delayed_command_to_executed_state_RMS_by_channel": _rms_by_channel(delayed_physical - executed_physical),
                "requested_command_to_executed_state_RMS_by_channel_including_delay": _rms_by_channel(requested_physical - executed_physical),
                "physical_channel_units": units,
                "estimated_additional_response_lag": _best_lag_by_channel(
                    selected,
                    input_field="delayed_physical",
                    output_field="executed_physical",
                ),
            }
            kinematic = None
        else:
            response = {
                "applicable": False,
                "reason": "kinematic path has no first-order response state",
            }
            semantics = sorted({str(value.kinematic_trace_semantics) for value in selected})
            kinematic = {
                "physical_rate_commands_reconstructed": True,
                "source": "plant_delivered_action multiplied by frozen environment action-to-rate scales",
                "RMS_by_channel": _rms_by_channel(executed_physical),
                "physical_channel_units": units,
                "raw_trace_semantics": semantics,
            }
        by_arm[arm] = {
            "episodes": len(selected),
            "requested_to_delayed_command": requested_to_delayed,
            "first_order_response": response,
            "kinematic_physical_rate_reconstruction": kinematic,
        }
    return {
        "semantic_warning": "requested-to-delayed-command mismatch and first-order response mismatch are distinct and are never pooled",
        "by_arm": by_arm,
    }


def qualification_screens(
    rows: Sequence[Mapping[str, Any]],
    by_arm: Mapping[str, Mapping[str, Any]],
    main_contrasts: Mapping[str, Any],
    *,
    spec: CampaignSpec,
) -> Dict[str, Any]:
    del rows
    nominal_name = arm_name(POLICY_ACTIVE, DYNAMICS_KINEMATIC, CURRENT_NONE)
    nominal = by_arm[nominal_name]
    nominal_pass = bool(
        nominal["terminal_success_count"] >= spec.nominal_terminal_min
        and nominal["tail80_success_count"] >= spec.nominal_tail80_min
        and nominal["ever_lock_count"] >= spec.nominal_lock_min
        and nominal["unsafe_track_start_count"] == 0
        and nominal["unsafe_track_end_count"] == 0
    )
    active_screens: Dict[str, str] = {}
    for dynamics in DYNAMICS:
        for current in CURRENTS:
            name = arm_name(POLICY_ACTIVE, dynamics, current)
            metrics = by_arm[name]
            if name == nominal_name:
                label = "NOMINAL_REPLICATION_PASS" if nominal_pass else "NOMINAL_REPLICATION_FAIL"
            else:
                label = (
                    "ROBUST_WITHIN_SCREEN"
                    if metrics["terminal_success_count"] >= spec.robust_terminal_min
                    and metrics["tail80_success_count"] >= spec.robust_tail80_min
                    and metrics["unsafe_track_start_count"] == 0
                    and metrics["unsafe_track_end_count"] == 0
                    else "OUTSIDE_ROBUSTNESS_SCREEN"
                )
            active_screens[name] = label

    policy_claims: Dict[str, str] = {}
    contrasts = main_contrasts["active_minus_fixed_within_execution_current"]
    for cell, contrast in contrasts.items():
        terminal = float(contrast["terminal_joint_success"]["risk_difference"])
        tail80 = float(contrast["tail80_joint_success"]["risk_difference"])
        policy_claims[cell] = (
            "ACTIVE_OUTPERFORMS_FIXED_ON_BOTH_BINARY_ENDPOINTS"
            if terminal > 0.0 and tail80 > 0.0
            else "NO_PRE_SPECIFIED_OUTPERFORMANCE_CLAIM"
        )
    return {
        "nominal_replication_pass": nominal_pass,
        "active_policy_absolute_screens": active_screens,
        "active_vs_fixed_claim_screen": policy_claims,
    }


def publication_arm_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the eight factorial cells into a compact publication table."""

    by_arm = result["by_arm"]
    execution = result["command_execution_diagnostics"]["by_arm"]
    rows: List[Dict[str, Any]] = []
    for specification in EXPECTED_ARMS:
        name = specification["name"]
        metrics = by_arm[name]
        first_track = metrics["first_TRACK_action_time_s"]
        e_p = metrics["terminal_localization_error_m"]
        e_f = metrics["terminal_formation_error_m"]
        tail_e_p = metrics["tail50_mean_localization_error_m"]
        tail_e_f = metrics["tail50_mean_formation_error_m"]
        runtime = metrics["maximum_combined_decision_runtime_s"]
        delay = execution[name]["requested_to_delayed_command"]
        response = execution[name]["first_order_response"]
        rows.append(
            {
                "arm": name,
                "policy": specification["policy"],
                "execution": specification["dynamics"],
                "current": specification["current"],
                "N": metrics["episodes"],
                "terminal_success_n": metrics["terminal_success_count"],
                "terminal_success_rate": metrics["terminal_success_count"] / metrics["episodes"],
                "Tail80_n": metrics["tail80_success_count"],
                "Tail80_rate": metrics["tail80_success_count"] / metrics["episodes"],
                "ever_lock_n": metrics["ever_lock_count"],
                "ever_lock_rate": metrics["ever_lock_count"] / metrics["episodes"],
                "first_TRACK_observed_n": 0 if first_track is None else first_track["count"],
                "first_TRACK_median_s": None if first_track is None else first_track["median"],
                "first_TRACK_p95_s": None if first_track is None else first_track["p95"],
                "terminal_e_p_median_m": e_p["median"],
                "terminal_e_p_p95_m": e_p["p95"],
                "terminal_e_p_max_m": e_p["max"],
                "terminal_e_f_median_m": e_f["median"],
                "terminal_e_f_p95_m": e_f["p95"],
                "terminal_e_f_max_m": e_f["max"],
                "tail50_mean_e_p_median_m": tail_e_p["median"],
                "tail50_mean_e_p_p95_m": tail_e_p["p95"],
                "tail50_mean_e_f_median_m": tail_e_f["median"],
                "tail50_mean_e_f_p95_m": tail_e_f["p95"],
                "truth_invalid_TRACK_starts_n": metrics["unsafe_track_start_count"],
                "truth_invalid_TRACK_ends_n": metrics["unsafe_track_end_count"],
                "decision_runtime_median_s": runtime["median"],
                "decision_runtime_p95_s": runtime["p95"],
                "decision_runtime_max_s": runtime["max"],
                "deadline_miss_episodes_n": metrics["decision_deadline_miss_episode_count"],
                "requested_to_delayed_normalized_RMS": delay["normalized_RMS_all_channels"],
                "first_order_response_applicable": response["applicable"],
            }
        )
    _require(len(rows) == 8, "publication arm table does not contain eight rows")
    return rows


def interaction_plot_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return long-form marginal cell data suitable for interaction plots."""

    rows: List[Dict[str, Any]] = []
    for specification in EXPECTED_ARMS:
        name = specification["name"]
        metrics = result["by_arm"][name]
        episodes = int(metrics["episodes"])

        def append(
            endpoint: str,
            estimate: Optional[float],
            *,
            observed: int,
            unit: str,
            median: Optional[float] = None,
            p95: Optional[float] = None,
        ) -> None:
            rows.append(
                {
                    "arm": name,
                    "policy": specification["policy"],
                    "execution": specification["dynamics"],
                    "current": specification["current"],
                    "endpoint": endpoint,
                    "estimate": estimate,
                    "median": median,
                    "p95": p95,
                    "observed_n": observed,
                    "missing_n": episodes - observed,
                    "unit": unit,
                }
            )

        append("terminal_joint_success", metrics["terminal_success_count"] / episodes, observed=episodes, unit="proportion")
        append("Tail80_joint_success", metrics["tail80_success_count"] / episodes, observed=episodes, unit="proportion")
        append("ever_lock", metrics["ever_lock_count"] / episodes, observed=episodes, unit="proportion")
        first_track = metrics["first_TRACK_action_time_s"]
        if first_track is not None:
            append(
                "first_TRACK_action_time",
                first_track["mean"],
                observed=first_track["count"],
                unit="s",
                median=first_track["median"],
                p95=first_track["p95"],
            )
        else:
            append(
                "first_TRACK_action_time",
                None,
                observed=0,
                unit="s",
            )
        for endpoint, key, unit in (
            ("terminal_e_p", "terminal_localization_error_m", "m"),
            ("terminal_e_f", "terminal_formation_error_m", "m"),
            ("tail50_mean_e_p", "tail50_mean_localization_error_m", "m"),
            ("tail50_mean_e_f", "tail50_mean_formation_error_m", "m"),
            ("decision_runtime_max_per_episode", "maximum_combined_decision_runtime_s", "s"),
        ):
            distribution = metrics[key]
            append(
                endpoint,
                distribution["mean"],
                observed=distribution["count"],
                unit=unit,
                median=distribution["median"],
                p95=distribution["p95"],
            )
        append(
            "truth_invalid_TRACK_starts",
            metrics["unsafe_track_start_count"] / episodes,
            observed=episodes,
            unit="count_per_episode",
        )
        append(
            "truth_invalid_TRACK_ends",
            metrics["unsafe_track_end_count"] / episodes,
            observed=episodes,
            unit="count_per_episode",
        )
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_publication_artifacts(
    result: Mapping[str, Any],
    output_directory: Path,
    *,
    campaign_directory: Path,
) -> Dict[str, str]:
    """Write compact artifacts outside the immutable campaign directory."""

    destination = output_directory.expanduser().resolve()
    campaign = campaign_directory.expanduser().resolve()
    try:
        destination.relative_to(campaign)
    except ValueError:
        pass
    else:
        _fail("analysis output directory must not be inside the campaign directory")
    destination.mkdir(parents=True, exist_ok=True)
    arm_rows = publication_arm_rows(result)
    plot_rows = interaction_plot_rows(result)
    compact = {
        "schema_version": 1,
        "analysis_version": result["analysis_version"],
        "campaign_source_manifest_sha256": result["contract_source_manifest_sha256"],
        "integrity_valid": result["integrity_valid"],
        "decision": result["decision"],
        "arms": arm_rows,
        "paired_main_effects": result["paired_main_effects"],
        "factorial_interactions_and_difference_in_differences": result[
            "factorial_interactions_and_difference_in_differences"
        ],
        "qualification_screens": result["qualification_screens"],
        "command_execution_semantics": result["command_execution_diagnostics"],
    }
    files = {
        "full_json": destination / "v40_postprocessed_full.json",
        "publication_json": destination / "v40_publication_results.json",
        "arms_csv": destination / "v40_publication_arms.csv",
        "interaction_plot_csv": destination / "v40_interaction_plot.csv",
        "episode_rows_csv": destination / "dynamic_plant_episode_rows.csv",
    }
    _write_json(files["full_json"], result)
    _write_json(files["publication_json"], compact)
    _write_csv(files["arms_csv"], arm_rows)
    _write_csv(files["interaction_plot_csv"], plot_rows)
    _write_csv(files["episode_rows_csv"], result["validated_episode_rows"])
    return {key: str(path) for key, path in files.items()}


def analyze_campaign(
    campaign: Path,
    *,
    environment_metadata: Optional[Path] = None,
    spec: CampaignSpec = QUALIFICATION_SPEC,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    root = campaign.expanduser().resolve()
    _require(root.is_dir(), f"campaign directory does not exist: {root}")
    contract, config, contract_status = validate_contract(
        root,
        metadata_override=environment_metadata,
        spec=spec,
    )
    rows, trace_diagnostics, record_status = load_and_validate_records(
        root,
        config=config,
        spec=spec,
    )
    by_arm = {
        arm: _arm_metrics([row for row in rows if row["arm"] == arm])
        for arm in EXPECTED_ARM_NAMES
    }
    main_contrasts = factorial_main_effect_contrasts(
        rows,
        bootstrap_replicates=bootstrap_replicates,
    )
    interactions = factorial_interactions(
        rows,
        bootstrap_replicates=bootstrap_replicates,
    )
    command_execution = aggregate_command_execution(trace_diagnostics)
    screens = qualification_screens(
        rows,
        by_arm,
        main_contrasts,
        spec=spec,
    )
    integrity = {
        **contract_status,
        **record_status,
        "final_holdout_absent_from_records": True,
        "campaign_summary_files_read": False,
    }
    decision = (
        "V40_QUALIFICATION_COMPLETE"
        if screens["nominal_replication_pass"]
        else "V40_INVALID_OR_NOMINAL_FAIL"
    )
    return {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_source": "per-episode JSON records and NPZ traces only",
        "campaign_directory_name": root.name,
        "contract_source_manifest_sha256": contract["source_manifest_sha256"],
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "base_seed": BOOTSTRAP_SEED,
            "generator": "NumPy PCG64",
        },
        "integrity_checks": integrity,
        "integrity_valid": True,
        "decision": decision,
        "validated_episode_rows": rows,
        "by_arm": by_arm,
        "paired_main_effects": main_contrasts,
        "factorial_interactions_and_difference_in_differences": interactions,
        "command_execution_diagnostics": command_execution,
        "qualification_screens": screens,
        "sealed_final_range": list(SEALED_FINAL_RANGE),
        "retuning_allowed": False,
    }


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="completed V40 qualification output directory")
    parser.add_argument(
        "--environment-metadata",
        type=Path,
        help="frozen metadata.json if the contract's recorded absolute path is unavailable",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        help="write compact JSON/CSV artifacts here; the directory must be outside the campaign",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    try:
        result = analyze_campaign(
            args.campaign,
            environment_metadata=args.environment_metadata,
        )
    except ValidationError as exc:
        print(f"V40 POSTPROCESSOR FAIL: {exc}", file=sys.stderr)
        return 2
    if args.analysis_output_dir is None:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    else:
        try:
            files = write_publication_artifacts(
                result,
                args.analysis_output_dir,
                campaign_directory=args.campaign,
            )
        except (OSError, ValidationError) as exc:
            print(f"V40 ARTIFACT WRITE FAIL: {exc}", file=sys.stderr)
            return 3
        print(
            json.dumps(
                {
                    "analysis": "PASS",
                    "integrity_valid": result["integrity_valid"],
                    "decision": result["decision"],
                    "artifacts": files,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
