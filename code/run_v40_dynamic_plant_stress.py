#!/usr/bin/env python3
"""Run the frozen V40 paired 2x2x2 qualification campaign sequentially."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import run_v20_positioning_ablation as runner20
import uuv_v19_observability as v19
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24
import uuv_v38_leader_source_ablation as v38
import uuv_v40_dynamic_plant_stress as v40


RUNNER_VERSION = "v40_factorial_runner_1.0"
PROTOCOL_NAME = "EXPERIMENT_PROTOCOL_V40_DYNAMIC_PLANT_STRESS.md"
DEFAULT_EPISODES = 100
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 40_049_900

LOCAL_SOURCE_NAMES = (
    PROTOCOL_NAME,
    "uuv_v40_dynamic_plant_stress.py",
    "run_v40_dynamic_plant_stress.py",
    "audit_v40_dynamic_plant_stress.py",
    "tests/test_uuv_v40_dynamic_plant_stress.py",
)
INHERITED_SOURCE_NAMES = (
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


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--coarse-candidates", type=int)
    parser.add_argument("--coarse-sweeps", type=int)
    parser.add_argument("--local-starts", type=int)
    parser.add_argument("--progress-every", type=int, default=4)
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    source_root = Path(
        os.environ.get("UUV_ART2_SOURCE_ROOT", root)
    ).expanduser().resolve()
    smoke = bool(args.smoke)
    maximum = len(v40.SMOKE_SEEDS) if smoke else DEFAULT_EPISODES
    episodes = maximum if args.episodes is None else int(args.episodes)
    start = int(args.episode_start)
    if episodes < 1 or start < 0 or start + episodes > maximum:
        raise ValueError("invalid V40 episode slice")
    if smoke:
        seeds = list(v40.SMOKE_SEEDS[start : start + episodes])
    else:
        seeds = list(
            range(
                v40.QUALIFICATION_START + start,
                v40.QUALIFICATION_START + start + episodes,
            )
        )
    for seed in seeds:
        v40.assert_seed_allowed(seed, smoke=smoke)
    output = (
        args.output_dir
        if args.output_dir is not None
        else source_root
        / (
            "experiments_v40_dynamic_current_factorial_smoke"
            if smoke
            else "experiments_v40_dynamic_current_factorial_qualification100"
        )
    ).expanduser().resolve()
    coarse_candidates = int(
        4096 if args.coarse_candidates is None else args.coarse_candidates
    )
    coarse_sweeps = int(
        2 if args.coarse_sweeps is None else args.coarse_sweeps
    )
    local_starts = int(48 if args.local_starts is None else args.local_starts)
    metadata = Path(runner20._default_metadata(source_root))
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    return {
        "root": root,
        "source_root": source_root,
        "output": output,
        "metadata": metadata,
        "smoke": smoke,
        "resume": bool(args.resume),
        "episode_start": start,
        "episodes": episodes,
        "seeds": seeds,
        "coarse_candidates": coarse_candidates,
        "coarse_sweeps": coarse_sweeps,
        "local_starts": local_starts,
        "publication_settings": bool(
            coarse_candidates == 4096
            and coarse_sweeps == 2
            and local_starts == 48
        ),
        "progress_every": max(1, int(args.progress_every)),
    }


def _sha256(path: Path) -> str:
    return runner20._sha256(path)


def _source_hashes(root: Path, source_root: Path) -> Dict[str, str]:
    paths: List[Tuple[str, Path]] = []
    for name in LOCAL_SOURCE_NAMES:
        paths.append((name, root / name))
    for name in INHERITED_SOURCE_NAMES:
        paths.append((name, source_root / name))
    missing = [str(path) for _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V40 source files: {missing}")
    return {name: _sha256(path) for name, path in paths}


def _fresh_qualification_audit(
    source_root: Path,
    output: Path,
) -> Dict[str, Any]:
    findings: List[str] = []
    filename = re.compile(r"seed_499\d{2}(?:\D|$)")
    content = re.compile(r'"episode_seed"\s*:\s*499\d{2}')
    for directory in sorted(source_root.glob("experiments_*")):
        if directory.resolve() == output.resolve() or not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or "source_snapshot" in path.parts:
                continue
            if filename.search(path.name):
                findings.append(str(path.relative_to(source_root)))
                continue
            if path.suffix == ".json":
                try:
                    if content.search(
                        path.read_text(encoding="utf-8", errors="ignore")
                    ):
                        findings.append(str(path.relative_to(source_root)))
                except OSError:
                    pass
    return {
        "range": [v40.QUALIFICATION_START, v40.QUALIFICATION_END],
        "fresh": not findings,
        "finding_count": len(findings),
        "findings": findings[:100],
    }


def _package_versions() -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for name in ("gymnasium", "numba", "scipy"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _contract(
    settings: Mapping[str, Any],
    cfg: Any,
    estimator: Any,
    freshness: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    sources = _source_hashes(
        Path(settings["root"]),
        Path(settings["source_root"]),
    )
    canonical = json.dumps(sources, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "experiment_version": v40.VERSION,
        "created_at_utc": runner20._utc_now(),
        "purpose": "reviewer-requested low-order dynamics and current robustness",
        "smoke": bool(settings["smoke"]),
        "episodes": int(settings["episodes"]),
        "episode_start": int(settings["episode_start"]),
        "seeds": list(settings["seeds"]),
        "arms": [arm.to_dict() for arm in v40.ARM_SPECS],
        "expected_runs": int(settings["episodes"]) * len(v40.ARM_SPECS),
        "publication_settings": bool(settings["publication_settings"]),
        "estimator_config": v19.config_to_dict(estimator),
        "factorial_contract": v40.condition_contract(),
        "fixed_horizon_actions": int(cfg.max_steps),
        "action_interval_s": float(cfg.action_dt),
        "plant_interval_s": float(cfg.sub_dt),
        "environment_metadata_path": str(settings["metadata"]),
        "environment_metadata_sha256": _sha256(Path(settings["metadata"])),
        "fresh_qualification_audit": freshness,
        "sealed_final_range": [v40.FINAL_START, v40.FINAL_END],
        "source_sha256": sources,
        "source_manifest_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "package_versions": _package_versions(),
        "retuning_allowed": False,
    }


def _immutable(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    return result


def _prepare(
    output: Path,
    contract: Mapping[str, Any],
    resume: bool,
    root: Path,
    source_root: Path,
) -> None:
    contract_path = output / "control" / "campaign_contract.json"
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if _immutable(previous) != _immutable(contract):
            raise RuntimeError("V40 resume contract differs from frozen contract")
        return
    runner20._write_json_atomic(contract_path, contract)
    snapshot = output / "control" / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in LOCAL_SOURCE_NAMES:
        source = root / name
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for name in INHERITED_SOURCE_NAMES:
        source = source_root / name
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _paths(
    output: Path,
    index: int,
    seed: int,
    arm_name: str,
) -> Tuple[Path, Path]:
    result = (
        output
        / "episode_results"
        / arm_name
        / f"episode_{index:04d}_seed_{seed}.json"
    )
    trace = (
        output
        / "traces_npz"
        / arm_name
        / f"episode_{index:04d}_seed_{seed}.npz"
    )
    return result, trace


def _flatten(summary: Mapping[str, Any]) -> Dict[str, Any]:
    exact = summary["gate"]["exact"]
    first_track = exact.get("first_transition_time_s")
    return {
        "episode_index": int(summary["episode_index"]),
        "episode_seed": int(summary["episode_seed"]),
        "arm": str(summary["arm"]),
        "policy": str(summary["policy_name"]),
        "dynamics": str(summary["dynamics_name"]),
        "current": str(summary["current_name"]),
        "terminal_joint_success": bool(summary["terminal_joint_success"]),
        "tail80_joint_success": bool(summary["tail80_joint_success"]),
        "dwell15_joint_success": bool(summary["dwell15_joint_success"]),
        "tail50_joint_occupancy": float(summary["tail50_joint_occupancy"]),
        "terminal_localization_error_m": float(
            summary["terminal_localization_error_m"]
        ),
        "terminal_formation_error_m": float(
            summary["terminal_formation_error_m"]
        ),
        "ever_locked": bool(summary["gate"]["ever_locked"]),
        "first_track_time_s": first_track,
        "unsafe_transition_count": int(exact["false_transition_count"]),
        "unsafe_track_start_count": int(
            exact["false_locked_action_start_count"]
        ),
        "unsafe_track_end_count": int(exact["false_locked_action_end_count"]),
        "audit_release_violation_count": int(
            summary["gate"]["audit_release_violation_count"]
        ),
        "mean_squared_action": float(summary["mean_squared_action"]),
        "maximum_combined_decision_runtime_s": float(
            summary["maximum_combined_decision_runtime_s"]
        ),
        "requested_delivered_action_rms": float(
            summary["plant"]["requested_delivered_action_rms"]
        ),
        "current_speed_rms_mps": float(
            summary["plant"]["current_speed_rms_mps"]
        ),
        "minimum_ground_speed_mps": float(
            summary["plant"]["minimum_ground_speed_mps"]
        ),
        "ground_speed_below_0p05_action_count": int(
            summary["plant"]["ground_speed_below_0p05_action_count"]
        ),
        "action_count": int(summary["action_count"]),
        "noise_tape_sha256": str(summary["noise_tape_sha256"]),
        "current_tape_sha256": str(summary["current_tape_sha256"]),
    }


def _distribution(values: Iterable[Optional[float]]) -> Optional[Dict[str, float]]:
    array = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if not array.size:
        return None
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "episodes": len(rows),
        "terminal_success_count": sum(
            bool(row["terminal_joint_success"]) for row in rows
        ),
        "tail80_success_count": sum(
            bool(row["tail80_joint_success"]) for row in rows
        ),
        "dwell15_success_count": sum(
            bool(row["dwell15_joint_success"]) for row in rows
        ),
        "ever_lock_count": sum(bool(row["ever_locked"]) for row in rows),
        "unsafe_transition_count": sum(
            int(row["unsafe_transition_count"]) for row in rows
        ),
        "unsafe_track_start_count": sum(
            int(row["unsafe_track_start_count"]) for row in rows
        ),
        "unsafe_track_end_count": sum(
            int(row["unsafe_track_end_count"]) for row in rows
        ),
        "audit_release_violation_count": sum(
            int(row["audit_release_violation_count"]) for row in rows
        ),
        "first_track_time_s": _distribution(
            row["first_track_time_s"] for row in rows
        ),
        "terminal_localization_error_m": _distribution(
            row["terminal_localization_error_m"] for row in rows
        ),
        "terminal_formation_error_m": _distribution(
            row["terminal_formation_error_m"] for row in rows
        ),
        "tail50_joint_occupancy": _distribution(
            row["tail50_joint_occupancy"] for row in rows
        ),
        "mean_squared_action": _distribution(
            row["mean_squared_action"] for row in rows
        ),
        "requested_delivered_action_rms": _distribution(
            row["requested_delivered_action_rms"] for row in rows
        ),
        "minimum_ground_speed_mps": _distribution(
            row["minimum_ground_speed_mps"] for row in rows
        ),
        "ground_speed_below_0p05_action_count": sum(
            int(row["ground_speed_below_0p05_action_count"])
            for row in rows
        ),
        "maximum_combined_decision_runtime_s": max(
            (
                float(row["maximum_combined_decision_runtime_s"])
                for row in rows
            ),
            default=0.0,
        ),
        "decision_deadline_miss_episode_count": sum(
            float(row["maximum_combined_decision_runtime_s"]) >= 2.0
            for row in rows
        ),
    }


def _bootstrap(values: Sequence[float], seed: int) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("empty paired difference")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        end = min(start + 1000, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            0,
            array.size,
            size=(end - start, array.size),
        )
        draws[start:end] = np.mean(array[indices], axis=1)
    return {
        "pairs": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "bootstrap_mean_95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": int(seed),
    }


def _mcnemar_exact(reference_only: int, treatment_only: int) -> Optional[float]:
    total = int(reference_only) + int(treatment_only)
    if total == 0:
        return None
    lower = min(int(reference_only), int(treatment_only))
    probability = sum(math.comb(total, k) for k in range(lower + 1)) / (2.0 ** total)
    return float(min(1.0, 2.0 * probability))


def _paired_contrast(
    reference_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    *,
    seed_offset: int,
) -> Dict[str, Any]:
    reference = {int(row["episode_seed"]): row for row in reference_rows}
    treatment = {int(row["episode_seed"]): row for row in treatment_rows}
    if set(reference) != set(treatment):
        raise RuntimeError("paired contrast has unequal seed support")
    result: Dict[str, Any] = {"pairs": len(reference)}
    for endpoint in ("terminal_joint_success", "tail80_joint_success"):
        reference_only = 0
        treatment_only = 0
        differences: List[float] = []
        for seed in sorted(reference):
            left = bool(reference[seed][endpoint])
            right = bool(treatment[seed][endpoint])
            reference_only += int(left and not right)
            treatment_only += int(right and not left)
            differences.append(float(right) - float(left))
        result[endpoint] = {
            "reference_only": reference_only,
            "treatment_only": treatment_only,
            "risk_difference": float(np.mean(differences)),
            "mcnemar_exact_two_sided_p": _mcnemar_exact(
                reference_only,
                treatment_only,
            ),
        }
    for index, endpoint in enumerate(
        ("terminal_localization_error_m", "terminal_formation_error_m")
    ):
        differences = [
            float(treatment[seed][endpoint]) - float(reference[seed][endpoint])
            for seed in sorted(reference)
        ]
        result[f"{endpoint}_treatment_minus_reference"] = _bootstrap(
            differences,
            BOOTSTRAP_SEED + seed_offset * 10 + index,
        )
    return result


def _factorial_contrasts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cells: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for policy in v40.POLICIES:
        for dynamics in v40.DYNAMICS:
            for current in v40.CURRENTS:
                cells[(policy, dynamics, current)] = [
                    row
                    for row in rows
                    if row["policy"] == policy
                    and row["dynamics"] == dynamics
                    and row["current"] == current
                ]
    active_vs_fixed: Dict[str, Any] = {}
    dynamic_vs_kinematic: Dict[str, Any] = {}
    current_vs_none: Dict[str, Any] = {}
    offset = 0
    for dynamics in v40.DYNAMICS:
        for current in v40.CURRENTS:
            name = f"{dynamics}__{current}"
            active_vs_fixed[name] = _paired_contrast(
                cells[(v40.POLICY_FIXED, dynamics, current)],
                cells[(v40.POLICY_ACTIVE, dynamics, current)],
                seed_offset=offset,
            )
            offset += 1
    for policy in v40.POLICIES:
        for current in v40.CURRENTS:
            name = f"{policy}__{current}"
            dynamic_vs_kinematic[name] = _paired_contrast(
                cells[(policy, v40.DYNAMICS_KINEMATIC, current)],
                cells[(policy, v40.DYNAMICS_LOW_ORDER, current)],
                seed_offset=offset,
            )
            offset += 1
    for policy in v40.POLICIES:
        for dynamics in v40.DYNAMICS:
            name = f"{policy}__{dynamics}"
            current_vs_none[name] = _paired_contrast(
                cells[(policy, dynamics, v40.CURRENT_NONE)],
                cells[(policy, dynamics, v40.CURRENT_VISIBLE)],
                seed_offset=offset,
            )
            offset += 1
    return {
        "treatment_minus_reference_convention": True,
        "active_vs_fixed_within_execution_current": active_vs_fixed,
        "dynamic_vs_kinematic_within_policy_current": dynamic_vs_kinematic,
        "current_vs_none_within_policy_execution": current_vs_none,
    }


def _pairing(outcomes: Mapping[str, v38.ArmOutcome]) -> Dict[str, Any]:
    if set(outcomes) != set(v40.ARM_BY_NAME):
        raise RuntimeError("V40 seed lacks one or more arms")
    ordered = [outcomes[arm.name] for arm in v40.ARM_SPECS]
    reference = ordered[0].summary
    for outcome in ordered[1:]:
        summary = outcome.summary
        for key in (
            "episode_seed",
            "episode_index",
            "noise_tape_sha256",
            "current_tape_sha256",
        ):
            if summary[key] != reference[key]:
                raise RuntimeError(f"V40 pairing differs in {key}")
        for key in ("initial_truth_m", "mission_support"):
            if not np.array_equal(
                np.asarray(
                    summary[key]["center_m"] if key == "mission_support" else summary[key]
                ),
                np.asarray(
                    reference[key]["center_m"] if key == "mission_support" else reference[key]
                ),
            ):
                raise RuntimeError(f"V40 pairing differs in {key}")
    return {
        "episode_seed": int(reference["episode_seed"]),
        "arm_count": len(ordered),
        "initial_state_equal": True,
        "sensor_noise_tape_equal": True,
        "current_tape_equal": True,
    }


def _reference_equivalence(
    v40_outcome: v38.ArmOutcome,
    direct_outcome: v38.ArmOutcome,
) -> Dict[str, Any]:
    excluded = {"planner_runtime_s"}
    common = sorted(set(v40_outcome.trace) & set(direct_outcome.trace) - excluded)
    unequal = [
        key
        for key in common
        if not np.array_equal(
            v40_outcome.trace[key],
            direct_outcome.trace[key],
            equal_nan=True,
        )
    ]
    return {
        "episode_seed": int(v40_outcome.summary["episode_seed"]),
        "policy": str(v40_outcome.summary["policy_name"]),
        "compared_trace_fields": len(common),
        "excluded_nondeterministic_trace_fields": sorted(excluded),
        "unequal_trace_fields": unequal,
        "bitwise_equal": not unequal,
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    elapsed_s: float,
    pairings: Sequence[Mapping[str, Any]],
    equivalence: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_arm = {
        arm.name: _metrics([row for row in rows if row["arm"] == arm.name])
        for arm in v40.ARM_SPECS
    }
    expected = int(contract["expected_runs"])
    smoke = bool(contract["smoke"])
    integrity = {
        "all_runs_complete": len(rows) == expected,
        "all_fixed_horizon": all(
            int(row["action_count"]) == int(contract["fixed_horizon_actions"])
            for row in rows
        ),
        "publication_settings": bool(contract["publication_settings"]),
        "pairing_complete": len(pairings) == int(contract["episodes"]),
        "pairing_all_eight_arms": all(
            int(record["arm_count"]) == len(v40.ARM_SPECS)
            for record in pairings
        ),
        "reference_equivalence": bool(
            (not smoke)
            or (
                len(equivalence) == 2 * int(contract["episodes"])
                and all(record["bitwise_equal"] for record in equivalence)
            )
        ),
        "audit_release_violations_zero": sum(
            int(row["audit_release_violation_count"]) for row in rows
        )
        == 0,
        "qualification_was_fresh": bool(
            smoke or contract["fresh_qualification_audit"]["fresh"]
        ),
        "final_holdout_sealed": contract["sealed_final_range"]
        == [v40.FINAL_START, v40.FINAL_END],
    }
    nominal_name = v40.ArmSpec(
        v40.POLICY_ACTIVE,
        v40.DYNAMICS_KINEMATIC,
        v40.CURRENT_NONE,
    ).name
    nominal = by_arm[nominal_name]
    nominal_pass = bool(
        smoke
        or (
            nominal["terminal_success_count"] >= 95
            and nominal["tail80_success_count"] >= 90
            and nominal["ever_lock_count"] >= 95
            and nominal["unsafe_track_start_count"] == 0
            and nominal["unsafe_track_end_count"] == 0
        )
    )
    screens: Dict[str, str] = {}
    for arm in v40.ARM_SPECS:
        if arm.policy != v40.POLICY_ACTIVE:
            continue
        metric = by_arm[arm.name]
        if arm.name == nominal_name:
            label = "NOMINAL_REPLICATION_PASS" if nominal_pass else "NOMINAL_REPLICATION_FAIL"
        else:
            label = (
                "ROBUST_WITHIN_SCREEN"
                if metric["terminal_success_count"] >= (1 if smoke else 90)
                and metric["tail80_success_count"] >= (1 if smoke else 85)
                and metric["unsafe_track_start_count"] == 0
                and metric["unsafe_track_end_count"] == 0
                else "OUTSIDE_ROBUSTNESS_SCREEN"
            )
        screens[arm.name] = label
    integrity_valid = all(integrity.values())
    decision = (
        "SMOKE_PASS"
        if smoke and integrity_valid
        else (
            "V40_QUALIFICATION_COMPLETE"
            if integrity_valid and nominal_pass
            else "V40_INVALID_OR_NOMINAL_FAIL"
        )
    )
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "analysis_version": "v40_factorial_analysis_1.0",
        "status": "complete",
        "smoke": smoke,
        "elapsed_wall_s": float(elapsed_s),
        "row_count": len(rows),
        "expected_row_count": expected,
        "integrity_checks": integrity,
        "integrity_valid": integrity_valid,
        "nominal_replication_pass": nominal_pass,
        "decision": decision,
        "by_arm": by_arm,
        "active_policy_screens": screens,
        "factorial_contrasts": _factorial_contrasts(rows),
        "pairing_records": list(pairings),
        "reference_equivalence_records": list(equivalence),
        "final_holdout_sealed": [v40.FINAL_START, v40.FINAL_END],
        "retuning_allowed": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    settings = _settings(args)
    root = Path(settings["root"])
    source_root = Path(settings["source_root"])
    output = Path(settings["output"])
    freshness = (
        None
        if settings["smoke"]
        else _fresh_qualification_audit(source_root, output)
    )
    if freshness is not None and not freshness["fresh"]:
        raise RuntimeError(
            f"V40 qualification seed audit failed: {freshness['findings'][:10]}"
        )
    cfg = runner20._load_environment_config(Path(settings["metadata"]))
    estimator_config = v19.BatchEstimatorConfig(
        coarse_candidates=int(settings["coarse_candidates"]),
        coarse_sweeps=int(settings["coarse_sweeps"]),
        local_starts=int(settings["local_starts"]),
        gate_mode="raw",
        candidate_radial_distribution="uniform_radius",
    )
    planner_config = v22.ActivePlannerConfig()
    lock_config = v24.AuditedLockConfig()
    contract = _contract(settings, cfg, estimator_config, freshness)
    _prepare(
        output,
        contract,
        bool(settings["resume"]),
        root,
        source_root,
    )
    frozen_sources = dict(contract["source_sha256"])
    rows: List[Dict[str, Any]] = []
    pairings: List[Dict[str, Any]] = []
    equivalence: List[Dict[str, Any]] = []
    completed = 0
    total = int(contract["expected_runs"])
    started = time.perf_counter()
    try:
        for local_index, seed in enumerate(settings["seeds"]):
            episode_index = int(settings["episode_start"]) + local_index
            tape = runner20._tape_for_episode(
                output,
                cfg,
                int(seed),
                episode_index,
            )
            outcomes: Dict[str, v38.ArmOutcome] = {}
            for arm in v40.ARM_SPECS:
                result_path, trace_path = _paths(
                    output,
                    episode_index,
                    int(seed),
                    arm.name,
                )
                if (
                    settings["resume"]
                    and result_path.is_file()
                    and trace_path.is_file()
                ):
                    summary = json.loads(result_path.read_text(encoding="utf-8"))
                    with np.load(trace_path, allow_pickle=False) as archive:
                        trace = {key: archive[key].copy() for key in archive.files}
                    outcome = v38.ArmOutcome(summary=summary, trace=trace)
                else:
                    outcome = v40.run_factorial_arm(
                        arm=arm,
                        cfg=cfg,
                        tape=tape,
                        episode_seed=int(seed),
                        episode_index=episode_index,
                        estimator_config=estimator_config,
                        lock_config=lock_config,
                        planner_config=planner_config,
                    )
                    runner20._write_json_atomic(result_path, outcome.summary)
                    runner20._write_npz_atomic(trace_path, outcome.trace)
                if outcome.summary["noise_tape_sha256"] != tape.content_sha256():
                    raise RuntimeError("V40 saved the wrong sensor-noise tape")
                if int(outcome.summary["action_count"]) != int(cfg.max_steps):
                    raise RuntimeError("V40 saved an incomplete episode")
                outcomes[arm.name] = outcome
                rows.append(_flatten(outcome.summary))
                completed += 1
                if _source_hashes(root, source_root) != frozen_sources:
                    raise RuntimeError("V40 source closure changed during execution")
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (total - completed)
                runner20._write_json_atomic(
                    output / "control" / "progress.json",
                    {
                        "status": "running",
                        "updated_at_utc": runner20._utc_now(),
                        "completed_runs": completed,
                        "total_runs": total,
                        "elapsed_wall_s": elapsed,
                        "estimated_remaining_s": eta,
                        "last_episode_index": episode_index,
                        "last_seed": int(seed),
                        "last_arm": arm.name,
                        "resumable": True,
                    },
                )
                if completed % int(settings["progress_every"]) == 0:
                    print(
                        f"[{runner20._utc_now()}] V40 {completed}/{total}; "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s; "
                        f"seed={seed} arm={arm.name}",
                        flush=True,
                    )
            pairings.append(_pairing(outcomes))

            if settings["smoke"]:
                for policy in v40.POLICIES:
                    reference_arm = v40.ArmSpec(
                        policy,
                        v40.DYNAMICS_KINEMATIC,
                        v40.CURRENT_NONE,
                    )
                    direct = v38.run_arm(
                        cfg=cfg,
                        tape=tape,
                        episode_seed=int(seed),
                        episode_index=episode_index,
                        source_name=v38.SOURCE_BOTH,
                        policy_name=policy,
                        estimator_config=estimator_config,
                        lock_config=lock_config,
                        planner_config=planner_config,
                    )
                    record = _reference_equivalence(
                        outcomes[reference_arm.name],
                        direct,
                    )
                    equivalence.append(record)
                    if not record["bitwise_equal"]:
                        raise RuntimeError(
                            "V40 kinematic/no-current path differs from V38: "
                            f"{record}"
                        )

        elapsed = time.perf_counter() - started
        runner20._write_csv_atomic(
            output / "episode_arm_summary.csv",
            rows,
        )
        summary = _aggregate(
            rows,
            contract,
            elapsed,
            pairings,
            equivalence,
        )
        runner20._write_json_atomic(output / "campaign_summary.json", summary)
        runner20._write_json_atomic(
            output / "decision.json",
            {
                "decision": summary["decision"],
                "integrity_valid": summary["integrity_valid"],
                "nominal_replication_pass": summary["nominal_replication_pass"],
                "active_policy_screens": summary["active_policy_screens"],
                "final_holdout_sealed": summary["final_holdout_sealed"],
                "retuning_allowed": False,
            },
        )
        runner20._write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "complete",
                "updated_at_utc": runner20._utc_now(),
                "completed_runs": completed,
                "total_runs": total,
                "elapsed_wall_s": elapsed,
                "estimated_remaining_s": 0.0,
                "resumable": True,
            },
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        elapsed = time.perf_counter() - started
        runner20._write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "failed",
                "updated_at_utc": runner20._utc_now(),
                "completed_runs": completed,
                "total_runs": total,
                "elapsed_wall_s": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "resumable": True,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
