#!/usr/bin/env python3
"""Run the frozen V35 sequential closed-loop stress campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
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
import uuv_v27_publication_baselines as v27
import uuv_v28_estimator_stress as v28
import uuv_v35_closed_loop_stress as v35


RUNNER_VERSION = "v35_closed_loop_stress_runner_1.0"
PROTOCOL_NAME = "EXPERIMENT_PROTOCOL_V35_CLOSED_LOOP_STRESS.md"
DEFAULT_SEED_START = 48_100
DEFAULT_EPISODES = 100
SMOKE_SEEDS = (49_566, 49_591)
BOOTSTRAP_REPLICATES = 50_000


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
    smoke = bool(args.smoke)
    default_count = len(SMOKE_SEEDS) if smoke else DEFAULT_EPISODES
    episodes = int(args.episodes if args.episodes is not None else default_count)
    if episodes < 1 or episodes > default_count:
        raise ValueError("invalid V35 episode count")
    episode_start = int(args.episode_start)
    if episode_start < 0 or episode_start + episodes > default_count:
        raise ValueError("invalid V35 episode slice")
    output = (
        args.output_dir
        if args.output_dir is not None
        else root
        / (
            "experiments_v35_closed_loop_stress_smoke"
            if smoke
            else "experiments_v35_closed_loop_stress_dev100"
        )
    ).expanduser().resolve()
    seeds = (
        list(SMOKE_SEEDS[episode_start : episode_start + episodes])
        if smoke
        else list(
            range(
                DEFAULT_SEED_START + episode_start,
                DEFAULT_SEED_START + episode_start + episodes,
            )
        )
    )
    for seed in seeds:
        v35.assert_v35_seed_allowed(seed)
    settings = {
        "root": root,
        "output": output,
        "metadata": runner20._default_metadata(root),
        "smoke": smoke,
        "resume": bool(args.resume),
        "episode_start": episode_start,
        "episodes": episodes,
        "seeds": seeds,
        "coarse_candidates": int(
            args.coarse_candidates
            if args.coarse_candidates is not None
            else (256 if smoke else 4096)
        ),
        "coarse_sweeps": int(
            args.coarse_sweeps
            if args.coarse_sweeps is not None
            else (1 if smoke else 2)
        ),
        "local_starts": int(
            args.local_starts
            if args.local_starts is not None
            else (8 if smoke else 48)
        ),
        "progress_every": max(1, int(args.progress_every)),
    }
    if not Path(settings["metadata"]).is_file():
        raise FileNotFoundError(settings["metadata"])
    return settings


def _sha256(path: Path) -> str:
    return runner20._sha256(path)


def _loaded_local_sources(root: Path) -> Dict[str, str]:
    root = root.resolve()
    paths = {
        root / PROTOCOL_NAME,
        root / "audit_v35_campaign.py",
        root / "tests/test_uuv_v35_closed_loop_stress.py",
    }
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename)
        if path.suffix in {".pyc", ".pyo"}:
            try:
                path = Path(importlib.util.source_from_cache(str(path)))
            except (ValueError, NotImplementedError):
                continue
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.suffix == ".py" and not any(
            part.startswith(".venv") for part in relative.parts
        ):
            paths.add(resolved)
    required = {
        PROTOCOL_NAME,
        "run_v35_closed_loop_stress.py",
        "uuv_v35_closed_loop_stress.py",
        "uuv_v24_audited_gate.py",
        "uuv_v22_active_acquisition.py",
        "uuv_v21_causal_lock.py",
        "uuv_v20_positioning_ablation.py",
        "uuv_v19_observability.py",
        "uuv_v30_profiled_bias.py",
        "uuv_v31_bias_evidence_gate.py",
        "uuv_v32_bias_time_to_evidence.py",
        "audit_v35_campaign.py",
        "tests/test_uuv_v35_closed_loop_stress.py",
    }
    manifest = {
        str(path.relative_to(root)): _sha256(path)
        for path in paths
        if path.is_file()
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"V35 source manifest missed: {missing}")
    return dict(sorted(manifest.items()))


def _package_versions() -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for name in ("gymnasium", "numba"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _fresh_seed_audit(root: Path, output: Path) -> Dict[str, Any]:
    """Reject prior structured scenario artifacts for 48100..48199."""

    findings: List[str] = []
    filename_pattern = re.compile(r"seed_(481\d{2})(?:\D|$)")
    json_pattern = re.compile(r'"episode_seed"\s*:\s*(481\d{2})')
    csv_pattern = re.compile(r"(?:^|,)(481\d{2})(?:,|$)")
    for directory in sorted(root.glob("experiments_*")):
        try:
            if directory.resolve() == output.resolve():
                continue
        except OSError:
            continue
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or "source_snapshot" in path.parts:
                continue
            match = filename_pattern.search(path.name)
            if match:
                findings.append(str(path.relative_to(root)))
                continue
            if path.suffix not in {".json", ".csv"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if json_pattern.search(text):
                findings.append(str(path.relative_to(root)))
            elif (
                path.suffix == ".csv"
                and text.splitlines()
                and "episode_seed" in text.splitlines()[0]
                and csv_pattern.search(text)
            ):
                findings.append(str(path.relative_to(root)))
    return {
        "range": [DEFAULT_SEED_START, DEFAULT_SEED_START + DEFAULT_EPISODES - 1],
        "fresh": not findings,
        "finding_count": len(findings),
        "findings": findings[:100],
    }


def _contract(
    settings: Mapping[str, Any],
    cfg: Any,
    estimator_config: v19.BatchEstimatorConfig,
    evaluator_config: v27.EvaluatorConfig,
    planner_config: v22.ActivePlannerConfig,
    lock_config: v24.AuditedLockConfig,
    freshness: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    manifest = _loaded_local_sources(Path(settings["root"]))
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v35.VERSION,
        "created_at_utc": runner20._utc_now(),
        "purpose": "frozen development-only V35 closed-loop stress campaign",
        "smoke": bool(settings["smoke"]),
        "episode_start": int(settings["episode_start"]),
        "episodes": int(settings["episodes"]),
        "seeds": list(settings["seeds"]),
        "conditions": list(v35.CONDITIONS),
        "bias_switch_conditions": list(v35.BIAS_SWITCH_CONDITIONS),
        "arm_condition_pairs": [list(value) for value in v35.arm_condition_pairs()],
        "expected_runs": int(settings["episodes"]) * len(v35.arm_condition_pairs()),
        "estimator_config": v19.config_to_dict(estimator_config),
        "evaluator_config": dict(evaluator_config.__dict__),
        "planner_config": dict(planner_config.__dict__),
        "audited_lock_config": lock_config.to_dict(),
        "model_decision_time_s": v35.MODEL_DECISION_TIME_S,
        "environment_metadata_path": str(settings["metadata"]),
        "environment_metadata_sha256": _sha256(Path(settings["metadata"])),
        "fixed_horizon_actions": int(cfg.max_steps),
        "stress_contract": v28.condition_contract(),
        "fresh_seed_audit": freshness,
        "reserved_final_range_untouched": [v35.RESERVED_START, v35.FINAL_END],
        "source_sha256": manifest,
        "source_manifest_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "package_versions": _package_versions(),
    }


def _immutable(contract: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(contract)
    value.pop("created_at_utc", None)
    return value


def _prepare(output: Path, contract: Mapping[str, Any], resume: bool) -> None:
    contract_path = output / "control" / "campaign_contract.json"
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if _immutable(previous) != _immutable(contract):
            raise RuntimeError("V35 resume contract differs from frozen contract")
    else:
        runner20._write_json_atomic(contract_path, contract)
        snapshot = output / "control" / "source_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        # The hash manifest is authoritative; copy only direct V35/protocol
        # sources to keep the campaign compact.
        local_root = Path(__file__).resolve().parent
        for name in (
            PROTOCOL_NAME,
            "uuv_v35_closed_loop_stress.py",
            "run_v35_closed_loop_stress.py",
            "audit_v35_campaign.py",
            "tests/test_uuv_v35_closed_loop_stress.py",
        ):
            source = local_root / name
            if source.is_file():
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())


def _result_paths(
    output: Path,
    episode_index: int,
    seed: int,
    condition: str,
    arm: str,
) -> Tuple[Path, Path]:
    result = (
        output
        / "episode_results"
        / condition
        / f"episode_{episode_index:04d}_seed_{seed}_{arm}.json"
    )
    trace = (
        output
        / "traces_npz"
        / condition
        / arm
        / f"episode_{episode_index:04d}_seed_{seed}.npz"
    )
    return result, trace


def _flatten(summary: Mapping[str, Any]) -> Dict[str, Any]:
    exact = summary["gate"]["exact"]
    return {
        "episode_index": int(summary["episode_index"]),
        "episode_seed": int(summary["episode_seed"]),
        "condition": str(summary["condition"]),
        "arm": str(summary["arm"]),
        "terminal_joint_success": bool(summary["terminal_joint_success"]),
        "tail80_joint_success": bool(summary["tail80_joint_success"]),
        "dwell15_joint_success": bool(summary["dwell15_joint_success"]),
        "tail50_joint_occupancy": float(summary["tail50_joint_occupancy"]),
        "terminal_localization_error_m": float(summary["terminal_localization_error_m"]),
        "terminal_formation_error_m": float(summary["terminal_formation_error_m"]),
        "ever_locked": bool(summary["gate"]["ever_locked"]),
        "unsafe_transition_count": int(exact["false_transition_count"]),
        "unsafe_track_start_count": int(exact["false_locked_action_start_count"]),
        "unsafe_track_end_count": int(exact["false_locked_action_end_count"]),
        "post300_unsafe_track_end_count": int(summary["gate"]["post300_unsafe_track_end_count"]),
        "audit_release_violation_count": int(summary["gate"]["audit_release_violation_count"]),
        "mean_squared_action": float(summary["mean_squared_action"]),
        "maximum_combined_decision_runtime_s": float(summary["maximum_combined_decision_runtime_s"]),
        "model_evaluated": bool(summary["model_switch"]["evaluated"]),
        "model_activated": bool(summary["model_switch"]["activated"]),
        "noise_tape_sha256": str(summary["noise_tape_sha256"]),
        "stress_tape_sha256": str(summary["stress_tape_sha256"]),
        "online_history_sha256": str(summary["online_history_sha256"]),
    }


def _distribution(values: Iterable[float]) -> Optional[Dict[str, float]]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    return {
        "episodes": count,
        "terminal_success_count": int(sum(bool(row["terminal_joint_success"]) for row in rows)),
        "tail80_success_count": int(sum(bool(row["tail80_joint_success"]) for row in rows)),
        "dwell15_success_count": int(sum(bool(row["dwell15_joint_success"]) for row in rows)),
        "ever_lock_count": int(sum(bool(row["ever_locked"]) for row in rows)),
        "unsafe_transition_count": int(sum(int(row["unsafe_transition_count"]) for row in rows)),
        "unsafe_track_start_count": int(sum(int(row["unsafe_track_start_count"]) for row in rows)),
        "unsafe_track_end_count": int(sum(int(row["unsafe_track_end_count"]) for row in rows)),
        "post300_unsafe_track_end_count": int(sum(int(row["post300_unsafe_track_end_count"]) for row in rows)),
        "audit_release_violation_count": int(sum(int(row["audit_release_violation_count"]) for row in rows)),
        "model_activation_count": int(sum(bool(row["model_activated"]) for row in rows)),
        "terminal_localization_error_m": _distribution(row["terminal_localization_error_m"] for row in rows),
        "terminal_formation_error_m": _distribution(row["terminal_formation_error_m"] for row in rows),
        "tail50_joint_occupancy": _distribution(row["tail50_joint_occupancy"] for row in rows),
        "mean_squared_action": _distribution(row["mean_squared_action"] for row in rows),
        "maximum_combined_decision_runtime_s": max(
            (float(row["maximum_combined_decision_runtime_s"]) for row in rows),
            default=0.0,
        ),
    }


def _bootstrap_mean(values: Sequence[float], seed: int) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    output = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        end = min(start + chunk, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, array.size, size=(end - start, array.size))
        output[start:end] = np.mean(array[indices], axis=1)
    return {
        "pairs": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "bootstrap_mean_95": [
            float(np.percentile(output, 2.5)),
            float(np.percentile(output, 97.5)),
        ],
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": int(seed),
    }


def _verify_treatment_pair(
    primary: v35.V35ArmOutcome,
    treatment: v35.V35ArmOutcome,
) -> Dict[str, Any]:
    for key in ("episode_seed", "condition", "noise_tape_sha256", "stress_tape_sha256"):
        if primary.summary[key] != treatment.summary[key]:
            raise RuntimeError(f"V35 treatment pair differs in {key}")
    fields = (
        "action_speed", "action_yaw", "action_pitch", "truth_x", "truth_y", "truth_z",
        "estimate_x", "estimate_y", "estimate_z", "phase_track",
        "gate_locked_after_update", "localization_error_m", "formation_error_truth_m",
    )
    times = np.asarray(primary.trace["time_s"], dtype=np.float64)
    mask = times <= v35.MODEL_DECISION_TIME_S + 1e-9
    for field in fields:
        left = np.asarray(primary.trace[field])[mask]
        right = np.asarray(treatment.trace[field])[mask]
        if left.dtype.kind in "fc":
            equal = np.array_equal(left, right, equal_nan=True)
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise RuntimeError(f"V35 treatment pair diverged before/at 300 s in {field}")
    return {
        "paired_through_time_s": v35.MODEL_DECISION_TIME_S,
        "verified_field_count": len(fields),
        "treatment_activated": bool(treatment.summary["model_switch"]["activated"]),
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    elapsed_s: float,
    pairing: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_cell: Dict[str, Any] = {}
    for condition, arm in v35.arm_condition_pairs():
        cell_rows = [
            row for row in rows
            if row["condition"] == condition and row["arm"] == arm
        ]
        by_cell[f"{condition}@{arm}"] = _arm_metrics(cell_rows)
    smoke = bool(contract["smoke"])
    expected = int(contract["expected_runs"])
    integrity_checks = {
        "all_runs_complete": len(rows) == expected,
        "all_fixed_horizon": True,
        "pairing_complete": len(pairing)
        == int(contract["episodes"]) * len(v35.BIAS_SWITCH_CONDITIONS),
        "maximum_runtime_below_2s": max(
            (float(row["maximum_combined_decision_runtime_s"]) for row in rows),
            default=float("inf"),
        ) < 2.0,
        "audit_release_violations_zero": sum(
            int(row["audit_release_violation_count"]) for row in rows
        ) == 0,
        "reserved_final_untouched": contract["reserved_final_range_untouched"]
        == [v35.RESERVED_START, v35.FINAL_END],
        "fresh_seed_audit_pass": bool(smoke or contract["fresh_seed_audit"]["fresh"]),
    }
    integrity_valid = bool(all(integrity_checks.values()))
    profile_effects: Dict[str, Any] = {}
    indexed = {
        (int(row["episode_seed"]), str(row["condition"]), str(row["arm"])): row
        for row in rows
    }
    for condition_index, condition in enumerate(v35.BIAS_SWITCH_CONDITIONS):
        differences: List[float] = []
        terminal_primary_only = terminal_treatment_only = 0
        tail_primary_only = tail_treatment_only = 0
        for seed in contract["seeds"]:
            primary = indexed[(int(seed), condition, v35.PRIMARY_ARM)]
            treatment = indexed[(int(seed), condition, v35.BIAS_SWITCH_ARM)]
            differences.append(
                float(primary["terminal_localization_error_m"])
                - float(treatment["terminal_localization_error_m"])
            )
            p_term = bool(primary["terminal_joint_success"])
            t_term = bool(treatment["terminal_joint_success"])
            terminal_primary_only += int(p_term and not t_term)
            terminal_treatment_only += int(t_term and not p_term)
            p_tail = bool(primary["tail80_joint_success"])
            t_tail = bool(treatment["tail80_joint_success"])
            tail_primary_only += int(p_tail and not t_tail)
            tail_treatment_only += int(t_tail and not p_tail)
        profile_effects[condition] = {
            "terminal_localization_improvement_m": _bootstrap_mean(
                differences, v35.BOOTSTRAP_SEED + condition_index
            ),
            "terminal_discordance": {
                "primary_only": terminal_primary_only,
                "treatment_only": terminal_treatment_only,
            },
            "tail80_discordance": {
                "primary_only": tail_primary_only,
                "treatment_only": tail_treatment_only,
            },
        }

    stress_decisions: Dict[str, str] = {}
    for condition in v35.CONDITIONS:
        metrics = by_cell[f"{condition}@{v35.PRIMARY_ARM}"]
        count = max(1, int(metrics["episodes"]))
        supported = bool(
            int(metrics["terminal_success_count"]) / count >= 0.90
            and int(metrics["tail80_success_count"]) / count >= 0.85
            and int(metrics["unsafe_transition_count"]) == 0
            and int(metrics["unsafe_track_start_count"]) == 0
            and int(metrics["unsafe_track_end_count"]) == 0
        )
        stress_decisions[condition] = (
            "SUPPORTED_STRESS_FAMILY" if supported else "UNSUPPORTED_STRESS_FAMILY"
        )

    if smoke:
        decision = "SMOKE_PASS" if integrity_valid else "SMOKE_FAIL"
        profile_decision = "SMOKE_ONLY"
        nominal_decision = "SMOKE_ONLY"
    else:
        nominal = by_cell[f"{v28.NOMINAL}@{v35.PRIMARY_ARM}"]
        nominal_decision = (
            "SUPPORT_V24_NOMINAL"
            if integrity_valid
            and nominal["terminal_success_count"] >= 95
            and nominal["tail80_success_count"] >= 90
            and nominal["ever_lock_count"] >= 95
            and nominal["unsafe_transition_count"] == 0
            and nominal["unsafe_track_start_count"] == 0
            and nominal["unsafe_track_end_count"] == 0
            else "DO_NOT_SUPPORT_V24_NOMINAL"
        )

        def cell(condition: str, arm: str) -> Mapping[str, Any]:
            return by_cell[f"{condition}@{arm}"]

        common_p = cell(v28.DOPPLER_COMMON_BIAS, v35.PRIMARY_ARM)
        common_t = cell(v28.DOPPLER_COMMON_BIAS, v35.BIAS_SWITCH_ARM)
        diff_p = cell(v28.DOPPLER_DIFFERENTIAL_BIAS, v35.PRIMARY_ARM)
        diff_t = cell(v28.DOPPLER_DIFFERENTIAL_BIAS, v35.BIAS_SWITCH_ARM)
        nominal_p = cell(v28.NOMINAL, v35.PRIMARY_ARM)
        nominal_t = cell(v28.NOMINAL, v35.BIAS_SWITCH_ARM)
        color_p = cell(v28.COLORED_NOISE, v35.PRIMARY_ARM)
        color_t = cell(v28.COLORED_NOISE, v35.BIAS_SWITCH_ARM)

        bias_endpoint_checks = []
        for condition, primary_metrics, treatment_metrics in (
            (v28.DOPPLER_COMMON_BIAS, common_p, common_t),
            (v28.DOPPLER_DIFFERENTIAL_BIAS, diff_p, diff_t),
        ):
            terminal_gain = treatment_metrics["terminal_success_count"] - primary_metrics["terminal_success_count"]
            tail_gain = treatment_metrics["tail80_success_count"] - primary_metrics["tail80_success_count"]
            effect = profile_effects[condition]["terminal_localization_improvement_m"]
            bias_endpoint_checks.append(
                bool(
                    treatment_metrics["model_activation_count"] >= 95
                    and treatment_metrics["episodes"] == 100
                    and treatment_metrics["terminal_localization_error_m"]["p95"] <= 3.0
                    and sum(
                        float(row["terminal_localization_error_m"]) < 7.0
                        for row in rows
                        if row["condition"] == condition and row["arm"] == v35.BIAS_SWITCH_ARM
                    ) >= 95
                    and effect["bootstrap_mean_95"][0] > 1.0
                    and max(terminal_gain, tail_gain) >= 5
                    and terminal_gain >= -2
                    and tail_gain >= -2
                    and treatment_metrics["post300_unsafe_track_end_count"] == 0
                )
            )
        activation_checks = bool(
            nominal_t["model_activation_count"] <= 5
            and color_t["model_activation_count"] <= 10
        )
        nominal_regression = bool(
            nominal_t["terminal_success_count"] >= nominal_p["terminal_success_count"] - 2
            and nominal_t["tail80_success_count"] >= nominal_p["tail80_success_count"] - 2
            and nominal_t["terminal_localization_error_m"]["p95"]
            <= nominal_p["terminal_localization_error_m"]["p95"] + 0.5
        )
        colored_regression = bool(
            color_t["terminal_success_count"] >= color_p["terminal_success_count"] - 5
            and color_t["tail80_success_count"] >= color_p["tail80_success_count"] - 5
            and color_t["terminal_localization_error_m"]["p95"]
            <= color_p["terminal_localization_error_m"]["p95"] + 1.0
        )
        profile_decision = (
            "RETAIN_EVIDENCE_QUALIFIED_PROFILED_BIAS_SWITCH"
            if integrity_valid
            and all(bias_endpoint_checks)
            and activation_checks
            and nominal_regression
            and colored_regression
            else "DO_NOT_RETAIN_PROFILED_SWITCH_FOR_MAIN_METHOD"
        )
        decision = "V35_COMPLETE" if integrity_valid else "V35_INVALID"

    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "analysis_version": "v35_analysis_1.0",
        "status": "complete",
        "smoke": smoke,
        "elapsed_wall_s": float(elapsed_s),
        "row_count": len(rows),
        "expected_row_count": expected,
        "integrity_checks": integrity_checks,
        "integrity_valid": integrity_valid,
        "decision": decision,
        "nominal_decision": nominal_decision,
        "profile_switch_decision": profile_decision,
        "stress_family_decisions": stress_decisions,
        "by_condition_arm": by_cell,
        "profile_paired_effects": profile_effects,
        "treatment_pairing": list(pairing),
        "development_only": True,
        "reserved_final_range_untouched": contract["reserved_final_range_untouched"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    settings = _settings(args)
    root = Path(settings["root"])
    output = Path(settings["output"])
    freshness = None
    if not settings["smoke"]:
        freshness = _fresh_seed_audit(root, output)
        if not freshness["fresh"]:
            raise RuntimeError(f"V35 fresh-seed audit failed: {freshness['findings'][:10]}")
    cfg = runner20._load_environment_config(Path(settings["metadata"]))
    estimator_config = v19.BatchEstimatorConfig(
        coarse_candidates=int(settings["coarse_candidates"]),
        coarse_sweeps=int(settings["coarse_sweeps"]),
        local_starts=int(settings["local_starts"]),
        gate_mode="raw",
        candidate_radial_distribution="uniform_radius",
    )
    evaluator_config = v27.EvaluatorConfig(
        measurement_sigma_mps=float(estimator_config.measurement_sigma_mps),
        coarse_candidates=int(settings["coarse_candidates"]),
        coarse_sweeps=int(settings["coarse_sweeps"]),
        local_starts=int(settings["local_starts"]),
        maximum_modes=int(estimator_config.maximum_modes),
    )
    planner_config = v22.ActivePlannerConfig()
    lock_config = v24.AuditedLockConfig()
    contract = _contract(
        settings,
        cfg,
        estimator_config,
        evaluator_config,
        planner_config,
        lock_config,
        freshness,
    )
    _prepare(output, contract, bool(settings["resume"]))
    expected_sources = dict(contract["source_sha256"])
    rows: List[Dict[str, Any]] = []
    pairing_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    completed = 0
    total = int(contract["expected_runs"])
    try:
        for local_index, seed in enumerate(settings["seeds"]):
            episode_index = int(settings["episode_start"]) + local_index
            tape = runner20._tape_for_episode(output, cfg, int(seed), episode_index)
            for condition in v35.CONDITIONS:
                stress_tape = v35.make_stress_tape(condition, episode_index)
                condition_outcomes: Dict[str, v35.V35ArmOutcome] = {}
                arms = [v35.PRIMARY_ARM]
                if condition in v35.BIAS_SWITCH_CONDITIONS:
                    arms.append(v35.BIAS_SWITCH_ARM)
                for arm in arms:
                    result_path, trace_path = _result_paths(
                        output, episode_index, int(seed), condition, arm
                    )
                    if settings["resume"] and result_path.is_file() and trace_path.is_file():
                        summary = json.loads(result_path.read_text(encoding="utf-8"))
                        with np.load(trace_path, allow_pickle=False) as archive:
                            trace = {key: archive[key].copy() for key in archive.files}
                        outcome = v35.V35ArmOutcome(summary=summary, trace=trace)
                    else:
                        outcome = v35.run_stressed_arm(
                            cfg=cfg,
                            tape=tape,
                            stress_tape=stress_tape,
                            episode_seed=int(seed),
                            episode_index=episode_index,
                            condition=condition,
                            arm=arm,
                            estimator_config=estimator_config,
                            evaluator_config=evaluator_config,
                            lock_config=lock_config,
                            planner_config=planner_config,
                        )
                        runner20._write_json_atomic(result_path, outcome.summary)
                        runner20._write_npz_atomic(trace_path, outcome.trace)
                    if outcome.summary["noise_tape_sha256"] != tape.content_sha256():
                        raise RuntimeError("V35 saved the wrong exogenous tape hash")
                    if outcome.summary["stress_tape_sha256"] != stress_tape.content_sha256():
                        raise RuntimeError("V35 saved the wrong stress tape hash")
                    if int(outcome.summary["action_count"]) != int(cfg.max_steps):
                        raise RuntimeError("V35 saved an incomplete trace")
                    condition_outcomes[arm] = outcome
                    rows.append(_flatten(outcome.summary))
                    completed += 1
                    if _loaded_local_sources(root) != expected_sources:
                        raise RuntimeError("V35 source closure changed during execution")
                    elapsed = float(time.perf_counter() - started)
                    eta = elapsed / max(completed, 1) * max(total - completed, 0)
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
                            "last_condition": condition,
                            "last_arm": arm,
                            "resumable": True,
                        },
                    )
                    if completed % int(settings["progress_every"]) == 0:
                        print(
                            f"[{runner20._utc_now()}] V35 {completed}/{total}; "
                            f"elapsed={elapsed:.1f}s eta={eta:.1f}s; "
                            f"seed={seed} condition={condition} arm={arm}",
                            flush=True,
                        )
                if condition in v35.BIAS_SWITCH_CONDITIONS:
                    pairing_records.append(
                        {
                            "episode_index": episode_index,
                            "episode_seed": int(seed),
                            "condition": condition,
                            **_verify_treatment_pair(
                                condition_outcomes[v35.PRIMARY_ARM],
                                condition_outcomes[v35.BIAS_SWITCH_ARM],
                            ),
                        }
                    )
        elapsed = float(time.perf_counter() - started)
        runner20._write_csv_atomic(output / "episode_arm_summary.csv", rows)
        summary = _aggregate(rows, contract, elapsed, pairing_records)
        runner20._write_json_atomic(output / "campaign_summary.json", summary)
        runner20._write_json_atomic(
            output / "decision.json",
            {
                "decision": summary["decision"],
                "nominal_decision": summary["nominal_decision"],
                "profile_switch_decision": summary["profile_switch_decision"],
                "integrity_valid": summary["integrity_valid"],
                "development_only": True,
                "reserved_final_range_untouched": summary["reserved_final_range_untouched"],
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
        elapsed = float(time.perf_counter() - started)
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
