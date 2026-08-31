#!/usr/bin/env python3
"""Run the paired V23-reference versus V24-audited gate campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

import run_v20_positioning_ablation as runner20
import run_v22_active_acquisition as runner22
import uuv_v19_observability as v19
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24


RUNNER_VERSION = "v24_audited_gate_runner_1.0"
DEFAULT_SEED_START = 49_500
DEFAULT_EPISODES = 100
SMOKE_SEED_START = 49_480
ARM_NAMES = ("v23_early_reference", "v24_audited_gate")
PROTOCOL_NAME = "EXPERIMENT_PROTOCOL_V24_AUDITED_GATE.md"


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--coarse-candidates", type=int)
    parser.add_argument("--coarse-sweeps", type=int)
    parser.add_argument("--local-starts", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=2)
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    smoke = bool(args.smoke)
    episodes = int(
        args.episodes if args.episodes is not None else (2 if smoke else DEFAULT_EPISODES)
    )
    seed_start = int(
        args.seed_start
        if args.seed_start is not None
        else (SMOKE_SEED_START if smoke else DEFAULT_SEED_START)
    )
    output = (
        args.output_dir
        if args.output_dir is not None
        else root
        / (
            "experiments_v24_audited_gate_smoke"
            if smoke
            else "experiments_v24_audited_gate_dev100"
        )
    ).expanduser().resolve()
    settings = {
        "root": root,
        "output": output,
        "metadata": runner20._default_metadata(root),
        "seed_start": seed_start,
        "episode_start": int(args.episode_start),
        "episodes": episodes,
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
        "smoke": smoke,
        "resume": bool(args.resume),
        "progress_every": max(1, int(args.progress_every)),
    }
    if episodes < 1 or settings["episode_start"] < 0:
        raise ValueError("episode count must be positive and start non-negative")
    first = seed_start + int(settings["episode_start"])
    last = first + episodes - 1
    for seed in (first, last):
        v19.assert_seed_is_not_sealed_final(seed)
    if first <= v19.SEALED_FINAL_SEED_END and last >= v19.SEALED_FINAL_SEED_START:
        raise PermissionError("V24 intersects the sealed final range")
    if not Path(settings["metadata"]).is_file():
        raise FileNotFoundError(settings["metadata"])
    return settings


def _loaded_local_sources(root: Path) -> Dict[str, str]:
    root = root.resolve()
    paths = {root / PROTOCOL_NAME}
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
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source-manifest files: {missing}")
    manifest = {
        str(path.resolve().relative_to(root)): runner20._sha256(path.resolve())
        for path in paths
    }
    required = {
        "run_v24_audited_gate.py",
        "uuv_v24_audited_gate.py",
        PROTOCOL_NAME,
        "run_v22_active_acquisition.py",
        "uuv_v22_active_acquisition.py",
        "run_v20_positioning_ablation.py",
        "uuv_v21_causal_lock.py",
        "uuv_v19_observability.py",
        "uuv_v18_resampling_guard.py",
    }
    absent = sorted(required - set(manifest))
    if absent:
        raise RuntimeError(f"source manifest missed required dependencies: {absent}")
    return dict(sorted(manifest.items()))


def _package_versions() -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for distribution in ("numba", "gymnasium"):
        try:
            values[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            values[distribution] = None
    return values


def _contract(
    settings: Mapping[str, Any],
    cfg: Any,
    estimator_config: v19.BatchEstimatorConfig,
    planner_config: v22.ActivePlannerConfig,
    lock_config: v24.AuditedLockConfig,
) -> Dict[str, Any]:
    root = Path(settings["root"])
    manifest = _loaded_local_sources(root)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v24.VERSION,
        "created_at_utc": runner20._utc_now(),
        "purpose": "paired development audit of the V23 early release gate",
        "seed_start": int(settings["seed_start"]),
        "episode_start": int(settings["episode_start"]),
        "episodes": int(settings["episodes"]),
        "arms": list(ARM_NAMES),
        "smoke": bool(settings["smoke"]),
        "estimator_config": v19.config_to_dict(estimator_config),
        "planner_config": dict(planner_config.__dict__),
        "reference_lock_config": dict(lock_config.base.__dict__),
        "audited_lock_config": lock_config.to_dict(),
        "environment_metadata_path": str(settings["metadata"]),
        "environment_metadata_sha256": runner20._sha256(Path(settings["metadata"])),
        "fixed_horizon_actions": int(cfg.max_steps),
        "source_sha256": manifest,
        "source_manifest_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "package_versions": _package_versions(),
        "sealed_final_seed_range_untouched": [
            v19.SEALED_FINAL_SEED_START,
            v19.SEALED_FINAL_SEED_END,
        ],
    }


def _immutable(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    return result


def _prepare(output: Path, contract: Mapping[str, Any], resume: bool) -> None:
    path = output / "control" / "campaign_contract.json"
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if _immutable(previous) != _immutable(contract):
            raise RuntimeError("resume contract differs from the V24 contract")
    else:
        runner20._write_json_atomic(path, contract)


def _flatten(summary: Mapping[str, Any]) -> Dict[str, Any]:
    row = runner22._flatten(summary)
    exact = summary.get("gate", {}).get("exact", {})
    for name, value in exact.items():
        row[f"exact_{name}"] = value
    row["gate_audit_release_violation_count"] = summary.get("gate", {}).get(
        "audit_release_violation_count"
    )
    return row


def _validate_outcome_identity(
    outcome: v22.ArmOutcome,
    *,
    arm: str,
    episode_index: int,
    seed: int,
    tape_sha256: str,
    expected_actions: int,
) -> None:
    summary = outcome.summary
    expected = {
        "version": v24.VERSION,
        "arm": arm,
        "episode_index": int(episode_index),
        "episode_seed": int(seed),
        "noise_tape_sha256": str(tape_sha256),
        "action_count": int(expected_actions),
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise RuntimeError(
                f"artifact identity mismatch in {name}: "
                f"expected {value!r}, got {summary.get(name)!r}"
            )
    required_trace_fields = {
        "time_s",
        "phase_track",
        "gate_locked_after_update",
        "localization_error_m",
    }
    missing = required_trace_fields - set(outcome.trace)
    if missing:
        raise RuntimeError(f"artifact trace is missing fields: {sorted(missing)}")
    for name in required_trace_fields:
        if np.asarray(outcome.trace[name]).shape != (int(expected_actions),):
            raise RuntimeError(f"artifact trace has wrong shape in {name}")


def _verify_pairing(outcomes: Mapping[str, v22.ArmOutcome], cfg: Any) -> Dict[str, Any]:
    if set(outcomes) != set(ARM_NAMES):
        raise RuntimeError("pair verification requires both prespecified arms")
    reference = outcomes[ARM_NAMES[0]]
    audited = outcomes[ARM_NAMES[1]]
    expected_cursor = {
        "substep_index": int(cfg.max_steps * round(cfg.action_dt / cfg.sub_dt)),
        "measurement_index": int(
            round(cfg.max_steps * cfg.action_dt / cfg.s_meas_period)
        ),
    }
    for arm, outcome in outcomes.items():
        if outcome.summary["noise_cursor"] != expected_cursor:
            raise RuntimeError(f"{arm} consumed the wrong amount of noise")
        if outcome.summary["noise_tape_sha256"] != reference.summary["noise_tape_sha256"]:
            raise RuntimeError("paired arms used different exogenous tapes")
        for field in ("initial_truth_m", "initial_leader_centroid_m"):
            if not np.allclose(
                np.asarray(outcome.summary[field], dtype=np.float64),
                np.asarray(reference.summary[field], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"paired arms differ in {field}")
    ref_phase = np.asarray(reference.trace["phase_track"], dtype=bool)
    audit_phase = np.asarray(audited.trace["phase_track"], dtype=bool)
    if ref_phase.shape != audit_phase.shape:
        raise RuntimeError("paired traces have different horizons")
    divergence = np.flatnonzero(ref_phase != audit_phase)
    intervention_index = int(divergence[0]) if divergence.size else int(ref_phase.size)
    fields = (
        "phase_track",
        "action_speed",
        "action_yaw",
        "action_pitch",
        "truth_x",
        "truth_y",
        "truth_z",
        "estimate_x",
        "estimate_y",
        "estimate_z",
        "localization_error_m",
    )
    for field in fields:
        left = np.asarray(reference.trace[field])[:intervention_index]
        right = np.asarray(audited.trace[field])[:intervention_index]
        if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
            raise RuntimeError(f"pre-treatment paired mismatch in {field}")
    return {
        "pre_treatment_match": True,
        "first_phase_divergence_index": (
            None if intervention_index == int(ref_phase.size) else intervention_index
        ),
        "first_phase_divergence_time_s": (
            None
            if intervention_index == int(ref_phase.size)
            else float(reference.trace["action_start_time_s"][intervention_index])
        ),
    }


def _percentiles(values: Sequence[float]) -> Optional[Dict[str, float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    if count < 1:
        raise ValueError("arm metrics require at least one episode")
    transition_times = [
        float(row["exact_first_transition_time_s"])
        for row in rows
        if row.get("exact_first_transition_time_s") is not None
    ]
    transition_errors = [
        float(row["exact_first_transition_error_m"])
        for row in rows
        if row.get("exact_first_transition_error_m") is not None
    ]
    max_transition_errors = [
        float(row["exact_maximum_transition_error_m"])
        for row in rows
        if row.get("exact_maximum_transition_error_m") is not None
    ]
    max_start_errors = [
        float(row["exact_maximum_locked_action_start_error_m"])
        for row in rows
        if row.get("exact_maximum_locked_action_start_error_m") is not None
    ]
    max_end_errors = [
        float(row["exact_maximum_locked_action_end_error_m"])
        for row in rows
        if row.get("exact_maximum_locked_action_end_error_m") is not None
    ]
    terminal_count = int(sum(bool(row["terminal_joint_success"]) for row in rows))
    lock_count = int(sum(int(row["exact_transition_count"]) > 0 for row in rows))
    runtime = [
        float(row.get("batch_maximum_update_runtime_s", 0.0))
        + float(row.get("planner_maximum_runtime_s", 0.0))
        for row in rows
    ]
    audit_values = [
        row.get("gate_audit_release_violation_count") for row in rows
    ]
    return {
        "episodes": count,
        "ever_locked_count": lock_count,
        "ever_locked_rate": float(lock_count / count),
        "terminal_joint_success_count": terminal_count,
        "terminal_joint_success_rate": float(terminal_count / count),
        "dwell15_joint_success_rate": float(
            np.mean([bool(row["dwell15_joint_success"]) for row in rows])
        ),
        "tail80_joint_success_rate": float(
            np.mean([bool(row["tail80_joint_success"]) for row in rows])
        ),
        "first_transition_time_s": _percentiles(transition_times),
        "first_transition_error_m": _percentiles(transition_errors),
        "transition_count": int(sum(int(row["exact_transition_count"]) for row in rows)),
        "unsafe_transition_count": int(
            sum(int(row["exact_false_transition_count"]) for row in rows)
        ),
        "unsafe_track_start_count": int(
            sum(int(row["exact_false_locked_action_start_count"]) for row in rows)
        ),
        "unsafe_track_end_count": int(
            sum(int(row["exact_false_locked_action_end_count"]) for row in rows)
        ),
        "maximum_transition_error_m": max(max_transition_errors, default=None),
        "maximum_track_start_error_m": max(max_start_errors, default=None),
        "maximum_track_end_error_m": max(max_end_errors, default=None),
        "maximum_combined_decision_runtime_s": float(max(runtime, default=0.0)),
        "mean_squared_action": float(
            np.mean([float(row["mean_squared_action"]) for row in rows])
        ),
        "audit_release_violation_count": (
            None
            if all(value is None for value in audit_values)
            else int(sum(int(value or 0) for value in audit_values))
        ),
    }


def _paired_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = {
        (int(row["episode_seed"]), str(row["arm"])): row for row in rows
    }
    seeds = sorted({key[0] for key in indexed})
    delays: List[float] = []
    faster = tied = later = missing_reference = missing_audited = 0
    terminal = {
        "both_success": 0,
        "reference_only": 0,
        "audited_only": 0,
        "neither_success": 0,
    }
    for seed in seeds:
        reference = indexed[(seed, ARM_NAMES[0])]
        audited = indexed[(seed, ARM_NAMES[1])]
        ref_time = reference.get("exact_first_transition_time_s")
        audit_time = audited.get("exact_first_transition_time_s")
        if ref_time is None:
            missing_reference += 1
        if audit_time is None:
            missing_audited += 1
        if ref_time is not None and audit_time is not None:
            delay = float(audit_time) - float(ref_time)
            delays.append(delay)
            if delay < -1e-12:
                faster += 1
            elif delay > 1e-12:
                later += 1
            else:
                tied += 1
        ref_success = bool(reference["terminal_joint_success"])
        audit_success = bool(audited["terminal_joint_success"])
        if ref_success and audit_success:
            terminal["both_success"] += 1
        elif ref_success:
            terminal["reference_only"] += 1
        elif audit_success:
            terminal["audited_only"] += 1
        else:
            terminal["neither_success"] += 1
    return {
        "episodes": len(seeds),
        "v24_minus_v23_transition_delay_s": _percentiles(delays),
        "audited_faster_count": faster,
        "tied_count": tied,
        "audited_later_count": later,
        "missing_reference_lock_count": missing_reference,
        "missing_audited_lock_count": missing_audited,
        "terminal_table": terminal,
    }


def _decision(
    by_arm: Mapping[str, Mapping[str, Any]],
    paired: Mapping[str, Any],
    *,
    complete_design: bool,
    smoke: bool,
    pairing_mismatch_count: int,
) -> tuple[str, Dict[str, bool]]:
    if smoke:
        return "SMOKE_ONLY", {}
    if not complete_design:
        return "INCOMPLETE", {}
    reference = by_arm[ARM_NAMES[0]]
    audited = by_arm[ARM_NAMES[1]]
    delays = paired["v24_minus_v23_transition_delay_s"]
    checks = {
        "ever_lock_rate": float(audited["ever_locked_rate"]) >= 0.95,
        "terminal_rate": float(audited["terminal_joint_success_rate"]) >= 0.95,
        "zero_unsafe_transitions": int(audited["unsafe_transition_count"]) == 0,
        "zero_unsafe_track_starts": int(audited["unsafe_track_start_count"]) == 0,
        "zero_unsafe_track_ends": int(audited["unsafe_track_end_count"]) == 0,
        "terminal_noninferiority": float(audited["terminal_joint_success_rate"])
        >= float(reference["terminal_joint_success_rate"]) - 0.02,
        "runtime": float(audited["maximum_combined_decision_runtime_s"]) < 2.0,
        "paired_median_delay": delays is not None and float(delays["p50"]) <= 30.0,
        "all_releases_audited": int(audited["audit_release_violation_count"]) == 0,
        "pre_treatment_pairing": int(pairing_mismatch_count) == 0,
    }
    return (
        "SUPPORT_V24_AUDITED_GATE"
        if all(checks.values())
        else "NO_SUPPORT_V24_AUDITED_GATE",
        checks,
    )


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    elapsed_s: float,
    pairing_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_arm = {
        arm: _arm_metrics([row for row in rows if row["arm"] == arm])
        for arm in ARM_NAMES
    }
    paired = _paired_metrics(rows)
    complete_design = bool(
        not contract["smoke"]
        and int(contract["seed_start"]) == DEFAULT_SEED_START
        and int(contract["episode_start"]) == 0
        and int(contract["episodes"]) == DEFAULT_EPISODES
        and all(
            metrics["episodes"] == DEFAULT_EPISODES for metrics in by_arm.values()
        )
    )
    mismatch_count = int(
        sum(not bool(item.get("pre_treatment_match")) for item in pairing_records)
    )
    decision, checks = _decision(
        by_arm,
        paired,
        complete_design=complete_design,
        smoke=bool(contract["smoke"]),
        pairing_mismatch_count=mismatch_count,
    )
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v24.VERSION,
        "completed_at_utc": runner20._utc_now(),
        "status": "complete",
        "decision": decision,
        "decision_checks": checks,
        "elapsed_wall_s": float(elapsed_s),
        "development_only": True,
        "no_rl_training": True,
        "episodes_per_arm": int(contract["episodes"]),
        "by_arm": by_arm,
        "paired_effect": paired,
        "pairing": {
            "pre_treatment_mismatch_count": mismatch_count,
            "records": list(pairing_records),
        },
        "sealed_final_range_untouched": contract[
            "sealed_final_seed_range_untouched"
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    settings = _settings(args)
    root = Path(settings["root"])
    output = Path(settings["output"])
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
    contract = _contract(
        settings, cfg, estimator_config, planner_config, lock_config
    )
    _prepare(output, contract, bool(settings["resume"]))
    expected_sources = dict(contract["source_sha256"])
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    pairing_records: List[Dict[str, Any]] = []
    completed = 0
    total = int(settings["episodes"]) * len(ARM_NAMES)
    try:
        for local_index in range(int(settings["episodes"])):
            episode_index = int(settings["episode_start"]) + local_index
            seed = int(settings["seed_start"]) + episode_index
            tape = runner20._tape_for_episode(output, cfg, seed, episode_index)
            outcomes: Dict[str, v22.ArmOutcome] = {}
            for arm in ARM_NAMES:
                result_path = (
                    output
                    / "episode_results"
                    / f"episode_{episode_index:04d}_seed_{seed}_{arm}.json"
                )
                trace_path = (
                    output
                    / "traces_npz"
                    / arm
                    / f"episode_{episode_index:04d}_seed_{seed}.npz"
                )
                if settings["resume"] and result_path.is_file() and trace_path.is_file():
                    summary = json.loads(result_path.read_text(encoding="utf-8"))
                    with np.load(trace_path, allow_pickle=False) as archive:
                        trace = {key: archive[key].copy() for key in archive.files}
                    outcome = v22.ArmOutcome(summary=summary, trace=trace)
                elif arm == ARM_NAMES[0]:
                    outcome = v24.augment_reference_outcome(
                        v22.run_active_acquisition_arm(
                            cfg=cfg,
                            tape=tape,
                            episode_seed=seed,
                            episode_index=episode_index,
                            estimator_config=estimator_config,
                            lock_config=lock_config.base,
                            planner_config=planner_config,
                        )
                    )
                else:
                    outcome = v24.run_audited_active_acquisition_arm(
                        cfg=cfg,
                        tape=tape,
                        episode_seed=seed,
                        episode_index=episode_index,
                        estimator_config=estimator_config,
                        lock_config=lock_config,
                        planner_config=planner_config,
                    )
                _validate_outcome_identity(
                    outcome,
                    arm=arm,
                    episode_index=episode_index,
                    seed=seed,
                    tape_sha256=tape.content_sha256(),
                    expected_actions=int(cfg.max_steps),
                )
                if not (
                    settings["resume"]
                    and result_path.is_file()
                    and trace_path.is_file()
                ):
                    runner20._write_json_atomic(result_path, outcome.summary)
                    runner20._write_npz_atomic(trace_path, outcome.trace)
                outcomes[arm] = outcome
                rows.append(_flatten(outcome.summary))
                completed += 1
                current_sources = _loaded_local_sources(root)
                if current_sources != expected_sources:
                    raise RuntimeError("local source closure changed after campaign start")
                elapsed = float(time.perf_counter() - started)
                eta = elapsed / max(completed, 1) * max(total - completed, 0)
                if completed % int(settings["progress_every"]) == 0:
                    print(
                        f"[{runner20._utc_now()}] completed {completed}/{total} arms; "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
                runner20._write_json_atomic(
                    output / "control" / "progress.json",
                    {
                        "status": "running",
                        "updated_at_utc": runner20._utc_now(),
                        "completed_arms": completed,
                        "total_arms": total,
                        "elapsed_wall_s": elapsed,
                        "estimated_remaining_s": eta,
                        "last_episode_index": episode_index,
                        "last_seed": seed,
                        "last_arm": arm,
                    },
                )
            pairing = _verify_pairing(outcomes, cfg)
            pairing_records.append(
                {"episode_index": episode_index, "episode_seed": seed, **pairing}
            )
        elapsed = float(time.perf_counter() - started)
        runner20._write_csv_atomic(output / "episode_arm_summary.csv", rows)
        summary = _aggregate(rows, contract, elapsed, pairing_records)
        runner20._write_json_atomic(output / "campaign_summary.json", summary)
        runner20._write_json_atomic(
            output / "decision.json",
            {
                "decision": summary["decision"],
                "decision_checks": summary["decision_checks"],
                "development_only": True,
                "sealed_final_range_untouched": contract[
                    "sealed_final_seed_range_untouched"
                ],
            },
        )
        runner20._write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "complete",
                "updated_at_utc": runner20._utc_now(),
                "completed_arms": completed,
                "total_arms": total,
                "elapsed_wall_s": elapsed,
                "estimated_remaining_s": 0.0,
            },
        )
        print(json.dumps(runner20._json_safe(summary), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        elapsed = float(time.perf_counter() - started)
        runner20._write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "failed",
                "updated_at_utc": runner20._utc_now(),
                "completed_arms": completed,
                "total_arms": total,
                "elapsed_wall_s": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
