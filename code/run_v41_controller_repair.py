#!/usr/bin/env python3
"""Run the frozen paired V41 post-TRACK controller-repair campaign."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

import run_v20_positioning_ablation as runner20
import uuv_v19_observability as v19
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24
import uuv_v38_leader_source_ablation as v38
import uuv_v40_dynamic_plant_stress as v40
import uuv_v41_controller_repair as v41


RUNNER_VERSION = "v41_controller_repair_runner_1.1"
PROTOCOL_NAME = "protocols/EXPERIMENT_PROTOCOL_V41_CONTROLLER_REPAIR.md"
DEFAULT_EPISODES = 100
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 41_051_000
LOCAL_SOURCE_NAMES = (
    "delay_aware_formation_tracker.py",
    "uuv_v41_controller_repair.py",
    "run_v41_controller_repair.py",
)
INHERITED_SOURCE_NAMES = (
    "uuv_v40_dynamic_plant_stress.py",
    "uuv_v38_leader_source_ablation.py",
    "uuv_v24_audited_gate.py",
    "uuv_v22_active_acquisition.py",
    "uuv_v21_causal_lock.py",
    "uuv_v20_positioning_ablation.py",
    "uuv_v19_observability.py",
    "uuv_v18_resampling_guard.py",
    "baseline_controllers_v11.py",
    "uuv_v11_online.py",
    "uuv_v11_rng.py",
)
SATURATION_THRESHOLD = 0.98
LOCALIZATION_NONINFERIORITY_MARGIN_M = 0.50
ACCEPTANCE = {
    "repaired_terminal_success_rate_min": 0.90,
    "repaired_tail80_success_rate_min": 0.85,
    "paired_terminal_success_gain_min": 0.20,
    "paired_tail80_success_gain_min": 0.40,
    "unsafe_track_count_max": 0,
    "terminal_localization_median_and_p95_margin_m": (
        LOCALIZATION_NONINFERIORITY_MARGIN_M
    ),
    "post_track_saturation_fraction_treatment_minus_reference_max": 0.0,
    "post_track_action_curvature_treatment_minus_reference_max": 0.0,
    "minimum_fraction_of_pairs_with_track_metrics": 0.90,
}


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
    return parser.parse_args(argv)


def _ordered_arms() -> Tuple[v41.ArmSpec, ...]:
    """Pair baseline and repaired controllers within each current condition."""

    lookup = {
        (str(arm.controller), str(arm.current)): arm for arm in v41.ARM_SPECS
    }
    ordered: List[v41.ArmSpec] = []
    for current in v40.CURRENTS:
        for controller in (v41.BASELINE_PID, v41.DELAY_AWARE):
            key = (controller, current)
            if key not in lookup:
                raise RuntimeError(f"V41 arm contract lacks {key}")
            ordered.append(lookup[key])
    if len(ordered) != 4 or len({arm.name for arm in ordered}) != 4:
        raise RuntimeError("V41 must define exactly four unique controller arms")
    return tuple(ordered)


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    source_root = Path(os.environ.get("UUV_ART2_SOURCE_ROOT", root)).expanduser().resolve()
    smoke = bool(args.smoke)
    maximum = len(v41.SMOKE_SEEDS) if smoke else DEFAULT_EPISODES
    episodes = maximum if args.episodes is None else int(args.episodes)
    start = int(args.episode_start)
    if episodes < 1 or start < 0 or start + episodes > maximum:
        raise ValueError("invalid V41 episode slice")
    if smoke:
        seeds = list(v41.SMOKE_SEEDS[start : start + episodes])
    else:
        seeds = list(
            range(
                int(v41.QUALIFICATION_START) + start,
                int(v41.QUALIFICATION_START) + start + episodes,
            )
        )
    for seed in seeds:
        v41.assert_seed_allowed(int(seed), smoke=smoke)
    if any(int(v41.FINAL_START) <= seed <= int(v41.FINAL_END) for seed in seeds):
        raise PermissionError("V41 campaign intersects the sealed final range")
    output = (
        args.output_dir
        if args.output_dir is not None
        else source_root
        / (
            "experiments_v41p1_controller_repair_smoke"
            if smoke
            else "experiments_v41p1_controller_repair_qualification100"
        )
    ).expanduser().resolve()
    coarse_candidates = int(4096 if args.coarse_candidates is None else args.coarse_candidates)
    coarse_sweeps = int(2 if args.coarse_sweeps is None else args.coarse_sweeps)
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
        "arms": _ordered_arms(),
        "coarse_candidates": coarse_candidates,
        "coarse_sweeps": coarse_sweeps,
        "local_starts": local_starts,
        "publication_settings": bool(
            coarse_candidates == 4096 and coarse_sweeps == 2 and local_starts == 48
        ),
    }


def _sha256(path: Path) -> str:
    return runner20._sha256(path)


def _source_paths(root: Path, source_root: Path) -> Dict[str, Path]:
    values: Dict[str, Path] = {
        PROTOCOL_NAME: root.parent / PROTOCOL_NAME,
    }
    for name in LOCAL_SOURCE_NAMES:
        values[name] = root / name
    for name in INHERITED_SOURCE_NAMES:
        values[name] = source_root / name
    missing = [str(path) for path in values.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V41 source closure: {missing}")
    return values


def _source_hashes(root: Path, source_root: Path) -> Dict[str, str]:
    return {
        name: _sha256(path)
        for name, path in sorted(_source_paths(root, source_root).items())
    }


def _package_versions() -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for name in ("gymnasium", "numba", "scipy", "tqdm"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _seed_reference_in_json(node: Any, selected: set[int]) -> set[int]:
    found: set[int] = set()
    if isinstance(node, Mapping):
        for key, child in node.items():
            if key in {"episode_seed", "seed"}:
                try:
                    value = int(child)
                except (TypeError, ValueError):
                    pass
                else:
                    if value in selected:
                        found.add(value)
            found.update(_seed_reference_in_json(child, selected))
    elif isinstance(node, list):
        for child in node:
            found.update(_seed_reference_in_json(child, selected))
    return found


def _fresh_seed_audit(
    source_root: Path,
    output: Path,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Reject qualification seeds already present in any other campaign artifact."""

    selected = {int(seed) for seed in seeds}
    token = re.compile(
        r"(?<!\d)(?:" + "|".join(str(seed) for seed in sorted(selected)) + r")(?!\d)"
    )
    findings: List[Dict[str, Any]] = []
    # In the public repository ``source_root`` is the ``code`` directory, while
    # campaign evidence also lives in top-level ``results`` and ``data``.  Audit
    # the repository as a whole so a copied or renamed artifact cannot make a
    # previously opened seed look fresh.
    audit_root = source_root.parent if source_root.name == "code" else source_root
    for path in sorted(audit_root.rglob("*")):
        if not path.is_file() or "source_snapshot" in path.parts or ".git" in path.parts:
            continue
        try:
            path.resolve().relative_to(output.resolve())
        except (OSError, ValueError):
            pass
        else:
            continue
        matched: set[int] = {int(value) for value in token.findall(path.name)}
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = None
            if value is not None:
                matched.update(_seed_reference_in_json(value, selected))
        elif path.suffix == ".csv":
            try:
                with path.open(
                    newline="",
                    encoding="utf-8",
                    errors="ignore",
                ) as handle:
                    reader = csv.DictReader(handle)
                    seed_fields = [
                        field
                        for field in ("episode_seed", "seed")
                        if field in (reader.fieldnames or ())
                    ]
                    for row in reader:
                        for field in seed_fields:
                            try:
                                episode_seed = int(row[field])
                            except (KeyError, TypeError, ValueError):
                                continue
                            if episode_seed in selected:
                                matched.add(episode_seed)
            except OSError:
                pass
        elif path.suffix == ".npz":
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if "episode_seed" in archive.files:
                        for value in np.asarray(archive["episode_seed"]).reshape(-1):
                            episode_seed = int(value)
                            if episode_seed in selected:
                                matched.add(episode_seed)
            except (OSError, ValueError, TypeError):
                pass
        if matched:
            try:
                relative = path.relative_to(audit_root)
            except ValueError:
                relative = path
            findings.append(
                {
                    "path": str(relative),
                    "seeds": sorted(matched),
                }
            )
            if len(findings) >= 100:
                break
    return {
        "selected_seed_range": [min(selected), max(selected)],
        "selected_seed_count": len(selected),
        "fresh": not findings,
        "finding_count": len(findings),
        "findings": findings,
    }


def _contract(
    settings: Mapping[str, Any],
    cfg: Any,
    estimator: Any,
    freshness: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    sources = _source_hashes(Path(settings["root"]), Path(settings["source_root"]))
    canonical = json.dumps(sources, sort_keys=True, separators=(",", ":"))
    full_qualification = bool(
        not settings["smoke"]
        and int(settings["episode_start"]) == 0
        and int(settings["episodes"]) == DEFAULT_EPISODES
        and list(settings["seeds"])
        == list(range(int(v41.QUALIFICATION_START), int(v41.QUALIFICATION_END) + 1))
    )
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "experiment_version": v41.VERSION,
        "created_at_utc": runner20._utc_now(),
        "purpose": "frozen paired repair of post-TRACK control under delay and low-order dynamics",
        "smoke": bool(settings["smoke"]),
        "full_qualification": full_qualification,
        "episodes": int(settings["episodes"]),
        "episode_start": int(settings["episode_start"]),
        "seeds": list(settings["seeds"]),
        "qualification_seed_range": [int(v41.QUALIFICATION_START), int(v41.QUALIFICATION_END)],
        "arms_in_execution_order": [arm.to_dict() for arm in settings["arms"]],
        "expected_runs": int(settings["episodes"]) * len(settings["arms"]),
        "publication_settings": bool(settings["publication_settings"]),
        "estimator_config": v19.config_to_dict(estimator),
        "controller_condition_contract": v41.condition_contract(),
        "fixed_horizon_actions": int(cfg.max_steps),
        "action_interval_s": float(cfg.action_dt),
        "plant_interval_s": float(cfg.sub_dt),
        "environment_metadata_path": str(settings["metadata"]),
        "environment_metadata_sha256": _sha256(Path(settings["metadata"])),
        "fresh_qualification_audit": freshness,
        "sealed_final_range": [int(v41.FINAL_START), int(v41.FINAL_END)],
        "source_sha256": sources,
        "source_manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "package_versions": _package_versions(),
        "acceptance_criteria": dict(ACCEPTANCE),
        "method_freeze": {
            "same_active_acquisition": True,
            "same_full_history_estimator": True,
            "same_audited_gate": True,
            "same_low_order_dynamic_plant": True,
            "same_exogenous_tapes_within_seed": True,
            "only_post_track_controller_changes": True,
        },
        "retuning_allowed_after_qualification_start": False,
    }


def _immutable(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    # Compare the exact JSON representation written by ``_write_json_atomic``.
    # This normalizes dataclass tuples (for example the three-channel slew
    # limit) to JSON lists, so an unchanged campaign can actually be resumed.
    return json.loads(
        json.dumps(
            runner20._json_safe(result),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _has_campaign_data(output: Path) -> bool:
    ignored = {
        Path("control/campaign.lock"),
        Path("control/runner.pid"),
    }
    if not output.exists():
        return False
    for path in output.rglob("*"):
        if path.is_file() and path.relative_to(output) not in ignored:
            return True
    return False


def _prepare(
    output: Path,
    contract: Mapping[str, Any],
    resume: bool,
    root: Path,
    source_root: Path,
) -> None:
    contract_path = output / "control" / "campaign_contract.json"
    if _has_campaign_data(output) and not resume:
        raise FileExistsError(f"output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if _immutable(previous) != _immutable(contract):
            raise RuntimeError("V41 resume contract differs from frozen contract")
        return
    runner20._write_json_atomic(contract_path, contract)
    snapshot = output / "control" / "source_snapshot"
    paths = _source_paths(root, source_root)
    for name, expected_sha in contract["source_sha256"].items():
        source = paths[name]
        if _sha256(source) != expected_sha:
            raise RuntimeError(f"source changed before snapshot: {name}")
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


@contextmanager
def _campaign_lock(output: Path) -> Iterator[None]:
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    lock_path = control / "campaign.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            pid_path = control / "runner.pid"
            owner = pid_path.read_text(encoding="utf-8").strip() if pid_path.is_file() else "unknown"
            raise RuntimeError(f"V41 campaign is already running (pid={owner})") from exc
        pid_path = control / "runner.pid"
        temporary = pid_path.with_name(pid_path.name + ".tmp")
        temporary.write_text(f"{os.getpid()}\n", encoding="utf-8")
        os.replace(temporary, pid_path)
        yield
    finally:
        pid_path = control / "runner.pid"
        try:
            if pid_path.is_file() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_path.unlink()
        except OSError:
            pass
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _log(output: Path, event: str, **values: Any) -> None:
    record = {"time_utc": runner20._utc_now(), "event": event, **values}
    path = output / "control" / "campaign.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(runner20._json_safe(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _paths(output: Path, index: int, seed: int, arm_name: str) -> Tuple[Path, Path]:
    return (
        output / "episode_results" / arm_name / f"episode_{index:04d}_seed_{seed}.json",
        output / "traces_npz" / arm_name / f"episode_{index:04d}_seed_{seed}.npz",
    )


def _load_outcome(result_path: Path, trace_path: Path) -> v38.ArmOutcome:
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in archive.files}
    return v38.ArmOutcome(summary=summary, trace=trace)


def _validate_outcome(
    outcome: v38.ArmOutcome,
    *,
    arm: v41.ArmSpec,
    tape: Any,
    cfg: Any,
    seed: int,
    episode_index: int,
) -> None:
    summary = outcome.summary
    expected = {
        "arm": arm.name,
        "controller_name": arm.controller,
        "current_name": arm.current,
        "episode_seed": int(seed),
        "episode_index": int(episode_index),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"V41 outcome mismatch for {key}: {summary.get(key)!r} != {value!r}")
    if summary.get("noise_tape_sha256") != tape.content_sha256():
        raise RuntimeError("V41 saved the wrong sensor-noise tape")
    if int(summary.get("action_count", -1)) != int(cfg.max_steps):
        raise RuntimeError("V41 saved an incomplete episode")
    if len(str(summary.get("current_tape_sha256", ""))) != 64:
        raise RuntimeError("V41 outcome lacks a valid current-tape hash")
    required = (
        "time_s",
        "phase_track",
        "action_speed",
        "action_yaw",
        "action_pitch",
        "localization_error_m",
        "formation_error_truth_m",
        "plant_requested_action",
        "plant_delivered_action",
    )
    for key in required:
        if key not in outcome.trace:
            raise RuntimeError(f"V41 trace lacks {key}")
        if np.asarray(outcome.trace[key]).shape[0] != int(cfg.max_steps):
            raise RuntimeError(f"V41 trace {key} has an invalid horizon")


def _first_track_index(trace: Mapping[str, np.ndarray]) -> Optional[int]:
    indices = np.flatnonzero(np.asarray(trace["phase_track"], dtype=bool))
    return None if indices.size == 0 else int(indices[0])


def _post_track_action_metrics(trace: Mapping[str, np.ndarray]) -> Dict[str, Optional[float]]:
    action = np.column_stack(
        [trace["action_speed"], trace["action_yaw"], trace["action_pitch"]]
    ).astype(np.float64)
    track = np.asarray(trace["phase_track"], dtype=bool)
    if not np.any(track):
        return {
            "post_track_action_count": 0,
            "post_track_saturation_fraction": None,
            "post_track_speed_saturation_fraction": None,
            "post_track_yaw_saturation_fraction": None,
            "post_track_pitch_saturation_fraction": None,
            "post_track_action_total_variation": None,
            "post_track_action_curvature_rms": None,
        }
    selected = np.abs(action[track]) >= SATURATION_THRESHOLD
    consecutive2 = track[1:] & track[:-1]
    difference = np.diff(action, axis=0)
    total_variation = (
        None
        if not np.any(consecutive2)
        else float(np.mean(np.linalg.norm(difference[consecutive2], axis=1)))
    )
    consecutive3 = track[2:] & track[1:-1] & track[:-2]
    curvature = action[2:] - 2.0 * action[1:-1] + action[:-2]
    curvature_rms = (
        None
        if not np.any(consecutive3)
        else float(np.sqrt(np.mean(np.sum(np.square(curvature[consecutive3]), axis=1))))
    )
    return {
        "post_track_action_count": int(np.sum(track)),
        "post_track_saturation_fraction": float(np.mean(np.any(selected, axis=1))),
        "post_track_speed_saturation_fraction": float(np.mean(selected[:, 0])),
        "post_track_yaw_saturation_fraction": float(np.mean(selected[:, 1])),
        "post_track_pitch_saturation_fraction": float(np.mean(selected[:, 2])),
        "post_track_action_total_variation": total_variation,
        "post_track_action_curvature_rms": curvature_rms,
    }


def _flatten(outcome: v38.ArmOutcome) -> Dict[str, Any]:
    summary = outcome.summary
    exact = summary["gate"]["exact"]
    values: Dict[str, Any] = {
        "episode_index": int(summary["episode_index"]),
        "episode_seed": int(summary["episode_seed"]),
        "arm": str(summary["arm"]),
        "controller": str(summary["controller_name"]),
        "current": str(summary["current_name"]),
        "terminal_joint_success": bool(summary["terminal_joint_success"]),
        "tail80_joint_success": bool(summary["tail80_joint_success"]),
        "dwell15_joint_success": bool(summary["dwell15_joint_success"]),
        "tail50_joint_occupancy": float(summary["tail50_joint_occupancy"]),
        "terminal_localization_error_m": float(summary["terminal_localization_error_m"]),
        "terminal_formation_error_m": float(summary["terminal_formation_error_m"]),
        "ever_locked": bool(summary["gate"]["ever_locked"]),
        "first_track_time_s": summary["gate"].get("first_track_action_time_s"),
        "unsafe_transition_count": int(exact["false_transition_count"]),
        "unsafe_track_start_count": int(exact["false_locked_action_start_count"]),
        "unsafe_track_end_count": int(exact["false_locked_action_end_count"]),
        "audit_release_violation_count": int(summary["gate"]["audit_release_violation_count"]),
        "mean_squared_action": float(summary["mean_squared_action"]),
        "maximum_combined_decision_runtime_s": float(summary["maximum_combined_decision_runtime_s"]),
        "requested_delivered_action_rms": float(summary["plant"]["requested_delivered_action_rms"]),
        "action_count": int(summary["action_count"]),
        "noise_tape_sha256": str(summary["noise_tape_sha256"]),
        "current_tape_sha256": str(summary["current_tape_sha256"]),
    }
    values.update(_post_track_action_metrics(outcome.trace))
    return values


def _equal_array(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))


def _pretrack_pairing(
    baseline: v38.ArmOutcome,
    repaired: v38.ArmOutcome,
    current: str,
) -> Dict[str, Any]:
    for key in ("episode_seed", "episode_index", "noise_tape_sha256", "current_tape_sha256"):
        if baseline.summary[key] != repaired.summary[key]:
            raise RuntimeError(f"V41 paired arms differ in {key}")
    if not np.array_equal(
        np.asarray(baseline.summary["initial_truth_m"]),
        np.asarray(repaired.summary["initial_truth_m"]),
    ):
        raise RuntimeError("V41 paired arms differ in initial truth")
    left_index = _first_track_index(baseline.trace)
    right_index = _first_track_index(repaired.trace)
    first = min(
        value for value in (left_index, right_index, int(baseline.summary["action_count"]))
        if value is not None
    )
    fields = (
        "phase_track", "gate_locked_after_update",
        "action_speed", "action_yaw", "action_pitch",
        "truth_x", "truth_y", "truth_z",
        "estimate_x", "estimate_y", "estimate_z",
        "formation_error_truth_m", "localization_error_m", "batch_local_radius95_m",
        "planner_utility", "planner_worst_radius_before_m", "planner_worst_radius_after_m",
        "planner_minimum_pair_chi2", "planner_hypothesis_count",
        "gate_release_predicate", "gate_release_pass_streak",
        "gate_hold_predicate", "gate_hold_failure_streak",
        "audit_release_checks_pass", "audit_hold_checks_pass",
        "current_dead_reckoning_x", "current_dead_reckoning_y", "current_dead_reckoning_z",
        "plant_requested_action", "plant_delivered_action", "plant_executed_rate_state",
        "water_current_mps", "body_velocity_through_water_mps", "ground_velocity_mps",
    )
    compared: List[str] = []
    unequal: List[str] = []
    for key in fields:
        if key not in baseline.trace or key not in repaired.trace:
            continue
        compared.append(key)
        if not _equal_array(baseline.trace[key][:first], repaired.trace[key][:first]):
            unequal.append(key)
    record = {
        "episode_seed": int(baseline.summary["episode_seed"]),
        "current": current,
        "first_track_action_index_baseline": left_index,
        "first_track_action_index_repaired": right_index,
        "common_pretrack_action_count": first,
        "compared_trace_fields": compared,
        "unequal_pretrack_trace_fields": unequal,
        "first_track_start_equal": left_index == right_index,
        "pretrack_bitwise_equal": not unequal and left_index == right_index,
    }
    if not record["pretrack_bitwise_equal"]:
        raise RuntimeError(f"V41 controller arms differ before TRACK: {record}")
    return record


def _reference_equivalence(
    candidate: v38.ArmOutcome,
    reference: v38.ArmOutcome,
) -> Dict[str, Any]:
    excluded = {"planner_runtime_s"}
    common = sorted(set(candidate.trace) & set(reference.trace) - excluded)
    unequal = [
        key for key in common
        if not _equal_array(candidate.trace[key], reference.trace[key])
    ]
    return {
        "episode_seed": int(candidate.summary["episode_seed"]),
        "current": str(candidate.summary["current_name"]),
        "compared_trace_fields": len(common),
        "excluded_nondeterministic_trace_fields": sorted(excluded),
        "unequal_trace_fields": unequal,
        "bitwise_equal": not unequal,
    }


def _distribution(values: Iterable[Optional[float]]) -> Optional[Dict[str, float]]:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if not array.size:
        return None
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    result: Dict[str, Any] = {
        "episodes": count,
        "terminal_success_count": sum(bool(row["terminal_joint_success"]) for row in rows),
        "tail80_success_count": sum(bool(row["tail80_joint_success"]) for row in rows),
        "dwell15_success_count": sum(bool(row["dwell15_joint_success"]) for row in rows),
        "ever_lock_count": sum(bool(row["ever_locked"]) for row in rows),
        "unsafe_transition_count": sum(int(row["unsafe_transition_count"]) for row in rows),
        "unsafe_track_start_count": sum(int(row["unsafe_track_start_count"]) for row in rows),
        "unsafe_track_end_count": sum(int(row["unsafe_track_end_count"]) for row in rows),
        "audit_release_violation_count": sum(int(row["audit_release_violation_count"]) for row in rows),
        "decision_deadline_miss_episode_count": sum(
            float(row["maximum_combined_decision_runtime_s"]) >= 2.0 for row in rows
        ),
    }
    result["terminal_success_rate"] = None if not count else result["terminal_success_count"] / count
    result["tail80_success_rate"] = None if not count else result["tail80_success_count"] / count
    result["ever_lock_rate"] = None if not count else result["ever_lock_count"] / count
    for field in (
        "first_track_time_s", "terminal_localization_error_m", "terminal_formation_error_m",
        "tail50_joint_occupancy", "mean_squared_action", "requested_delivered_action_rms",
        "post_track_saturation_fraction", "post_track_speed_saturation_fraction",
        "post_track_yaw_saturation_fraction", "post_track_pitch_saturation_fraction",
        "post_track_action_total_variation", "post_track_action_curvature_rms",
        "maximum_combined_decision_runtime_s",
    ):
        result[field] = _distribution(row[field] for row in rows)
    return result


def _bootstrap_mean(values: Sequence[float], seed: int) -> List[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        end = min(start + 1000, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, array.size, size=(end - start, array.size))
        draws[start:end] = np.mean(array[indices], axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _mcnemar_exact(reference_only: int, treatment_only: int) -> Optional[float]:
    total = int(reference_only) + int(treatment_only)
    if total == 0:
        return None
    lower = min(int(reference_only), int(treatment_only))
    probability = sum(math.comb(total, k) for k in range(lower + 1)) / (2.0 ** total)
    return float(min(1.0, 2.0 * probability))


def _paired_binary(
    reference: Mapping[int, Mapping[str, Any]],
    treatment: Mapping[int, Mapping[str, Any]],
    field: str,
) -> Dict[str, Any]:
    reference_only = treatment_only = 0
    differences: List[float] = []
    for seed in sorted(reference):
        left = bool(reference[seed][field])
        right = bool(treatment[seed][field])
        reference_only += int(left and not right)
        treatment_only += int(right and not left)
        differences.append(float(right) - float(left))
    return {
        "pairs": len(differences),
        "reference_only": reference_only,
        "treatment_only": treatment_only,
        "risk_difference": float(np.mean(differences)),
        "mcnemar_exact_two_sided_p": _mcnemar_exact(reference_only, treatment_only),
    }


def _paired_numeric(
    reference: Mapping[int, Mapping[str, Any]],
    treatment: Mapping[int, Mapping[str, Any]],
    field: str,
    seed: int,
) -> Dict[str, Any]:
    differences: List[float] = []
    for episode_seed in sorted(reference):
        left = reference[episode_seed][field]
        right = treatment[episode_seed][field]
        if left is None or right is None:
            continue
        left_value, right_value = float(left), float(right)
        if math.isfinite(left_value) and math.isfinite(right_value):
            differences.append(right_value - left_value)
    if not differences:
        return {"pairs": 0, "treatment_minus_reference": None}
    return {
        "pairs": len(differences),
        "treatment_minus_reference": _distribution(differences),
        "bootstrap_mean_95": _bootstrap_mean(differences, seed),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    current: str,
    seed_offset: int,
) -> Dict[str, Any]:
    reference = {
        int(row["episode_seed"]): row
        for row in rows
        if row["current"] == current and row["controller"] == v41.BASELINE_PID
    }
    treatment = {
        int(row["episode_seed"]): row
        for row in rows
        if row["current"] == current and row["controller"] == v41.DELAY_AWARE
    }
    if set(reference) != set(treatment):
        raise RuntimeError("V41 paired contrast has unequal seed support")
    result: Dict[str, Any] = {
        "reference": v41.BASELINE_PID,
        "treatment": v41.DELAY_AWARE,
        "current": current,
        "pairs": len(reference),
    }
    for field in ("terminal_joint_success", "tail80_joint_success", "dwell15_joint_success"):
        result[field] = _paired_binary(reference, treatment, field)
    numeric = (
        "terminal_localization_error_m", "terminal_formation_error_m",
        "tail50_joint_occupancy", "mean_squared_action",
        "post_track_saturation_fraction", "post_track_action_total_variation",
        "post_track_action_curvature_rms",
    )
    for index, field in enumerate(numeric):
        result[field] = _paired_numeric(
            reference, treatment, field, BOOTSTRAP_SEED + seed_offset * 100 + index
        )
    return result


def _mean_difference(contrast: Mapping[str, Any], field: str) -> Optional[float]:
    value = contrast[field].get("treatment_minus_reference")
    return None if value is None else float(value["mean"])


def _acceptance_screens(
    by_arm: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    episodes: int,
) -> Dict[str, Any]:
    screens: Dict[str, Any] = {}
    for current in v40.CURRENTS:
        baseline_name = v41.ArmSpec(v41.BASELINE_PID, current).name
        repaired_name = v41.ArmSpec(v41.DELAY_AWARE, current).name
        baseline = by_arm[baseline_name]
        repaired = by_arm[repaired_name]
        contrast = contrasts[current]
        localization_baseline = baseline["terminal_localization_error_m"]
        localization_repaired = repaired["terminal_localization_error_m"]
        localization_ok = bool(
            localization_baseline is not None
            and localization_repaired is not None
            and localization_repaired["median"]
            <= localization_baseline["median"] + LOCALIZATION_NONINFERIORITY_MARGIN_M
            and localization_repaired["p95"]
            <= localization_baseline["p95"] + LOCALIZATION_NONINFERIORITY_MARGIN_M
        )
        saturation_difference = _mean_difference(contrast, "post_track_saturation_fraction")
        curvature_difference = _mean_difference(contrast, "post_track_action_curvature_rms")
        metric_pairs = min(
            int(contrast["post_track_saturation_fraction"]["pairs"]),
            int(contrast["post_track_action_curvature_rms"]["pairs"]),
        )
        checks = {
            "repaired_terminal_success": float(repaired["terminal_success_rate"])
            >= ACCEPTANCE["repaired_terminal_success_rate_min"],
            "repaired_tail80_success": float(repaired["tail80_success_rate"])
            >= ACCEPTANCE["repaired_tail80_success_rate_min"],
            "paired_terminal_gain": float(
                contrast["terminal_joint_success"]["risk_difference"]
            ) >= ACCEPTANCE["paired_terminal_success_gain_min"],
            "paired_tail80_gain": float(
                contrast["tail80_joint_success"]["risk_difference"]
            ) >= ACCEPTANCE["paired_tail80_success_gain_min"],
            "zero_unsafe_track": (
                int(baseline["unsafe_transition_count"])
                + int(baseline["unsafe_track_start_count"])
                + int(baseline["unsafe_track_end_count"])
                + int(repaired["unsafe_transition_count"])
                + int(repaired["unsafe_track_start_count"])
                + int(repaired["unsafe_track_end_count"])
            ) == 0,
            "localization_noninferior": localization_ok,
            "track_metric_pair_coverage": metric_pairs
            >= math.ceil(episodes * ACCEPTANCE["minimum_fraction_of_pairs_with_track_metrics"]),
            "post_track_saturation_reduced": (
                saturation_difference is not None and saturation_difference < 0.0
            ),
            "post_track_action_curvature_reduced": (
                curvature_difference is not None and curvature_difference < 0.0
            ),
        }
        screens[current] = {
            "pass": all(checks.values()),
            "checks": checks,
            "observed": {
                "repaired_terminal_success_rate": repaired["terminal_success_rate"],
                "repaired_tail80_success_rate": repaired["tail80_success_rate"],
                "paired_terminal_success_gain": contrast["terminal_joint_success"]["risk_difference"],
                "paired_tail80_success_gain": contrast["tail80_joint_success"]["risk_difference"],
                "localization_median_baseline_m": localization_baseline["median"],
                "localization_median_repaired_m": localization_repaired["median"],
                "localization_p95_baseline_m": localization_baseline["p95"],
                "localization_p95_repaired_m": localization_repaired["p95"],
                "saturation_fraction_paired_mean_difference": saturation_difference,
                "action_curvature_paired_mean_difference": curvature_difference,
                "valid_track_metric_pairs": metric_pairs,
            },
        }
    return screens


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    elapsed_s: float,
    pairings: Sequence[Mapping[str, Any]],
    equivalence: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    arms = [item["name"] for item in contract["arms_in_execution_order"]]
    by_arm = {name: _metrics([row for row in rows if row["arm"] == name]) for name in arms}
    contrasts = {
        current: _paired_contrast(rows, current, offset)
        for offset, current in enumerate(v40.CURRENTS)
    }
    expected = int(contract["expected_runs"])
    smoke = bool(contract["smoke"])
    integrity = {
        "all_runs_complete": len(rows) == expected,
        "all_fixed_horizon": all(
            int(row["action_count"]) == int(contract["fixed_horizon_actions"]) for row in rows
        ),
        "publication_estimator_settings": bool(
            smoke or contract["publication_settings"]
        ),
        "pairing_complete": len(pairings) == 2 * int(contract["episodes"]),
        "pretrack_equivalence": all(record["pretrack_bitwise_equal"] for record in pairings),
        "baseline_reference_equivalence": bool(
            (not smoke)
            or (
                len(equivalence) == 2 * int(contract["episodes"])
                and all(record["bitwise_equal"] for record in equivalence)
            )
        ),
        "audit_release_violations_zero": sum(
            int(row["audit_release_violation_count"]) for row in rows
        ) == 0,
        "qualification_was_fresh": bool(
            smoke or contract["fresh_qualification_audit"]["fresh"]
        ),
        "final_holdout_sealed": contract["sealed_final_range"]
        == [int(v41.FINAL_START), int(v41.FINAL_END)],
    }
    integrity_valid = all(integrity.values())
    screens = (
        None
        if smoke or not contract["full_qualification"]
        else _acceptance_screens(by_arm, contrasts, int(contract["episodes"]))
    )
    acceptance_pass = bool(
        screens is not None and all(screen["pass"] for screen in screens.values())
    )
    if smoke:
        decision = "SMOKE_PASS" if integrity_valid else "SMOKE_FAIL"
    elif not contract["full_qualification"]:
        decision = "PARTIAL_QUALIFICATION_COMPLETE" if integrity_valid else "PARTIAL_QUALIFICATION_INVALID"
    else:
        decision = (
            "CONTROLLER_REPAIR_ACCEPT"
            if integrity_valid and acceptance_pass
            else "CONTROLLER_REPAIR_REJECT_OR_INVALID"
        )
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "analysis_version": "v41_controller_repair_analysis_1.0",
        "status": "complete",
        "smoke": smoke,
        "full_qualification": bool(contract["full_qualification"]),
        "elapsed_wall_s": float(elapsed_s),
        "row_count": len(rows),
        "expected_row_count": expected,
        "integrity_checks": integrity,
        "integrity_valid": integrity_valid,
        "decision": decision,
        "acceptance_pass": acceptance_pass,
        "acceptance_criteria": dict(ACCEPTANCE),
        "acceptance_screens_by_current": screens,
        "by_arm": by_arm,
        "paired_delay_aware_minus_baseline_by_current": contrasts,
        "pairing_records": list(pairings),
        "v40_baseline_reference_equivalence_records": list(equivalence),
        "final_holdout_sealed": [int(v41.FINAL_START), int(v41.FINAL_END)],
        "retuning_allowed": False,
    }


def _duration(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "?"
    seconds = max(0, int(round(value)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def _progress(
    output: Path,
    *,
    status: str,
    completed: int,
    total: int,
    elapsed_s: float,
    eta_s: Optional[float],
    seed: Optional[int] = None,
    arm: Optional[str] = None,
    error: Optional[str] = None,
    traceback_text: Optional[str] = None,
) -> None:
    expected_finish = (
        None
        if eta_s is None
        else (datetime.now(timezone.utc) + timedelta(seconds=float(eta_s))).isoformat()
    )
    value: Dict[str, Any] = {
        "status": status,
        "updated_at_utc": runner20._utc_now(),
        "pid": os.getpid(),
        "completed_runs": int(completed),
        "total_runs": int(total),
        "elapsed_wall_s": float(elapsed_s),
        "estimated_remaining_s": eta_s,
        "estimated_finish_utc": expected_finish,
        "last_seed": seed,
        "last_arm": arm,
        "resumable": True,
    }
    if error is not None:
        value["error"] = error
    if traceback_text is not None:
        value["traceback"] = traceback_text
    runner20._write_json_atomic(output / "control" / "progress.json", value)


def _run(settings: Mapping[str, Any]) -> int:
    root = Path(settings["root"])
    source_root = Path(settings["source_root"])
    output = Path(settings["output"])
    freshness = None if settings["smoke"] else _fresh_seed_audit(
        source_root, output, settings["seeds"]
    )
    if freshness is not None and not freshness["fresh"]:
        raise RuntimeError(f"V41 qualification seed audit failed: {freshness['findings'][:10]}")
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
    _prepare(output, contract, bool(settings["resume"]), root, source_root)
    frozen_sources = dict(contract["source_sha256"])
    rows: List[Dict[str, Any]] = []
    pairings: List[Dict[str, Any]] = []
    equivalence: List[Dict[str, Any]] = []
    completed = 0
    executed_durations: List[float] = []
    total = int(contract["expected_runs"])
    started = time.perf_counter()
    _log(
        output,
        "campaign_started",
        pid=os.getpid(),
        resume=bool(settings["resume"]),
        total_runs=total,
        seeds=[settings["seeds"][0], settings["seeds"][-1]],
    )
    _progress(output, status="running", completed=0, total=total, elapsed_s=0.0, eta_s=None)
    bar = tqdm(
        total=total,
        desc="V41 controller repair",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.08,
        leave=True,
    )
    try:
        for local_index, seed in enumerate(settings["seeds"]):
            episode_index = int(settings["episode_start"]) + local_index
            tape = runner20._tape_for_episode(output, cfg, int(seed), episode_index)
            outcomes: Dict[str, v38.ArmOutcome] = {}
            for arm in settings["arms"]:
                result_path, trace_path = _paths(output, episode_index, int(seed), arm.name)
                resumed = bool(
                    settings["resume"] and result_path.is_file() and trace_path.is_file()
                )
                if bool(result_path.is_file()) != bool(trace_path.is_file()):
                    raise RuntimeError(f"incomplete V41 arm artifact pair: {result_path}, {trace_path}")
                bar.set_postfix_str(f"seed={seed} arm={arm.controller}/{arm.current} running")
                if resumed:
                    outcome = _load_outcome(result_path, trace_path)
                else:
                    run_started = time.perf_counter()
                    outcome = v41.run_controller_arm(
                        arm=arm,
                        cfg=cfg,
                        tape=tape,
                        episode_seed=int(seed),
                        episode_index=episode_index,
                        estimator_config=estimator_config,
                        lock_config=lock_config,
                        planner_config=planner_config,
                    )
                    executed_durations.append(float(time.perf_counter() - run_started))
                    runner20._write_json_atomic(result_path, outcome.summary)
                    runner20._write_npz_atomic(trace_path, outcome.trace)
                _validate_outcome(
                    outcome,
                    arm=arm,
                    tape=tape,
                    cfg=cfg,
                    seed=int(seed),
                    episode_index=episode_index,
                )
                outcomes[arm.name] = outcome
                row = _flatten(outcome)
                rows.append(row)
                completed += 1
                if _source_hashes(root, source_root) != frozen_sources:
                    raise RuntimeError("V41 source closure changed during execution")
                pending = total - completed
                eta = None if not executed_durations else float(np.mean(executed_durations) * pending)
                elapsed = float(time.perf_counter() - started)
                _progress(
                    output,
                    status="running",
                    completed=completed,
                    total=total,
                    elapsed_s=elapsed,
                    eta_s=eta,
                    seed=int(seed),
                    arm=arm.name,
                )
                _log(
                    output,
                    "run_completed" if not resumed else "run_resumed",
                    completed_runs=completed,
                    total_runs=total,
                    seed=int(seed),
                    episode_index=episode_index,
                    arm=arm.name,
                    terminal_joint_success=row["terminal_joint_success"],
                    tail80_joint_success=row["tail80_joint_success"],
                    elapsed_wall_s=elapsed,
                    estimated_remaining_s=eta,
                )
                bar.update(1)
                bar.set_postfix_str(
                    f"seed={seed} arm={arm.controller}/{arm.current} ETA={_duration(eta)}"
                )

            for current in v40.CURRENTS:
                baseline = outcomes[v41.ArmSpec(v41.BASELINE_PID, current).name]
                repaired = outcomes[v41.ArmSpec(v41.DELAY_AWARE, current).name]
                pairing = _pretrack_pairing(baseline, repaired, current)
                pairings.append(pairing)
                _log(output, "pretrack_pair_validated", **pairing)

            if settings["smoke"]:
                for current in v40.CURRENTS:
                    bar.set_postfix_str(f"seed={seed} V40 baseline reference/{current}")
                    reference_arm = v40.ArmSpec(
                        v40.POLICY_ACTIVE,
                        v40.DYNAMICS_LOW_ORDER,
                        current,
                    )
                    reference = v40.run_factorial_arm(
                        arm=reference_arm,
                        cfg=cfg,
                        tape=tape,
                        episode_seed=int(seed),
                        episode_index=episode_index,
                        estimator_config=estimator_config,
                        lock_config=lock_config,
                        planner_config=planner_config,
                    )
                    candidate = outcomes[v41.ArmSpec(v41.BASELINE_PID, current).name]
                    record = _reference_equivalence(candidate, reference)
                    equivalence.append(record)
                    if not record["bitwise_equal"]:
                        raise RuntimeError(f"V41 baseline differs from V40: {record}")
                    _log(output, "v40_baseline_reference_validated", **record)

        elapsed = float(time.perf_counter() - started)
        runner20._write_csv_atomic(output / "episode_arm_summary.csv", rows)
        summary = _aggregate(rows, contract, elapsed, pairings, equivalence)
        runner20._write_json_atomic(output / "campaign_summary.json", summary)
        runner20._write_json_atomic(
            output / "decision.json",
            {
                "decision": summary["decision"],
                "integrity_valid": summary["integrity_valid"],
                "acceptance_pass": summary["acceptance_pass"],
                "acceptance_screens_by_current": summary["acceptance_screens_by_current"],
                "final_holdout_sealed": summary["final_holdout_sealed"],
                "retuning_allowed": False,
            },
        )
        _progress(
            output,
            status="complete",
            completed=completed,
            total=total,
            elapsed_s=elapsed,
            eta_s=0.0,
        )
        _log(output, "campaign_completed", decision=summary["decision"], elapsed_wall_s=elapsed)
        bar.close()
        print(
            f"V41 complete: decision={summary['decision']} elapsed={_duration(elapsed)} "
            f"output={output}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        elapsed = float(time.perf_counter() - started)
        trace_text = traceback.format_exc()
        error = f"{type(exc).__name__}: {exc}"
        _progress(
            output,
            status="failed",
            completed=completed,
            total=total,
            elapsed_s=elapsed,
            eta_s=None,
            error=error,
            traceback_text=trace_text,
        )
        crash = {
            "status": "failed",
            "failed_at_utc": runner20._utc_now(),
            "pid": os.getpid(),
            "completed_runs": completed,
            "total_runs": total,
            "error": error,
            "traceback": trace_text,
            "resume_command": f"{sys.executable} {Path(__file__).resolve()} --resume"
            + (" --smoke" if settings["smoke"] else ""),
        }
        runner20._write_json_atomic(output / "control" / "crash_state.json", crash)
        _log(output, "campaign_failed", **crash)
        bar.close()
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    settings = _settings(args)
    output = Path(settings["output"])
    with _campaign_lock(output):
        return _run(settings)


if __name__ == "__main__":
    raise SystemExit(main())
