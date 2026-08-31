#!/usr/bin/env python3
"""Run the paired 3x2 leader-source/acquisition-policy ablation."""

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
import uuv_v38_leader_source_ablation as v38


RUNNER_VERSION = "v38_leader_source_ablation_runner_1.1"
PROTOCOL_NAME = "EXPERIMENT_PROTOCOL_V38_LEADER_SOURCE_ABLATION.md"
DEFAULT_SEED_START = 48_400
DEFAULT_EPISODES = 100
SMOKE_SEEDS = (48_598, 48_599)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 38_200_100


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
    parser.add_argument("--progress-every", type=int, default=2)
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    smoke = bool(args.smoke)
    maximum = len(SMOKE_SEEDS) if smoke else DEFAULT_EPISODES
    episodes = int(args.episodes if args.episodes is not None else maximum)
    start = int(args.episode_start)
    if episodes < 1 or start < 0 or start + episodes > maximum:
        raise ValueError("invalid V38 episode slice")
    seeds = (
        list(SMOKE_SEEDS[start : start + episodes])
        if smoke
        else list(range(DEFAULT_SEED_START + start, DEFAULT_SEED_START + start + episodes))
    )
    for seed in seeds:
        v38.assert_seed_allowed(seed)
    output = (
        args.output_dir
        if args.output_dir is not None
        else root
        / (
            "experiments_v38_leader_source_ablation_smoke"
            if smoke
            else "experiments_v38_leader_source_ablation_dev100"
        )
    ).expanduser().resolve()
    # Smoke deliberately retains the publication settings.  Optional overrides
    # exist only for unit/debug runs and are recorded in the contract.
    settings = {
        "root": root,
        "output": output,
        "metadata": runner20._default_metadata(root),
        "smoke": smoke,
        "resume": bool(args.resume),
        "episode_start": start,
        "episodes": episodes,
        "seeds": seeds,
        "coarse_candidates": int(
            4096 if args.coarse_candidates is None else args.coarse_candidates
        ),
        "coarse_sweeps": int(
            2 if args.coarse_sweeps is None else args.coarse_sweeps
        ),
        "local_starts": int(
            48 if args.local_starts is None else args.local_starts
        ),
        "publication_settings": bool(
            (4096 if args.coarse_candidates is None else args.coarse_candidates)
            == 4096
            and (2 if args.coarse_sweeps is None else args.coarse_sweeps) == 2
            and (48 if args.local_starts is None else args.local_starts) == 48
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
    test_relative = Path("tests/test_uuv_v38_leader_source_ablation.py")
    test_path = root / test_relative
    if not test_path.is_file():
        test_path = root.parent / test_relative
    paths = {
        root / PROTOCOL_NAME,
        root / "audit_v38_leader_source_ablation.py",
        test_path,
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
        "uuv_v38_leader_source_ablation.py",
        "run_v38_leader_source_ablation.py",
        "audit_v38_leader_source_ablation.py",
        "tests/test_uuv_v38_leader_source_ablation.py",
        "uuv_v24_audited_gate.py",
        "uuv_v22_active_acquisition.py",
        "uuv_v21_causal_lock.py",
        "uuv_v20_positioning_ablation.py",
        "uuv_v19_observability.py",
    }
    manifest: Dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            name = str(path.relative_to(root))
        except ValueError:
            try:
                name = str(path.relative_to(root.parent))
            except ValueError as error:
                raise RuntimeError(
                    f"V38 source lies outside the public repository: {path}"
                ) from error
        manifest[name] = _sha256(path)
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"V38 source manifest missed {missing}")
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
    findings: List[str] = []
    seed_pattern = (
        f"{DEFAULT_SEED_START // 100:03d}"
        if DEFAULT_SEED_START % 100 == 0 and DEFAULT_EPISODES == 100
        else None
    )
    if seed_pattern is None:
        raise RuntimeError("fresh-seed audit expects one aligned 100-seed block")
    filename_pattern = re.compile(rf"seed_({seed_pattern}\d{{2}})(?:\D|$)")
    json_pattern = re.compile(
        rf'"episode_seed"\s*:\s*({seed_pattern}\d{{2}})'
    )
    for directory in sorted(root.glob("experiments_*")):
        if not directory.is_dir():
            continue
        try:
            if directory.resolve() == output.resolve():
                continue
        except OSError:
            continue
        for path in directory.rglob("*"):
            if (
                not path.is_file()
                or "source_snapshot" in path.parts
                or path.suffix not in {".json", ".csv", ".npz"}
            ):
                continue
            if filename_pattern.search(path.name):
                findings.append(str(path.relative_to(root)))
                continue
            if path.suffix == ".json":
                try:
                    if json_pattern.search(
                        path.read_text(encoding="utf-8", errors="ignore")
                    ):
                        findings.append(str(path.relative_to(root)))
                except OSError:
                    pass
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
    planner_config: v22.ActivePlannerConfig,
    lock_config: v24.AuditedLockConfig,
    freshness: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    manifest = _loaded_local_sources(Path(settings["root"]))
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v38.VERSION,
        "created_at_utc": runner20._utc_now(),
        "purpose": "publication-grade paired 3x2 Doppler-source ablation",
        "smoke": bool(settings["smoke"]),
        "publication_settings": bool(settings["publication_settings"]),
        "episode_start": int(settings["episode_start"]),
        "episodes": int(settings["episodes"]),
        "seeds": list(settings["seeds"]),
        "source_names": list(v38.SOURCE_NAMES),
        "source_masks": {
            name: list(mask) for name, mask in v38.SOURCE_MASKS.items()
        },
        "policies": list(v38.POLICIES),
        "arms": [
            {
                "source_name": source,
                "policy_name": policy,
                "arm": v38.arm_name(source, policy),
            }
            for source, policy in v38.arm_pairs()
        ],
        "expected_runs": int(settings["episodes"]) * len(v38.arm_pairs()),
        "estimator_config": v19.config_to_dict(estimator_config),
        "planner_config": dict(planner_config.__dict__),
        "audited_lock_config": lock_config.to_dict(),
        "mission_support": {
            "center_source": "shared reset-time centroid of both physical leaders",
            "radius_min_m": float(cfg.start_rho_min),
            "radius_max_m": float(cfg.start_rho_max),
            "identical_within_each_six-arm_pair": True,
        },
        "environment_metadata_path": str(settings["metadata"]),
        "environment_metadata_sha256": _sha256(Path(settings["metadata"])),
        "fixed_horizon_actions": int(cfg.max_steps),
        "fresh_seed_audit": freshness,
        "paired_contrasts": [
            "active_vs_fixed_within_each_source",
            "both_vs_leader1_within_each_policy",
            "both_vs_leader2_within_each_policy",
        ],
        "primary_endpoints": [
            "terminal_joint_success",
            "tail80_joint_success",
            "common_60s_localization_on_identical_fixed_trajectory",
            "terminal_localization_error_m",
            "terminal_formation_error_m",
            "unsafe_track_counts",
        ],
        "information_boundary": {
            "excluded_doppler_and_leader_state_removed_before_estimator": True,
            "excluded_leader_removed_from_planner_measurement_prediction": True,
            "common_two_leader_velocity_reference_for_all_s_turns": True,
            "common_two_leader_velocity_reference_for_candidate_anchor": True,
            "common_two_leader_centroid_prior_for_all_arms": True,
            "both_broadcasts_retained_for_formation_reference": True,
        },
        "reserved_final_range_untouched": [v38.RESERVED_START, v38.FINAL_END],
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
            raise RuntimeError("V38 resume contract differs from frozen contract")
    else:
        runner20._write_json_atomic(contract_path, contract)
        snapshot = output / "control" / "source_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parent
        for name in (
            PROTOCOL_NAME,
            "uuv_v38_leader_source_ablation.py",
            "run_v38_leader_source_ablation.py",
            "audit_v38_leader_source_ablation.py",
            "tests/test_uuv_v38_leader_source_ablation.py",
        ):
            source = root / name
            if source.is_file():
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())


def _paths(
    output: Path,
    episode_index: int,
    seed: int,
    source: str,
    policy: str,
) -> Tuple[Path, Path]:
    arm = v38.arm_name(source, policy)
    return (
        output
        / "episode_results"
        / f"episode_{episode_index:04d}_seed_{seed}_{arm}.json",
        output
        / "traces_npz"
        / arm
        / f"episode_{episode_index:04d}_seed_{seed}.npz",
    )


def _flatten(summary: Mapping[str, Any]) -> Dict[str, Any]:
    exact = summary["gate"]["exact"]
    return {
        "episode_index": int(summary["episode_index"]),
        "episode_seed": int(summary["episode_seed"]),
        "source_name": str(summary["source_name"]),
        "policy_name": str(summary["policy_name"]),
        "arm": str(summary["arm"]),
        "source_mask_l1": bool(summary["source_mask"][0]),
        "source_mask_l2": bool(summary["source_mask"][1]),
        "active_source_count": int(summary["active_source_count"]),
        "measurement_row_count": int(summary["measurement_row_count"]),
        "active_scalar_measurement_count": int(
            summary["active_scalar_measurement_count"]
        ),
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
        "checkpoint_60s_localization_error_m": float(
            summary["checkpoint_60s"]["localization_error_m"]
        ),
        "checkpoint_60s_localization_below_7m": bool(
            summary["checkpoint_60s"]["localization_below_7m"]
        ),
        "checkpoint_60s_phase_track": bool(
            summary["checkpoint_60s"]["phase_track"]
        ),
        "checkpoint_60s_gate_locked_after_update": bool(
            summary["checkpoint_60s"]["gate_locked_after_update"]
        ),
        "ever_locked": bool(summary["gate"]["ever_locked"]),
        "first_track_action_time_s": summary["gate"]["first_track_action_time_s"],
        "unsafe_transition_count": int(exact["false_transition_count"]),
        "unsafe_track_start_count": int(
            exact["false_locked_action_start_count"]
        ),
        "unsafe_track_end_count": int(exact["false_locked_action_end_count"]),
        "audit_release_violation_count": int(
            summary["gate"]["audit_release_violation_count"]
        ),
        "false_confidence_action_count": int(
            summary["robustness"]["false_confidence_action_count"]
        ),
        "maximum_localization_error_m": float(
            summary["robustness"]["maximum_localization_error_m"]
        ),
        "maximum_track_localization_error_m": (
            summary["robustness"]["maximum_track_localization_error_m"]
        ),
        "mean_squared_action": float(summary["mean_squared_action"]),
        "maximum_combined_decision_runtime_s": float(
            summary["maximum_combined_decision_runtime_s"]
        ),
        "planner_decision_count": int(summary["planner"]["decision_count"]),
        "noise_tape_sha256": str(summary["noise_tape_sha256"]),
        "masked_history_sha256": str(summary["masked_history_sha256"]),
    }


def _distribution(values: Iterable[float]) -> Optional[Dict[str, float]]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _wilson(successes: int, count: int) -> List[float]:
    n = int(count)
    if n < 1:
        return [float("nan"), float("nan")]
    p = float(successes) / n
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def _cell_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    terminal = int(sum(bool(row["terminal_joint_success"]) for row in rows))
    tail = int(sum(bool(row["tail80_joint_success"]) for row in rows))
    lock = int(sum(bool(row["ever_locked"]) for row in rows))
    checkpoint_success = int(
        sum(bool(row["checkpoint_60s_localization_below_7m"]) for row in rows)
    )
    track_max = [
        float(row["maximum_track_localization_error_m"])
        for row in rows
        if row["maximum_track_localization_error_m"] is not None
    ]
    lock_times = [
        float(row["first_track_action_time_s"])
        for row in rows
        if row["first_track_action_time_s"] is not None
    ]
    return {
        "episodes": count,
        "terminal_success_count": terminal,
        "terminal_success_rate": 0.0 if count == 0 else terminal / count,
        "terminal_success_wilson95": _wilson(terminal, count),
        "tail80_success_count": tail,
        "tail80_success_rate": 0.0 if count == 0 else tail / count,
        "tail80_success_wilson95": _wilson(tail, count),
        "dwell15_success_count": int(
            sum(bool(row["dwell15_joint_success"]) for row in rows)
        ),
        "ever_lock_count": lock,
        "ever_lock_rate": 0.0 if count == 0 else lock / count,
        "checkpoint_60s_localization_success_count": checkpoint_success,
        "checkpoint_60s_localization_success_rate": (
            0.0 if count == 0 else checkpoint_success / count
        ),
        "checkpoint_60s_localization_success_wilson95": _wilson(
            checkpoint_success,
            count,
        ),
        "checkpoint_60s_localization_error_m": _distribution(
            row["checkpoint_60s_localization_error_m"] for row in rows
        ),
        "unsafe_transition_count": int(
            sum(int(row["unsafe_transition_count"]) for row in rows)
        ),
        "unsafe_track_start_count": int(
            sum(int(row["unsafe_track_start_count"]) for row in rows)
        ),
        "unsafe_track_end_count": int(
            sum(int(row["unsafe_track_end_count"]) for row in rows)
        ),
        "audit_release_violation_count": int(
            sum(int(row["audit_release_violation_count"]) for row in rows)
        ),
        "false_confidence_action_count": int(
            sum(int(row["false_confidence_action_count"]) for row in rows)
        ),
        "terminal_localization_error_m": _distribution(
            row["terminal_localization_error_m"] for row in rows
        ),
        "terminal_formation_error_m": _distribution(
            row["terminal_formation_error_m"] for row in rows
        ),
        "maximum_localization_error_m": _distribution(
            row["maximum_localization_error_m"] for row in rows
        ),
        "maximum_track_localization_error_m": _distribution(track_max),
        "first_track_action_time_s": _distribution(lock_times),
        "tail50_joint_occupancy": _distribution(
            row["tail50_joint_occupancy"] for row in rows
        ),
        "mean_squared_action": _distribution(
            row["mean_squared_action"] for row in rows
        ),
        "maximum_combined_decision_runtime_s": max(
            (
                float(row["maximum_combined_decision_runtime_s"])
                for row in rows
            ),
            default=0.0,
        ),
    }


def _bootstrap_mean(values: Sequence[float], seed: int) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("paired bootstrap requires at least one pair")
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


def _discordance_p(left_only: int, right_only: int) -> float:
    n = int(left_only) + int(right_only)
    if n == 0:
        return 1.0
    lower = min(int(left_only), int(right_only))
    probability = sum(math.comb(n, k) for k in range(lower + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * probability))


def _paired_contrast(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    left_label: str,
    right_label: str,
) -> Dict[str, Any]:
    left_by_seed = {int(row["episode_seed"]): row for row in left}
    right_by_seed = {int(row["episode_seed"]): row for row in right}
    if set(left_by_seed) != set(right_by_seed):
        raise RuntimeError("paired contrast seed sets differ")
    ordered = sorted(left_by_seed)
    localization_improvement: List[float] = []
    formation_improvement: List[float] = []
    checkpoint_localization_improvement: List[float] = []
    terminal_left_only = terminal_right_only = 0
    tail_left_only = tail_right_only = 0
    checkpoint_left_only = checkpoint_right_only = 0
    for episode_seed in ordered:
        lrow = left_by_seed[episode_seed]
        rrow = right_by_seed[episode_seed]
        # Positive improvement means the right-hand method has lower error.
        localization_improvement.append(
            float(lrow["terminal_localization_error_m"])
            - float(rrow["terminal_localization_error_m"])
        )
        formation_improvement.append(
            float(lrow["terminal_formation_error_m"])
            - float(rrow["terminal_formation_error_m"])
        )
        checkpoint_localization_improvement.append(
            float(lrow["checkpoint_60s_localization_error_m"])
            - float(rrow["checkpoint_60s_localization_error_m"])
        )
        l_term = bool(lrow["terminal_joint_success"])
        r_term = bool(rrow["terminal_joint_success"])
        terminal_left_only += int(l_term and not r_term)
        terminal_right_only += int(r_term and not l_term)
        l_tail = bool(lrow["tail80_joint_success"])
        r_tail = bool(rrow["tail80_joint_success"])
        tail_left_only += int(l_tail and not r_tail)
        tail_right_only += int(r_tail and not l_tail)
        l_checkpoint = bool(lrow["checkpoint_60s_localization_below_7m"])
        r_checkpoint = bool(rrow["checkpoint_60s_localization_below_7m"])
        checkpoint_left_only += int(l_checkpoint and not r_checkpoint)
        checkpoint_right_only += int(r_checkpoint and not l_checkpoint)
    count = len(ordered)
    left_terminal = sum(
        bool(left_by_seed[value]["terminal_joint_success"]) for value in ordered
    )
    right_terminal = sum(
        bool(right_by_seed[value]["terminal_joint_success"]) for value in ordered
    )
    left_tail = sum(
        bool(left_by_seed[value]["tail80_joint_success"]) for value in ordered
    )
    right_tail = sum(
        bool(right_by_seed[value]["tail80_joint_success"]) for value in ordered
    )
    left_checkpoint = sum(
        bool(left_by_seed[value]["checkpoint_60s_localization_below_7m"])
        for value in ordered
    )
    right_checkpoint = sum(
        bool(right_by_seed[value]["checkpoint_60s_localization_below_7m"])
        for value in ordered
    )
    return {
        "left": left_label,
        "right": right_label,
        "pairs": count,
        "terminal_success_rate_difference_right_minus_left": (
            (right_terminal - left_terminal) / max(count, 1)
        ),
        "tail80_success_rate_difference_right_minus_left": (
            (right_tail - left_tail) / max(count, 1)
        ),
        "checkpoint_60s_localization_success_rate_difference_right_minus_left": (
            (right_checkpoint - left_checkpoint) / max(count, 1)
        ),
        "terminal_discordance": {
            "left_only": terminal_left_only,
            "right_only": terminal_right_only,
            "exact_mcnemar_p_two_sided": _discordance_p(
                terminal_left_only,
                terminal_right_only,
            ),
        },
        "tail80_discordance": {
            "left_only": tail_left_only,
            "right_only": tail_right_only,
            "exact_mcnemar_p_two_sided": _discordance_p(
                tail_left_only,
                tail_right_only,
            ),
        },
        "checkpoint_60s_localization_discordance": {
            "left_only": checkpoint_left_only,
            "right_only": checkpoint_right_only,
            "exact_mcnemar_p_two_sided": _discordance_p(
                checkpoint_left_only,
                checkpoint_right_only,
            ),
        },
        "checkpoint_60s_localization_improvement_m": _bootstrap_mean(
            checkpoint_localization_improvement,
            seed + 2,
        ),
        "terminal_localization_improvement_m": _bootstrap_mean(
            localization_improvement,
            seed,
        ),
        "terminal_formation_improvement_m": _bootstrap_mean(
            formation_improvement,
            seed + 1,
        ),
    }


def _verify_seed_pairing(
    outcomes: Mapping[Tuple[str, str], v38.ArmOutcome],
) -> Dict[str, Any]:
    if set(outcomes) != set(v38.arm_pairs()):
        raise RuntimeError("V38 seed pairing lacks one or more arms")
    summaries = [value.summary for value in outcomes.values()]
    tape_hashes = {str(value["noise_tape_sha256"]) for value in summaries}
    seeds = {int(value["episode_seed"]) for value in summaries}
    supports = {
        json.dumps(value["mission_support"], sort_keys=True)
        for value in summaries
    }
    if len(tape_hashes) != 1 or len(seeds) != 1 or len(supports) != 1:
        raise RuntimeError("V38 paired arms differ in tape, seed, or mission support")
    return {
        "episode_seed": next(iter(seeds)),
        "noise_tape_sha256": next(iter(tape_hashes)),
        "arm_count": len(outcomes),
        "common_mission_support": True,
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    elapsed_s: float,
    pairing: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_cell: Dict[str, Any] = {}
    indexed_rows: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for source, policy in v38.arm_pairs():
        cell = [
            row
            for row in rows
            if row["source_name"] == source and row["policy_name"] == policy
        ]
        indexed_rows[(source, policy)] = cell
        by_cell[v38.arm_name(source, policy)] = _cell_metrics(cell)
    expected = int(contract["expected_runs"])
    scalar_counts_valid = all(
        int(row["active_scalar_measurement_count"])
        == int(row["measurement_row_count"])
        * int(row["active_source_count"])
        for row in rows
    )
    masks_valid = all(
        (
            bool(row["source_mask_l1"]),
            bool(row["source_mask_l2"]),
        )
        == v38.SOURCE_MASKS[str(row["source_name"])]
        for row in rows
    )
    fixed_planner_zero = all(
        int(row["planner_decision_count"]) == 0
        for row in rows
        if row["policy_name"] == v38.POLICY_FIXED
    )
    fixed_checkpoint_pre_release = all(
        not bool(row["checkpoint_60s_phase_track"])
        and not bool(row["checkpoint_60s_gate_locked_after_update"])
        for row in rows
        if row["policy_name"] == v38.POLICY_FIXED
    )
    integrity_checks = {
        "all_runs_complete": len(rows) == expected,
        "all_six_arms_paired": len(pairing) == int(contract["episodes"]),
        "publication_settings": bool(contract["publication_settings"]),
        "active_scalar_denominators_valid": scalar_counts_valid,
        "source_masks_valid": masks_valid,
        "fixed_policy_never_called_belief_planner": fixed_planner_zero,
        "fixed_60s_checkpoint_precedes_release": fixed_checkpoint_pre_release,
        "audit_release_violations_zero": sum(
            int(row["audit_release_violation_count"]) for row in rows
        )
        == 0,
        "maximum_runtime_below_2s": max(
            (
                float(row["maximum_combined_decision_runtime_s"])
                for row in rows
            ),
            default=float("inf"),
        )
        < 2.0,
        "reserved_final_untouched": contract["reserved_final_range_untouched"]
        == [v38.RESERVED_START, v38.FINAL_END],
        "fresh_seed_audit_pass": bool(
            contract["smoke"] or contract["fresh_seed_audit"]["fresh"]
        ),
    }
    integrity_valid = bool(all(integrity_checks.values()))
    contrasts: Dict[str, Any] = {}
    contrast_index = 0
    for source in v38.SOURCE_NAMES:
        contrasts[f"active_vs_fixed@{source}"] = _paired_contrast(
            indexed_rows[(source, v38.POLICY_FIXED)],
            indexed_rows[(source, v38.POLICY_ACTIVE)],
            seed=BOOTSTRAP_SEED + 10 * contrast_index,
            left_label=v38.arm_name(source, v38.POLICY_FIXED),
            right_label=v38.arm_name(source, v38.POLICY_ACTIVE),
        )
        contrast_index += 1
    for policy in v38.POLICIES:
        for single in (v38.SOURCE_L1, v38.SOURCE_L2):
            contrasts[f"both_vs_{single}@{policy}"] = _paired_contrast(
                indexed_rows[(single, policy)],
                indexed_rows[(v38.SOURCE_BOTH, policy)],
                seed=BOOTSTRAP_SEED + 10 * contrast_index,
                left_label=v38.arm_name(single, policy),
                right_label=v38.arm_name(v38.SOURCE_BOTH, policy),
            )
            contrast_index += 1

    if bool(contract["smoke"]):
        claim_decision = "SMOKE_ONLY"
        decision = "SMOKE_PASS" if integrity_valid else "SMOKE_FAIL"
    else:
        both_active = by_cell[
            v38.arm_name(v38.SOURCE_BOTH, v38.POLICY_ACTIVE)
        ]
        active_vs_fixed = contrasts[
            f"active_vs_fixed@{v38.SOURCE_BOTH}"
        ]
        both_vs_l1 = contrasts[
            f"both_vs_{v38.SOURCE_L1}@{v38.POLICY_ACTIVE}"
        ]
        both_vs_l2 = contrasts[
            f"both_vs_{v38.SOURCE_L2}@{v38.POLICY_ACTIVE}"
        ]
        both_vs_l1_fixed = contrasts[
            f"both_vs_{v38.SOURCE_L1}@{v38.POLICY_FIXED}"
        ]
        both_vs_l2_fixed = contrasts[
            f"both_vs_{v38.SOURCE_L2}@{v38.POLICY_FIXED}"
        ]
        safety = bool(
            both_active["unsafe_transition_count"] == 0
            and both_active["unsafe_track_start_count"] == 0
            and both_active["unsafe_track_end_count"] == 0
        )
        operational = bool(
            both_active["terminal_success_rate"] >= 0.90
            and both_active["tail80_success_rate"] >= 0.85
            and both_active["terminal_localization_error_m"]["p95"] < 7.0
        )
        active_benefit = bool(
            active_vs_fixed[
                "terminal_success_rate_difference_right_minus_left"
            ]
            >= 0.05
            and active_vs_fixed[
                "terminal_localization_improvement_m"
            ]["mean"]
            > 0.0
        )
        two_source_benefit = bool(
            both_vs_l1[
                "terminal_success_rate_difference_right_minus_left"
            ]
            >= 0.05
            and both_vs_l2[
                "terminal_success_rate_difference_right_minus_left"
            ]
            >= 0.05
            and both_vs_l1[
                "terminal_localization_improvement_m"
            ]["mean"]
            > 0.0
            and both_vs_l2[
                "terminal_localization_improvement_m"
            ]["mean"]
            > 0.0
            and both_vs_l1_fixed[
                "checkpoint_60s_localization_success_rate_difference_right_minus_left"
            ]
            >= 0.05
            and both_vs_l2_fixed[
                "checkpoint_60s_localization_success_rate_difference_right_minus_left"
            ]
            >= 0.05
            and both_vs_l1_fixed[
                "checkpoint_60s_localization_improvement_m"
            ]["mean"]
            > 0.0
            and both_vs_l2_fixed[
                "checkpoint_60s_localization_improvement_m"
            ]["mean"]
            > 0.0
        )
        claim_decision = (
            "SUPPORT_TWO_DOPPLER_REFERENCE_ACTIVE_LOCALIZATION_CLAIM"
            if integrity_valid
            and safety
            and operational
            and active_benefit
            and two_source_benefit
            else "DO_NOT_SUPPORT_TWO_DOPPLER_REFERENCE_ACTIVE_LOCALIZATION_CLAIM"
        )
        decision = "V38_COMPLETE" if integrity_valid else "V38_INVALID"

    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "analysis_version": "v38_analysis_1.1",
        "status": "complete",
        "smoke": bool(contract["smoke"]),
        "publication_settings": bool(contract["publication_settings"]),
        "elapsed_wall_s": float(elapsed_s),
        "row_count": len(rows),
        "expected_row_count": expected,
        "integrity_checks": integrity_checks,
        "integrity_valid": integrity_valid,
        "decision": decision,
        "claim_decision": claim_decision,
        "by_arm": by_cell,
        "paired_contrasts": contrasts,
        "seed_pairing": list(pairing),
        "development_only": True,
        "reserved_final_range_untouched": contract[
            "reserved_final_range_untouched"
        ],
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
            raise RuntimeError(
                f"V38 fresh-seed audit failed: {freshness['findings'][:10]}"
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
    contract = _contract(
        settings,
        cfg,
        estimator_config,
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
            tape = runner20._tape_for_episode(
                output,
                cfg,
                int(seed),
                episode_index,
            )
            outcomes: Dict[Tuple[str, str], v38.ArmOutcome] = {}
            for source, policy in v38.arm_pairs():
                result_path, trace_path = _paths(
                    output,
                    episode_index,
                    int(seed),
                    source,
                    policy,
                )
                if (
                    settings["resume"]
                    and result_path.is_file()
                    and trace_path.is_file()
                ):
                    summary = json.loads(
                        result_path.read_text(encoding="utf-8")
                    )
                    with np.load(trace_path, allow_pickle=False) as archive:
                        trace = {
                            key: archive[key].copy() for key in archive.files
                        }
                    outcome = v38.ArmOutcome(summary=summary, trace=trace)
                else:
                    outcome = v38.run_arm(
                        cfg=cfg,
                        tape=tape,
                        episode_seed=int(seed),
                        episode_index=episode_index,
                        source_name=source,
                        policy_name=policy,
                        estimator_config=estimator_config,
                        lock_config=lock_config,
                        planner_config=planner_config,
                    )
                    runner20._write_json_atomic(result_path, outcome.summary)
                    runner20._write_npz_atomic(trace_path, outcome.trace)
                if outcome.summary["noise_tape_sha256"] != tape.content_sha256():
                    raise RuntimeError("V38 saved the wrong exogenous tape hash")
                if int(outcome.summary["action_count"]) != int(cfg.max_steps):
                    raise RuntimeError("V38 saved an incomplete trace")
                outcomes[(source, policy)] = outcome
                rows.append(_flatten(outcome.summary))
                completed += 1
                if _loaded_local_sources(root) != expected_sources:
                    raise RuntimeError("V38 source closure changed during execution")
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
                        "last_source": source,
                        "last_policy": policy,
                        "resumable": True,
                    },
                )
                if completed % int(settings["progress_every"]) == 0:
                    print(
                        f"[{runner20._utc_now()}] V38 {completed}/{total}; "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s; "
                        f"seed={seed} source={source} policy={policy}",
                        flush=True,
                    )
            pairing_records.append(_verify_seed_pairing(outcomes))
        elapsed = float(time.perf_counter() - started)
        runner20._write_csv_atomic(output / "episode_arm_summary.csv", rows)
        summary = _aggregate(rows, contract, elapsed, pairing_records)
        runner20._write_json_atomic(output / "campaign_summary.json", summary)
        runner20._write_json_atomic(
            output / "decision.json",
            {
                "decision": summary["decision"],
                "claim_decision": summary["claim_decision"],
                "integrity_valid": summary["integrity_valid"],
                "development_only": True,
                "reserved_final_range_untouched": summary[
                    "reserved_final_range_untouched"
                ],
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
