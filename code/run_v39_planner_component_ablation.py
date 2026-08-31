#!/usr/bin/env python3
"""Run the paired four-arm V39 planner-component ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import run_v20_positioning_ablation as runner20
import run_v38_leader_source_ablation as runner38
import uuv_v19_observability as v19
import uuv_v24_audited_gate as v24
import uuv_v39_planner_component_ablation as v39


RUNNER_VERSION = "v39_planner_component_ablation_runner_1.0"
PROTOCOL_NAME = "EXPERIMENT_PROTOCOL_V39_PLANNER_COMPONENT_ABLATION.md"
AUDITOR_NAME = "audit_v39_planner_component_ablation.py"
TEST_NAME = "tests/test_uuv_v39_planner_component_ablation.py"
DEFAULT_SEED_START = 48_800
DEFAULT_EPISODES = 100
SMOKE_SEEDS = (48_998, 48_999)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 39_200_100
RUNTIME_THRESHOLD_S = 2.0
MATERIAL_RATE_DIFFERENCE = 0.05


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
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    smoke = bool(args.smoke)
    maximum = len(SMOKE_SEEDS) if smoke else DEFAULT_EPISODES
    episodes = int(args.episodes if args.episodes is not None else maximum)
    start = int(args.episode_start)
    if episodes < 1 or start < 0 or start + episodes > maximum:
        raise ValueError("invalid V39 episode slice")
    seeds = (
        list(SMOKE_SEEDS[start : start + episodes])
        if smoke
        else list(
            range(
                DEFAULT_SEED_START + start,
                DEFAULT_SEED_START + start + episodes,
            )
        )
    )
    for seed in seeds:
        v39.assert_seed_allowed(seed)
    coarse_candidates = int(
        4096 if args.coarse_candidates is None else args.coarse_candidates
    )
    coarse_sweeps = int(
        2 if args.coarse_sweeps is None else args.coarse_sweeps
    )
    local_starts = int(
        48 if args.local_starts is None else args.local_starts
    )
    output = (
        args.output_dir
        if args.output_dir is not None
        else root
        / (
            "experiments_v39_planner_component_ablation_smoke"
            if smoke
            else "experiments_v39_planner_component_ablation_dev100"
        )
    ).expanduser().resolve()
    metadata = runner20._default_metadata(root)
    if not Path(metadata).is_file():
        raise FileNotFoundError(metadata)
    return {
        "root": root,
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


def _loaded_local_sources(root: Path) -> Dict[str, str]:
    root = root.resolve()
    paths = {
        root / PROTOCOL_NAME,
        root / AUDITOR_NAME,
        root / TEST_NAME,
        Path(__file__).resolve(),
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
        AUDITOR_NAME,
        TEST_NAME,
        "uuv_v39_planner_component_ablation.py",
        "run_v39_planner_component_ablation.py",
        "uuv_v38_leader_source_ablation.py",
        "uuv_v24_audited_gate.py",
        "uuv_v22_active_acquisition.py",
        "uuv_v21_causal_lock.py",
        "uuv_v20_positioning_ablation.py",
        "uuv_v19_observability.py",
    }
    manifest = {
        str(path.relative_to(root)): _sha256(path)
        for path in paths
        if path.is_file()
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"V39 source manifest missed {missing}")
    return dict(sorted(manifest.items()))


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


def _seed_references(path: Path, selected: set[int]) -> List[int]:
    found: set[int] = set()
    if path.suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return sorted(found)

        def visit(node: Any) -> None:
            if isinstance(node, Mapping):
                for key, child in node.items():
                    if key == "episode_seed":
                        try:
                            episode_seed = int(child)
                        except (TypeError, ValueError):
                            pass
                        else:
                            if episode_seed in selected:
                                found.add(episode_seed)
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(value)
    elif path.suffix == ".csv":
        try:
            with path.open(
                newline="",
                encoding="utf-8",
                errors="ignore",
            ) as handle:
                reader = csv.DictReader(handle)
                if "episode_seed" in (reader.fieldnames or ()):
                    for row in reader:
                        try:
                            value = int(row["episode_seed"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if value in selected:
                            found.add(value)
        except OSError:
            pass
    elif path.suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "episode_seed" in archive.files:
                    values = np.asarray(archive["episode_seed"]).reshape(-1)
                    for child in values:
                        episode_seed = int(child)
                        if episode_seed in selected:
                            found.add(episode_seed)
        except (OSError, ValueError, TypeError):
            pass
    return sorted(found)


def _fresh_seed_audit(
    root: Path,
    output: Path,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    selected = {int(seed) for seed in seeds}
    ordered = sorted(selected)
    findings: List[Dict[str, Any]] = []
    directories = []
    for directory in sorted(root.glob("experiments_*")):
        if not directory.is_dir():
            continue
        try:
            if directory.resolve() == output.resolve():
                continue
        except OSError:
            continue
        directories.append(directory)
    if selected and directories:
        alternatives = "|".join(str(value) for value in sorted(selected))
        pattern = (
            rf'"episode_seed"\s*:\s*(?:{alternatives})(?:[^0-9]|$)'
        )
        command = [
            "rg",
            "--files-with-matches",
            "--glob",
            "*.json",
            "--glob",
            "!**/source_snapshot/**",
            pattern,
            *(str(directory) for directory in directories),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            candidate_paths = [
                path
                for directory in directories
                for path in directory.rglob("*.json")
                if "source_snapshot" not in path.parts
            ]
        else:
            if completed.returncode not in (0, 1):
                raise RuntimeError(
                    f"exact episode_seed freshness search failed: "
                    f"{completed.stderr.strip()}"
                )
            candidate_paths = [
                Path(value)
                for value in completed.stdout.splitlines()
                if value.strip()
            ]
        for path in candidate_paths:
            if not path.is_absolute():
                path = root / path
            references = _seed_references(path, selected)
            if not references:
                continue
            findings.append(
                {
                    "path": str(path.relative_to(root)),
                    "seeds": references,
                }
            )
    consecutive = bool(
        ordered
        and ordered == list(range(ordered[0], ordered[0] + len(ordered)))
    )
    return {
        "range": [ordered[0], ordered[-1]],
        "count": len(ordered),
        "consecutive": consecutive,
        "fresh": not findings,
        "evidence_field": "exact JSON key episode_seed",
        "finding_count": len(findings),
        "findings": findings[:100],
    }


def _contract(
    settings: Mapping[str, Any],
    cfg: Any,
    estimator_config: v19.BatchEstimatorConfig,
    lock_config: v24.AuditedLockConfig,
    freshness: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    manifest = _loaded_local_sources(Path(settings["root"]))
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    arm_contracts: List[Dict[str, Any]] = []
    for arm in v39.ARM_NAMES:
        config = v39.planner_config_for_arm(arm)
        arm_contracts.append(
            {
                "arm": arm,
                "planner_kind": (
                    "uniform_random_feasible"
                    if arm == v39.ARM_RANDOM
                    else "belief_conditioned"
                ),
                "planner_config": (
                    None if config is None else config.to_dict()
                ),
            }
        )
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "experiment_version": v39.VERSION,
        "created_at_utc": runner20._utc_now(),
        "purpose": (
            "publication-grade paired two-link planner-component ablation"
        ),
        "protocol": PROTOCOL_NAME,
        "smoke": bool(settings["smoke"]),
        "publication_settings": bool(settings["publication_settings"]),
        "episode_start": int(settings["episode_start"]),
        "episodes": int(settings["episodes"]),
        "seeds": list(settings["seeds"]),
        "arms": arm_contracts,
        "expected_runs": int(settings["episodes"]) * len(v39.ARM_NAMES),
        "source_mask": list(v39.SOURCE_MASK),
        "estimator_config": v19.config_to_dict(estimator_config),
        "audited_lock_config": lock_config.to_dict(),
        "environment_metadata_path": str(settings["metadata"]),
        "environment_metadata_sha256": _sha256(Path(settings["metadata"])),
        "fixed_horizon_actions": int(cfg.max_steps),
        "fresh_seed_audit": freshness,
        "paired_contrasts": [
            "full_active_vs_no_pair_term",
            "full_active_vs_best_hypothesis_only",
            "no_pair_term_vs_best_hypothesis_only",
            "full_active_vs_random_feasible",
        ],
        "material_success_rate_difference": MATERIAL_RATE_DIFFERENCE,
        "runtime_threshold_s": RUNTIME_THRESHOLD_S,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "random_policy": {
            "domain": v39.POLICY_RNG_DOMAIN,
            "distribution": "uniform_over_frozen_candidate_bank",
            "environment_noise_rng_independent": True,
        },
        "truth_boundary": {
            "truth_used_after_completed_action_for_scoring_only": True,
            "legacy_pf_excluded_from_controller": True,
            "random_selector_receives_belief": False,
            "random_selector_receives_truth": False,
        },
        "reserved_final_range_untouched": [
            v39.RESERVED_START,
            v39.FINAL_END,
        ],
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


def _prepare(
    output: Path,
    root: Path,
    contract: Mapping[str, Any],
    resume: bool,
) -> None:
    contract_path = output / "control" / "campaign_contract.json"
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if _immutable(previous) != _immutable(contract):
            raise RuntimeError("V39 resume contract differs from frozen contract")
        return
    runner20._write_json_atomic(contract_path, contract)
    snapshot = output / "control" / "source_snapshot"
    for name, expected_sha in contract["source_sha256"].items():
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if _sha256(source) != expected_sha:
            raise RuntimeError(f"source changed before snapshot: {name}")
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _paths(
    output: Path,
    episode_index: int,
    seed: int,
    arm: str,
) -> Tuple[Path, Path]:
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
    planner = summary["planner"]
    random_policy = planner["random_policy"]
    return {
        "episode_index": int(summary["episode_index"]),
        "episode_seed": int(summary["episode_seed"]),
        "arm": str(summary["arm"]),
        "source_mask_l1": bool(summary["source_mask"][0]),
        "source_mask_l2": bool(summary["source_mask"][1]),
        "measurement_row_count": int(summary["measurement_row_count"]),
        "active_scalar_measurement_count": int(
            summary["active_scalar_measurement_count"]
        ),
        "action_count": int(summary["action_count"]),
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
        "first_track_action_time_s": summary["gate"][
            "first_track_action_time_s"
        ],
        "first_track_analysis_time_s": float(
            summary["gate"]["first_track_analysis_time_s"]
        ),
        "unsafe_transition_count": int(exact["false_transition_count"]),
        "unsafe_track_start_count": int(
            exact["false_locked_action_start_count"]
        ),
        "unsafe_track_end_count": int(
            exact["false_locked_action_end_count"]
        ),
        "audit_release_violation_count": int(
            summary["gate"]["audit_release_violation_count"]
        ),
        "false_confidence_action_count": int(
            summary["robustness"]["false_confidence_action_count"]
        ),
        "maximum_localization_error_m": float(
            summary["robustness"]["maximum_localization_error_m"]
        ),
        "maximum_track_localization_error_m": summary["robustness"][
            "maximum_track_localization_error_m"
        ],
        "mean_squared_action": float(summary["mean_squared_action"]),
        "maximum_combined_decision_runtime_s": float(
            summary["maximum_combined_decision_runtime_s"]
        ),
        "planner_kind": str(planner["kind"]),
        "planner_decision_count": int(planner["decision_count"]),
        "maximum_saved_hypothesis_count": int(
            planner["maximum_saved_hypothesis_count"]
        ),
        "policy_seed": (
            None if random_policy is None else int(random_policy["policy_seed"])
        ),
        "policy_index_stream_sha256": (
            None
            if random_policy is None
            else str(random_policy["policy_index_stream_sha256"])
        ),
        "noise_tape_sha256": str(summary["noise_tape_sha256"]),
        "noise_cursor_json": json.dumps(
            summary["noise_cursor"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "mission_support_json": json.dumps(
            summary["mission_support"],
            sort_keys=True,
            separators=(",", ":"),
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
        indices = rng.integers(
            0,
            array.size,
            size=(end - start, array.size),
        )
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


def _paired_contrast(
    comparator: Sequence[Mapping[str, Any]],
    full: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    comparator_label: str,
    full_label: str,
) -> Dict[str, Any]:
    left = {int(row["episode_seed"]): row for row in comparator}
    right = {int(row["episode_seed"]): row for row in full}
    if set(left) != set(right):
        raise RuntimeError("V39 paired contrast seed sets differ")
    ordered = sorted(left)
    terminal_left_only = terminal_right_only = 0
    tail_left_only = tail_right_only = 0
    localization_improvement: List[float] = []
    formation_improvement: List[float] = []
    lock_time_improvement: List[float] = []
    occupancy_improvement: List[float] = []
    effort_improvement: List[float] = []
    for episode_seed in ordered:
        lrow = left[episode_seed]
        rrow = right[episode_seed]
        lterm = bool(lrow["terminal_joint_success"])
        rterm = bool(rrow["terminal_joint_success"])
        terminal_left_only += int(lterm and not rterm)
        terminal_right_only += int(rterm and not lterm)
        ltail = bool(lrow["tail80_joint_success"])
        rtail = bool(rrow["tail80_joint_success"])
        tail_left_only += int(ltail and not rtail)
        tail_right_only += int(rtail and not ltail)
        # Positive values favor the full method.
        localization_improvement.append(
            float(lrow["terminal_localization_error_m"])
            - float(rrow["terminal_localization_error_m"])
        )
        formation_improvement.append(
            float(lrow["terminal_formation_error_m"])
            - float(rrow["terminal_formation_error_m"])
        )
        lock_time_improvement.append(
            float(lrow["first_track_analysis_time_s"])
            - float(rrow["first_track_analysis_time_s"])
        )
        occupancy_improvement.append(
            float(rrow["tail50_joint_occupancy"])
            - float(lrow["tail50_joint_occupancy"])
        )
        effort_improvement.append(
            float(lrow["mean_squared_action"])
            - float(rrow["mean_squared_action"])
        )
    count = len(ordered)
    terminal_left = sum(
        bool(left[value]["terminal_joint_success"]) for value in ordered
    )
    terminal_right = sum(
        bool(right[value]["terminal_joint_success"]) for value in ordered
    )
    tail_left = sum(
        bool(left[value]["tail80_joint_success"]) for value in ordered
    )
    tail_right = sum(
        bool(right[value]["tail80_joint_success"]) for value in ordered
    )
    return {
        "comparator": comparator_label,
        "full": full_label,
        "pairs": count,
        "terminal_success_rate_difference_full_minus_comparator": (
            (terminal_right - terminal_left) / max(count, 1)
        ),
        "tail80_success_rate_difference_full_minus_comparator": (
            (tail_right - tail_left) / max(count, 1)
        ),
        "terminal_discordance": {
            "comparator_only": terminal_left_only,
            "full_only": terminal_right_only,
            "exact_mcnemar_p_two_sided": runner38._discordance_p(
                terminal_left_only,
                terminal_right_only,
            ),
        },
        "tail80_discordance": {
            "comparator_only": tail_left_only,
            "full_only": tail_right_only,
            "exact_mcnemar_p_two_sided": runner38._discordance_p(
                tail_left_only,
                tail_right_only,
            ),
        },
        "terminal_localization_improvement_m": _bootstrap_mean(
            localization_improvement,
            seed,
        ),
        "terminal_formation_improvement_m": _bootstrap_mean(
            formation_improvement,
            seed + 1,
        ),
        "first_track_time_improvement_s": _bootstrap_mean(
            lock_time_improvement,
            seed + 2,
        ),
        "tail_occupancy_improvement": _bootstrap_mean(
            occupancy_improvement,
            seed + 3,
        ),
        "mean_squared_action_improvement": _bootstrap_mean(
            effort_improvement,
            seed + 4,
        ),
    }


def _cell_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    terminal = int(sum(bool(row["terminal_joint_success"]) for row in rows))
    tail = int(sum(bool(row["tail80_joint_success"]) for row in rows))
    lock = int(sum(bool(row["ever_locked"]) for row in rows))
    track_max = [
        float(row["maximum_track_localization_error_m"])
        for row in rows
        if row["maximum_track_localization_error_m"] is not None
    ]
    return {
        "episodes": count,
        "terminal_success_count": terminal,
        "terminal_success_rate": terminal / max(count, 1),
        "terminal_success_wilson95": runner38._wilson(terminal, count),
        "tail80_success_count": tail,
        "tail80_success_rate": tail / max(count, 1),
        "tail80_success_wilson95": runner38._wilson(tail, count),
        "dwell15_success_count": int(
            sum(bool(row["dwell15_joint_success"]) for row in rows)
        ),
        "ever_lock_count": lock,
        "ever_lock_rate": lock / max(count, 1),
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
        "terminal_localization_error_m": runner38._distribution(
            row["terminal_localization_error_m"] for row in rows
        ),
        "terminal_formation_error_m": runner38._distribution(
            row["terminal_formation_error_m"] for row in rows
        ),
        "maximum_localization_error_m": runner38._distribution(
            row["maximum_localization_error_m"] for row in rows
        ),
        "maximum_track_localization_error_m": runner38._distribution(track_max),
        "first_track_analysis_time_s": runner38._distribution(
            row["first_track_analysis_time_s"] for row in rows
        ),
        "tail50_joint_occupancy": runner38._distribution(
            row["tail50_joint_occupancy"] for row in rows
        ),
        "mean_squared_action": runner38._distribution(
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


def _verify_seed_pairing(
    outcomes: Mapping[str, v39.ArmOutcome],
) -> Dict[str, Any]:
    if set(outcomes) != set(v39.ARM_NAMES):
        raise RuntimeError("V39 seed pairing lacks one or more arms")
    summaries = [value.summary for value in outcomes.values()]
    tape_hashes = {str(value["noise_tape_sha256"]) for value in summaries}
    seeds = {int(value["episode_seed"]) for value in summaries}
    supports = {
        json.dumps(value["mission_support"], sort_keys=True)
        for value in summaries
    }
    noise_cursors = {
        json.dumps(value["noise_cursor"], sort_keys=True)
        for value in summaries
    }
    if (
        len(tape_hashes) != 1
        or len(seeds) != 1
        or len(supports) != 1
        or len(noise_cursors) != 1
    ):
        raise RuntimeError(
            "V39 paired arms differ in tape, seed, support, or cursor"
        )

    active_traces = [
        outcomes[arm].trace for arm in v39.ACTIVE_ARMS
    ]
    first_decisions: List[int] = []
    for trace in active_traces:
        decision = np.flatnonzero(
            np.asarray(trace["planner_candidate_count"], dtype=np.int64) > 0
        )
        if decision.size == 0:
            raise RuntimeError("V39 active arm never called the planner")
        first_decisions.append(int(decision[0]))
    if len(set(first_decisions)) != 1:
        raise RuntimeError("active arms have different first planner times")
    first = first_decisions[0]
    prefix_fields = (
        "action_speed",
        "action_yaw",
        "action_pitch",
        "truth_x",
        "truth_y",
        "truth_z",
    )
    reference = active_traces[0]
    for trace in active_traces[1:]:
        for field in prefix_fields:
            if not np.array_equal(
                np.asarray(reference[field])[:first],
                np.asarray(trace[field])[:first],
            ):
                raise RuntimeError(
                    f"active-arm common prefix differs for {field}"
                )
    full_hash = str(
        np.asarray(
            outcomes[v39.ARM_FULL].trace["planner_hypothesis_sha256"]
        )[first]
    )
    no_pair_hash = str(
        np.asarray(
            outcomes[v39.ARM_NO_PAIR].trace[
                "planner_hypothesis_sha256"
            ]
        )[first]
    )
    if not full_hash or full_hash != no_pair_hash:
        raise RuntimeError(
            "full and no-pair arms evaluated different initial hypotheses"
        )
    return {
        "episode_seed": next(iter(seeds)),
        "noise_tape_sha256": next(iter(tape_hashes)),
        "arm_count": len(outcomes),
        "common_mission_support": True,
        "common_noise_cursor": True,
        "active_common_prefix_actions": first,
        "full_no_pair_initial_hypothesis_sha256": full_hash,
    }


def _material_decision(
    contrast: Mapping[str, Any],
    full: Mapping[str, Any],
    comparator: Mapping[str, Any],
    *,
    positive: str,
    negative: str,
    integrity_valid: bool,
) -> str:
    safety = bool(
        int(full["unsafe_transition_count"])
        <= int(comparator["unsafe_transition_count"])
        and int(full["unsafe_track_start_count"])
        <= int(comparator["unsafe_track_start_count"])
        and int(full["unsafe_track_end_count"])
        <= int(comparator["unsafe_track_end_count"])
    )
    material = bool(
        float(
            contrast[
                "terminal_success_rate_difference_full_minus_comparator"
            ]
        )
        >= MATERIAL_RATE_DIFFERENCE
        or float(
            contrast[
                "tail80_success_rate_difference_full_minus_comparator"
            ]
        )
        >= MATERIAL_RATE_DIFFERENCE
    )
    return positive if integrity_valid and safety and material else negative


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    elapsed_s: float,
    pairing: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    indexed: Dict[str, List[Mapping[str, Any]]] = {}
    by_arm: Dict[str, Any] = {}
    for arm in v39.ARM_NAMES:
        cell = [row for row in rows if row["arm"] == arm]
        indexed[arm] = cell
        by_arm[arm] = _cell_metrics(cell)
    expected = int(contract["expected_runs"])
    integrity_checks = {
        "all_runs_complete": len(rows) == expected,
        "all_four_arms_paired": len(pairing) == int(contract["episodes"]),
        "publication_settings": bool(contract["publication_settings"]),
        "fixed_horizon_complete": all(
            int(row["action_count"]) == int(contract["fixed_horizon_actions"])
            for row in rows
        ),
        "two_sources_active": all(
            bool(row["source_mask_l1"]) and bool(row["source_mask_l2"])
            for row in rows
        ),
        "active_scalar_denominators_valid": all(
            int(row["active_scalar_measurement_count"])
            == 2 * int(row["measurement_row_count"])
            for row in rows
        ),
        "best_arm_at_most_one_hypothesis": all(
            int(row["maximum_saved_hypothesis_count"]) <= 1
            for row in indexed[v39.ARM_BEST_ONLY]
        ),
        "random_arm_has_no_belief_hypotheses": all(
            int(row["maximum_saved_hypothesis_count"]) == 0
            and row["planner_kind"] == "uniform_random_feasible"
            for row in indexed[v39.ARM_RANDOM]
        ),
        "random_policy_tapes_present": all(
            row["policy_seed"] is not None
            and bool(row["policy_index_stream_sha256"])
            for row in indexed[v39.ARM_RANDOM]
        ),
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
        < RUNTIME_THRESHOLD_S,
        "fresh_seed_audit_pass": bool(
            contract["smoke"] or contract["fresh_seed_audit"]["fresh"]
        ),
        "reserved_final_untouched": contract[
            "reserved_final_range_untouched"
        ]
        == [v39.RESERVED_START, v39.FINAL_END],
    }
    integrity_valid = bool(all(integrity_checks.values()))
    full = indexed[v39.ARM_FULL]
    contrasts = {
        "full_vs_no_pair": _paired_contrast(
            indexed[v39.ARM_NO_PAIR],
            full,
            seed=BOOTSTRAP_SEED,
            comparator_label=v39.ARM_NO_PAIR,
            full_label=v39.ARM_FULL,
        ),
        "full_vs_best_only": _paired_contrast(
            indexed[v39.ARM_BEST_ONLY],
            full,
            seed=BOOTSTRAP_SEED + 10,
            comparator_label=v39.ARM_BEST_ONLY,
            full_label=v39.ARM_FULL,
        ),
        "no_pair_vs_best_only": _paired_contrast(
            indexed[v39.ARM_BEST_ONLY],
            indexed[v39.ARM_NO_PAIR],
            seed=BOOTSTRAP_SEED + 20,
            comparator_label=v39.ARM_BEST_ONLY,
            full_label=v39.ARM_NO_PAIR,
        ),
        "full_vs_random": _paired_contrast(
            indexed[v39.ARM_RANDOM],
            full,
            seed=BOOTSTRAP_SEED + 30,
            comparator_label=v39.ARM_RANDOM,
            full_label=v39.ARM_FULL,
        ),
    }
    full_metrics = by_arm[v39.ARM_FULL]
    operational = bool(
        full_metrics["terminal_success_rate"] >= 0.90
        and full_metrics["tail80_success_rate"] >= 0.85
        and full_metrics["terminal_localization_error_m"] is not None
        and full_metrics["terminal_localization_error_m"]["p95"] < 7.0
        and full_metrics["unsafe_transition_count"] == 0
        and full_metrics["unsafe_track_start_count"] == 0
        and full_metrics["unsafe_track_end_count"] == 0
    )
    if bool(contract["smoke"]):
        component_decisions = {
            "pair_term": "SMOKE_ONLY",
            "retained_hypotheses": "SMOKE_ONLY",
            "informed_selection": "SMOKE_ONLY",
        }
        decision = "SMOKE_PASS" if integrity_valid else "SMOKE_FAIL"
    else:
        component_decisions = {
            "pair_term": _material_decision(
                contrasts["full_vs_no_pair"],
                full_metrics,
                by_arm[v39.ARM_NO_PAIR],
                positive="MATERIAL_PAIR_TERM_EFFECT",
                negative="NO_MATERIAL_PAIR_TERM_EFFECT_DETECTED",
                integrity_valid=integrity_valid,
            ),
            "retained_hypotheses": _material_decision(
                contrasts["full_vs_best_only"],
                full_metrics,
                by_arm[v39.ARM_BEST_ONLY],
                positive="MATERIAL_RETAINED_HYPOTHESIS_EFFECT",
                negative=(
                    "NO_MATERIAL_RETAINED_HYPOTHESIS_EFFECT_DETECTED"
                ),
                integrity_valid=integrity_valid,
            ),
            "informed_selection": _material_decision(
                contrasts["full_vs_random"],
                full_metrics,
                by_arm[v39.ARM_RANDOM],
                positive="MATERIAL_INFORMED_SELECTION_EFFECT",
                negative="NO_MATERIAL_INFORMED_SELECTION_EFFECT_DETECTED",
                integrity_valid=integrity_valid,
            ),
        }
        decision = "V39_COMPLETE" if integrity_valid else "V39_INVALID"
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "analysis_version": "v39_analysis_1.0",
        "status": "complete",
        "smoke": bool(contract["smoke"]),
        "publication_settings": bool(contract["publication_settings"]),
        "elapsed_wall_s": float(elapsed_s),
        "row_count": len(rows),
        "expected_row_count": expected,
        "integrity_checks": integrity_checks,
        "integrity_valid": integrity_valid,
        "decision": decision,
        "full_arm_operational_gate": operational,
        "component_decisions": component_decisions,
        "material_rate_difference": MATERIAL_RATE_DIFFERENCE,
        "by_arm": by_arm,
        "paired_contrasts": contrasts,
        "seed_pairing": list(pairing),
        "development_only": True,
        "reserved_final_range_untouched": contract[
            "reserved_final_range_untouched"
        ],
        "interpretation_guard": (
            "A negative material-effect decision means no >=5 percentage-point "
            "effect was detected under this protocol; it is not equivalence."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    settings = _settings(_parse(argv))
    root = Path(settings["root"])
    output = Path(settings["output"])
    freshness = None
    if not settings["smoke"]:
        freshness = _fresh_seed_audit(root, output, settings["seeds"])
        if not freshness["fresh"]:
            raise RuntimeError(
                f"V39 fresh-seed audit failed: {freshness['findings'][:10]}"
            )
    cfg = runner20._load_environment_config(Path(settings["metadata"]))
    estimator_config = v19.BatchEstimatorConfig(
        coarse_candidates=int(settings["coarse_candidates"]),
        coarse_sweeps=int(settings["coarse_sweeps"]),
        local_starts=int(settings["local_starts"]),
        gate_mode="raw",
        candidate_radial_distribution="uniform_radius",
    )
    lock_config = v24.AuditedLockConfig()
    contract = _contract(
        settings,
        cfg,
        estimator_config,
        lock_config,
        freshness,
    )
    _prepare(output, root, contract, bool(settings["resume"]))
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
            outcomes: Dict[str, v39.ArmOutcome] = {}
            for arm in v39.ARM_NAMES:
                result_path, trace_path = _paths(
                    output,
                    episode_index,
                    int(seed),
                    arm,
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
                    outcome = v39.ArmOutcome(summary=summary, trace=trace)
                else:
                    outcome = v39.run_arm(
                        cfg=cfg,
                        tape=tape,
                        episode_seed=int(seed),
                        episode_index=episode_index,
                        arm=arm,
                        estimator_config=estimator_config,
                        lock_config=lock_config,
                    )
                    runner20._write_json_atomic(result_path, outcome.summary)
                    runner20._write_npz_atomic(trace_path, outcome.trace)
                if outcome.summary["noise_tape_sha256"] != tape.content_sha256():
                    raise RuntimeError("V39 saved the wrong noise-tape hash")
                if int(outcome.summary["action_count"]) != int(cfg.max_steps):
                    raise RuntimeError("V39 saved an incomplete trace")
                outcomes[arm] = outcome
                rows.append(_flatten(outcome.summary))
                completed += 1
                if _loaded_local_sources(root) != expected_sources:
                    raise RuntimeError(
                        "V39 loaded source closure changed during execution"
                    )
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
                        "last_arm": arm,
                        "resumable": True,
                    },
                )
                if completed % int(settings["progress_every"]) == 0:
                    print(
                        f"[{runner20._utc_now()}] V39 {completed}/{total}; "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s; "
                        f"seed={seed} arm={arm}",
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
                "integrity_valid": summary["integrity_valid"],
                "full_arm_operational_gate": summary[
                    "full_arm_operational_gate"
                ],
                "component_decisions": summary["component_decisions"],
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
