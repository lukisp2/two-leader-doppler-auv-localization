#!/usr/bin/env python3
"""Run the frozen V28 estimator stress and calibration campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import run_v27_publication_baselines as runner27
import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27
import uuv_v28_estimator_stress as v28


RUNNER_VERSION = "v28_estimator_stress_runner_1.1"
PROTOCOL_NAME = "EXPERIMENT_PROTOCOL_V28_ESTIMATOR_STRESS.md"
ARMS = (v27.GLOBAL_FULL, v27.LOCAL_NLS6, v27.PF_LW_16384)
PREFIXES_S = (120.0, 440.0)
SMOKE_EPISODE_INDICES = (0, 73)
FULL_EPISODE_INDICES = tuple(range(100))
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 28_045_000

SOURCE_FILES = (
    "run_v28_estimator_stress.py",
    "uuv_v28_estimator_stress.py",
    "audit_v28_campaign.py",
    PROTOCOL_NAME,
    "tests/test_uuv_v28_estimator_stress.py",
    "uuv_v27_publication_baselines.py",
    "uuv_v19_observability.py",
)


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    smoke = bool(args.smoke)
    archive = (
        args.archive_dir
        or root
        / "experiments_v19_observability_estimator_benchmark_dev100"
        / "measurement_archive"
    ).expanduser().resolve()
    output = (
        args.output_dir
        or root / f"experiments_v28_estimator_stress_{'smoke' if smoke else 'dev100'}"
    ).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(archive)
    if int(args.progress_every) < 1:
        raise ValueError("progress-every must be positive")
    config = (
        v27.EvaluatorConfig(
            coarse_candidates=512,
            coarse_sweeps=1,
            local_starts=8,
            maximum_modes=8,
            pf_particles_small=512,
            pf_particles_large=2048,
        )
        if smoke
        else v27.EvaluatorConfig()
    )
    return {
        "root": root,
        "archive": archive,
        "output": output,
        "smoke": smoke,
        "resume": bool(args.resume),
        "progress_every": int(args.progress_every),
        "config": config,
        "episode_indices": SMOKE_EPISODE_INDICES if smoke else FULL_EPISODE_INDICES,
    }


def _source_manifest(root: Path) -> Dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = v27.sha256_file(path)
    return result


def _stable_runtime_environment() -> Dict[str, Any]:
    """Return reproducibility metadata without per-process identity.

    A PID changes on every invocation and therefore cannot be part of a
    resumable campaign contract.  The original V28 run used runner 1.0 and
    retained its frozen source snapshot; runner 1.1 removes that transient
    field for future campaigns.
    """
    environment = dict(runner27._runtime_environment())
    environment.pop("pid", None)
    return environment


def _contract(settings: Mapping[str, Any]) -> Dict[str, Any]:
    source_rows = [
        runner27._validate_source_episode(
            runner27._episode_directory(settings["archive"], int(index)), int(index)
        )
        for index in settings["episode_indices"]
    ]
    value = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "stress_version": v28.VERSION,
        "estimator_version": v27.VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": v27.sha256_file(settings["root"] / PROTOCOL_NAME),
        "development_only": True,
        "no_rl_training": True,
        "smoke": bool(settings["smoke"]),
        "source_archive": str(settings["archive"]),
        "episode_indices": list(settings["episode_indices"]),
        "episode_seeds": [row["episode_seed"] for row in source_rows],
        "prefixes_s": list(PREFIXES_S),
        "arms": list(ARMS),
        "stress_contract": v28.condition_contract(),
        "estimator_config": v27.config_to_dict(settings["config"]),
        "source_episodes": source_rows,
        "source_manifest": _source_manifest(settings["root"]),
        "reserved_seed_range_untouched": [v27.RESERVED_SEED_START, v27.RESERVED_SEED_END],
        "sealed_final_seed_range_untouched": [v27.FINAL_SEED_START, v27.FINAL_SEED_END],
        "truth_boundary": "all condition/arm/checkpoint outputs for an episode are persisted before truth is opened",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "runtime_environment": _stable_runtime_environment(),
    }
    value["contract_sha256"] = runner27._json_hash(value)
    return value


def _prepare(settings: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    output: Path = settings["output"]
    if output.exists() and not settings["resume"]:
        raise FileExistsError(f"V28 output exists; use --resume: {output}")
    for relative in ("control/source_snapshot", "unscored", "episode_results"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    contract_path = output / "control" / "campaign_contract.json"
    if contract_path.exists():
        if runner27._read_json(contract_path) != runner27._json_safe(contract):
            raise RuntimeError("V28 resume contract mismatch")
        return
    runner27._write_json_atomic(contract_path, contract)
    for relative in SOURCE_FILES:
        source = settings["root"] / relative
        destination = output / "control" / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    runner27._write_json_atomic(
        output / "control" / "source_manifest.json",
        {
            "files": contract["source_manifest"],
            "manifest_sha256": runner27._json_hash(contract["source_manifest"]),
        },
    )


def _stem(episode_index: int, condition: str, prefix_s: float, arm: str) -> str:
    return (
        f"episode_{int(episode_index):04d}_{condition}_"
        f"prefix_{int(round(prefix_s)):04d}_{arm}"
    )


def _paths(
    output: Path, episode_index: int, condition: str, prefix_s: float, arm: str
) -> Tuple[Path, Path]:
    stem = _stem(episode_index, condition, prefix_s, arm) + ".json"
    return output / "unscored" / stem, output / "episode_results" / stem


def _stats(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _nearest_rank(values: Iterable[Optional[float]], probability: float) -> Optional[float]:
    finite = sorted(
        float(value) for value in values if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return None
    rank = max(1, int(math.ceil(float(probability) * len(finite))))
    return finite[min(rank - 1, len(finite) - 1)]


def _bootstrap(differences: np.ndarray) -> Dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    samples = []
    remaining = BOOTSTRAP_RESAMPLES
    while remaining:
        chunk = min(5000, remaining)
        indices = rng.integers(0, values.size, size=(chunk, values.size), endpoint=False)
        samples.append(np.mean(values[indices], axis=1))
        remaining -= chunk
    bootstrap = np.concatenate(samples)
    return {
        "mean": float(np.mean(values)),
        "lower95": float(np.percentile(bootstrap, 2.5)),
        "upper95": float(np.percentile(bootstrap, 97.5)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_cell: Dict[str, Any] = {}
    paired: Dict[str, Any] = {}
    for condition in v28.CONDITIONS:
        for prefix in PREFIXES_S:
            for arm in ARMS:
                subset = [
                    row
                    for row in rows
                    if row["condition"] == condition
                    and row["arm"] == arm
                    and float(row["prefix_s"]) == float(prefix)
                ]
                coverages = [
                    row["score"]["nominal_radius95_covers"]
                    for row in subset
                    if row["score"]["nominal_radius95_covers"] is not None
                ]
                ratios = [
                    float(row["score"]["endpoint_position_error_m"])
                    / float(row["unscored"]["nominal_radius95_m"])
                    for row in subset
                    if row["unscored"]["nominal_radius95_m"] is not None
                    and float(row["unscored"]["nominal_radius95_m"]) > 0.0
                ]
                by_cell[f"{condition}|{arm}@{int(prefix)}"] = {
                    "condition": condition,
                    "arm": arm,
                    "prefix_s": float(prefix),
                    "count": len(subset),
                    "success_le_7m_count": int(
                        sum(bool(row["score"]["success_le_7m"]) for row in subset)
                    ),
                    "endpoint_error_m": _stats(
                        row["score"]["endpoint_position_error_m"] for row in subset
                    ),
                    "runtime_s": _stats(row["unscored"]["runtime_s"] for row in subset),
                    "runtime_p99_nearest_rank_s": _nearest_rank(
                        (row["unscored"]["runtime_s"] for row in subset), 0.99
                    ),
                    "coverage_defined_count": len(coverages),
                    "coverage_count": int(sum(bool(value) for value in coverages)),
                    "coverage_rate": (
                        None
                        if not coverages
                        else float(np.mean(np.asarray(coverages, dtype=np.float64)))
                    ),
                    "error_over_radius_q95": (
                        None if not ratios else float(np.percentile(np.asarray(ratios), 95.0))
                    ),
                }
            global_rows = {
                int(row["episode_index"]): float(row["score"]["endpoint_position_error_m"])
                for row in rows
                if row["condition"] == condition
                and row["arm"] == v27.GLOBAL_FULL
                and float(row["prefix_s"]) == float(prefix)
            }
            for arm in (v27.LOCAL_NLS6, v27.PF_LW_16384):
                comparator = {
                    int(row["episode_index"]): float(row["score"]["endpoint_position_error_m"])
                    for row in rows
                    if row["condition"] == condition
                    and row["arm"] == arm
                    and float(row["prefix_s"]) == float(prefix)
                }
                if set(global_rows) != set(comparator):
                    raise RuntimeError("V28 paired episode sets differ")
                differences = np.asarray(
                    [comparator[index] - global_rows[index] for index in sorted(global_rows)],
                    dtype=np.float64,
                )
                paired[f"{condition}|global_full_vs_{arm}@{int(prefix)}"] = _bootstrap(
                    differences
                )
    return {"cell_count": len(rows), "by_condition_arm_prefix": by_cell, "paired_error_gain_m": paired}


def _decision(
    aggregate: Mapping[str, Any],
    *,
    smoke: bool,
    deterministic_valid: bool,
    audit_valid: bool,
) -> Dict[str, Any]:
    if smoke:
        valid = bool(deterministic_valid and audit_valid)
        return {
            "decision": "SMOKE_PASS" if valid else "SMOKE_FAIL",
            "smoke_only": True,
            "scientific_interpretation_forbidden": True,
            "deterministic_repeat_valid": bool(deterministic_valid),
            "independent_audit_valid": bool(audit_valid),
        }
    table = aggregate["by_condition_arm_prefix"]
    failing = []
    runtime_failures = []
    for condition in v28.CONDITIONS:
        early = table[f"{condition}|global_full@120"]
        late = table[f"{condition}|global_full@440"]
        early_required = 100 if condition == v28.NOMINAL else 80
        late_required = 100 if condition == v28.NOMINAL else 90
        if int(early["success_le_7m_count"]) < early_required or int(
            late["success_le_7m_count"]
        ) < late_required:
            failing.append(
                {
                    "condition": condition,
                    "success_120": int(early["success_le_7m_count"]),
                    "required_120": early_required,
                    "success_440": int(late["success_le_7m_count"]),
                    "required_440": late_required,
                }
            )
        runtime = late["runtime_p99_nearest_rank_s"]
        if runtime is None or float(runtime) >= 2.0:
            runtime_failures.append({"condition": condition, "runtime_p99_s": runtime})
    material_headroom = []
    for condition in v28.CONDITIONS:
        global_success = int(
            table[f"{condition}|global_full@440"]["success_le_7m_count"]
        )
        local_success = int(
            table[f"{condition}|local_nls6@440"]["success_le_7m_count"]
        )
        paired = aggregate["paired_error_gain_m"][
            f"{condition}|global_full_vs_local_nls6@440"
        ]
        if global_success - local_success >= 5 and float(paired["lower95"]) > 0.5:
            material_headroom.append(
                {
                    "condition": condition,
                    "success_gain_pp": global_success - local_success,
                    "paired_error_gain_m": paired,
                }
            )
    calibration_flags = []
    for condition in v28.CONDITIONS:
        for arm in ARMS:
            row = table[f"{condition}|{arm}@440"]
            coverage = row["coverage_rate"]
            if coverage is not None and float(coverage) < 0.90:
                calibration_flags.append(
                    {
                        "condition": condition,
                        "arm": arm,
                        "coverage_rate": coverage,
                        "error_over_radius_q95": row["error_over_radius_q95"],
                    }
                )
    robust = bool(audit_valid and not failing and not runtime_failures)
    return {
        "decision": (
            "PROCEED_TO_RESERVED_CLOSED_LOOP_QUALIFICATION"
            if robust
            else "MODEL_EXTENSION_REQUIRED_BEFORE_CLOSED_LOOP"
        ),
        "development_only": True,
        "final_authorized": False,
        "integrity_valid": bool(audit_valid),
        "failing_stress_families": failing,
        "runtime_failures": runtime_failures,
        "material_global_search_headroom": material_headroom,
        "point_estimator_recommendation": (
            "RETAIN_GLOBAL_FULL"
            if material_headroom
            else "SIMPLIFY_TO_LOCAL_NLS6_PENDING_GATE_TEST"
        ),
        "calibration_flags": calibration_flags,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    settings = _settings(args)
    contract = _contract(settings)
    _prepare(settings, contract)
    output: Path = settings["output"]
    started = time.perf_counter()
    runner27._write_json_atomic(
        output / "run_state.json",
        {
            "status": "running",
            "started_at_utc": runner27._utc_now(),
            "contract_sha256": contract["contract_sha256"],
        },
    )
    source_by_index = {
        int(row["episode_index"]): row for row in contract["source_episodes"]
    }
    rows: List[Dict[str, Any]] = []
    deterministic_checks: List[Dict[str, Any]] = []
    total = len(settings["episode_indices"])
    for position, episode_index in enumerate(settings["episode_indices"], start=1):
        source = source_by_index[int(episode_index)]
        online = v19.load_online_history(Path(source["online_path"]))
        stressed_by_condition = {
            condition: v28.apply_stress(online, condition, int(episode_index))
            for condition in v28.CONDITIONS
        }
        unscored_by_key: Dict[Tuple[str, float, str], Dict[str, Any]] = {}
        # Truth remains unopened throughout this complete estimator phase.
        for condition, stressed_full in stressed_by_condition.items():
            stress_hash = v28.history_sha256(stressed_full)
            for prefix_s in PREFIXES_S:
                history = stressed_full.prefix(prefix_s)
                for arm in ARMS:
                    unscored_path, _ = _paths(
                        output, episode_index, condition, prefix_s, arm
                    )
                    if unscored_path.exists() and settings["resume"]:
                        wrapper = runner27._read_json(unscored_path)
                    else:
                        estimator = v27.evaluate_arm(
                            arm,
                            history,
                            source["support_radius_min_m"],
                            source["support_radius_max_m"],
                            settings["config"],
                        )
                        wrapper = {
                            "schema_version": 1,
                            "contract_sha256": contract["contract_sha256"],
                            "episode_index": int(episode_index),
                            "condition": condition,
                            "prefix_s": float(prefix_s),
                            "arm": arm,
                            "online_inputs_sha256": source["online_sha256"],
                            "stressed_history_sha256": stress_hash,
                            "payload": estimator.to_unscored_dict(),
                        }
                        runner27._write_json_atomic(unscored_path, wrapper)
                    identity = (
                        wrapper.get("contract_sha256") == contract["contract_sha256"]
                        and wrapper.get("episode_index") == int(episode_index)
                        and wrapper.get("condition") == condition
                        and float(wrapper.get("prefix_s", -1)) == float(prefix_s)
                        and wrapper.get("arm") == arm
                        and wrapper.get("stressed_history_sha256") == stress_hash
                    )
                    if not identity or "payload" not in wrapper:
                        raise RuntimeError(f"V28 unscored identity mismatch: {unscored_path}")
                    unscored_by_key[(condition, float(prefix_s), arm)] = wrapper

        if settings["smoke"] and int(episode_index) == SMOKE_EPISODE_INDICES[0]:
            for condition, stressed_full in stressed_by_condition.items():
                history = stressed_full.prefix(440.0)
                for arm in ARMS:
                    repeated = v27.evaluate_arm(
                        arm,
                        history,
                        source["support_radius_min_m"],
                        source["support_radius_max_m"],
                        settings["config"],
                    )
                    original = v27.output_from_unscored_dict(
                        unscored_by_key[(condition, 440.0, arm)]["payload"]
                    )
                    valid = (
                        v27.deterministic_payload(original)
                        == v27.deterministic_payload(repeated)
                    )
                    deterministic_checks.append(
                        {"condition": condition, "arm": arm, "valid": bool(valid)}
                    )
                    if not valid:
                        raise RuntimeError(
                            f"V28 deterministic repeat failed: {condition}/{arm}"
                        )

        truth = v19.load_truth_diagnostics(Path(source["truth_path"]))
        for condition, stressed_full in stressed_by_condition.items():
            for prefix_s in PREFIXES_S:
                history = stressed_full.prefix(prefix_s)
                for arm in ARMS:
                    wrapper = unscored_by_key[(condition, float(prefix_s), arm)]
                    estimator = v27.output_from_unscored_dict(wrapper["payload"])
                    score = v27.score_output(estimator, truth, history)
                    unscored_path, scored_path = _paths(
                        output, episode_index, condition, prefix_s, arm
                    )
                    row = {
                        "schema_version": 1,
                        "contract_sha256": contract["contract_sha256"],
                        "episode_index": int(episode_index),
                        "episode_seed": int(source["episode_seed"]),
                        "condition": condition,
                        "prefix_s": float(prefix_s),
                        "arm": arm,
                        "online_inputs_sha256": source["online_sha256"],
                        "truth_labels_sha256": source["truth_sha256"],
                        "stressed_history_sha256": wrapper["stressed_history_sha256"],
                        "unscored_sha256": v27.sha256_file(unscored_path),
                        "unscored": wrapper["payload"],
                        "score": score,
                    }
                    if scored_path.exists() and settings["resume"]:
                        if runner27._read_json(scored_path) != runner27._json_safe(row):
                            raise RuntimeError(f"V28 scored resume mismatch: {scored_path}")
                    else:
                        runner27._write_json_atomic(scored_path, row)
                    rows.append(row)
        if position % settings["progress_every"] == 0 or position == total:
            print(
                f"V28 {'smoke' if settings['smoke'] else 'dev'} episode "
                f"{position}/{total} complete",
                flush=True,
            )

    expected_cells = len(settings["episode_indices"]) * len(v28.CONDITIONS) * len(PREFIXES_S) * len(ARMS)
    if len(rows) != expected_cells:
        raise RuntimeError(f"V28 expected {expected_cells} rows, found {len(rows)}")
    aggregate = _aggregate(rows)
    preliminary = {
        "schema_version": 1,
        "status": "complete_pre_audit",
        "runner_version": RUNNER_VERSION,
        "stress_version": v28.VERSION,
        "estimator_version": v27.VERSION,
        "contract_sha256": contract["contract_sha256"],
        "smoke": bool(settings["smoke"]),
        "development_only": True,
        "episode_count": len(settings["episode_indices"]),
        "cell_count": len(rows),
        "deterministic_checks": deterministic_checks,
        "aggregate": aggregate,
        "elapsed_wall_s": float(time.perf_counter() - started),
        "completed_at_utc": runner27._utc_now(),
    }
    runner27._write_json_atomic(output / "campaign_summary.json", preliminary)
    import audit_v28_campaign

    audit = audit_v28_campaign.audit_campaign(output)
    runner27._write_json_atomic(output / "independent_audit.json", audit)
    deterministic_valid = (
        all(bool(row["valid"]) for row in deterministic_checks)
        if settings["smoke"]
        else True
    )
    decision = _decision(
        aggregate,
        smoke=bool(settings["smoke"]),
        deterministic_valid=deterministic_valid,
        audit_valid=bool(audit.get("valid", False)),
    )
    final = dict(preliminary)
    final.update(
        {
            "status": "complete" if audit.get("valid", False) else "audit_failed",
            "integrity_valid": bool(audit.get("valid", False)),
            "independent_audit": audit,
            "decision": decision["decision"],
        }
    )
    runner27._write_json_atomic(output / "campaign_summary.json", final)
    runner27._write_json_atomic(
        output / "decision.json",
        {"contract_sha256": contract["contract_sha256"], **decision},
    )
    runner27._write_json_atomic(
        output / "run_state.json",
        {
            "status": final["status"],
            "decision": decision["decision"],
            "elapsed_wall_s": float(time.perf_counter() - started),
            "completed_at_utc": runner27._utc_now(),
            "contract_sha256": contract["contract_sha256"],
        },
    )
    print(json.dumps(runner27._json_safe(decision), indent=2, sort_keys=True), flush=True)
    return 0 if audit.get("valid", False) and decision["decision"] != "SMOKE_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
