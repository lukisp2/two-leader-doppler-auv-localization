#!/usr/bin/env python3
"""Run the frozen V27 common estimator evaluator.

The runner writes every estimator output before loading that episode's truth
labels.  It then scores the persisted outputs, aggregates paired results and
invokes the independent artifact auditor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27


RUNNER_VERSION = "v27_publication_baselines_runner_1.0"
PROTOCOL_NAME = "EXPERIMENT_PROTOCOL_V27_PUBLICATION_BASELINES.md"
FULL_PREFIXES_S = (30.0, 60.0, 120.0, 240.0, 440.0)
SMOKE_PREFIXES_S = (30.0, 120.0, 440.0)
SMOKE_EPISODE_INDICES = (0, 73)
FULL_EPISODE_INDICES = tuple(range(100))
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 27_045_000

SOURCE_FILES = (
    "run_v27_publication_baselines.py",
    "uuv_v27_publication_baselines.py",
    "audit_v27_campaign.py",
    PROTOCOL_NAME,
    "tests/test_uuv_v27_publication_baselines.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _write_json_atomic(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen V27 estimator baselines")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def _default_archive(root: Path) -> Path:
    return root / "experiments_v19_observability_estimator_benchmark_dev100" / "measurement_archive"


def _default_output(root: Path, smoke: bool) -> Path:
    suffix = "smoke" if smoke else "dev100"
    return root / f"experiments_v27_publication_baselines_{suffix}"


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    smoke = bool(args.smoke)
    archive = (args.archive_dir or _default_archive(root)).expanduser().resolve()
    output = (args.output_dir or _default_output(root, smoke)).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"V19 measurement archive is missing: {archive}")
    if output == archive or output in archive.parents or archive in output.parents:
        raise ValueError("output and source archive must be disjoint")
    if int(args.progress_every) < 1:
        raise ValueError("progress-every must be positive")
    if smoke:
        config = v27.EvaluatorConfig(
            coarse_candidates=512,
            coarse_sweeps=1,
            local_starts=8,
            maximum_modes=8,
            pf_particles_small=512,
            pf_particles_large=2048,
        )
        episode_indices = SMOKE_EPISODE_INDICES
        prefixes = SMOKE_PREFIXES_S
    else:
        config = v27.EvaluatorConfig()
        episode_indices = FULL_EPISODE_INDICES
        prefixes = FULL_PREFIXES_S
    return {
        "root": root,
        "archive": archive,
        "output": output,
        "smoke": smoke,
        "resume": bool(args.resume),
        "progress_every": int(args.progress_every),
        "config": config,
        "episode_indices": episode_indices,
        "prefixes_s": prefixes,
    }


def _episode_directory(archive: Path, episode_index: int) -> Path:
    matches = sorted(archive.glob(f"episode_{int(episode_index):04d}_seed_*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one V19 source directory for episode {episode_index}, found {len(matches)}"
        )
    return matches[0]


def _validate_source_episode(directory: Path, episode_index: int) -> Dict[str, Any]:
    metadata_path = directory / "capture_metadata.json"
    online_path = directory / "online_inputs.npz"
    truth_path = directory / "truth_labels.npz"
    metadata = _read_json(metadata_path)
    seed = int(metadata.get("episode_seed", -1))
    if int(metadata.get("episode_index", -1)) != int(episode_index):
        raise RuntimeError("V19 episode index mismatch")
    if seed != v19.V181_DEV_SEED_START + int(episode_index):
        raise RuntimeError("V19 episode seed mismatch")
    v27.assert_v27_development_seed(seed)
    if metadata.get("controller") != v19.PRIMARY_CONTROLLER:
        raise RuntimeError("V27 requires the frozen deterministic-greedy source")
    if not bool(metadata.get("replay_integrity", {}).get("passed", False)):
        raise RuntimeError("V19 replay integrity did not pass")
    if v27.sha256_file(online_path) != str(metadata.get("online_inputs_sha256")):
        raise RuntimeError("online input hash mismatch")
    if v27.sha256_file(truth_path) != str(metadata.get("truth_labels_sha256")):
        raise RuntimeError("truth-label hash mismatch")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("V19 metadata lacks provenance")
    trace_path = Path(str(provenance.get("trace_path"))).expanduser().resolve()
    trace_sha256 = str(provenance.get("trace_sha256"))
    if not trace_path.is_file() or v27.sha256_file(trace_path) != trace_sha256:
        raise RuntimeError("frozen legacy PF trace is missing or changed")
    return {
        "episode_index": int(episode_index),
        "episode_seed": seed,
        "directory": str(directory.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": v27.sha256_file(metadata_path),
        "online_path": str(online_path.resolve()),
        "online_sha256": str(metadata["online_inputs_sha256"]),
        "truth_path": str(truth_path.resolve()),
        "truth_sha256": str(metadata["truth_labels_sha256"]),
        "trace_path": str(trace_path),
        "trace_sha256": trace_sha256,
        "support_radius_min_m": float(metadata["support_radius_min_m"]),
        "support_radius_max_m": float(metadata["support_radius_max_m"]),
    }


def _source_manifest(root: Path) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"V27 source file is missing: {source}")
        manifest[relative] = v27.sha256_file(source)
    return manifest


def _runtime_environment() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pid": os.getpid(),
    }


def _build_contract(settings: Mapping[str, Any]) -> Dict[str, Any]:
    sources = [
        _validate_source_episode(
            _episode_directory(settings["archive"], int(index)), int(index)
        )
        for index in settings["episode_indices"]
    ]
    contract = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "evaluator_version": v27.VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": v27.sha256_file(settings["root"] / PROTOCOL_NAME),
        "development_only": True,
        "no_rl_training": True,
        "smoke": bool(settings["smoke"]),
        "source_archive": str(settings["archive"]),
        "episode_indices": list(settings["episode_indices"]),
        "episode_seeds": [row["episode_seed"] for row in sources],
        "prefixes_s": list(settings["prefixes_s"]),
        "arms": list(v27.ARM_NAMES),
        "evaluator_config": v27.config_to_dict(settings["config"]),
        "source_episodes": sources,
        "source_manifest": _source_manifest(settings["root"]),
        "reserved_seed_range_untouched": [v27.RESERVED_SEED_START, v27.RESERVED_SEED_END],
        "sealed_final_seed_range_untouched": [v27.FINAL_SEED_START, v27.FINAL_SEED_END],
        "truth_boundary": "all unscored cells for an episode are persisted before truth_labels.npz is opened",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "runtime_environment": _runtime_environment(),
    }
    contract["contract_sha256"] = _json_hash(contract)
    return contract


def _prepare(settings: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    output = settings["output"]
    if output.exists() and not settings["resume"]:
        raise FileExistsError(f"V27 output already exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "control" / "source_snapshot").mkdir(parents=True, exist_ok=True)
    (output / "unscored").mkdir(parents=True, exist_ok=True)
    (output / "episode_results").mkdir(parents=True, exist_ok=True)
    contract_path = output / "control" / "campaign_contract.json"
    if contract_path.exists():
        existing = _read_json(contract_path)
        if existing != _json_safe(contract):
            raise RuntimeError("resume contract differs from frozen V27 contract")
    else:
        _write_json_atomic(contract_path, contract)
        for relative in SOURCE_FILES:
            source = settings["root"] / relative
            destination = output / "control" / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        _write_json_atomic(
            output / "control" / "source_manifest.json",
            {
                "files": contract["source_manifest"],
                "manifest_sha256": _json_hash(contract["source_manifest"]),
            },
        )


def _cell_stem(episode_index: int, prefix_s: float, arm: str) -> str:
    return f"episode_{int(episode_index):04d}_prefix_{int(round(prefix_s)):04d}_{arm}"


def _unscored_path(output: Path, episode_index: int, prefix_s: float, arm: str) -> Path:
    return output / "unscored" / f"{_cell_stem(episode_index, prefix_s, arm)}.json"


def _scored_path(output: Path, episode_index: int, prefix_s: float, arm: str) -> Path:
    return output / "episode_results" / f"{_cell_stem(episode_index, prefix_s, arm)}.json"


def _write_or_validate_unscored(
    path: Path,
    wrapper: Mapping[str, Any],
    *,
    resume: bool,
) -> Dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        existing = _read_json(path)
        expected_identity = {
            key: wrapper[key]
            for key in ("contract_sha256", "episode_index", "prefix_s", "arm", "online_inputs_sha256")
        }
        if any(existing.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError(f"unscored resume identity mismatch: {path}")
        return existing
    _write_json_atomic(path, wrapper)
    return dict(wrapper)


def _percentile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(np.percentile(np.asarray(finite), q))


def _nearest_rank(values: Iterable[Optional[float]], probability: float) -> Optional[float]:
    finite = sorted(
        float(value) for value in values if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return None
    rank = max(1, int(math.ceil(float(probability) * len(finite))))
    return finite[min(rank - 1, len(finite) - 1)]


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


def _bootstrap_mean_ci(differences: np.ndarray) -> Dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap requires finite one-dimensional differences")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    means: List[np.ndarray] = []
    remaining = BOOTSTRAP_RESAMPLES
    while remaining > 0:
        chunk = min(5000, remaining)
        indices = rng.integers(0, values.size, size=(chunk, values.size), endpoint=False)
        means.append(np.mean(values[indices], axis=1))
        remaining -= chunk
    samples = np.concatenate(means)
    return {
        "mean": float(np.mean(values)),
        "lower95": float(np.percentile(samples, 2.5)),
        "upper95": float(np.percentile(samples, 97.5)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], smoke: bool) -> Dict[str, Any]:
    expected = len(v27.ARM_NAMES) * (len(SMOKE_PREFIXES_S) if smoke else len(FULL_PREFIXES_S)) * (
        len(SMOKE_EPISODE_INDICES) if smoke else len(FULL_EPISODE_INDICES)
    )
    if len(rows) != expected:
        raise RuntimeError(f"V27 expected {expected} scored cells, found {len(rows)}")
    by_arm_prefix: Dict[str, Any] = {}
    for arm in v27.ARM_NAMES:
        for prefix in (SMOKE_PREFIXES_S if smoke else FULL_PREFIXES_S):
            subset = [
                row
                for row in rows
                if row["arm"] == arm
                and math.isclose(float(row["prefix_s"]), float(prefix), rel_tol=0.0, abs_tol=1e-12)
            ]
            coverage_values = [
                row["score"]["nominal_radius95_covers"]
                for row in subset
                if row["score"]["nominal_radius95_covers"] is not None
            ]
            by_arm_prefix[f"{arm}@{int(prefix)}"] = {
                "arm": arm,
                "prefix_s": float(prefix),
                "count": len(subset),
                "success_lt_7m_count": int(sum(bool(row["score"]["success_lt_7m"]) for row in subset)),
                "success_le_7m_count": int(sum(bool(row["score"]["success_le_7m"]) for row in subset)),
                "endpoint_error_m": _stats(row["score"]["endpoint_position_error_m"] for row in subset),
                "runtime_s": _stats(row["unscored"]["runtime_s"] for row in subset),
                "runtime_p99_nearest_rank_s": _nearest_rank(
                    (row["unscored"]["runtime_s"] for row in subset), 0.99
                ),
                "nominal_radius95_m": _stats(row["unscored"]["nominal_radius95_m"] for row in subset),
                "coverage_defined_count": len(coverage_values),
                "coverage_count": int(sum(bool(value) for value in coverage_values)),
                "coverage_rate": (
                    None if not coverage_values else float(np.mean(np.asarray(coverage_values, dtype=np.float64)))
                ),
            }
    paired: Dict[str, Any] = {}
    prefixes = SMOKE_PREFIXES_S if smoke else FULL_PREFIXES_S
    for prefix in prefixes:
        primary = {
            int(row["episode_index"]): float(row["score"]["endpoint_position_error_m"])
            for row in rows
            if row["arm"] == v27.GLOBAL_FULL
            and math.isclose(float(row["prefix_s"]), float(prefix), rel_tol=0.0, abs_tol=1e-12)
        }
        for arm in v27.ARM_NAMES:
            if arm == v27.GLOBAL_FULL:
                continue
            comparator = {
                int(row["episode_index"]): float(row["score"]["endpoint_position_error_m"])
                for row in rows
                if row["arm"] == arm
                and math.isclose(float(row["prefix_s"]), float(prefix), rel_tol=0.0, abs_tol=1e-12)
            }
            if set(primary) != set(comparator):
                raise RuntimeError("paired V27 episode sets differ")
            # Positive means global_full has lower endpoint error.
            differences = np.asarray(
                [comparator[index] - primary[index] for index in sorted(primary)],
                dtype=np.float64,
            )
            paired[f"global_full_vs_{arm}@{int(prefix)}"] = _bootstrap_mean_ci(differences)
    return {"cell_count": len(rows), "by_arm_prefix": by_arm_prefix, "paired_error_gain_m": paired}


def _decision(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    *,
    smoke: bool,
    deterministic_valid: bool,
    independent_audit_valid: bool,
) -> Dict[str, Any]:
    if smoke:
        passed = bool(deterministic_valid and independent_audit_valid)
        return {
            "decision": "SMOKE_PASS" if passed else "SMOKE_FAIL",
            "smoke_only": True,
            "scientific_interpretation_forbidden": True,
            "deterministic_repeat_valid": bool(deterministic_valid),
            "independent_audit_valid": bool(independent_audit_valid),
        }
    summary = aggregate["by_arm_prefix"]
    success_120 = int(summary["global_full@120"]["success_le_7m_count"])
    success_440 = int(summary["global_full@440"]["success_le_7m_count"])
    runtime_p99 = summary["global_full@440"]["runtime_p99_nearest_rank_s"]
    global_by_episode = {
        (int(row["episode_index"]), int(round(float(row["prefix_s"])))): bool(
            row["score"]["success_le_7m"]
        )
        for row in rows
        if row["arm"] == v27.GLOBAL_FULL
    }
    joint_240_440 = sum(
        global_by_episode[(index, 240)] and global_by_episode[(index, 440)]
        for index in FULL_EPISODE_INDICES
    )
    comparator_440 = [
        int(summary[f"{arm}@440"]["success_le_7m_count"])
        for arm in v27.ARM_NAMES
        if arm not in {v27.GLOBAL_FULL, v27.LEGACY_PF}
    ]
    best_comparator = max(comparator_440)
    checks = {
        "integrity_and_audit": bool(independent_audit_valid),
        "global_full_120_success_ge_95": success_120 >= 95,
        "global_full_joint_240_440_success_ge_95": joint_240_440 >= 95,
        "global_full_440_success_ge_99": success_440 >= 99,
        "global_full_440_runtime_p99_lt_2s": runtime_p99 is not None and float(runtime_p99) < 2.0,
        "global_full_not_more_than_1pp_below_best_corrected": success_440 >= best_comparator - 1,
    }
    passed = all(checks.values())
    return {
        "decision": (
            "PROCEED_TO_CLOSED_LOOP_STRESS_DESIGN"
            if passed
            else "REVISE_ESTIMATOR_BEFORE_STRESS"
        ),
        "development_only": True,
        "final_authorized": False,
        "checks": checks,
        "global_full_success_120": success_120,
        "global_full_joint_success_240_440": int(joint_240_440),
        "global_full_success_440": success_440,
        "best_corrected_comparator_success_440": int(best_comparator),
        "global_full_runtime_p99_440_s": runtime_p99,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    settings = _settings(args)
    contract = _build_contract(settings)
    _prepare(settings, contract)
    output: Path = settings["output"]
    started = time.perf_counter()
    _write_json_atomic(
        output / "run_state.json",
        {
            "status": "running",
            "started_at_utc": _utc_now(),
            "contract_sha256": contract["contract_sha256"],
            "smoke": settings["smoke"],
        },
    )
    source_by_index = {
        int(row["episode_index"]): row for row in contract["source_episodes"]
    }
    deterministic_checks: List[Dict[str, Any]] = []
    scored_rows: List[Dict[str, Any]] = []
    total_episodes = len(settings["episode_indices"])
    for position, episode_index in enumerate(settings["episode_indices"], start=1):
        source = source_by_index[int(episode_index)]
        history_full = v19.load_online_history(Path(source["online_path"]))
        # Phase A: no truth archive is opened in this loop.
        episode_unscored: Dict[Tuple[float, str], Dict[str, Any]] = {}
        for prefix_s in settings["prefixes_s"]:
            history = history_full.prefix(float(prefix_s))
            for arm in v27.ARM_NAMES:
                path = _unscored_path(output, episode_index, prefix_s, arm)
                if path.exists() and settings["resume"]:
                    wrapper = _write_or_validate_unscored(
                        path,
                        {
                            "contract_sha256": contract["contract_sha256"],
                            "episode_index": int(episode_index),
                            "prefix_s": float(prefix_s),
                            "arm": arm,
                            "online_inputs_sha256": source["online_sha256"],
                        },
                        resume=True,
                    )
                else:
                    estimator = v27.evaluate_arm(
                        arm,
                        history,
                        source["support_radius_min_m"],
                        source["support_radius_max_m"],
                        settings["config"],
                        legacy_trace_path=Path(source["trace_path"]),
                        legacy_trace_sha256=source["trace_sha256"],
                    )
                    wrapper = {
                        "schema_version": 1,
                        "contract_sha256": contract["contract_sha256"],
                        "episode_index": int(episode_index),
                        "prefix_s": float(prefix_s),
                        "arm": arm,
                        "online_inputs_sha256": source["online_sha256"],
                        "payload": estimator.to_unscored_dict(),
                    }
                    _write_or_validate_unscored(path, wrapper, resume=False)
                if "payload" not in wrapper:
                    raise RuntimeError(f"unscored payload is missing: {path}")
                episode_unscored[(float(prefix_s), arm)] = wrapper

        if settings["smoke"] and int(episode_index) == int(SMOKE_EPISODE_INDICES[0]):
            prefix_s = float(SMOKE_PREFIXES_S[0])
            history = history_full.prefix(prefix_s)
            for arm in v27.ARM_NAMES:
                repeated = v27.evaluate_arm(
                    arm,
                    history,
                    source["support_radius_min_m"],
                    source["support_radius_max_m"],
                    settings["config"],
                    legacy_trace_path=Path(source["trace_path"]),
                    legacy_trace_sha256=source["trace_sha256"],
                )
                original = v27.output_from_unscored_dict(
                    episode_unscored[(prefix_s, arm)]["payload"]
                )
                valid = v27.deterministic_payload(original) == v27.deterministic_payload(repeated)
                deterministic_checks.append({"arm": arm, "prefix_s": prefix_s, "valid": bool(valid)})
                if not valid:
                    raise RuntimeError(f"deterministic repeat failed for {arm}")

        # Phase B: only now may diagnostic truth be opened for this episode.
        truth = v19.load_truth_diagnostics(Path(source["truth_path"]))
        for prefix_s in settings["prefixes_s"]:
            history = history_full.prefix(float(prefix_s))
            for arm in v27.ARM_NAMES:
                unscored = episode_unscored[(float(prefix_s), arm)]
                estimator = v27.output_from_unscored_dict(unscored["payload"])
                score = v27.score_output(estimator, truth, history)
                row = {
                    "schema_version": 1,
                    "contract_sha256": contract["contract_sha256"],
                    "episode_index": int(episode_index),
                    "episode_seed": int(source["episode_seed"]),
                    "prefix_s": float(prefix_s),
                    "arm": arm,
                    "online_inputs_sha256": source["online_sha256"],
                    "truth_labels_sha256": source["truth_sha256"],
                    "unscored_sha256": v27.sha256_file(
                        _unscored_path(output, episode_index, prefix_s, arm)
                    ),
                    "unscored": unscored["payload"],
                    "score": score,
                }
                scored_path = _scored_path(output, episode_index, prefix_s, arm)
                if scored_path.exists() and settings["resume"]:
                    existing = _read_json(scored_path)
                    if existing != _json_safe(row):
                        raise RuntimeError(f"scored resume mismatch: {scored_path}")
                else:
                    _write_json_atomic(scored_path, row)
                scored_rows.append(row)
        if position % settings["progress_every"] == 0 or position == total_episodes:
            print(
                f"V27 {'smoke' if settings['smoke'] else 'dev'} "
                f"episode {position}/{total_episodes} complete",
                flush=True,
            )

    aggregate = _aggregate(scored_rows, bool(settings["smoke"]))
    preliminary = {
        "schema_version": 1,
        "status": "complete_pre_audit",
        "runner_version": RUNNER_VERSION,
        "evaluator_version": v27.VERSION,
        "contract_sha256": contract["contract_sha256"],
        "smoke": bool(settings["smoke"]),
        "development_only": True,
        "no_rl_training": True,
        "episode_count": len(settings["episode_indices"]),
        "cell_count": len(scored_rows),
        "deterministic_checks": deterministic_checks,
        "aggregate": aggregate,
        "elapsed_wall_s": float(time.perf_counter() - started),
        "completed_at_utc": _utc_now(),
    }
    _write_json_atomic(output / "campaign_summary.json", preliminary)

    import audit_v27_campaign

    audit = audit_v27_campaign.audit_campaign(output)
    _write_json_atomic(output / "independent_audit.json", audit)
    audit_valid = bool(audit.get("valid", False))
    deterministic_valid = all(bool(row["valid"]) for row in deterministic_checks) if settings["smoke"] else True
    decision = _decision(
        scored_rows,
        aggregate,
        smoke=bool(settings["smoke"]),
        deterministic_valid=deterministic_valid,
        independent_audit_valid=audit_valid,
    )
    final_summary = dict(preliminary)
    final_summary.update(
        {
            "status": "complete" if audit_valid else "audit_failed",
            "integrity_valid": audit_valid,
            "independent_audit": audit,
            "decision": decision["decision"],
        }
    )
    _write_json_atomic(output / "campaign_summary.json", final_summary)
    _write_json_atomic(
        output / "decision.json",
        {
            "contract_sha256": contract["contract_sha256"],
            **decision,
        },
    )
    _write_json_atomic(
        output / "run_state.json",
        {
            "status": final_summary["status"],
            "completed_at_utc": _utc_now(),
            "elapsed_wall_s": float(time.perf_counter() - started),
            "contract_sha256": contract["contract_sha256"],
            "decision": decision["decision"],
        },
    )
    print(json.dumps(_json_safe(decision), indent=2, sort_keys=True), flush=True)
    return 0 if audit_valid and decision["decision"] != "SMOKE_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
