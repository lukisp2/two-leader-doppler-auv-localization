#!/usr/bin/env python3
"""Run the frozen append-only V34 arrival-cost MHE-60 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import uuv_v19_observability as v19
import uuv_v34_mhe60_baseline as v34


RUNNER_VERSION = "v34_mhe60_baseline_runner_1.0"
PROTOCOL_NAME = "EXPERIMENT_PROTOCOL_V34_MHE60_BASELINE.md"
SMOKE_EPISODE_INDICES = (0, 73)
FULL_EPISODE_INDICES = tuple(range(100))
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 34_045_000

SOURCE_FILES = (
    "run_v34_mhe60_baseline.py",
    "uuv_v34_mhe60_baseline.py",
    "audit_v34_mhe60_campaign.py",
    PROTOCOL_NAME,
    "tests/test_uuv_v34_mhe60_baseline.py",
    "uuv_v19_observability.py",
    "uuv_v27_publication_baselines.py",
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
    parser = argparse.ArgumentParser(description="Run frozen V34 MHE-60 baseline")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--reference-v27-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def _default_archive(root: Path) -> Path:
    return root / "experiments_v19_observability_estimator_benchmark_dev100" / "measurement_archive"


def _default_reference(root: Path) -> Path:
    return root / "experiments_v27_publication_baselines_dev100"


def _default_output(root: Path, smoke: bool) -> Path:
    suffix = "smoke" if smoke else "dev100"
    return root / f"experiments_v34_mhe60_baseline_{suffix}"


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    smoke = bool(args.smoke)
    archive = (args.archive_dir or _default_archive(root)).expanduser().resolve()
    reference = (args.reference_v27_dir or _default_reference(root)).expanduser().resolve()
    output = (args.output_dir or _default_output(root, smoke)).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"V19 measurement archive is missing: {archive}")
    if not reference.is_dir():
        raise FileNotFoundError(f"V27 reference campaign is missing: {reference}")
    if any(output == item or output in item.parents or item in output.parents for item in (archive, reference)):
        raise ValueError("V34 output, source archive and V27 reference must be disjoint")
    if int(args.progress_every) < 1:
        raise ValueError("progress-every must be positive")
    if smoke:
        config = v34.MHEConfig(
            coarse_candidates=512,
            coarse_sweeps=1,
            local_starts=8,
            maximum_modes=8,
        )
        episode_indices = SMOKE_EPISODE_INDICES
    else:
        config = v34.MHEConfig()
        episode_indices = FULL_EPISODE_INDICES
    return {
        "root": root,
        "archive": archive,
        "reference": reference,
        "output": output,
        "smoke": smoke,
        "resume": bool(args.resume),
        "progress_every": int(args.progress_every),
        "config": config,
        "episode_indices": episode_indices,
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
        raise RuntimeError("V34 source episode index mismatch")
    if seed != v19.V181_DEV_SEED_START + int(episode_index):
        raise RuntimeError("V34 source seed mismatch")
    v34.assert_v34_development_seed(seed)
    if metadata.get("controller") != v19.PRIMARY_CONTROLLER:
        raise RuntimeError("V34 requires the frozen deterministic-greedy source")
    if not bool(metadata.get("replay_integrity", {}).get("passed", False)):
        raise RuntimeError("V19 replay integrity did not pass")
    if v34.sha256_file(online_path) != str(metadata.get("online_inputs_sha256")):
        raise RuntimeError("V34 online-input hash mismatch")
    if v34.sha256_file(truth_path) != str(metadata.get("truth_labels_sha256")):
        raise RuntimeError("V34 truth-label hash mismatch")
    return {
        "episode_index": int(episode_index),
        "episode_seed": seed,
        "directory": str(directory.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": v34.sha256_file(metadata_path),
        "online_path": str(online_path.resolve()),
        "online_sha256": str(metadata["online_inputs_sha256"]),
        "truth_path": str(truth_path.resolve()),
        "truth_sha256": str(metadata["truth_labels_sha256"]),
        "support_radius_min_m": float(metadata["support_radius_min_m"]),
        "support_radius_max_m": float(metadata["support_radius_max_m"]),
    }


def _reference_path(reference: Path, episode_index: int, checkpoint_s: float) -> Path:
    return reference / "episode_results" / (
        f"episode_{int(episode_index):04d}_prefix_{int(round(checkpoint_s)):04d}_global_full.json"
    )


def _reference_manifest(reference: Path, episode_indices: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for episode_index in episode_indices:
        for checkpoint_s in v34.CHECKPOINTS_S:
            path = _reference_path(reference, int(episode_index), float(checkpoint_s))
            if not path.is_file():
                raise FileNotFoundError(f"V27 global_full reference is missing: {path}")
            rows.append(
                {
                    "episode_index": int(episode_index),
                    "checkpoint_s": float(checkpoint_s),
                    "path": str(path.resolve()),
                    "sha256": v34.sha256_file(path),
                }
            )
    return rows


def _source_manifest(root: Path) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"V34 source file is missing: {source}")
        manifest[relative] = v34.sha256_file(source)
    return manifest


def _smoke_prerequisite(root: Path, smoke: bool) -> Optional[Dict[str, Any]]:
    if smoke:
        return None
    directory = root / "experiments_v34_mhe60_baseline_smoke"
    decision_path = directory / "decision.json"
    audit_path = directory / "independent_audit.json"
    if not decision_path.is_file() or not audit_path.is_file():
        raise RuntimeError("V34 full run requires a completed audited smoke campaign")
    decision = _read_json(decision_path)
    audit = _read_json(audit_path)
    if decision.get("decision") != "SMOKE_PASS" or not bool(audit.get("valid", False)):
        raise RuntimeError("V34 full run refuses a failed smoke prerequisite")
    return {
        "directory": str(directory.resolve()),
        "decision_sha256": v34.sha256_file(decision_path),
        "audit_sha256": v34.sha256_file(audit_path),
        "contract_sha256": str(decision.get("contract_sha256", "")),
    }


def _build_contract(settings: Mapping[str, Any]) -> Dict[str, Any]:
    sources = [
        _validate_source_episode(
            _episode_directory(settings["archive"], int(index)), int(index)
        )
        for index in settings["episode_indices"]
    ]
    reference_summary = settings["reference"] / "campaign_summary.json"
    reference_decision = settings["reference"] / "decision.json"
    reference_contract = settings["reference"] / "control" / "campaign_contract.json"
    for path in (reference_summary, reference_decision, reference_contract):
        if not path.is_file():
            raise FileNotFoundError(f"V27 reference control artifact is missing: {path}")
    contract = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "evaluator_version": v34.VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": v34.sha256_file(settings["root"] / PROTOCOL_NAME),
        "development_only": True,
        "append_only_v27_unchanged": True,
        "no_rl_training": True,
        "smoke": bool(settings["smoke"]),
        "source_archive": str(settings["archive"]),
        "episode_indices": list(settings["episode_indices"]),
        "episode_seeds": [row["episode_seed"] for row in sources],
        "checkpoints_s": list(v34.CHECKPOINTS_S),
        "arm": v34.ARM_NAME,
        "evaluator_config": v34.config_to_dict(settings["config"]),
        "source_episodes": sources,
        "source_manifest": _source_manifest(settings["root"]),
        "v27_reference": {
            "directory": str(settings["reference"]),
            "campaign_summary_sha256": v34.sha256_file(reference_summary),
            "decision_sha256": v34.sha256_file(reference_decision),
            "campaign_contract_sha256": v34.sha256_file(reference_contract),
            "global_full_cells": _reference_manifest(
                settings["reference"], settings["episode_indices"]
            ),
        },
        "smoke_prerequisite": _smoke_prerequisite(
            settings["root"], bool(settings["smoke"])
        ),
        "reserved_seed_range_untouched": [
            v34.RESERVED_SEED_START,
            v34.RESERVED_SEED_END,
        ],
        "sealed_final_seed_range_untouched": [
            v34.FINAL_SEED_START,
            v34.FINAL_SEED_END,
        ],
        "truth_boundary": "all five unscored MHE outputs for an episode are persisted before truth_labels.npz is loaded",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "runtime_environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pid": os.getpid(),
        },
    }
    contract["contract_sha256"] = _json_hash(contract)
    return contract


def _prepare(settings: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    output: Path = settings["output"]
    if output.exists() and not settings["resume"]:
        raise FileExistsError(f"V34 output already exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "control" / "source_snapshot").mkdir(parents=True, exist_ok=True)
    (output / "unscored").mkdir(parents=True, exist_ok=True)
    (output / "episode_results").mkdir(parents=True, exist_ok=True)
    contract_path = output / "control" / "campaign_contract.json"
    if contract_path.exists():
        existing = _read_json(contract_path)
        if existing != _json_safe(contract):
            raise RuntimeError("resume contract differs from frozen V34 contract")
        return
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


def _run_unit_tests(root: Path, output: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "tests.test_uuv_v34_mhe60_baseline"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    report = {
        "command": [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "tests.test_uuv_v34_mhe60_baseline",
        ],
        "returncode": int(completed.returncode),
        "passed": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_wall_s": float(time.perf_counter() - started),
    }
    _write_json_atomic(output / "unit_test_report.json", report)
    if completed.returncode != 0:
        raise RuntimeError("V34 unit tests failed; full execution is forbidden")
    return report


def _stem(episode_index: int, checkpoint_s: float) -> str:
    return f"episode_{int(episode_index):04d}_checkpoint_{int(round(checkpoint_s)):04d}_{v34.ARM_NAME}"


def _unscored_path(output: Path, episode_index: int, checkpoint_s: float) -> Path:
    return output / "unscored" / f"{_stem(episode_index, checkpoint_s)}.json"


def _scored_path(output: Path, episode_index: int, checkpoint_s: float) -> Path:
    return output / "episode_results" / f"{_stem(episode_index, checkpoint_s)}.json"


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


def _bootstrap_mean_ci(differences: np.ndarray) -> Dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("V34 paired bootstrap requires finite one-dimensional values")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    samples: List[np.ndarray] = []
    remaining = BOOTSTRAP_RESAMPLES
    while remaining > 0:
        chunk = min(5000, remaining)
        indices = rng.integers(0, values.size, size=(chunk, values.size), endpoint=False)
        samples.append(np.mean(values[indices], axis=1))
        remaining -= chunk
    distribution = np.concatenate(samples)
    return {
        "mean": float(np.mean(values)),
        "lower95": float(np.percentile(distribution, 2.5)),
        "upper95": float(np.percentile(distribution, 97.5)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _load_reference_rows(contract: Mapping[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for reference in contract["v27_reference"]["global_full_cells"]:
        path = Path(str(reference["path"]))
        if v34.sha256_file(path) != str(reference["sha256"]):
            raise RuntimeError(f"V27 reference cell changed: {path}")
        row = _read_json(path)
        if row.get("arm") != "global_full":
            raise RuntimeError("V34 reference cell is not V27 global_full")
        key = (int(reference["episode_index"]), int(round(float(reference["checkpoint_s"]))))
        if int(row.get("episode_index", -1)) != key[0] or int(round(float(row.get("prefix_s", -1)))) != key[1]:
            raise RuntimeError("V27 reference identity mismatch")
        rows[key] = row
    return rows


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = len(contract["episode_indices"]) * len(v34.CHECKPOINTS_S)
    if len(rows) != expected:
        raise RuntimeError(f"V34 expected {expected} scored cells, found {len(rows)}")
    references = _load_reference_rows(contract)
    by_checkpoint: Dict[str, Any] = {}
    paired: Dict[str, Any] = {}
    for checkpoint in v34.CHECKPOINTS_S:
        endpoint = int(round(checkpoint))
        subset = [row for row in rows if int(round(float(row["checkpoint_s"]))) == endpoint]
        coverage = [
            row["score"]["nominal_radius95_covers"]
            for row in subset
            if row["score"]["nominal_radius95_covers"] is not None
        ]
        reference_success = sum(
            bool(references[(int(row["episode_index"]), endpoint)]["score"]["success_le_7m"])
            for row in subset
        )
        differences = np.asarray(
            [
                float(row["score"]["endpoint_position_error_m"])
                - float(references[(int(row["episode_index"]), endpoint)]["score"]["endpoint_position_error_m"])
                for row in sorted(subset, key=lambda value: int(value["episode_index"]))
            ],
            dtype=np.float64,
        )
        by_checkpoint[str(endpoint)] = {
            "checkpoint_s": float(checkpoint),
            "count": len(subset),
            "success_lt_7m_count": int(sum(bool(row["score"]["success_lt_7m"]) for row in subset)),
            "success_le_7m_count": int(sum(bool(row["score"]["success_le_7m"]) for row in subset)),
            "v27_global_full_success_le_7m_count": int(reference_success),
            "false_confidence_count": int(
                sum(bool(row["score"]["false_confidence_radius_lt_7_error_gt_7"]) for row in subset)
            ),
            "endpoint_error_m": _stats(row["score"]["endpoint_position_error_m"] for row in subset),
            "initial_error_m": _stats(row["score"]["initial_position_error_m"] for row in subset),
            "nominal_radius95_m": _stats(row["unscored"]["nominal_radius95_m"] for row in subset),
            "coverage_defined_count": len(coverage),
            "coverage_count": int(sum(bool(value) for value in coverage)),
            "coverage_rate": None if not coverage else float(np.mean(np.asarray(coverage, dtype=np.float64))),
            "checkpoint_update_runtime_s": _stats(
                row["unscored"]["checkpoint_update_runtime_s"] for row in subset
            ),
            "cumulative_runtime_s": _stats(row["unscored"]["cumulative_runtime_s"] for row in subset),
            "converged_count": int(sum(bool(row["unscored"]["diagnostics"]["last_converged"]) for row in subset)),
        }
        paired[str(endpoint)] = _bootstrap_mean_ci(differences)
    final_rows = [
        row for row in rows if int(round(float(row["checkpoint_s"]))) == 440
    ]
    all_post_init_latencies = [
        float(latency)
        for row in final_rows
        for latency in row["unscored"]["post_init_update_latencies_s"]
    ]
    return {
        "cell_count": len(rows),
        "by_checkpoint": by_checkpoint,
        "paired_mhe_minus_v27_global_full_error_m": paired,
        "all_post_init_update_latency_s": {
            **_stats(all_post_init_latencies),
            "p99_nearest_rank": _nearest_rank(all_post_init_latencies, 0.99),
            "update_count": len(all_post_init_latencies),
        },
    }


def _decision(
    aggregate: Mapping[str, Any],
    *,
    smoke: bool,
    deterministic_valid: bool,
    audit_valid: bool,
    unit_tests_valid: bool,
) -> Dict[str, Any]:
    if smoke:
        passed = bool(deterministic_valid and audit_valid and unit_tests_valid)
        return {
            "decision": "SMOKE_PASS" if passed else "SMOKE_FAIL",
            "smoke_only": True,
            "scientific_interpretation_forbidden": True,
            "deterministic_repeat_valid": bool(deterministic_valid),
            "independent_audit_valid": bool(audit_valid),
            "unit_tests_valid": bool(unit_tests_valid),
        }
    by_checkpoint = aggregate["by_checkpoint"]
    late = (120, 240, 440)
    success_checks = {
        str(checkpoint): int(by_checkpoint[str(checkpoint)]["success_le_7m_count"]) >= 99
        for checkpoint in late
    }
    reference_checks = {
        str(checkpoint): int(by_checkpoint[str(checkpoint)]["success_le_7m_count"])
        >= int(by_checkpoint[str(checkpoint)]["v27_global_full_success_le_7m_count"]) - 1
        for checkpoint in late
    }
    false_confidence_checks = {
        str(checkpoint): int(by_checkpoint[str(checkpoint)]["false_confidence_count"]) <= 1
        for checkpoint in late
    }
    paired_440 = aggregate["paired_mhe_minus_v27_global_full_error_m"]["440"]
    latency_p99 = aggregate["all_post_init_update_latency_s"]["p99_nearest_rank"]
    checks = {
        "integrity_audit_and_tests": bool(audit_valid and unit_tests_valid),
        "all_500_cells_finite": int(aggregate["cell_count"]) == 500,
        "success_ge_99_each_late_checkpoint": all(success_checks.values()),
        "not_more_than_one_success_below_v27_each_late_checkpoint": all(reference_checks.values()),
        "paired_mean_excess_error_upper95_le_0p5m_at_440": float(paired_440["upper95"]) <= 0.5,
        "all_update_latency_p99_lt_1s": latency_p99 is not None and float(latency_p99) < 1.0,
        "false_confidence_le_1_each_late_checkpoint": all(false_confidence_checks.values()),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "MHE60_COMPETITIVE_NOMINAL_BASELINE"
            if passed
            else "MHE60_NOT_NONINFERIOR"
        ),
        "development_only": True,
        "final_authorized": False,
        "checks": checks,
        "late_success_checks": success_checks,
        "late_reference_success_checks": reference_checks,
        "late_false_confidence_checks": false_confidence_checks,
        "paired_mean_excess_error_440_m": paired_440,
        "all_update_latency_p99_s": latency_p99,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    settings = _settings(args)
    contract = _build_contract(settings)
    _prepare(settings, contract)
    output: Path = settings["output"]
    unit_tests = _run_unit_tests(settings["root"], output)
    started = time.perf_counter()
    _write_json_atomic(
        output / "run_state.json",
        {
            "status": "running",
            "started_at_utc": _utc_now(),
            "contract_sha256": contract["contract_sha256"],
            "smoke": bool(settings["smoke"]),
        },
    )
    source_by_index = {
        int(row["episode_index"]): row for row in contract["source_episodes"]
    }
    deterministic_checks: List[Dict[str, Any]] = []
    scored_rows: List[Dict[str, Any]] = []
    for ordinal, episode_index in enumerate(settings["episode_indices"], start=1):
        source = source_by_index[int(episode_index)]
        history = v19.load_online_history(Path(source["online_path"]))
        # Phase A: estimate and persist every checkpoint before loading truth.
        outputs = v34.evaluate_mhe_history(
            history,
            float(source["support_radius_min_m"]),
            float(source["support_radius_max_m"]),
            settings["config"],
        )
        episode_unscored: Dict[float, Dict[str, Any]] = {}
        for estimator_output in outputs:
            checkpoint = float(estimator_output.checkpoint_s)
            wrapper = {
                "schema_version": 1,
                "contract_sha256": contract["contract_sha256"],
                "episode_index": int(episode_index),
                "checkpoint_s": checkpoint,
                "arm": v34.ARM_NAME,
                "online_inputs_sha256": source["online_sha256"],
                "payload": estimator_output.to_unscored_dict(),
            }
            path = _unscored_path(output, int(episode_index), checkpoint)
            if path.exists() and settings["resume"]:
                existing = _read_json(path)
                if any(
                    existing.get(key) != wrapper.get(key)
                    for key in (
                        "contract_sha256",
                        "episode_index",
                        "checkpoint_s",
                        "arm",
                        "online_inputs_sha256",
                    )
                ):
                    raise RuntimeError(f"V34 unscored resume identity mismatch: {path}")
                persisted_output = v34.output_from_unscored_dict(existing["payload"])
                if v34.deterministic_payload(persisted_output) != v34.deterministic_payload(estimator_output):
                    raise RuntimeError(f"V34 unscored resume payload mismatch: {path}")
                wrapper = existing
            else:
                _write_json_atomic(path, wrapper)
            episode_unscored[checkpoint] = wrapper

        if bool(settings["smoke"]) and int(episode_index) == int(SMOKE_EPISODE_INDICES[0]):
            repeated = v34.evaluate_mhe_history(
                history,
                float(source["support_radius_min_m"]),
                float(source["support_radius_max_m"]),
                settings["config"],
            )
            for repeated_output in repeated:
                checkpoint = float(repeated_output.checkpoint_s)
                original = v34.output_from_unscored_dict(
                    episode_unscored[checkpoint]["payload"]
                )
                valid = v34.deterministic_payload(original) == v34.deterministic_payload(repeated_output)
                deterministic_checks.append(
                    {"checkpoint_s": checkpoint, "valid": bool(valid)}
                )
                if not valid:
                    raise RuntimeError(f"V34 deterministic repeat failed at {checkpoint:g} s")

        # Phase B: only after all five unscored files exist may truth be loaded.
        truth = v19.load_truth_diagnostics(Path(source["truth_path"]))
        for checkpoint in v34.CHECKPOINTS_S:
            wrapper = episode_unscored[float(checkpoint)]
            estimator_output = v34.output_from_unscored_dict(wrapper["payload"])
            score = v34.score_output(estimator_output, truth, history)
            row = {
                "schema_version": 1,
                "contract_sha256": contract["contract_sha256"],
                "episode_index": int(episode_index),
                "episode_seed": int(source["episode_seed"]),
                "checkpoint_s": float(checkpoint),
                "arm": v34.ARM_NAME,
                "online_inputs_sha256": source["online_sha256"],
                "truth_labels_sha256": source["truth_sha256"],
                "unscored_sha256": v34.sha256_file(
                    _unscored_path(output, int(episode_index), float(checkpoint))
                ),
                "unscored": wrapper["payload"],
                "score": score,
            }
            scored_path = _scored_path(output, int(episode_index), float(checkpoint))
            if scored_path.exists() and settings["resume"]:
                existing = _read_json(scored_path)
                if existing != _json_safe(row):
                    raise RuntimeError(f"V34 scored resume mismatch: {scored_path}")
            else:
                _write_json_atomic(scored_path, row)
            scored_rows.append(row)
        if ordinal % int(settings["progress_every"]) == 0 or ordinal == len(settings["episode_indices"]):
            print(
                f"V34 {'smoke' if settings['smoke'] else 'dev'} episode "
                f"{ordinal}/{len(settings['episode_indices'])} complete",
                flush=True,
            )

    aggregate = _aggregate(scored_rows, contract)
    preliminary = {
        "schema_version": 1,
        "status": "complete_pre_audit",
        "runner_version": RUNNER_VERSION,
        "evaluator_version": v34.VERSION,
        "contract_sha256": contract["contract_sha256"],
        "smoke": bool(settings["smoke"]),
        "development_only": True,
        "append_only_v27_unchanged": True,
        "no_rl_training": True,
        "episode_count": len(settings["episode_indices"]),
        "cell_count": len(scored_rows),
        "deterministic_checks": deterministic_checks,
        "unit_tests": unit_tests,
        "aggregate": aggregate,
        "elapsed_wall_s": float(time.perf_counter() - started),
        "completed_at_utc": _utc_now(),
    }
    _write_json_atomic(output / "campaign_summary.json", preliminary)

    import audit_v34_mhe60_campaign

    audit = audit_v34_mhe60_campaign.audit_campaign(output)
    _write_json_atomic(output / "independent_audit.json", audit)
    audit_valid = bool(audit.get("valid", False))
    deterministic_valid = (
        all(bool(row["valid"]) for row in deterministic_checks)
        if settings["smoke"]
        else True
    )
    decision = _decision(
        aggregate,
        smoke=bool(settings["smoke"]),
        deterministic_valid=deterministic_valid,
        audit_valid=audit_valid,
        unit_tests_valid=bool(unit_tests["passed"]),
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
        {"contract_sha256": contract["contract_sha256"], **decision},
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

