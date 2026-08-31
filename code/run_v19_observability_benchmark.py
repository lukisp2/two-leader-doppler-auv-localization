#!/usr/bin/env python3
"""Run the fail-closed V19 observability/estimator development benchmark.

This launcher never trains a policy and never opens the sealed final seed
range.  It replays the frozen V18.1 deterministic-greedy trajectories,
persists online inputs separately from simulator truth, and evaluates the
recorded-noisy and structural-noiseless initial-position problems at causal
time prefixes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import uuv_v19_observability as v19


RUNNER_VERSION = "v19_observability_benchmark_runner_1.0"
CONTRACT_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
DEFAULT_PREFIXES_S = (30.0, 60.0, 120.0, 240.0, 440.0)
SEARCH_SEED_BASE = 19_001
ACTION_INTERVAL_S = 2.0
SUPPORT_RADIUS_MIN_M = 120.0
SUPPORT_RADIUS_MAX_M = 350.0
STRUCTURAL_ERROR_GATE_M = 1.0
STRUCTURAL_EQUIVALENT_RMSE_GAP_MPS = 1e-6
NOISY_ERROR_GATE_M = 7.0

SOURCE_SNAPSHOT_NAMES = (
    "run_v19_observability_benchmark.py",
    "uuv_v19_observability.py",
    "EXPERIMENT_PROTOCOL_V19_OBSERVABILITY.md",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
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


def _load_json_object(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    fields: List[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if value is None else value
                    for key, value in _json_safe(dict(row)).items()
                }
            )
    os.replace(temporary, destination)


def _same_json(left: Any, right: Any) -> bool:
    return _json_safe(left) == _json_safe(right)


def _percentile(values: Iterable[Any], q: float) -> Optional[float]:
    finite = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            finite.append(number)
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=np.float64), float(q)))


def _nearest_rank_percentile(values: Iterable[Any], q: float) -> Optional[float]:
    """Return the prespecified finite-sample nearest-rank percentile."""

    finite = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            finite.append(number)
    if not finite:
        return None
    ordered = np.sort(np.asarray(finite, dtype=np.float64))
    rank = max(1, int(math.ceil(float(q) * ordered.size / 100.0)))
    return float(ordered[min(rank - 1, ordered.size - 1)])


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            finite.append(number)
    return None if not finite else float(np.mean(finite))


def _default_evaluation_directory(source_root: Path) -> Path:
    return (
        source_root
        / "experiments_v18_1_guard_ablation_dev_3seed"
        / "evaluations"
        / "dev100_seed_28001_range_45000_45099"
    )


def _default_output_directory(source_root: Path, smoke: bool) -> Path:
    suffix = "smoke" if smoke else "dev100"
    return source_root / f"experiments_v19_observability_estimator_benchmark_{suffix}"


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen V18.1 development episodes and run the V19 "
            "structural/noisy multi-start observability benchmark."
        )
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="frozen V18.1 seed-28001 development evaluation directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new V19 output directory (existing output requires --resume)",
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=int,
        help="number of consecutive development episodes (default: 100; smoke: 2)",
    )
    parser.add_argument(
        "--prefixes",
        type=float,
        nargs="+",
        default=list(DEFAULT_PREFIXES_S),
        metavar="SECONDS",
        help="strictly increasing causal endpoints (default: 30 60 120 240 440)",
    )
    parser.add_argument("--coarse-candidates", type=int)
    parser.add_argument("--coarse-sweeps", type=int)
    parser.add_argument("--local-starts", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use small default episode/search counts; still development-only",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only if the immutable contract and every input hash match",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def _resolved_settings(args: argparse.Namespace) -> Dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    evaluation = (
        args.evaluation_dir
        if args.evaluation_dir is not None
        else _default_evaluation_directory(source_root)
    ).expanduser().resolve()
    output = (
        args.output_dir
        if args.output_dir is not None
        else _default_output_directory(source_root, bool(args.smoke))
    ).expanduser().resolve()
    episodes = int(args.episodes if args.episodes is not None else (2 if args.smoke else 100))
    coarse_candidates = int(
        args.coarse_candidates
        if args.coarse_candidates is not None
        else (512 if args.smoke else 4096)
    )
    coarse_sweeps = int(
        args.coarse_sweeps
        if args.coarse_sweeps is not None
        else (1 if args.smoke else 2)
    )
    local_starts = int(
        args.local_starts
        if args.local_starts is not None
        else (12 if args.smoke else 48)
    )
    prefixes = tuple(float(value) for value in args.prefixes)
    if not evaluation.is_dir():
        raise FileNotFoundError(f"V18.1 evaluation directory is missing: {evaluation}")
    if output == evaluation or evaluation in output.parents or output in evaluation.parents:
        raise ValueError("output and frozen evaluation directories must be disjoint")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    episode_start = int(args.episode_start)
    episode_end = episode_start + episodes - 1
    if episode_start < 0 or episode_end > 99:
        raise ValueError("V19 accepts only V18.1 development episode indices 0..99")
    if not prefixes or any(not math.isfinite(value) for value in prefixes):
        raise ValueError("prefixes must be finite and non-empty")
    if any(value <= 0.0 or value > 440.0 for value in prefixes):
        raise ValueError("prefixes must lie in (0, 440]")
    if any(right <= left for left, right in zip(prefixes, prefixes[1:])):
        raise ValueError("prefixes must be strictly increasing")
    for required in (120.0, 240.0, 440.0):
        if not any(
            math.isclose(value, required, rel_tol=0.0, abs_tol=1e-12)
            for value in prefixes
        ):
            raise ValueError(
                "prefixes must include the sustained-lock checkpoints "
                "120, 240, and 440 s"
            )
    if not math.isclose(prefixes[-1], 440.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the final causal prefix must be 440 s")
    if int(args.progress_every) < 1:
        raise ValueError("progress-every must be positive")
    config = v19.BatchEstimatorConfig(
        coarse_candidates=coarse_candidates,
        coarse_sweeps=coarse_sweeps,
        local_starts=local_starts,
    )
    episode_indices = tuple(range(episode_start, episode_end + 1))
    episode_seeds = tuple(v19.V181_DEV_SEED_START + index for index in episode_indices)
    for seed in episode_seeds:
        v19.assert_seed_is_not_sealed_final(seed)
    return {
        "source_root": source_root,
        "evaluation_directory": evaluation,
        "output_directory": output,
        "episode_indices": episode_indices,
        "episode_seeds": episode_seeds,
        "prefixes_s": prefixes,
        "config": config,
        "smoke": bool(args.smoke),
        "resume": bool(args.resume),
        "progress_every": int(args.progress_every),
    }


def _validate_v181_metadata(evaluation: Path) -> Dict[str, Any]:
    metadata_path = evaluation / "metadata.json"
    metadata = _load_json_object(metadata_path, label="V18.1 metadata")
    if metadata.get("evaluation_mode") != "dev":
        raise RuntimeError("V19 refuses a non-development V18.1 evaluation")
    options = metadata.get("evaluation_options")
    if not isinstance(options, Mapping):
        raise RuntimeError("V18.1 metadata lacks evaluation_options")
    controllers = options.get("controllers")
    if not isinstance(controllers, list) or v19.PRIMARY_CONTROLLER not in controllers:
        raise RuntimeError("V18.1 evaluation lacks deterministic_greedy_grid")
    if int(options.get("episodes", -1)) != 100:
        raise RuntimeError("V19 requires the complete frozen 100-episode dev source")
    seed_protocol = metadata.get("seed_protocol")
    if not isinstance(seed_protocol, Mapping):
        raise RuntimeError("V18.1 metadata lacks seed_protocol")
    expected_seed_fields = {
        "mode": "dev",
        "formula": "episode_seed = 45000 + episode_index",
        "episode_index_start": 0,
        "episode_index_end_inclusive": 99,
    }
    for field, expected in expected_seed_fields.items():
        if seed_protocol.get(field) != expected:
            raise RuntimeError(f"unexpected V18.1 seed protocol field {field!r}")
    v181_contract = metadata.get("v181_contract")
    if not isinstance(v181_contract, Mapping):
        raise RuntimeError("V18.1 metadata lacks its diagnostic contract")
    if v181_contract.get("development_range") != "45000..45099":
        raise RuntimeError("V18.1 development range is not the frozen range")
    if v181_contract.get("final_status") != "sealed and unauthorized":
        raise RuntimeError("V18.1 final-range seal is absent")
    return metadata


def _source_manifest(source_root: Path, evaluation: Path) -> Dict[str, Any]:
    v19_files: Dict[str, Dict[str, str]] = {}
    for name in SOURCE_SNAPSHOT_NAMES:
        path = source_root / name
        if not path.is_file():
            raise FileNotFoundError(f"required V19 source is missing: {path}")
        v19_files[name] = {"path": str(path), "sha256": _sha256_file(path)}

    frozen_dir = evaluation.parents[1] / "control" / "sequential_3seed" / "execution_source"
    metadata = _validate_v181_metadata(evaluation)
    recorded = metadata.get("source_sha256")
    if not isinstance(recorded, Mapping) or not recorded:
        raise RuntimeError("V18.1 metadata lacks a frozen source inventory")
    actual: Dict[str, str] = {}
    for name, expected_hash in sorted(recorded.items()):
        path = frozen_dir / str(name)
        if not path.is_file():
            raise FileNotFoundError(f"frozen V18.1 source is missing: {path}")
        digest = _sha256_file(path)
        if digest != str(expected_hash):
            raise RuntimeError(f"frozen V18.1 source hash mismatch: {name}")
        actual[str(name)] = digest
    return {
        "schema_version": 1,
        "v19_source": v19_files,
        "v181_execution_source_directory": str(frozen_dir.resolve()),
        "v181_frozen_source_sha256": actual,
    }


def _input_manifest(
    evaluation: Path,
    episode_indices: Sequence[int],
    episode_seeds: Sequence[int],
) -> Dict[str, Any]:
    metadata = _validate_v181_metadata(evaluation)
    trace_manifest_path = evaluation / "trace_sha256_manifest.json"
    trace_manifest = _load_json_object(
        trace_manifest_path, label="V18.1 trace SHA-256 manifest"
    )
    trace_entries = trace_manifest.get("files")
    if not isinstance(trace_entries, list):
        raise RuntimeError("V18.1 trace manifest lacks files")
    recorded_trace = {
        (str(item.get("controller")), int(item.get("episode_index", -1))): item
        for item in trace_entries
        if isinstance(item, Mapping)
    }
    tape_entries = metadata.get("noise_tapes")
    if not isinstance(tape_entries, list):
        raise RuntimeError("V18.1 metadata lacks noise tape inventory")
    recorded_tape = {
        int(item.get("episode_index", -1)): item
        for item in tape_entries
        if isinstance(item, Mapping)
    }
    episodes: List[Dict[str, Any]] = []
    for index, seed in zip(episode_indices, episode_seeds):
        v19.assert_seed_is_not_sealed_final(int(seed))
        trace_item = recorded_trace.get((v19.PRIMARY_CONTROLLER, int(index)))
        tape_item = recorded_tape.get(int(index))
        if trace_item is None or tape_item is None:
            raise RuntimeError(f"frozen input manifest lacks episode {index}")
        trace_path = evaluation / str(trace_item["relative_path"])
        tape_path = evaluation / str(tape_item["path"])
        if not trace_path.is_file() or not tape_path.is_file():
            raise FileNotFoundError(f"frozen inputs are missing for episode {index}")
        trace_hash = _sha256_file(trace_path)
        tape_file_hash = _sha256_file(tape_path)
        with np.load(tape_path, allow_pickle=False) as tape_archive:
            if "content_sha256" not in tape_archive.files:
                raise RuntimeError(f"noise tape lacks content digest for episode {index}")
            tape_content_hash = str(tape_archive["content_sha256"].item())
        if trace_hash != str(trace_item["sha256"]):
            raise RuntimeError(f"recorded trace hash mismatch for episode {index}")
        if tape_content_hash != str(tape_item["sha256"]):
            raise RuntimeError(f"recorded tape content hash mismatch for episode {index}")
        episodes.append(
            {
                "episode_index": int(index),
                "episode_seed": int(seed),
                "trace_path": str(trace_path.resolve()),
                "trace_sha256": trace_hash,
                "noise_tape_path": str(tape_path.resolve()),
                "noise_tape_sha256": tape_file_hash,
                "noise_tape_content_sha256": tape_content_hash,
            }
        )
    return {
        "schema_version": 1,
        "evaluation_directory": str(evaluation.resolve()),
        "evaluation_metadata_path": str((evaluation / "metadata.json").resolve()),
        "evaluation_metadata_sha256": _sha256_file(evaluation / "metadata.json"),
        "trace_manifest_path": str(trace_manifest_path.resolve()),
        "trace_manifest_sha256": _sha256_file(trace_manifest_path),
        "controller": v19.PRIMARY_CONTROLLER,
        "episodes": episodes,
    }


def _search_seed_map(prefixes: Sequence[float]) -> Dict[str, Dict[str, int]]:
    return {
        f"{prefix:g}": {
            "recorded_noisy": SEARCH_SEED_BASE + 2 * index,
            "structural_noiseless": SEARCH_SEED_BASE + 2 * index + 1,
        }
        for index, prefix in enumerate(prefixes)
    }


def _campaign_contract(
    settings: Mapping[str, Any],
    source_manifest_sha256: str,
    input_manifest_sha256: str,
    support_radius_min_m: float,
    support_radius_max_m: float,
) -> Dict[str, Any]:
    prefixes = tuple(settings["prefixes_s"])
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "estimator_version": v19.VERSION,
        "benchmark_kind": "development_observability_estimator_diagnostic",
        "training_enabled": False,
        "controller": v19.PRIMARY_CONTROLLER,
        "source_root": str(settings["source_root"]),
        "evaluation_directory": str(settings["evaluation_directory"]),
        "output_directory": str(settings["output_directory"]),
        "episode_indices": list(settings["episode_indices"]),
        "episode_seeds": list(settings["episode_seeds"]),
        "development_range": "45000..45099",
        "sealed_final_range": "50000..50999",
        "sealed_final_status": "unauthorized and not accessed",
        "prefixes_s": list(prefixes),
        "estimator_config": asdict(settings["config"]),
        "search_seed_policy": {
            "independent_of_episode_seed": True,
            "mapping": _search_seed_map(prefixes),
        },
        "support_prior": {
            "center": "mean(first leader position - first leader velocity * first measurement time)",
            "radius_min_m": float(support_radius_min_m),
            "radius_max_m": float(support_radius_max_m),
        },
        "exact_replay_tolerance": v19.REPLAY_TOLERANCE,
        "source_manifest_sha256": source_manifest_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "decision_screens": {
            "structural_error_gate_m": STRUCTURAL_ERROR_GATE_M,
            "structural_equivalent_alternative_separation_m": settings[
                "config"
            ].alternative_separation_m,
            "structural_equivalent_rmse_gap_mps": STRUCTURAL_EQUIVALENT_RMSE_GAP_MPS,
            "noisy_error_gate_m": NOISY_ERROR_GATE_M,
            "required_noisy_success_fraction_240s": 0.95,
            "required_noisy_success_fraction_440s": 0.99,
            "required_joint_noisy_success_fraction_240s_and_440s": 0.95,
            "runtime_p99_limit_s": ACTION_INTERVAL_S,
            "runtime_p99_definition": "finite-sample nearest-rank p99 of solver-only runtime",
            "sustained_lock_checkpoints_s": [120.0, 240.0, 440.0],
            "certificate_gate": "deferred; no local-covariance or likelihood-gap certificate is evaluated",
        },
        "actual_estimator_boundary": {
            "input_archive": "online_inputs.npz only",
            "support_center_derived_from_online_leader broadcasts": True,
            "episode_seed_not_passed_to_search": True,
            "truth_loaded_only_after_actual_unscored_output_is_committed": True,
        },
        "structural_problem_is_oracle_diagnostic": True,
        "final_authorized": False,
    }


def _freeze_v19_sources(
    source_root: Path, execution_source: Path, source_manifest: Mapping[str, Any]
) -> None:
    if execution_source.exists():
        raise FileExistsError(f"execution-source snapshot already exists: {execution_source}")
    execution_source.mkdir(parents=True)
    entries = source_manifest["v19_source"]
    for name in SOURCE_SNAPSHOT_NAMES:
        source = source_root / name
        target = execution_source / name
        shutil.copy2(source, target)
        if _sha256_file(target) != str(entries[name]["sha256"]):
            raise RuntimeError(f"V19 source snapshot mismatch: {name}")
        target.chmod(0o444)
    execution_source.chmod(0o555)


def _verify_v19_snapshot(
    execution_source: Path, source_manifest: Mapping[str, Any]
) -> None:
    entries = source_manifest["v19_source"]
    for name in SOURCE_SNAPSHOT_NAMES:
        path = execution_source / name
        if not path.is_file() or _sha256_file(path) != str(entries[name]["sha256"]):
            raise RuntimeError(f"V19 execution-source snapshot mismatch: {name}")


def _stage_or_verify_campaign(
    settings: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    source_root = Path(settings["source_root"])
    evaluation = Path(settings["evaluation_directory"])
    output = Path(settings["output_directory"])
    resume = bool(settings["resume"])
    if output.exists() and not resume:
        raise FileExistsError(f"output already exists; use --resume: {output}")
    if not output.exists() and resume:
        raise FileNotFoundError(f"cannot resume missing output: {output}")

    source_manifest = _source_manifest(source_root, evaluation)
    input_manifest = _input_manifest(
        evaluation, settings["episode_indices"], settings["episode_seeds"]
    )
    metadata = _validate_v181_metadata(evaluation)
    environment_config = metadata.get("environment_config")
    if not isinstance(environment_config, Mapping):
        raise RuntimeError("V18.1 metadata lacks environment_config")
    if not math.isclose(
        float(environment_config.get("start_rho_min")),
        SUPPORT_RADIUS_MIN_M,
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        float(environment_config.get("start_rho_max")),
        SUPPORT_RADIUS_MAX_M,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("frozen V18.1 support shell differs from 120..350 m")
    # V19 uses explicit frozen constants.  It never obtains the shell from a
    # per-episode capture metadata or truth-label archive.
    support_min = SUPPORT_RADIUS_MIN_M
    support_max = SUPPORT_RADIUS_MAX_M

    if not resume:
        output.mkdir(parents=True)
        control = output / "control"
        control.mkdir()
        _write_json_atomic(control / "source_manifest.json", source_manifest)
        _write_json_atomic(control / "input_manifest.json", input_manifest)
        _freeze_v19_sources(
            source_root, control / "execution_source", source_manifest
        )
        contract = _campaign_contract(
            settings,
            _sha256_file(control / "source_manifest.json"),
            _sha256_file(control / "input_manifest.json"),
            support_min,
            support_max,
        )
        _write_json_atomic(control / "campaign_contract.json", contract)
        contract_hash = _sha256_file(control / "campaign_contract.json")
        _write_json_atomic(
            control / "campaign_manifest.json",
            {
                "schema_version": 1,
                "created_at_utc": _utc_now(),
                "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "interpreter": str(Path(sys.executable).resolve()),
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "contract_sha256": contract_hash,
                "final_authorized": False,
            },
        )
        _write_json_atomic(
            control / "progress.json",
            {
                "schema_version": 1,
                "contract_sha256": contract_hash,
                "completed": {},
            },
        )
    else:
        control = output / "control"
        recorded_source = _load_json_object(
            control / "source_manifest.json", label="V19 source manifest"
        )
        recorded_input = _load_json_object(
            control / "input_manifest.json", label="V19 input manifest"
        )
        if not _same_json(recorded_source, source_manifest):
            raise RuntimeError("V19 source manifest changed; resume refused")
        if not _same_json(recorded_input, input_manifest):
            raise RuntimeError("V19 frozen input manifest changed; resume refused")
        _verify_v19_snapshot(control / "execution_source", source_manifest)
        contract = _campaign_contract(
            settings,
            _sha256_file(control / "source_manifest.json"),
            _sha256_file(control / "input_manifest.json"),
            support_min,
            support_max,
        )
        recorded_contract = _load_json_object(
            control / "campaign_contract.json", label="V19 campaign contract"
        )
        if not _same_json(recorded_contract, contract):
            raise RuntimeError("V19 immutable campaign contract changed; resume refused")
        contract = recorded_contract
        contract_hash = _sha256_file(control / "campaign_contract.json")

    if not resume:
        contract = _load_json_object(
            output / "control" / "campaign_contract.json", label="V19 campaign contract"
        )
    return source_manifest, input_manifest, contract, contract_hash


@contextmanager
def _frozen_v181_imports(source_manifest: Mapping[str, Any]):
    frozen = str(source_manifest["v181_execution_source_directory"])
    sys.path.insert(0, frozen)
    try:
        yield
    finally:
        try:
            sys.path.remove(frozen)
        except ValueError:
            pass


def _support_center_from_online(history: v19.OnlineDopplerHistory) -> np.ndarray:
    return v19.initial_leader_centroid_from_history(history)


def _candidate_seed(contract: Mapping[str, Any], problem: str, prefix: float) -> int:
    mapping = contract["search_seed_policy"]["mapping"]
    key = f"{float(prefix):g}"
    return int(mapping[key][problem])


def _forward_prediction_rmse(
    full_history: v19.OnlineDopplerHistory,
    previous_prefix_s: Optional[float],
    prefix_s: float,
    previous_position_m: Optional[np.ndarray],
) -> Tuple[Optional[float], int]:
    if previous_prefix_s is None or previous_position_m is None:
        return None, 0
    selection = (full_history.t_s > float(previous_prefix_s) + 1e-12) & (
        full_history.t_s <= float(prefix_s) + 1e-12
    )
    count = int(np.sum(selection))
    if count == 0:
        return None, 0
    forward_window = full_history.take(selection)
    residual = forward_window.doppler_measured_mps - v19.predict_doppler(
        previous_position_m, forward_window
    )
    return float(np.sqrt(np.mean(residual * residual))), count


def _run_prefix_estimates(
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: v19.BatchEstimatorConfig,
    prefixes_s: Sequence[float],
    contract: Mapping[str, Any],
    problem: str,
) -> Tuple[Dict[float, v19.BatchEstimate], Dict[str, Any]]:
    estimates: Dict[float, v19.BatchEstimate] = {}
    records: List[Dict[str, Any]] = []
    previous_prefix: Optional[float] = None
    previous_position: Optional[np.ndarray] = None
    for prefix in prefixes_s:
        prefix_history = history.prefix(float(prefix))
        search_seed = _candidate_seed(contract, problem, float(prefix))
        estimate = v19.estimate_initial_position_multistart(
            prefix_history,
            center_m,
            radius_min_m,
            radius_max_m,
            config,
            candidate_seed=search_seed,
        )
        forward_rmse, forward_count = _forward_prediction_rmse(
            history,
            previous_prefix,
            float(prefix),
            previous_position,
        )
        estimates[float(prefix)] = estimate
        record = estimate.to_dict()
        record.update(
            {
                "problem": problem,
                "prefix_s": float(prefix),
                "measurement_count": prefix_history.measurement_count,
                "candidate_seed": search_seed,
                "forward_prediction_origin_prefix_s": previous_prefix,
                "forward_prediction_measurement_count": forward_count,
                "forward_prediction_rmse_mps": forward_rmse,
            }
        )
        records.append(record)
        previous_prefix = float(prefix)
        previous_position = estimate.best.initial_position_m.copy()
    return estimates, {
        "schema_version": 1,
        "problem": problem,
        "estimator_config": asdict(config),
        "support_center_m": center_m.tolist(),
        "support_radius_min_m": float(radius_min_m),
        "support_radius_max_m": float(radius_max_m),
        "search_seed_scope": "problem_and_prefix_only",
        "diagnostic_labels_used_by_search": problem == "structural_noiseless",
        "prefix_estimates": records,
    }


def _course_difference_from_velocity(velocity: np.ndarray) -> float:
    yaw = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
    return v19.circular_difference_deg(float(yaw[0]), float(yaw[1]))


def _prefix_geometry(
    history: v19.OnlineDopplerHistory, prefix_s: float
) -> Dict[str, float]:
    prefix = history.prefix(prefix_s)
    positions = prefix.leader_position_m[-1]
    velocities = prefix.leader_velocity_mps[-1]
    speeds = np.linalg.norm(velocities, axis=1)
    return {
        "leader_separation_m": float(np.linalg.norm(positions[1] - positions[0])),
        "leader_course_difference_deg": _course_difference_from_velocity(velocities),
        "leader_speed_difference_mps": float(abs(speeds[1] - speeds[0])),
        "leader_relative_speed_mps": float(np.linalg.norm(velocities[1] - velocities[0])),
    }


def _load_pf_prefix_metrics(
    trace_path: Path,
    prefixes_s: Sequence[float],
    episode_index: int,
    episode_seed: int,
) -> Dict[float, Dict[str, Any]]:
    """Read frozen PF diagnostics only in the post-estimation scoring phase."""

    requested = (
        "localization_error_diagnostic_m",
        "pf_std_largest_eigen_raw_m",
        "nees_diagnostic",
        "coverage_95_diagnostic",
        "formation_error_est_online_m",
        "formation_error_true_diagnostic_m",
        "fim_online_win_eig_min",
    )
    covariance_fields = (
        "pf_cov_xx",
        "pf_cov_xy",
        "pf_cov_xz",
        "pf_cov_yy",
        "pf_cov_yz",
        "pf_cov_zz",
    )
    with np.load(Path(trace_path), allow_pickle=False) as archive:
        required = {
            "episode_index",
            "episode_seed",
            "controller",
            "step",
            "t_s",
            *requested,
            *covariance_fields,
        }
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"frozen trace lacks PF prefix fields: {sorted(missing)}")
        if not np.all(np.asarray(archive["episode_index"]) == int(episode_index)):
            raise RuntimeError("frozen trace episode-index mismatch")
        if not np.all(np.asarray(archive["episode_seed"]) == int(episode_seed)):
            raise RuntimeError("frozen trace episode-seed mismatch")
        controller = np.asarray(archive["controller"]).astype(str)
        if not np.all(controller == v19.PRIMARY_CONTROLLER):
            raise RuntimeError("V19 refuses a trace from a non-primary controller")
        times = np.asarray(archive["t_s"], dtype=np.float64)
        output: Dict[float, Dict[str, Any]] = {}
        for prefix in prefixes_s:
            index = int(round(float(prefix) / ACTION_INTERVAL_S)) - 1
            if index < 0 or index >= times.size:
                raise RuntimeError(f"frozen trace lacks the {prefix:g} s endpoint row")
            if not math.isclose(
                float(times[index]), float(prefix), rel_tol=0.0, abs_tol=1e-8
            ):
                raise RuntimeError(f"frozen trace time mismatch at {prefix:g} s")
            if int(np.asarray(archive["step"])[index]) != int(round(float(prefix) / 2.0)):
                raise RuntimeError(f"trace step/time mismatch at {prefix:g} s")
            values: Dict[str, Any] = {
                "trace_step": int(np.asarray(archive["step"])[index])
            }
            for field in requested:
                values[field] = float(np.asarray(archive[field])[index])
            xx, xy, xz, yy, yz, zz = (
                float(np.asarray(archive[field])[index]) for field in covariance_fields
            )
            covariance = np.array(
                [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64
            )
            eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
            values.update(
                {
                    "covariance_trace_m2": float(np.trace(covariance)),
                    "covariance_eigenvalue_min_m2": float(eigenvalues[0]),
                    "covariance_eigenvalue_max_m2": float(eigenvalues[-1]),
                }
            )
            output[float(prefix)] = values
    return output


def _score_row(
    *,
    episode_index: int,
    episode_seed: int,
    prefix_s: float,
    problem: str,
    estimate: v19.BatchEstimate,
    history: v19.OnlineDopplerHistory,
    truth: v19.ReplayTruthDiagnostics,
    unscored_record: Mapping[str, Any],
    capture: v19.ReplayCapture,
    initial_elevation_deg: float,
    pf_prefix_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    errors = v19.estimate_diagnostic_errors(estimate, history, truth)
    best = estimate.best
    alternative_rmse_gap = None
    if estimate.alternative_mode_index is not None:
        alternative_rmse_gap = float(
            estimate.modes[estimate.alternative_mode_index].residual_rmse_mps
            - best.residual_rmse_mps
        )
    equivalent_alternative = bool(
        estimate.alternative_mode_index is not None
        and estimate.alternative_distance_m is not None
        and estimate.alternative_distance_m >= 7.0
        and alternative_rmse_gap is not None
        and alternative_rmse_gap <= STRUCTURAL_EQUIVALENT_RMSE_GAP_MPS
    )
    structural_pass = None
    noisy_pass = None
    if problem == "structural_noiseless":
        structural_pass = bool(
            errors["initial_position_error_m"] <= STRUCTURAL_ERROR_GATE_M
            and not equivalent_alternative
        )
    else:
        noisy_pass = bool(errors["endpoint_position_error_m"] <= NOISY_ERROR_GATE_M)
    row: Dict[str, Any] = {
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "problem": problem,
        "prefix_s": float(prefix_s),
        "measurement_count": history.measurement_count,
        "candidate_seed": int(unscored_record["candidate_seed"]),
        "initial_position_error_m": errors["initial_position_error_m"],
        "endpoint_position_error_m": errors["endpoint_position_error_m"],
        "residual_rmse_mps": best.residual_rmse_mps,
        "coarse_best_rmse_mps": estimate.coarse_best_rmse_mps,
        "runtime_s": estimate.runtime_s,
        "mode_count": len(estimate.modes),
        "clustered_mode_count": estimate.clustered_mode_count,
        "best_converged": best.converged,
        "best_iterations": best.iterations,
        "best_local_radius95_m": best.local_radius95_m,
        "best_local_covariance_valid": best.local_covariance_valid,
        "best_hessian_rank": best.hessian_rank,
        "best_hessian_condition_number": best.hessian_condition_number,
        "best_hessian_eigenvalue_min": float(np.min(best.hessian_eigenvalues)),
        "alternative_distance_m": estimate.alternative_distance_m,
        "alternative_delta_sse_mps2": estimate.alternative_delta_sse_mps2,
        "alternative_delta_chi2": estimate.alternative_delta_chi2,
        "alternative_rmse_gap_mps": alternative_rmse_gap,
        "measurement_equivalent_alternative": equivalent_alternative,
        "forward_prediction_origin_prefix_s": unscored_record.get(
            "forward_prediction_origin_prefix_s"
        ),
        "forward_prediction_measurement_count": unscored_record.get(
            "forward_prediction_measurement_count"
        ),
        "forward_prediction_rmse_mps": unscored_record.get(
            "forward_prediction_rmse_mps"
        ),
        "structural_gate_pass": structural_pass,
        "noisy_7m_gate_pass": noisy_pass,
        "truth_initial_radius_m": capture.initial_pf_metrics["truth_initial_radius_m"],
        "truth_initial_elevation_deg": initial_elevation_deg,
    }
    row.update(_prefix_geometry(history, prefix_s))
    row.update(
        {
            f"initial_pf_{key}": value
            for key, value in capture.initial_pf_metrics.items()
        }
    )
    row.update(
        {
            f"frozen_pf_final_{key}": value
            for key, value in capture.frozen_pf_endpoint.items()
        }
    )
    row.update(
        {f"frozen_pf_prefix_{key}": value for key, value in pf_prefix_metrics.items()}
    )
    return row


def _annotate_episode_lock(rows: List[Dict[str, Any]]) -> None:
    """Add checkpoint-sustained lock and time-to-lock diagnostics in place."""

    checkpoints = (120.0, 240.0, 440.0)
    definitions = (
        ("recorded_noisy", "noisy_7m_gate_pass", "noisy_7m"),
        ("structural_noiseless", "structural_gate_pass", "structural_1m"),
    )
    for problem, gate_field, label in definitions:
        selected = [row for row in rows if row["problem"] == problem]
        by_prefix = {float(row["prefix_s"]): row for row in selected}
        if any(prefix not in by_prefix for prefix in checkpoints):
            raise RuntimeError(f"missing sustained-lock checkpoint for {problem}")
        sustained: Dict[float, bool] = {}
        for prefix in checkpoints:
            sustained[prefix] = all(
                by_prefix[later].get(gate_field) is True
                for later in checkpoints
                if later >= prefix
            )
        time_to_lock = next(
            (prefix for prefix in checkpoints if sustained[prefix]), None
        )
        for row in selected:
            prefix = float(row["prefix_s"])
            row[f"time_to_sustained_{label}_lock_s"] = time_to_lock
            row[f"sustained_{label}_lock_through_440"] = (
                sustained[prefix] if prefix in sustained else None
            )


def _capture_paths(output: Path, index: int, seed: int) -> Tuple[Path, Path, Path, Path]:
    stem = f"episode_{index:04d}_seed_{seed}"
    archive = output / "measurement_archive" / stem
    result = output / "episode_results" / f"{stem}.json"
    actual = output / "estimator_outputs" / f"{stem}_actual_unscored.json"
    structural = (
        output / "estimator_outputs" / f"{stem}_structural_oracle_unscored.json"
    )
    return archive, result, actual, structural


def _capture_file_hashes(archive: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for name in ("online_inputs.npz", "truth_labels.npz", "capture_metadata.json"):
        path = archive / name
        if not path.is_file():
            raise FileNotFoundError(f"incomplete replay capture: {path}")
        result[name] = _sha256_file(path)
    return result


def _verify_committed_result(
    result_path: Path,
    expected_sha256: str,
    contract_sha256: str,
    episode_index: int,
    episode_seed: int,
    archive: Path,
) -> Dict[str, Any]:
    if not result_path.is_file() or _sha256_file(result_path) != expected_sha256:
        raise RuntimeError(f"committed episode result changed: {result_path}")
    result = _load_json_object(result_path, label="V19 episode result")
    expected = {
        "contract_sha256": contract_sha256,
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"committed episode result field changed: {key}")
    current_capture = _capture_file_hashes(archive)
    if result.get("capture_sha256") != current_capture:
        raise RuntimeError("committed replay-capture hashes changed")
    for key in ("actual_unscored_path", "structural_unscored_path"):
        path = Path(str(result[key]))
        hash_key = key.replace("_path", "_sha256")
        if not path.is_file() or _sha256_file(path) != str(result[hash_key]):
            raise RuntimeError(f"committed estimator output changed: {path}")
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("episode result lacks summary rows")
    return result


def _verify_episode_inputs(
    episode_input: Mapping[str, Any], episode_index: int, episode_seed: int
) -> None:
    if int(episode_input["episode_index"]) != int(episode_index):
        raise RuntimeError("input manifest episode index mismatch")
    if int(episode_input["episode_seed"]) != int(episode_seed):
        raise RuntimeError("input manifest episode seed mismatch")
    v19.assert_seed_is_not_sealed_final(episode_seed)
    for kind in ("trace", "noise_tape"):
        path = Path(str(episode_input[f"{kind}_path"]))
        if not path.is_file() or _sha256_file(path) != str(
            episode_input[f"{kind}_sha256"]
        ):
            raise RuntimeError(f"frozen {kind} changed for episode {episode_index}")


def _run_episode(
    *,
    settings: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha256: str,
    source_manifest: Mapping[str, Any],
    episode_input: Mapping[str, Any],
    episode_index: int,
    episode_seed: int,
) -> Dict[str, Any]:
    output = Path(settings["output_directory"])
    evaluation = Path(settings["evaluation_directory"])
    config = settings["config"]
    prefixes = tuple(settings["prefixes_s"])
    radius_min = float(contract["support_prior"]["radius_min_m"])
    radius_max = float(contract["support_prior"]["radius_max_m"])
    archive, result_path, actual_path, structural_path = _capture_paths(
        output, episode_index, episode_seed
    )
    _verify_episode_inputs(episode_input, episode_index, episode_seed)

    capture_files = tuple(
        archive / name
        for name in ("online_inputs.npz", "truth_labels.npz", "capture_metadata.json")
    )
    present = [path.exists() for path in capture_files]
    capture_runtime = 0.0
    if any(present) and not all(present):
        raise RuntimeError(f"partial replay capture exists; refusing overwrite: {archive}")
    if not all(present):
        started = time.perf_counter()
        with _frozen_v181_imports(source_manifest):
            capture = v19.capture_v181_episode(
                evaluation,
                episode_index,
                episode_seed,
                controller=v19.PRIMARY_CONTROLLER,
                replay_tolerance=v19.REPLAY_TOLERANCE,
            )
        capture_runtime = float(time.perf_counter() - started)
        v19.save_replay_capture(archive, capture)
        del capture

    # The deployable/noisy estimator phase deliberately opens the online
    # allowlist only.  Its support center is reconstructed from leader
    # broadcasts, and its deterministic search seed is prefix-only.
    online_path = archive / "online_inputs.npz"
    online_sha256 = _sha256_file(online_path)
    online = v19.load_online_history(online_path)
    center = _support_center_from_online(online)
    actual_estimates, actual_unscored = _run_prefix_estimates(
        online,
        center,
        radius_min,
        radius_max,
        config,
        prefixes,
        contract,
        "recorded_noisy",
    )
    actual_unscored.update(
        {
            "online_inputs_sha256": online_sha256,
        }
    )
    _write_json_atomic(actual_path, actual_unscored)

    # Truth is first loaded only after the unscored noisy-estimator result is
    # durable.  From this point onward all truth use is diagnostic/scoring.
    capture = v19.load_replay_capture(archive)
    capture_hashes = _capture_file_hashes(archive)
    if capture.episode_index != episode_index or capture.episode_seed != episode_seed:
        raise RuntimeError("replay capture belongs to a different episode")
    if capture.controller != v19.PRIMARY_CONTROLLER:
        raise RuntimeError("V19 refuses a capture from a non-primary controller")
    if not bool(capture.replay_integrity.get("passed")):
        raise RuntimeError("replay capture did not pass exact replay")
    pf_prefix = _load_pf_prefix_metrics(
        Path(str(episode_input["trace_path"])),
        prefixes,
        episode_index,
        episode_seed,
    )
    structural_full = v19.structural_noiseless_history(capture)
    structural_estimates, structural_unscored = _run_prefix_estimates(
        structural_full,
        center,
        radius_min,
        radius_max,
        config,
        prefixes,
        contract,
        "structural_noiseless",
    )
    structural_unscored.update(
        {
            "oracle_diagnostic": True,
            "online_inputs_sha256": capture_hashes["online_inputs.npz"],
            "truth_labels_sha256": capture_hashes["truth_labels.npz"],
        }
    )
    _write_json_atomic(structural_path, structural_unscored)

    offset = (
        capture.truth.initial_follower_position_m
        - capture.truth.initial_leader_centroid_m
    )
    radius = max(float(np.linalg.norm(offset)), 1e-12)
    elevation = float(np.degrees(np.arcsin(np.clip(offset[2] / radius, -1.0, 1.0))))
    actual_records = {
        float(item["prefix_s"]): item
        for item in actual_unscored["prefix_estimates"]
    }
    structural_records = {
        float(item["prefix_s"]): item
        for item in structural_unscored["prefix_estimates"]
    }
    rows: List[Dict[str, Any]] = []
    for prefix in prefixes:
        actual_history = online.prefix(prefix)
        rows.append(
            _score_row(
                episode_index=episode_index,
                episode_seed=episode_seed,
                prefix_s=prefix,
                problem="recorded_noisy",
                estimate=actual_estimates[prefix],
                history=actual_history,
                truth=capture.truth,
                unscored_record=actual_records[prefix],
                capture=capture,
                initial_elevation_deg=elevation,
                pf_prefix_metrics=pf_prefix[prefix],
            )
        )
        structural_history = structural_full.prefix(prefix)
        rows.append(
            _score_row(
                episode_index=episode_index,
                episode_seed=episode_seed,
                prefix_s=prefix,
                problem="structural_noiseless",
                estimate=structural_estimates[prefix],
                history=structural_history,
                truth=capture.truth,
                unscored_record=structural_records[prefix],
                capture=capture,
                initial_elevation_deg=elevation,
                pf_prefix_metrics=pf_prefix[prefix],
            )
        )
    _annotate_episode_lock(rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "contract_sha256": contract_sha256,
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "controller": v19.PRIMARY_CONTROLLER,
        "capture_runtime_s": capture_runtime,
        "capture_sha256": capture_hashes,
        "replay_integrity": dict(capture.replay_integrity),
        "capture_provenance": dict(capture.provenance),
        "actual_unscored_path": str(actual_path.resolve()),
        "actual_unscored_sha256": _sha256_file(actual_path),
        "structural_unscored_path": str(structural_path.resolve()),
        "structural_unscored_sha256": _sha256_file(structural_path),
        "rows": rows,
        "final_authorized": False,
    }
    _write_json_atomic(result_path, result)
    return _load_json_object(result_path, label="new V19 episode result")


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted(
        {(str(row["problem"]), float(row["prefix_s"])) for row in rows},
        key=lambda item: (item[1], item[0]),
    )
    result: List[Dict[str, Any]] = []
    for problem, prefix in keys:
        group = [
            row
            for row in rows
            if row["problem"] == problem and float(row["prefix_s"]) == prefix
        ]
        result.append(
            {
                "problem": problem,
                "prefix_s": prefix,
                "count": len(group),
                "initial_error_p50_m": _percentile(
                    (row["initial_position_error_m"] for row in group), 50
                ),
                "initial_error_p95_m": _percentile(
                    (row["initial_position_error_m"] for row in group), 95
                ),
                "initial_error_p99_m": _percentile(
                    (row["initial_position_error_m"] for row in group), 99
                ),
                "initial_error_max_m": _percentile(
                    (row["initial_position_error_m"] for row in group), 100
                ),
                "endpoint_error_p50_m": _percentile(
                    (row["endpoint_position_error_m"] for row in group), 50
                ),
                "endpoint_error_p95_m": _percentile(
                    (row["endpoint_position_error_m"] for row in group), 95
                ),
                "endpoint_error_p99_m": _percentile(
                    (row["endpoint_position_error_m"] for row in group), 99
                ),
                "endpoint_error_max_m": _percentile(
                    (row["endpoint_position_error_m"] for row in group), 100
                ),
                "residual_rmse_mean_mps": _mean(
                    row["residual_rmse_mps"] for row in group
                ),
                "forward_prediction_rmse_mean_mps": _mean(
                    row["forward_prediction_rmse_mps"] for row in group
                ),
                "runtime_p50_s": _percentile((row["runtime_s"] for row in group), 50),
                "runtime_p99_nearest_rank_s": _nearest_rank_percentile(
                    (row["runtime_s"] for row in group), 99
                ),
                "runtime_max_s": _percentile((row["runtime_s"] for row in group), 100),
                "equivalent_alternative_count": int(
                    sum(bool(row["measurement_equivalent_alternative"]) for row in group)
                ),
                "structural_gate_pass_count": int(
                    sum(row["structural_gate_pass"] is True for row in group)
                ),
                "noisy_7m_gate_pass_count": int(
                    sum(row["noisy_7m_gate_pass"] is True for row in group)
                ),
            }
        )
    return result


def _bin_label(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for index, label in enumerate(labels):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        final = index == len(labels) - 1
        if value >= lower and (value < upper or (final and value <= upper)):
            return label
    return "outside"


def _stratified_final_summary(
    final_actual: Sequence[Mapping[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    definitions = {
        "truth_initial_radius_m": (
            (120.0, 180.0, 240.0, 300.0, 350.0),
            ("120-180", "180-240", "240-300", "300-350"),
        ),
        "abs_initial_elevation_deg": (
            (0.0, 15.0, 30.0, 90.0),
            ("0-15", "15-30", "30-90"),
        ),
        "leader_course_difference_deg": (
            (0.0, 15.0, 45.0, 90.0, 180.0),
            ("0-15", "15-45", "45-90", "90-180"),
        ),
        "leader_speed_difference_mps": (
            (0.0, 0.5, 1.0, 2.1),
            ("0-0.5", "0.5-1.0", "1.0-2.1"),
        ),
    }
    output: Dict[str, List[Dict[str, Any]]] = {}
    for field, (edges, labels) in definitions.items():
        grouped: Dict[str, List[Mapping[str, Any]]] = {label: [] for label in labels}
        grouped["outside"] = []
        for row in final_actual:
            if field == "abs_initial_elevation_deg":
                value = abs(float(row["truth_initial_elevation_deg"]))
            else:
                value = float(row[field])
            grouped[_bin_label(value, edges, labels)].append(row)
        output[field] = []
        for label in (*labels, "outside"):
            group = grouped[label]
            if not group:
                continue
            output[field].append(
                {
                    "bin": label,
                    "count": len(group),
                    "noisy_7m_success_count": int(
                        sum(row["noisy_7m_gate_pass"] is True for row in group)
                    ),
                    "endpoint_error_p50_m": _percentile(
                        (row["endpoint_position_error_m"] for row in group), 50
                    ),
                    "endpoint_error_p95_m": _percentile(
                        (row["endpoint_position_error_m"] for row in group), 95
                    ),
                    "endpoint_error_max_m": _percentile(
                        (row["endpoint_position_error_m"] for row in group), 100
                    ),
                }
            )
    return output


def _decision(
    rows: Sequence[Mapping[str, Any]], final_prefix_s: float
) -> Dict[str, Any]:
    structural = [
        row
        for row in rows
        if row["problem"] == "structural_noiseless"
        and float(row["prefix_s"]) == final_prefix_s
    ]
    noisy = [
        row
        for row in rows
        if row["problem"] == "recorded_noisy"
        and float(row["prefix_s"]) == final_prefix_s
    ]
    if not structural or len(structural) != len(noisy):
        raise RuntimeError("final-prefix decision rows are incomplete")
    noisy_240 = [
        row
        for row in rows
        if row["problem"] == "recorded_noisy"
        and float(row["prefix_s"]) == 240.0
    ]
    if len(noisy_240) != len(noisy):
        raise RuntimeError("240 s noisy decision rows are incomplete")
    structural_pass_count = sum(row["structural_gate_pass"] is True for row in structural)
    structural_go = structural_pass_count == len(structural)
    noisy_pass_count = sum(row["noisy_7m_gate_pass"] is True for row in noisy)
    noisy_240_pass_count = sum(
        row["noisy_7m_gate_pass"] is True for row in noisy_240
    )
    joint_240_440_pass_count = sum(
        row.get("sustained_noisy_7m_lock_through_440") is True
        for row in noisy_240
    )
    required_noisy = int(math.ceil(0.99 * len(noisy) - 1e-12))
    required_noisy_240 = int(math.ceil(0.95 * len(noisy) - 1e-12))
    required_joint_240_440 = int(math.ceil(0.95 * len(noisy) - 1e-12))
    runtime_p99 = _nearest_rank_percentile((row["runtime_s"] for row in noisy), 99)
    estimator_screen_observed_pass = bool(
        noisy_pass_count >= required_noisy
        and noisy_240_pass_count >= required_noisy_240
        and joint_240_440_pass_count >= required_joint_240_440
        and runtime_p99 is not None
        and runtime_p99 < ACTION_INTERVAL_S
    )
    episode_indices = sorted({int(row["episode_index"]) for row in noisy})
    decision_eligible = episode_indices == list(range(100))
    if not decision_eligible:
        code = "SMOKE_OR_PARTIAL_COMPLETE_NO_DECISION"
        next_step = "run the immutable 100-episode development contract before applying engineering gates"
    elif not structural_go:
        code = "STOP_GEOMETRY"
        next_step = "change sensing geometry, maneuver, or sensors before estimator/control work"
    elif not estimator_screen_observed_pass:
        code = "STOP_ESTIMATOR"
        next_step = "improve the estimator/model; do not train a controller"
    else:
        code = "PASS_ENGINEERING_SCREEN_ONLY"
        next_step = (
            "freeze a global certificate and independent >=1000-scenario validation; "
            "then run the deterministic estimator/MPC 2x2 opportunity test"
        )
    return {
        "schema_version": 1,
        "decision_code": code,
        "engineering_screen_eligible": decision_eligible,
        "structural_observability": {
            "go": structural_go if decision_eligible else None,
            "observed_pass": structural_go,
            "pass_count": structural_pass_count,
            "scenario_count": len(structural),
        },
        "recorded_noisy_estimator_screen": {
            "go": estimator_screen_observed_pass if decision_eligible else None,
            "observed_pass": estimator_screen_observed_pass,
            "success_count": noisy_pass_count,
            "required_success_count": required_noisy,
            "success_240s_count": noisy_240_pass_count,
            "required_success_240s_count": required_noisy_240,
            "joint_success_240s_and_440s_count": joint_240_440_pass_count,
            "required_joint_success_240s_and_440s_count": required_joint_240_440,
            "scenario_count": len(noisy),
            "runtime_p99_nearest_rank_s": runtime_p99,
            "runtime_limit_s": ACTION_INTERVAL_S,
            "certificate_gate": "deferred and not evaluated",
        },
        "formal_global_certificate_available": False,
        "development_results_are_not_fresh_validation": True,
        "recommended_next_step": next_step,
        "training_authorized": False,
        "final_authorized": False,
    }


def _sustained_lock_summary(
    rows: Sequence[Mapping[str, Any]], problem: str, label: str
) -> Dict[str, Any]:
    final_rows = [
        row
        for row in rows
        if row["problem"] == problem and float(row["prefix_s"]) == 440.0
    ]
    field = f"time_to_sustained_{label}_lock_s"
    counts = {
        f"time_to_lock_{checkpoint:g}s_count": int(
            sum(
                row.get(field) is not None
                and math.isclose(
                    float(row[field]), checkpoint, rel_tol=0.0, abs_tol=1e-12
                )
                for row in final_rows
            )
        )
        for checkpoint in (120.0, 240.0, 440.0)
    }
    counts["never_sustained_lock_count"] = int(
        sum(row.get(field) is None for row in final_rows)
    )
    counts.update(
        {
            f"sustained_by_{checkpoint:g}s_count": int(
                sum(
                    row.get(field) is not None
                    and float(row[field]) <= checkpoint
                    for row in final_rows
                )
            )
            for checkpoint in (120.0, 240.0, 440.0)
        }
    )
    return {"episode_count": len(final_rows), **counts}


def _write_campaign_outputs(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    contract_sha256: str,
    prefixes_s: Sequence[float],
) -> Dict[str, Any]:
    ordered_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            int(row["episode_index"]),
            float(row["prefix_s"]),
            str(row["problem"]),
        ),
    )
    aggregate = _aggregate_rows(ordered_rows)
    final_prefix = float(prefixes_s[-1])
    final_actual = [
        row
        for row in ordered_rows
        if row["problem"] == "recorded_noisy"
        and float(row["prefix_s"]) == final_prefix
    ]
    decision = _decision(ordered_rows, final_prefix)
    nearest = [row["initial_pf_nearest_particle_distance_m"] for row in final_actual]
    within7 = [row["initial_pf_particle_count_within_7m"] for row in final_actual]
    within20 = [row["initial_pf_particle_count_within_20m"] for row in final_actual]
    summary = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "contract_sha256": contract_sha256,
        "status": "completed",
        "finished_at_utc": _utc_now(),
        "episode_count": len(results),
        "row_count": len(ordered_rows),
        "prefixes_s": list(prefixes_s),
        "aggregate_by_problem_and_prefix": aggregate,
        "initial_pf_support_audit": {
            "nearest_particle_distance_p50_m": _percentile(nearest, 50),
            "nearest_particle_distance_p95_m": _percentile(nearest, 95),
            "nearest_particle_distance_max_m": _percentile(nearest, 100),
            "episodes_with_particle_within_7m": int(sum(int(value) > 0 for value in within7)),
            "episodes_with_particle_within_20m": int(sum(int(value) > 0 for value in within20)),
            "episode_count": len(final_actual),
        },
        "sustained_lock_120_240_440": {
            "recorded_noisy_7m": _sustained_lock_summary(
                ordered_rows, "recorded_noisy", "noisy_7m"
            ),
            "structural_noiseless_1m": _sustained_lock_summary(
                ordered_rows, "structural_noiseless", "structural_1m"
            ),
        },
        "stratified_recorded_noisy_440s": _stratified_final_summary(final_actual),
        "decision": decision,
        "training_performed": False,
        "final_authorized": False,
    }
    _write_json_atomic(output / "episode_prefix_summary.json", ordered_rows)
    _write_csv_atomic(output / "episode_prefix_summary.csv", ordered_rows)
    _write_json_atomic(output / "campaign_summary.json", summary)
    _write_json_atomic(output / "decision.json", decision)
    return summary


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    settings = _resolved_settings(args)
    output = Path(settings["output_directory"])
    source_manifest, input_manifest, contract, contract_hash = _stage_or_verify_campaign(
        settings
    )
    progress_path = output / "control" / "progress.json"
    run_state_path = output / "run_state.json"
    _write_json_atomic(
        run_state_path,
        {
            "status": "running",
            "updated_at_utc": _utc_now(),
            "contract_sha256": contract_hash,
            "final_authorized": False,
        },
    )
    try:
        progress = _load_json_object(progress_path, label="V19 progress")
        if progress.get("contract_sha256") != contract_hash:
            raise RuntimeError("progress file belongs to another campaign contract")
        completed = progress.get("completed")
        if not isinstance(completed, dict):
            raise RuntimeError("progress file lacks completed mapping")
        input_by_index = {
            int(item["episode_index"]): item for item in input_manifest["episodes"]
        }
        results: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        total = len(settings["episode_indices"])
        for position, (episode_index, episode_seed) in enumerate(
            zip(settings["episode_indices"], settings["episode_seeds"]), start=1
        ):
            key = str(episode_index)
            archive, result_path, _, _ = _capture_paths(
                output, episode_index, episode_seed
            )
            committed = completed.get(key)
            if committed is not None:
                if not isinstance(committed, Mapping):
                    raise RuntimeError(f"invalid progress entry for episode {episode_index}")
                result = _verify_committed_result(
                    result_path,
                    str(committed["result_sha256"]),
                    contract_hash,
                    episode_index,
                    episode_seed,
                    archive,
                )
            elif result_path.exists():
                # Adopt only a complete, self-consistent atomic result left
                # between result commit and progress commit.
                candidate_hash = _sha256_file(result_path)
                result = _verify_committed_result(
                    result_path,
                    candidate_hash,
                    contract_hash,
                    episode_index,
                    episode_seed,
                    archive,
                )
                completed[key] = {"result_sha256": candidate_hash}
                progress["completed"] = completed
                _write_json_atomic(progress_path, progress)
            else:
                result = _run_episode(
                    settings=settings,
                    contract=contract,
                    contract_sha256=contract_hash,
                    source_manifest=source_manifest,
                    episode_input=input_by_index[episode_index],
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                )
                result_hash = _sha256_file(result_path)
                completed[key] = {"result_sha256": result_hash}
                progress["completed"] = completed
                _write_json_atomic(progress_path, progress)
            results.append(result)
            rows.extend(dict(row) for row in result["rows"])
            if position % int(settings["progress_every"]) == 0 or position == total:
                print(
                    f"V19 episode {position}/{total}: index={episode_index}, "
                    f"seed={episode_seed}, completed={len(completed)}",
                    flush=True,
                )
        summary = _write_campaign_outputs(
            output,
            rows,
            results,
            contract_hash,
            settings["prefixes_s"],
        )
        _write_json_atomic(
            run_state_path,
            {
                "status": "completed",
                "updated_at_utc": _utc_now(),
                "contract_sha256": contract_hash,
                "episode_count": len(results),
                "decision_code": summary["decision"]["decision_code"],
                "training_performed": False,
                "final_authorized": False,
            },
        )
        print(
            f"V19 complete: {summary['decision']['decision_code']}; output={output}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        _write_json_atomic(
            run_state_path,
            {
                "status": "failed",
                "updated_at_utc": _utc_now(),
                "contract_sha256": contract_hash,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "training_performed": False,
                "final_authorized": False,
            },
        )
        raise


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
