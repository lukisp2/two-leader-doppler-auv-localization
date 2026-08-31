#!/usr/bin/env python3
"""Run the V20 paired positioning ablation (no RL and no training)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from uuv_v11_rng import EpisodeSeedPlan, ExogenousNoiseTape
from uuv_v18_resampling_guard import UUV3DConfig
import uuv_v19_observability as v19
import uuv_v20_positioning_ablation as v20


RUNNER_VERSION = "v20_positioning_ablation_runner_1.0"
DEFAULT_SEED_START = 46_000
DEFAULT_EPISODES = 100
SOURCE_NAMES = (
    "run_v20_positioning_ablation.py",
    "uuv_v20_positioning_ablation.py",
    "EXPERIMENT_PROTOCOL_V20_POSITIONING.md",
    "uuv_v19_observability.py",
    "baseline_controllers_v11.py",
    "uuv_v18_resampling_guard.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
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


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(dict(row)))
    os.replace(temporary, destination)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired PF/batch/oracle positioning with a common deterministic PID."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--environment-metadata", type=Path)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--arms", nargs="+", choices=v20.ARM_NAMES, default=list(v20.ARM_NAMES))
    parser.add_argument("--coarse-candidates", type=int)
    parser.add_argument("--coarse-sweeps", type=int)
    parser.add_argument("--local-starts", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def _default_metadata(root: Path) -> Path:
    return (
        root
        / "experiments_v18_1_guard_ablation_dev_3seed"
        / "evaluations"
        / "dev100_seed_28001_range_45000_45099"
        / "metadata.json"
    )


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    smoke = bool(args.smoke)
    episodes = int(args.episodes if args.episodes is not None else (3 if smoke else 100))
    output = (
        args.output_dir
        if args.output_dir is not None
        else root
        / (
            "experiments_v20_positioning_ablation_smoke"
            if smoke
            else "experiments_v20_positioning_ablation_dev100"
        )
    ).expanduser().resolve()
    metadata = (
        args.environment_metadata
        if args.environment_metadata is not None
        else _default_metadata(root)
    ).expanduser().resolve()
    settings = {
        "root": root,
        "output": output,
        "metadata": metadata,
        "seed_start": int(args.seed_start),
        "episode_start": int(args.episode_start),
        "episodes": episodes,
        "arms": list(args.arms),
        "coarse_candidates": int(
            args.coarse_candidates
            if args.coarse_candidates is not None
            else (512 if smoke else 4096)
        ),
        "coarse_sweeps": int(
            args.coarse_sweeps if args.coarse_sweeps is not None else (1 if smoke else 2)
        ),
        "local_starts": int(
            args.local_starts if args.local_starts is not None else (12 if smoke else 48)
        ),
        "smoke": smoke,
        "resume": bool(args.resume),
        "progress_every": max(1, int(args.progress_every)),
    }
    if episodes < 1 or settings["episode_start"] < 0:
        raise ValueError("episode count must be positive and episode start non-negative")
    first_seed = settings["seed_start"] + settings["episode_start"]
    last_seed = first_seed + episodes - 1
    for seed in (first_seed, last_seed):
        v19.assert_seed_is_not_sealed_final(seed)
    if first_seed <= v19.SEALED_FINAL_SEED_END and last_seed >= v19.SEALED_FINAL_SEED_START:
        raise PermissionError("V20 development run intersects the sealed final seed range")
    if not metadata.is_file():
        raise FileNotFoundError(f"missing frozen environment metadata: {metadata}")
    return settings


def _load_environment_config(metadata_path: Path) -> UUV3DConfig:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    frozen = metadata.get("environment_config")
    if not isinstance(frozen, dict):
        raise ValueError("metadata lacks environment_config")
    values = dict(frozen)
    values.update(
        {
            "v11_controller_id": "pid_track",
            "v11_generate_noise_tape": False,
            "v11_env_rank": 0,
        }
    )
    cfg = UUV3DConfig(**values)
    reconstructed = asdict(cfg)
    allowed = {"v11_controller_id", "v11_generate_noise_tape", "v11_env_rank"}
    for key, frozen_value in frozen.items():
        if key not in allowed and reconstructed.get(key) != frozen_value:
            raise RuntimeError(f"environment config drift for {key}")
    if str(cfg.v11_controller_id) != "pid_track":
        raise RuntimeError("V20 requires the direct three-channel PID action path")
    return cfg


def _contract(settings: Mapping[str, Any], cfg: UUV3DConfig) -> Dict[str, Any]:
    root = Path(settings["root"])
    source_hashes = {
        name: _sha256(root / name)
        for name in SOURCE_NAMES
    }
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v20.VERSION,
        "created_at_utc": _utc_now(),
        "purpose": "engineering positioning bottleneck ablation; no RL and no training",
        "seed_start": int(settings["seed_start"]),
        "episode_start": int(settings["episode_start"]),
        "episodes": int(settings["episodes"]),
        "arms": list(settings["arms"]),
        "track_start_s": v20.TRACK_START_S,
        "global_refresh_s": list(v20.GLOBAL_REFRESH_S),
        "estimator_config": {
            "coarse_candidates": int(settings["coarse_candidates"]),
            "coarse_sweeps": int(settings["coarse_sweeps"]),
            "local_starts": int(settings["local_starts"]),
            "gate_mode": "raw",
        },
        "environment_metadata_path": str(settings["metadata"]),
        "environment_metadata_sha256": _sha256(Path(settings["metadata"])),
        "environment_overrides": {
            "v11_controller_id": "pid_track direct three-channel actions",
            "v11_generate_noise_tape": False,
            "v11_env_rank": 0,
        },
        "fixed_horizon_actions": int(cfg.max_steps),
        "fixed_horizon_s": float(cfg.max_steps * cfg.action_dt),
        "source_sha256": source_hashes,
        "sealed_final_seed_range_untouched": [
            v19.SEALED_FINAL_SEED_START,
            v19.SEALED_FINAL_SEED_END,
        ],
    }


def _immutable_contract(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    return result


def _prepare_output(settings: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    output = Path(settings["output"])
    contract_path = output / "control" / "campaign_contract.json"
    if output.exists() and not bool(settings["resume"]):
        if any(output.iterdir()):
            raise FileExistsError(f"output exists; pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if _immutable_contract(previous) != _immutable_contract(contract):
            raise RuntimeError("resume contract differs from existing V20 campaign")
    else:
        _write_json_atomic(contract_path, contract)


def _tape_for_episode(output: Path, cfg: UUV3DConfig, seed: int, episode_index: int) -> ExogenousNoiseTape:
    path = output / "noise_tapes" / f"episode_{episode_index:04d}_seed_{seed}.npz"
    plan = EpisodeSeedPlan(root_seed=int(seed), episode_index=0, env_rank=0)
    if path.is_file():
        tape = ExogenousNoiseTape.load_npz(path)
        if tape.seed_plan != plan:
            raise RuntimeError("saved noise tape has the wrong seed plan")
        return tape
    substeps = int(cfg.max_steps) * int(round(float(cfg.action_dt) / float(cfg.sub_dt)))
    measurements = int(math.ceil(float(cfg.max_steps * cfg.action_dt) / float(cfg.s_meas_period))) + 2
    tape = ExogenousNoiseTape.generate(
        plan,
        n_substeps=substeps,
        n_doppler_measurements=measurements,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    tape.save_npz(temporary)
    os.replace(temporary, path)
    return tape


def _flatten_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    batch = summary.get("batch", {})
    return {
        key: value
        for key, value in summary.items()
        if key not in {"batch", "noise_cursor", "initial_truth_m", "initial_leader_centroid_m", "state_at_track_start_m"}
    } | {
        "batch_measurement_count": batch.get("measurement_count", 0),
        "batch_global_solve_count": batch.get("global_solve_count", 0),
        "batch_local_solve_count": batch.get("local_solve_count", 0),
        "batch_total_runtime_s": batch.get("total_runtime_s", 0.0),
        "batch_initial_position_error_m": batch.get("initial_position_error_m"),
    }


def _verify_pairing(outcomes: Mapping[str, v20.ArmOutcome], cfg: UUV3DConfig) -> None:
    names = list(outcomes)
    reference = outcomes[names[0]]
    reference_summary = reference.summary
    reference_trace = reference.trace
    expected_cursor = {
        "substep_index": int(cfg.max_steps * round(cfg.action_dt / cfg.sub_dt)),
        "measurement_index": int(round(cfg.max_steps * cfg.action_dt / cfg.s_meas_period)),
    }
    acquisition_actions = int(round(v20.TRACK_START_S / float(cfg.action_dt)))
    for name in names:
        outcome = outcomes[name]
        if outcome.summary["noise_cursor"] != expected_cursor:
            raise RuntimeError(f"{name} consumed the wrong amount of exogenous noise")
        if outcome.summary["noise_tape_sha256"] != reference_summary["noise_tape_sha256"]:
            raise RuntimeError("paired arms used different noise tapes")
        for field in ("initial_truth_m", "initial_leader_centroid_m", "state_at_track_start_m"):
            if not np.allclose(
                np.asarray(outcome.summary[field], dtype=np.float64),
                np.asarray(reference_summary[field], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"paired arm initial/acquisition state mismatch in {field}")
        for field in ("action_speed", "action_yaw", "action_pitch", "truth_x", "truth_y", "truth_z"):
            if not np.allclose(
                np.asarray(outcome.trace[field])[:acquisition_actions],
                np.asarray(reference_trace[field])[:acquisition_actions],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"paired arms diverged during common acquisition: {field}")
        action = np.column_stack(
            [outcome.trace["action_speed"], outcome.trace["action_yaw"], outcome.trace["action_pitch"]]
        )
        if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0 + 1e-7):
            raise RuntimeError(f"{name} produced an invalid direct plant action")
    if "batch_pid" in outcomes:
        count = int(outcomes["batch_pid"].summary["batch"]["measurement_count"])
        if count != expected_cursor["measurement_index"]:
            raise RuntimeError(f"batch history has {count} measurements, expected {expected_cursor['measurement_index']}")


def _aggregate(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], elapsed_s: float) -> Dict[str, Any]:
    by_arm: Dict[str, Dict[str, Any]] = {}
    for arm in contract["arms"]:
        selected = [row for row in rows if row["arm"] == arm]
        if not selected:
            continue
        by_arm[arm] = {
            "episodes": len(selected),
            "terminal_joint_success_count": int(sum(bool(row["terminal_joint_success"]) for row in selected)),
            "terminal_joint_success_rate": float(np.mean([bool(row["terminal_joint_success"]) for row in selected])),
            "dwell15_joint_success_rate": float(np.mean([bool(row["dwell15_joint_success"]) for row in selected])),
            "tail80_joint_success_rate": float(np.mean([bool(row["tail80_joint_success"]) for row in selected])),
            "terminal_formation_error_median": float(np.median([float(row["terminal_formation_error_m"]) for row in selected])),
            "terminal_localization_error_median": float(np.median([float(row["terminal_localization_error_m"]) for row in selected])),
            "mean_formation_error_after_120_m": float(np.mean([float(row["mean_formation_error_after_120_m"]) for row in selected])),
            "mean_localization_error_after_120_m": float(np.mean([float(row["mean_localization_error_after_120_m"]) for row in selected])),
            "mean_squared_action": float(np.mean([float(row["mean_squared_action"]) for row in selected])),
            "batch_runtime_total_s": float(np.sum([float(row.get("batch_total_runtime_s", 0.0)) for row in selected])),
        }
    deltas: Dict[str, Optional[float]] = {}
    if "pf_pid" in by_arm and "oracle_pid" in by_arm:
        deltas["oracle_minus_pf_terminal_joint_pp"] = 100.0 * (
            by_arm["oracle_pid"]["terminal_joint_success_rate"]
            - by_arm["pf_pid"]["terminal_joint_success_rate"]
        )
    if "batch_pid" in by_arm and "oracle_pid" in by_arm:
        deltas["oracle_minus_batch_terminal_joint_pp"] = 100.0 * (
            by_arm["oracle_pid"]["terminal_joint_success_rate"]
            - by_arm["batch_pid"]["terminal_joint_success_rate"]
        )
    if "pf_pid" in by_arm and "batch_pid" in by_arm:
        deltas["batch_minus_pf_terminal_joint_pp"] = 100.0 * (
            by_arm["batch_pid"]["terminal_joint_success_rate"]
            - by_arm["pf_pid"]["terminal_joint_success_rate"]
        )
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v20.VERSION,
        "completed_at_utc": _utc_now(),
        "status": "complete",
        "elapsed_wall_s": float(elapsed_s),
        "no_rl_training": True,
        "development_only": True,
        "episodes_per_arm": int(contract["episodes"]),
        "by_arm": by_arm,
        "paired_deltas": deltas,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    settings = _settings(args)
    cfg = _load_environment_config(Path(settings["metadata"]))
    estimator_config = v19.BatchEstimatorConfig(
        coarse_candidates=int(settings["coarse_candidates"]),
        coarse_sweeps=int(settings["coarse_sweeps"]),
        local_starts=int(settings["local_starts"]),
        gate_mode="raw",
    )
    contract = _contract(settings, cfg)
    _prepare_output(settings, contract)
    output = Path(settings["output"])
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    total_arms = int(settings["episodes"]) * len(settings["arms"])
    completed_arms = 0
    try:
        for local_index in range(int(settings["episodes"])):
            episode_index = int(settings["episode_start"]) + local_index
            seed = int(settings["seed_start"]) + episode_index
            tape = _tape_for_episode(output, cfg, seed, episode_index)
            outcomes: Dict[str, v20.ArmOutcome] = {}
            for arm in settings["arms"]:
                result_path = output / "episode_results" / f"episode_{episode_index:04d}_seed_{seed}_{arm}.json"
                trace_path = output / "traces_npz" / arm / f"episode_{episode_index:04d}_seed_{seed}.npz"
                if bool(settings["resume"]) and result_path.is_file() and trace_path.is_file():
                    summary = json.loads(result_path.read_text(encoding="utf-8"))
                    with np.load(trace_path, allow_pickle=False) as archive:
                        trace = {key: archive[key].copy() for key in archive.files}
                    outcome = v20.ArmOutcome(summary=summary, trace=trace)
                else:
                    outcome = v20.run_positioning_arm(
                        cfg=cfg,
                        tape=tape,
                        episode_seed=seed,
                        episode_index=episode_index,
                        arm=arm,
                        estimator_config=estimator_config,
                    )
                    _write_json_atomic(result_path, outcome.summary)
                    _write_npz_atomic(trace_path, outcome.trace)
                outcomes[arm] = outcome
                rows.append(_flatten_summary(outcome.summary))
                completed_arms += 1
                elapsed = time.perf_counter() - started
                rate = elapsed / max(completed_arms, 1)
                eta = rate * max(total_arms - completed_arms, 0)
                if completed_arms % int(settings["progress_every"]) == 0:
                    print(
                        f"[{_utc_now()}] completed {completed_arms}/{total_arms} arms; "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
                _write_json_atomic(
                    output / "control" / "progress.json",
                    {
                        "status": "running",
                        "updated_at_utc": _utc_now(),
                        "completed_arms": completed_arms,
                        "total_arms": total_arms,
                        "elapsed_wall_s": elapsed,
                        "estimated_remaining_s": eta,
                        "last_episode_index": episode_index,
                        "last_arm": arm,
                    },
                )
            _verify_pairing(outcomes, cfg)
        elapsed = time.perf_counter() - started
        _write_csv_atomic(output / "episode_arm_summary.csv", rows)
        summary = _aggregate(rows, contract, elapsed)
        _write_json_atomic(output / "campaign_summary.json", summary)
        _write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "complete",
                "updated_at_utc": _utc_now(),
                "completed_arms": completed_arms,
                "total_arms": total_arms,
                "elapsed_wall_s": elapsed,
                "estimated_remaining_s": 0.0,
            },
        )
        print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        elapsed = time.perf_counter() - started
        _write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "failed",
                "updated_at_utc": _utc_now(),
                "completed_arms": completed_arms,
                "total_arms": total_arms,
                "elapsed_wall_s": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
