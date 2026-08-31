#!/usr/bin/env python3
"""Run the paired V21 causal-lock development campaign."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from uuv_v18_resampling_guard import UUV3DConfig
import uuv_v19_observability as v19
import uuv_v20_positioning_ablation as v20
import run_v20_positioning_ablation as runner20
import uuv_v21_causal_lock as v21


RUNNER_VERSION = "v21_causal_lock_runner_1.0"
DEFAULT_SEED_START = 47_000
DEFAULT_EPISODES = 100
ARM_NAMES = ("causal_batch_pid", "fixed120_batch_pid", "pf_pid")
SOURCE_NAMES = (
    "run_v21_causal_lock.py",
    "uuv_v21_causal_lock.py",
    "EXPERIMENT_PROTOCOL_V21_CAUSAL_LOCK.md",
    "run_v20_positioning_ablation.py",
    "uuv_v20_positioning_ablation.py",
    "uuv_v19_observability.py",
    "baseline_controllers_v11.py",
    "uuv_v18_resampling_guard.py",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired causal-lock/fixed-lock/PF development arms."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--environment-metadata", type=Path)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARM_NAMES,
        default=list(ARM_NAMES),
    )
    parser.add_argument("--coarse-candidates", type=int)
    parser.add_argument("--coarse-sweeps", type=int)
    parser.add_argument("--local-starts", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    smoke = bool(args.smoke)
    episodes = int(args.episodes if args.episodes is not None else (2 if smoke else 100))
    output = (
        args.output_dir
        if args.output_dir is not None
        else root
        / (
            "experiments_v21_causal_lock_smoke"
            if smoke
            else "experiments_v21_causal_lock_dev100"
        )
    ).expanduser().resolve()
    metadata = (
        args.environment_metadata
        if args.environment_metadata is not None
        else runner20._default_metadata(root)
    ).expanduser().resolve()
    settings: Dict[str, Any] = {
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
        raise ValueError("episode count must be positive and episode start non-negative")
    first_seed = settings["seed_start"] + settings["episode_start"]
    last_seed = first_seed + episodes - 1
    for seed in (first_seed, last_seed):
        v19.assert_seed_is_not_sealed_final(seed)
    if first_seed <= v19.SEALED_FINAL_SEED_END and last_seed >= v19.SEALED_FINAL_SEED_START:
        raise PermissionError("V21 development run intersects the sealed final range")
    if not metadata.is_file():
        raise FileNotFoundError(f"missing frozen environment metadata: {metadata}")
    return settings


def _contract(
    settings: Mapping[str, Any],
    cfg: UUV3DConfig,
    lock_config: v21.CausalLockConfig,
) -> Dict[str, Any]:
    root = Path(settings["root"])
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v21.VERSION,
        "created_at_utc": runner20._utc_now(),
        "purpose": "development confirmation of truth-free ACQUIRE-to-TRACK lock",
        "seed_start": int(settings["seed_start"]),
        "episode_start": int(settings["episode_start"]),
        "episodes": int(settings["episodes"]),
        "arms": list(settings["arms"]),
        "global_refresh_s": list(v21.GLOBAL_REFRESH_S),
        "estimator_config": {
            "coarse_candidates": int(settings["coarse_candidates"]),
            "coarse_sweeps": int(settings["coarse_sweeps"]),
            "local_starts": int(settings["local_starts"]),
            "gate_mode": "raw",
            "candidate_radial_distribution": "uniform_radius",
        },
        "lock_config": lock_config.__dict__,
        "environment_metadata_path": str(settings["metadata"]),
        "environment_metadata_sha256": runner20._sha256(Path(settings["metadata"])),
        "fixed_horizon_actions": int(cfg.max_steps),
        "fixed_horizon_s": float(cfg.max_steps * cfg.action_dt),
        "source_sha256": {
            name: runner20._sha256(root / name) for name in SOURCE_NAMES
        },
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
    if output.exists() and not bool(settings["resume"]) and any(output.iterdir()):
        raise FileExistsError(f"output exists; pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if _immutable_contract(previous) != _immutable_contract(contract):
            raise RuntimeError("resume contract differs from existing V21 campaign")
    else:
        runner20._write_json_atomic(contract_path, contract)


def _augment_v20_outcome(
    outcome: v20.ArmOutcome,
    arm_name: str,
) -> v20.ArmOutcome:
    summary = dict(outcome.summary)
    trace = dict(outcome.trace)
    phase = np.asarray(trace["phase_track"], dtype=bool)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    times = np.asarray(trace["time_s"], dtype=np.float64)
    first = np.flatnonzero(phase)
    first_index = None if first.size == 0 else int(first[0])
    truth_ready = np.isfinite(localization) & (
        localization < v21.TERMINAL_LOCALIZATION_GATE_M
    )
    truth_ready_time = v21._first_window_time(
        times, truth_ready, v21.TRUTH_READY_DWELL_ACTIONS
    )
    first_action_time = None if first_index is None else float(times[first_index] - 2.0)
    false_mask = phase & (
        (~np.isfinite(localization))
        | (localization >= v21.TERMINAL_LOCALIZATION_GATE_M)
    )
    summary["arm"] = arm_name
    summary["gate"] = {
        "first_track_action_time_s": first_action_time,
        "first_track_localization_error_m": (
            None if first_index is None else float(localization[first_index])
        ),
        "truth_ready_time_s": truth_ready_time,
        "lock_delay_from_truth_ready_s": (
            None
            if first_action_time is None or truth_ready_time is None
            else float(first_action_time - truth_ready_time)
        ),
        "ever_locked": bool(first_index is not None),
        "false_lock_episode": bool(np.any(false_mask)),
        "false_locked_action_count": int(np.sum(false_mask)),
        "locked_action_count": int(np.sum(phase)),
        "lock_count": int(first_index is not None),
        "unlock_count": 0,
        "transitions": [],
    }
    return v20.ArmOutcome(summary=summary, trace=trace)


def _run_arm(
    *,
    arm: str,
    cfg: UUV3DConfig,
    tape: Any,
    seed: int,
    episode_index: int,
    estimator_config: v19.BatchEstimatorConfig,
    lock_config: v21.CausalLockConfig,
) -> Any:
    if arm == "causal_batch_pid":
        return v21.run_causal_lock_arm(
            cfg=cfg,
            tape=tape,
            episode_seed=seed,
            episode_index=episode_index,
            estimator_config=estimator_config,
            lock_config=lock_config,
        )
    v20_arm = "batch_pid" if arm == "fixed120_batch_pid" else "pf_pid"
    base = v20.run_positioning_arm(
        cfg=cfg,
        tape=tape,
        episode_seed=seed,
        episode_index=episode_index,
        arm=v20_arm,
        estimator_config=estimator_config,
    )
    return _augment_v20_outcome(base, arm)


def _flatten(summary: Mapping[str, Any]) -> Dict[str, Any]:
    gate = summary.get("gate", {})
    batch = summary.get("batch", {})
    excluded = {
        "gate",
        "batch",
        "noise_cursor",
        "initial_truth_m",
        "initial_leader_centroid_m",
        "state_at_track_start_m",
    }
    row = {key: value for key, value in summary.items() if key not in excluded}
    row.update(
        {
            "gate_first_track_action_time_s": gate.get("first_track_action_time_s"),
            "gate_first_track_localization_error_m": gate.get(
                "first_track_localization_error_m"
            ),
            "gate_truth_ready_time_s": gate.get("truth_ready_time_s"),
            "gate_lock_delay_from_truth_ready_s": gate.get(
                "lock_delay_from_truth_ready_s"
            ),
            "gate_ever_locked": gate.get("ever_locked", False),
            "gate_false_lock_episode": gate.get("false_lock_episode", False),
            "gate_false_locked_action_count": gate.get(
                "false_locked_action_count", 0
            ),
            "gate_locked_action_count": gate.get("locked_action_count", 0),
            "gate_lock_count": gate.get("lock_count", 0),
            "gate_unlock_count": gate.get("unlock_count", 0),
            "batch_total_runtime_s": batch.get("total_runtime_s", 0.0),
            "batch_maximum_update_runtime_s": batch.get(
                "maximum_update_runtime_s", 0.0
            ),
        }
    )
    return row


def _verify_pairing(outcomes: Mapping[str, Any], cfg: UUV3DConfig) -> None:
    names = list(outcomes)
    reference = outcomes[names[0]]
    expected_cursor = {
        "substep_index": int(cfg.max_steps * round(cfg.action_dt / cfg.sub_dt)),
        "measurement_index": int(round(cfg.max_steps * cfg.action_dt / cfg.s_meas_period)),
    }
    acquisition_actions = int(round(120.0 / float(cfg.action_dt)))
    for name, outcome in outcomes.items():
        if outcome.summary["noise_cursor"] != expected_cursor:
            raise RuntimeError(f"{name} consumed the wrong amount of noise")
        if outcome.summary["noise_tape_sha256"] != reference.summary["noise_tape_sha256"]:
            raise RuntimeError("paired arms used different noise tapes")
        for field in ("initial_truth_m", "initial_leader_centroid_m"):
            if not np.allclose(
                np.asarray(outcome.summary[field], dtype=np.float64),
                np.asarray(reference.summary[field], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"paired arms differ in {field}")
        for field in (
            "action_speed",
            "action_yaw",
            "action_pitch",
            "truth_x",
            "truth_y",
            "truth_z",
        ):
            if not np.allclose(
                np.asarray(outcome.trace[field])[:acquisition_actions],
                np.asarray(reference.trace[field])[:acquisition_actions],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"paired arms diverged before 120 s in {field}")


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    elapsed_s: float,
) -> Dict[str, Any]:
    by_arm: Dict[str, Dict[str, Any]] = {}
    for arm in contract["arms"]:
        selected = [row for row in rows if row["arm"] == arm]
        if not selected:
            continue
        lock_times = [
            float(row["gate_first_track_action_time_s"])
            for row in selected
            if row.get("gate_first_track_action_time_s") is not None
        ]
        lock_errors = [
            float(row["gate_first_track_localization_error_m"])
            for row in selected
            if row.get("gate_first_track_localization_error_m") is not None
        ]
        maximum_updates = [
            float(row.get("batch_maximum_update_runtime_s", 0.0))
            for row in selected
        ]
        by_arm[arm] = {
            "episodes": len(selected),
            "terminal_joint_success_count": int(
                sum(bool(row["terminal_joint_success"]) for row in selected)
            ),
            "terminal_joint_success_rate": float(
                np.mean([bool(row["terminal_joint_success"]) for row in selected])
            ),
            "dwell15_joint_success_rate": float(
                np.mean([bool(row["dwell15_joint_success"]) for row in selected])
            ),
            "tail80_joint_success_rate": float(
                np.mean([bool(row["tail80_joint_success"]) for row in selected])
            ),
            "ever_locked_count": int(
                sum(bool(row["gate_ever_locked"]) for row in selected)
            ),
            "false_lock_episode_count": int(
                sum(bool(row["gate_false_lock_episode"]) for row in selected)
            ),
            "false_locked_action_count": int(
                sum(int(row["gate_false_locked_action_count"]) for row in selected)
            ),
            "lock_time_s_median": (
                None if not lock_times else float(np.median(lock_times))
            ),
            "lock_time_s_p95": (
                None if not lock_times else float(np.percentile(lock_times, 95))
            ),
            "lock_time_s_max": None if not lock_times else float(max(lock_times)),
            "first_lock_error_m_p95": (
                None if not lock_errors else float(np.percentile(lock_errors, 95))
            ),
            "first_lock_error_m_max": (
                None if not lock_errors else float(max(lock_errors))
            ),
            "batch_runtime_total_s": float(
                sum(float(row.get("batch_total_runtime_s", 0.0)) for row in selected)
            ),
            "maximum_update_runtime_s": float(max(maximum_updates, default=0.0)),
        }
    causal = by_arm.get("causal_batch_pid")
    decision = "NOT_RUN"
    if causal is not None:
        passed = bool(
            causal["false_lock_episode_count"] == 0
            and causal["ever_locked_count"] >= 95
            and causal["terminal_joint_success_count"] >= 95
            and causal["maximum_update_runtime_s"] < 2.0
        )
        decision = "PASS_ENGINEERING_SCREEN" if passed else "FAIL_ENGINEERING_SCREEN"
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v21.VERSION,
        "completed_at_utc": runner20._utc_now(),
        "status": "complete",
        "decision": decision,
        "elapsed_wall_s": float(elapsed_s),
        "development_only": True,
        "no_rl_training": True,
        "episodes_per_arm": int(contract["episodes"]),
        "by_arm": by_arm,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    settings = _settings(args)
    cfg = runner20._load_environment_config(Path(settings["metadata"]))
    estimator_config = v19.BatchEstimatorConfig(
        coarse_candidates=int(settings["coarse_candidates"]),
        coarse_sweeps=int(settings["coarse_sweeps"]),
        local_starts=int(settings["local_starts"]),
        gate_mode="raw",
        candidate_radial_distribution="uniform_radius",
    )
    lock_config = v21.CausalLockConfig()
    contract = _contract(settings, cfg, lock_config)
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
            tape = runner20._tape_for_episode(output, cfg, seed, episode_index)
            outcomes: Dict[str, Any] = {}
            for arm in settings["arms"]:
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
                if bool(settings["resume"]) and result_path.is_file() and trace_path.is_file():
                    summary = json.loads(result_path.read_text(encoding="utf-8"))
                    with np.load(trace_path, allow_pickle=False) as archive:
                        trace = {key: archive[key].copy() for key in archive.files}
                    outcome = v21.ArmOutcome(summary=summary, trace=trace)
                else:
                    outcome = _run_arm(
                        arm=arm,
                        cfg=cfg,
                        tape=tape,
                        seed=seed,
                        episode_index=episode_index,
                        estimator_config=estimator_config,
                        lock_config=lock_config,
                    )
                    runner20._write_json_atomic(result_path, outcome.summary)
                    runner20._write_npz_atomic(trace_path, outcome.trace)
                outcomes[arm] = outcome
                rows.append(_flatten(outcome.summary))
                completed_arms += 1
                elapsed = time.perf_counter() - started
                seconds_per_arm = elapsed / max(completed_arms, 1)
                eta = seconds_per_arm * max(total_arms - completed_arms, 0)
                if completed_arms % int(settings["progress_every"]) == 0:
                    print(
                        f"[{runner20._utc_now()}] completed {completed_arms}/{total_arms} "
                        f"arms; elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
                runner20._write_json_atomic(
                    output / "control" / "progress.json",
                    {
                        "status": "running",
                        "updated_at_utc": runner20._utc_now(),
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
        runner20._write_csv_atomic(output / "episode_arm_summary.csv", rows)
        summary = _aggregate(rows, contract, elapsed)
        runner20._write_json_atomic(output / "campaign_summary.json", summary)
        runner20._write_json_atomic(
            output / "decision.json",
            {
                "decision": summary["decision"],
                "development_only": True,
                "sealed_final_range_untouched": [
                    v19.SEALED_FINAL_SEED_START,
                    v19.SEALED_FINAL_SEED_END,
                ],
            },
        )
        runner20._write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "complete",
                "updated_at_utc": runner20._utc_now(),
                "completed_arms": completed_arms,
                "total_arms": total_arms,
                "elapsed_wall_s": elapsed,
                "estimated_remaining_s": 0.0,
            },
        )
        print(json.dumps(runner20._json_safe(summary), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        elapsed = time.perf_counter() - started
        runner20._write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "failed",
                "updated_at_utc": runner20._utc_now(),
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
