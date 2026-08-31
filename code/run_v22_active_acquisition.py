#!/usr/bin/env python3
"""Run the paired V22 S-turn versus belief-FIM acquisition campaign."""

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

import run_v20_positioning_ablation as runner20
import run_v21_causal_lock as runner21
import uuv_v19_observability as v19
import uuv_v21_causal_lock as v21
import uuv_v22_active_acquisition as v22


RUNNER_VERSION = "v22_active_acquisition_runner_1.0"
DEFAULT_SEED_START = 48_000
DEFAULT_EPISODES = 100
ARM_NAMES = ("s_turn_causal", "belief_fim_causal")
SOURCE_NAMES = (
    "run_v22_active_acquisition.py",
    "uuv_v22_active_acquisition.py",
    "EXPERIMENT_PROTOCOL_V22_ACTIVE_ACQUISITION.md",
    "run_v21_causal_lock.py",
    "uuv_v21_causal_lock.py",
    "EXPERIMENT_PROTOCOL_V21_CAUSAL_LOCK.md",
    "uuv_v20_positioning_ablation.py",
    "uuv_v19_observability.py",
    "uuv_v18_resampling_guard.py",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V22 active-acquisition ablation.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--environment-metadata", type=Path)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(ARM_NAMES))
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
        else root / ("experiments_v22_active_acquisition_smoke" if smoke else "experiments_v22_active_acquisition_dev100")
    ).expanduser().resolve()
    metadata = (
        args.environment_metadata
        if args.environment_metadata is not None
        else runner20._default_metadata(root)
    ).expanduser().resolve()
    settings = {
        "root": root,
        "output": output,
        "metadata": metadata,
        "seed_start": int(args.seed_start),
        "episode_start": int(args.episode_start),
        "episodes": episodes,
        "arms": list(args.arms),
        "coarse_candidates": int(args.coarse_candidates if args.coarse_candidates is not None else (256 if smoke else 4096)),
        "coarse_sweeps": int(args.coarse_sweeps if args.coarse_sweeps is not None else (1 if smoke else 2)),
        "local_starts": int(args.local_starts if args.local_starts is not None else (8 if smoke else 48)),
        "smoke": smoke,
        "resume": bool(args.resume),
        "progress_every": max(1, int(args.progress_every)),
    }
    first = settings["seed_start"] + settings["episode_start"]
    last = first + episodes - 1
    if episodes < 1 or settings["episode_start"] < 0:
        raise ValueError("invalid episode range")
    for seed in (first, last):
        v19.assert_seed_is_not_sealed_final(seed)
    if first <= v19.SEALED_FINAL_SEED_END and last >= v19.SEALED_FINAL_SEED_START:
        raise PermissionError("V22 intersects the sealed final range")
    if not metadata.is_file():
        raise FileNotFoundError(f"missing environment metadata: {metadata}")
    return settings


def _contract(
    settings: Mapping[str, Any],
    cfg: Any,
    estimator_config: v19.BatchEstimatorConfig,
    lock_config: v21.CausalLockConfig,
    planner_config: v22.ActivePlannerConfig,
) -> Dict[str, Any]:
    root = Path(settings["root"])
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v22.VERSION,
        "created_at_utc": runner20._utc_now(),
        "purpose": "development comparison of fixed and belief-FIM acquisition",
        "seed_start": int(settings["seed_start"]),
        "episode_start": int(settings["episode_start"]),
        "episodes": int(settings["episodes"]),
        "arms": list(settings["arms"]),
        "estimator_config": v19.config_to_dict(estimator_config),
        "lock_config": lock_config.__dict__,
        "planner_config": planner_config.__dict__,
        "environment_metadata_path": str(settings["metadata"]),
        "environment_metadata_sha256": runner20._sha256(Path(settings["metadata"])),
        "fixed_horizon_actions": int(cfg.max_steps),
        "source_sha256": {name: runner20._sha256(root / name) for name in SOURCE_NAMES},
        "sealed_final_seed_range_untouched": [v19.SEALED_FINAL_SEED_START, v19.SEALED_FINAL_SEED_END],
    }


def _immutable(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    return result


def _prepare(settings: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    output = Path(settings["output"])
    path = output / "control" / "campaign_contract.json"
    if output.exists() and not settings["resume"] and any(output.iterdir()):
        raise FileExistsError(f"output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if _immutable(previous) != _immutable(contract):
            raise RuntimeError("resume contract differs")
    else:
        runner20._write_json_atomic(path, contract)


def _rename_s_turn(outcome: v21.ArmOutcome) -> v21.ArmOutcome:
    summary = dict(outcome.summary)
    summary["arm"] = "s_turn_causal"
    summary["planner"] = {"decision_count": 0, "total_runtime_s": 0.0, "maximum_runtime_s": 0.0}
    return v21.ArmOutcome(summary=summary, trace=outcome.trace)


def _flatten(summary: Mapping[str, Any]) -> Dict[str, Any]:
    row = runner21._flatten(summary)
    row.pop("planner", None)
    planner = summary.get("planner", {})
    row.update({
        "planner_decision_count": planner.get("decision_count", 0),
        "planner_total_runtime_s": planner.get("total_runtime_s", 0.0),
        "planner_maximum_runtime_s": planner.get("maximum_runtime_s", 0.0),
        "mean_formation_error_after_30_m": summary.get("mean_formation_error_after_30_m"),
    })
    return row


def _verify_pairing(outcomes: Mapping[str, Any], cfg: Any) -> None:
    names = list(outcomes)
    reference = outcomes[names[0]]
    expected_cursor = {
        "substep_index": int(cfg.max_steps * round(cfg.action_dt / cfg.sub_dt)),
        "measurement_index": int(round(cfg.max_steps * cfg.action_dt / cfg.s_meas_period)),
    }
    common_actions = int(round(30.0 / float(cfg.action_dt)))
    for name, outcome in outcomes.items():
        if outcome.summary["noise_cursor"] != expected_cursor:
            raise RuntimeError(f"{name} consumed wrong noise count")
        if outcome.summary["noise_tape_sha256"] != reference.summary["noise_tape_sha256"]:
            raise RuntimeError("noise tape mismatch")
        if not np.allclose(outcome.summary["initial_truth_m"], reference.summary["initial_truth_m"], rtol=0.0, atol=1e-12):
            raise RuntimeError("initial truth mismatch")
        for field in ("action_speed", "action_yaw", "action_pitch", "truth_x", "truth_y", "truth_z"):
            if not np.allclose(np.asarray(outcome.trace[field])[:common_actions], np.asarray(reference.trace[field])[:common_actions], rtol=0.0, atol=1e-12):
                raise RuntimeError(f"arms diverged before active treatment in {field}")


def _arm_metrics(selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    lock_times = [float(row["gate_first_track_action_time_s"]) for row in selected if row.get("gate_first_track_action_time_s") is not None]
    delays = [float(row["gate_lock_delay_from_truth_ready_s"]) for row in selected if row.get("gate_lock_delay_from_truth_ready_s") is not None]
    runtime = [float(row.get("batch_maximum_update_runtime_s", 0.0)) + float(row.get("planner_maximum_runtime_s", 0.0)) for row in selected]
    return {
        "episodes": len(selected),
        "terminal_joint_success_count": int(sum(bool(row["terminal_joint_success"]) for row in selected)),
        "terminal_joint_success_rate": float(np.mean([bool(row["terminal_joint_success"]) for row in selected])),
        "dwell15_joint_success_rate": float(np.mean([bool(row["dwell15_joint_success"]) for row in selected])),
        "tail80_joint_success_rate": float(np.mean([bool(row["tail80_joint_success"]) for row in selected])),
        "ever_locked_count": int(sum(bool(row["gate_ever_locked"]) for row in selected)),
        "false_lock_episode_count": int(sum(bool(row["gate_false_lock_episode"]) for row in selected)),
        "false_locked_action_count": int(sum(int(row["gate_false_locked_action_count"]) for row in selected)),
        "lock_time_s_median": None if not lock_times else float(np.median(lock_times)),
        "lock_time_s_p95": None if not lock_times else float(np.percentile(lock_times, 95)),
        "lock_time_s_max": None if not lock_times else float(max(lock_times)),
        "lock_delay_s_median": None if not delays else float(np.median(delays)),
        "maximum_combined_decision_runtime_s": float(max(runtime, default=0.0)),
        "mean_squared_action": float(np.mean([float(row["mean_squared_action"]) for row in selected])),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], elapsed_s: float) -> Dict[str, Any]:
    by_arm = {arm: _arm_metrics([row for row in rows if row["arm"] == arm]) for arm in contract["arms"]}
    effect: Dict[str, Any] = {}
    decision = "NOT_RUN"
    if "s_turn_causal" in by_arm and "belief_fim_causal" in by_arm:
        fixed = by_arm["s_turn_causal"]
        active = by_arm["belief_fim_causal"]
        if fixed["lock_time_s_median"] is not None and active["lock_time_s_median"] is not None:
            improvement = float(fixed["lock_time_s_median"] - active["lock_time_s_median"])
            fraction = improvement / max(float(fixed["lock_time_s_median"]), 1e-9)
        else:
            improvement, fraction = float("nan"), float("nan")
        effect = {"median_lock_improvement_s": improvement, "median_lock_improvement_fraction": fraction}
        safe = active["false_lock_episode_count"] == 0 and active["maximum_combined_decision_runtime_s"] < 2.0
        effective = math.isfinite(improvement) and (improvement >= 30.0 or fraction >= 0.20)
        noninferior = active["terminal_joint_success_rate"] >= fixed["terminal_joint_success_rate"] - 0.02
        decision = "SUPPORT_ACTIVE_EFFECT" if safe and effective and noninferior else "NO_SUPPORTED_ACTIVE_EFFECT"
    return {
        "runner_version": RUNNER_VERSION,
        "experiment_version": v22.VERSION,
        "completed_at_utc": runner20._utc_now(),
        "status": "complete",
        "decision": decision,
        "elapsed_wall_s": float(elapsed_s),
        "development_only": True,
        "no_rl_training": True,
        "episodes_per_arm": int(contract["episodes"]),
        "by_arm": by_arm,
        "paired_effect": effect,
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
    planner_config = v22.ActivePlannerConfig()
    contract = _contract(settings, cfg, estimator_config, lock_config, planner_config)
    _prepare(settings, contract)
    output = Path(settings["output"])
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    total = int(settings["episodes"]) * len(settings["arms"])
    completed = 0
    try:
        for local_index in range(int(settings["episodes"])):
            episode_index = int(settings["episode_start"]) + local_index
            seed = int(settings["seed_start"]) + episode_index
            tape = runner20._tape_for_episode(output, cfg, seed, episode_index)
            outcomes: Dict[str, Any] = {}
            for arm in settings["arms"]:
                result_path = output / "episode_results" / f"episode_{episode_index:04d}_seed_{seed}_{arm}.json"
                trace_path = output / "traces_npz" / arm / f"episode_{episode_index:04d}_seed_{seed}.npz"
                if settings["resume"] and result_path.is_file() and trace_path.is_file():
                    summary = json.loads(result_path.read_text(encoding="utf-8"))
                    with np.load(trace_path, allow_pickle=False) as archive:
                        trace = {key: archive[key].copy() for key in archive.files}
                    outcome = v22.ArmOutcome(summary=summary, trace=trace)
                elif arm == "s_turn_causal":
                    outcome = _rename_s_turn(v21.run_causal_lock_arm(cfg=cfg, tape=tape, episode_seed=seed, episode_index=episode_index, estimator_config=estimator_config, lock_config=lock_config))
                else:
                    outcome = v22.run_active_acquisition_arm(cfg=cfg, tape=tape, episode_seed=seed, episode_index=episode_index, estimator_config=estimator_config, lock_config=lock_config, planner_config=planner_config)
                if not (settings["resume"] and result_path.is_file() and trace_path.is_file()):
                    runner20._write_json_atomic(result_path, outcome.summary)
                    runner20._write_npz_atomic(trace_path, outcome.trace)
                outcomes[arm] = outcome
                rows.append(_flatten(outcome.summary))
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / max(completed, 1) * max(total - completed, 0)
                if completed % int(settings["progress_every"]) == 0:
                    print(f"[{runner20._utc_now()}] completed {completed}/{total} arms; elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
                runner20._write_json_atomic(output / "control" / "progress.json", {"status": "running", "updated_at_utc": runner20._utc_now(), "completed_arms": completed, "total_arms": total, "elapsed_wall_s": elapsed, "estimated_remaining_s": eta, "last_episode_index": episode_index, "last_arm": arm})
            _verify_pairing(outcomes, cfg)
        elapsed = time.perf_counter() - started
        runner20._write_csv_atomic(output / "episode_arm_summary.csv", rows)
        summary = _aggregate(rows, contract, elapsed)
        runner20._write_json_atomic(output / "campaign_summary.json", summary)
        runner20._write_json_atomic(output / "decision.json", {"decision": summary["decision"], "development_only": True, "sealed_final_range_untouched": [v19.SEALED_FINAL_SEED_START, v19.SEALED_FINAL_SEED_END]})
        runner20._write_json_atomic(output / "control" / "progress.json", {"status": "complete", "updated_at_utc": runner20._utc_now(), "completed_arms": completed, "total_arms": total, "elapsed_wall_s": elapsed, "estimated_remaining_s": 0.0})
        print(json.dumps(runner20._json_safe(summary), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        elapsed = time.perf_counter() - started
        runner20._write_json_atomic(output / "control" / "progress.json", {"status": "failed", "updated_at_utc": runner20._utc_now(), "completed_arms": completed, "total_arms": total, "elapsed_wall_s": elapsed, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    sys.exit(main())
