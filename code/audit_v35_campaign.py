#!/usr/bin/env python3
"""Independently audit V35 artifacts without trusting runner decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import uuv_v19_observability as v19
import uuv_v21_causal_lock as v21
import uuv_v24_audited_gate as v24
import uuv_v28_estimator_stress as v28
import uuv_v35_closed_loop_stress as v35


AUDITOR_VERSION = "v35_independent_auditor_1.0"


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(
    campaign: Path,
    episode_index: int,
    seed: int,
    condition: str,
    arm: str,
) -> Tuple[Path, Path]:
    result = (
        campaign
        / "episode_results"
        / condition
        / f"episode_{episode_index:04d}_seed_{seed}_{arm}.json"
    )
    trace = (
        campaign
        / "traces_npz"
        / condition
        / arm
        / f"episode_{episode_index:04d}_seed_{seed}.npz"
    )
    return result, trace


def _history(trace: Mapping[str, np.ndarray]) -> v19.OnlineDopplerHistory:
    return v19.OnlineDopplerHistory(
        t_s=np.asarray(trace["online_t_s"], dtype=np.float64),
        dead_reckoned_displacement_m=np.asarray(
            trace["online_dead_reckoned_displacement_m"], dtype=np.float64
        ),
        leader_position_m=np.asarray(trace["online_leader_position_m"], dtype=np.float64),
        leader_velocity_mps=np.asarray(trace["online_leader_velocity_mps"], dtype=np.float64),
        follower_velocity_measured_mps=np.asarray(
            trace["online_follower_velocity_measured_mps"], dtype=np.float64
        ),
        doppler_measured_mps=np.asarray(trace["online_doppler_measured_mps"], dtype=np.float64),
        historical_pf_gate_factor=np.asarray(
            trace["online_historical_pf_gate_factor"], dtype=np.float64
        ),
    )


def _trace_score(trace: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    exact = v24.exact_trace_score(trace)
    formation = np.asarray(trace["formation_error_truth_m"], dtype=np.float64)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    joint = (formation < v21.TERMINAL_FORMATION_GATE_M) & (
        localization < v21.TERMINAL_LOCALIZATION_GATE_M
    )
    gate = np.asarray(trace["gate_locked_after_update"], dtype=bool)
    previous = np.concatenate([np.asarray([False]), gate[:-1]])
    transitions = gate & ~previous
    audit_violations = int(
        np.sum(
            transitions
            & ~np.asarray(trace["audit_release_checks_pass"], dtype=bool)
        )
    )
    times = np.asarray(trace["time_s"], dtype=np.float64)
    phase = np.asarray(trace["phase_track"], dtype=bool)
    post300_unsafe = int(
        np.sum(
            (times > v35.MODEL_DECISION_TIME_S + 1e-9)
            & phase
            & ((~np.isfinite(localization)) | (localization >= 7.0))
        )
    )
    return {
        "exact": exact,
        "terminal_joint_success": bool(joint[-1]),
        "tail80_joint_success": bool(float(np.mean(joint[-50:])) >= 0.8),
        "dwell15_joint_success": bool(np.all(joint[-15:])),
        "tail50_joint_occupancy": float(np.mean(joint[-50:])),
        "terminal_localization_error_m": float(localization[-1]),
        "terminal_formation_error_m": float(formation[-1]),
        "audit_release_violation_count": audit_violations,
        "post300_unsafe_track_end_count": post300_unsafe,
    }


def _check_exact(saved: Mapping[str, Any], audited: Mapping[str, Any]) -> None:
    for key in (
        "transition_count",
        "false_transition_count",
        "locked_action_count",
        "false_locked_action_start_count",
        "false_locked_action_end_count",
    ):
        if int(saved[key]) != int(audited[key]):
            raise RuntimeError(f"V35 exact metric differs: {key}")
    for key in (
        "first_transition_time_s",
        "first_transition_error_m",
        "maximum_transition_error_m",
        "first_locked_action_time_s",
        "maximum_locked_action_start_error_m",
        "maximum_locked_action_end_error_m",
    ):
        left = saved.get(key)
        right = audited.get(key)
        if left is None or right is None:
            if left is not None or right is not None:
                raise RuntimeError(f"V35 exact availability differs: {key}")
        elif not math.isclose(float(left), float(right), abs_tol=1e-10):
            raise RuntimeError(f"V35 exact value differs: {key}")


def _check_summary(summary: Mapping[str, Any], score: Mapping[str, Any]) -> None:
    for key in (
        "terminal_joint_success",
        "tail80_joint_success",
        "dwell15_joint_success",
    ):
        if bool(summary[key]) != bool(score[key]):
            raise RuntimeError(f"V35 summary differs: {key}")
    for key in (
        "tail50_joint_occupancy",
        "terminal_localization_error_m",
        "terminal_formation_error_m",
    ):
        if not math.isclose(float(summary[key]), float(score[key]), abs_tol=1e-10):
            raise RuntimeError(f"V35 summary differs: {key}")
    if int(summary["gate"]["audit_release_violation_count"]) != int(
        score["audit_release_violation_count"]
    ):
        raise RuntimeError("V35 audit-release violation count differs")
    if int(summary["gate"]["post300_unsafe_track_end_count"]) != int(
        score["post300_unsafe_track_end_count"]
    ):
        raise RuntimeError("V35 post-300 unsafe count differs")
    _check_exact(summary["gate"]["exact"], score["exact"])


def _compare_pair(
    primary: Mapping[str, np.ndarray], treatment: Mapping[str, np.ndarray]
) -> None:
    times = np.asarray(primary["time_s"], dtype=np.float64)
    mask = times <= v35.MODEL_DECISION_TIME_S + 1e-9
    for field in (
        "action_speed", "action_yaw", "action_pitch", "truth_x", "truth_y", "truth_z",
        "estimate_x", "estimate_y", "estimate_z", "phase_track",
        "gate_locked_after_update", "localization_error_m", "formation_error_truth_m",
    ):
        left = np.asarray(primary[field])[mask]
        right = np.asarray(treatment[field])[mask]
        if not np.array_equal(left, right, equal_nan=True):
            raise RuntimeError(f"V35 audited pair diverged by 300 s: {field}")
    primary_history = _history(primary).prefix(v35.MODEL_DECISION_TIME_S)
    treatment_history = _history(treatment).prefix(v35.MODEL_DECISION_TIME_S)
    if v28.history_sha256(primary_history) != v28.history_sha256(treatment_history):
        raise RuntimeError("V35 paired online histories differ through 300 s")


def audit(campaign: Path) -> Dict[str, Any]:
    root = Path(campaign).expanduser().resolve()
    contract_path = root / "control" / "campaign_contract.json"
    summary_path = root / "campaign_summary.json"
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    campaign_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else None
    )
    issues: List[str] = []
    audited_rows = 0
    pair_count = 0
    loaded: Dict[Tuple[int, str, str], Mapping[str, np.ndarray]] = {}
    source_root = Path(__file__).resolve().parent
    for relative, expected in contract["source_sha256"].items():
        path = source_root / relative
        if not path.is_file() or _sha256(path) != expected:
            issues.append(f"source hash differs: {relative}")
    try:
        if contract["reserved_final_range_untouched"] != [49_900, 50_999]:
            issues.append("reserved/final declaration differs")
        for seed in contract["seeds"]:
            v35.assert_v35_seed_allowed(int(seed))
    except Exception as exc:
        issues.append(f"seed-boundary error: {exc}")

    for local_index, seed in enumerate(contract["seeds"]):
        episode_index = int(contract["episode_start"]) + local_index
        for condition, arm in v35.arm_condition_pairs():
            result_path, trace_path = _paths(
                root, episode_index, int(seed), condition, arm
            )
            try:
                if not result_path.is_file() or not trace_path.is_file():
                    raise FileNotFoundError(f"missing {condition}/{arm}/seed {seed}")
                summary = json.loads(result_path.read_text(encoding="utf-8"))
                with np.load(trace_path, allow_pickle=False) as archive:
                    trace = {key: archive[key].copy() for key in archive.files}
                if np.asarray(trace["time_s"]).shape != (
                    int(contract["fixed_horizon_actions"]),
                ):
                    raise RuntimeError("fixed-horizon trace length differs")
                stress_tape = v35.make_stress_tape(condition, episode_index)
                if summary["stress_tape_sha256"] != stress_tape.content_sha256():
                    raise RuntimeError("stress tape hash differs")
                history = _history(trace)
                if summary["online_history_sha256"] != v28.history_sha256(history):
                    raise RuntimeError("online history hash differs")
                if int(summary["retained_measurement_count"]) != history.measurement_count:
                    raise RuntimeError("retained measurement count differs")
                if int(summary["raw_measurement_count"]) != 440:
                    raise RuntimeError("raw measurement count differs")
                score = _trace_score(trace)
                _check_summary(summary, score)
                if arm == v35.PRIMARY_ARM and bool(summary["model_switch"]["evaluated"]):
                    raise RuntimeError("primary arm evaluated the model switch")
                if arm == v35.BIAS_SWITCH_ARM:
                    marker = np.asarray(trace["model_decision_evaluated"], dtype=bool)
                    if int(np.sum(marker)) != 1 or not math.isclose(
                        float(np.asarray(trace["time_s"])[np.flatnonzero(marker)[0]]),
                        v35.MODEL_DECISION_TIME_S,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError("one-shot 300-s decision marker differs")
                loaded[(int(seed), condition, arm)] = trace
                audited_rows += 1
            except Exception as exc:
                issues.append(f"seed={seed} condition={condition} arm={arm}: {exc}")

        for condition in v35.BIAS_SWITCH_CONDITIONS:
            try:
                _compare_pair(
                    loaded[(int(seed), condition, v35.PRIMARY_ARM)],
                    loaded[(int(seed), condition, v35.BIAS_SWITCH_ARM)],
                )
                pair_count += 1
            except Exception as exc:
                issues.append(f"pair seed={seed} condition={condition}: {exc}")

    expected_rows = int(contract["expected_runs"])
    expected_pairs = int(contract["episodes"]) * len(v35.BIAS_SWITCH_CONDITIONS)
    if audited_rows != expected_rows:
        issues.append(f"audited row count {audited_rows} != {expected_rows}")
    if pair_count != expected_pairs:
        issues.append(f"audited pair count {pair_count} != {expected_pairs}")
    if campaign_summary is not None:
        if int(campaign_summary["row_count"]) != expected_rows:
            issues.append("campaign summary row count differs")
        if bool(campaign_summary["integrity_valid"]) != (not issues):
            issues.append("runner integrity flag differs from independent audit")

    result = {
        "schema_version": 1,
        "auditor_version": AUDITOR_VERSION,
        "campaign": str(root),
        "valid": not issues,
        "issues": issues,
        "audited_rows": audited_rows,
        "expected_rows": expected_rows,
        "audited_pairs": pair_count,
        "expected_pairs": expected_pairs,
        "reserved_final_range_untouched": contract["reserved_final_range_untouched"],
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    result = audit(args.campaign)
    output = Path(args.campaign).expanduser().resolve() / "independent_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

