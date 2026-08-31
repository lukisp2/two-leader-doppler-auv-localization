#!/usr/bin/env python3
"""Independently audit V38 source masks, causal metrics, and six-arm pairing."""

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
import uuv_v38_leader_source_ablation as v38


AUDITOR_VERSION = "v38_independent_causal_auditor_1.1"


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
    source: str,
    policy: str,
) -> Tuple[Path, Path]:
    arm = v38.arm_name(source, policy)
    return (
        campaign
        / "episode_results"
        / f"episode_{episode_index:04d}_seed_{seed}_{arm}.json",
        campaign
        / "traces_npz"
        / arm
        / f"episode_{episode_index:04d}_seed_{seed}.npz",
    )


def _full_history(trace: Mapping[str, np.ndarray]) -> v19.OnlineDopplerHistory:
    return v19.OnlineDopplerHistory(
        t_s=np.asarray(trace["online_t_s"], dtype=np.float64),
        dead_reckoned_displacement_m=np.asarray(
            trace["online_dead_reckoned_displacement_m"],
            dtype=np.float64,
        ),
        leader_position_m=np.asarray(
            trace["online_leader_position_m"],
            dtype=np.float64,
        ),
        leader_velocity_mps=np.asarray(
            trace["online_leader_velocity_mps"],
            dtype=np.float64,
        ),
        follower_velocity_measured_mps=np.asarray(
            trace["online_follower_velocity_measured_mps"],
            dtype=np.float64,
        ),
        doppler_measured_mps=np.asarray(
            trace["online_doppler_measured_mps"],
            dtype=np.float64,
        ),
        historical_pf_gate_factor=np.asarray(
            trace["online_historical_pf_gate_factor"],
            dtype=np.float64,
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
    transitions = gate & ~np.concatenate([np.asarray([False]), gate[:-1]])
    audit_release_violations = int(
        np.sum(
            transitions
            & ~np.asarray(trace["audit_release_checks_pass"], dtype=bool)
        )
    )
    radius = np.asarray(trace["batch_local_radius95_m"], dtype=np.float64)
    false_confidence = (
        np.isfinite(radius)
        & (radius < 7.0)
        & ((~np.isfinite(localization)) | (localization >= 7.0))
    )
    phase = np.asarray(trace["phase_track"], dtype=bool)
    track_errors = localization[phase & np.isfinite(localization)]
    return {
        "exact": exact,
        "terminal_joint_success": bool(joint[-1]),
        "tail80_joint_success": bool(float(np.mean(joint[-50:])) >= 0.8),
        "dwell15_joint_success": bool(np.all(joint[-15:])),
        "tail50_joint_occupancy": float(np.mean(joint[-50:])),
        "terminal_localization_error_m": float(localization[-1]),
        "terminal_formation_error_m": float(formation[-1]),
        "audit_release_violation_count": audit_release_violations,
        "false_confidence_action_count": int(np.sum(false_confidence)),
        "maximum_localization_error_m": float(
            np.nanmax(localization[np.isfinite(localization)])
        ),
        "maximum_track_localization_error_m": (
            None if track_errors.size == 0 else float(np.max(track_errors))
        ),
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
            raise RuntimeError(f"exact causal score differs: {key}")
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
                raise RuntimeError(f"exact score availability differs: {key}")
        elif not math.isclose(float(left), float(right), abs_tol=1e-10):
            raise RuntimeError(f"exact score value differs: {key}")


def _check_summary(summary: Mapping[str, Any], score: Mapping[str, Any]) -> None:
    for key in (
        "terminal_joint_success",
        "tail80_joint_success",
        "dwell15_joint_success",
    ):
        if bool(summary[key]) != bool(score[key]):
            raise RuntimeError(f"saved task score differs: {key}")
    for key in (
        "tail50_joint_occupancy",
        "terminal_localization_error_m",
        "terminal_formation_error_m",
    ):
        if not math.isclose(float(summary[key]), float(score[key]), abs_tol=1e-10):
            raise RuntimeError(f"saved task value differs: {key}")
    if int(summary["gate"]["audit_release_violation_count"]) != int(
        score["audit_release_violation_count"]
    ):
        raise RuntimeError("audit-release violation count differs")
    for key in (
        "false_confidence_action_count",
        "maximum_localization_error_m",
        "maximum_track_localization_error_m",
    ):
        left = summary["robustness"][key]
        right = score[key]
        if left is None or right is None:
            if left is not None or right is not None:
                raise RuntimeError(f"robustness availability differs: {key}")
        elif not math.isclose(float(left), float(right), abs_tol=1e-10):
            raise RuntimeError(f"robustness value differs: {key}")
    _check_exact(summary["gate"]["exact"], score["exact"])


def _check_masked_gate_metrics(
    trace: Mapping[str, np.ndarray],
    masked: v38.MaskedDopplerHistory,
) -> None:
    times = np.asarray(trace["time_s"], dtype=np.float64)
    estimates = np.column_stack(
        (
            trace["estimate_x"],
            trace["estimate_y"],
            trace["estimate_z"],
        )
    ).astype(np.float64)
    current_dead_reckoning = np.column_stack(
        (
            trace["current_dead_reckoning_x"],
            trace["current_dead_reckoning_y"],
            trace["current_dead_reckoning_z"],
        )
    ).astype(np.float64)
    full_saved = np.asarray(
        trace["batch_full_residual_rmse_mps"],
        dtype=np.float64,
    )
    recent_saved = np.asarray(
        trace["batch_recent_residual_rmse_mps"],
        dtype=np.float64,
    )
    full_counts = np.asarray(
        trace["gate_full_residual_scalar_count"],
        dtype=np.int64,
    )
    recent_counts = np.asarray(
        trace["gate_recent_residual_scalar_count"],
        dtype=np.int64,
    )
    causal_row_counts = np.asarray(
        trace["estimator_measurement_row_count"],
        dtype=np.int64,
    )
    active = masked.active_source_count
    for row, now in enumerate(times):
        count = int(causal_row_counts[row])
        if count < 1 or count > masked.measurement_count:
            raise RuntimeError("causal measurement-row count is out of bounds")
        if float(masked.t_s[count - 1]) > float(now) + 1e-7:
            raise RuntimeError("estimator used a future Doppler row")
        if now + 1e-9 < v21.ESTIMATION_START_S:
            if full_counts[row] != 0 or recent_counts[row] != 0:
                raise RuntimeError("gate residual count became active before 30 s")
            continue
        expected_full = count * active
        expected_recent = min(20, count) * active
        if full_counts[row] != expected_full:
            raise RuntimeError("full RMSE scalar denominator differs")
        if recent_counts[row] != expected_recent:
            raise RuntimeError("recent RMSE scalar denominator differs")
        if not np.all(np.isfinite(estimates[row])):
            raise RuntimeError("estimator missing after estimation start")
        history = masked.take(slice(0, count))
        initial_position = estimates[row] - current_dead_reckoning[row]
        residual = (
            history.doppler_measured_mps
            - v38.predict_doppler(initial_position, history)
        )
        audited_full = float(math.sqrt(float(np.mean(residual * residual))))
        recent = history.take(
            slice(max(0, history.measurement_count - 20), history.measurement_count)
        )
        recent_residual = (
            recent.doppler_measured_mps
            - v38.predict_doppler(initial_position, recent)
        )
        audited_recent = float(
            math.sqrt(float(np.mean(recent_residual * recent_residual)))
        )
        if not math.isclose(full_saved[row], audited_full, abs_tol=1e-9):
            raise RuntimeError("masked full residual RMSE differs")
        if not math.isclose(recent_saved[row], audited_recent, abs_tol=1e-9):
            raise RuntimeError("masked recent residual RMSE differs")


def _recompute_claim_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    integrity_valid: bool,
    smoke: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    if smoke:
        return (
            "SMOKE_PASS" if integrity_valid else "SMOKE_FAIL",
            "SMOKE_ONLY",
            {},
        )

    indexed: Dict[Tuple[str, str], Dict[int, Mapping[str, Any]]] = {}
    for source, policy in v38.arm_pairs():
        indexed[(source, policy)] = {
            int(row["episode_seed"]): row
            for row in rows
            if row["source_name"] == source and row["policy_name"] == policy
        }

    def rate(source: str, policy: str, key: str) -> float:
        values = list(indexed[(source, policy)].values())
        return float(np.mean([bool(row[key]) for row in values]))

    def p95(source: str, policy: str, key: str) -> float:
        values = [
            float(row[key]) for row in indexed[(source, policy)].values()
        ]
        return float(np.percentile(np.asarray(values, dtype=np.float64), 95))

    def paired_improvement(
        left: Tuple[str, str],
        right: Tuple[str, str],
        key: str,
    ) -> float:
        left_rows = indexed[left]
        right_rows = indexed[right]
        if set(left_rows) != set(right_rows):
            raise RuntimeError("independent claim contrast seed sets differ")
        return float(
            np.mean(
                [
                    float(left_rows[seed][key])
                    - float(right_rows[seed][key])
                    for seed in sorted(left_rows)
                ]
            )
        )

    both_active_rows = list(
        indexed[(v38.SOURCE_BOTH, v38.POLICY_ACTIVE)].values()
    )
    safety = bool(
        sum(int(row["unsafe_transition_count"]) for row in both_active_rows) == 0
        and sum(int(row["unsafe_track_start_count"]) for row in both_active_rows)
        == 0
        and sum(int(row["unsafe_track_end_count"]) for row in both_active_rows)
        == 0
    )
    operational = bool(
        rate(v38.SOURCE_BOTH, v38.POLICY_ACTIVE, "terminal_joint_success")
        >= 0.90
        and rate(v38.SOURCE_BOTH, v38.POLICY_ACTIVE, "tail80_joint_success")
        >= 0.85
        and p95(
            v38.SOURCE_BOTH,
            v38.POLICY_ACTIVE,
            "terminal_localization_error_m",
        )
        < 7.0
    )
    active_benefit = bool(
        rate(v38.SOURCE_BOTH, v38.POLICY_ACTIVE, "terminal_joint_success")
        - rate(v38.SOURCE_BOTH, v38.POLICY_FIXED, "terminal_joint_success")
        >= 0.05
        and paired_improvement(
            (v38.SOURCE_BOTH, v38.POLICY_FIXED),
            (v38.SOURCE_BOTH, v38.POLICY_ACTIVE),
            "terminal_localization_error_m",
        )
        > 0.0
    )
    active_source_checks: List[bool] = []
    checkpoint_source_checks: List[bool] = []
    for single in (v38.SOURCE_L1, v38.SOURCE_L2):
        active_source_checks.append(
            bool(
                rate(
                    v38.SOURCE_BOTH,
                    v38.POLICY_ACTIVE,
                    "terminal_joint_success",
                )
                - rate(single, v38.POLICY_ACTIVE, "terminal_joint_success")
                >= 0.05
                and paired_improvement(
                    (single, v38.POLICY_ACTIVE),
                    (v38.SOURCE_BOTH, v38.POLICY_ACTIVE),
                    "terminal_localization_error_m",
                )
                > 0.0
            )
        )
        checkpoint_source_checks.append(
            bool(
                rate(
                    v38.SOURCE_BOTH,
                    v38.POLICY_FIXED,
                    "checkpoint_60s_localization_below_7m",
                )
                - rate(
                    single,
                    v38.POLICY_FIXED,
                    "checkpoint_60s_localization_below_7m",
                )
                >= 0.05
                and paired_improvement(
                    (single, v38.POLICY_FIXED),
                    (v38.SOURCE_BOTH, v38.POLICY_FIXED),
                    "checkpoint_60s_localization_error_m",
                )
                > 0.0
            )
        )
    two_source_benefit = bool(
        all(active_source_checks) and all(checkpoint_source_checks)
    )
    support = bool(
        integrity_valid
        and safety
        and operational
        and active_benefit
        and two_source_benefit
    )
    inputs = {
        "safety": safety,
        "operational": operational,
        "active_benefit": active_benefit,
        "active_source_checks": active_source_checks,
        "checkpoint_source_checks": checkpoint_source_checks,
        "two_source_benefit": two_source_benefit,
    }
    return (
        "V38_COMPLETE" if integrity_valid else "V38_INVALID",
        (
            "SUPPORT_TWO_DOPPLER_REFERENCE_ACTIVE_LOCALIZATION_CLAIM"
            if support
            else "DO_NOT_SUPPORT_TWO_DOPPLER_REFERENCE_ACTIVE_LOCALIZATION_CLAIM"
        ),
        inputs,
    )


def audit(campaign: Path) -> Dict[str, Any]:
    root = Path(campaign).expanduser().resolve()
    contract_path = root / "control" / "campaign_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    issues: List[str] = []
    claim_rows: List[Dict[str, Any]] = []
    audited_rows = 0
    audited_pairs = 0
    source_root = Path(__file__).resolve().parent
    for relative, expected in contract["source_sha256"].items():
        path = source_root / relative
        if not path.is_file() or _sha256(path) != expected:
            issues.append(f"source hash differs: {relative}")
    try:
        if contract["reserved_final_range_untouched"] != [49_900, 50_999]:
            issues.append("reserved/final declaration differs")
        for seed in contract["seeds"]:
            v38.assert_seed_allowed(int(seed))
    except Exception as exc:
        issues.append(f"seed-boundary error: {exc}")

    recomputed_by_arm: Dict[str, Dict[str, int]] = {
        v38.arm_name(source, policy): {
            "episodes": 0,
            "terminal_success_count": 0,
            "tail80_success_count": 0,
            "checkpoint_60s_localization_success_count": 0,
            "unsafe_transition_count": 0,
            "unsafe_track_start_count": 0,
            "unsafe_track_end_count": 0,
        }
        for source, policy in v38.arm_pairs()
    }
    for local_index, seed in enumerate(contract["seeds"]):
        episode_index = int(contract["episode_start"]) + local_index
        tape_hashes = set()
        supports = set()
        fixed_traces: Dict[str, Dict[str, np.ndarray]] = {}
        arm_count = 0
        for source, policy in v38.arm_pairs():
            result_path, trace_path = _paths(
                root,
                episode_index,
                int(seed),
                source,
                policy,
            )
            try:
                if not result_path.is_file() or not trace_path.is_file():
                    raise FileNotFoundError(f"missing {source}/{policy}/seed {seed}")
                summary = json.loads(result_path.read_text(encoding="utf-8"))
                with np.load(trace_path, allow_pickle=False) as archive:
                    trace = {key: archive[key].copy() for key in archive.files}
                if np.asarray(trace["time_s"]).shape != (
                    int(contract["fixed_horizon_actions"]),
                ):
                    raise RuntimeError("fixed-horizon trace length differs")
                mask = v38.SOURCE_MASKS[source]
                if tuple(bool(value) for value in summary["source_mask"]) != mask:
                    raise RuntimeError("saved source mask differs")
                if not np.all(
                    np.asarray(trace["source_mask_l1"], dtype=bool) == mask[0]
                ) or not np.all(
                    np.asarray(trace["source_mask_l2"], dtype=bool) == mask[1]
                ):
                    raise RuntimeError("trace source mask differs")
                full = _full_history(trace)
                masked = v38.MaskedDopplerHistory.from_full(full, mask)
                if summary["masked_history_sha256"] != masked.content_sha256():
                    raise RuntimeError("masked history hash differs")
                if int(summary["measurement_row_count"]) != masked.measurement_count:
                    raise RuntimeError("measurement-row count differs")
                if (
                    int(summary["active_scalar_measurement_count"])
                    != masked.scalar_measurement_count
                ):
                    raise RuntimeError("active scalar count differs")
                _check_masked_gate_metrics(trace, masked)
                score = _trace_score(trace)
                _check_summary(summary, score)
                checkpoint = summary["checkpoint_60s"]
                checkpoint_indices = np.flatnonzero(
                    np.isclose(
                        np.asarray(trace["time_s"], dtype=np.float64),
                        v38.COMMON_PRE_RELEASE_CHECKPOINT_S,
                        rtol=0.0,
                        atol=1e-6,
                    )
                )
                if checkpoint_indices.size != 1:
                    raise RuntimeError("common pre-release checkpoint differs")
                checkpoint_index = int(checkpoint_indices[0])
                checkpoint_error = float(
                    np.asarray(
                        trace["localization_error_m"],
                        dtype=np.float64,
                    )[checkpoint_index]
                )
                if abs(
                    float(checkpoint["localization_error_m"])
                    - checkpoint_error
                ) > 1e-9:
                    raise RuntimeError("checkpoint localization error differs")
                if bool(checkpoint["localization_below_7m"]) != bool(
                    checkpoint_error < 7.0
                ):
                    raise RuntimeError("checkpoint localization endpoint differs")
                checkpoint_phase = bool(
                    np.asarray(trace["phase_track"], dtype=np.float64)[
                        checkpoint_index
                    ]
                    > 0.5
                )
                checkpoint_locked = bool(
                    np.asarray(
                        trace["gate_locked_after_update"],
                        dtype=np.float64,
                    )[checkpoint_index]
                    > 0.5
                )
                if (
                    bool(checkpoint["phase_track"]) != checkpoint_phase
                    or bool(checkpoint["gate_locked_after_update"])
                    != checkpoint_locked
                ):
                    raise RuntimeError("checkpoint release state differs")
                if policy == v38.POLICY_FIXED and (
                    checkpoint_phase or checkpoint_locked
                ):
                    raise RuntimeError("fixed arm released by 60-s checkpoint")
                support = summary["mission_support"]
                expected_center = v19.initial_leader_centroid_from_history(full)
                if not np.allclose(
                    np.asarray(support["center_m"], dtype=np.float64),
                    expected_center,
                    rtol=0.0,
                    atol=1e-9,
                ):
                    raise RuntimeError("mission-support center differs from leader centroid")
                if (
                    abs(
                        float(support["radius_min_m"])
                        - float(contract["mission_support"]["radius_min_m"])
                    )
                    > 1e-12
                    or abs(
                        float(support["radius_max_m"])
                        - float(contract["mission_support"]["radius_max_m"])
                    )
                    > 1e-12
                ):
                    raise RuntimeError("mission-support radii differ from contract")
                boundary = summary["information_boundary"]
                for key in (
                    "estimator_receives_masked_history_only",
                    "planner_information_model_uses_masked_sources_only",
                    "common_prior_uses_both_initial_leader_broadcasts",
                    "common_s_turn_uses_both_leader_velocity_broadcasts",
                    "planner_candidate_anchor_uses_common_s_turn",
                    "formation_reference_uses_both_leader_broadcasts",
                    "mission_support_uses_common_initial_leader_broadcasts",
                    "legacy_pf_excluded_from_controller",
                    "simulator_truth_scoring_only",
                ):
                    if not bool(boundary[key]):
                        raise RuntimeError(f"information-boundary flag differs: {key}")
                if policy == v38.POLICY_FIXED and int(
                    summary["planner"]["decision_count"]
                ) != 0:
                    raise RuntimeError("fixed arm invoked belief planner")
                if policy == v38.POLICY_FIXED:
                    fixed_traces[source] = trace
                tape_hashes.add(str(summary["noise_tape_sha256"]))
                supports.add(
                    json.dumps(summary["mission_support"], sort_keys=True)
                )
                arm = v38.arm_name(source, policy)
                aggregate = recomputed_by_arm[arm]
                aggregate["episodes"] += 1
                aggregate["terminal_success_count"] += int(
                    score["terminal_joint_success"]
                )
                aggregate["tail80_success_count"] += int(
                    score["tail80_joint_success"]
                )
                aggregate["checkpoint_60s_localization_success_count"] += int(
                    checkpoint_error < 7.0
                )
                exact = score["exact"]
                aggregate["unsafe_transition_count"] += int(
                    exact["false_transition_count"]
                )
                aggregate["unsafe_track_start_count"] += int(
                    exact["false_locked_action_start_count"]
                )
                aggregate["unsafe_track_end_count"] += int(
                    exact["false_locked_action_end_count"]
                )
                claim_rows.append(
                    {
                        "episode_seed": int(seed),
                        "source_name": source,
                        "policy_name": policy,
                        "terminal_joint_success": bool(
                            score["terminal_joint_success"]
                        ),
                        "tail80_joint_success": bool(
                            score["tail80_joint_success"]
                        ),
                        "terminal_localization_error_m": float(
                            score["terminal_localization_error_m"]
                        ),
                        "terminal_formation_error_m": float(
                            score["terminal_formation_error_m"]
                        ),
                        "checkpoint_60s_localization_error_m": checkpoint_error,
                        "checkpoint_60s_localization_below_7m": bool(
                            checkpoint_error < 7.0
                        ),
                        "unsafe_transition_count": int(
                            exact["false_transition_count"]
                        ),
                        "unsafe_track_start_count": int(
                            exact["false_locked_action_start_count"]
                        ),
                        "unsafe_track_end_count": int(
                            exact["false_locked_action_end_count"]
                        ),
                        "maximum_combined_decision_runtime_s": float(
                            summary["maximum_combined_decision_runtime_s"]
                        ),
                    }
                )
                audited_rows += 1
                arm_count += 1
            except Exception as exc:
                issues.append(f"seed {seed} {source}/{policy}: {exc}")
        if set(fixed_traces) == set(v38.SOURCE_NAMES):
            reference = fixed_traces[v38.SOURCE_L1]
            common_length = int(
                np.sum(
                    np.asarray(reference["time_s"], dtype=np.float64)
                    <= v38.COMMON_PRE_RELEASE_CHECKPOINT_S + 1e-6
                )
            )
            if np.any(
                np.asarray(reference["phase_track"], dtype=np.float64)[
                    :common_length
                ]
                > 0.5
            ):
                issues.append(
                    f"seed {seed}: TRACK started before common checkpoint"
                )
            for source in (v38.SOURCE_L2, v38.SOURCE_BOTH):
                candidate = fixed_traces[source]
                if np.any(
                    np.asarray(candidate["phase_track"], dtype=np.float64)[
                        :common_length
                    ]
                    > 0.5
                ):
                    issues.append(
                        f"seed {seed}: TRACK started before common checkpoint"
                    )
                for field in ("action_speed", "action_yaw", "action_pitch"):
                    if not np.array_equal(
                        np.asarray(reference[field])[:common_length],
                        np.asarray(candidate[field])[:common_length],
                    ):
                        issues.append(
                            f"seed {seed}: fixed S-turn differs through checkpoint"
                        )
                        break
                for field in ("truth_x", "truth_y", "truth_z"):
                    if not np.array_equal(
                        np.asarray(reference[field])[:common_length],
                        np.asarray(candidate[field])[:common_length],
                    ):
                        issues.append(
                            f"seed {seed}: fixed physical state differs through checkpoint"
                        )
                        break
                reference_online_length = int(
                    np.sum(
                        np.asarray(reference["online_t_s"], dtype=np.float64)
                        <= v38.COMMON_PRE_RELEASE_CHECKPOINT_S + 1e-6
                    )
                )
                candidate_online_length = int(
                    np.sum(
                        np.asarray(candidate["online_t_s"], dtype=np.float64)
                        <= v38.COMMON_PRE_RELEASE_CHECKPOINT_S + 1e-6
                    )
                )
                if candidate_online_length != reference_online_length:
                    issues.append(
                        f"seed {seed}: fixed online-history length differs"
                    )
                else:
                    for field in (
                        "online_t_s",
                        "online_dead_reckoned_displacement_m",
                        "online_leader_position_m",
                        "online_leader_velocity_mps",
                        "online_follower_velocity_measured_mps",
                        "online_doppler_measured_mps",
                        "online_historical_pf_gate_factor",
                    ):
                        if not np.array_equal(
                            np.asarray(reference[field])[
                                :reference_online_length
                            ],
                            np.asarray(candidate[field])[
                                :candidate_online_length
                            ],
                        ):
                            issues.append(
                                f"seed {seed}: fixed online history differs: {field}"
                            )
                            break
        else:
            issues.append(f"seed {seed}: fixed-arm action audit incomplete")
        if arm_count == len(v38.arm_pairs()) and len(tape_hashes) == 1 and len(supports) == 1:
            audited_pairs += 1
        else:
            issues.append(
                f"seed {seed}: six-arm tape/support pairing failed"
            )

    campaign_summary_path = root / "campaign_summary.json"
    saved_campaign: Optional[Mapping[str, Any]] = None
    if campaign_summary_path.is_file():
        saved_campaign = json.loads(
            campaign_summary_path.read_text(encoding="utf-8")
        )
        for arm, audited in recomputed_by_arm.items():
            saved = saved_campaign["by_arm"].get(arm)
            if saved is None:
                issues.append(f"campaign summary missing arm {arm}")
                continue
            for key, value in audited.items():
                if int(saved[key]) != int(value):
                    issues.append(f"campaign aggregate differs: {arm}/{key}")
    else:
        issues.append("campaign summary missing")

    expected_rows = int(contract["expected_runs"])
    preliminary_valid = bool(
        not issues
        and audited_rows == expected_rows
        and audited_pairs == int(contract["episodes"])
        and len(claim_rows) == expected_rows
        and bool(contract["publication_settings"])
        and bool(
            contract["smoke"]
            or contract["fresh_seed_audit"]["fresh"]
        )
        and max(
            (
                float(row["maximum_combined_decision_runtime_s"])
                for row in claim_rows
            ),
            default=float("inf"),
        )
        < 2.0
    )
    recomputed_decision, recomputed_claim, claim_inputs = (
        _recompute_claim_decision(
            claim_rows,
            integrity_valid=preliminary_valid,
            smoke=bool(contract["smoke"]),
        )
    )
    if saved_campaign is not None:
        if str(saved_campaign.get("decision")) != recomputed_decision:
            issues.append("runner decision differs from independent decision")
        if str(saved_campaign.get("claim_decision")) != recomputed_claim:
            issues.append("runner claim decision differs from independent decision")
        if bool(saved_campaign.get("integrity_valid")) != preliminary_valid:
            issues.append("runner integrity decision differs from independent audit")
    valid = bool(preliminary_valid and not issues)
    return {
        "auditor_version": AUDITOR_VERSION,
        "campaign": str(root),
        "valid": valid,
        "audited_rows": audited_rows,
        "expected_rows": expected_rows,
        "audited_six_arm_pairs": audited_pairs,
        "expected_six_arm_pairs": int(contract["episodes"]),
        "recomputed_by_arm": recomputed_by_arm,
        "recomputed_decision": recomputed_decision,
        "recomputed_claim_decision": recomputed_claim,
        "recomputed_claim_inputs": claim_inputs,
        "issue_count": len(issues),
        "issues": issues,
        "information_boundary_audited": True,
        "reserved_final_range_untouched": [v38.RESERVED_START, v38.FINAL_END],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    result = audit(args.campaign)
    output = Path(args.campaign).expanduser().resolve() / "independent_audit.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
