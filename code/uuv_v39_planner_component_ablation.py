#!/usr/bin/env python3
"""Paired component ablations for the frozen two-link active-acquisition loop.

The module reuses the V38 two-link estimator, audited gate, measurement
boundary, formation controller, and fixed-horizon semantics.  It changes only
the ACQUIRE action selector.  Simulator truth is read after each completed
action for scoring and never enters an action decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from uuv_v11_rng import ExogenousNoiseTape
from uuv_v18_resampling_guard import UUV3DConfig, UUVTwoLeader3DPFEnv
import uuv_v19_observability as v19
import uuv_v20_positioning_ablation as v20
import uuv_v21_causal_lock as v21
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24
import uuv_v38_leader_source_ablation as v38


VERSION = "v39_planner_component_ablation_1.0"
ARM_FULL = "full_active"
ARM_NO_PAIR = "no_pair_term"
ARM_BEST_ONLY = "best_hypothesis_only"
ARM_RANDOM = "random_feasible"
ARM_NAMES: Tuple[str, ...] = (
    ARM_FULL,
    ARM_NO_PAIR,
    ARM_BEST_ONLY,
    ARM_RANDOM,
)
ACTIVE_ARMS: Tuple[str, ...] = (ARM_FULL, ARM_NO_PAIR, ARM_BEST_ONLY)
SOURCE_MASK: Tuple[bool, bool] = (True, True)
RESERVED_START = 49_900
FINAL_END = 50_999
POLICY_RNG_DOMAIN = "v39-random-feasible-policy-tape-1"
NO_LOCK_ANALYSIS_TIME_S = 442.0


def assert_seed_allowed(seed: int) -> None:
    value = int(seed)
    if RESERVED_START <= value <= FINAL_END:
        raise PermissionError(
            f"V39 refuses reserved/final seed {value}; "
            f"{RESERVED_START}..{FINAL_END} remain closed"
        )


@dataclass(frozen=True)
class PlannerConfig:
    """V22-compatible planner settings with a legal zero pair weight."""

    horizon_s: float = 30.0
    prediction_dt_s: float = 1.0
    maximum_hypotheses: int = 16
    hypothesis_cluster_radius_m: float = 1.0
    covariance_axis_scale: float = 2.0
    pair_separation_m: float = 7.0
    pair_weight: float = 0.25
    action_energy_weight: float = 0.05
    action_change_weight: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.horizon_s,
            self.prediction_dt_s,
            self.hypothesis_cluster_radius_m,
            self.covariance_axis_scale,
            self.pair_separation_m,
            self.action_energy_weight,
            self.action_change_weight,
        )
        if any(
            (not math.isfinite(float(value))) or float(value) <= 0.0
            for value in positive
        ):
            raise ValueError("positive planner constants must be finite")
        if (
            not math.isfinite(float(self.pair_weight))
            or float(self.pair_weight) < 0.0
        ):
            raise ValueError("pair weight must be finite and nonnegative")
        if int(self.maximum_hypotheses) < 1:
            raise ValueError("maximum_hypotheses must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def planner_config_for_arm(arm: str) -> Optional[PlannerConfig]:
    if arm == ARM_FULL:
        return PlannerConfig(maximum_hypotheses=16, pair_weight=0.25)
    if arm == ARM_NO_PAIR:
        return PlannerConfig(maximum_hypotheses=16, pair_weight=0.0)
    if arm == ARM_BEST_ONLY:
        # With one hypothesis no pair exists, so the nominal frozen weight is
        # mathematically inactive.  This arm changes only hypothesis capacity.
        return PlannerConfig(maximum_hypotheses=1, pair_weight=0.25)
    if arm == ARM_RANDOM:
        return None
    raise ValueError(f"unknown V39 arm {arm!r}")


def policy_seed_for_episode(episode_seed: int) -> int:
    payload = (
        f"{POLICY_RNG_DOMAIN}|episode_seed={int(episode_seed)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def policy_index_stream_sha256(
    policy_seed: int,
    candidate_counts: Sequence[int],
    candidate_indices: Sequence[int],
) -> str:
    counts = np.asarray(candidate_counts, dtype="<i8")
    indices = np.asarray(candidate_indices, dtype="<i8")
    if counts.shape != indices.shape:
        raise ValueError("candidate counts and indices must have equal shape")
    digest = hashlib.sha256()
    digest.update(POLICY_RNG_DOMAIN.encode("utf-8"))
    digest.update(np.asarray([int(policy_seed)], dtype="<u8").tobytes())
    digest.update(counts.tobytes())
    digest.update(indices.tobytes())
    return digest.hexdigest()


class RandomFeasiblePlanner:
    """Uniform uninformed selector over the exact frozen candidate bank."""

    def __init__(self, episode_seed: int) -> None:
        self.policy_seed = policy_seed_for_episode(int(episode_seed))
        self.rng = np.random.Generator(np.random.PCG64(self.policy_seed))
        self.candidate_counts: List[int] = []
        self.candidate_indices: List[int] = []
        self.total_runtime_s = 0.0
        self.maximum_runtime_s = 0.0
        self.decision_count = 0
        self.last_candidate_index = -1

    def action(
        self,
        env: UUVTwoLeader3DPFEnv,
        *,
        time_s: float,
    ) -> v22.PlannerDecision:
        started = time.perf_counter()
        fallback = v38.common_s_turn_action(env, float(time_s))
        candidates = v22._candidate_actions(fallback)
        count = int(candidates.shape[0])
        index = int(self.rng.integers(0, count))
        selected = np.asarray(candidates[index], dtype=np.float32)
        elapsed = float(time.perf_counter() - started)
        self.candidate_counts.append(count)
        self.candidate_indices.append(index)
        self.last_candidate_index = index
        self.total_runtime_s += elapsed
        self.maximum_runtime_s = max(self.maximum_runtime_s, elapsed)
        self.decision_count += 1
        return v22.PlannerDecision(
            action=selected,
            utility=0.0,
            worst_radius_before_m=0.0,
            worst_radius_after_m=0.0,
            minimum_pair_chi2=0.0,
            hypothesis_count=0,
            candidate_count=count,
            runtime_s=elapsed,
        )

    def tape_sha256(self) -> str:
        return policy_index_stream_sha256(
            self.policy_seed,
            self.candidate_counts,
            self.candidate_indices,
        )


class AuditedBeliefPlanner(v38.MaskedBeliefFIMPlanner):
    """Frozen belief planner that exposes a hash of each evaluated hypothesis set."""

    def __init__(
        self,
        config: PlannerConfig,
        source_mask: Sequence[bool],
    ) -> None:
        super().__init__(config, source_mask)
        self.last_hypothesis_sha256 = ""

    def _hypotheses(
        self,
        estimator: v38.LeaderMaskedStreamingEstimator,
        history: v38.MaskedDopplerHistory,
    ) -> np.ndarray:
        hypotheses = super()._hypotheses(estimator, history)
        digest = hashlib.sha256()
        digest.update(VERSION.encode("utf-8"))
        digest.update(np.asarray(hypotheses.shape, dtype="<i8").tobytes())
        digest.update(
            np.ascontiguousarray(hypotheses, dtype="<f8").tobytes()
        )
        self.last_hypothesis_sha256 = digest.hexdigest()
        return hypotheses


@dataclass(frozen=True)
class ArmOutcome:
    summary: Mapping[str, Any]
    trace: Mapping[str, np.ndarray]


def _audit_trace_values(
    gate: v38.LeaderMaskedAuditedGate,
) -> Dict[str, float]:
    return v38._audit_trace_values(gate)


def run_arm(
    *,
    cfg: UUV3DConfig,
    tape: ExogenousNoiseTape,
    episode_seed: int,
    episode_index: int,
    arm: str,
    estimator_config: v19.BatchEstimatorConfig,
    lock_config: v24.AuditedLockConfig,
) -> ArmOutcome:
    """Run one fixed-horizon arm using two admitted Doppler references."""

    if arm not in ARM_NAMES:
        raise ValueError(f"unknown V39 arm {arm!r}")
    assert_seed_allowed(episode_seed)
    planner_config = planner_config_for_arm(arm)
    env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    env.attach_exogenous_noise_tape(tape)
    env.reset(seed=int(episode_seed))
    support = v38.mission_support_from_initial_leaders(cfg, env.pL1, env.pL2)
    recorder = v20.OnlineHistoryRecorder(env)
    estimator = v38.LeaderMaskedStreamingEstimator(
        estimator_config=estimator_config,
        lock_config=lock_config,
        source_mask=SOURCE_MASK,
        support=support,
    )
    belief_planner: Optional[AuditedBeliefPlanner] = None
    random_planner: Optional[RandomFeasiblePlanner] = None
    if arm in ACTIVE_ARMS:
        if planner_config is None:
            raise RuntimeError("active arm has no planner configuration")
        belief_planner = AuditedBeliefPlanner(
            planner_config,
            SOURCE_MASK,
        )
    else:
        random_planner = RandomFeasiblePlanner(episode_seed)

    fields = (
        "step", "action_start_time_s", "time_s", "phase_track",
        "gate_locked_after_update", "action_speed", "action_yaw", "action_pitch",
        "truth_x", "truth_y", "truth_z", "estimate_x", "estimate_y", "estimate_z",
        "formation_error_truth_m", "localization_error_m", "batch_local_radius95_m",
        "planner_utility", "planner_worst_radius_before_m",
        "planner_worst_radius_after_m", "planner_minimum_pair_chi2",
        "planner_hypothesis_count", "planner_candidate_count",
        "planner_hypothesis_sha256", "planner_random_candidate_index",
        "planner_runtime_s",
        "gate_release_predicate", "gate_release_pass_streak",
        "gate_hold_predicate", "gate_hold_failure_streak",
        "audit_release_checks_pass", "audit_hold_checks_pass",
        "local_to_primary_m", "local_to_confirmation_m",
        "primary_to_confirmation_m", "primary_radius95_m",
        "confirmation_radius95_m", "primary_rmse_mps",
        "confirmation_rmse_mps", "global_evidence_time_s",
        "active_source_count", "gate_full_residual_scalar_count",
        "gate_recent_residual_scalar_count",
        "gate_forward_residual_scalar_count",
        "batch_full_residual_rmse_mps",
        "batch_recent_residual_rmse_mps",
        "estimator_measurement_row_count",
        "current_dead_reckoning_x", "current_dead_reckoning_y",
        "current_dead_reckoning_z",
        "source_mask_l1", "source_mask_l2",
    )
    trace_rows: Dict[str, List[Any]] = {field: [] for field in fields}
    actions: List[np.ndarray] = []
    maximum_combined_runtime_s = 0.0
    try:
        for _ in range(int(cfg.max_steps)):
            action_start = float(env.t)
            track = bool(estimator.gate.is_locked and estimator.has_solution)
            decision: Optional[v22.PlannerDecision] = None
            random_index = -1
            if track:
                action = v20.pid_action_for_position(
                    env,
                    np.asarray(
                        estimator.state.endpoint_position_m,
                        dtype=np.float64,
                    ),
                )
            elif not estimator.has_solution:
                action = v38.common_s_turn_action(env, action_start)
            elif belief_planner is not None:
                history_before = v38.MaskedDopplerHistory.from_full(
                    recorder.history(),
                    SOURCE_MASK,
                )
                decision = belief_planner.action(
                    env,
                    history=history_before,
                    current_dead_reckoning_m=(
                        recorder.accumulated_dead_reckoning_m
                    ),
                    estimator=estimator,
                    time_s=action_start,
                )
                action = decision.action
            else:
                if random_planner is None:
                    raise RuntimeError("random arm lost its action selector")
                decision = random_planner.action(env, time_s=action_start)
                random_index = int(random_planner.last_candidate_index)
                action = decision.action

            action = np.asarray(action, dtype=np.float32)
            if (
                action.shape != (3,)
                or not np.all(np.isfinite(action))
                or np.any(action < -1.0)
                or np.any(action > 1.0)
            ):
                raise RuntimeError("V39 produced an invalid action")
            actions.append(action.astype(np.float64))
            _, _, terminated, truncated, _ = env.step(action)
            masked_history = v38.MaskedDopplerHistory.from_full(
                recorder.history(),
                SOURCE_MASK,
            )
            update_started = time.perf_counter()
            estimator.update(
                masked_history,
                decision_time_s=float(env.step_count) * float(cfg.action_dt),
                current_dead_reckoning_m=(
                    recorder.accumulated_dead_reckoning_m
                ),
            )
            update_runtime = float(time.perf_counter() - update_started)
            combined_runtime = update_runtime + (
                0.0 if decision is None else float(decision.runtime_s)
            )
            maximum_combined_runtime_s = max(
                maximum_combined_runtime_s,
                combined_runtime,
            )

            # Truth is first read after the action and estimator update.
            truth = np.asarray(env.pF, dtype=np.float64).copy()
            endpoint = (
                np.asarray(
                    estimator.state.endpoint_position_m,
                    dtype=np.float64,
                ).copy()
                if estimator.has_solution
                else np.full(3, np.nan, dtype=np.float64)
            )
            _, desired = env._formation_desired()
            formation = float(np.linalg.norm(truth - np.asarray(desired)))
            localization = float(np.linalg.norm(endpoint - truth))
            mode = estimator.state.latest_mode
            values: Dict[str, Any] = {
                "step": int(env.step_count),
                "action_start_time_s": action_start,
                "time_s": float(env.t),
                "phase_track": float(track),
                "gate_locked_after_update": float(estimator.gate.is_locked),
                "action_speed": float(action[0]),
                "action_yaw": float(action[1]),
                "action_pitch": float(action[2]),
                "truth_x": float(truth[0]),
                "truth_y": float(truth[1]),
                "truth_z": float(truth[2]),
                "estimate_x": float(endpoint[0]),
                "estimate_y": float(endpoint[1]),
                "estimate_z": float(endpoint[2]),
                "formation_error_truth_m": formation,
                "localization_error_m": localization,
                "batch_local_radius95_m": (
                    float("nan")
                    if mode is None
                    else float(mode.local_radius95_m)
                ),
                "planner_utility": (
                    float("nan") if decision is None else decision.utility
                ),
                "planner_worst_radius_before_m": (
                    float("nan")
                    if decision is None
                    else decision.worst_radius_before_m
                ),
                "planner_worst_radius_after_m": (
                    float("nan")
                    if decision is None
                    else decision.worst_radius_after_m
                ),
                "planner_minimum_pair_chi2": (
                    float("nan")
                    if decision is None
                    else decision.minimum_pair_chi2
                ),
                "planner_hypothesis_count": (
                    0 if decision is None else decision.hypothesis_count
                ),
                "planner_candidate_count": (
                    0 if decision is None else decision.candidate_count
                ),
                "planner_hypothesis_sha256": (
                    ""
                    if decision is None or belief_planner is None
                    else belief_planner.last_hypothesis_sha256
                ),
                "planner_random_candidate_index": random_index,
                "planner_runtime_s": (
                    0.0 if decision is None else decision.runtime_s
                ),
                "batch_full_residual_rmse_mps": (
                    float("nan")
                    if mode is None
                    else float(mode.residual_rmse_mps)
                ),
                "batch_recent_residual_rmse_mps": (
                    float(estimator.state.recent_residual_rmse_mps)
                    if mode is not None
                    else float("nan")
                ),
                "estimator_measurement_row_count": int(
                    masked_history.measurement_count
                ),
                "current_dead_reckoning_x": float(
                    recorder.accumulated_dead_reckoning_m[0]
                ),
                "current_dead_reckoning_y": float(
                    recorder.accumulated_dead_reckoning_m[1]
                ),
                "current_dead_reckoning_z": float(
                    recorder.accumulated_dead_reckoning_m[2]
                ),
                "source_mask_l1": 1.0,
                "source_mask_l2": 1.0,
            }
            values.update(_audit_trace_values(estimator.gate))
            for field in fields:
                trace_rows[field].append(values[field])
            if terminated or truncated:
                if int(env.step_count) != int(cfg.max_steps):
                    raise RuntimeError("V39 arm ended before fixed horizon")
                break
        cursor = env._v11_noise_cursor
        if cursor is None:
            raise RuntimeError("V39 environment lost its noise tape")
        noise_cursor = dict(cursor.state_dict())
    finally:
        recorder.restore()
        env.close()

    trace = {
        field: np.asarray(values) for field, values in trace_rows.items()
    }
    if trace["time_s"].shape != (int(cfg.max_steps),):
        raise RuntimeError("V39 trace does not contain the fixed horizon")
    full_history = recorder.history()
    masked_history = v38.MaskedDopplerHistory.from_full(
        full_history,
        SOURCE_MASK,
    )
    trace.update(
        {
            "online_t_s": full_history.t_s.copy(),
            "online_dead_reckoned_displacement_m": (
                full_history.dead_reckoned_displacement_m.copy()
            ),
            "online_leader_position_m": full_history.leader_position_m.copy(),
            "online_leader_velocity_mps": (
                full_history.leader_velocity_mps.copy()
            ),
            "online_follower_velocity_measured_mps": (
                full_history.follower_velocity_measured_mps.copy()
            ),
            "online_doppler_measured_mps": (
                full_history.doppler_measured_mps.copy()
            ),
            "online_historical_pf_gate_factor": (
                full_history.historical_pf_gate_factor.copy()
            ),
        }
    )
    times = np.asarray(trace["time_s"], dtype=np.float64)
    formation = np.asarray(trace["formation_error_truth_m"], dtype=np.float64)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    joint = (formation < v21.TERMINAL_FORMATION_GATE_M) & (
        localization < v21.TERMINAL_LOCALIZATION_GATE_M
    )
    exact = v24.exact_trace_score(trace)
    gate_after = np.asarray(trace["gate_locked_after_update"], dtype=bool)
    transitions = gate_after & ~np.concatenate(
        [np.asarray([False]), gate_after[:-1]]
    )
    audit_release_violation_count = int(
        np.sum(
            transitions
            & ~np.asarray(trace["audit_release_checks_pass"], dtype=bool)
        )
    )
    phase = np.asarray(trace["phase_track"], dtype=bool)
    finite_localization = np.isfinite(localization)
    track_errors = localization[phase & finite_localization]
    false_confidence = (
        np.isfinite(trace["batch_local_radius95_m"])
        & (np.asarray(trace["batch_local_radius95_m"]) < 7.0)
        & ((~finite_localization) | (localization >= 7.0))
    )
    actions_array = np.asarray(actions, dtype=np.float64)
    planner = belief_planner if belief_planner is not None else random_planner
    if planner is None:
        raise RuntimeError("V39 arm has no acquisition selector")
    random_metadata: Optional[Dict[str, Any]] = None
    if random_planner is not None:
        random_metadata = {
            "policy_rng_domain": POLICY_RNG_DOMAIN,
            "policy_seed": int(random_planner.policy_seed),
            "candidate_counts": list(random_planner.candidate_counts),
            "candidate_indices": list(random_planner.candidate_indices),
            "policy_index_stream_sha256": random_planner.tape_sha256(),
        }
    summary: Dict[str, Any] = {
        "version": VERSION,
        "arm": arm,
        "source_name": v38.SOURCE_BOTH,
        "source_mask": list(SOURCE_MASK),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "noise_tape_sha256": tape.content_sha256(),
        "masked_history_sha256": masked_history.content_sha256(),
        "action_count": int(times.size),
        "measurement_row_count": int(masked_history.measurement_count),
        "active_scalar_measurement_count": int(
            masked_history.scalar_measurement_count
        ),
        "noise_cursor": noise_cursor,
        "mission_support": support.to_dict(),
        "terminal_formation_error_m": float(formation[-1]),
        "terminal_localization_error_m": float(localization[-1]),
        "terminal_joint_success": bool(joint[-1]),
        "dwell15_joint_success": bool(np.all(joint[-v21.DWELL_ACTIONS :])),
        "tail50_joint_occupancy": float(
            np.mean(joint[-v21.TAIL_WINDOW_ACTIONS :])
        ),
        "tail80_joint_success": bool(
            float(np.mean(joint[-v21.TAIL_WINDOW_ACTIONS :])) >= 0.8
        ),
        "time_to_sustained_joint_lock_s": v21._first_sustained_time(
            times,
            joint,
        ),
        "mean_squared_action": float(
            np.mean(np.sum(actions_array * actions_array, axis=1))
        ),
        "mean_formation_error_after_30_m": float(
            np.mean(formation[times >= 30.0])
        ),
        "maximum_combined_decision_runtime_s": float(
            maximum_combined_runtime_s
        ),
        "robustness": {
            "maximum_localization_error_m": float(
                np.nanmax(localization[finite_localization])
            ),
            "maximum_track_localization_error_m": (
                None if track_errors.size == 0 else float(np.max(track_errors))
            ),
            "false_confidence_action_count": int(np.sum(false_confidence)),
            "terminal_localization_below_7m": bool(localization[-1] < 7.0),
            "terminal_formation_below_8m": bool(formation[-1] < 8.0),
        },
        "gate": {
            "ever_locked": bool(exact["transition_count"] > 0),
            "first_track_action_time_s": exact["first_locked_action_time_s"],
            "first_track_analysis_time_s": (
                NO_LOCK_ANALYSIS_TIME_S
                if exact["first_locked_action_time_s"] is None
                else float(exact["first_locked_action_time_s"])
            ),
            "lock_count": int(estimator.gate.state.lock_count),
            "unlock_count": int(estimator.gate.state.unlock_count),
            "audit_release_violation_count": audit_release_violation_count,
            "exact": exact,
            "transitions": list(estimator.gate.state.transitions),
            "full_residual_scalar_count": int(
                estimator.gate.full_residual_scalar_count
            ),
            "recent_residual_scalar_count": int(
                estimator.gate.recent_residual_scalar_count
            ),
            "forward_residual_scalar_count": int(
                estimator.gate.forward_residual_scalar_count
            ),
        },
        "batch": {
            "global_solve_count": int(estimator.state.global_solve_count),
            "local_solve_count": int(estimator.state.local_solve_count),
            "total_runtime_s": float(estimator.state.total_runtime_s),
            "maximum_update_runtime_s": float(
                estimator.state.maximum_update_runtime_s
            ),
            "global_records": list(estimator.global_records),
        },
        "planner": {
            "kind": (
                "belief_conditioned"
                if belief_planner is not None
                else "uniform_random_feasible"
            ),
            "config": (
                None if planner_config is None else planner_config.to_dict()
            ),
            "decision_count": int(planner.decision_count),
            "total_runtime_s": float(planner.total_runtime_s),
            "maximum_runtime_s": float(planner.maximum_runtime_s),
            "maximum_saved_hypothesis_count": int(
                np.max(trace["planner_hypothesis_count"], initial=0)
            ),
            "random_policy": random_metadata,
        },
        "information_boundary": {
            "both_doppler_links_admitted": True,
            "estimator_and_gate_identical_across_arms": True,
            "track_pid_identical_across_arms": True,
            "random_selector_receives_belief": False,
            "random_selector_receives_truth": False,
            "legacy_pf_excluded_from_controller": True,
            "simulator_truth_scoring_only": True,
        },
        "sealed_seed_range_untouched": [RESERVED_START, FINAL_END],
    }
    return ArmOutcome(summary=summary, trace=trace)


__all__ = [
    "VERSION",
    "ARM_FULL",
    "ARM_NO_PAIR",
    "ARM_BEST_ONLY",
    "ARM_RANDOM",
    "ARM_NAMES",
    "ACTIVE_ARMS",
    "SOURCE_MASK",
    "RESERVED_START",
    "FINAL_END",
    "POLICY_RNG_DOMAIN",
    "NO_LOCK_ANALYSIS_TIME_S",
    "PlannerConfig",
    "RandomFeasiblePlanner",
    "AuditedBeliefPlanner",
    "ArmOutcome",
    "assert_seed_allowed",
    "planner_config_for_arm",
    "policy_seed_for_episode",
    "policy_index_stream_sha256",
    "run_arm",
]
