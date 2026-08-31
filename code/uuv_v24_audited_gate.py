#!/usr/bin/env python3
"""Audited ACQUIRE-to-TRACK gate for two-leader Doppler navigation.

V24 leaves the frozen V22 estimator, active-acquisition planner and PID
controller unchanged.  Its treatment is limited to the lock state machine:
the current local solution must agree with both independently seeded global
solutions, and both global solutions must satisfy explicit radius/RMSE limits.
Simulator truth is used only after each action to score the experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from uuv_v11_rng import ExogenousNoiseTape
from uuv_v18_resampling_guard import UUV3DConfig, UUVTwoLeader3DPFEnv
import uuv_v19_observability as v19
import uuv_v20_positioning_ablation as v20
import uuv_v21_causal_lock as v21
import uuv_v22_active_acquisition as v22


VERSION = "v24_audited_gate_1.0"


@dataclass(frozen=True)
class AuditedLockConfig:
    """Prespecified V24 gate, including the unchanged V23-early base gate."""

    base: v21.CausalLockConfig = field(
        default_factory=lambda: v21.CausalLockConfig(
            minimum_release_time_s=60.0,
            release_search_agreement_m=2.0,
            release_global_stability_m=3.0,
            release_alternative_delta_chi2=13.82,
        )
    )
    release_local_global_agreement_m: float = 2.0
    release_global_radius95_m: float = 7.0
    release_global_rmse_mps: float = 0.08
    hold_local_global_agreement_m: float = 7.0
    hold_global_radius95_m: float = 10.0
    hold_global_rmse_mps: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.release_local_global_agreement_m,
            self.release_global_radius95_m,
            self.release_global_rmse_mps,
            self.hold_local_global_agreement_m,
            self.hold_global_radius95_m,
            self.hold_global_rmse_mps,
        )
        if any(
            (not math.isfinite(float(value))) or float(value) <= 0.0
            for value in values
        ):
            raise ValueError("audited lock thresholds must be finite and positive")
        if self.hold_local_global_agreement_m < self.release_local_global_agreement_m:
            raise ValueError("hold agreement must not be tighter than release agreement")
        if self.hold_global_radius95_m < self.release_global_radius95_m:
            raise ValueError("hold radius must not be tighter than release radius")
        if self.hold_global_rmse_mps < self.release_global_rmse_mps:
            raise ValueError("hold RMSE must not be tighter than release RMSE")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": dict(self.base.__dict__),
            "release_local_global_agreement_m": float(
                self.release_local_global_agreement_m
            ),
            "release_global_radius95_m": float(self.release_global_radius95_m),
            "release_global_rmse_mps": float(self.release_global_rmse_mps),
            "hold_local_global_agreement_m": float(
                self.hold_local_global_agreement_m
            ),
            "hold_global_radius95_m": float(self.hold_global_radius95_m),
            "hold_global_rmse_mps": float(self.hold_global_rmse_mps),
        }


class AuditedCausalLockGate(v21.CausalLockGate):
    """V21 truth-free state machine with explicit local/global audit checks."""

    AUDIT_RELEASE_CHECKS: Tuple[str, ...] = (
        "local_primary_agreement",
        "local_confirmation_agreement",
        "primary_radius95",
        "confirmation_radius95",
        "primary_rmse",
        "confirmation_rmse",
    )
    AUDIT_HOLD_CHECKS: Tuple[str, ...] = AUDIT_RELEASE_CHECKS

    def __init__(self, config: AuditedLockConfig) -> None:
        super().__init__(config.base)
        self.audit_config = config
        self.last_metrics: Dict[str, float] = {}
        self.last_release_checks: Dict[str, bool] = {}
        self.last_hold_checks: Dict[str, bool] = {}

    @staticmethod
    def _finite_at_most(value: float, threshold: float) -> bool:
        return bool(math.isfinite(float(value)) and float(value) <= float(threshold))

    def _audit_checks(
        self,
        *,
        local_position_m: np.ndarray,
        evidence: v21.GlobalLockEvidence,
        release: bool,
    ) -> Dict[str, bool]:
        primary = evidence.primary.best
        confirmation = evidence.confirmation.best
        primary_position = np.asarray(primary.initial_position_m, dtype=np.float64)
        confirmation_position = np.asarray(
            confirmation.initial_position_m, dtype=np.float64
        )
        local_primary = float(np.linalg.norm(local_position_m - primary_position))
        local_confirmation = float(
            np.linalg.norm(local_position_m - confirmation_position)
        )
        self.last_metrics = {
            "local_to_primary_m": local_primary,
            "local_to_confirmation_m": local_confirmation,
            "primary_to_confirmation_m": float(
                np.linalg.norm(primary_position - confirmation_position)
            ),
            "primary_radius95_m": float(primary.local_radius95_m),
            "confirmation_radius95_m": float(confirmation.local_radius95_m),
            "primary_rmse_mps": float(primary.residual_rmse_mps),
            "confirmation_rmse_mps": float(confirmation.residual_rmse_mps),
            "global_evidence_time_s": float(evidence.time_s),
        }
        if release:
            agreement = float(self.audit_config.release_local_global_agreement_m)
            radius = float(self.audit_config.release_global_radius95_m)
            rmse = float(self.audit_config.release_global_rmse_mps)
        else:
            agreement = float(self.audit_config.hold_local_global_agreement_m)
            radius = float(self.audit_config.hold_global_radius95_m)
            rmse = float(self.audit_config.hold_global_rmse_mps)
        return {
            "local_primary_agreement": self._finite_at_most(
                local_primary, agreement
            ),
            "local_confirmation_agreement": self._finite_at_most(
                local_confirmation, agreement
            ),
            "primary_radius95": self._finite_at_most(
                primary.local_radius95_m, radius
            ),
            "confirmation_radius95": self._finite_at_most(
                confirmation.local_radius95_m, radius
            ),
            "primary_rmse": self._finite_at_most(primary.residual_rmse_mps, rmse),
            "confirmation_rmse": self._finite_at_most(
                confirmation.residual_rmse_mps, rmse
            ),
        }

    def evaluate(
        self,
        *,
        now_s: float,
        mode: v19.BatchMode,
        initial_position_m: Sequence[float],
        recent_residual_rmse_mps: float,
        global_evidence: Optional[v21.GlobalLockEvidence],
    ) -> bool:
        now = float(now_s)
        position = np.asarray(initial_position_m, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("gate initial position must contain three finite values")
        mode_position = np.asarray(mode.initial_position_m, dtype=np.float64)
        if not np.allclose(position, mode_position, rtol=0.0, atol=1e-10):
            raise ValueError("gate position differs from the audited local mode")
        stable = self._local_stability(position)
        release_checks: Dict[str, bool] = {
            "minimum_time": now + 1e-9
            >= float(self.config.minimum_release_time_s),
            "local_valid": v21._mode_is_locally_valid(mode),
            "local_radius95": self._finite_at_most(
                mode.local_radius95_m, self.config.release_radius95_m
            ),
            "full_rmse": self._finite_at_most(
                mode.residual_rmse_mps, self.config.release_full_rmse_mps
            ),
            "recent_rmse": self._finite_at_most(
                recent_residual_rmse_mps, self.config.release_recent_rmse_mps
            ),
            "local_stability": stable,
            "global_available": global_evidence is not None,
        }
        if global_evidence is not None:
            release_checks.update(global_evidence.release_checks(now, self.config))
            release_checks.update(
                self._audit_checks(
                    local_position_m=position,
                    evidence=global_evidence,
                    release=True,
                )
            )
        else:
            self.last_metrics = {}
        self.last_release_checks = dict(release_checks)
        release_ok = bool(all(release_checks.values()))
        self.state.last_release_predicate = release_ok

        if not self.state.locked:
            self.last_hold_checks = {}
            self.state.release_pass_streak = (
                self.state.release_pass_streak + 1 if release_ok else 0
            )
            self.state.hold_failure_streak = 0
            self.state.last_failed_checks = tuple(
                name for name, passed in release_checks.items() if not passed
            )
            if self.state.release_pass_streak >= int(
                self.config.release_consecutive_actions
            ):
                self.state.locked = True
                self.state.lock_count += 1
                self.state.last_lock_time_s = now
                if self.state.first_lock_time_s is None:
                    self.state.first_lock_time_s = now
                self.state.transitions.append(
                    {
                        "time_s": now,
                        "from": "ACQUIRE",
                        "to": "TRACK",
                        "audit_metrics": dict(self.last_metrics),
                    }
                )
                self.state.last_failed_checks = ()
            return self.state.locked

        hold_checks: Dict[str, bool] = {
            "local_valid": v21._mode_is_locally_valid(mode),
            "local_radius95": self._finite_at_most(
                mode.local_radius95_m, self.config.hold_radius95_m
            ),
            "full_rmse": self._finite_at_most(
                mode.residual_rmse_mps, self.config.hold_full_rmse_mps
            ),
            "recent_rmse": self._finite_at_most(
                recent_residual_rmse_mps, self.config.hold_recent_rmse_mps
            ),
            "global_available": global_evidence is not None,
        }
        if global_evidence is not None:
            hold_checks.update(global_evidence.hold_checks(now, self.config))
            hold_checks.update(
                self._audit_checks(
                    local_position_m=position,
                    evidence=global_evidence,
                    release=False,
                )
            )
        self.last_hold_checks = dict(hold_checks)
        hold_ok = bool(all(hold_checks.values()))
        self.state.last_hold_predicate = hold_ok
        self.state.last_failed_checks = tuple(
            name for name, passed in hold_checks.items() if not passed
        )
        self.state.hold_failure_streak = (
            0 if hold_ok else self.state.hold_failure_streak + 1
        )
        material_disagreement = bool(
            global_evidence is not None
            and (
                not hold_checks.get("local_primary_agreement", False)
                or not hold_checks.get("local_confirmation_agreement", False)
            )
        )
        if material_disagreement or self.state.hold_failure_streak >= int(
            self.config.loss_consecutive_actions
        ):
            failed = list(self.state.last_failed_checks)
            self.state.locked = False
            self.state.unlock_count += 1
            self.state.last_unlock_time_s = now
            self.state.release_pass_streak = 0
            self.state.hold_failure_streak = 0
            self.state.force_global_refresh = True
            self.state.transitions.append(
                {
                    "time_s": now,
                    "from": "TRACK",
                    "to": "ACQUIRE",
                    "failed_checks": failed,
                }
            )
        return self.state.locked


class AuditedCausalStreamingEstimator(v21.CausalStreamingEstimator):
    """Unchanged V21 estimator wired to :class:`AuditedCausalLockGate`."""

    def __init__(
        self,
        *,
        estimator_config: v19.BatchEstimatorConfig,
        lock_config: AuditedLockConfig,
        support_radius_min_m: float,
        support_radius_max_m: float,
    ) -> None:
        super().__init__(
            estimator_config=estimator_config,
            lock_config=lock_config.base,
            support_radius_min_m=support_radius_min_m,
            support_radius_max_m=support_radius_max_m,
        )
        self.gate = AuditedCausalLockGate(lock_config)


def exact_trace_score(trace: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Score releases and TRACK actions at their exact causal indices."""

    gate_after = np.asarray(trace["gate_locked_after_update"], dtype=bool)
    phase = np.asarray(trace["phase_track"], dtype=bool)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    times = np.asarray(trace["time_s"], dtype=np.float64)
    action_times = np.asarray(trace["action_start_time_s"], dtype=np.float64)
    if not (
        gate_after.shape
        == phase.shape
        == localization.shape
        == times.shape
        == action_times.shape
    ):
        raise ValueError("exact-score trace fields have inconsistent shapes")
    previous = np.concatenate([np.asarray([False]), gate_after[:-1]])
    transitions = np.flatnonzero(gate_after & ~previous)
    phase_indices = np.flatnonzero(phase)
    if phase_indices.size and np.any(phase_indices == 0):
        raise RuntimeError("TRACK action has no preceding causal state")
    transition_errors = localization[transitions]
    transition_times = times[transitions]
    start_errors = localization[phase_indices - 1] if phase_indices.size else np.asarray([])
    end_errors = localization[phase_indices]
    def optional_first(values: np.ndarray) -> Optional[float]:
        if values.size == 0 or not math.isfinite(float(values[0])):
            return None
        return float(values[0])

    def optional_max(values: np.ndarray) -> Optional[float]:
        finite_values = values[np.isfinite(values)]
        return None if finite_values.size == 0 else float(np.max(finite_values))

    limit = float(v21.TERMINAL_LOCALIZATION_GATE_M)
    transition_unsafe = (~np.isfinite(transition_errors)) | (
        transition_errors >= limit
    )
    start_unsafe = (~np.isfinite(start_errors)) | (start_errors >= limit)
    end_unsafe = (~np.isfinite(end_errors)) | (end_errors >= limit)
    return {
        "transition_count": int(transitions.size),
        "first_transition_time_s": optional_first(transition_times),
        "first_transition_error_m": optional_first(transition_errors),
        "maximum_transition_error_m": optional_max(transition_errors),
        "false_transition_count": int(np.sum(transition_unsafe)),
        "locked_action_count": int(phase_indices.size),
        "first_locked_action_time_s": (
            None if phase_indices.size == 0 else float(action_times[phase_indices[0]])
        ),
        "first_locked_action_start_error_m": optional_first(start_errors),
        "first_locked_action_end_error_m": optional_first(end_errors),
        "maximum_locked_action_start_error_m": optional_max(start_errors),
        "maximum_locked_action_end_error_m": optional_max(end_errors),
        "false_locked_action_start_count": int(np.sum(start_unsafe)),
        "false_locked_action_end_count": int(np.sum(end_unsafe)),
    }


def _audit_trace_values(gate: AuditedCausalLockGate) -> Dict[str, float]:
    metrics = gate.last_metrics
    release = gate.last_release_checks
    hold = gate.last_hold_checks
    return {
        "gate_release_predicate": float(gate.state.last_release_predicate),
        "gate_release_pass_streak": float(gate.state.release_pass_streak),
        "gate_hold_predicate": float(gate.state.last_hold_predicate),
        "gate_hold_failure_streak": float(gate.state.hold_failure_streak),
        "audit_release_checks_pass": float(
            bool(release)
            and all(release.get(name, False) for name in gate.AUDIT_RELEASE_CHECKS)
        ),
        "audit_hold_checks_pass": float(
            bool(hold)
            and all(hold.get(name, False) for name in gate.AUDIT_HOLD_CHECKS)
        ),
        **{
            name: float(metrics.get(name, float("nan")))
            for name in (
                "local_to_primary_m",
                "local_to_confirmation_m",
                "primary_to_confirmation_m",
                "primary_radius95_m",
                "confirmation_radius95_m",
                "primary_rmse_mps",
                "confirmation_rmse_mps",
                "global_evidence_time_s",
            )
        },
    }


def run_audited_active_acquisition_arm(
    *,
    cfg: UUV3DConfig,
    tape: ExogenousNoiseTape,
    episode_seed: int,
    episode_index: int,
    estimator_config: v19.BatchEstimatorConfig,
    lock_config: AuditedLockConfig,
    planner_config: v22.ActivePlannerConfig,
) -> v22.ArmOutcome:
    """Run V22 active acquisition with the V24 audited gate."""

    env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    env.attach_exogenous_noise_tape(tape)
    env.reset(seed=int(episode_seed))
    initial_truth = np.asarray(env.pF, dtype=np.float64).copy()
    initial_leader_centroid = 0.5 * (
        np.asarray(env.pL1, dtype=np.float64) + np.asarray(env.pL2, dtype=np.float64)
    )
    recorder = v20.OnlineHistoryRecorder(env)
    estimator = AuditedCausalStreamingEstimator(
        estimator_config=estimator_config,
        lock_config=lock_config,
        support_radius_min_m=float(cfg.start_rho_min),
        support_radius_max_m=float(cfg.start_rho_max),
    )
    planner = v22.BeliefFIMPlanner(planner_config)
    fields = (
        "step", "action_start_time_s", "time_s", "phase_track",
        "gate_locked_after_update", "action_speed", "action_yaw", "action_pitch",
        "truth_x", "truth_y", "truth_z", "estimate_x", "estimate_y", "estimate_z",
        "formation_error_truth_m", "localization_error_m", "batch_local_radius95_m",
        "planner_utility", "planner_worst_radius_before_m",
        "planner_worst_radius_after_m", "planner_minimum_pair_chi2",
        "planner_hypothesis_count", "planner_runtime_s",
        "gate_release_predicate", "gate_release_pass_streak",
        "gate_hold_predicate", "gate_hold_failure_streak",
        "audit_release_checks_pass", "audit_hold_checks_pass",
        "local_to_primary_m", "local_to_confirmation_m",
        "primary_to_confirmation_m", "primary_radius95_m",
        "confirmation_radius95_m", "primary_rmse_mps",
        "confirmation_rmse_mps", "global_evidence_time_s",
    )
    trace_rows: Dict[str, List[Any]] = {name: [] for name in fields}
    actions: List[np.ndarray] = []
    try:
        for _ in range(int(cfg.max_steps)):
            action_start = float(env.t)
            track = bool(estimator.gate.is_locked and estimator.has_solution)
            decision: Optional[v22.PlannerDecision] = None
            if track:
                action = v20.pid_action_for_position(
                    env,
                    np.asarray(estimator.state.endpoint_position_m, dtype=np.float64),
                )
            elif not estimator.has_solution:
                action = v20.position_independent_acquisition_action(env, action_start)
            else:
                decision = planner.action(
                    env,
                    history=recorder.history(),
                    current_dead_reckoning_m=recorder.accumulated_dead_reckoning_m,
                    estimator=estimator,
                    time_s=action_start,
                )
                action = decision.action
            actions.append(np.asarray(action, dtype=np.float64).copy())
            _, _, terminated, truncated, _ = env.step(action)
            estimator.update(
                recorder.history(),
                decision_time_s=float(env.step_count) * float(cfg.action_dt),
                current_dead_reckoning_m=recorder.accumulated_dead_reckoning_m,
            )
            truth = np.asarray(env.pF, dtype=np.float64).copy()
            endpoint = (
                np.asarray(estimator.state.endpoint_position_m, dtype=np.float64).copy()
                if estimator.has_solution
                else np.full(3, np.nan, dtype=np.float64)
            )
            _, desired = env._formation_desired()
            formation = float(np.linalg.norm(truth - np.asarray(desired)))
            localization = float(np.linalg.norm(endpoint - truth))
            mode = estimator.state.latest_mode
            radius95 = float("nan") if mode is None else float(mode.local_radius95_m)
            base_values = {
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
                "batch_local_radius95_m": radius95,
                "planner_utility": float("nan") if decision is None else decision.utility,
                "planner_worst_radius_before_m": float("nan") if decision is None else decision.worst_radius_before_m,
                "planner_worst_radius_after_m": float("nan") if decision is None else decision.worst_radius_after_m,
                "planner_minimum_pair_chi2": float("nan") if decision is None else decision.minimum_pair_chi2,
                "planner_hypothesis_count": 0 if decision is None else decision.hypothesis_count,
                "planner_runtime_s": 0.0 if decision is None else decision.runtime_s,
            }
            base_values.update(_audit_trace_values(estimator.gate))
            for name in fields:
                trace_rows[name].append(base_values[name])
            if terminated or truncated:
                if int(env.step_count) != int(cfg.max_steps):
                    raise RuntimeError("V24 arm ended before fixed horizon")
                break
        cursor = env._v11_noise_cursor
        if cursor is None:
            raise RuntimeError("V24 environment lost its noise tape")
        noise_cursor = dict(cursor.state_dict())
    finally:
        recorder.restore()
        env.close()

    trace = {name: np.asarray(values) for name, values in trace_rows.items()}
    expected_shape = (int(cfg.max_steps),)
    if trace["time_s"].shape != expected_shape:
        raise RuntimeError("V24 trace does not contain the fixed action horizon")
    times = np.asarray(trace["time_s"], dtype=np.float64)
    formation = np.asarray(trace["formation_error_truth_m"], dtype=np.float64)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    joint = (formation < v21.TERMINAL_FORMATION_GATE_M) & (
        localization < v21.TERMINAL_LOCALIZATION_GATE_M
    )
    truth_ready = np.isfinite(localization) & (
        localization < v21.TERMINAL_LOCALIZATION_GATE_M
    )
    truth_ready_time = v21._first_window_time(
        times, truth_ready, v21.TRUTH_READY_DWELL_ACTIONS
    )
    exact = exact_trace_score(trace)
    gate_after = np.asarray(trace["gate_locked_after_update"], dtype=bool)
    previous_gate = np.concatenate([np.asarray([False]), gate_after[:-1]])
    transition_mask = gate_after & ~previous_gate
    audit_release_violation_count = int(
        np.sum(
            transition_mask
            & ~np.asarray(trace["audit_release_checks_pass"], dtype=bool)
        )
    )
    first_action_time = exact["first_locked_action_time_s"]
    tail = joint[-v21.TAIL_WINDOW_ACTIONS :]
    dwell = joint[-v21.DWELL_ACTIONS :]
    action_array = np.asarray(actions, dtype=np.float64)
    state = estimator.state
    gate_state = estimator.gate.state
    false_episode = bool(
        exact["false_transition_count"]
        or exact["false_locked_action_start_count"]
        or exact["false_locked_action_end_count"]
    )
    summary: Dict[str, Any] = {
        "version": VERSION,
        "arm": "v24_audited_gate",
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "noise_tape_sha256": tape.content_sha256(),
        "action_count": int(times.size),
        "noise_cursor": noise_cursor,
        "initial_truth_m": initial_truth.tolist(),
        "initial_leader_centroid_m": initial_leader_centroid.tolist(),
        "terminal_formation_error_m": float(formation[-1]),
        "terminal_localization_error_m": float(localization[-1]),
        "terminal_joint_success": bool(joint[-1]),
        "dwell15_joint_success": bool(np.all(dwell)),
        "tail50_joint_occupancy": float(np.mean(tail)),
        "tail80_joint_success": bool(float(np.mean(tail)) >= 0.8),
        "time_to_sustained_joint_lock_s": v21._first_sustained_time(times, joint),
        "mean_squared_action": float(
            np.mean(np.sum(action_array * action_array, axis=1))
        ),
        "mean_formation_error_after_30_m": float(
            np.mean(formation[times >= 30.0])
        ),
        "gate": {
            "first_track_action_time_s": first_action_time,
            "first_track_localization_error_m": exact[
                "first_locked_action_start_error_m"
            ],
            "truth_ready_time_s": truth_ready_time,
            "lock_delay_from_truth_ready_s": (
                None
                if first_action_time is None or truth_ready_time is None
                else float(first_action_time - truth_ready_time)
            ),
            "ever_locked": bool(exact["transition_count"] > 0),
            "false_lock_episode": false_episode,
            "false_locked_action_count": int(
                exact["false_locked_action_start_count"]
            ),
            "locked_action_count": int(exact["locked_action_count"]),
            "lock_count": int(gate_state.lock_count),
            "unlock_count": int(gate_state.unlock_count),
            "audit_release_violation_count": audit_release_violation_count,
            "transitions": list(gate_state.transitions),
            "exact": exact,
        },
        "batch": {
            "measurement_count": int(recorder.measurement_count),
            "global_solve_count": int(state.global_solve_count),
            "local_solve_count": int(state.local_solve_count),
            "total_runtime_s": float(state.total_runtime_s),
            "maximum_update_runtime_s": float(state.maximum_update_runtime_s),
            "global_records": list(estimator.global_records),
        },
        "planner": {
            "decision_count": int(planner.decision_count),
            "total_runtime_s": float(planner.total_runtime_s),
            "maximum_runtime_s": float(planner.maximum_runtime_s),
        },
    }
    return v22.ArmOutcome(summary=summary, trace=trace)


def augment_reference_outcome(outcome: v22.ArmOutcome) -> v22.ArmOutcome:
    """Attach the same exact causal score to the frozen V23 reference arm."""

    summary = dict(outcome.summary)
    gate = dict(summary["gate"])
    exact = exact_trace_score(outcome.trace)
    gate["exact"] = exact
    gate["first_track_action_time_s"] = exact["first_locked_action_time_s"]
    gate["first_track_localization_error_m"] = exact[
        "first_locked_action_start_error_m"
    ]
    truth_ready_time = gate.get("truth_ready_time_s")
    gate["lock_delay_from_truth_ready_s"] = (
        None
        if exact["first_locked_action_time_s"] is None or truth_ready_time is None
        else float(exact["first_locked_action_time_s"] - float(truth_ready_time))
    )
    gate["ever_locked"] = bool(exact["transition_count"] > 0)
    gate["false_lock_episode"] = bool(
        exact["false_transition_count"]
        or exact["false_locked_action_start_count"]
        or exact["false_locked_action_end_count"]
    )
    gate["false_locked_action_count"] = int(
        exact["false_locked_action_start_count"]
    )
    gate["locked_action_count"] = int(exact["locked_action_count"])
    summary["gate"] = gate
    summary["version"] = VERSION
    summary["arm"] = "v23_early_reference"
    return v22.ArmOutcome(summary=summary, trace=outcome.trace)


__all__ = [
    "AuditedCausalLockGate",
    "AuditedCausalStreamingEstimator",
    "AuditedLockConfig",
    "VERSION",
    "augment_reference_outcome",
    "exact_trace_score",
    "run_audited_active_acquisition_arm",
]
