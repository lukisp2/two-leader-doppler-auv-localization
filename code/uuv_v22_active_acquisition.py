#!/usr/bin/env python3
"""Belief-conditioned deterministic acquisition for the V21 causal gate."""

from __future__ import annotations

import itertools
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


VERSION = "v22_belief_fim_acquisition_1.0"


@dataclass(frozen=True)
class ActivePlannerConfig:
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
        numeric = (
            self.horizon_s,
            self.prediction_dt_s,
            self.hypothesis_cluster_radius_m,
            self.covariance_axis_scale,
            self.pair_separation_m,
            self.pair_weight,
            self.action_energy_weight,
            self.action_change_weight,
        )
        if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in numeric):
            raise ValueError("active-planner constants must be finite and positive")
        if int(self.maximum_hypotheses) < 1:
            raise ValueError("maximum_hypotheses must be positive")


@dataclass(frozen=True)
class PlannerDecision:
    action: np.ndarray
    utility: float
    worst_radius_before_m: float
    worst_radius_after_m: float
    minimum_pair_chi2: float
    hypothesis_count: int
    candidate_count: int
    runtime_s: float


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _action_velocity(
    action: np.ndarray,
    env: UUVTwoLeader3DPFEnv,
) -> np.ndarray:
    cfg = env.cfg
    command = np.asarray(action, dtype=np.float64)
    speed = float(
        np.clip(
            float(env._v11_speed_meas)
            + command[0] * float(cfg.rl_speed_delta_per_step),
            float(cfg.f_min_speed),
            float(cfg.f_max_speed),
        )
    )
    yaw = math.radians(
        float(env._v11_yaw_meas)
        + command[1] * float(cfg.rl_yaw_per_step_deg)
    )
    pitch_deg = float(
        np.clip(
            float(env._v11_pitch_meas)
            + command[2] * float(cfg.rl_pitch_per_step_deg),
            float(cfg.pitch_min_deg),
            float(cfg.pitch_max_deg),
        )
    )
    pitch = math.radians(pitch_deg)
    horizontal = speed * math.cos(pitch)
    return np.asarray(
        [
            horizontal * math.cos(yaw),
            horizontal * math.sin(yaw),
            speed * math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _candidate_actions(s_turn: np.ndarray) -> np.ndarray:
    values: List[np.ndarray] = [np.zeros(3, dtype=np.float64)]
    for speed, yaw, pitch in itertools.product(
        (-1.0, 0.0, 1.0),
        (-1.0, -0.5, 0.0, 0.5, 1.0),
        (-1.0, 0.0, 1.0),
    ):
        candidate = np.asarray([speed, yaw, pitch], dtype=np.float64)
        if not np.allclose(candidate, 0.0):
            values.append(candidate)
    fallback = np.asarray(s_turn, dtype=np.float64)
    if all(not np.allclose(fallback, candidate, atol=1e-12) for candidate in values):
        values.append(fallback)
    return np.asarray(values, dtype=np.float64)


def _cluster_hypotheses(
    values: Sequence[np.ndarray],
    *,
    radius_m: float,
    maximum: int,
) -> np.ndarray:
    kept: List[np.ndarray] = []
    for value in values:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        if all(float(np.linalg.norm(point - previous)) >= float(radius_m) for previous in kept):
            kept.append(point.copy())
        if len(kept) >= int(maximum):
            break
    if not kept:
        raise RuntimeError("belief planner received no finite hypothesis")
    return np.asarray(kept, dtype=np.float64)


class BeliefFIMPlanner:
    """Worst-hypothesis E-optimal Doppler planner with mode discrimination."""

    def __init__(self, config: ActivePlannerConfig) -> None:
        self.config = config
        self.previous_action = np.zeros(3, dtype=np.float64)
        self.total_runtime_s = 0.0
        self.maximum_runtime_s = 0.0
        self.decision_count = 0

    def _hypotheses(
        self,
        estimator: v21.CausalStreamingEstimator,
        history: v19.OnlineDopplerHistory,
    ) -> np.ndarray:
        state = estimator.state
        if state.initial_position_m is None or state.latest_mode is None:
            raise RuntimeError("belief requested before estimator initialization")
        values: List[np.ndarray] = [
            np.asarray(state.initial_position_m, dtype=np.float64).copy()
        ]
        evidence = state.latest_global_evidence
        if evidence is not None:
            for estimate in (evidence.primary, evidence.confirmation):
                values.extend(
                    np.asarray(mode.initial_position_m, dtype=np.float64).copy()
                    for mode in estimate.modes
                )
        mode = state.latest_mode
        covariance = np.asarray(mode.local_covariance_m2, dtype=np.float64)
        if mode.local_covariance_valid and covariance.shape == (3, 3) and np.all(np.isfinite(covariance)):
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(
                    0.5 * (covariance + covariance.T)
                )
                center = v19.initial_leader_centroid_from_history(history)
                for index in range(3):
                    scale = float(self.config.covariance_axis_scale) * math.sqrt(
                        max(float(eigenvalues[index]), 0.0)
                    )
                    delta = scale * eigenvectors[:, index]
                    for sign in (-1.0, 1.0):
                        values.append(
                            v19.project_to_shell(
                                np.asarray(state.initial_position_m) + sign * delta,
                                center,
                                estimator.support_radius_min_m,
                                estimator.support_radius_max_m,
                            )
                        )
            except np.linalg.LinAlgError:
                pass
        return _cluster_hypotheses(
            values,
            radius_m=self.config.hypothesis_cluster_radius_m,
            maximum=self.config.maximum_hypotheses,
        )

    def _evaluate_velocity(
        self,
        *,
        velocity_mps: np.ndarray,
        hypotheses_p0_m: np.ndarray,
        current_dead_reckoning_m: np.ndarray,
        base_information: np.ndarray,
        leader_position_m: np.ndarray,
        leader_velocity_mps: np.ndarray,
        sigma_mps: float,
    ) -> Tuple[float, float, float]:
        dt = float(self.config.prediction_dt_s)
        count = int(round(float(self.config.horizon_s) / dt))
        tau = dt * np.arange(1, count + 1, dtype=np.float64)
        current_positions = hypotheses_p0_m + current_dead_reckoning_m[None, :]
        follower = (
            current_positions[:, None, None, :]
            + tau[None, :, None, None] * velocity_mps[None, None, None, :]
        )
        leaders = (
            leader_position_m[None, :, :]
            + tau[:, None, None] * leader_velocity_mps[None, :, :]
        )
        relative = leaders[None, :, :, :] - follower
        rho = np.maximum(np.linalg.norm(relative, axis=3), 1e-9)
        line_of_sight = relative / rho[:, :, :, None]
        relative_velocity = leader_velocity_mps - velocity_mps[None, :]
        projection = np.sum(
            line_of_sight * relative_velocity[None, None, :, :], axis=3
        )
        jacobian = -(
            relative_velocity[None, None, :, :]
            - projection[:, :, :, None] * line_of_sight
        ) / rho[:, :, :, None]
        future_information = np.einsum(
            "mtli,mtlj->mij", jacobian, jacobian, optimize=True
        )
        total_information = base_information + future_information
        eigenvalues = np.linalg.eigvalsh(total_information)
        minimum_eigenvalues = np.maximum(eigenvalues[:, 0], 1e-18)
        radii = (
            v19.CHI2_3_95_SQRT
            * float(sigma_mps)
            / np.sqrt(minimum_eigenvalues)
        )
        worst_after = float(np.max(radii))
        base_eigenvalues = np.linalg.eigvalsh(base_information)
        base_minimum = np.maximum(base_eigenvalues[:, 0], 1e-18)
        worst_before = float(
            np.max(
                v19.CHI2_3_95_SQRT
                * float(sigma_mps)
                / np.sqrt(base_minimum)
            )
        )
        prediction = -projection.reshape(hypotheses_p0_m.shape[0], -1)
        pair_values: List[float] = []
        for left in range(hypotheses_p0_m.shape[0]):
            for right in range(left + 1, hypotheses_p0_m.shape[0]):
                if float(
                    np.linalg.norm(hypotheses_p0_m[left] - hypotheses_p0_m[right])
                ) < float(self.config.pair_separation_m):
                    continue
                difference = prediction[left] - prediction[right]
                pair_values.append(
                    float(difference @ difference)
                    / max(float(sigma_mps) ** 2, 1e-18)
                )
        minimum_pair = 0.0 if not pair_values else float(min(pair_values))
        return worst_before, worst_after, minimum_pair

    def action(
        self,
        env: UUVTwoLeader3DPFEnv,
        *,
        history: v19.OnlineDopplerHistory,
        current_dead_reckoning_m: Sequence[float],
        estimator: v21.CausalStreamingEstimator,
        time_s: float,
    ) -> PlannerDecision:
        started = time.perf_counter()
        fallback = v20.position_independent_acquisition_action(env, float(time_s))
        if not estimator.has_solution or estimator.state.latest_mode is None:
            elapsed = float(time.perf_counter() - started)
            self.total_runtime_s += elapsed
            self.maximum_runtime_s = max(self.maximum_runtime_s, elapsed)
            self.decision_count += 1
            self.previous_action = np.asarray(fallback, dtype=np.float64)
            return PlannerDecision(
                action=np.asarray(fallback, dtype=np.float32),
                utility=0.0,
                worst_radius_before_m=float("inf"),
                worst_radius_after_m=float("inf"),
                minimum_pair_chi2=0.0,
                hypothesis_count=0,
                candidate_count=1,
                runtime_s=elapsed,
            )

        hypotheses = self._hypotheses(estimator, history)
        base_information: List[np.ndarray] = []
        for position in hypotheses:
            _, jacobian = v19.residual_and_jacobian(
                position,
                history,
                estimator.estimator_config,
            )
            base_information.append(jacobian.T @ jacobian)
        base = np.asarray(base_information, dtype=np.float64)
        candidates = _candidate_actions(fallback)
        current_dr = np.asarray(current_dead_reckoning_m, dtype=np.float64)
        leaders = np.asarray([env.pL1, env.pL2], dtype=np.float64)
        leader_velocities = np.asarray([env.vL1, env.vL2], dtype=np.float64)
        sigma = float(estimator.estimator_config.measurement_sigma_mps)
        best_index = 0
        best_utility = -float("inf")
        best_metrics = (float("inf"), float("inf"), 0.0)
        for index, candidate in enumerate(candidates):
            velocity = _action_velocity(candidate, env)
            before, after, pair = self._evaluate_velocity(
                velocity_mps=velocity,
                hypotheses_p0_m=hypotheses,
                current_dead_reckoning_m=current_dr,
                base_information=base,
                leader_position_m=leaders,
                leader_velocity_mps=leader_velocities,
                sigma_mps=sigma,
            )
            radius_reduction = before - after
            utility = (
                radius_reduction
                + float(self.config.pair_weight) * math.log1p(max(pair, 0.0))
                - float(self.config.action_energy_weight)
                * float(candidate @ candidate)
                - float(self.config.action_change_weight)
                * float(np.sum((candidate - self.previous_action) ** 2))
            )
            if math.isfinite(utility) and utility > best_utility:
                best_index = index
                best_utility = float(utility)
                best_metrics = (before, after, pair)
        if not math.isfinite(best_utility):
            raise RuntimeError("active planner produced no finite candidate utility")
        selected = np.asarray(candidates[best_index], dtype=np.float32)
        self.previous_action = np.asarray(selected, dtype=np.float64)
        elapsed = float(time.perf_counter() - started)
        self.total_runtime_s += elapsed
        self.maximum_runtime_s = max(self.maximum_runtime_s, elapsed)
        self.decision_count += 1
        return PlannerDecision(
            action=selected,
            utility=best_utility,
            worst_radius_before_m=float(best_metrics[0]),
            worst_radius_after_m=float(best_metrics[1]),
            minimum_pair_chi2=float(best_metrics[2]),
            hypothesis_count=int(hypotheses.shape[0]),
            candidate_count=int(candidates.shape[0]),
            runtime_s=elapsed,
        )


@dataclass(frozen=True)
class ArmOutcome:
    summary: Mapping[str, Any]
    trace: Mapping[str, np.ndarray]


def run_active_acquisition_arm(
    *,
    cfg: UUV3DConfig,
    tape: ExogenousNoiseTape,
    episode_seed: int,
    episode_index: int,
    estimator_config: v19.BatchEstimatorConfig,
    lock_config: v21.CausalLockConfig,
    planner_config: ActivePlannerConfig,
) -> ArmOutcome:
    env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    env.attach_exogenous_noise_tape(tape)
    env.reset(seed=int(episode_seed))
    initial_truth = np.asarray(env.pF, dtype=np.float64).copy()
    initial_leader_centroid = 0.5 * (
        np.asarray(env.pL1, dtype=np.float64) + np.asarray(env.pL2, dtype=np.float64)
    )
    recorder = v20.OnlineHistoryRecorder(env)
    estimator = v21.CausalStreamingEstimator(
        estimator_config=estimator_config,
        lock_config=lock_config,
        support_radius_min_m=float(cfg.start_rho_min),
        support_radius_max_m=float(cfg.start_rho_max),
    )
    planner = BeliefFIMPlanner(planner_config)
    fields = (
        "step", "action_start_time_s", "time_s", "phase_track",
        "gate_locked_after_update", "action_speed", "action_yaw", "action_pitch",
        "truth_x", "truth_y", "truth_z", "estimate_x", "estimate_y", "estimate_z",
        "formation_error_truth_m", "localization_error_m", "batch_local_radius95_m",
        "planner_utility", "planner_worst_radius_before_m",
        "planner_worst_radius_after_m", "planner_minimum_pair_chi2",
        "planner_hypothesis_count", "planner_runtime_s",
    )
    trace_rows: Dict[str, List[Any]] = {name: [] for name in fields}
    actions: List[np.ndarray] = []
    try:
        for _ in range(int(cfg.max_steps)):
            action_start = float(env.t)
            track = bool(estimator.gate.is_locked and estimator.has_solution)
            decision: Optional[PlannerDecision] = None
            if track:
                action = v20.pid_action_for_position(
                    env, np.asarray(estimator.state.endpoint_position_m, dtype=np.float64)
                )
            elif not estimator.has_solution:
                action = v20.position_independent_acquisition_action(
                    env, action_start
                )
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
                if estimator.has_solution else np.full(3, np.nan)
            )
            _, desired = env._formation_desired()
            formation = float(np.linalg.norm(truth - np.asarray(desired)))
            localization = float(np.linalg.norm(endpoint - truth))
            mode = estimator.state.latest_mode
            radius95 = float("nan") if mode is None else float(mode.local_radius95_m)
            trace_rows["step"].append(int(env.step_count))
            trace_rows["action_start_time_s"].append(action_start)
            trace_rows["time_s"].append(float(env.t))
            trace_rows["phase_track"].append(float(track))
            trace_rows["gate_locked_after_update"].append(float(estimator.gate.is_locked))
            trace_rows["action_speed"].append(float(action[0]))
            trace_rows["action_yaw"].append(float(action[1]))
            trace_rows["action_pitch"].append(float(action[2]))
            trace_rows["truth_x"].append(float(truth[0]))
            trace_rows["truth_y"].append(float(truth[1]))
            trace_rows["truth_z"].append(float(truth[2]))
            trace_rows["estimate_x"].append(float(endpoint[0]))
            trace_rows["estimate_y"].append(float(endpoint[1]))
            trace_rows["estimate_z"].append(float(endpoint[2]))
            trace_rows["formation_error_truth_m"].append(formation)
            trace_rows["localization_error_m"].append(localization)
            trace_rows["batch_local_radius95_m"].append(radius95)
            trace_rows["planner_utility"].append(float("nan") if decision is None else decision.utility)
            trace_rows["planner_worst_radius_before_m"].append(float("nan") if decision is None else decision.worst_radius_before_m)
            trace_rows["planner_worst_radius_after_m"].append(float("nan") if decision is None else decision.worst_radius_after_m)
            trace_rows["planner_minimum_pair_chi2"].append(float("nan") if decision is None else decision.minimum_pair_chi2)
            trace_rows["planner_hypothesis_count"].append(0 if decision is None else decision.hypothesis_count)
            trace_rows["planner_runtime_s"].append(0.0 if decision is None else decision.runtime_s)
            if terminated or truncated:
                if int(env.step_count) != int(cfg.max_steps):
                    raise RuntimeError("V22 arm ended before fixed horizon")
                break
        cursor = env._v11_noise_cursor
        if cursor is None:
            raise RuntimeError("V22 environment lost its noise tape")
        noise_cursor = dict(cursor.state_dict())
    finally:
        recorder.restore()
        env.close()

    trace = {name: np.asarray(values) for name, values in trace_rows.items()}
    times = np.asarray(trace["time_s"], dtype=np.float64)
    action_times = np.asarray(trace["action_start_time_s"], dtype=np.float64)
    formation = np.asarray(trace["formation_error_truth_m"], dtype=np.float64)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    phase = np.asarray(trace["phase_track"], dtype=bool)
    finite = np.isfinite(localization)
    joint = (formation < v21.TERMINAL_FORMATION_GATE_M) & (
        localization < v21.TERMINAL_LOCALIZATION_GATE_M
    )
    truth_ready = finite & (localization < v21.TERMINAL_LOCALIZATION_GATE_M)
    truth_ready_time = v21._first_window_time(
        times, truth_ready, v21.TRUTH_READY_DWELL_ACTIONS
    )
    indices = np.flatnonzero(phase)
    first_index = None if indices.size == 0 else int(indices[0])
    first_track_time = None if first_index is None else float(action_times[first_index])
    false_mask = phase & ((~finite) | (localization >= v21.TERMINAL_LOCALIZATION_GATE_M))
    tail = joint[-v21.TAIL_WINDOW_ACTIONS:]
    dwell = joint[-v21.DWELL_ACTIONS:]
    action_array = np.asarray(actions, dtype=np.float64)
    state = estimator.state
    gate_state = estimator.gate.state
    summary: Dict[str, Any] = {
        "version": VERSION,
        "arm": "belief_fim_causal",
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
        "mean_squared_action": float(np.mean(np.sum(action_array * action_array, axis=1))),
        "mean_formation_error_after_30_m": float(np.mean(formation[times >= 30.0])),
        "gate": {
            "first_track_action_time_s": first_track_time,
            "first_track_localization_error_m": None if first_index is None else float(localization[first_index]),
            "truth_ready_time_s": truth_ready_time,
            "lock_delay_from_truth_ready_s": None if first_track_time is None or truth_ready_time is None else float(first_track_time - truth_ready_time),
            "ever_locked": bool(first_index is not None),
            "false_lock_episode": bool(np.any(false_mask)),
            "false_locked_action_count": int(np.sum(false_mask)),
            "locked_action_count": int(np.sum(phase)),
            "lock_count": int(gate_state.lock_count),
            "unlock_count": int(gate_state.unlock_count),
            "transitions": list(gate_state.transitions),
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
    return ArmOutcome(summary=summary, trace=trace)


__all__ = [
    "ActivePlannerConfig",
    "BeliefFIMPlanner",
    "PlannerDecision",
    "VERSION",
    "run_active_acquisition_arm",
]
