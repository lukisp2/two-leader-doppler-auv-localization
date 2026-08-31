#!/usr/bin/env python3
"""Causal positioning ablation for the two-leader UUV problem.

V20 deliberately does not train or load reinforcement learning.  Every arm
executes the same position-independent acquisition manoeuvre for 120 seconds.
After that point the same restored PID controller receives one of three
position sources: the legacy particle filter, the causal V19 batch estimate,
or simulator truth (an explicitly diagnostic oracle ceiling).

The particle filter continues to run in every environment because it is part
of the frozen V18 simulator implementation.  In the ``batch_pid`` arm its
state is never read by the external controller; raw Doppler and dead-reckoning
measurements are merely captured at the existing PF call boundary.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from baseline_controllers_v11 import pid_tracking_action_from_info
from uuv_v11_rng import ExogenousNoiseTape
from uuv_v18_resampling_guard import UUV3DConfig, UUVTwoLeader3DPFEnv
import uuv_v19_observability as v19


VERSION = "v20_positioning_ablation_1.0"
ARM_NAMES: Tuple[str, ...] = ("pf_pid", "batch_pid", "oracle_pid")
TRACK_START_S = 120.0
GLOBAL_REFRESH_S: Tuple[float, ...] = (120.0, 240.0, 360.0, 440.0)
TERMINAL_FORMATION_GATE_M = 8.0
TERMINAL_LOCALIZATION_GATE_M = 7.0
TAIL_WINDOW_ACTIONS = 50
DWELL_ACTIONS = 15


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _velocity_to_direct_action(
    desired_velocity_mps: Sequence[float],
    *,
    measured_speed_mps: float,
    measured_yaw_deg: float,
    measured_pitch_deg: float,
    cfg: UUV3DConfig,
) -> np.ndarray:
    """Convert a desired velocity into the simulator's normalized increments."""

    velocity = np.asarray(desired_velocity_mps, dtype=np.float64)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("desired velocity must contain three finite values")
    speed_xy = float(np.linalg.norm(velocity[:2]))
    speed_des = float(
        np.clip(
            np.linalg.norm(velocity),
            float(cfg.f_min_speed),
            float(cfg.f_max_speed),
        )
    )
    if speed_xy > 1e-12:
        yaw_des = math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
    else:
        yaw_des = float(measured_yaw_deg)
    pitch_des = math.degrees(math.atan2(float(velocity[2]), max(speed_xy, 1e-12)))
    pitch_des = float(
        np.clip(pitch_des, float(cfg.pitch_min_deg), float(cfg.pitch_max_deg))
    )
    return np.clip(
        np.asarray(
            [
                (speed_des - float(measured_speed_mps))
                / max(float(cfg.rl_speed_delta_per_step), 1e-9),
                _wrap180(yaw_des - float(measured_yaw_deg))
                / max(float(cfg.rl_yaw_per_step_deg), 1e-9),
                (pitch_des - float(measured_pitch_deg))
                / max(float(cfg.rl_pitch_per_step_deg), 1e-9),
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    ).astype(np.float32)


def position_independent_acquisition_action(
    env: UUVTwoLeader3DPFEnv, time_s: float
) -> np.ndarray:
    """Deterministic S-turn using only broadcasts and onboard kinematics.

    No follower position, PF state, covariance, reward, planner or simulator
    truth is read here.  The common manoeuvre follows the mean leader velocity
    while adding bounded horizontal and vertical excitation.
    """

    v_center = 0.5 * (
        np.asarray(env.vL1, dtype=np.float64)
        + np.asarray(env.vL2, dtype=np.float64)
    )
    horizontal = np.asarray(v_center[:2], dtype=np.float64)
    norm_xy = float(np.linalg.norm(horizontal))
    if norm_xy <= 1e-9:
        forward = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward = np.asarray(
            [horizontal[0] / norm_xy, horizontal[1] / norm_xy, 0.0],
            dtype=np.float64,
        )
    side = np.asarray([-forward[1], forward[0], 0.0], dtype=np.float64)
    vertical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    t = float(time_s)
    desired = (
        v_center
        + 1.20 * math.sin(2.0 * math.pi * t / 40.0) * side
        + 0.35 * math.cos(2.0 * math.pi * t / 70.0) * forward
        + 0.75 * math.sin(2.0 * math.pi * t / 55.0 + 0.4) * vertical
    )
    return _velocity_to_direct_action(
        desired,
        measured_speed_mps=float(env._v11_speed_meas),
        measured_yaw_deg=float(env._v11_yaw_meas),
        measured_pitch_deg=float(env._v11_pitch_meas),
        cfg=env.cfg,
    )


def pid_action_for_position(
    env: UUVTwoLeader3DPFEnv, estimated_position_m: Sequence[float]
) -> np.ndarray:
    """Restored PID action built from an explicit online-only allowlist."""

    estimate = np.asarray(estimated_position_m, dtype=np.float64)
    if estimate.shape != (3,) or not np.all(np.isfinite(estimate)):
        raise ValueError("estimated follower position must contain three finite values")
    _, desired = env._formation_desired()
    info: Dict[str, float] = {
        "pFhat_x": float(estimate[0]),
        "pFhat_y": float(estimate[1]),
        "pFhat_z": float(estimate[2]),
        "pFdes_x": float(desired[0]),
        "pFdes_y": float(desired[1]),
        "pFdes_z": float(desired[2]),
        "leader1_speed": float(env.leader1_speed),
        "leader2_speed": float(env.leader2_speed),
        "yaw_L1": float(env.yaw_L1),
        "yaw_L2": float(env.yaw_L2),
        "speed_F": float(env._v11_speed_meas),
        "yaw_F": float(env._v11_yaw_meas),
        "pitch_F": float(env._v11_pitch_meas),
    }
    return pid_tracking_action_from_info(
        info,
        env.cfg,
        mode="pid_track",
        step_count=int(env.step_count),
    )


class OnlineHistoryRecorder:
    """Capture the deployable V19 history without changing PF behaviour."""

    def __init__(self, env: UUVTwoLeader3DPFEnv) -> None:
        self.env = env
        self._accumulated_dr = np.zeros(3, dtype=np.float64)
        self._time: List[float] = []
        self._dr: List[np.ndarray] = []
        self._leader_position: List[np.ndarray] = []
        self._leader_velocity: List[np.ndarray] = []
        self._follower_velocity_measured: List[np.ndarray] = []
        self._doppler: List[np.ndarray] = []
        self._gate: List[np.ndarray] = []
        self._original_predict = env.pf.predict
        self._original_update = env.pf.update_doppler

        def capture_predict(vF_meas: np.ndarray, dt: float) -> Any:
            self._accumulated_dr += np.asarray(vF_meas, dtype=np.float64) * float(dt)
            return self._original_predict(vF_meas, dt)

        def capture_update(
            pL_list: List[np.ndarray],
            vL_list: List[np.ndarray],
            vF_meas: np.ndarray,
            s_meas_list: List[Optional[float]],
            **kwargs: Any,
        ) -> Any:
            if any(value is None for value in s_meas_list):
                raise RuntimeError("V20 encountered a missing Doppler measurement")
            gate = np.asarray(
                kwargs.get("gate_factors", [1.0, 1.0]), dtype=np.float64
            )
            if gate.shape != (2,) or not np.all(np.isfinite(gate)):
                gate = np.ones(2, dtype=np.float64)
            self._time.append(float(env.next_s_time))
            self._dr.append(self._accumulated_dr.copy())
            self._leader_position.append(
                np.asarray(pL_list, dtype=np.float64).copy()
            )
            self._leader_velocity.append(
                np.asarray(vL_list, dtype=np.float64).copy()
            )
            self._follower_velocity_measured.append(
                np.asarray(vF_meas, dtype=np.float64).copy()
            )
            self._doppler.append(
                np.asarray(s_meas_list, dtype=np.float64).copy()
            )
            # The V20 estimator uses raw Doppler.  Ones make the saved online
            # history independent of every PF gating decision as well.
            self._gate.append(np.ones(2, dtype=np.float64))
            return self._original_update(
                pL_list,
                vL_list,
                vF_meas,
                s_meas_list,
                **kwargs,
            )

        env.pf.predict = capture_predict  # type: ignore[method-assign]
        env.pf.update_doppler = capture_update  # type: ignore[method-assign]

    @property
    def measurement_count(self) -> int:
        return len(self._time)

    @property
    def accumulated_dead_reckoning_m(self) -> np.ndarray:
        return self._accumulated_dr.copy()

    def history(self) -> v19.OnlineDopplerHistory:
        if not self._time:
            raise RuntimeError("no measurements have been captured")
        return v19.OnlineDopplerHistory(
            t_s=np.asarray(self._time, dtype=np.float64),
            dead_reckoned_displacement_m=np.asarray(self._dr, dtype=np.float64),
            leader_position_m=np.asarray(self._leader_position, dtype=np.float64),
            leader_velocity_mps=np.asarray(self._leader_velocity, dtype=np.float64),
            follower_velocity_measured_mps=np.asarray(
                self._follower_velocity_measured, dtype=np.float64
            ),
            doppler_measured_mps=np.asarray(self._doppler, dtype=np.float64),
            historical_pf_gate_factor=np.asarray(self._gate, dtype=np.float64),
        )

    def restore(self) -> None:
        self.env.pf.predict = self._original_predict  # type: ignore[method-assign]
        self.env.pf.update_doppler = self._original_update  # type: ignore[method-assign]


@dataclass
class StreamingBatchState:
    initial_position_m: Optional[np.ndarray] = None
    endpoint_position_m: Optional[np.ndarray] = None
    covariance_m2: Optional[np.ndarray] = None
    latest_mode: Optional[v19.BatchMode] = None
    total_runtime_s: float = 0.0
    global_runtime_s: float = 0.0
    local_runtime_s: float = 0.0
    global_solve_count: int = 0
    local_solve_count: int = 0
    latest_global_alternative_delta_chi2: Optional[float] = None
    latest_global_time_s: Optional[float] = None


class StreamingBatchEstimator:
    """Full-history V19 solve with causal global refreshes and warm starts."""

    def __init__(
        self,
        *,
        config: v19.BatchEstimatorConfig,
        support_radius_min_m: float,
        support_radius_max_m: float,
        episode_index: int,
    ) -> None:
        self.config = config
        self.support_radius_min_m = float(support_radius_min_m)
        self.support_radius_max_m = float(support_radius_max_m)
        self.episode_index = int(episode_index)
        self.state = StreamingBatchState()
        self.global_records: List[Dict[str, Any]] = []

    def update(
        self,
        history: v19.OnlineDopplerHistory,
        *,
        decision_time_s: float,
        current_dead_reckoning_m: Sequence[float],
    ) -> None:
        """Update at an action boundary using only measurements available then.

        The frozen simulator can have its latest integer-second Doppler sample
        one second behind the nominal 2-second action boundary because of
        floating-point accumulation.  ``current_dead_reckoning_m`` therefore
        propagates the estimated initial position to the exact decision time
        rather than to the latest Doppler timestamp.
        """

        now = float(decision_time_s)
        current_dr = np.asarray(current_dead_reckoning_m, dtype=np.float64)
        if current_dr.shape != (3,) or not np.all(np.isfinite(current_dr)):
            raise ValueError("current dead reckoning must contain three finite values")
        if float(history.t_s[-1]) > now + 1e-9:
            raise RuntimeError("batch history contains a future Doppler measurement")
        if now + 1e-9 < TRACK_START_S:
            return
        is_global = any(abs(now - checkpoint) <= 1e-9 for checkpoint in GLOBAL_REFRESH_S)
        center = v19.initial_leader_centroid_from_history(history)
        started = time.perf_counter()
        if is_global or self.state.initial_position_m is None:
            candidate_seed = 20_001 + int(round(now))
            estimate = v19.estimate_initial_position_multistart(
                history,
                center,
                self.support_radius_min_m,
                self.support_radius_max_m,
                self.config,
                candidate_seed=candidate_seed,
            )
            mode = estimate.best
            elapsed = float(time.perf_counter() - started)
            self.state.global_runtime_s += elapsed
            self.state.global_solve_count += 1
            self.state.latest_global_alternative_delta_chi2 = (
                None
                if estimate.alternative_delta_chi2 is None
                else float(estimate.alternative_delta_chi2)
            )
            self.state.latest_global_time_s = now
            self.global_records.append(
                {
                    "time_s": now,
                    "candidate_seed": candidate_seed,
                    "runtime_s": elapsed,
                    "residual_rmse_mps": float(mode.residual_rmse_mps),
                    "converged": bool(mode.converged),
                    "hessian_rank": int(mode.hessian_rank),
                    "hessian_condition_number": float(mode.hessian_condition_number),
                    "local_covariance_valid": bool(mode.local_covariance_valid),
                    "local_radius95_m": float(mode.local_radius95_m),
                    "alternative_delta_chi2": self.state.latest_global_alternative_delta_chi2,
                    "initial_position_m": np.asarray(
                        mode.initial_position_m, dtype=np.float64
                    ).tolist(),
                }
            )
        else:
            mode = v19.refine_damped_gauss_newton(
                self.state.initial_position_m,
                history,
                center,
                self.support_radius_min_m,
                self.support_radius_max_m,
                self.config,
            )
            elapsed = float(time.perf_counter() - started)
            self.state.local_runtime_s += elapsed
            self.state.local_solve_count += 1
        self.state.total_runtime_s += elapsed
        self.state.latest_mode = mode
        self.state.initial_position_m = np.asarray(
            mode.initial_position_m, dtype=np.float64
        ).copy()
        self.state.endpoint_position_m = (
            self.state.initial_position_m
            + current_dr
        )
        self.state.covariance_m2 = np.asarray(
            mode.local_covariance_m2, dtype=np.float64
        ).copy()

    @property
    def has_solution(self) -> bool:
        value = self.state.endpoint_position_m
        return value is not None and value.shape == (3,) and bool(np.all(np.isfinite(value)))


@dataclass(frozen=True)
class ArmOutcome:
    summary: Mapping[str, Any]
    trace: Mapping[str, np.ndarray]


def _first_sustained_time(times: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if times.shape != mask.shape:
        raise ValueError("times and mask must have identical shapes")
    suffix = True
    first: Optional[float] = None
    for index in range(mask.size - 1, -1, -1):
        suffix = bool(suffix and bool(mask[index]))
        if suffix:
            first = float(times[index])
    return first


def _json_float(value: float) -> Optional[float]:
    number = float(value)
    return number if math.isfinite(number) else None


def run_positioning_arm(
    *,
    cfg: UUV3DConfig,
    tape: ExogenousNoiseTape,
    episode_seed: int,
    episode_index: int,
    arm: str,
    estimator_config: v19.BatchEstimatorConfig,
) -> ArmOutcome:
    """Run one fixed-horizon V20 arm on a supplied immutable noise tape."""

    arm_name = str(arm)
    if arm_name not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm_name!r}; expected one of {ARM_NAMES}")
    env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    env.attach_exogenous_noise_tape(tape)
    env.reset(seed=int(episode_seed))
    initial_truth = np.asarray(env.pF, dtype=np.float64).copy()
    initial_leader_centroid = 0.5 * (
        np.asarray(env.pL1, dtype=np.float64)
        + np.asarray(env.pL2, dtype=np.float64)
    )
    recorder: Optional[OnlineHistoryRecorder] = None
    batch: Optional[StreamingBatchEstimator] = None
    if arm_name == "batch_pid":
        recorder = OnlineHistoryRecorder(env)
        batch = StreamingBatchEstimator(
            config=estimator_config,
            support_radius_min_m=float(cfg.start_rho_min),
            support_radius_max_m=float(cfg.start_rho_max),
            episode_index=int(episode_index),
        )

    trace_rows: Dict[str, List[Any]] = {
        "step": [],
        "time_s": [],
        "phase_track": [],
        "action_speed": [],
        "action_yaw": [],
        "action_pitch": [],
        "truth_x": [],
        "truth_y": [],
        "truth_z": [],
        "estimate_x": [],
        "estimate_y": [],
        "estimate_z": [],
        "formation_error_truth_m": [],
        "localization_error_m": [],
        "batch_local_radius95_m": [],
    }
    state_at_track_start: Optional[np.ndarray] = None
    actions: List[np.ndarray] = []
    try:
        track_start_step = int(round(TRACK_START_S / float(cfg.action_dt)))
        for _ in range(int(cfg.max_steps)):
            now = float(env.t)
            track = int(env.step_count) >= track_start_step
            estimate_for_action: Optional[np.ndarray] = None
            if not track:
                action = position_independent_acquisition_action(env, now)
            elif arm_name == "pf_pid":
                estimate_for_action = np.asarray(env.pf.mean, dtype=np.float64).copy()
                action = pid_action_for_position(env, estimate_for_action)
            elif arm_name == "oracle_pid":
                estimate_for_action = np.asarray(env.pF, dtype=np.float64).copy()
                action = pid_action_for_position(env, estimate_for_action)
            else:
                if batch is None or not batch.has_solution:
                    action = position_independent_acquisition_action(env, now)
                else:
                    estimate_for_action = np.asarray(
                        batch.state.endpoint_position_m, dtype=np.float64
                    ).copy()
                    action = pid_action_for_position(env, estimate_for_action)
            if state_at_track_start is None and track:
                state_at_track_start = np.asarray(env.pF, dtype=np.float64).copy()
            actions.append(np.asarray(action, dtype=np.float64).copy())
            _, _, terminated, truncated, _ = env.step(action)
            if recorder is not None and batch is not None:
                batch.update(
                    recorder.history(),
                    decision_time_s=float(env.step_count) * float(cfg.action_dt),
                    current_dead_reckoning_m=recorder.accumulated_dead_reckoning_m,
                )

            truth = np.asarray(env.pF, dtype=np.float64).copy()
            if arm_name == "pf_pid":
                endpoint_estimate = np.asarray(env.pf.mean, dtype=np.float64).copy()
            elif arm_name == "oracle_pid":
                endpoint_estimate = truth.copy()
            elif batch is not None and batch.has_solution:
                endpoint_estimate = np.asarray(
                    batch.state.endpoint_position_m, dtype=np.float64
                ).copy()
            else:
                endpoint_estimate = np.full(3, np.nan, dtype=np.float64)
            _, desired = env._formation_desired()
            formation_error = float(np.linalg.norm(truth - np.asarray(desired)))
            localization_error = float(np.linalg.norm(endpoint_estimate - truth))
            radius95 = float("nan")
            if batch is not None and batch.state.latest_mode is not None:
                radius95 = float(batch.state.latest_mode.local_radius95_m)
            trace_rows["step"].append(int(env.step_count))
            trace_rows["time_s"].append(float(env.t))
            trace_rows["phase_track"].append(float(track))
            trace_rows["action_speed"].append(float(action[0]))
            trace_rows["action_yaw"].append(float(action[1]))
            trace_rows["action_pitch"].append(float(action[2]))
            trace_rows["truth_x"].append(float(truth[0]))
            trace_rows["truth_y"].append(float(truth[1]))
            trace_rows["truth_z"].append(float(truth[2]))
            trace_rows["estimate_x"].append(float(endpoint_estimate[0]))
            trace_rows["estimate_y"].append(float(endpoint_estimate[1]))
            trace_rows["estimate_z"].append(float(endpoint_estimate[2]))
            trace_rows["formation_error_truth_m"].append(formation_error)
            trace_rows["localization_error_m"].append(localization_error)
            trace_rows["batch_local_radius95_m"].append(radius95)
            if terminated or truncated:
                if int(env.step_count) != int(cfg.max_steps):
                    raise RuntimeError("V20 arm ended before the fixed 440 s horizon")
                break
        cursor = env._v11_noise_cursor
        if cursor is None:
            raise RuntimeError("V20 environment lost its attached noise tape")
        noise_cursor = dict(cursor.state_dict())
    finally:
        if recorder is not None:
            recorder.restore()
        env.close()

    trace = {name: np.asarray(values) for name, values in trace_rows.items()}
    if trace["time_s"].shape != (int(cfg.max_steps),):
        raise RuntimeError("V20 trace does not contain the fixed number of actions")
    times = np.asarray(trace["time_s"], dtype=np.float64)
    formation = np.asarray(trace["formation_error_truth_m"], dtype=np.float64)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    joint = (formation < TERMINAL_FORMATION_GATE_M) & (
        localization < TERMINAL_LOCALIZATION_GATE_M
    )
    tail = joint[-TAIL_WINDOW_ACTIONS:]
    dwell = joint[-DWELL_ACTIONS:]
    track_mask = times >= TRACK_START_S
    batch_summary: Dict[str, Any] = {
        "measurement_count": 0,
        "global_solve_count": 0,
        "local_solve_count": 0,
        "total_runtime_s": 0.0,
        "global_runtime_s": 0.0,
        "local_runtime_s": 0.0,
        "global_records": [],
    }
    if recorder is not None and batch is not None:
        batch_summary.update(
            {
                "measurement_count": int(recorder.measurement_count),
                "global_solve_count": int(batch.state.global_solve_count),
                "local_solve_count": int(batch.state.local_solve_count),
                "total_runtime_s": float(batch.state.total_runtime_s),
                "global_runtime_s": float(batch.state.global_runtime_s),
                "local_runtime_s": float(batch.state.local_runtime_s),
                "global_records": list(batch.global_records),
                "initial_position_error_m": (
                    None
                    if batch.state.initial_position_m is None
                    else float(
                        np.linalg.norm(batch.state.initial_position_m - initial_truth)
                    )
                ),
            }
        )
    summary: Dict[str, Any] = {
        "version": VERSION,
        "arm": arm_name,
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "noise_tape_sha256": tape.content_sha256(),
        "track_start_s": TRACK_START_S,
        "action_count": int(times.size),
        "noise_cursor": noise_cursor,
        "initial_truth_m": initial_truth.tolist(),
        "initial_leader_centroid_m": initial_leader_centroid.tolist(),
        "state_at_track_start_m": (
            None if state_at_track_start is None else state_at_track_start.tolist()
        ),
        "terminal_formation_error_m": float(formation[-1]),
        "terminal_localization_error_m": float(localization[-1]),
        "terminal_formation_success": bool(
            formation[-1] < TERMINAL_FORMATION_GATE_M
        ),
        "terminal_localization_success": bool(
            localization[-1] < TERMINAL_LOCALIZATION_GATE_M
        ),
        "terminal_joint_success": bool(joint[-1]),
        "dwell15_joint_success": bool(np.all(dwell)),
        "tail50_joint_occupancy": float(np.mean(tail)),
        "tail80_joint_success": bool(float(np.mean(tail)) >= 0.8),
        "time_to_sustained_joint_lock_s": _first_sustained_time(times, joint),
        "mean_formation_error_after_120_m": float(np.mean(formation[track_mask])),
        "max_formation_error_after_120_m": float(np.max(formation[track_mask])),
        "mean_localization_error_after_120_m": _json_float(
            float(np.nanmean(localization[track_mask]))
        ),
        "mean_squared_action": float(np.mean(np.sum(action_array * action_array, axis=1))),
        "batch": batch_summary,
    }
    return ArmOutcome(summary=summary, trace=trace)


__all__ = [
    "ARM_NAMES",
    "GLOBAL_REFRESH_S",
    "TRACK_START_S",
    "ArmOutcome",
    "OnlineHistoryRecorder",
    "StreamingBatchEstimator",
    "pid_action_for_position",
    "position_independent_acquisition_action",
    "run_positioning_arm",
]
