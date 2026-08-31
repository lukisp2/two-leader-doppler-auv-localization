#!/usr/bin/env python3
"""Append-only V35 closed-loop stress and causal bias-model switch.

This module imports the frozen V24/V30--V32 implementations.  It does not
modify their sources.  Simulator truth is accepted only by the outer episode
runner after each action for scoring; the online classes below operate on the
deployable V19 history and broadcast/DR view.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from uuv_v11_rng import ExogenousNoiseTape
from uuv_v18_resampling_guard import UUV3DConfig, UUVTwoLeader3DPFEnv
import uuv_v19_observability as v19
import uuv_v20_positioning_ablation as v20
import uuv_v21_causal_lock as v21
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24
import uuv_v27_publication_baselines as v27
import uuv_v28_estimator_stress as v28
import uuv_v30_profiled_bias as v30
import uuv_v32_bias_time_to_evidence as v32


VERSION = "v35_closed_loop_stress_1.0"
PRIMARY_ARM = "v24_nominal_model"
BIAS_SWITCH_ARM = "v35_bias_gate_300"
ARMS = (PRIMARY_ARM, BIAS_SWITCH_ARM)

CONDITIONS: Tuple[str, ...] = (
    v28.NOMINAL,
    v28.DOPPLER_COMMON_BIAS,
    v28.DOPPLER_DIFFERENTIAL_BIAS,
    v28.DOPPLER_SCALE,
    v28.COLORED_NOISE,
    v28.DROPOUT,
    v28.BROADCAST_DELAY,
    v28.DR_SCALE,
)
BIAS_SWITCH_CONDITIONS: Tuple[str, ...] = (
    v28.NOMINAL,
    v28.DOPPLER_COMMON_BIAS,
    v28.DOPPLER_DIFFERENTIAL_BIAS,
    v28.COLORED_NOISE,
)

MODEL_DECISION_TIME_S = 300.0
PROFILE_PREVIOUS_TIME_S = 240.0
STRESS_MEASUREMENT_COUNT = 440
BOOTSTRAP_SEED = 35_048_100
RESERVED_START = 49_900
FINAL_END = 50_999


def assert_v35_seed_allowed(seed: int) -> None:
    value = int(seed)
    if RESERVED_START <= value <= FINAL_END:
        raise PermissionError(
            f"V35 refuses reserved/final seed {value}; "
            f"{RESERVED_START}..{FINAL_END} remain closed"
        )


def arm_condition_pairs() -> Tuple[Tuple[str, str], ...]:
    pairs: List[Tuple[str, str]] = [
        (condition, PRIMARY_ARM) for condition in CONDITIONS
    ]
    pairs.extend((condition, BIAS_SWITCH_ARM) for condition in BIAS_SWITCH_CONDITIONS)
    return tuple(pairs)


@dataclass(frozen=True)
class StressTape:
    """Arm-independent deterministic perturbations for one condition/episode."""

    condition: str
    episode_index: int
    colored_noise_mps: np.ndarray
    dropout_keep: np.ndarray

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unsupported V35 condition {self.condition!r}")
        colored = np.asarray(self.colored_noise_mps, dtype=np.float64)
        keep = np.asarray(self.dropout_keep, dtype=bool)
        if colored.shape != (STRESS_MEASUREMENT_COUNT, 2):
            raise ValueError("colored-noise tape has the wrong shape")
        if keep.shape != (STRESS_MEASUREMENT_COUNT,):
            raise ValueError("dropout tape has the wrong shape")
        if not np.all(np.isfinite(colored)):
            raise ValueError("colored-noise tape contains non-finite values")
        object.__setattr__(self, "colored_noise_mps", colored.copy())
        object.__setattr__(self, "dropout_keep", keep.copy())

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(VERSION.encode("utf-8"))
        digest.update(self.condition.encode("utf-8"))
        digest.update(np.asarray([self.episode_index], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(self.colored_noise_mps).tobytes())
        digest.update(np.ascontiguousarray(self.dropout_keep, dtype=np.uint8).tobytes())
        return digest.hexdigest()


def make_stress_tape(condition: str, episode_index: int) -> StressTape:
    name = str(condition)
    if name not in CONDITIONS:
        raise ValueError(name)
    index = int(episode_index)
    if index < 0:
        raise ValueError("episode index must be non-negative")
    code = int(v28.CONDITION_CODE[name])
    rng = np.random.Generator(
        np.random.PCG64(int(v28.BASE_DESIGN_SEED) + 1009 * index + code)
    )
    colored = np.zeros((STRESS_MEASUREMENT_COUNT, 2), dtype=np.float64)
    keep = np.ones(STRESS_MEASUREMENT_COUNT, dtype=bool)
    if name == v28.COLORED_NOISE:
        colored[0] = rng.normal(
            0.0, float(v28.COLORED_STATIONARY_SD_MPS), size=2
        )
        innovation_sd = float(v28.COLORED_STATIONARY_SD_MPS) * math.sqrt(
            1.0 - float(v28.COLORED_RHO) ** 2
        )
        for row in range(1, STRESS_MEASUREMENT_COUNT):
            colored[row] = (
                float(v28.COLORED_RHO) * colored[row - 1]
                + rng.normal(0.0, innovation_sd, size=2)
            )
    elif name == v28.DROPOUT:
        keep = rng.random(STRESS_MEASUREMENT_COUNT) >= float(
            v28.DROPOUT_PROBABILITY
        )
        # Match the V28 nested-prefix contract.
        keep[0] = True
        keep[119] = True
        keep[-1] = True
    return StressTape(
        condition=name,
        episode_index=index,
        colored_noise_mps=colored,
        dropout_keep=keep,
    )


def _formation_desired_from_broadcast(
    cfg: UUV3DConfig,
    leader_position_m: np.ndarray,
    leader_velocity_mps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(leader_position_m, dtype=np.float64)
    velocities = np.asarray(leader_velocity_mps, dtype=np.float64)
    if positions.shape != (2, 3) or velocities.shape != (2, 3):
        raise ValueError("leader broadcast must contain two 3-D states")
    center = 0.5 * (positions[0] + positions[1])
    velocity = 0.5 * (velocities[0] + velocities[1])
    horizontal = np.asarray([velocity[0], velocity[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    forward = (
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if norm < 1e-9
        else horizontal / norm
    )
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    right_norm = float(np.linalg.norm(right))
    right = (
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        if right_norm < 1e-9
        else right / right_norm
    )
    desired = center - float(cfg.d_back) * forward + float(cfg.d_right) * right
    desired[2] = float(center[2] + cfg.dz_offset)
    return center, desired


class OnlineEnvironmentView:
    """Truth-free duck-typed view consumed by frozen V20/V22 actions."""

    def __init__(
        self,
        env: UUVTwoLeader3DPFEnv,
        leader_position_m: np.ndarray,
        leader_velocity_mps: np.ndarray,
        measured_speed_mps: float,
        measured_yaw_deg: float,
        measured_pitch_deg: float,
    ) -> None:
        self._env = env
        positions = np.asarray(leader_position_m, dtype=np.float64)
        velocities = np.asarray(leader_velocity_mps, dtype=np.float64)
        self.pL1 = positions[0].copy()
        self.pL2 = positions[1].copy()
        self.vL1 = velocities[0].copy()
        self.vL2 = velocities[1].copy()
        self._v11_speed_meas = float(measured_speed_mps)
        self._v11_yaw_meas = float(measured_yaw_deg)
        self._v11_pitch_meas = float(measured_pitch_deg)
        self.leader1_speed = float(np.linalg.norm(self.vL1[:2]))
        self.leader2_speed = float(np.linalg.norm(self.vL2[:2]))
        self.yaw_L1 = float(math.degrees(math.atan2(self.vL1[1], self.vL1[0])) % 360.0)
        self.yaw_L2 = float(math.degrees(math.atan2(self.vL2[1], self.vL2[0])) % 360.0)

    @property
    def cfg(self) -> UUV3DConfig:
        return self._env.cfg

    @property
    def step_count(self) -> int:
        return int(self._env.step_count)

    def _formation_desired(self) -> Tuple[np.ndarray, np.ndarray]:
        return _formation_desired_from_broadcast(
            self.cfg,
            np.asarray([self.pL1, self.pL2], dtype=np.float64),
            np.asarray([self.vL1, self.vL2], dtype=np.float64),
        )


class CausalStressRecorder:
    """Capture the deployable stressed V19 history without exposing truth."""

    def __init__(
        self,
        env: UUVTwoLeader3DPFEnv,
        stress_tape: StressTape,
    ) -> None:
        self.env = env
        self.stress_tape = stress_tape
        self.condition = stress_tape.condition
        self._accumulated_dr = np.zeros(3, dtype=np.float64)
        self._time: List[float] = []
        self._dr: List[np.ndarray] = []
        self._leader_position: List[np.ndarray] = []
        self._leader_velocity: List[np.ndarray] = []
        self._follower_velocity: List[np.ndarray] = []
        self._doppler: List[np.ndarray] = []
        self._gate: List[np.ndarray] = []
        self._broadcast_buffer: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque()
        self._latest_broadcast_position: Optional[np.ndarray] = None
        self._latest_broadcast_velocity: Optional[np.ndarray] = None
        self._raw_measurement_count = 0
        self._original_predict = env.pf.predict
        self._original_update = env.pf.update_doppler

        def capture_predict(vF_meas: np.ndarray, dt: float) -> Any:
            measured = np.asarray(vF_meas, dtype=np.float64)
            if self.condition == v28.DR_SCALE:
                measured = float(v28.DR_SCALE_FACTOR) * measured
            self._accumulated_dr += measured * float(dt)
            return self._original_predict(vF_meas, dt)

        def capture_update(
            pL_list: List[np.ndarray],
            vL_list: List[np.ndarray],
            vF_meas: np.ndarray,
            s_meas_list: List[Optional[float]],
            **kwargs: Any,
        ) -> Any:
            if any(value is None for value in s_meas_list):
                raise RuntimeError("V35 received a missing raw Doppler value")
            row = int(self._raw_measurement_count)
            if row >= STRESS_MEASUREMENT_COUNT:
                raise RuntimeError("V35 stress tape exhausted")
            now = float(env.next_s_time)
            leader_position = np.asarray(pL_list, dtype=np.float64)
            leader_velocity = np.asarray(vL_list, dtype=np.float64)
            self._broadcast_buffer.append(
                (now, leader_position.copy(), leader_velocity.copy())
            )
            broadcast_position, broadcast_velocity = self._select_broadcast(now)
            self._latest_broadcast_position = broadcast_position.copy()
            self._latest_broadcast_velocity = broadcast_velocity.copy()

            measured_velocity = np.asarray(vF_meas, dtype=np.float64)
            if self.condition == v28.DR_SCALE:
                measured_velocity = float(v28.DR_SCALE_FACTOR) * measured_velocity
            doppler = np.asarray(s_meas_list, dtype=np.float64)
            if self.condition == v28.DOPPLER_COMMON_BIAS:
                doppler = doppler + np.asarray([0.03, 0.03], dtype=np.float64)
            elif self.condition == v28.DOPPLER_DIFFERENTIAL_BIAS:
                doppler = doppler + np.asarray([0.03, -0.03], dtype=np.float64)
            elif self.condition == v28.DOPPLER_SCALE:
                doppler = 1.02 * doppler
            elif self.condition == v28.COLORED_NOISE:
                doppler = doppler + self.stress_tape.colored_noise_mps[row]

            keep = bool(self.stress_tape.dropout_keep[row])
            if self.condition != v28.DROPOUT or keep:
                self._time.append(now)
                self._dr.append(self._accumulated_dr.copy())
                self._leader_position.append(broadcast_position.copy())
                self._leader_velocity.append(broadcast_velocity.copy())
                self._follower_velocity.append(measured_velocity.copy())
                self._doppler.append(doppler.copy())
                self._gate.append(np.ones(2, dtype=np.float64))
            self._raw_measurement_count += 1
            # The legacy PF remains outside the V35 controller information path.
            return self._original_update(
                pL_list,
                vL_list,
                vF_meas,
                s_meas_list,
                **kwargs,
            )

        env.pf.predict = capture_predict  # type: ignore[method-assign]
        env.pf.update_doppler = capture_update  # type: ignore[method-assign]

    def _select_broadcast(self, now_s: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.condition != v28.BROADCAST_DELAY:
            _, positions, velocities = self._broadcast_buffer[-1]
            return positions, velocities
        target = float(now_s) - float(v28.BROADCAST_DELAY_S)
        selected = self._broadcast_buffer[0]
        for item in self._broadcast_buffer:
            if float(item[0]) <= target + 1e-12:
                selected = item
            else:
                break
        # Retain only enough history for the fixed delay plus one sample.
        while len(self._broadcast_buffer) > 5:
            self._broadcast_buffer.popleft()
        return selected[1], selected[2]

    @property
    def measurement_count(self) -> int:
        return len(self._time)

    @property
    def raw_measurement_count(self) -> int:
        return int(self._raw_measurement_count)

    @property
    def accumulated_dead_reckoning_m(self) -> np.ndarray:
        return self._accumulated_dr.copy()

    def history(self) -> v19.OnlineDopplerHistory:
        if not self._time:
            raise RuntimeError("V35 has no retained Doppler measurement")
        return v19.OnlineDopplerHistory(
            t_s=np.asarray(self._time, dtype=np.float64),
            dead_reckoned_displacement_m=np.asarray(self._dr, dtype=np.float64),
            leader_position_m=np.asarray(self._leader_position, dtype=np.float64),
            leader_velocity_mps=np.asarray(self._leader_velocity, dtype=np.float64),
            follower_velocity_measured_mps=np.asarray(
                self._follower_velocity, dtype=np.float64
            ),
            doppler_measured_mps=np.asarray(self._doppler, dtype=np.float64),
            historical_pf_gate_factor=np.asarray(self._gate, dtype=np.float64),
        )

    def online_view(self) -> OnlineEnvironmentView:
        positions = (
            np.asarray([self.env.pL1, self.env.pL2], dtype=np.float64)
            if self._latest_broadcast_position is None
            else self._latest_broadcast_position
        )
        velocities = (
            np.asarray([self.env.vL1, self.env.vL2], dtype=np.float64)
            if self._latest_broadcast_velocity is None
            else self._latest_broadcast_velocity
        )
        speed = float(self.env._v11_speed_meas)
        if self.condition == v28.DR_SCALE:
            speed *= float(v28.DR_SCALE_FACTOR)
        return OnlineEnvironmentView(
            self.env,
            positions,
            velocities,
            speed,
            float(self.env._v11_yaw_meas),
            float(self.env._v11_pitch_meas),
        )

    def restore(self) -> None:
        self.env.pf.predict = self._original_predict  # type: ignore[method-assign]
        self.env.pf.update_doppler = self._original_update  # type: ignore[method-assign]


def _profile_mode_to_batch(mode: v30.ProfiledBiasMode) -> v19.BatchMode:
    covariance = mode.position_covariance_m2
    valid = bool(
        mode.converged
        and covariance is not None
        and mode.nominal_radius95_m is not None
        and math.isfinite(float(mode.nominal_radius95_m))
    )
    if covariance is None:
        cov = np.full((3, 3), np.nan, dtype=np.float64)
        eigenvalues = np.zeros(3, dtype=np.float64)
        rank = 0
        condition = float("inf")
    else:
        cov = 0.5 * (
            np.asarray(covariance, dtype=np.float64)
            + np.asarray(covariance, dtype=np.float64).T
        )
        try:
            precision = np.linalg.pinv(cov, hermitian=True)
            eigenvalues = np.maximum(np.linalg.eigvalsh(precision), 0.0)
            rank = int(np.linalg.matrix_rank(precision, tol=1e-10))
            positive = eigenvalues[eigenvalues > 1e-15]
            condition = (
                float("inf")
                if positive.size < 3
                else float(np.max(positive) / np.min(positive))
            )
            valid = bool(valid and rank == 3 and np.all(np.isfinite(cov)))
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(3, dtype=np.float64)
            rank = 0
            condition = float("inf")
            valid = False
    return v19.BatchMode(
        initial_position_m=np.asarray(mode.initial_position_m, dtype=np.float64).copy(),
        residual_sse_mps2=float(mode.residual_sse_mps2),
        residual_rmse_mps=float(mode.residual_rmse_mps),
        iterations=int(mode.iterations),
        converged=bool(mode.converged),
        hessian_eigenvalues=eigenvalues,
        hessian_rank=rank,
        hessian_condition_number=condition,
        local_covariance_m2=cov,
        local_covariance_valid=valid,
        local_radius95_m=(
            float(mode.nominal_radius95_m)
            if mode.nominal_radius95_m is not None
            else float("inf")
        ),
    )


def _profile_estimate_to_batch(
    estimate: v30.ProfiledBiasEstimate,
) -> v19.BatchEstimate:
    modes = tuple(_profile_mode_to_batch(mode) for mode in estimate.modes)
    alternative_index: Optional[int] = None
    if estimate.alternative_distance_m is not None:
        for index, mode in enumerate(estimate.modes[1:], start=1):
            if (
                float(np.linalg.norm(mode.initial_position_m - estimate.best.initial_position_m))
                >= 7.0
            ):
                alternative_index = index
                break
    delta_sse = (
        None
        if alternative_index is None
        else float(
            estimate.modes[alternative_index].residual_sse_mps2
            - estimate.best.residual_sse_mps2
        )
    )
    return v19.BatchEstimate(
        modes=modes,
        runtime_s=float(estimate.runtime_s),
        coarse_best_rmse_mps=float(estimate.best.residual_rmse_mps),
        candidate_count=int(estimate.candidate_count),
        refined_start_count=int(estimate.refined_start_count),
        clustered_mode_count=int(estimate.clustered_mode_count),
        alternative_mode_index=alternative_index,
        alternative_distance_m=estimate.alternative_distance_m,
        alternative_delta_sse_mps2=delta_sse,
        alternative_delta_chi2=estimate.alternative_delta_chi2,
    )


def _profile_residual_rmse(
    mode: v30.ProfiledBiasMode,
    history: v19.OnlineDopplerHistory,
) -> float:
    prediction = v19.predict_doppler(mode.initial_position_m, history)
    residual = (
        history.doppler_measured_mps
        - prediction
        - np.asarray(mode.link_bias_mps, dtype=np.float64)[None, :]
    )
    return float(math.sqrt(float(np.mean(residual * residual))))


class ProfiledAuditedStreamingEstimator:
    """V24-compatible streaming estimator after the one-shot model switch."""

    def __init__(
        self,
        *,
        estimator_config: v19.BatchEstimatorConfig,
        evaluator_config: v27.EvaluatorConfig,
        lock_config: v24.AuditedLockConfig,
        support_radius_min_m: float,
        support_radius_max_m: float,
        previous_gate: v24.AuditedCausalLockGate,
        previous_state: v21.CausalEstimatorState,
        switch_time_s: float,
    ) -> None:
        self.estimator_config = estimator_config
        self.evaluator_config = evaluator_config
        self.support_radius_min_m = float(support_radius_min_m)
        self.support_radius_max_m = float(support_radius_max_m)
        self.gate = v24.AuditedCausalLockGate(lock_config)
        self.gate.state = copy.deepcopy(previous_gate.state)
        self.state = copy.deepcopy(previous_state)
        self.global_records: List[Dict[str, Any]] = []
        self._previous_profile_mode: Optional[v30.ProfiledBiasMode] = None
        self._previous_global_measurement_count: Optional[int] = None
        self.current_profile_mode: Optional[v30.ProfiledBiasMode] = None
        self.switch_time_s = float(switch_time_s)
        self.profiled_runtime_s = 0.0
        self.profiled_solve_count = 0
        self.activation_runtime_s = 0.0
        self._force_unlock_for_model_switch()

    @property
    def has_solution(self) -> bool:
        value = self.state.endpoint_position_m
        return bool(
            value is not None
            and np.asarray(value).shape == (3,)
            and np.all(np.isfinite(value))
        )

    def _force_unlock_for_model_switch(self) -> None:
        state = self.gate.state
        was_locked = bool(state.locked)
        state.locked = False
        state.release_pass_streak = 0
        state.hold_failure_streak = 0
        state.force_global_refresh = False
        state.last_release_predicate = False
        state.last_hold_predicate = False
        state.last_failed_checks = ("measurement_model_switch",)
        if was_locked:
            state.unlock_count += 1
            state.last_unlock_time_s = self.switch_time_s
            state.transitions.append(
                {
                    "time_s": self.switch_time_s,
                    "from": "TRACK",
                    "to": "ACQUIRE",
                    "failed_checks": ["measurement_model_switch"],
                }
            )

    def _global_config(self, seed: int) -> v27.EvaluatorConfig:
        base = self.evaluator_config
        return v27.EvaluatorConfig(
            measurement_sigma_mps=base.measurement_sigma_mps,
            coarse_candidates=base.coarse_candidates,
            coarse_sweeps=base.coarse_sweeps,
            local_starts=base.local_starts,
            maximum_modes=base.maximum_modes,
            window_s=base.window_s,
            pf_particles_small=base.pf_particles_small,
            pf_particles_large=base.pf_particles_large,
            pf_ess_fraction=base.pf_ess_fraction,
            pf_kernel_h=base.pf_kernel_h,
            global_candidate_seed=int(seed),
            window_candidate_seed=base.window_candidate_seed,
            pf_design_seed=base.pf_design_seed,
        )

    def _profile_global(
        self,
        history: v19.OnlineDopplerHistory,
        seed: int,
    ) -> v30.ProfiledBiasEstimate:
        center = v19.initial_leader_centroid_from_history(history)
        estimate = v30.estimate_profiled_bias_global(
            history,
            center,
            self.support_radius_min_m,
            self.support_radius_max_m,
            self._global_config(seed),
        )
        self.profiled_runtime_s += float(estimate.runtime_s)
        self.profiled_solve_count += 1
        return estimate

    def activate(
        self,
        history: v19.OnlineDopplerHistory,
        *,
        current_dead_reckoning_m: Sequence[float],
    ) -> None:
        started = time.perf_counter()
        previous_history = history.prefix(PROFILE_PREVIOUS_TIME_S)
        previous = self._profile_global(previous_history, 21_001)
        primary = self._profile_global(history, 21_001)
        confirmation = self._profile_global(history, 1_021_001)
        current_mode = primary.best
        validation_mask = history.t_s > PROFILE_PREVIOUS_TIME_S + 1e-12
        validation = history.take(validation_mask)
        forward_rmse = _profile_residual_rmse(previous.best, validation)
        stability = float(
            np.linalg.norm(
                current_mode.initial_position_m - previous.best.initial_position_m
            )
        )
        primary_batch = _profile_estimate_to_batch(primary)
        confirmation_batch = _profile_estimate_to_batch(confirmation)
        evidence = v21.GlobalLockEvidence(
            time_s=MODEL_DECISION_TIME_S,
            primary=primary_batch,
            confirmation=confirmation_batch,
            search_agreement_m=float(
                np.linalg.norm(
                    primary.best.initial_position_m
                    - confirmation.best.initial_position_m
                )
            ),
            stability_from_previous_m=stability,
            forward_prediction_rmse_mps=forward_rmse,
            forward_prediction_sample_count=int(validation.measurement_count),
        )
        self.current_profile_mode = current_mode
        mode = primary_batch.best
        current_dr = np.asarray(current_dead_reckoning_m, dtype=np.float64)
        self.state.initial_position_m = mode.initial_position_m.copy()
        self.state.endpoint_position_m = mode.initial_position_m + current_dr
        self.state.latest_mode = mode
        self.state.latest_global_evidence = evidence
        self.state.recent_residual_rmse_mps = _profile_residual_rmse(
            current_mode, history.take(slice(max(0, history.measurement_count - 20), None))
        )
        self._previous_profile_mode = current_mode
        self._previous_global_measurement_count = int(history.measurement_count)
        self.gate.evaluate(
            now_s=MODEL_DECISION_TIME_S,
            mode=mode,
            initial_position_m=mode.initial_position_m,
            recent_residual_rmse_mps=self.state.recent_residual_rmse_mps,
            global_evidence=evidence,
        )
        elapsed = float(time.perf_counter() - started)
        self.activation_runtime_s = elapsed
        self.state.total_runtime_s += elapsed
        self.state.global_runtime_s += elapsed
        self.state.global_solve_count += 3
        self.state.maximum_update_runtime_s = max(
            float(self.state.maximum_update_runtime_s), elapsed
        )
        self.global_records.append(
            {
                "time_s": MODEL_DECISION_TIME_S,
                "activation": True,
                "retrospective_time_s": PROFILE_PREVIOUS_TIME_S,
                "runtime_s": elapsed,
                "link_bias_mps": current_mode.link_bias_mps.tolist(),
                **evidence.to_dict(),
            }
        )

    def _forward_residual(
        self,
        history: v19.OnlineDopplerHistory,
    ) -> Tuple[Optional[float], int]:
        if (
            self._previous_profile_mode is None
            or self._previous_global_measurement_count is None
        ):
            return None, 0
        start = int(self._previous_global_measurement_count)
        if start >= history.measurement_count:
            return None, 0
        held_out = history.take(slice(start, history.measurement_count))
        return (
            _profile_residual_rmse(self._previous_profile_mode, held_out),
            int(held_out.measurement_count),
        )

    def update(
        self,
        history: v19.OnlineDopplerHistory,
        *,
        decision_time_s: float,
        current_dead_reckoning_m: Sequence[float],
    ) -> None:
        now = float(decision_time_s)
        current_dr = np.asarray(current_dead_reckoning_m, dtype=np.float64)
        if current_dr.shape != (3,) or not np.all(np.isfinite(current_dr)):
            raise ValueError("invalid current dead reckoning")
        center = v19.initial_leader_centroid_from_history(history)
        forced = self.gate.consume_force_global_refresh()
        global_update = bool(forced or v21.CausalStreamingEstimator._scheduled_global(now))
        started = time.perf_counter()
        if global_update:
            primary = self._profile_global(history, 21_001)
            confirmation = self._profile_global(history, 1_021_001)
            forward_rmse, forward_count = self._forward_residual(history)
            previous_position = (
                None
                if self._previous_profile_mode is None
                else self._previous_profile_mode.initial_position_m
            )
            stability = (
                None
                if previous_position is None
                else float(
                    np.linalg.norm(primary.best.initial_position_m - previous_position)
                )
            )
            primary_batch = _profile_estimate_to_batch(primary)
            confirmation_batch = _profile_estimate_to_batch(confirmation)
            evidence = v21.GlobalLockEvidence(
                time_s=now,
                primary=primary_batch,
                confirmation=confirmation_batch,
                search_agreement_m=float(
                    np.linalg.norm(
                        primary.best.initial_position_m
                        - confirmation.best.initial_position_m
                    )
                ),
                stability_from_previous_m=stability,
                forward_prediction_rmse_mps=forward_rmse,
                forward_prediction_sample_count=forward_count,
            )
            profile_mode = primary.best
            mode = primary_batch.best
            self.state.latest_global_evidence = evidence
            self._previous_profile_mode = profile_mode
            self._previous_global_measurement_count = int(history.measurement_count)
            self.global_records.append(
                {
                    "time_s": now,
                    "forced": forced,
                    "runtime_s": float(time.perf_counter() - started),
                    "link_bias_mps": profile_mode.link_bias_mps.tolist(),
                    **evidence.to_dict(),
                }
            )
            self.state.global_solve_count += 2
        else:
            if self.current_profile_mode is None:
                raise RuntimeError("profiled local update before activation")
            profile_mode = v30.refine_profiled_bias(
                self.current_profile_mode.initial_position_m,
                history,
                center,
                self.support_radius_min_m,
                self.support_radius_max_m,
                self.estimator_config,
            )
            mode = _profile_mode_to_batch(profile_mode)
            self.profiled_solve_count += 1
            self.state.local_solve_count += 1
        elapsed = float(time.perf_counter() - started)
        self.profiled_runtime_s += elapsed
        self.current_profile_mode = profile_mode
        self.state.total_runtime_s += elapsed
        if global_update:
            self.state.global_runtime_s += elapsed
        else:
            self.state.local_runtime_s += elapsed
        self.state.maximum_update_runtime_s = max(
            float(self.state.maximum_update_runtime_s), elapsed
        )
        self.state.initial_position_m = mode.initial_position_m.copy()
        self.state.endpoint_position_m = mode.initial_position_m + current_dr
        self.state.latest_mode = mode
        recent = history.take(slice(max(0, history.measurement_count - 20), None))
        self.state.recent_residual_rmse_mps = _profile_residual_rmse(
            profile_mode, recent
        )
        self.gate.evaluate(
            now_s=now,
            mode=mode,
            initial_position_m=mode.initial_position_m,
            recent_residual_rmse_mps=self.state.recent_residual_rmse_mps,
            global_evidence=self.state.latest_global_evidence,
        )


@dataclass(frozen=True)
class V35ArmOutcome:
    summary: Mapping[str, Any]
    trace: Mapping[str, np.ndarray]


def _first_sustained_time(times: np.ndarray, mask: np.ndarray) -> Optional[float]:
    return v21._first_sustained_time(times, mask)


def _json_float(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def _audit_trace_values(gate: v24.AuditedCausalLockGate) -> Dict[str, float]:
    return v24._audit_trace_values(gate)


def run_stressed_arm(
    *,
    cfg: UUV3DConfig,
    tape: ExogenousNoiseTape,
    stress_tape: StressTape,
    episode_seed: int,
    episode_index: int,
    condition: str,
    arm: str,
    estimator_config: v19.BatchEstimatorConfig,
    evaluator_config: v27.EvaluatorConfig,
    lock_config: v24.AuditedLockConfig,
    planner_config: v22.ActivePlannerConfig,
) -> V35ArmOutcome:
    if condition not in CONDITIONS or arm not in ARMS:
        raise ValueError("unknown V35 condition/arm")
    if arm == BIAS_SWITCH_ARM and condition not in BIAS_SWITCH_CONDITIONS:
        raise ValueError("bias-switch arm is not contracted for this condition")
    assert_v35_seed_allowed(episode_seed)
    env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    env.attach_exogenous_noise_tape(tape)
    env.reset(seed=int(episode_seed))
    initial_truth = np.asarray(env.pF, dtype=np.float64).copy()
    initial_centroid = 0.5 * (
        np.asarray(env.pL1, dtype=np.float64)
        + np.asarray(env.pL2, dtype=np.float64)
    )
    recorder = CausalStressRecorder(env, stress_tape)
    estimator: Any = v24.AuditedCausalStreamingEstimator(
        estimator_config=estimator_config,
        lock_config=lock_config,
        support_radius_min_m=float(cfg.start_rho_min),
        support_radius_max_m=float(cfg.start_rho_max),
    )
    planner = v22.BeliefFIMPlanner(planner_config)
    model_decision_evidence: Optional[Mapping[str, Any]] = None
    model_decision_runtime_s = 0.0
    profile_activated = False
    switch_position_jump_m: Optional[float] = None
    switch_after_row_index: Optional[int] = None

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
        "model_decision_evaluated", "model_bias_activated",
        "profile_model_active", "model_decision_runtime_s",
    )
    trace_rows: Dict[str, List[Any]] = {field: [] for field in fields}
    actions: List[np.ndarray] = []
    maximum_combined_runtime_s = 0.0
    try:
        for _ in range(int(cfg.max_steps)):
            action_start = float(env.t)
            track = bool(estimator.gate.is_locked and estimator.has_solution)
            decision: Optional[v22.PlannerDecision] = None
            online_view = recorder.online_view()
            if track:
                action = v20.pid_action_for_position(
                    online_view,  # type: ignore[arg-type]
                    np.asarray(estimator.state.endpoint_position_m, dtype=np.float64),
                )
            elif not estimator.has_solution:
                action = v20.position_independent_acquisition_action(
                    online_view, action_start  # type: ignore[arg-type]
                )
            else:
                decision = planner.action(
                    online_view,  # type: ignore[arg-type]
                    history=recorder.history(),
                    current_dead_reckoning_m=recorder.accumulated_dead_reckoning_m,
                    estimator=estimator,
                    time_s=action_start,
                )
                action = decision.action
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (3,) or not np.all(np.isfinite(action)):
                raise RuntimeError("V35 produced a non-finite action")
            actions.append(np.asarray(action, dtype=np.float64).copy())
            _, _, terminated, truncated, _ = env.step(action)
            update_started = time.perf_counter()
            estimator.update(
                recorder.history(),
                decision_time_s=float(env.step_count) * float(cfg.action_dt),
                current_dead_reckoning_m=recorder.accumulated_dead_reckoning_m,
            )
            update_runtime = float(time.perf_counter() - update_started)

            # Snapshot the ordinary post-update state.  The one-shot model
            # treatment is applied at the boundary after this row, so both
            # paired traces remain identical through 300 s.
            truth = np.asarray(env.pF, dtype=np.float64).copy()
            endpoint = (
                np.asarray(estimator.state.endpoint_position_m, dtype=np.float64).copy()
                if estimator.has_solution
                else np.full(3, np.nan, dtype=np.float64)
            )
            _, desired_truth = env._formation_desired()
            formation = float(np.linalg.norm(truth - np.asarray(desired_truth)))
            localization = float(np.linalg.norm(endpoint - truth))
            mode = estimator.state.latest_mode
            radius = float("nan") if mode is None else float(mode.local_radius95_m)
            pre_switch_gate_locked = bool(estimator.gate.is_locked)
            profile_active_before_row = bool(profile_activated)
            gate_values = _audit_trace_values(estimator.gate)

            evaluated_now = False
            activated_now = False
            if (
                arm == BIAS_SWITCH_ARM
                and model_decision_evidence is None
                and math.isclose(float(env.t), MODEL_DECISION_TIME_S, abs_tol=1e-8)
            ):
                evaluated_now = True
                evaluation_started = time.perf_counter()
                evaluation = v32.evaluate_checkpoint(
                    recorder.history(),
                    MODEL_DECISION_TIME_S,
                    float(cfg.start_rho_min),
                    float(cfg.start_rho_max),
                    evaluator_config,
                )
                model_decision_evidence = evaluation.evidence.to_dict()
                model_decision_runtime_s = float(time.perf_counter() - evaluation_started)
                if bool(evaluation.evidence.activate_bias_model):
                    nominal_endpoint = endpoint.copy()
                    profiled = ProfiledAuditedStreamingEstimator(
                        estimator_config=estimator_config,
                        evaluator_config=evaluator_config,
                        lock_config=lock_config,
                        support_radius_min_m=float(cfg.start_rho_min),
                        support_radius_max_m=float(cfg.start_rho_max),
                        previous_gate=estimator.gate,
                        previous_state=estimator.state,
                        switch_time_s=MODEL_DECISION_TIME_S,
                    )
                    profiled.global_records = list(estimator.global_records)
                    profiled.activate(
                        recorder.history(),
                        current_dead_reckoning_m=recorder.accumulated_dead_reckoning_m,
                    )
                    estimator = profiled
                    profile_activated = True
                    activated_now = True
                    switch_after_row_index = int(env.step_count - 1)
                    switch_position_jump_m = float(
                        np.linalg.norm(
                            np.asarray(estimator.state.endpoint_position_m)
                            - nominal_endpoint
                        )
                    )
                    model_decision_runtime_s += float(profiled.activation_runtime_s)

            combined_runtime = (
                update_runtime
                + (0.0 if decision is None else float(decision.runtime_s))
                + (model_decision_runtime_s if evaluated_now else 0.0)
            )
            maximum_combined_runtime_s = max(
                maximum_combined_runtime_s, combined_runtime
            )
            base_values: Dict[str, Any] = {
                "step": int(env.step_count),
                "action_start_time_s": action_start,
                "time_s": float(env.t),
                "phase_track": float(track),
                "gate_locked_after_update": float(pre_switch_gate_locked),
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
                "batch_local_radius95_m": radius,
                "planner_utility": float("nan") if decision is None else decision.utility,
                "planner_worst_radius_before_m": float("nan") if decision is None else decision.worst_radius_before_m,
                "planner_worst_radius_after_m": float("nan") if decision is None else decision.worst_radius_after_m,
                "planner_minimum_pair_chi2": float("nan") if decision is None else decision.minimum_pair_chi2,
                "planner_hypothesis_count": 0 if decision is None else decision.hypothesis_count,
                "planner_runtime_s": 0.0 if decision is None else decision.runtime_s,
                "model_decision_evaluated": float(evaluated_now),
                "model_bias_activated": float(activated_now),
                "profile_model_active": float(profile_active_before_row),
                "model_decision_runtime_s": model_decision_runtime_s if evaluated_now else 0.0,
            }
            base_values.update(gate_values)
            for field in fields:
                trace_rows[field].append(base_values[field])
            if terminated or truncated:
                if int(env.step_count) != int(cfg.max_steps):
                    raise RuntimeError("V35 arm ended before the fixed horizon")
                break
        cursor = env._v11_noise_cursor
        if cursor is None:
            raise RuntimeError("V35 environment lost its noise tape")
        noise_cursor = dict(cursor.state_dict())
    finally:
        recorder.restore()
        env.close()

    trace = {field: np.asarray(values) for field, values in trace_rows.items()}
    if trace["time_s"].shape != (int(cfg.max_steps),):
        raise RuntimeError("V35 trace has the wrong fixed-horizon length")
    if recorder.raw_measurement_count != STRESS_MEASUREMENT_COUNT:
        raise RuntimeError("V35 did not consume the complete stress tape")
    times = np.asarray(trace["time_s"], dtype=np.float64)
    formation = np.asarray(trace["formation_error_truth_m"], dtype=np.float64)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    joint = (formation < v21.TERMINAL_FORMATION_GATE_M) & (
        localization < v21.TERMINAL_LOCALIZATION_GATE_M
    )
    exact = v24.exact_trace_score(trace)
    gate_after = np.asarray(trace["gate_locked_after_update"], dtype=bool)
    transition = gate_after & ~np.concatenate([np.asarray([False]), gate_after[:-1]])
    audit_release_violation_count = int(
        np.sum(
            transition
            & ~np.asarray(trace["audit_release_checks_pass"], dtype=bool)
        )
    )
    post300 = times > MODEL_DECISION_TIME_S + 1e-9
    phase = np.asarray(trace["phase_track"], dtype=bool)
    post300_unsafe_track_end_count = int(
        np.sum(post300 & phase & ((~np.isfinite(localization)) | (localization >= 7.0)))
    )
    action_array = np.asarray(actions, dtype=np.float64)
    tail = joint[-v21.TAIL_WINDOW_ACTIONS :]
    dwell = joint[-v21.DWELL_ACTIONS :]
    history = recorder.history()
    trace.update(
        {
            "online_t_s": history.t_s.copy(),
            "online_dead_reckoned_displacement_m": history.dead_reckoned_displacement_m.copy(),
            "online_leader_position_m": history.leader_position_m.copy(),
            "online_leader_velocity_mps": history.leader_velocity_mps.copy(),
            "online_follower_velocity_measured_mps": history.follower_velocity_measured_mps.copy(),
            "online_doppler_measured_mps": history.doppler_measured_mps.copy(),
            "online_historical_pf_gate_factor": history.historical_pf_gate_factor.copy(),
        }
    )
    gate_state = estimator.gate.state
    profile_mode = getattr(estimator, "current_profile_mode", None)
    summary: Dict[str, Any] = {
        "version": VERSION,
        "arm": arm,
        "condition": condition,
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "noise_tape_sha256": tape.content_sha256(),
        "stress_tape_sha256": stress_tape.content_sha256(),
        "online_history_sha256": v28.history_sha256(history),
        "action_count": int(times.size),
        "retained_measurement_count": int(recorder.measurement_count),
        "raw_measurement_count": int(recorder.raw_measurement_count),
        "noise_cursor": noise_cursor,
        "initial_truth_m": initial_truth.tolist(),
        "initial_leader_centroid_m": initial_centroid.tolist(),
        "terminal_formation_error_m": float(formation[-1]),
        "terminal_localization_error_m": float(localization[-1]),
        "terminal_joint_success": bool(joint[-1]),
        "dwell15_joint_success": bool(np.all(dwell)),
        "tail50_joint_occupancy": float(np.mean(tail)),
        "tail80_joint_success": bool(float(np.mean(tail)) >= 0.8),
        "time_to_sustained_joint_lock_s": _first_sustained_time(times, joint),
        "mean_squared_action": float(np.mean(np.sum(action_array * action_array, axis=1))),
        "maximum_combined_decision_runtime_s": float(maximum_combined_runtime_s),
        "gate": {
            "ever_locked": bool(exact["transition_count"] > 0),
            "lock_count": int(gate_state.lock_count),
            "unlock_count": int(gate_state.unlock_count),
            "audit_release_violation_count": audit_release_violation_count,
            "exact": exact,
            "transitions": list(gate_state.transitions),
            "post300_unsafe_track_end_count": post300_unsafe_track_end_count,
        },
        "model_switch": {
            "evaluated": model_decision_evidence is not None,
            "activated": bool(profile_activated),
            "decision_time_s": MODEL_DECISION_TIME_S if model_decision_evidence is not None else None,
            "decision_runtime_s": float(model_decision_runtime_s),
            "evidence": model_decision_evidence,
            "switch_after_row_index": switch_after_row_index,
            "position_jump_m": switch_position_jump_m,
            "terminal_link_bias_mps": (
                None if profile_mode is None else profile_mode.link_bias_mps.tolist()
            ),
        },
        "batch": {
            "global_solve_count": int(estimator.state.global_solve_count),
            "local_solve_count": int(estimator.state.local_solve_count),
            "total_runtime_s": float(estimator.state.total_runtime_s),
            "maximum_update_runtime_s": float(estimator.state.maximum_update_runtime_s),
            "global_records": list(estimator.global_records),
        },
        "sealed_seed_range_untouched": [RESERVED_START, FINAL_END],
    }
    return V35ArmOutcome(summary=summary, trace=trace)


__all__ = [
    "VERSION",
    "PRIMARY_ARM",
    "BIAS_SWITCH_ARM",
    "ARMS",
    "CONDITIONS",
    "BIAS_SWITCH_CONDITIONS",
    "MODEL_DECISION_TIME_S",
    "RESERVED_START",
    "FINAL_END",
    "StressTape",
    "make_stress_tape",
    "CausalStressRecorder",
    "OnlineEnvironmentView",
    "ProfiledAuditedStreamingEstimator",
    "V35ArmOutcome",
    "arm_condition_pairs",
    "assert_v35_seed_allowed",
    "run_stressed_arm",
]
