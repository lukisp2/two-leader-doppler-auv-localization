#!/usr/bin/env python3
"""Truth-free acquisition-to-tracking lock for two-leader Doppler navigation.

V21 imports the frozen V19/V20 implementation and adds an independently
testable confidence gate.  The estimator/gate path never receives simulator
truth, the legacy particle-filter state, reward, or success fields.  Truth is
used only after each action by :func:`run_causal_lock_arm` to score false
locks and task performance.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from uuv_v11_rng import ExogenousNoiseTape
from uuv_v18_resampling_guard import UUV3DConfig, UUVTwoLeader3DPFEnv
import uuv_v19_observability as v19
import uuv_v20_positioning_ablation as v20


VERSION = "v21_causal_lock_1.0"
GLOBAL_REFRESH_S: Tuple[float, ...] = (
    30.0,
    60.0,
    90.0,
    120.0,
    180.0,
    240.0,
    300.0,
    360.0,
    420.0,
    440.0,
)
ESTIMATION_START_S = 30.0
TERMINAL_FORMATION_GATE_M = 8.0
TERMINAL_LOCALIZATION_GATE_M = 7.0
TAIL_WINDOW_ACTIONS = 50
DWELL_ACTIONS = 15
TRUTH_READY_DWELL_ACTIONS = 15


def _finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(float(value))


@dataclass(frozen=True)
class CausalLockConfig:
    """Prespecified V21 release and hysteresis thresholds."""

    minimum_release_time_s: float = 120.0
    release_radius95_m: float = 7.0
    release_full_rmse_mps: float = 0.08
    release_recent_rmse_mps: float = 0.08
    release_global_age_s: float = 65.0
    release_search_agreement_m: float = 3.0
    release_global_stability_m: float = 5.0
    release_forward_rmse_mps: float = 0.08
    release_forward_min_samples: int = 20
    release_alternative_delta_chi2: float = 11.345
    release_local_stability_m: float = 2.0
    release_local_stability_count: int = 3
    release_consecutive_actions: int = 3
    hold_radius95_m: float = 10.0
    hold_full_rmse_mps: float = 0.10
    hold_recent_rmse_mps: float = 0.10
    hold_search_agreement_m: float = 7.0
    hold_forward_rmse_mps: float = 0.10
    hold_alternative_delta_chi2: float = 7.815
    loss_consecutive_actions: int = 3

    def __post_init__(self) -> None:
        positive = (
            self.minimum_release_time_s,
            self.release_radius95_m,
            self.release_full_rmse_mps,
            self.release_recent_rmse_mps,
            self.release_global_age_s,
            self.release_search_agreement_m,
            self.release_global_stability_m,
            self.release_forward_rmse_mps,
            self.release_alternative_delta_chi2,
            self.release_local_stability_m,
            self.hold_radius95_m,
            self.hold_full_rmse_mps,
            self.hold_recent_rmse_mps,
            self.hold_search_agreement_m,
            self.hold_forward_rmse_mps,
            self.hold_alternative_delta_chi2,
        )
        if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in positive):
            raise ValueError("lock thresholds must be finite and positive")
        counts = (
            self.release_forward_min_samples,
            self.release_local_stability_count,
            self.release_consecutive_actions,
            self.loss_consecutive_actions,
        )
        if any(int(value) < 1 for value in counts):
            raise ValueError("lock counters must be positive")


def _mode_is_locally_valid(mode: v19.BatchMode) -> bool:
    return bool(
        mode.converged
        and int(mode.hessian_rank) == 3
        and bool(mode.local_covariance_valid)
        and math.isfinite(float(mode.local_radius95_m))
        and math.isfinite(float(mode.residual_rmse_mps))
    )


def _alternative_is_rejected(
    estimate: v19.BatchEstimate,
    threshold_delta_chi2: float,
) -> bool:
    if estimate.alternative_mode_index is None:
        return True
    if estimate.alternative_distance_m is None:
        return False
    if float(estimate.alternative_distance_m) < 7.0:
        return True
    return bool(
        _finite(estimate.alternative_delta_chi2)
        and float(estimate.alternative_delta_chi2) >= float(threshold_delta_chi2)
    )


@dataclass(frozen=True)
class GlobalLockEvidence:
    time_s: float
    primary: v19.BatchEstimate
    confirmation: v19.BatchEstimate
    search_agreement_m: float
    stability_from_previous_m: Optional[float]
    forward_prediction_rmse_mps: Optional[float]
    forward_prediction_sample_count: int

    def release_checks(
        self,
        now_s: float,
        config: CausalLockConfig,
    ) -> Dict[str, bool]:
        return {
            "global_fresh": float(now_s) - float(self.time_s)
            <= float(config.release_global_age_s) + 1e-9,
            "primary_valid": _mode_is_locally_valid(self.primary.best),
            "confirmation_valid": _mode_is_locally_valid(self.confirmation.best),
            "search_agreement": math.isfinite(float(self.search_agreement_m))
            and float(self.search_agreement_m)
            <= float(config.release_search_agreement_m),
            "global_stability": _finite(self.stability_from_previous_m)
            and float(self.stability_from_previous_m)
            <= float(config.release_global_stability_m),
            "forward_count": int(self.forward_prediction_sample_count)
            >= int(config.release_forward_min_samples),
            "forward_rmse": _finite(self.forward_prediction_rmse_mps)
            and float(self.forward_prediction_rmse_mps)
            <= float(config.release_forward_rmse_mps),
            "primary_mode_clear": _alternative_is_rejected(
                self.primary, config.release_alternative_delta_chi2
            ),
            "confirmation_mode_clear": _alternative_is_rejected(
                self.confirmation, config.release_alternative_delta_chi2
            ),
        }

    def hold_checks(
        self,
        now_s: float,
        config: CausalLockConfig,
    ) -> Dict[str, bool]:
        return {
            "global_fresh": float(now_s) - float(self.time_s)
            <= float(config.release_global_age_s) + 1e-9,
            "primary_valid": _mode_is_locally_valid(self.primary.best),
            "confirmation_valid": _mode_is_locally_valid(self.confirmation.best),
            "search_agreement": math.isfinite(float(self.search_agreement_m))
            and float(self.search_agreement_m)
            <= float(config.hold_search_agreement_m),
            "forward_rmse": _finite(self.forward_prediction_rmse_mps)
            and float(self.forward_prediction_rmse_mps)
            <= float(config.hold_forward_rmse_mps),
            "primary_mode_clear": _alternative_is_rejected(
                self.primary, config.hold_alternative_delta_chi2
            ),
            "confirmation_mode_clear": _alternative_is_rejected(
                self.confirmation, config.hold_alternative_delta_chi2
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": float(self.time_s),
            "search_agreement_m": float(self.search_agreement_m),
            "stability_from_previous_m": self.stability_from_previous_m,
            "forward_prediction_rmse_mps": self.forward_prediction_rmse_mps,
            "forward_prediction_sample_count": int(
                self.forward_prediction_sample_count
            ),
            "primary": self.primary.to_dict(),
            "confirmation": self.confirmation.to_dict(),
        }


@dataclass
class CausalLockState:
    locked: bool = False
    release_pass_streak: int = 0
    hold_failure_streak: int = 0
    first_lock_time_s: Optional[float] = None
    last_lock_time_s: Optional[float] = None
    last_unlock_time_s: Optional[float] = None
    lock_count: int = 0
    unlock_count: int = 0
    force_global_refresh: bool = False
    last_release_predicate: bool = False
    last_hold_predicate: bool = False
    last_failed_checks: Tuple[str, ...] = ()
    transitions: List[Dict[str, Any]] = field(default_factory=list)


class CausalLockGate:
    """Pure truth-free release/hold state machine."""

    def __init__(self, config: CausalLockConfig) -> None:
        self.config = config
        self.state = CausalLockState()
        self._local_positions: Deque[np.ndarray] = deque(
            maxlen=int(config.release_local_stability_count)
        )

    @property
    def is_locked(self) -> bool:
        return bool(self.state.locked)

    def consume_force_global_refresh(self) -> bool:
        value = bool(self.state.force_global_refresh)
        self.state.force_global_refresh = False
        return value

    def _local_stability(self, position_m: np.ndarray) -> bool:
        self._local_positions.append(np.asarray(position_m, dtype=np.float64).copy())
        if len(self._local_positions) < int(self.config.release_local_stability_count):
            return False
        newest = self._local_positions[-1]
        return bool(
            max(float(np.linalg.norm(value - newest)) for value in self._local_positions)
            <= float(self.config.release_local_stability_m)
        )

    def evaluate(
        self,
        *,
        now_s: float,
        mode: v19.BatchMode,
        initial_position_m: Sequence[float],
        recent_residual_rmse_mps: float,
        global_evidence: Optional[GlobalLockEvidence],
    ) -> bool:
        now = float(now_s)
        position = np.asarray(initial_position_m, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("gate initial position must contain three finite values")
        stable = self._local_stability(position)
        release_checks: Dict[str, bool] = {
            "minimum_time": now + 1e-9 >= float(self.config.minimum_release_time_s),
            "local_valid": _mode_is_locally_valid(mode),
            "local_radius95": math.isfinite(float(mode.local_radius95_m))
            and float(mode.local_radius95_m) <= float(self.config.release_radius95_m),
            "full_rmse": math.isfinite(float(mode.residual_rmse_mps))
            and float(mode.residual_rmse_mps)
            <= float(self.config.release_full_rmse_mps),
            "recent_rmse": math.isfinite(float(recent_residual_rmse_mps))
            and float(recent_residual_rmse_mps)
            <= float(self.config.release_recent_rmse_mps),
            "local_stability": stable,
            "global_available": global_evidence is not None,
        }
        if global_evidence is not None:
            release_checks.update(global_evidence.release_checks(now, self.config))
        release_ok = bool(all(release_checks.values()))
        self.state.last_release_predicate = release_ok

        if not self.state.locked:
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
                    {"time_s": now, "from": "ACQUIRE", "to": "TRACK"}
                )
                self.state.last_failed_checks = ()
            return self.state.locked

        hold_checks: Dict[str, bool] = {
            "local_valid": _mode_is_locally_valid(mode),
            "local_radius95": math.isfinite(float(mode.local_radius95_m))
            and float(mode.local_radius95_m) <= float(self.config.hold_radius95_m),
            "full_rmse": math.isfinite(float(mode.residual_rmse_mps))
            and float(mode.residual_rmse_mps) <= float(self.config.hold_full_rmse_mps),
            "recent_rmse": math.isfinite(float(recent_residual_rmse_mps))
            and float(recent_residual_rmse_mps)
            <= float(self.config.hold_recent_rmse_mps),
            "global_available": global_evidence is not None,
        }
        if global_evidence is not None:
            hold_checks.update(global_evidence.hold_checks(now, self.config))
        hold_ok = bool(all(hold_checks.values()))
        self.state.last_hold_predicate = hold_ok
        self.state.last_failed_checks = tuple(
            name for name, passed in hold_checks.items() if not passed
        )
        self.state.hold_failure_streak = (
            0 if hold_ok else self.state.hold_failure_streak + 1
        )
        if self.state.hold_failure_streak >= int(self.config.loss_consecutive_actions):
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
                    "failed_checks": list(self.state.last_failed_checks),
                }
            )
        return self.state.locked


@dataclass
class CausalEstimatorState:
    initial_position_m: Optional[np.ndarray] = None
    endpoint_position_m: Optional[np.ndarray] = None
    latest_mode: Optional[v19.BatchMode] = None
    latest_global_evidence: Optional[GlobalLockEvidence] = None
    recent_residual_rmse_mps: float = float("inf")
    total_runtime_s: float = 0.0
    global_runtime_s: float = 0.0
    local_runtime_s: float = 0.0
    maximum_update_runtime_s: float = 0.0
    global_solve_count: int = 0
    local_solve_count: int = 0


class CausalStreamingEstimator:
    """V19 full-information estimator plus independent global confirmation."""

    def __init__(
        self,
        *,
        estimator_config: v19.BatchEstimatorConfig,
        lock_config: CausalLockConfig,
        support_radius_min_m: float,
        support_radius_max_m: float,
    ) -> None:
        self.estimator_config = estimator_config
        self.support_radius_min_m = float(support_radius_min_m)
        self.support_radius_max_m = float(support_radius_max_m)
        self.gate = CausalLockGate(lock_config)
        self.state = CausalEstimatorState()
        self.global_records: List[Dict[str, Any]] = []
        self._previous_global_position_m: Optional[np.ndarray] = None
        self._previous_global_measurement_count: Optional[int] = None

    @property
    def has_solution(self) -> bool:
        value = self.state.endpoint_position_m
        return bool(
            value is not None
            and np.asarray(value).shape == (3,)
            and np.all(np.isfinite(value))
        )

    @staticmethod
    def _scheduled_global(now_s: float) -> bool:
        return any(abs(float(now_s) - checkpoint) <= 1e-7 for checkpoint in GLOBAL_REFRESH_S)

    def _forward_residual(
        self,
        history: v19.OnlineDopplerHistory,
    ) -> Tuple[Optional[float], int]:
        if (
            self._previous_global_position_m is None
            or self._previous_global_measurement_count is None
        ):
            return None, 0
        start = int(self._previous_global_measurement_count)
        if start >= history.measurement_count:
            return None, 0
        held_out = history.take(slice(start, history.measurement_count))
        prediction = v19.predict_doppler(
            self._previous_global_position_m,
            held_out,
        )
        residual = held_out.doppler_measured_mps - prediction
        return float(math.sqrt(float(np.mean(residual * residual)))), int(
            held_out.measurement_count
        )

    def _recent_residual(
        self,
        history: v19.OnlineDopplerHistory,
        position_m: np.ndarray,
    ) -> float:
        start = max(0, history.measurement_count - 20)
        recent = history.take(slice(start, history.measurement_count))
        prediction = v19.predict_doppler(position_m, recent)
        residual = recent.doppler_measured_mps - prediction
        return float(math.sqrt(float(np.mean(residual * residual))))

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
            raise ValueError("current dead reckoning must contain three finite values")
        if float(history.t_s[-1]) > now + 1e-9:
            raise RuntimeError("causal history contains a future Doppler measurement")
        if now + 1e-9 < ESTIMATION_START_S:
            return

        center = v19.initial_leader_centroid_from_history(history)
        forced = self.gate.consume_force_global_refresh()
        global_update = bool(
            forced
            or self._scheduled_global(now)
            or self.state.initial_position_m is None
        )
        started = time.perf_counter()
        if global_update:
            primary_seed = 21_001 + int(round(now))
            confirmation_seed = 1_021_001 + int(round(now))
            primary = v19.estimate_initial_position_multistart(
                history,
                center,
                self.support_radius_min_m,
                self.support_radius_max_m,
                self.estimator_config,
                candidate_seed=primary_seed,
            )
            confirmation = v19.estimate_initial_position_multistart(
                history,
                center,
                self.support_radius_min_m,
                self.support_radius_max_m,
                self.estimator_config,
                candidate_seed=confirmation_seed,
            )
            mode = primary.best
            forward_rmse, forward_count = self._forward_residual(history)
            stability = (
                None
                if self._previous_global_position_m is None
                else float(
                    np.linalg.norm(
                        mode.initial_position_m - self._previous_global_position_m
                    )
                )
            )
            agreement = float(
                np.linalg.norm(
                    mode.initial_position_m - confirmation.best.initial_position_m
                )
            )
            evidence = GlobalLockEvidence(
                time_s=now,
                primary=primary,
                confirmation=confirmation,
                search_agreement_m=agreement,
                stability_from_previous_m=stability,
                forward_prediction_rmse_mps=forward_rmse,
                forward_prediction_sample_count=forward_count,
            )
            self.state.latest_global_evidence = evidence
            self._previous_global_position_m = np.asarray(
                mode.initial_position_m, dtype=np.float64
            ).copy()
            self._previous_global_measurement_count = history.measurement_count
            elapsed = float(time.perf_counter() - started)
            self.state.global_runtime_s += elapsed
            self.state.global_solve_count += 2
            self.global_records.append(
                {
                    "time_s": now,
                    "forced": forced,
                    "primary_candidate_seed": primary_seed,
                    "confirmation_candidate_seed": confirmation_seed,
                    "runtime_s": elapsed,
                    **evidence.to_dict(),
                }
            )
        else:
            if self.state.initial_position_m is None:
                raise RuntimeError("local solve requested before global initialization")
            mode = v19.refine_damped_gauss_newton(
                self.state.initial_position_m,
                history,
                center,
                self.support_radius_min_m,
                self.support_radius_max_m,
                self.estimator_config,
            )
            elapsed = float(time.perf_counter() - started)
            self.state.local_runtime_s += elapsed
            self.state.local_solve_count += 1

        self.state.total_runtime_s += elapsed
        self.state.maximum_update_runtime_s = max(
            self.state.maximum_update_runtime_s, elapsed
        )
        self.state.latest_mode = mode
        self.state.initial_position_m = np.asarray(
            mode.initial_position_m, dtype=np.float64
        ).copy()
        self.state.endpoint_position_m = self.state.initial_position_m + current_dr
        self.state.recent_residual_rmse_mps = self._recent_residual(
            history,
            self.state.initial_position_m,
        )
        self.gate.evaluate(
            now_s=now,
            mode=mode,
            initial_position_m=self.state.initial_position_m,
            recent_residual_rmse_mps=self.state.recent_residual_rmse_mps,
            global_evidence=self.state.latest_global_evidence,
        )


@dataclass(frozen=True)
class ArmOutcome:
    summary: Mapping[str, Any]
    trace: Mapping[str, np.ndarray]


def _first_window_time(
    times: np.ndarray,
    mask: np.ndarray,
    window: int,
) -> Optional[float]:
    count = int(window)
    if times.shape != mask.shape:
        raise ValueError("times and mask must have identical shapes")
    if count < 1 or count > mask.size:
        return None
    values = np.convolve(mask.astype(np.int64), np.ones(count, dtype=np.int64), mode="valid")
    candidates = np.flatnonzero(values == count)
    return None if candidates.size == 0 else float(times[int(candidates[0])])


def _first_sustained_time(times: np.ndarray, mask: np.ndarray) -> Optional[float]:
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


def run_causal_lock_arm(
    *,
    cfg: UUV3DConfig,
    tape: ExogenousNoiseTape,
    episode_seed: int,
    episode_index: int,
    estimator_config: v19.BatchEstimatorConfig,
    lock_config: CausalLockConfig,
) -> ArmOutcome:
    """Run one causal V21 arm; truth is kept outside estimator and gate."""

    env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    env.attach_exogenous_noise_tape(tape)
    env.reset(seed=int(episode_seed))
    initial_truth = np.asarray(env.pF, dtype=np.float64).copy()
    initial_leader_centroid = 0.5 * (
        np.asarray(env.pL1, dtype=np.float64)
        + np.asarray(env.pL2, dtype=np.float64)
    )
    recorder = v20.OnlineHistoryRecorder(env)
    estimator = CausalStreamingEstimator(
        estimator_config=estimator_config,
        lock_config=lock_config,
        support_radius_min_m=float(cfg.start_rho_min),
        support_radius_max_m=float(cfg.start_rho_max),
    )
    trace_rows: Dict[str, List[Any]] = {
        "step": [],
        "action_start_time_s": [],
        "time_s": [],
        "phase_track": [],
        "gate_locked_after_update": [],
        "gate_release_predicate": [],
        "gate_release_pass_streak": [],
        "gate_hold_failure_streak": [],
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
        "batch_recent_residual_rmse_mps": [],
    }
    actions: List[np.ndarray] = []
    try:
        for _ in range(int(cfg.max_steps)):
            action_start = float(env.t)
            track_for_action = bool(estimator.gate.is_locked and estimator.has_solution)
            if track_for_action:
                action = v20.pid_action_for_position(
                    env,
                    np.asarray(estimator.state.endpoint_position_m, dtype=np.float64),
                )
            else:
                action = v20.position_independent_acquisition_action(env, action_start)
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
            formation_error = float(np.linalg.norm(truth - np.asarray(desired)))
            localization_error = float(np.linalg.norm(endpoint - truth))
            mode = estimator.state.latest_mode
            radius95 = float("nan") if mode is None else float(mode.local_radius95_m)
            recent_rmse = float(estimator.state.recent_residual_rmse_mps)
            gate_state = estimator.gate.state
            trace_rows["step"].append(int(env.step_count))
            trace_rows["action_start_time_s"].append(action_start)
            trace_rows["time_s"].append(float(env.t))
            trace_rows["phase_track"].append(float(track_for_action))
            trace_rows["gate_locked_after_update"].append(float(estimator.gate.is_locked))
            trace_rows["gate_release_predicate"].append(
                float(gate_state.last_release_predicate)
            )
            trace_rows["gate_release_pass_streak"].append(
                int(gate_state.release_pass_streak)
            )
            trace_rows["gate_hold_failure_streak"].append(
                int(gate_state.hold_failure_streak)
            )
            trace_rows["action_speed"].append(float(action[0]))
            trace_rows["action_yaw"].append(float(action[1]))
            trace_rows["action_pitch"].append(float(action[2]))
            trace_rows["truth_x"].append(float(truth[0]))
            trace_rows["truth_y"].append(float(truth[1]))
            trace_rows["truth_z"].append(float(truth[2]))
            trace_rows["estimate_x"].append(float(endpoint[0]))
            trace_rows["estimate_y"].append(float(endpoint[1]))
            trace_rows["estimate_z"].append(float(endpoint[2]))
            trace_rows["formation_error_truth_m"].append(formation_error)
            trace_rows["localization_error_m"].append(localization_error)
            trace_rows["batch_local_radius95_m"].append(radius95)
            trace_rows["batch_recent_residual_rmse_mps"].append(recent_rmse)
            if terminated or truncated:
                if int(env.step_count) != int(cfg.max_steps):
                    raise RuntimeError("V21 arm ended before the fixed horizon")
                break
        cursor = env._v11_noise_cursor
        if cursor is None:
            raise RuntimeError("V21 environment lost its attached noise tape")
        noise_cursor = dict(cursor.state_dict())
    finally:
        recorder.restore()
        env.close()

    trace = {name: np.asarray(values) for name, values in trace_rows.items()}
    expected_shape = (int(cfg.max_steps),)
    if trace["time_s"].shape != expected_shape:
        raise RuntimeError("V21 trace does not contain the fixed number of actions")
    times = np.asarray(trace["time_s"], dtype=np.float64)
    action_times = np.asarray(trace["action_start_time_s"], dtype=np.float64)
    formation = np.asarray(trace["formation_error_truth_m"], dtype=np.float64)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    phase_track = np.asarray(trace["phase_track"], dtype=bool)
    finite_localization = np.isfinite(localization)
    joint = (formation < TERMINAL_FORMATION_GATE_M) & (
        localization < TERMINAL_LOCALIZATION_GATE_M
    )
    truth_ready = finite_localization & (
        localization < TERMINAL_LOCALIZATION_GATE_M
    )
    truth_ready_time = _first_window_time(
        times,
        truth_ready,
        TRUTH_READY_DWELL_ACTIONS,
    )
    track_indices = np.flatnonzero(phase_track)
    first_track_time = (
        None if track_indices.size == 0 else float(action_times[int(track_indices[0])])
    )
    first_track_error = (
        None if track_indices.size == 0 else float(localization[int(track_indices[0])])
    )
    false_locked_action = phase_track & (
        (~finite_localization) | (localization >= TERMINAL_LOCALIZATION_GATE_M)
    )
    tail = joint[-TAIL_WINDOW_ACTIONS:]
    dwell = joint[-DWELL_ACTIONS:]
    action_array = np.asarray(actions, dtype=np.float64)
    finite_after_estimation = (times >= ESTIMATION_START_S) & finite_localization
    lock_delay = (
        None
        if first_track_time is None or truth_ready_time is None
        else float(first_track_time - truth_ready_time)
    )
    state = estimator.state
    gate_state = estimator.gate.state
    summary: Dict[str, Any] = {
        "version": VERSION,
        "arm": "causal_batch_pid",
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
        "time_to_sustained_joint_lock_s": _first_sustained_time(times, joint),
        "mean_squared_action": float(
            np.mean(np.sum(action_array * action_array, axis=1))
        ),
        "mean_localization_error_after_30_m": _json_float(
            float(np.nanmean(localization[finite_after_estimation]))
        ),
        "gate": {
            "first_track_action_time_s": first_track_time,
            "first_track_localization_error_m": first_track_error,
            "truth_ready_time_s": truth_ready_time,
            "lock_delay_from_truth_ready_s": lock_delay,
            "ever_locked": bool(track_indices.size > 0),
            "false_lock_episode": bool(np.any(false_locked_action)),
            "false_locked_action_count": int(np.sum(false_locked_action)),
            "locked_action_count": int(np.sum(phase_track)),
            "lock_count": int(gate_state.lock_count),
            "unlock_count": int(gate_state.unlock_count),
            "transitions": list(gate_state.transitions),
        },
        "batch": {
            "measurement_count": int(recorder.measurement_count),
            "global_solve_count": int(state.global_solve_count),
            "local_solve_count": int(state.local_solve_count),
            "total_runtime_s": float(state.total_runtime_s),
            "global_runtime_s": float(state.global_runtime_s),
            "local_runtime_s": float(state.local_runtime_s),
            "maximum_update_runtime_s": float(state.maximum_update_runtime_s),
            "global_records": list(estimator.global_records),
        },
    }
    return ArmOutcome(summary=summary, trace=trace)


__all__ = [
    "CausalLockConfig",
    "CausalLockGate",
    "CausalStreamingEstimator",
    "GlobalLockEvidence",
    "GLOBAL_REFRESH_S",
    "VERSION",
    "run_causal_lock_arm",
]
