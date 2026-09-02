#!/usr/bin/env python3
"""Delay-aware formation tracker using deployable online signals only.

The controller in this module is intentionally separate from the restored
post-lock controller in :mod:`baseline_controllers_v11`.  It is a new
experimental tracker for a plant with a known command delay and first-order
lag in surge acceleration, yaw rate, and pitch rate.

The design combines four standard control mechanisms:

* a Smith-style predictor propagates the measured follower motion through the
  commands that are already waiting in the delay queue;
* an outer formation loop constructs a leader-feed-forward velocity reference
  from the *predicted* position error;
* a critically damped, model-based inner loop commands the lagged rates;
* reference and command governors, plus a bumpless ACQUIRE--TRACK transfer,
  prevent the repeated saturation seen with the memoryless baseline.

No simulator truth, particle-filter state, reward, or future noise is accepted
by the public interface.  ``TrackingObservation`` contains only a causal
position estimate, the broadcast-derived desired formation point and leader
velocities, and the onboard dead-reckoning velocity representation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Sequence, Tuple

import numpy as np


Array3 = np.ndarray


def _finite_vec3(value: Sequence[float], name: str) -> Array3:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite values")
    return array.copy()


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _wrap360(angle_deg: float) -> float:
    return float(angle_deg) % 360.0


def _smoothstep01(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _speed_yaw_pitch_to_velocity(state: Sequence[float]) -> Array3:
    speed, yaw_deg, pitch_deg = _finite_vec3(state, "speed/yaw/pitch")
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    horizontal = float(speed) * math.cos(pitch)
    return np.asarray(
        [
            horizontal * math.cos(yaw),
            horizontal * math.sin(yaw),
            float(speed) * math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _velocity_to_speed_yaw_pitch(
    velocity_mps: Sequence[float],
    *,
    fallback_yaw_deg: float,
) -> Array3:
    velocity = _finite_vec3(velocity_mps, "velocity_mps")
    speed = float(np.linalg.norm(velocity))
    horizontal = float(np.linalg.norm(velocity[:2]))
    yaw = (
        float(fallback_yaw_deg)
        if horizontal <= 1e-12
        else math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
    )
    pitch = math.degrees(math.atan2(float(velocity[2]), max(horizontal, 1e-12)))
    return np.asarray([speed, _wrap360(yaw), pitch], dtype=np.float64)


def _limit_vector_delta(previous: Array3, target: Array3, maximum: float) -> Array3:
    difference = np.asarray(target, dtype=np.float64) - np.asarray(
        previous, dtype=np.float64
    )
    norm = float(np.linalg.norm(difference))
    if norm <= float(maximum) or norm <= 1e-15:
        return np.asarray(target, dtype=np.float64).copy()
    return np.asarray(previous, dtype=np.float64) + difference * (
        float(maximum) / norm
    )


@dataclass(frozen=True)
class TrackingObservation:
    """Causal inputs available when one action is selected.

    ``measured_speed_mps``, ``measured_yaw_deg``, and
    ``measured_pitch_deg`` use the same bottom-track dead-reckoning convention
    as the frozen experiment.  The two leader velocities are broadcasts; their
    mean advances the desired formation point across the predictor horizon.
    """

    time_s: float
    estimated_position_m: Sequence[float]
    desired_position_m: Sequence[float]
    leader1_velocity_mps: Sequence[float]
    leader2_velocity_mps: Sequence[float]
    measured_speed_mps: float
    measured_yaw_deg: float
    measured_pitch_deg: float

    def validated(self) -> "ValidatedTrackingObservation":
        scalars = np.asarray(
            [
                self.time_s,
                self.measured_speed_mps,
                self.measured_yaw_deg,
                self.measured_pitch_deg,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("tracking observation contains a non-finite scalar")
        if float(self.measured_speed_mps) < 0.0:
            raise ValueError("measured speed must be non-negative")
        return ValidatedTrackingObservation(
            time_s=float(self.time_s),
            estimated_position_m=_finite_vec3(
                self.estimated_position_m, "estimated_position_m"
            ),
            desired_position_m=_finite_vec3(
                self.desired_position_m, "desired_position_m"
            ),
            leader1_velocity_mps=_finite_vec3(
                self.leader1_velocity_mps, "leader1_velocity_mps"
            ),
            leader2_velocity_mps=_finite_vec3(
                self.leader2_velocity_mps, "leader2_velocity_mps"
            ),
            measured_state=np.asarray(
                [
                    float(self.measured_speed_mps),
                    _wrap360(float(self.measured_yaw_deg)),
                    float(self.measured_pitch_deg),
                ],
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True)
class ValidatedTrackingObservation:
    time_s: float
    estimated_position_m: Array3
    desired_position_m: Array3
    leader1_velocity_mps: Array3
    leader2_velocity_mps: Array3
    measured_state: Array3


@dataclass(frozen=True)
class DelayAwareTrackerConfig:
    """Frozen controller and known actuator-model parameters.

    The natural frequencies are conservative analytical choices: each is
    below the reciprocal of the two-second command delay, while damping is
    critical.  They are exposed explicitly so a campaign can preregister a
    different design without hiding gains in runner code.
    """

    action_dt_s: float = 2.0
    predictor_substeps: int = 20
    command_delay_actions: int = 1
    surge_acceleration_time_constant_s: float = 5.0
    yaw_rate_time_constant_s: float = 2.0
    pitch_rate_time_constant_s: float = 3.0

    speed_increment_per_action_mps: float = 0.4
    yaw_increment_per_action_deg: float = 20.0
    pitch_increment_per_action_deg: float = 14.0
    maximum_yaw_rate_deg_s: float = 60.0
    maximum_pitch_rate_deg_s: float = 45.0
    minimum_speed_mps: float = 0.2
    maximum_speed_mps: float = 5.0
    minimum_pitch_deg: float = -45.0
    maximum_pitch_deg: float = 45.0

    # Preserve the outer-loop correction authority of the frozen reference
    # controller.  The reference independently limits its along-track and
    # cross-track corrections to 1.65 m/s, so its largest feasible horizontal
    # correction has Euclidean norm sqrt(2) * 1.65 m/s.  The delay-aware
    # intervention changes command execution, not the formation-error gains.
    along_gain_per_s: float = 0.055
    lateral_gain_per_s: float = 0.065
    vertical_gain_per_s: float = 0.055
    maximum_horizontal_correction_mps: float = 2.3334523779156067
    maximum_vertical_correction_mps: float = 1.20
    maximum_reference_acceleration_mps2: float = 0.25

    speed_natural_frequency_rad_s: float = 0.18
    yaw_natural_frequency_rad_s: float = 0.26
    pitch_natural_frequency_rad_s: float = 0.22
    damping_ratio: float = 1.0
    measured_rate_blend: float = 0.30

    bumpless_transfer_actions: int = 6
    maximum_action_delta: Tuple[float, float, float] = (0.35, 0.30, 0.30)

    def __post_init__(self) -> None:
        positive = {
            "action_dt_s": self.action_dt_s,
            "predictor_substeps": self.predictor_substeps,
            "speed_increment_per_action_mps": self.speed_increment_per_action_mps,
            "yaw_increment_per_action_deg": self.yaw_increment_per_action_deg,
            "pitch_increment_per_action_deg": self.pitch_increment_per_action_deg,
            "maximum_yaw_rate_deg_s": self.maximum_yaw_rate_deg_s,
            "maximum_pitch_rate_deg_s": self.maximum_pitch_rate_deg_s,
            "maximum_speed_mps": self.maximum_speed_mps,
            "maximum_reference_acceleration_mps2": self.maximum_reference_acceleration_mps2,
            "speed_natural_frequency_rad_s": self.speed_natural_frequency_rad_s,
            "yaw_natural_frequency_rad_s": self.yaw_natural_frequency_rad_s,
            "pitch_natural_frequency_rad_s": self.pitch_natural_frequency_rad_s,
            "damping_ratio": self.damping_ratio,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = {
            "command_delay_actions": self.command_delay_actions,
            "surge_acceleration_time_constant_s": self.surge_acceleration_time_constant_s,
            "yaw_rate_time_constant_s": self.yaw_rate_time_constant_s,
            "pitch_rate_time_constant_s": self.pitch_rate_time_constant_s,
            "bumpless_transfer_actions": self.bumpless_transfer_actions,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.predictor_substeps) != self.predictor_substeps:
            raise ValueError("predictor_substeps must be an integer")
        if int(self.command_delay_actions) != self.command_delay_actions:
            raise ValueError("command_delay_actions must be an integer")
        if int(self.bumpless_transfer_actions) != self.bumpless_transfer_actions:
            raise ValueError("bumpless_transfer_actions must be an integer")
        if not 0.0 <= float(self.measured_rate_blend) <= 1.0:
            raise ValueError("measured_rate_blend must lie in [0, 1]")
        if float(self.minimum_speed_mps) < 0.0 or not (
            float(self.maximum_speed_mps) > float(self.minimum_speed_mps)
        ):
            raise ValueError("invalid speed bounds")
        if not float(self.maximum_pitch_deg) > float(self.minimum_pitch_deg):
            raise ValueError("invalid pitch bounds")
        action_delta = np.asarray(self.maximum_action_delta, dtype=np.float64)
        if action_delta.shape != (3,) or not np.all(np.isfinite(action_delta)):
            raise ValueError("maximum_action_delta must contain three finite values")
        if np.any(action_delta <= 0.0) or np.any(action_delta > 2.0):
            raise ValueError("maximum_action_delta entries must lie in (0, 2]")

    @classmethod
    def from_simulator_config(
        cls,
        cfg: Any,
        *,
        command_delay_actions: int,
        surge_acceleration_time_constant_s: float,
        yaw_rate_time_constant_s: float,
        pitch_rate_time_constant_s: float,
        **controller_overrides: Any,
    ) -> "DelayAwareTrackerConfig":
        """Copy only public numeric limits from a simulator config object."""

        values = {
            "action_dt_s": float(cfg.action_dt),
            "command_delay_actions": int(command_delay_actions),
            "surge_acceleration_time_constant_s": float(
                surge_acceleration_time_constant_s
            ),
            "yaw_rate_time_constant_s": float(yaw_rate_time_constant_s),
            "pitch_rate_time_constant_s": float(pitch_rate_time_constant_s),
            "speed_increment_per_action_mps": float(
                cfg.rl_speed_delta_per_step
            ),
            "yaw_increment_per_action_deg": float(cfg.rl_yaw_per_step_deg),
            "pitch_increment_per_action_deg": float(cfg.rl_pitch_per_step_deg),
            "maximum_yaw_rate_deg_s": float(cfg.max_yaw_rate_deg_s),
            "maximum_pitch_rate_deg_s": float(cfg.max_pitch_rate_deg_s),
            "minimum_speed_mps": float(cfg.f_min_speed),
            "maximum_speed_mps": float(cfg.f_max_speed),
            "minimum_pitch_deg": float(cfg.pitch_min_deg),
            "maximum_pitch_deg": float(cfg.pitch_max_deg),
        }
        values.update(controller_overrides)
        return cls(**values)

    @property
    def normalized_rate_scales(self) -> Array3:
        return np.asarray(
            [
                self.speed_increment_per_action_mps / self.action_dt_s,
                min(
                    self.maximum_yaw_rate_deg_s,
                    self.yaw_increment_per_action_deg / self.action_dt_s,
                ),
                min(
                    self.maximum_pitch_rate_deg_s,
                    self.pitch_increment_per_action_deg / self.action_dt_s,
                ),
            ],
            dtype=np.float64,
        )

    @property
    def time_constants_s(self) -> Array3:
        return np.asarray(
            [
                self.surge_acceleration_time_constant_s,
                self.yaw_rate_time_constant_s,
                self.pitch_rate_time_constant_s,
            ],
            dtype=np.float64,
        )

    @property
    def natural_frequencies_rad_s(self) -> Array3:
        return np.asarray(
            [
                self.speed_natural_frequency_rad_s,
                self.yaw_natural_frequency_rad_s,
                self.pitch_natural_frequency_rad_s,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class TrackerDiagnostics:
    predictor_horizon_s: float
    predicted_position_m: Array3
    predicted_state: Array3
    predicted_rate_state: Array3
    predicted_formation_error_m: Array3
    unconstrained_reference_velocity_mps: Array3
    governed_reference_velocity_mps: Array3
    raw_action: Array3
    blended_action: Array3
    returned_action: Array3
    bumpless_blend: float
    command_delta_limited: Tuple[bool, bool, bool]


class DelayAwareFormationTracker:
    """Stateful Smith-predictor/cascade tracker.

    Call :meth:`action` once per action interval only while TRACK is active.
    On the first call, pass the last ACQUIRE command as ``previous_action``;
    this seeds both the plant-delay queue and the bumpless-transfer baseline.
    Call :meth:`reset` after loss of lock or at the start of a new episode.
    """

    def __init__(self, config: Optional[DelayAwareTrackerConfig] = None) -> None:
        self.config = config or DelayAwareTrackerConfig()
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._last_time_s: Optional[float] = None
        self._last_measured_state: Optional[Array3] = None
        self._rate_state = np.zeros(3, dtype=np.float64)
        self._pending_actions: Deque[Array3] = deque()
        self._last_interval_delivered_action = np.zeros(3, dtype=np.float64)
        self._last_action = np.zeros(3, dtype=np.float64)
        self._reference_velocity: Optional[Array3] = None
        self._reference_state: Optional[Array3] = None
        self._track_action_count = 0
        self.last_diagnostics: Optional[TrackerDiagnostics] = None

    @property
    def initialized(self) -> bool:
        """Whether the controller currently owns an active TRACK segment."""

        return bool(self._initialized)

    @property
    def track_action_count(self) -> int:
        """Number of commands issued since the last ACQUIRE--TRACK transfer."""

        return int(self._track_action_count)

    def _initialize(
        self,
        observation: ValidatedTrackingObservation,
        previous_action: Sequence[float],
    ) -> None:
        seed_action = np.clip(
            _finite_vec3(previous_action, "previous_action"), -1.0, 1.0
        )
        delay = int(self.config.command_delay_actions)
        self._pending_actions = deque(seed_action.copy() for _ in range(delay))
        self._last_interval_delivered_action = seed_action.copy()
        self._last_action = seed_action.copy()
        self._last_time_s = float(observation.time_s)
        self._last_measured_state = observation.measured_state.copy()
        self._rate_state = np.zeros(3, dtype=np.float64)
        self._reference_velocity = _speed_yaw_pitch_to_velocity(
            observation.measured_state
        )
        self._reference_state = observation.measured_state.copy()
        self._track_action_count = 0
        self._initialized = True

    def _requested_rates(self, action: Array3) -> Array3:
        return np.asarray(action, dtype=np.float64) * self.config.normalized_rate_scales

    def _advance_state_exact(
        self,
        state: Array3,
        rates: Array3,
        action: Array3,
        duration_s: float,
    ) -> Tuple[Array3, Array3]:
        requested = self._requested_rates(action)
        tau = self.config.time_constants_s
        duration = float(duration_s)
        updated_state = np.asarray(state, dtype=np.float64).copy()
        updated_rates = np.asarray(rates, dtype=np.float64).copy()
        for index in range(3):
            if tau[index] <= 0.0:
                state_increment = requested[index] * duration
                rate_end = requested[index]
            else:
                decay = math.exp(-duration / float(tau[index]))
                state_increment = (
                    requested[index] * duration
                    + (rates[index] - requested[index])
                    * tau[index]
                    * (1.0 - decay)
                )
                rate_end = requested[index] + (
                    rates[index] - requested[index]
                ) * decay
            updated_state[index] += state_increment
            updated_rates[index] = rate_end
        updated_state[0] = float(
            np.clip(
                updated_state[0],
                self.config.minimum_speed_mps,
                self.config.maximum_speed_mps,
            )
        )
        updated_state[1] = _wrap360(float(updated_state[1]))
        updated_state[2] = float(
            np.clip(
                updated_state[2],
                self.config.minimum_pitch_deg,
                self.config.maximum_pitch_deg,
            )
        )
        return updated_state, updated_rates

    def _observe_rate_state(
        self, observation: ValidatedTrackingObservation
    ) -> None:
        assert self._last_time_s is not None
        assert self._last_measured_state is not None
        elapsed = float(observation.time_s - self._last_time_s)
        if elapsed <= 0.0:
            raise ValueError("tracking observation times must increase strictly")
        _, model_rates = self._advance_state_exact(
            self._last_measured_state,
            self._rate_state,
            self._last_interval_delivered_action,
            elapsed,
        )
        measured_rates = np.asarray(
            [
                (observation.measured_state[0] - self._last_measured_state[0])
                / elapsed,
                _wrap180(
                    observation.measured_state[1]
                    - self._last_measured_state[1]
                )
                / elapsed,
                (observation.measured_state[2] - self._last_measured_state[2])
                / elapsed,
            ],
            dtype=np.float64,
        )
        # Dead-reckoning angle noise can make a two-point derivative much
        # larger than any physical response.  The clipping limits are based on
        # the known command channels, not on simulator truth.
        derivative_limit = 2.0 * self.config.normalized_rate_scales
        measured_rates = np.clip(
            measured_rates, -derivative_limit, derivative_limit
        )
        blend = float(self.config.measured_rate_blend)
        self._rate_state = (1.0 - blend) * model_rates + blend * measured_rates
        self._last_time_s = float(observation.time_s)
        self._last_measured_state = observation.measured_state.copy()

    def _predict_pending_execution(
        self, observation: ValidatedTrackingObservation
    ) -> Tuple[Array3, Array3, Array3]:
        state = observation.measured_state.copy()
        rates = self._rate_state.copy()
        position = observation.estimated_position_m.copy()
        substeps = int(self.config.predictor_substeps)
        dt = float(self.config.action_dt_s) / float(substeps)
        for pending_action in self._pending_actions:
            for _ in range(substeps):
                velocity_before = _speed_yaw_pitch_to_velocity(state)
                state, rates = self._advance_state_exact(
                    state, rates, pending_action, dt
                )
                velocity_after = _speed_yaw_pitch_to_velocity(state)
                position += 0.5 * (velocity_before + velocity_after) * dt
        return position, state, rates

    def _formation_velocity_reference(
        self,
        observation: ValidatedTrackingObservation,
        predicted_position_m: Array3,
    ) -> Tuple[Array3, Array3, Array3]:
        cfg = self.config
        leader_velocity = 0.5 * (
            observation.leader1_velocity_mps
            + observation.leader2_velocity_mps
        )
        horizon = float(cfg.command_delay_actions) * float(cfg.action_dt_s)
        future_desired = observation.desired_position_m + horizon * leader_velocity
        error = future_desired - predicted_position_m

        horizontal_leader = leader_velocity[:2]
        horizontal_norm = float(np.linalg.norm(horizontal_leader))
        if horizontal_norm > 1e-9:
            forward = horizontal_leader / horizontal_norm
        else:
            predicted_velocity = _speed_yaw_pitch_to_velocity(
                np.asarray(
                    [
                        max(float(self.config.minimum_speed_mps), 1e-9),
                        float(observation.measured_state[1]),
                        0.0,
                    ],
                    dtype=np.float64,
                )
            )[:2]
            forward = predicted_velocity / max(
                float(np.linalg.norm(predicted_velocity)), 1e-9
            )
        side = np.asarray([forward[1], -forward[0]], dtype=np.float64)
        along = float(np.dot(error[:2], forward))
        lateral = float(np.dot(error[:2], side))
        horizontal_correction = (
            cfg.along_gain_per_s * along * forward
            + cfg.lateral_gain_per_s * lateral * side
        )
        correction_norm = float(np.linalg.norm(horizontal_correction))
        if correction_norm > float(cfg.maximum_horizontal_correction_mps):
            horizontal_correction *= (
                float(cfg.maximum_horizontal_correction_mps) / correction_norm
            )
        vertical_correction = float(
            np.clip(
                cfg.vertical_gain_per_s * float(error[2]),
                -cfg.maximum_vertical_correction_mps,
                cfg.maximum_vertical_correction_mps,
            )
        )
        target = leader_velocity.copy()
        target[:2] += horizontal_correction
        target[2] += vertical_correction
        return target, error, leader_velocity

    def _govern_reference(self, target_velocity: Array3) -> Array3:
        assert self._reference_velocity is not None
        maximum_delta = (
            float(self.config.maximum_reference_acceleration_mps2)
            * float(self.config.action_dt_s)
        )
        governed = _limit_vector_delta(
            self._reference_velocity, target_velocity, maximum_delta
        )
        speed = float(np.linalg.norm(governed))
        if speed > float(self.config.maximum_speed_mps):
            governed *= float(self.config.maximum_speed_mps) / speed
        elif speed < float(self.config.minimum_speed_mps):
            if speed > 1e-12:
                governed *= float(self.config.minimum_speed_mps) / speed
            else:
                yaw = math.radians(
                    float(
                        self._last_measured_state[1]
                        if self._last_measured_state is not None
                        else 0.0
                    )
                )
                governed = np.asarray(
                    [
                        self.config.minimum_speed_mps * math.cos(yaw),
                        self.config.minimum_speed_mps * math.sin(yaw),
                        0.0,
                    ],
                    dtype=np.float64,
                )
        self._reference_velocity = governed.copy()
        return governed

    def _model_based_action(
        self,
        predicted_state: Array3,
        predicted_rates: Array3,
        reference_velocity: Array3,
    ) -> Array3:
        cfg = self.config
        reference_state = _velocity_to_speed_yaw_pitch(
            reference_velocity,
            fallback_yaw_deg=float(predicted_state[1]),
        )
        reference_state[0] = float(
            np.clip(
                reference_state[0], cfg.minimum_speed_mps, cfg.maximum_speed_mps
            )
        )
        reference_state[2] = float(
            np.clip(
                reference_state[2], cfg.minimum_pitch_deg, cfg.maximum_pitch_deg
            )
        )
        if self._reference_state is None:
            reference_rates = np.zeros(3, dtype=np.float64)
        else:
            reference_rates = np.asarray(
                [
                    (reference_state[0] - self._reference_state[0])
                    / cfg.action_dt_s,
                    _wrap180(reference_state[1] - self._reference_state[1])
                    / cfg.action_dt_s,
                    (reference_state[2] - self._reference_state[2])
                    / cfg.action_dt_s,
                ],
                dtype=np.float64,
            )
        self._reference_state = reference_state.copy()

        error = reference_state - predicted_state
        error[1] = _wrap180(float(error[1]))
        tau = cfg.time_constants_s
        omega = cfg.natural_frequencies_rad_s
        zeta = float(cfg.damping_ratio)
        requested_rates = np.empty(3, dtype=np.float64)
        for index in range(3):
            if tau[index] <= 0.0:
                requested_rates[index] = reference_rates[index] + omega[index] * error[index]
            else:
                # Pole placement for q_dot=r, tau*r_dot=u-r, with a
                # piecewise-linear reference over one action interval.
                requested_rates[index] = (
                    tau[index] * omega[index] ** 2 * error[index]
                    + (1.0 - 2.0 * zeta * omega[index] * tau[index])
                    * predicted_rates[index]
                    + 2.0
                    * zeta
                    * omega[index]
                    * tau[index]
                    * reference_rates[index]
                )
        return np.clip(
            requested_rates / cfg.normalized_rate_scales, -1.0, 1.0
        )

    def _bumpless_and_slew_limited(self, raw_action: Array3) -> Tuple[Array3, Array3, float, Tuple[bool, bool, bool]]:
        transfer_actions = int(self.config.bumpless_transfer_actions)
        if transfer_actions <= 1:
            blend = 1.0
        else:
            blend = _smoothstep01(
                float(self._track_action_count) / float(transfer_actions - 1)
            )
        blended = self._last_action + blend * (raw_action - self._last_action)
        maximum_delta = np.asarray(
            self.config.maximum_action_delta, dtype=np.float64
        )
        delta = blended - self._last_action
        limited_delta = np.clip(delta, -maximum_delta, maximum_delta)
        limited_flags = tuple(
            bool(abs(delta[index] - limited_delta[index]) > 1e-12)
            for index in range(3)
        )
        returned = np.clip(self._last_action + limited_delta, -1.0, 1.0)
        return blended, returned, float(blend), limited_flags

    def action(
        self,
        observation: TrackingObservation,
        *,
        previous_action: Optional[Sequence[float]] = None,
    ) -> Array3:
        """Return one normalized command and update the causal controller state.

        ``previous_action`` is required on the first call after :meth:`reset`
        and rejected later.  Requiring it prevents an implicit zero command at
        the ACQUIRE--TRACK boundary.
        """

        obs = observation.validated()
        if not self._initialized:
            if previous_action is None:
                raise ValueError(
                    "previous_action is required on the first TRACK action"
                )
            self._initialize(obs, previous_action)
        else:
            if previous_action is not None:
                raise ValueError(
                    "previous_action may be supplied only on the first TRACK action"
                )
            self._observe_rate_state(obs)

        predicted_position, predicted_state, predicted_rates = (
            self._predict_pending_execution(obs)
        )
        target_velocity, predicted_error, _ = self._formation_velocity_reference(
            obs, predicted_position
        )
        governed_reference = self._govern_reference(target_velocity)
        raw_action = self._model_based_action(
            predicted_state, predicted_rates, governed_reference
        )
        blended, returned, transfer_blend, limited = (
            self._bumpless_and_slew_limited(raw_action)
        )

        if self._pending_actions:
            delivered_next = self._pending_actions.popleft()
            self._pending_actions.append(returned.copy())
        else:
            delivered_next = returned.copy()
        self._last_interval_delivered_action = delivered_next.copy()
        self._last_action = returned.copy()
        self._track_action_count += 1
        self.last_diagnostics = TrackerDiagnostics(
            predictor_horizon_s=(
                float(self.config.command_delay_actions)
                * float(self.config.action_dt_s)
            ),
            predicted_position_m=predicted_position.copy(),
            predicted_state=predicted_state.copy(),
            predicted_rate_state=predicted_rates.copy(),
            predicted_formation_error_m=predicted_error.copy(),
            unconstrained_reference_velocity_mps=target_velocity.copy(),
            governed_reference_velocity_mps=governed_reference.copy(),
            raw_action=raw_action.copy(),
            blended_action=blended.copy(),
            returned_action=returned.copy(),
            bumpless_blend=float(transfer_blend),
            command_delta_limited=limited,
        )
        return returned.astype(np.float32)


__all__ = [
    "DelayAwareFormationTracker",
    "DelayAwareTrackerConfig",
    "TrackerDiagnostics",
    "TrackingObservation",
]
