#!/usr/bin/env python3
"""Paired post-lock controller repair experiment.

This module leaves the causal full-history estimator, active acquisition
planner, audited ACQUIRE--TRACK gate, sensor-noise tape, and V40 low-order
plant unchanged.  The sole intervention is the controller called while the
audited gate is in TRACK:

* ``baseline_pid`` is the frozen memoryless post-lock controller;
* ``delay_aware`` is the causal stateful tracker in
  :mod:`delay_aware_formation_tracker`.

Both controller arms are crossed with no current and the same
bottom-track-visible current used in V40.  Simulator truth remains available
only to the inherited scorer.  The experiment deliberately uses a new
qualification range and continues to reject the sealed final range.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from delay_aware_formation_tracker import (
    DelayAwareFormationTracker,
    DelayAwareTrackerConfig,
    TrackerDiagnostics,
    TrackingObservation,
)
import uuv_v11_online as v11
from uuv_v18_resampling_guard import UUV3DConfig
import uuv_v20_positioning_ablation as v20
import uuv_v38_leader_source_ablation as v38
import uuv_v40_dynamic_plant_stress as v40


VERSION = "v41_controller_repair_1.1"

POLICY = v38.POLICY_ACTIVE
DYNAMICS = v40.DYNAMICS_LOW_ORDER

BASELINE_PID = "baseline_pid"
DELAY_AWARE = "delay_aware"
CONTROLLERS: Tuple[str, ...] = (BASELINE_PID, DELAY_AWARE)

CURRENT_NONE = v40.CURRENT_NONE
CURRENT_VISIBLE = v40.CURRENT_VISIBLE
CURRENTS: Tuple[str, ...] = (CURRENT_NONE, CURRENT_VISIBLE)

SMOKE_SEEDS = (49_566, 49_591)
FINAL_START = 50_000
FINAL_END = 50_999
QUALIFICATION_START = 51_000
QUALIFICATION_END = 51_099


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the paired 2-controller by 2-current experiment."""

    controller: str
    current: str

    def __post_init__(self) -> None:
        if self.controller not in CONTROLLERS:
            raise ValueError(f"unknown V41 controller {self.controller!r}")
        if self.current not in CURRENTS:
            raise ValueError(f"unknown V41 current {self.current!r}")

    @property
    def policy(self) -> str:
        return POLICY

    @property
    def dynamics(self) -> str:
        return DYNAMICS

    @property
    def name(self) -> str:
        return (
            f"{self.policy}__{self.dynamics}__"
            f"{self.controller}__{self.current}"
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "policy": self.policy,
            "dynamics": self.dynamics,
            "controller": self.controller,
            "current": self.current,
        }


ARM_SPECS: Tuple[ArmSpec, ...] = tuple(
    ArmSpec(controller=controller, current=current)
    for controller in CONTROLLERS
    for current in CURRENTS
)
ARM_BY_NAME: Mapping[str, ArmSpec] = {arm.name: arm for arm in ARM_SPECS}


def arm_specs() -> Tuple[ArmSpec, ...]:
    return ARM_SPECS


def assert_seed_allowed(seed: int, *, smoke: bool = False) -> None:
    """Enforce the preregistered V41 seed boundary."""

    value = int(seed)
    if FINAL_START <= value <= FINAL_END:
        raise PermissionError(f"sealed final seed {value} remains unavailable")
    if smoke:
        if value not in SMOKE_SEEDS:
            raise PermissionError(
                "smoke is restricted to the two previously opened seeds"
            )
        return
    if not QUALIFICATION_START <= value <= QUALIFICATION_END:
        raise PermissionError(
            "V41 qualification uses exactly seeds 51000..51099"
        )


def _assert_seed_allowed_automatic(seed: int) -> None:
    value = int(seed)
    assert_seed_allowed(value, smoke=value in SMOKE_SEEDS)


def _leader_velocity_from_broadcast(
    speed_mps: float,
    yaw_deg: float,
) -> np.ndarray:
    """Build a leader velocity from broadcast speed and heading only."""

    return np.asarray(
        v11._v8.vel_from_speed_yaw_pitch(
            float(speed_mps),
            float(yaw_deg),
            0.0,
        ),
        dtype=np.float64,
    )


def delay_aware_config_for_environment(
    cfg: UUV3DConfig,
) -> DelayAwareTrackerConfig:
    """Return the frozen controller config from public plant constants."""

    plant = v40.PLANT_PARAMETERS[DYNAMICS]
    return DelayAwareTrackerConfig.from_simulator_config(
        cfg,
        command_delay_actions=int(plant.command_delay_actions),
        surge_acceleration_time_constant_s=float(
            plant.surge_acceleration_time_constant_s
        ),
        yaw_rate_time_constant_s=float(plant.yaw_rate_time_constant_s),
        pitch_rate_time_constant_s=float(plant.pitch_rate_time_constant_s),
    )


@dataclass(frozen=True)
class _ControllerRecord:
    step_index: int
    segment_start: bool
    diagnostics: TrackerDiagnostics


class _DelayAwareControllerBridge:
    """Adapt the V20 controller callback without crossing the truth boundary."""

    def __init__(
        self,
        cfg: UUV3DConfig,
        tracker: Optional[DelayAwareFormationTracker] = None,
    ) -> None:
        self.tracker = tracker or DelayAwareFormationTracker(
            delay_aware_config_for_environment(cfg)
        )
        self.last_track_step: Optional[int] = None
        self.segment_count = 0
        self.reacquisition_reset_count = 0
        self.records: Dict[int, _ControllerRecord] = {}

    def __call__(
        self,
        env: Any,
        estimated_position_m: Sequence[float],
    ) -> np.ndarray:
        step_index = int(env.step_count)
        consecutive = (
            self.last_track_step is not None
            and step_index == self.last_track_step + 1
        )
        segment_start = not consecutive
        if segment_start:
            if self.tracker.initialized:
                self.tracker.reset()
                self.reacquisition_reset_count += 1
            self.segment_count += 1

        _, desired = env._formation_desired()
        observation = TrackingObservation(
            time_s=float(env.t),
            estimated_position_m=np.asarray(
                estimated_position_m, dtype=np.float64
            ),
            desired_position_m=np.asarray(desired, dtype=np.float64),
            leader1_velocity_mps=_leader_velocity_from_broadcast(
                env.leader1_speed, env.yaw_L1
            ),
            leader2_velocity_mps=_leader_velocity_from_broadcast(
                env.leader2_speed, env.yaw_L2
            ),
            measured_speed_mps=float(env._v11_speed_meas),
            measured_yaw_deg=float(env._v11_yaw_meas),
            measured_pitch_deg=float(env._v11_pitch_meas),
        )
        if segment_start:
            previous_action = (
                np.asarray(env._v40_requested_actions[-1], dtype=np.float64)
                if env._v40_requested_actions
                else np.zeros(3, dtype=np.float64)
            )
            action = self.tracker.action(
                observation,
                previous_action=previous_action,
            )
        else:
            action = self.tracker.action(observation)

        diagnostics = self.tracker.last_diagnostics
        if diagnostics is None:
            raise RuntimeError("delay-aware tracker omitted diagnostics")
        if step_index in self.records:
            raise RuntimeError("controller was called twice for one action")
        self.records[step_index] = _ControllerRecord(
            step_index=step_index,
            segment_start=segment_start,
            diagnostics=diagnostics,
        )
        self.last_track_step = step_index
        return np.asarray(action, dtype=np.float32)


_VECTOR_DIAGNOSTICS: Mapping[str, str] = {
    "controller_predicted_position_m": "predicted_position_m",
    "controller_predicted_state": "predicted_state",
    "controller_predicted_rate_state": "predicted_rate_state",
    "controller_predicted_formation_error_m": "predicted_formation_error_m",
    "controller_unconstrained_reference_velocity_mps": (
        "unconstrained_reference_velocity_mps"
    ),
    "controller_governed_reference_velocity_mps": (
        "governed_reference_velocity_mps"
    ),
    "controller_raw_action": "raw_action",
    "controller_blended_action": "blended_action",
    "controller_returned_action": "returned_action",
}


def _controller_trace(
    action_count: int,
    bridge: Optional[_DelayAwareControllerBridge],
) -> Dict[str, np.ndarray]:
    n = int(action_count)
    trace: Dict[str, np.ndarray] = {
        name: np.full((n, 3), np.nan, dtype=np.float64)
        for name in _VECTOR_DIAGNOSTICS
    }
    trace.update(
        {
            "controller_active": np.zeros(n, dtype=bool),
            "controller_segment_start": np.zeros(n, dtype=bool),
            "controller_bumpless_blend": np.full(n, np.nan, dtype=np.float64),
            "controller_command_delta_limited": np.zeros(
                (n, 3), dtype=bool
            ),
        }
    )
    if bridge is None:
        return trace
    for step_index, record in bridge.records.items():
        if not 0 <= step_index < n:
            raise RuntimeError("controller diagnostic index is outside trace")
        diagnostics = record.diagnostics
        trace["controller_active"][step_index] = True
        trace["controller_segment_start"][step_index] = record.segment_start
        trace["controller_bumpless_blend"][step_index] = float(
            diagnostics.bumpless_blend
        )
        trace["controller_command_delta_limited"][step_index] = np.asarray(
            diagnostics.command_delta_limited, dtype=bool
        )
        for trace_name, attribute in _VECTOR_DIAGNOSTICS.items():
            trace[trace_name][step_index] = np.asarray(
                getattr(diagnostics, attribute), dtype=np.float64
            )
    return trace


def _rms(values: np.ndarray) -> Optional[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(array))))


def _channel_rms(values: np.ndarray) -> Optional[list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] == 0:
        return None
    return np.sqrt(np.mean(np.square(array), axis=0)).tolist()


def _track_metrics(
    trace: Mapping[str, np.ndarray],
    cfg: UUV3DConfig,
) -> Dict[str, Any]:
    phase = np.asarray(trace["phase_track"], dtype=bool)
    requested = np.asarray(trace["plant_requested_action"], dtype=np.float64)
    delivered = np.asarray(trace["plant_delivered_action"], dtype=np.float64)
    executed_rate = np.asarray(
        trace["plant_executed_rate_state"], dtype=np.float64
    )
    if requested.shape != delivered.shape or requested.shape != executed_rate.shape:
        raise RuntimeError("V41 plant trace shapes do not agree")
    if requested.shape != (phase.size, 3):
        raise RuntimeError("V41 action trace is not aligned with phase")

    normalized_rate_scales = np.asarray(
        [
            float(cfg.rl_speed_delta_per_step) / float(cfg.action_dt),
            min(
                float(cfg.max_yaw_rate_deg_s),
                float(cfg.rl_yaw_per_step_deg) / float(cfg.action_dt),
            ),
            min(
                float(cfg.max_pitch_rate_deg_s),
                float(cfg.rl_pitch_per_step_deg) / float(cfg.action_dt),
            ),
        ],
        dtype=np.float64,
    )
    if np.any(normalized_rate_scales <= 0.0):
        raise RuntimeError("invalid V41 command-channel scales")
    executed_normalized = executed_rate / normalized_rate_scales[None, :]

    saturation = np.abs(requested) >= 1.0 - 1e-7
    track_actions = requested[phase]
    track_saturation = saturation[phase]
    consecutive = phase[1:] & phase[:-1]
    track_delta = (requested[1:] - requested[:-1])[consecutive]
    requested_delivered = (requested - delivered)[phase]
    requested_executed = (requested - executed_normalized)[phase]
    delivered_executed = (delivered - executed_normalized)[phase]

    if track_actions.shape[0] == 0:
        saturation_count = 0
        saturation_fraction = None
        per_channel_saturation = None
    else:
        any_saturation = np.any(track_saturation, axis=1)
        saturation_count = int(np.sum(any_saturation))
        saturation_fraction = float(np.mean(any_saturation))
        per_channel_saturation = np.mean(track_saturation, axis=0).tolist()

    phase_start = phase & ~np.concatenate(
        [np.asarray([False]), phase[:-1]]
    )
    return {
        "track_action_count": int(track_actions.shape[0]),
        "track_segment_count": int(np.sum(phase_start)),
        "any_channel_saturation_count": saturation_count,
        "any_channel_saturation_fraction": saturation_fraction,
        "per_channel_saturation_fraction": per_channel_saturation,
        "consecutive_track_transition_count": int(track_delta.shape[0]),
        "action_delta_rms": _rms(track_delta),
        "action_delta_rms_per_channel": _channel_rms(track_delta),
        "maximum_absolute_action_delta": (
            None
            if track_delta.size == 0
            else float(np.max(np.abs(track_delta)))
        ),
        "maximum_absolute_action_delta_per_channel": (
            None
            if track_delta.shape[0] == 0
            else np.max(np.abs(track_delta), axis=0).tolist()
        ),
        "requested_delivered_action_mismatch_rms": _rms(
            requested_delivered
        ),
        "requested_delivered_action_mismatch_rms_per_channel": _channel_rms(
            requested_delivered
        ),
        "requested_executed_rate_mismatch_rms": _rms(requested_executed),
        "requested_executed_rate_mismatch_rms_per_channel": _channel_rms(
            requested_executed
        ),
        "delivered_executed_rate_mismatch_rms": _rms(delivered_executed),
        "delivered_executed_rate_mismatch_rms_per_channel": _channel_rms(
            delivered_executed
        ),
    }


def _plant_summary(
    env: v40.DynamicCurrentEnv,
    trace: Mapping[str, np.ndarray],
    current_name: str,
) -> Dict[str, Any]:
    """Reproduce the V40 plant summary without changing its semantics."""

    return {
        "parameters": env.plant_parameters.to_dict(),
        "bottom_track_current_visible": current_name == CURRENT_VISIBLE,
        "planner_uses_execution_model": False,
        "planner_uses_current_model": False,
        "requested_delivered_action_rms": float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.asarray(trace["plant_requested_action"])
                        - np.asarray(trace["plant_delivered_action"])
                    )
                )
            )
        ),
        "executed_rate_rms": np.sqrt(
            np.mean(
                np.square(trace["plant_executed_rate_state"]),
                axis=0,
            )
        ).tolist(),
        "current_speed_rms_mps": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(trace["water_current_mps"]),
                        axis=1,
                    )
                )
            )
        ),
        "minimum_ground_speed_mps": float(
            np.min(np.linalg.norm(trace["ground_velocity_mps"], axis=1))
        ),
        "ground_speed_below_0p05_action_count": int(
            np.sum(np.linalg.norm(trace["ground_velocity_mps"], axis=1) < 0.05)
        ),
    }


def run_controller_arm(
    *,
    arm: ArmSpec,
    cfg: UUV3DConfig,
    tape: Any,
    episode_seed: int,
    episode_index: int,
    estimator_config: Any,
    lock_config: Any,
    planner_config: Any,
) -> v38.ArmOutcome:
    """Run one frozen active-acquisition arm with one post-lock controller."""

    if arm.name not in ARM_BY_NAME:
        raise ValueError("unknown V41 arm")
    _assert_seed_allowed_automatic(episode_seed)
    holder: Dict[str, v40.DynamicCurrentEnv] = {}
    original_factory = v38.UUVTwoLeader3DPFEnv
    original_seed_guard = v38.assert_seed_allowed
    original_pid = v20.pid_action_for_position
    bridge: Optional[_DelayAwareControllerBridge] = None

    def factory(
        *,
        cfg: UUV3DConfig,
        render_mode: str = "none",
    ) -> v40.DynamicCurrentEnv:
        env = v40.DynamicCurrentEnv(
            cfg=cfg,
            render_mode=render_mode,
            dynamics=DYNAMICS,
            current=arm.current,
            episode_seed=int(episode_seed),
        )
        holder["env"] = env
        return env

    v38.UUVTwoLeader3DPFEnv = factory  # type: ignore[assignment]
    v38.assert_seed_allowed = _assert_seed_allowed_automatic  # type: ignore[assignment]
    if arm.controller == DELAY_AWARE:
        bridge = _DelayAwareControllerBridge(cfg)
        v20.pid_action_for_position = bridge  # type: ignore[assignment]
    try:
        base = v38.run_arm(
            cfg=cfg,
            tape=tape,
            episode_seed=int(episode_seed),
            episode_index=int(episode_index),
            source_name=v38.SOURCE_BOTH,
            policy_name=POLICY,
            estimator_config=estimator_config,
            lock_config=lock_config,
            planner_config=planner_config,
        )
    finally:
        v20.pid_action_for_position = original_pid
        v38.UUVTwoLeader3DPFEnv = original_factory
        v38.assert_seed_allowed = original_seed_guard

    env = holder.get("env")
    if env is None:
        raise RuntimeError("V41 environment factory was not used")
    trace = dict(base.trace)
    trace.update(env.plant_trace())
    trace.update(_controller_trace(int(env.step_count), bridge))

    scales = delay_aware_config_for_environment(cfg).normalized_rate_scales
    executed_normalized = (
        np.asarray(trace["plant_executed_rate_state"], dtype=np.float64)
        / scales[None, :]
    )
    trace["plant_executed_rate_normalized"] = executed_normalized
    trace["plant_requested_executed_rate_mismatch"] = (
        np.asarray(trace["plant_requested_action"], dtype=np.float64)
        - executed_normalized
    )

    phase = np.asarray(trace["phase_track"], dtype=bool)
    controller_active = np.asarray(trace["controller_active"], dtype=bool)
    if bridge is not None and not np.array_equal(controller_active, phase):
        raise RuntimeError(
            "delay-aware diagnostic calls do not match TRACK actions"
        )
    if bridge is None and np.any(controller_active):
        raise RuntimeError("baseline arm unexpectedly contains tracker calls")

    summary = dict(base.summary)
    controller_config = (
        None
        if bridge is None
        else asdict(bridge.tracker.config)
    )
    summary.update(
        {
            "version": VERSION,
            "arm": arm.name,
            "policy_name": POLICY,
            "dynamics_name": DYNAMICS,
            "controller_name": arm.controller,
            "current_name": arm.current,
            "current_tape_sha256": env.current_tape_sha256,
            "plant": _plant_summary(env, trace, arm.current),
            "controller": {
                "name": arm.controller,
                "config": controller_config,
                "causal_inputs_only": True,
                "online_inputs": [
                    "estimated_follower_position",
                    "desired_formation_position",
                    "leader_speed_and_heading_broadcasts",
                    "bottom_track_speed_yaw_pitch",
                    "own_requested_action_history",
                ],
                "simulator_truth_scoring_only": True,
                "planner_unchanged": True,
                "estimator_unchanged": True,
                "audited_gate_unchanged": True,
                "segment_count": (
                    0 if bridge is None else int(bridge.segment_count)
                ),
                "reacquisition_reset_count": (
                    0
                    if bridge is None
                    else int(bridge.reacquisition_reset_count)
                ),
                "track_metrics": _track_metrics(trace, cfg),
            },
        }
    )
    return v38.ArmOutcome(summary=summary, trace=trace)


def baseline_equivalence_report(
    candidate: v38.ArmOutcome,
    reference: v38.ArmOutcome,
) -> Dict[str, Any]:
    """Compare deterministic common traces while excluding runtime fields.

    This is intended for the two previously opened smoke seeds.  Controller
    diagnostics added by V41 have no counterpart in V40 and are excluded.
    ``planner_runtime_s`` is measured wall time and is likewise excluded.
    """

    excluded_prefixes = ("controller_",)
    excluded_keys = {"planner_runtime_s"}
    common = sorted(set(candidate.trace) & set(reference.trace))
    compared = [
        key
        for key in common
        if key not in excluded_keys
        and not key.startswith(excluded_prefixes)
    ]
    mismatched = []
    for key in compared:
        left = np.asarray(candidate.trace[key])
        right = np.asarray(reference.trace[key])
        if left.shape != right.shape or not np.array_equal(
            left, right, equal_nan=True
        ):
            mismatched.append(key)
    return {
        "bitwise_equal_outside_runtime": not mismatched,
        "compared_trace_count": len(compared),
        "mismatched_traces": mismatched,
        "candidate_noise_tape_sha256": candidate.summary.get(
            "noise_tape_sha256"
        ),
        "reference_noise_tape_sha256": reference.summary.get(
            "noise_tape_sha256"
        ),
    }


def condition_contract() -> Dict[str, Any]:
    default_config = asdict(DelayAwareTrackerConfig())
    return {
        "version": VERSION,
        "design": "paired_2_controllers_x_2_currents",
        "policy": POLICY,
        "dynamics": DYNAMICS,
        "controllers": list(CONTROLLERS),
        "currents": list(CURRENTS),
        "arms": [arm.to_dict() for arm in ARM_SPECS],
        "plant_parameters": v40.PLANT_PARAMETERS[DYNAMICS].to_dict(),
        "delay_aware_controller_defaults": default_config,
        "smoke_seeds": list(SMOKE_SEEDS),
        "qualification_seeds": [QUALIFICATION_START, QUALIFICATION_END],
        "sealed_final_range": [FINAL_START, FINAL_END],
        "episodes": 100,
        "runs": 400,
        "sequential_within_seed": True,
        "paired_sensor_noise_tape": True,
        "paired_current_tape": True,
        "estimator_planner_gate_frozen": True,
        "truth_scoring_only": True,
        "retuning_allowed": False,
    }


__all__ = [
    "VERSION",
    "POLICY",
    "DYNAMICS",
    "BASELINE_PID",
    "DELAY_AWARE",
    "CONTROLLERS",
    "CURRENT_NONE",
    "CURRENT_VISIBLE",
    "CURRENTS",
    "SMOKE_SEEDS",
    "FINAL_START",
    "FINAL_END",
    "QUALIFICATION_START",
    "QUALIFICATION_END",
    "ArmSpec",
    "ARM_SPECS",
    "ARM_BY_NAME",
    "arm_specs",
    "assert_seed_allowed",
    "delay_aware_config_for_environment",
    "run_controller_arm",
    "baseline_equivalence_report",
    "condition_contract",
]
