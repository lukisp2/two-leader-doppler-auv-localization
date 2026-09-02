from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from delay_aware_formation_tracker import (
    DelayAwareFormationTracker,
    DelayAwareTrackerConfig,
    TrackingObservation,
)


def _observation(
    *,
    time_s: float = 120.0,
    estimate=(0.0, 0.0, 0.0),
    desired=(0.0, 0.0, 0.0),
    leader_velocity=(2.0, 0.0, 0.0),
    measured_speed: float = 2.0,
    measured_yaw: float = 0.0,
    measured_pitch: float = 0.0,
) -> TrackingObservation:
    return TrackingObservation(
        time_s=time_s,
        estimated_position_m=estimate,
        desired_position_m=desired,
        leader1_velocity_mps=leader_velocity,
        leader2_velocity_mps=leader_velocity,
        measured_speed_mps=measured_speed,
        measured_yaw_deg=measured_yaw,
        measured_pitch_deg=measured_pitch,
    )


def test_config_factory_copies_only_explicit_plant_limits() -> None:
    @dataclass
    class Config:
        action_dt: float = 2.0
        rl_speed_delta_per_step: float = 0.4
        rl_yaw_per_step_deg: float = 20.0
        rl_pitch_per_step_deg: float = 14.0
        max_yaw_rate_deg_s: float = 60.0
        max_pitch_rate_deg_s: float = 45.0
        f_min_speed: float = 0.2
        f_max_speed: float = 5.0
        pitch_min_deg: float = -45.0
        pitch_max_deg: float = 45.0
        pF: tuple[float, float, float] = (999.0, 999.0, 999.0)

    controller_config = DelayAwareTrackerConfig.from_simulator_config(
        Config(),
        command_delay_actions=1,
        surge_acceleration_time_constant_s=5.0,
        yaw_rate_time_constant_s=2.0,
        pitch_rate_time_constant_s=3.0,
    )
    assert controller_config.command_delay_actions == 1
    assert np.array_equal(
        controller_config.normalized_rate_scales,
        np.asarray([0.2, 10.0, 7.0]),
    )
    assert not hasattr(controller_config, "pF")


def test_default_outer_loop_preserves_reference_controller_authority() -> None:
    config = DelayAwareTrackerConfig()
    assert config.along_gain_per_s == pytest.approx(0.055)
    assert config.lateral_gain_per_s == pytest.approx(0.065)
    assert config.vertical_gain_per_s == pytest.approx(0.055)
    assert config.maximum_horizontal_correction_mps == pytest.approx(
        np.sqrt(2.0) * 1.65
    )
    assert config.maximum_vertical_correction_mps == pytest.approx(1.20)


def test_first_track_action_is_exactly_bumpless() -> None:
    tracker = DelayAwareFormationTracker()
    acquire_action = np.asarray([0.7, -0.6, 0.4], dtype=np.float32)
    returned = tracker.action(
        _observation(desired=(60.0, 40.0, -20.0)),
        previous_action=acquire_action,
    )
    assert np.array_equal(returned, acquire_action)
    assert tracker.last_diagnostics is not None
    assert tracker.last_diagnostics.bumpless_blend == 0.0
    assert tracker.last_diagnostics.predictor_horizon_s == pytest.approx(2.0)


def test_smith_predictor_propagates_position_through_pending_delay() -> None:
    config = DelayAwareTrackerConfig(bumpless_transfer_actions=1)
    tracker = DelayAwareFormationTracker(config)
    tracker.action(_observation(), previous_action=(0.0, 0.0, 0.0))
    diagnostics = tracker.last_diagnostics
    assert diagnostics is not None
    # Constant 2 m/s motion over the known one-action (2 s) queue.
    assert diagnostics.predicted_position_m == pytest.approx([4.0, 0.0, 0.0])
    assert diagnostics.predicted_state == pytest.approx([2.0, 0.0, 0.0])


def test_smith_predictor_uses_queued_yaw_command_and_known_lag() -> None:
    config = DelayAwareTrackerConfig(bumpless_transfer_actions=1)
    tracker = DelayAwareFormationTracker(config)
    tracker.action(_observation(), previous_action=(0.0, 1.0, 0.0))
    diagnostics = tracker.last_diagnostics
    assert diagnostics is not None
    expected_yaw = 20.0 - 20.0 * (1.0 - np.exp(-1.0))
    expected_rate = 10.0 * (1.0 - np.exp(-1.0))
    assert diagnostics.predicted_state[1] == pytest.approx(expected_yaw, abs=1e-12)
    assert diagnostics.predicted_rate_state[1] == pytest.approx(
        expected_rate, abs=1e-12
    )


def test_commands_are_bounded_and_slew_limited_after_transfer() -> None:
    config = DelayAwareTrackerConfig(
        bumpless_transfer_actions=1,
        maximum_action_delta=(0.12, 0.10, 0.08),
    )
    tracker = DelayAwareFormationTracker(config)
    actions = []
    previous = np.asarray([0.8, -0.8, 0.8])
    for index in range(12):
        action = tracker.action(
            _observation(
                time_s=120.0 + 2.0 * index,
                estimate=(0.0, 0.0, 0.0),
                desired=(200.0, 200.0, -100.0),
            ),
            previous_action=previous if index == 0 else None,
        )
        actions.append(np.asarray(action, dtype=np.float64))
    values = np.asarray(actions)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values)) <= 1.0
    assert np.all(
        np.abs(np.diff(values, axis=0))
        <= np.asarray(config.maximum_action_delta)[None, :] + 2e-7
    )


def test_reference_governor_limits_vector_acceleration() -> None:
    config = DelayAwareTrackerConfig(
        bumpless_transfer_actions=1,
        maximum_reference_acceleration_mps2=0.10,
    )
    tracker = DelayAwareFormationTracker(config)
    tracker.action(
        _observation(desired=(200.0, 200.0, -100.0)),
        previous_action=(0.0, 0.0, 0.0),
    )
    diagnostics = tracker.last_diagnostics
    assert diagnostics is not None
    initial_velocity = np.asarray([2.0, 0.0, 0.0])
    reference_change = np.linalg.norm(
        diagnostics.governed_reference_velocity_mps - initial_velocity
    )
    assert reference_change == pytest.approx(0.20, abs=1e-12)
    assert np.linalg.norm(
        diagnostics.unconstrained_reference_velocity_mps - initial_velocity
    ) > reference_change


def test_yaw_derivative_observer_uses_wrapped_difference() -> None:
    config = DelayAwareTrackerConfig(
        bumpless_transfer_actions=1,
        measured_rate_blend=1.0,
    )
    tracker = DelayAwareFormationTracker(config)
    tracker.action(
        _observation(measured_yaw=359.0),
        previous_action=(0.0, 0.0, 0.0),
    )
    tracker.action(_observation(time_s=122.0, measured_yaw=1.0))
    diagnostics = tracker.last_diagnostics
    assert diagnostics is not None
    # The measured rate is +1 deg/s, not -179 deg/s. It decays while the
    # pending zero command is propagated through the Smith horizon.
    assert 0.0 < diagnostics.predicted_rate_state[1] < 1.0


def test_matched_reference_and_zero_error_produce_zero_raw_command() -> None:
    config = DelayAwareTrackerConfig(
        command_delay_actions=0,
        bumpless_transfer_actions=1,
    )
    tracker = DelayAwareFormationTracker(config)
    returned = tracker.action(
        _observation(), previous_action=(0.0, 0.0, 0.0)
    )
    diagnostics = tracker.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.raw_action == pytest.approx([0.0, 0.0, 0.0], abs=1e-14)
    assert returned == pytest.approx([0.0, 0.0, 0.0], abs=1e-14)


def test_reset_replays_the_same_causal_command_sequence() -> None:
    tracker = DelayAwareFormationTracker()

    def run_once() -> np.ndarray:
        rows = []
        for index in range(8):
            rows.append(
                tracker.action(
                    _observation(
                        time_s=120.0 + 2.0 * index,
                        estimate=(2.0 * index, 3.0, -2.0),
                        desired=(2.0 * index + 15.0, 0.0, 0.0),
                    ),
                    previous_action=(0.2, -0.1, 0.3) if index == 0 else None,
                )
            )
        return np.asarray(rows)

    first = run_once()
    tracker.reset()
    second = run_once()
    assert np.array_equal(first, second)


def test_invalid_or_noncausal_call_contract_is_rejected() -> None:
    tracker = DelayAwareFormationTracker()
    with pytest.raises(ValueError, match="previous_action is required"):
        tracker.action(_observation())
    tracker.action(_observation(), previous_action=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="only on the first"):
        tracker.action(
            _observation(time_s=122.0), previous_action=(0.0, 0.0, 0.0)
        )
    with pytest.raises(ValueError, match="increase strictly"):
        tracker.action(_observation(time_s=120.0))
    with pytest.raises(ValueError, match="three finite"):
        TrackingObservation(
            time_s=124.0,
            estimated_position_m=(np.nan, 0.0, 0.0),
            desired_position_m=(0.0, 0.0, 0.0),
            leader1_velocity_mps=(2.0, 0.0, 0.0),
            leader2_velocity_mps=(2.0, 0.0, 0.0),
            measured_speed_mps=2.0,
            measured_yaw_deg=0.0,
            measured_pitch_deg=0.0,
        ).validated()


def test_independent_low_order_closed_loop_reduces_three_axis_error() -> None:
    """Numerical sanity check with the frozen delayed low-order equations."""

    config = DelayAwareTrackerConfig(bumpless_transfer_actions=4)
    tracker = DelayAwareFormationTracker(config)
    state = np.asarray([2.0, 0.0, 0.0], dtype=np.float64)
    rates = np.zeros(3, dtype=np.float64)
    position = np.asarray([-25.0, 18.0, -12.0], dtype=np.float64)
    desired = np.zeros(3, dtype=np.float64)
    leader_velocity = np.asarray([2.0, 0.0, 0.0], dtype=np.float64)
    queue = [np.zeros(3, dtype=np.float64)]
    initial_error = float(np.linalg.norm(desired - position))

    for index in range(100):
        observation = _observation(
            time_s=120.0 + index * config.action_dt_s,
            estimate=position,
            desired=desired,
            leader_velocity=leader_velocity,
            measured_speed=float(state[0]),
            measured_yaw=float(state[1]),
            measured_pitch=float(state[2]),
        )
        action = tracker.action(
            observation,
            previous_action=(0.0, 0.0, 0.0) if index == 0 else None,
        )
        delivered = queue.pop(0)
        queue.append(np.asarray(action, dtype=np.float64))
        dt = config.action_dt_s / config.predictor_substeps
        for _ in range(config.predictor_substeps):
            velocity_before = np.asarray(
                [
                    state[0]
                    * np.cos(np.deg2rad(state[2]))
                    * np.cos(np.deg2rad(state[1])),
                    state[0]
                    * np.cos(np.deg2rad(state[2]))
                    * np.sin(np.deg2rad(state[1])),
                    state[0] * np.sin(np.deg2rad(state[2])),
                ]
            )
            state, rates = tracker._advance_state_exact(
                state, rates, delivered, dt
            )
            velocity_after = np.asarray(
                [
                    state[0]
                    * np.cos(np.deg2rad(state[2]))
                    * np.cos(np.deg2rad(state[1])),
                    state[0]
                    * np.cos(np.deg2rad(state[2]))
                    * np.sin(np.deg2rad(state[1])),
                    state[0] * np.sin(np.deg2rad(state[2])),
                ]
            )
            position += 0.5 * (velocity_before + velocity_after) * dt
            desired += leader_velocity * dt

    final_error = float(np.linalg.norm(desired - position))
    assert final_error < 0.35 * initial_error
    assert np.all(np.isfinite(position))
