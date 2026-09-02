from __future__ import annotations

import numpy as np
import pytest

import run_v41_controller_repair as runner
import uuv_v38_leader_source_ablation as v38
import uuv_v40_dynamic_plant_stress as v40
import uuv_v41_controller_repair as v41


def test_runner_orders_paired_controllers_within_each_current() -> None:
    observed = [(arm.controller, arm.current) for arm in runner._ordered_arms()]
    assert observed == [
        (v41.BASELINE_PID, v40.CURRENT_NONE),
        (v41.DELAY_AWARE, v40.CURRENT_NONE),
        (v41.BASELINE_PID, v40.CURRENT_VISIBLE),
        (v41.DELAY_AWARE, v40.CURRENT_VISIBLE),
    ]


def test_immutable_contract_normalizes_tuples_to_saved_json_lists() -> None:
    left = {"created_at_utc": "first", "nested": {"value": (0.3, 0.2)}}
    right = {"created_at_utc": "second", "nested": {"value": [0.3, 0.2]}}
    assert runner._immutable(left) == runner._immutable(right)


def test_post_track_metrics_ignore_acquire_gaps() -> None:
    trace = {
        "phase_track": np.asarray([False, True, True, False, True, True]),
        "action_speed": np.asarray([1.0, 0.0, 1.0, -1.0, 0.2, 0.2]),
        "action_yaw": np.asarray([1.0, 0.0, 0.0, -1.0, 0.2, 0.2]),
        "action_pitch": np.asarray([1.0, 0.0, 0.0, -1.0, 0.2, 0.2]),
    }
    result = runner._post_track_action_metrics(trace)
    assert result["post_track_action_count"] == 4
    assert result["post_track_saturation_fraction"] == pytest.approx(0.25)
    # Only 1->2 and 4->5 are consecutive TRACK transitions.  The large
    # ACQUIRE-gap jump is intentionally absent.
    assert result["post_track_action_total_variation"] == pytest.approx(0.5)
    assert result["post_track_action_curvature_rms"] is None


def _paired_outcome(actions: np.ndarray, first_track: int) -> v38.ArmOutcome:
    n = actions.shape[0]
    phase = np.arange(n) >= first_track
    trace = {
        "phase_track": phase,
        "gate_locked_after_update": np.arange(n) >= first_track - 1,
        "action_speed": actions[:, 0],
        "action_yaw": actions[:, 1],
        "action_pitch": actions[:, 2],
        "truth_x": np.arange(n, dtype=float),
        "truth_y": np.zeros(n),
        "truth_z": np.zeros(n),
        "plant_requested_action": actions.copy(),
        "plant_delivered_action": actions.copy(),
        "plant_executed_rate_state": actions.copy(),
    }
    return v38.ArmOutcome(
        summary={
            "episode_seed": 49_566,
            "episode_index": 0,
            "noise_tape_sha256": "n" * 64,
            "current_tape_sha256": "c" * 64,
            "initial_truth_m": [1.0, 2.0, 3.0],
            "action_count": n,
        },
        trace=trace,
    )


def test_pretrack_pairing_allows_only_post_track_controller_divergence() -> None:
    baseline_action = np.zeros((6, 3))
    repaired_action = baseline_action.copy()
    repaired_action[3:, 0] = 0.5
    record = runner._pretrack_pairing(
        _paired_outcome(baseline_action, 3),
        _paired_outcome(repaired_action, 3),
        v40.CURRENT_NONE,
    )
    assert record["pretrack_bitwise_equal"]
    assert record["common_pretrack_action_count"] == 3

    repaired_action[2, 0] = 0.25
    with pytest.raises(RuntimeError, match="before TRACK"):
        runner._pretrack_pairing(
            _paired_outcome(baseline_action, 3),
            _paired_outcome(repaired_action, 3),
            v40.CURRENT_NONE,
        )
