from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from uuv_v11_rng import EpisodeSeedPlan, ExogenousNoiseTape
from uuv_v18_resampling_guard import UUV3DConfig
import run_v20_positioning_ablation as runner20
import uuv_v19_observability as v19
import uuv_v20_positioning_ablation as v20
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24
import uuv_v38_leader_source_ablation as v38
import uuv_v40_dynamic_plant_stress as v40
import uuv_v41_controller_repair as v41


def test_contract_contains_exactly_the_four_preregistered_arms() -> None:
    observed = {(arm.controller, arm.current) for arm in v41.ARM_SPECS}
    expected = {
        (controller, current)
        for controller in v41.CONTROLLERS
        for current in v41.CURRENTS
    }
    assert observed == expected
    assert len(v41.ARM_SPECS) == 4
    assert len({arm.name for arm in v41.ARM_SPECS}) == 4
    for arm in v41.ARM_SPECS:
        assert arm.policy == v38.POLICY_ACTIVE
        assert arm.dynamics == v40.DYNAMICS_LOW_ORDER

    contract = v41.condition_contract()
    assert contract["design"] == "paired_2_controllers_x_2_currents"
    assert contract["qualification_seeds"] == [51_000, 51_099]
    assert contract["sealed_final_range"] == [50_000, 50_999]
    assert contract["runs"] == 400
    assert contract["retuning_allowed"] is False


def test_seed_boundaries_do_not_reopen_the_final_holdout() -> None:
    for seed in v41.SMOKE_SEEDS:
        v41.assert_seed_allowed(seed, smoke=True)
    v41.assert_seed_allowed(51_000)
    v41.assert_seed_allowed(51_099)
    for seed in (50_000, 50_999):
        with pytest.raises(PermissionError, match="sealed final"):
            v41.assert_seed_allowed(seed)
    with pytest.raises(PermissionError):
        v41.assert_seed_allowed(49_900)
    with pytest.raises(PermissionError):
        v41.assert_seed_allowed(51_000, smoke=True)
    with pytest.raises(PermissionError):
        v41.assert_seed_allowed(51_100)


def test_controller_config_matches_the_low_order_plant_and_action_limits() -> None:
    cfg = UUV3DConfig()
    controller = v41.delay_aware_config_for_environment(cfg)
    plant = v40.PLANT_PARAMETERS[v40.DYNAMICS_LOW_ORDER]
    assert controller.action_dt_s == pytest.approx(cfg.action_dt)
    assert controller.command_delay_actions == plant.command_delay_actions
    assert controller.surge_acceleration_time_constant_s == pytest.approx(5.0)
    assert controller.yaw_rate_time_constant_s == pytest.approx(2.0)
    assert controller.pitch_rate_time_constant_s == pytest.approx(3.0)
    assert controller.normalized_rate_scales == pytest.approx(
        [
            cfg.rl_speed_delta_per_step / cfg.action_dt,
            min(cfg.max_yaw_rate_deg_s, cfg.rl_yaw_per_step_deg / cfg.action_dt),
            min(
                cfg.max_pitch_rate_deg_s,
                cfg.rl_pitch_per_step_deg / cfg.action_dt,
            ),
        ]
    )


class _OnlineOnlyFakeEnv:
    def __init__(self, cfg: UUV3DConfig) -> None:
        self.cfg = cfg
        self.step_count = 0
        self.t = 120.0
        self.leader1_speed = 2.0
        self.leader2_speed = 2.2
        self.yaw_L1 = 10.0
        self.yaw_L2 = 15.0
        self._v11_speed_meas = 2.1
        self._v11_yaw_meas = 12.0
        self._v11_pitch_meas = 1.0
        self._v40_requested_actions = [
            np.asarray([0.6, -0.4, 0.2], dtype=np.float64)
        ]

    def _formation_desired(self):
        return None, np.asarray([20.0 + self.t, 4.0, -3.0])


def test_bridge_is_bumpless_and_resets_after_a_track_gap() -> None:
    env = _OnlineOnlyFakeEnv(UUV3DConfig())
    bridge = v41._DelayAwareControllerBridge(env.cfg)
    estimate = np.asarray([env.t, 0.0, 0.0])

    first = bridge(env, estimate)
    assert np.array_equal(
        first, np.asarray(env._v40_requested_actions[-1], dtype=np.float32)
    )
    env._v40_requested_actions.append(first.copy())
    env.step_count = 1
    env.t += env.cfg.action_dt
    second = bridge(env, estimate + np.asarray([2.0, 0.0, 0.0]))
    assert np.all(np.isfinite(second))

    # One ACQUIRE action occurs before TRACK is entered again.
    env._v40_requested_actions.extend(
        [second.copy(), np.asarray([-0.2, 0.3, -0.1])]
    )
    env.step_count = 3
    env.t += 2.0 * env.cfg.action_dt
    third = bridge(env, estimate + np.asarray([6.0, 0.0, 0.0]))
    assert np.array_equal(
        third, np.asarray(env._v40_requested_actions[-1], dtype=np.float32)
    )
    assert bridge.segment_count == 2
    assert bridge.reacquisition_reset_count == 1
    assert bridge.records[0].segment_start
    assert not bridge.records[1].segment_start
    assert bridge.records[3].segment_start


def test_baseline_diagnostic_trace_is_explicitly_empty() -> None:
    trace = v41._controller_trace(5, None)
    assert not np.any(trace["controller_active"])
    assert not np.any(trace["controller_segment_start"])
    assert not np.any(trace["controller_command_delta_limited"])
    for name in v41._VECTOR_DIAGNOSTICS:
        assert trace[name].shape == (5, 3)
        assert np.all(np.isnan(trace[name]))


def test_track_metrics_exclude_acquire_and_cross_segment_deltas() -> None:
    cfg = UUV3DConfig()
    requested = np.asarray(
        [
            [1.0, 1.0, 1.0],       # ACQUIRE: must not count.
            [0.2, 0.4, -0.1],
            [0.4, 0.0, -0.1],
            [-1.0, -1.0, -1.0],    # ACQUIRE gap.
            [0.5, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    phase = np.asarray([False, True, True, False, True])
    scales = v41.delay_aware_config_for_environment(
        cfg
    ).normalized_rate_scales
    delivered = 0.5 * requested
    executed = 0.25 * requested * scales[None, :]
    trace = {
        "phase_track": phase,
        "plant_requested_action": requested,
        "plant_delivered_action": delivered,
        "plant_executed_rate_state": executed,
        "controller_segment_start": np.asarray(
            [False, True, False, False, True]
        ),
    }
    metrics = v41._track_metrics(trace, cfg)
    assert metrics["track_action_count"] == 3
    assert metrics["track_segment_count"] == 2
    assert metrics["consecutive_track_transition_count"] == 1
    assert metrics["any_channel_saturation_count"] == 1
    assert metrics["any_channel_saturation_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["action_delta_rms_per_channel"] == pytest.approx(
        [0.2, 0.4, 0.0]
    )
    assert metrics["requested_delivered_action_mismatch_rms"] is not None
    assert metrics["requested_executed_rate_mismatch_rms"] is not None


def test_all_monkeypatches_are_restored_when_inherited_run_fails(monkeypatch) -> None:
    original_factory = v38.UUVTwoLeader3DPFEnv
    original_guard = v38.assert_seed_allowed
    original_pid = v20.pid_action_for_position

    def fail(**_kwargs):
        raise RuntimeError("intentional inherited failure")

    monkeypatch.setattr(v38, "run_arm", fail)
    arm = v41.ArmSpec(v41.DELAY_AWARE, v41.CURRENT_NONE)
    with pytest.raises(RuntimeError, match="intentional inherited failure"):
        v41.run_controller_arm(
            arm=arm,
            cfg=UUV3DConfig(),
            tape=object(),
            episode_seed=v41.SMOKE_SEEDS[0],
            episode_index=0,
            estimator_config=object(),
            lock_config=object(),
            planner_config=object(),
        )
    assert v38.UUVTwoLeader3DPFEnv is original_factory
    assert v38.assert_seed_allowed is original_guard
    assert v20.pid_action_for_position is original_pid


def test_baseline_equivalence_report_ignores_runtime_and_v41_diagnostics() -> None:
    summary = {"noise_tape_sha256": "same"}
    reference = v38.ArmOutcome(
        summary=summary,
        trace={
            "action_speed": np.asarray([0.0, np.nan]),
            "planner_runtime_s": np.asarray([0.1, 0.2]),
        },
    )
    candidate = v38.ArmOutcome(
        summary=summary,
        trace={
            "action_speed": np.asarray([0.0, np.nan]),
            "planner_runtime_s": np.asarray([9.0, 8.0]),
            "controller_active": np.asarray([False, False]),
        },
    )
    report = v41.baseline_equivalence_report(candidate, reference)
    assert report["bitwise_equal_outside_runtime"]
    assert report["mismatched_traces"] == []

    changed = v38.ArmOutcome(
        summary=summary,
        trace={
            "action_speed": np.asarray([0.0, 0.25]),
            "planner_runtime_s": np.asarray([9.0, 8.0]),
        },
    )
    report = v41.baseline_equivalence_report(changed, reference)
    assert not report["bitwise_equal_outside_runtime"]
    assert report["mismatched_traces"] == ["action_speed"]


@pytest.mark.skipif(
    os.environ.get("UUV_V41_RUN_SMOKE_EQUIVALENCE") != "1",
    reason="set UUV_V41_RUN_SMOKE_EQUIVALENCE=1 for the full two-run smoke",
)
def test_full_baseline_smoke_is_bitwise_v40_equivalent() -> None:
    """Execute the inherited simulator twice on an already opened seed."""

    repository = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").is_file()
    )
    archive_root = Path(
        os.environ.get("UUV_ARCHIVE_ROOT", repository / "data" / "raw")
    ).expanduser().resolve()
    metadata = (
        archive_root
        / "estimator_benchmark"
        / "v18_replay_fixture"
        / "metadata.json"
    )
    cfg = runner20._load_environment_config(metadata)
    seed = v41.SMOKE_SEEDS[0]
    substeps = int(cfg.max_steps) * int(
        round(float(cfg.action_dt) / float(cfg.sub_dt))
    )
    measurements = int(
        math.ceil(
            float(cfg.max_steps * cfg.action_dt) / float(cfg.s_meas_period)
        )
    ) + 2
    tape = ExogenousNoiseTape.generate(
        EpisodeSeedPlan(root_seed=seed, episode_index=0, env_rank=0),
        n_substeps=substeps,
        n_doppler_measurements=measurements,
    )
    estimator = v19.BatchEstimatorConfig(
        coarse_candidates=64,
        coarse_sweeps=1,
        local_starts=2,
        gate_mode="raw",
        candidate_radial_distribution="uniform_radius",
    )
    lock = v24.AuditedLockConfig()
    planner = v22.ActivePlannerConfig()
    candidate = v41.run_controller_arm(
        arm=v41.ArmSpec(v41.BASELINE_PID, v41.CURRENT_NONE),
        cfg=cfg,
        tape=tape,
        episode_seed=seed,
        episode_index=0,
        estimator_config=estimator,
        lock_config=lock,
        planner_config=planner,
    )
    reference = v40.run_factorial_arm(
        arm=v40.ArmSpec(
            v40.POLICY_ACTIVE,
            v40.DYNAMICS_LOW_ORDER,
            v40.CURRENT_NONE,
        ),
        cfg=cfg,
        tape=tape,
        episode_seed=seed,
        episode_index=0,
        estimator_config=estimator,
        lock_config=lock,
        planner_config=planner,
    )
    report = v41.baseline_equivalence_report(candidate, reference)
    assert report["bitwise_equal_outside_runtime"], report
