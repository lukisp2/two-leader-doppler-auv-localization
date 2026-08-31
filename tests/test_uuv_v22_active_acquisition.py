from types import SimpleNamespace

import numpy as np
import pytest

import run_v22_active_acquisition as runner22
import uuv_v22_active_acquisition as v22


class _KinematicsOnlyEnv:
    def __init__(self) -> None:
        self._v11_speed_meas = 2.0
        self._v11_yaw_meas = 10.0
        self._v11_pitch_meas = 0.0
        self.cfg = SimpleNamespace(
            rl_speed_delta_per_step=0.4,
            rl_yaw_per_step_deg=20.0,
            rl_pitch_per_step_deg=14.0,
            f_min_speed=0.2,
            f_max_speed=5.0,
            pitch_min_deg=-45.0,
            pitch_max_deg=45.0,
        )

    @property
    def pF(self):
        raise AssertionError("planner attempted to read follower truth")


def test_candidate_grid_is_finite_bounded_and_contains_zero() -> None:
    candidates = v22._candidate_actions(np.asarray([0.25, -0.25, 0.5]))
    assert candidates.shape == (46, 3)
    assert np.all(np.isfinite(candidates))
    assert np.all(np.abs(candidates) <= 1.0)
    assert np.any(np.all(candidates == 0.0, axis=1))


def test_action_to_velocity_uses_only_onboard_kinematics() -> None:
    velocity = v22._action_velocity(np.asarray([1.0, -0.5, 1.0]), _KinematicsOnlyEnv())
    assert velocity.shape == (3,)
    assert np.all(np.isfinite(velocity))
    assert np.linalg.norm(velocity) == pytest.approx(2.4)


def test_future_information_cannot_increase_worst_radius() -> None:
    planner = v22.BeliefFIMPlanner(v22.ActivePlannerConfig())
    base = np.repeat((1e-4 * np.eye(3))[None, :, :], 2, axis=0)
    before, after, pair = planner._evaluate_velocity(
        velocity_mps=np.asarray([2.0, 0.0, 0.0]),
        hypotheses_p0_m=np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        current_dead_reckoning_m=np.zeros(3),
        base_information=base,
        leader_position_m=np.asarray(
            [[100.0, -50.0, -30.0], [100.0, 50.0, -70.0]]
        ),
        leader_velocity_mps=np.asarray(
            [[2.0, 0.2, 0.0], [1.8, -0.1, 0.0]]
        ),
        sigma_mps=0.05,
    )
    assert np.isfinite([before, after, pair]).all()
    assert after <= before
    assert pair >= 0.0


def test_runner_refuses_sealed_final_seed() -> None:
    args = runner22._parse_args(["--seed-start", "50000", "--episodes", "1"])
    with pytest.raises(PermissionError):
        runner22._settings(args)
