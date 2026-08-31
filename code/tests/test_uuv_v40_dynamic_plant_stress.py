from __future__ import annotations

import itertools

import numpy as np
import pytest

import uuv_v40_dynamic_plant_stress as v40


def _arm_specs():
    """Accept either a frozen tuple or a function returning that tuple."""

    if hasattr(v40, "ARM_SPECS"):
        return tuple(v40.ARM_SPECS)
    return tuple(v40.arm_specs())


def test_seed_boundaries_keep_final_holdout_sealed() -> None:
    assert tuple(v40.SMOKE_SEEDS) == (49_566, 49_591)
    for seed in v40.SMOKE_SEEDS:
        v40.assert_seed_allowed(seed, smoke=True)
    v40.assert_seed_allowed(49_900)
    v40.assert_seed_allowed(49_999)
    with pytest.raises(PermissionError):
        v40.assert_seed_allowed(50_000)
    with pytest.raises(PermissionError):
        v40.assert_seed_allowed(49_899)
    with pytest.raises(PermissionError):
        v40.assert_seed_allowed(49_900, smoke=True)


def test_factorial_contract_has_exactly_eight_unique_arms() -> None:
    assert tuple(v40.POLICIES) == ("fixed_s_turn", "belief_active")
    assert tuple(v40.DYNAMICS) == (
        v40.DYNAMICS_KINEMATIC,
        v40.DYNAMICS_LOW_ORDER,
    )
    assert tuple(v40.CURRENTS) == (v40.CURRENT_NONE, v40.CURRENT_VISIBLE)

    specs = _arm_specs()
    observed = {(s.policy, s.dynamics, s.current) for s in specs}
    expected = set(itertools.product(v40.POLICIES, v40.DYNAMICS, v40.CURRENTS))
    assert len(specs) == 8
    assert len(observed) == 8
    assert observed == expected
    assert len({s.name for s in specs}) == 8


def test_low_order_execution_constants_are_frozen() -> None:
    reference = v40.PLANT_PARAMETERS[v40.DYNAMICS_KINEMATIC]
    dynamic = v40.PLANT_PARAMETERS[v40.DYNAMICS_LOW_ORDER]

    assert reference.command_delay_actions == 0
    assert not reference.has_first_order_response
    assert dynamic.command_delay_actions == 1
    assert dynamic.surge_acceleration_time_constant_s == pytest.approx(5.0)
    assert dynamic.yaw_rate_time_constant_s == pytest.approx(2.0)
    assert dynamic.pitch_rate_time_constant_s == pytest.approx(3.0)
    assert dynamic.has_first_order_response


def test_first_order_update_is_exact_stable_and_monotone() -> None:
    assert v40._first_order(1.0, -2.0, 0.0, 0.1) == -2.0

    value = 0.0
    values = []
    for _ in range(100):
        value = v40._first_order(value, 1.0, 5.0, 0.1)
        values.append(value)
    assert np.all(np.diff(values) > 0.0)
    assert values[-1] == pytest.approx(1.0 - np.exp(-2.0), abs=1e-14)
    assert 0.0 < values[-1] < 1.0


def test_current_tape_is_reproducible_horizontal_and_content_addressed() -> None:
    first = v40.CurrentTape.generate(seed=49_566, horizon_s=440.0, dt_s=0.1)
    second = v40.CurrentTape.generate(seed=49_566, horizon_s=440.0, dt_s=0.1)
    other = v40.CurrentTape.generate(seed=49_591, horizon_s=440.0, dt_s=0.1)

    assert np.array_equal(first.horizontal_velocity_mps, second.horizontal_velocity_mps)
    assert first.content_sha256() == second.content_sha256()
    assert len(first.content_sha256()) == 64
    assert first.content_sha256() != other.content_sha256()
    assert first.horizontal_velocity_mps.shape == (4_401, 2)
    assert first.gauss_markov_velocity_mps.shape == (4_401, 2)
    assert first.steady_horizontal_velocity_mps.shape == (2,)
    assert np.linalg.norm(first.steady_horizontal_velocity_mps) == pytest.approx(0.30)
    assert np.array_equal(
        first.horizontal_velocity_mps,
        first.steady_horizontal_velocity_mps[None, :]
        + first.gauss_markov_velocity_mps,
    )


def test_current_tape_uses_stationary_gauss_markov_initialization() -> None:
    # Across independent seeds, the marginal moments should approach the frozen
    # stationary model. This also catches zero-initialized transient tapes.
    samples = np.asarray(
        [
            v40.CurrentTape.generate(seed=seed, horizon_s=0.0, dt_s=0.1)
            .gauss_markov_velocity_mps[0]
            for seed in range(49_000, 49_800)
        ]
    )
    assert np.all(np.abs(np.mean(samples, axis=0)) < 0.006)
    assert np.all(np.abs(np.std(samples, axis=0, ddof=1) - 0.05) < 0.006)


def test_current_parameters_and_contract_are_explicit() -> None:
    assert v40.CURRENT_STEADY_SPEED_MPS == pytest.approx(0.30)
    assert v40.CURRENT_GM_SIGMA_MPS == pytest.approx(0.05)
    assert v40.CURRENT_GM_TAU_S == pytest.approx(120.0)

    contract = v40.condition_contract()
    assert contract["qualification_seeds"] == [49_900, 49_999]
    assert contract["sealed_final_range"] == [50_000, 50_999]
    assert contract["runs"] == 800
    assert contract["policies"] == list(v40.POLICIES)
    assert contract["dynamics"] == list(v40.DYNAMICS)
    assert contract["currents"] == list(v40.CURRENTS)
    assert len(contract["arms"]) == 8
    assert contract["retuning_allowed"] is False


def test_same_seed_current_tape_is_policy_and_plant_independent() -> None:
    tape = v40.CurrentTape.generate(seed=49_566, horizon_s=440.0, dt_s=0.1)
    digest = tape.content_sha256()
    for spec in _arm_specs():
        # Arm metadata must not alter the exogenous current realization. The
        # current switch may zero its application, but never regenerates it.
        regenerated = v40.CurrentTape.generate(seed=49_566, horizon_s=440.0, dt_s=0.1)
        assert regenerated.content_sha256() == digest, spec.name
