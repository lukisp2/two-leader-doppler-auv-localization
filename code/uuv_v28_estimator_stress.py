#!/usr/bin/env python3
"""Truth-free online-history stress transforms for V28."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

import numpy as np

import uuv_v19_observability as v19


VERSION = "v28_estimator_stress_1.0"

NOMINAL = "nominal"
DOPPLER_COMMON_BIAS = "doppler_common_bias_p003"
DOPPLER_DIFFERENTIAL_BIAS = "doppler_differential_bias_003"
DOPPLER_SCALE = "doppler_scale_102"
COLORED_NOISE = "colored_noise_rho09_sd003"
DROPOUT = "dropout_iid10"
BROADCAST_DELAY = "broadcast_delay_2s"
BROADCAST_OFFSET = "broadcast_offset_2m"
DR_SCALE = "dead_reckoning_scale_101"
DR_DRIFT = "dead_reckoning_drift_001"

CONDITIONS: Tuple[str, ...] = (
    NOMINAL,
    DOPPLER_COMMON_BIAS,
    DOPPLER_DIFFERENTIAL_BIAS,
    DOPPLER_SCALE,
    COLORED_NOISE,
    DROPOUT,
    BROADCAST_DELAY,
    BROADCAST_OFFSET,
    DR_SCALE,
    DR_DRIFT,
)

CONDITION_CODE = {name: index for index, name in enumerate(CONDITIONS)}
BASE_DESIGN_SEED = 28_000_000
COLORED_RHO = 0.9
COLORED_STATIONARY_SD_MPS = 0.03
DROPOUT_PROBABILITY = 0.10
BROADCAST_DELAY_S = 2.0
BROADCAST_OFFSETS_M = np.asarray(
    [[2.0, -1.0, 0.5], [-2.0, 1.0, -0.5]], dtype=np.float64
)
DR_SCALE_FACTOR = 1.01
DR_DRIFT_VELOCITY_MPS = np.asarray(
    [0.0087287156, -0.0043643578, 0.0021821789], dtype=np.float64
)


def _rng(condition: str, episode_index: int) -> np.random.Generator:
    code = CONDITION_CODE[str(condition)]
    seed = BASE_DESIGN_SEED + 1009 * int(episode_index) + code
    return np.random.Generator(np.random.PCG64(seed))


def _history(
    source: v19.OnlineDopplerHistory,
    *,
    selection: Any = slice(None),
    dead_reckoned_displacement_m: np.ndarray | None = None,
    leader_position_m: np.ndarray | None = None,
    leader_velocity_mps: np.ndarray | None = None,
    follower_velocity_measured_mps: np.ndarray | None = None,
    doppler_measured_mps: np.ndarray | None = None,
) -> v19.OnlineDopplerHistory:
    selected = source.take(selection)
    return v19.OnlineDopplerHistory(
        t_s=selected.t_s.copy(),
        dead_reckoned_displacement_m=(
            selected.dead_reckoned_displacement_m.copy()
            if dead_reckoned_displacement_m is None
            else np.asarray(dead_reckoned_displacement_m, dtype=np.float64).copy()
        ),
        leader_position_m=(
            selected.leader_position_m.copy()
            if leader_position_m is None
            else np.asarray(leader_position_m, dtype=np.float64).copy()
        ),
        leader_velocity_mps=(
            selected.leader_velocity_mps.copy()
            if leader_velocity_mps is None
            else np.asarray(leader_velocity_mps, dtype=np.float64).copy()
        ),
        follower_velocity_measured_mps=(
            selected.follower_velocity_measured_mps.copy()
            if follower_velocity_measured_mps is None
            else np.asarray(follower_velocity_measured_mps, dtype=np.float64).copy()
        ),
        doppler_measured_mps=(
            selected.doppler_measured_mps.copy()
            if doppler_measured_mps is None
            else np.asarray(doppler_measured_mps, dtype=np.float64).copy()
        ),
        historical_pf_gate_factor=selected.historical_pf_gate_factor.copy(),
    )


def apply_stress(
    source: v19.OnlineDopplerHistory,
    condition: str,
    episode_index: int,
) -> v19.OnlineDopplerHistory:
    """Return a deterministic stressed online history without accepting truth."""

    name = str(condition)
    if name not in CONDITIONS:
        raise ValueError(f"unknown V28 stress condition {name!r}")
    if int(episode_index) < 0:
        raise ValueError("episode_index must be non-negative")
    if name == NOMINAL:
        return _history(source)
    if name == DOPPLER_COMMON_BIAS:
        return _history(
            source,
            doppler_measured_mps=source.doppler_measured_mps + 0.03,
        )
    if name == DOPPLER_DIFFERENTIAL_BIAS:
        return _history(
            source,
            doppler_measured_mps=(
                source.doppler_measured_mps
                + np.asarray([0.03, -0.03], dtype=np.float64)[None, :]
            ),
        )
    if name == DOPPLER_SCALE:
        return _history(
            source,
            doppler_measured_mps=1.02 * source.doppler_measured_mps,
        )
    if name == COLORED_NOISE:
        rng = _rng(name, episode_index)
        noise = np.empty_like(source.doppler_measured_mps, dtype=np.float64)
        noise[0] = rng.normal(0.0, COLORED_STATIONARY_SD_MPS, size=2)
        innovation_sd = COLORED_STATIONARY_SD_MPS * np.sqrt(
            1.0 - COLORED_RHO**2
        )
        for index in range(1, source.measurement_count):
            noise[index] = (
                COLORED_RHO * noise[index - 1]
                + rng.normal(0.0, innovation_sd, size=2)
            )
        return _history(
            source,
            doppler_measured_mps=source.doppler_measured_mps + noise,
        )
    if name == DROPOUT:
        rng = _rng(name, episode_index)
        keep = rng.random(source.measurement_count) >= DROPOUT_PROBABILITY
        keep[0] = True
        keep[-1] = True
        for checkpoint in (120.0, 440.0):
            matches = np.flatnonzero(
                np.isclose(source.t_s, checkpoint, rtol=0.0, atol=1e-9)
            )
            if matches.size:
                keep[int(matches[0])] = True
        if int(np.sum(keep)) < 3:
            raise RuntimeError("dropout stress retained too few rows")
        return _history(source, selection=keep)
    if name == BROADCAST_DELAY:
        delayed_indices = np.empty(source.measurement_count, dtype=np.int64)
        for index, now in enumerate(source.t_s):
            target = float(now) - BROADCAST_DELAY_S
            delayed_indices[index] = max(
                0,
                int(np.searchsorted(source.t_s, target, side="right") - 1),
            )
        return _history(
            source,
            leader_position_m=source.leader_position_m[delayed_indices],
            leader_velocity_mps=source.leader_velocity_mps[delayed_indices],
        )
    if name == BROADCAST_OFFSET:
        return _history(
            source,
            leader_position_m=(
                source.leader_position_m + BROADCAST_OFFSETS_M[None, :, :]
            ),
        )
    if name == DR_SCALE:
        return _history(
            source,
            dead_reckoned_displacement_m=(
                DR_SCALE_FACTOR * source.dead_reckoned_displacement_m
            ),
            follower_velocity_measured_mps=(
                DR_SCALE_FACTOR * source.follower_velocity_measured_mps
            ),
        )
    if name == DR_DRIFT:
        return _history(
            source,
            dead_reckoned_displacement_m=(
                source.dead_reckoned_displacement_m
                + source.t_s[:, None] * DR_DRIFT_VELOCITY_MPS[None, :]
            ),
            follower_velocity_measured_mps=(
                source.follower_velocity_measured_mps
                + DR_DRIFT_VELOCITY_MPS[None, :]
            ),
        )
    raise AssertionError("unreachable V28 stress dispatch")


def history_sha256(history: v19.OnlineDopplerHistory) -> str:
    digest = hashlib.sha256()
    for name in (
        "t_s",
        "dead_reckoned_displacement_m",
        "leader_position_m",
        "leader_velocity_mps",
        "follower_velocity_measured_mps",
        "doppler_measured_mps",
        "historical_pf_gate_factor",
    ):
        array = np.ascontiguousarray(getattr(history, name), dtype=np.float64)
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def condition_contract() -> Dict[str, Any]:
    return {
        "conditions": list(CONDITIONS),
        "condition_codes": dict(CONDITION_CODE),
        "base_design_seed": BASE_DESIGN_SEED,
        "doppler_common_bias_mps": [0.03, 0.03],
        "doppler_differential_bias_mps": [0.03, -0.03],
        "doppler_scale": 1.02,
        "colored_rho": COLORED_RHO,
        "colored_stationary_sd_mps": COLORED_STATIONARY_SD_MPS,
        "dropout_probability": DROPOUT_PROBABILITY,
        "dropout_forced_times_s": [1.0, 120.0, 440.0],
        "broadcast_delay_s": BROADCAST_DELAY_S,
        "broadcast_offsets_m": BROADCAST_OFFSETS_M.tolist(),
        "dead_reckoning_scale": DR_SCALE_FACTOR,
        "dead_reckoning_drift_velocity_mps": DR_DRIFT_VELOCITY_MPS.tolist(),
    }


__all__ = [
    "VERSION",
    "CONDITIONS",
    "NOMINAL",
    "DOPPLER_COMMON_BIAS",
    "DOPPLER_DIFFERENTIAL_BIAS",
    "DOPPLER_SCALE",
    "COLORED_NOISE",
    "DROPOUT",
    "BROADCAST_DELAY",
    "BROADCAST_OFFSET",
    "DR_SCALE",
    "DR_DRIFT",
    "apply_stress",
    "history_sha256",
    "condition_contract",
]
