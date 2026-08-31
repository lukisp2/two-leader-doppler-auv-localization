#!/usr/bin/env python3
"""V19 observability and estimator benchmark primitives.

V19 deliberately contains no reinforcement-learning training code.  It
replays frozen V18.1 plant actions and exogenous-noise tapes, captures the
online Doppler/odometry history, and solves a low-dimensional multi-start
batch problem for the unknown initial follower position.

Simulator truth is kept in :class:`ReplayTruthDiagnostics`; the estimator
accepts only :class:`OnlineDopplerHistory`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


VERSION = "v19_observability_estimator_benchmark_1.0"
SEALED_FINAL_SEED_START = 50_000
SEALED_FINAL_SEED_END = 50_999
V181_DEV_SEED_START = 45_000
V181_DEV_SEED_END = 45_099
PRIMARY_CONTROLLER = "deterministic_greedy_grid"
REPLAY_TOLERANCE = 1e-12
CHI2_3_95_SQRT = 2.7954834829151074
FROZEN_V181_ENVIRONMENT_SOURCES = (
    "uuv_v8_temporal_infofix.py",
    "uuv_v10_info_tracking.py",
    "uuv_v11_online.py",
    "uuv_v11_rng.py",
    "uuv_v12_hybrid.py",
    "uuv_v13_info_ray.py",
    "uuv_v14_info_ray.py",
    "uuv_v15_contextual_gain.py",
    "uuv_v16_temporal_escalation.py",
    "uuv_v17_bounded_escalation.py",
    "uuv_v18_resampling_guard.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_seed_is_not_sealed_final(seed: int) -> None:
    value = int(seed)
    if SEALED_FINAL_SEED_START <= value <= SEALED_FINAL_SEED_END:
        raise PermissionError(
            f"V19 refuses sealed final seed {value}; "
            f"{SEALED_FINAL_SEED_START}..{SEALED_FINAL_SEED_END} remain unauthorized"
        )


def circular_difference_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def initial_leader_centroid_from_history(
    history: "OnlineDopplerHistory",
) -> np.ndarray:
    """Recover the known reset-time leader centroid from online broadcasts.

    The frozen V18 plant uses constant leader velocities.  Rewinding the first
    one-second broadcast avoids reading the simulator-truth archive merely to
    obtain the known centre of the initialization shell.
    """

    first_time = float(history.t_s[0])
    initial_leaders = (
        history.leader_position_m[0]
        - first_time * history.leader_velocity_mps[0]
    )
    center = np.mean(initial_leaders, axis=0)
    if not np.all(np.isfinite(center)):
        raise ValueError("inferred initial leader centroid is non-finite")
    return np.asarray(center, dtype=np.float64)


def _finite_array(name: str, value: Any, shape: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class OnlineDopplerHistory:
    """The deployable inputs of the V19 initial-position estimator."""

    t_s: np.ndarray
    dead_reckoned_displacement_m: np.ndarray
    leader_position_m: np.ndarray
    leader_velocity_mps: np.ndarray
    follower_velocity_measured_mps: np.ndarray
    doppler_measured_mps: np.ndarray
    historical_pf_gate_factor: np.ndarray

    def __post_init__(self) -> None:
        t_s = np.asarray(self.t_s, dtype=np.float64)
        if t_s.ndim != 1 or t_s.size == 0:
            raise ValueError("t_s must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(t_s)) or np.any(np.diff(t_s) <= 0.0):
            raise ValueError("measurement times must be finite and strictly increasing")
        n = int(t_s.size)
        arrays = {
            "dead_reckoned_displacement_m": _finite_array(
                "dead_reckoned_displacement_m",
                self.dead_reckoned_displacement_m,
                (n, 3),
            ),
            "leader_position_m": _finite_array(
                "leader_position_m", self.leader_position_m, (n, 2, 3)
            ),
            "leader_velocity_mps": _finite_array(
                "leader_velocity_mps", self.leader_velocity_mps, (n, 2, 3)
            ),
            "follower_velocity_measured_mps": _finite_array(
                "follower_velocity_measured_mps",
                self.follower_velocity_measured_mps,
                (n, 3),
            ),
            "doppler_measured_mps": _finite_array(
                "doppler_measured_mps", self.doppler_measured_mps, (n, 2)
            ),
            "historical_pf_gate_factor": _finite_array(
                "historical_pf_gate_factor",
                self.historical_pf_gate_factor,
                (n, 2),
            ),
        }
        if np.any(arrays["historical_pf_gate_factor"] < 0.0) or np.any(
            arrays["historical_pf_gate_factor"] > 1.0
        ):
            raise ValueError("historical PF gate factors must lie in [0, 1]")
        object.__setattr__(self, "t_s", t_s)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @property
    def measurement_count(self) -> int:
        return int(self.t_s.size)

    def prefix(self, end_time_s: float) -> "OnlineDopplerHistory":
        end = float(end_time_s)
        mask = self.t_s <= end + 1e-12
        if not np.any(mask):
            raise ValueError(f"no Doppler measurements at or before {end:g} s")
        return OnlineDopplerHistory(
            t_s=self.t_s[mask].copy(),
            dead_reckoned_displacement_m=self.dead_reckoned_displacement_m[mask].copy(),
            leader_position_m=self.leader_position_m[mask].copy(),
            leader_velocity_mps=self.leader_velocity_mps[mask].copy(),
            follower_velocity_measured_mps=self.follower_velocity_measured_mps[mask].copy(),
            doppler_measured_mps=self.doppler_measured_mps[mask].copy(),
            historical_pf_gate_factor=self.historical_pf_gate_factor[mask].copy(),
        )

    def first_fraction(self, fraction: float) -> "OnlineDopplerHistory":
        value = float(fraction)
        if not 0.0 < value <= 1.0:
            raise ValueError("fraction must lie in (0, 1]")
        count = max(1, int(math.floor(value * self.measurement_count)))
        return self.take(slice(0, count))

    def take(self, selection: Any) -> "OnlineDopplerHistory":
        return OnlineDopplerHistory(
            t_s=np.atleast_1d(self.t_s[selection]).copy(),
            dead_reckoned_displacement_m=np.atleast_2d(
                self.dead_reckoned_displacement_m[selection]
            ).copy(),
            leader_position_m=np.asarray(self.leader_position_m[selection])[
                None, ...
            ].copy()
            if np.asarray(self.leader_position_m[selection]).ndim == 2
            else np.asarray(self.leader_position_m[selection]).copy(),
            leader_velocity_mps=np.asarray(self.leader_velocity_mps[selection])[
                None, ...
            ].copy()
            if np.asarray(self.leader_velocity_mps[selection]).ndim == 2
            else np.asarray(self.leader_velocity_mps[selection]).copy(),
            follower_velocity_measured_mps=np.atleast_2d(
                self.follower_velocity_measured_mps[selection]
            ).copy(),
            doppler_measured_mps=np.atleast_2d(
                self.doppler_measured_mps[selection]
            ).copy(),
            historical_pf_gate_factor=np.atleast_2d(
                self.historical_pf_gate_factor[selection]
            ).copy(),
        )


@dataclass(frozen=True)
class ReplayTruthDiagnostics:
    """Simulator-only values kept outside the estimator input boundary."""

    initial_follower_position_m: np.ndarray
    initial_leader_centroid_m: np.ndarray
    true_displacement_m: np.ndarray
    follower_velocity_true_mps: np.ndarray

    def __post_init__(self) -> None:
        p0 = _finite_array(
            "initial_follower_position_m", self.initial_follower_position_m, (3,)
        )
        center = _finite_array(
            "initial_leader_centroid_m", self.initial_leader_centroid_m, (3,)
        )
        displacement = np.asarray(self.true_displacement_m, dtype=np.float64)
        velocity = np.asarray(self.follower_velocity_true_mps, dtype=np.float64)
        if displacement.ndim != 2 or displacement.shape[1] != 3:
            raise ValueError("true_displacement_m must have shape (N, 3)")
        if velocity.shape != displacement.shape:
            raise ValueError("true velocity must match true displacement shape")
        if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(velocity)):
            raise ValueError("truth diagnostics contain non-finite values")
        object.__setattr__(self, "initial_follower_position_m", p0)
        object.__setattr__(self, "initial_leader_centroid_m", center)
        object.__setattr__(self, "true_displacement_m", displacement)
        object.__setattr__(self, "follower_velocity_true_mps", velocity)


@dataclass(frozen=True)
class ReplayCapture:
    online: OnlineDopplerHistory
    truth: ReplayTruthDiagnostics
    episode_index: int
    episode_seed: int
    controller: str
    support_radius_min_m: float
    support_radius_max_m: float
    initial_pf_metrics: Mapping[str, float]
    frozen_pf_endpoint: Mapping[str, float]
    scenario_metrics: Mapping[str, float]
    replay_integrity: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class BatchEstimatorConfig:
    coarse_candidates: int = 4096
    coarse_sweeps: int = 2
    local_starts: int = 48
    maximum_modes: int = 12
    maximum_iterations: int = 80
    coarse_start_separation_m: float = 5.0
    mode_cluster_radius_m: float = 1.0
    alternative_separation_m: float = 7.0
    initial_damping: float = 1e-3
    gradient_tolerance: float = 1e-10
    step_tolerance_m: float = 1e-8
    measurement_sigma_mps: float = 0.05
    gate_mode: str = "raw"
    candidate_radial_distribution: str = "uniform_radius"

    def __post_init__(self) -> None:
        if int(self.coarse_candidates) < 64:
            raise ValueError("coarse_candidates must be at least 64")
        if int(self.coarse_sweeps) < 1:
            raise ValueError("coarse_sweeps must be positive")
        if int(self.local_starts) < 2:
            raise ValueError("local_starts must be at least 2")
        if int(self.maximum_modes) < 2:
            raise ValueError("maximum_modes must be at least 2")
        if int(self.maximum_iterations) < 1:
            raise ValueError("maximum_iterations must be positive")
        if self.gate_mode not in {"raw", "historical_soft"}:
            raise ValueError("gate_mode must be raw or historical_soft")
        if self.candidate_radial_distribution not in {
            "uniform_radius",
            "uniform_volume",
        }:
            raise ValueError(
                "candidate_radial_distribution must be uniform_radius or uniform_volume"
            )
        positive = (
            self.coarse_start_separation_m,
            self.mode_cluster_radius_m,
            self.alternative_separation_m,
            self.initial_damping,
            self.gradient_tolerance,
            self.step_tolerance_m,
            self.measurement_sigma_mps,
        )
        if any((not np.isfinite(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("batch-estimator numeric constants must be finite and positive")


@dataclass(frozen=True)
class BatchMode:
    initial_position_m: np.ndarray
    residual_sse_mps2: float
    residual_rmse_mps: float
    iterations: int
    converged: bool
    hessian_eigenvalues: np.ndarray
    hessian_rank: int
    hessian_condition_number: float
    local_covariance_m2: np.ndarray
    local_covariance_valid: bool
    local_radius95_m: float

    def to_dict(self) -> Dict[str, Any]:
        covariance = np.asarray(self.local_covariance_m2, dtype=np.float64)
        return {
            "initial_position_m": np.asarray(self.initial_position_m).tolist(),
            "residual_sse_mps2": float(self.residual_sse_mps2),
            "residual_rmse_mps": float(self.residual_rmse_mps),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "hessian_eigenvalues": np.asarray(self.hessian_eigenvalues).tolist(),
            "hessian_rank": int(self.hessian_rank),
            "hessian_condition_number": (
                float(self.hessian_condition_number)
                if np.isfinite(self.hessian_condition_number)
                else None
            ),
            "local_covariance_m2": (
                covariance.tolist()
                if np.all(np.isfinite(covariance))
                else None
            ),
            "local_covariance_valid": bool(self.local_covariance_valid),
            "local_radius95_m": (
                float(self.local_radius95_m)
                if np.isfinite(self.local_radius95_m)
                else None
            ),
        }


@dataclass(frozen=True)
class BatchEstimate:
    modes: Tuple[BatchMode, ...]
    runtime_s: float
    coarse_best_rmse_mps: float
    candidate_count: int
    refined_start_count: int
    clustered_mode_count: int
    alternative_mode_index: Optional[int]
    alternative_distance_m: Optional[float]
    alternative_delta_sse_mps2: Optional[float]
    alternative_delta_chi2: Optional[float]

    @property
    def best(self) -> BatchMode:
        if not self.modes:
            raise RuntimeError("batch estimate contains no modes")
        return self.modes[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime_s": float(self.runtime_s),
            "coarse_best_rmse_mps": float(self.coarse_best_rmse_mps),
            "candidate_count": int(self.candidate_count),
            "refined_start_count": int(self.refined_start_count),
            "mode_count": len(self.modes),
            "clustered_mode_count": int(self.clustered_mode_count),
            "alternative_mode_index": self.alternative_mode_index,
            "alternative_distance_m": self.alternative_distance_m,
            "alternative_delta_sse_mps2": self.alternative_delta_sse_mps2,
            "alternative_delta_chi2": self.alternative_delta_chi2,
            "modes": [mode.to_dict() for mode in self.modes],
        }


def _measurement_weights(
    history: OnlineDopplerHistory, config: BatchEstimatorConfig
) -> np.ndarray:
    if config.gate_mode == "raw":
        return np.ones_like(history.doppler_measured_mps, dtype=np.float64)
    return np.clip(history.historical_pf_gate_factor, 1e-3, 1.0)


def predict_doppler(
    initial_position_m: np.ndarray, history: OnlineDopplerHistory
) -> np.ndarray:
    position = np.asarray(initial_position_m, dtype=np.float64).reshape(3)
    follower = position[None, :] + history.dead_reckoned_displacement_m
    relative_position = history.leader_position_m - follower[:, None, :]
    rho = np.maximum(np.linalg.norm(relative_position, axis=2), 1e-12)
    line_of_sight = relative_position / rho[:, :, None]
    relative_velocity = (
        history.leader_velocity_mps
        - history.follower_velocity_measured_mps[:, None, :]
    )
    return -np.sum(line_of_sight * relative_velocity, axis=2)


def predict_doppler_many(
    initial_positions_m: np.ndarray,
    history: OnlineDopplerHistory,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    candidates = np.asarray(initial_positions_m, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("initial_positions_m must have shape (M, 3)")
    outputs: List[np.ndarray] = []
    for start in range(0, candidates.shape[0], max(1, int(chunk_size))):
        candidate = candidates[start : start + max(1, int(chunk_size))]
        follower = (
            candidate[:, None, None, :]
            + history.dead_reckoned_displacement_m[None, :, None, :]
        )
        relative_position = history.leader_position_m[None, :, :, :] - follower
        rho = np.maximum(np.linalg.norm(relative_position, axis=3), 1e-12)
        line_of_sight = relative_position / rho[:, :, :, None]
        relative_velocity = (
            history.leader_velocity_mps[None, :, :, :]
            - history.follower_velocity_measured_mps[None, :, None, :]
        )
        outputs.append(-np.sum(line_of_sight * relative_velocity, axis=3))
    return np.concatenate(outputs, axis=0)


def residual_and_jacobian(
    initial_position_m: np.ndarray,
    history: OnlineDopplerHistory,
    config: BatchEstimatorConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    position = np.asarray(initial_position_m, dtype=np.float64).reshape(3)
    follower = position[None, :] + history.dead_reckoned_displacement_m
    relative_position = history.leader_position_m - follower[:, None, :]
    rho = np.maximum(np.linalg.norm(relative_position, axis=2), 1e-12)
    line_of_sight = relative_position / rho[:, :, None]
    relative_velocity = (
        history.leader_velocity_mps
        - history.follower_velocity_measured_mps[:, None, :]
    )
    projection = np.sum(line_of_sight * relative_velocity, axis=2)
    prediction = -projection
    residual = history.doppler_measured_mps - prediction

    # The residual derivative with respect to follower p0 equals the Doppler
    # Jacobian written with respect to the leader-minus-follower relative vector.
    jacobian = -(
        relative_velocity - projection[:, :, None] * line_of_sight
    ) / rho[:, :, None]
    sqrt_weight = np.sqrt(_measurement_weights(history, config))
    return (
        (sqrt_weight * residual).reshape(-1),
        (sqrt_weight[:, :, None] * jacobian).reshape(-1, 3),
    )


def deterministic_shell_candidates(
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    count: int,
    seed: int,
    *,
    radial_distribution: str = "uniform_radius",
) -> np.ndarray:
    center = np.asarray(center_m, dtype=np.float64).reshape(3)
    radius_min = float(radius_min_m)
    radius_max = float(radius_max_m)
    n = int(count)
    if not (0.0 <= radius_min < radius_max):
        raise ValueError("invalid shell radii")
    if n < 1:
        raise ValueError("candidate count must be positive")
    if radial_distribution not in {"uniform_radius", "uniform_volume"}:
        raise ValueError("unknown radial distribution")
    def radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
        values = np.zeros(indices.shape, dtype=np.float64)
        factor = 1.0 / float(base)
        work = indices.astype(np.int64).copy()
        while np.any(work > 0):
            values += factor * (work % base)
            work //= base
            factor /= float(base)
        return values

    # A shifted 3-D Halton design provides deterministic, stratified support
    # without adding a SciPy/Sobol dependency to the frozen NumPy runtime.
    indices = np.arange(1, n + 1, dtype=np.int64)
    unit = np.column_stack(
        (
            radical_inverse(indices, 2),
            radical_inverse(indices, 3),
            radical_inverse(indices, 5),
        )
    )
    shift = np.random.default_rng(int(seed)).random(3)
    unit = np.mod(unit + shift[None, :], 1.0)
    u = unit[:, 0]
    if radial_distribution == "uniform_radius":
        radius = radius_min + u * (radius_max - radius_min)
    else:
        radius = (
            radius_min**3 + u * (radius_max**3 - radius_min**3)
        ) ** (1.0 / 3.0)
    azimuth = 2.0 * math.pi * unit[:, 1]
    z_component = 2.0 * unit[:, 2] - 1.0
    horizontal = np.sqrt(np.maximum(1.0 - z_component * z_component, 0.0))
    direction = np.column_stack(
        (
            horizontal * np.cos(azimuth),
            horizontal * np.sin(azimuth),
            z_component,
        )
    )
    return center[None, :] + direction * radius[:, None]


def project_to_shell(
    position_m: np.ndarray,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
) -> np.ndarray:
    position = np.asarray(position_m, dtype=np.float64).reshape(3)
    center = np.asarray(center_m, dtype=np.float64).reshape(3)
    offset = position - center
    radius = float(np.linalg.norm(offset))
    if radius <= 1e-12:
        return center + np.array([float(radius_min_m), 0.0, 0.0])
    clipped = float(np.clip(radius, float(radius_min_m), float(radius_max_m)))
    return center + offset * (clipped / radius)


def _mode_from_solution(
    position_m: np.ndarray,
    history: OnlineDopplerHistory,
    config: BatchEstimatorConfig,
    iterations: int,
    converged: bool,
    *,
    center_m: Optional[np.ndarray] = None,
    radius_min_m: Optional[float] = None,
    radius_max_m: Optional[float] = None,
) -> BatchMode:
    residual, jacobian = residual_and_jacobian(position_m, history, config)
    sse = float(residual @ residual)
    rmse = float(math.sqrt(sse / max(residual.size, 1)))
    hessian = jacobian.T @ jacobian
    hessian = 0.5 * (hessian + hessian.T)
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(3, dtype=np.float64)
    largest_eigenvalue = float(max(np.max(eigenvalues), 0.0))
    rank_threshold = max(1e-12, 1e-10 * largest_eigenvalue)
    hessian_rank = int(np.sum(eigenvalues > rank_threshold))
    if hessian_rank == 3 and float(eigenvalues[0]) > 0.0:
        hessian_condition = float(largest_eigenvalue / float(eigenvalues[0]))
    else:
        hessian_condition = float("inf")
    constrained_boundary = False
    if (
        center_m is not None
        and radius_min_m is not None
        and radius_max_m is not None
    ):
        solution_radius = float(
            np.linalg.norm(
                np.asarray(position_m, dtype=np.float64)
                - np.asarray(center_m, dtype=np.float64)
            )
        )
        boundary_tolerance = max(1e-6, 1e-8 * float(radius_max_m))
        constrained_boundary = (
            abs(solution_radius - float(radius_min_m)) <= boundary_tolerance
            or abs(solution_radius - float(radius_max_m)) <= boundary_tolerance
        )
    covariance_valid = bool(
        hessian_rank == 3
        and np.isfinite(hessian_condition)
        and hessian_condition <= 1e10
        and not constrained_boundary
    )
    dof = max(int(residual.size) - 3, 1)
    empirical_variance = max(sse / dof, float(config.measurement_sigma_mps) ** 2)
    if covariance_valid:
        covariance = empirical_variance * np.linalg.inv(hessian)
        covariance = 0.5 * (covariance + covariance.T)
        try:
            maximum_variance = float(
                max(np.max(np.linalg.eigvalsh(covariance)), 0.0)
            )
        except np.linalg.LinAlgError:
            maximum_variance = float("inf")
        radius95 = CHI2_3_95_SQRT * math.sqrt(maximum_variance)
        if not np.isfinite(radius95):
            covariance_valid = False
    else:
        covariance = np.full((3, 3), np.nan, dtype=np.float64)
        radius95 = float("inf")
    if not covariance_valid:
        covariance = np.full((3, 3), np.nan, dtype=np.float64)
        radius95 = float("inf")
    return BatchMode(
        initial_position_m=np.asarray(position_m, dtype=np.float64).copy(),
        residual_sse_mps2=sse,
        residual_rmse_mps=rmse,
        iterations=int(iterations),
        converged=bool(converged),
        hessian_eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        hessian_rank=hessian_rank,
        hessian_condition_number=hessian_condition,
        local_covariance_m2=np.asarray(covariance, dtype=np.float64),
        local_covariance_valid=covariance_valid,
        local_radius95_m=float(radius95),
    )


def refine_damped_gauss_newton(
    initial_position_m: np.ndarray,
    history: OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: BatchEstimatorConfig,
) -> BatchMode:
    position = project_to_shell(
        initial_position_m, center_m, radius_min_m, radius_max_m
    )
    damping = float(config.initial_damping)
    converged = False
    iterations = 0
    for iterations in range(1, int(config.maximum_iterations) + 1):
        residual, jacobian = residual_and_jacobian(position, history, config)
        cost = 0.5 * float(residual @ residual)
        gradient = jacobian.T @ residual
        if float(np.linalg.norm(gradient, ord=np.inf)) <= float(
            config.gradient_tolerance
        ):
            converged = True
            break
        normal = jacobian.T @ jacobian
        scale = np.maximum(np.diag(normal), 1e-12)
        system = normal + damping * np.diag(scale)
        try:
            step = np.linalg.solve(system, -gradient)
        except np.linalg.LinAlgError:
            damping = min(damping * 10.0, 1e18)
            continue
        if float(np.linalg.norm(step)) <= float(config.step_tolerance_m):
            converged = True
            break
        trial = project_to_shell(
            position + step, center_m, radius_min_m, radius_max_m
        )
        trial_residual, _ = residual_and_jacobian(trial, history, config)
        trial_cost = 0.5 * float(trial_residual @ trial_residual)
        if trial_cost < cost:
            improvement = cost - trial_cost
            position = trial
            damping = max(damping * 0.3, 1e-15)
            if improvement <= 1e-14 * max(1.0, cost):
                converged = True
                break
        else:
            damping = min(damping * 10.0, 1e18)
    return _mode_from_solution(
        position,
        history,
        config,
        iterations,
        converged,
        center_m=np.asarray(center_m, dtype=np.float64),
        radius_min_m=float(radius_min_m),
        radius_max_m=float(radius_max_m),
    )


def _coarse_weighted_sse(
    candidates: np.ndarray,
    history: OnlineDopplerHistory,
    config: BatchEstimatorConfig,
) -> np.ndarray:
    prediction = predict_doppler_many(candidates, history)
    residual = history.doppler_measured_mps[None, :, :] - prediction
    weights = _measurement_weights(history, config)[None, :, :]
    return np.sum(weights * residual * residual, axis=(1, 2))


def _select_spatially_separated_starts(
    candidates: np.ndarray,
    costs: np.ndarray,
    count: int,
    separation_m: float,
) -> List[np.ndarray]:
    selected: List[np.ndarray] = []
    for index in np.argsort(np.asarray(costs, dtype=np.float64)):
        candidate = np.asarray(candidates[int(index)], dtype=np.float64)
        if all(
            float(np.linalg.norm(candidate - previous)) >= float(separation_m)
            for previous in selected
        ):
            selected.append(candidate.copy())
        if len(selected) >= int(count):
            break
    if len(selected) < 2:
        raise RuntimeError("coarse search did not produce enough separated starts")
    return selected


def _cluster_refined_modes(
    modes: Iterable[BatchMode], config: BatchEstimatorConfig
) -> Tuple[BatchMode, ...]:
    kept: List[BatchMode] = []
    for mode in sorted(modes, key=lambda item: item.residual_sse_mps2):
        if all(
            float(np.linalg.norm(mode.initial_position_m - previous.initial_position_m))
            >= float(config.mode_cluster_radius_m)
            for previous in kept
        ):
            kept.append(mode)
    if not kept:
        raise RuntimeError("local refinement produced no modes")
    return tuple(kept)


def estimate_initial_position_multistart(
    history: OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: BatchEstimatorConfig,
    *,
    candidate_seed: int,
) -> BatchEstimate:
    started = time.perf_counter()
    candidates = np.concatenate(
        [
            deterministic_shell_candidates(
                center_m,
                radius_min_m,
                radius_max_m,
                int(config.coarse_candidates),
                int(candidate_seed) + 104_729 * sweep,
                radial_distribution=config.candidate_radial_distribution,
            )
            for sweep in range(int(config.coarse_sweeps))
        ],
        axis=0,
    )
    costs = _coarse_weighted_sse(candidates, history, config)
    starts = _select_spatially_separated_starts(
        candidates,
        costs,
        int(config.local_starts),
        float(config.coarse_start_separation_m),
    )
    refined = [
        refine_damped_gauss_newton(
            start,
            history,
            center_m,
            radius_min_m,
            radius_max_m,
            config,
        )
        for start in starts
    ]
    all_modes = _cluster_refined_modes(refined, config)
    best = all_modes[0]
    alternative_all_index: Optional[int] = None
    alternative_distance: Optional[float] = None
    alternative_delta_sse: Optional[float] = None
    alternative_delta_chi2: Optional[float] = None
    for index, mode in enumerate(all_modes[1:], start=1):
        distance = float(
            np.linalg.norm(mode.initial_position_m - best.initial_position_m)
        )
        if distance >= float(config.alternative_separation_m):
            alternative_all_index = index
            alternative_distance = distance
            alternative_delta_sse = float(
                mode.residual_sse_mps2 - best.residual_sse_mps2
            )
            alternative_delta_chi2 = float(
                alternative_delta_sse
                / max(float(config.measurement_sigma_mps) ** 2, 1e-18)
            )
            break
    reported_modes = list(all_modes[: int(config.maximum_modes)])
    alternative_index: Optional[int] = None
    if alternative_all_index is not None:
        alternative_mode = all_modes[alternative_all_index]
        try:
            alternative_index = next(
                index
                for index, mode in enumerate(reported_modes)
                if mode is alternative_mode
            )
        except StopIteration:
            if len(reported_modes) >= int(config.maximum_modes):
                reported_modes[-1] = alternative_mode
                alternative_index = len(reported_modes) - 1
            else:
                reported_modes.append(alternative_mode)
                alternative_index = len(reported_modes) - 1
    modes = tuple(reported_modes)
    coarse_best_rmse = float(
        math.sqrt(float(np.min(costs)) / (2 * history.measurement_count))
    )
    return BatchEstimate(
        modes=modes,
        runtime_s=float(time.perf_counter() - started),
        coarse_best_rmse_mps=coarse_best_rmse,
        candidate_count=int(candidates.shape[0]),
        refined_start_count=len(refined),
        clustered_mode_count=len(all_modes),
        alternative_mode_index=alternative_index,
        alternative_distance_m=alternative_distance,
        alternative_delta_sse_mps2=alternative_delta_sse,
        alternative_delta_chi2=alternative_delta_chi2,
    )


def structural_noiseless_history(capture: ReplayCapture) -> OnlineDopplerHistory:
    online = capture.online
    truth = capture.truth
    if truth.true_displacement_m.shape[0] != online.measurement_count:
        raise ValueError("truth history length differs from online history")
    provisional = OnlineDopplerHistory(
        t_s=online.t_s.copy(),
        dead_reckoned_displacement_m=truth.true_displacement_m.copy(),
        leader_position_m=online.leader_position_m.copy(),
        leader_velocity_mps=online.leader_velocity_mps.copy(),
        follower_velocity_measured_mps=truth.follower_velocity_true_mps.copy(),
        doppler_measured_mps=np.zeros_like(online.doppler_measured_mps),
        historical_pf_gate_factor=np.ones_like(online.historical_pf_gate_factor),
    )
    noiseless = predict_doppler(truth.initial_follower_position_m, provisional)
    return OnlineDopplerHistory(
        t_s=provisional.t_s,
        dead_reckoned_displacement_m=provisional.dead_reckoned_displacement_m,
        leader_position_m=provisional.leader_position_m,
        leader_velocity_mps=provisional.leader_velocity_mps,
        follower_velocity_measured_mps=provisional.follower_velocity_measured_mps,
        doppler_measured_mps=noiseless,
        historical_pf_gate_factor=provisional.historical_pf_gate_factor,
    )


def _trace_endpoint_values(trace: Mapping[str, np.ndarray]) -> Dict[str, float]:
    requested = (
        "localization_error_diagnostic_m",
        "pf_std_largest_eigen_raw_m",
        "nees_diagnostic",
        "formation_error_est_online_m",
        "formation_error_true_diagnostic_m",
        "fim_online_win_eig_min",
    )
    result: Dict[str, float] = {}
    for name in requested:
        if name not in trace:
            raise KeyError(f"frozen trace lacks {name!r}")
        result[name] = float(np.asarray(trace[name])[-1])
    return result


def _atomic_savez(path: Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)


def save_online_history(path: Path, history: OnlineDopplerHistory) -> None:
    """Write the strict estimator allowlist, with no seed, truth, or noise tape."""

    _atomic_savez(
        Path(path),
        t_s=history.t_s,
        dead_reckoned_displacement_m=history.dead_reckoned_displacement_m,
        leader_position_m=history.leader_position_m,
        leader_velocity_mps=history.leader_velocity_mps,
        follower_velocity_measured_mps=history.follower_velocity_measured_mps,
        doppler_measured_mps=history.doppler_measured_mps,
        historical_pf_gate_factor=history.historical_pf_gate_factor,
        schema_version=np.array(1, dtype=np.int64),
    )


def load_online_history(path: Path) -> OnlineDopplerHistory:
    allowed = {
        "t_s",
        "dead_reckoned_displacement_m",
        "leader_position_m",
        "leader_velocity_mps",
        "follower_velocity_measured_mps",
        "doppler_measured_mps",
        "historical_pf_gate_factor",
        "schema_version",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != allowed:
            raise ValueError("online history contains missing or forbidden fields")
        if int(archive["schema_version"].item()) != 1:
            raise ValueError("unsupported online-history schema")
        return OnlineDopplerHistory(
            t_s=archive["t_s"].copy(),
            dead_reckoned_displacement_m=archive[
                "dead_reckoned_displacement_m"
            ].copy(),
            leader_position_m=archive["leader_position_m"].copy(),
            leader_velocity_mps=archive["leader_velocity_mps"].copy(),
            follower_velocity_measured_mps=archive[
                "follower_velocity_measured_mps"
            ].copy(),
            doppler_measured_mps=archive["doppler_measured_mps"].copy(),
            historical_pf_gate_factor=archive[
                "historical_pf_gate_factor"
            ].copy(),
        )


def save_truth_diagnostics(path: Path, truth: ReplayTruthDiagnostics) -> None:
    _atomic_savez(
        Path(path),
        diagnostic_initial_follower_position_m=truth.initial_follower_position_m,
        diagnostic_initial_leader_centroid_m=truth.initial_leader_centroid_m,
        diagnostic_true_displacement_m=truth.true_displacement_m,
        diagnostic_follower_velocity_true_mps=truth.follower_velocity_true_mps,
        schema_version=np.array(1, dtype=np.int64),
    )


def load_truth_diagnostics(path: Path) -> ReplayTruthDiagnostics:
    required = {
        "diagnostic_initial_follower_position_m",
        "diagnostic_initial_leader_centroid_m",
        "diagnostic_true_displacement_m",
        "diagnostic_follower_velocity_true_mps",
        "schema_version",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("truth-label archive has an invalid schema")
        if int(archive["schema_version"].item()) != 1:
            raise ValueError("unsupported truth-label schema")
        return ReplayTruthDiagnostics(
            initial_follower_position_m=archive[
                "diagnostic_initial_follower_position_m"
            ].copy(),
            initial_leader_centroid_m=archive[
                "diagnostic_initial_leader_centroid_m"
            ].copy(),
            true_displacement_m=archive["diagnostic_true_displacement_m"].copy(),
            follower_velocity_true_mps=archive[
                "diagnostic_follower_velocity_true_mps"
            ].copy(),
        )


def _capture_metadata(capture: ReplayCapture) -> Dict[str, Any]:
    metadata = {
        "version": VERSION,
        "episode_index": capture.episode_index,
        "episode_seed": capture.episode_seed,
        "controller": capture.controller,
        "support_radius_min_m": capture.support_radius_min_m,
        "support_radius_max_m": capture.support_radius_max_m,
        "initial_pf_metrics": dict(capture.initial_pf_metrics),
        "frozen_pf_endpoint": dict(capture.frozen_pf_endpoint),
        "scenario_metrics": dict(capture.scenario_metrics),
        "replay_integrity": dict(capture.replay_integrity),
        "provenance": dict(capture.provenance),
        "truth_fields_are_diagnostic_only": True,
    }
    return metadata


def save_replay_capture(directory: Path, capture: ReplayCapture) -> Mapping[str, str]:
    """Persist online inputs and truth labels in physically separate archives."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    online_path = root / "online_inputs.npz"
    truth_path = root / "truth_labels.npz"
    metadata_path = root / "capture_metadata.json"
    save_online_history(online_path, capture.online)
    save_truth_diagnostics(truth_path, capture.truth)
    metadata = _capture_metadata(capture)
    metadata["online_inputs_sha256"] = sha256_file(online_path)
    metadata["truth_labels_sha256"] = sha256_file(truth_path)
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, metadata_path)
    return {
        "online_inputs": str(online_path),
        "truth_labels": str(truth_path),
        "metadata": str(metadata_path),
    }


def load_replay_capture(directory: Path) -> ReplayCapture:
    root = Path(directory)
    metadata = json.loads((root / "capture_metadata.json").read_text(encoding="utf-8"))
    online_path = root / "online_inputs.npz"
    truth_path = root / "truth_labels.npz"
    if sha256_file(online_path) != str(metadata["online_inputs_sha256"]):
        raise ValueError("online-input archive digest mismatch")
    if sha256_file(truth_path) != str(metadata["truth_labels_sha256"]):
        raise ValueError("truth-label archive digest mismatch")
    online = load_online_history(online_path)
    truth = load_truth_diagnostics(truth_path)
    return ReplayCapture(
        online=online,
        truth=truth,
        episode_index=int(metadata["episode_index"]),
        episode_seed=int(metadata["episode_seed"]),
        controller=str(metadata["controller"]),
        support_radius_min_m=float(metadata["support_radius_min_m"]),
        support_radius_max_m=float(metadata["support_radius_max_m"]),
        initial_pf_metrics=dict(metadata["initial_pf_metrics"]),
        frozen_pf_endpoint=dict(metadata["frozen_pf_endpoint"]),
        scenario_metrics=dict(metadata["scenario_metrics"]),
        replay_integrity=dict(metadata["replay_integrity"]),
        provenance=dict(metadata["provenance"]),
    )


def capture_v181_episode(
    evaluation_directory: Path,
    episode_index: int,
    episode_seed: int,
    *,
    controller: str = PRIMARY_CONTROLLER,
    replay_tolerance: float = REPLAY_TOLERANCE,
) -> ReplayCapture:
    """Replay one frozen V18.1 episode and capture every PF measurement call."""

    if str(controller) != PRIMARY_CONTROLLER:
        raise PermissionError(
            f"V19 primary benchmark accepts only {PRIMARY_CONTROLLER!r}"
        )
    if not 0 <= int(episode_index) <= 99:
        raise ValueError("V19 episode index must lie in the frozen dev100 block")
    assert_seed_is_not_sealed_final(episode_seed)
    if int(episode_seed) != V181_DEV_SEED_START + int(episode_index):
        raise ValueError("V18.1 development seed/index relation is inconsistent")
    evaluation_directory = Path(evaluation_directory).resolve()
    metadata_path = evaluation_directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing frozen evaluation metadata: {metadata_path}")
    evaluation_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    seed_protocol = evaluation_metadata.get("seed_protocol", {})
    if str(seed_protocol.get("mode", "")) != "dev":
        raise PermissionError("V19 accepts only the frozen V18.1 development campaign")
    if str(seed_protocol.get("formula", "")) != (
        "episode_seed = 45000 + episode_index"
    ):
        raise ValueError("unexpected V18.1 development seed formula")
    evaluation_options = evaluation_metadata.get("evaluation_options", {})
    if (
        str(evaluation_options.get("mode", "")) != "dev"
        or int(evaluation_options.get("episodes", -1)) != 100
        or PRIMARY_CONTROLLER not in evaluation_options.get("controllers", [])
        or "/seed_28001/" not in str(evaluation_options.get("model_path", ""))
    ):
        raise ValueError("evaluation directory is not the frozen seed-28001 dev100 source")

    frozen_source_hashes = evaluation_metadata.get("source_sha256", {})
    source_root = Path(__file__).resolve().parent
    checked_source_hashes: Dict[str, str] = {}
    for filename in FROZEN_V181_ENVIRONMENT_SOURCES:
        source_path = source_root / filename
        expected_hash = frozen_source_hashes.get(filename)
        if expected_hash is None:
            raise ValueError(f"frozen metadata lacks source hash for {filename}")
        actual_hash = sha256_file(source_path)
        if actual_hash != str(expected_hash):
            raise RuntimeError(
                f"frozen environment source drift for {filename}: "
                f"{actual_hash} != {expected_hash}"
            )
        checked_source_hashes[filename] = actual_hash

    trace_path = (
        evaluation_directory
        / "traces_npz"
        / str(controller)
        / f"episode_{int(episode_index):04d}_seed_{int(episode_seed)}.npz"
    )
    tape_path = (
        evaluation_directory
        / "noise_tapes"
        / f"episode_{int(episode_index):04d}_seed_{int(episode_seed)}.npz"
    )
    if not trace_path.is_file() or not tape_path.is_file():
        raise FileNotFoundError(
            f"missing frozen trace/tape for episode {episode_index}, seed {episode_seed}"
        )

    # Imports are local so the pure estimator mathematics remains testable in
    # a minimal NumPy-only process.
    from uuv_v11_rng import ExogenousNoiseTape
    from uuv_v18_resampling_guard import UUV3DConfig, UUVTwoLeader3DPFEnv

    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    required_actions = (
        "applied_action_speed",
        "applied_action_yaw",
        "applied_action_pitch",
    )
    if any(name not in trace for name in required_actions):
        raise KeyError("frozen trace lacks applied plant actions")
    steps = int(np.asarray(trace[required_actions[0]]).size)
    if steps != 220:
        raise ValueError(f"frozen V18.1 trace has {steps} actions, expected 220")
    for name in required_actions:
        values = np.asarray(trace[name], dtype=np.float64)
        if values.shape != (steps,) or not np.all(np.isfinite(values)):
            raise ValueError(f"frozen action field {name!r} is invalid")
    if not np.array_equal(np.asarray(trace.get("step")), np.arange(1, steps + 1)):
        raise ValueError("frozen trace step index is not exactly 1..220")
    if not np.allclose(
        np.asarray(trace.get("t_s"), dtype=np.float64),
        2.0 * np.arange(1, steps + 1, dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("frozen trace time grid differs from 2..440 s")
    if not np.all(np.asarray(trace.get("episode_index")) == int(episode_index)):
        raise ValueError("trace episode index differs from requested episode")
    if not np.all(np.asarray(trace.get("episode_seed")) == int(episode_seed)):
        raise ValueError("trace episode seed differs from requested seed")
    if not np.all(np.asarray(trace.get("controller")) == str(controller)):
        raise ValueError("trace controller differs from requested controller")

    tape = ExogenousNoiseTape.load_npz(tape_path)
    if tape.n_substeps != 4400 or tape.n_doppler_measurements != 442:
        raise ValueError("noise-tape dimensions differ from the frozen V18.1 contract")
    tape_file_sha256 = sha256_file(tape_path)
    tape_content_sha256 = tape.content_sha256()
    tape_manifest_matches = [
        item
        for item in evaluation_metadata.get("noise_tapes", [])
        if int(item.get("episode_index", -1)) == int(episode_index)
        and int(item.get("episode_seed", -1)) == int(episode_seed)
    ]
    if len(tape_manifest_matches) != 1:
        raise ValueError("frozen metadata has no unique matching noise tape")
    tape_manifest = tape_manifest_matches[0]
    if tape_content_sha256 != str(tape_manifest.get("sha256", "")):
        raise RuntimeError("noise-tape content hash differs from frozen metadata")
    if not np.all(
        np.asarray(trace.get("noise_tape_sha256")) == tape_content_sha256
    ):
        raise RuntimeError("trace noise-tape hash differs from replay tape")

    frozen_environment_config = evaluation_metadata.get("environment_config")
    if not isinstance(frozen_environment_config, dict):
        raise ValueError("frozen metadata lacks environment_config")
    replay_environment_config = dict(frozen_environment_config)
    replay_environment_config.update(
        {
            # Saved values are already-applied three-channel plant actions.
            "v11_controller_id": "pid_track_exc",
            "v11_generate_noise_tape": False,
        }
    )
    cfg = UUV3DConfig(**replay_environment_config)
    reconstructed_config = asdict(cfg)
    for name, frozen_value in frozen_environment_config.items():
        if name == "v11_controller_id":
            continue
        if reconstructed_config.get(name) != frozen_value:
            raise RuntimeError(f"frozen environment config drift for {name}")
    env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    env.attach_exogenous_noise_tape(tape)
    env.reset(seed=int(episode_seed))

    initial_follower = np.asarray(env.pF, dtype=np.float64).copy()
    initial_centroid = 0.5 * (
        np.asarray(env.pL1, dtype=np.float64)
        + np.asarray(env.pL2, dtype=np.float64)
    )
    initial_particles = np.asarray(env.pf.p, dtype=np.float64).copy()
    initial_weights = np.asarray(env.pf.w, dtype=np.float64).copy()
    initial_distance = np.linalg.norm(
        initial_particles - initial_follower[None, :], axis=1
    )
    support_min = float(cfg.start_rho_min)
    support_max = float(cfg.start_rho_max)
    initial_pf_metrics = {
        "nearest_particle_distance_m": float(np.min(initial_distance)),
        "particle_count_within_7m": int(np.sum(initial_distance < 7.0)),
        "particle_count_within_20m": int(np.sum(initial_distance < 20.0)),
        "particle_mass_within_7m": float(np.sum(initial_weights[initial_distance < 7.0])),
        "particle_mass_within_20m": float(
            np.sum(initial_weights[initial_distance < 20.0])
        ),
        "truth_initial_radius_m": float(
            np.linalg.norm(initial_follower - initial_centroid)
        ),
        "pf_particle_radius_mean_m": float(
            np.mean(np.linalg.norm(initial_particles - initial_centroid[None, :], axis=1))
        ),
    }

    measurement_time: List[float] = []
    dr_displacement: List[np.ndarray] = []
    leader_position: List[np.ndarray] = []
    leader_velocity: List[np.ndarray] = []
    follower_velocity_measured: List[np.ndarray] = []
    doppler_measured: List[np.ndarray] = []
    gate_factor: List[np.ndarray] = []
    true_displacement: List[np.ndarray] = []
    follower_velocity_true: List[np.ndarray] = []
    accumulated_dr = np.zeros(3, dtype=np.float64)

    original_predict = env.pf.predict
    original_update = env.pf.update_doppler

    def capture_predict(vF_meas: np.ndarray, dt: float) -> Any:
        accumulated_dr[:] += np.asarray(vF_meas, dtype=np.float64) * float(dt)
        return original_predict(vF_meas, dt)

    def capture_update(
        pL_list: List[np.ndarray],
        vL_list: List[np.ndarray],
        vF_meas: np.ndarray,
        s_meas_list: List[Optional[float]],
        **kwargs: Any,
    ) -> Any:
        if any(value is None for value in s_meas_list):
            raise RuntimeError("V19 primary replay encountered a missing Doppler sample")
        gates = kwargs.get("gate_factors", [1.0, 1.0])
        measurement_time.append(float(env.next_s_time))
        dr_displacement.append(accumulated_dr.copy())
        leader_position.append(np.asarray(pL_list, dtype=np.float64).copy())
        leader_velocity.append(np.asarray(vL_list, dtype=np.float64).copy())
        follower_velocity_measured.append(
            np.asarray(vF_meas, dtype=np.float64).copy()
        )
        doppler_measured.append(np.asarray(s_meas_list, dtype=np.float64).copy())
        gate_factor.append(np.asarray(gates, dtype=np.float64).copy())
        true_displacement.append(
            np.asarray(env.pF, dtype=np.float64).copy() - initial_follower
        )
        follower_velocity_true.append(np.asarray(env.vF, dtype=np.float64).copy())
        return original_update(
            pL_list,
            vL_list,
            vF_meas,
            s_meas_list,
            **kwargs,
        )

    env.pf.predict = capture_predict  # type: ignore[method-assign]
    env.pf.update_doppler = capture_update  # type: ignore[method-assign]

    comparison_fields = (
        "pF_true_x",
        "pF_true_y",
        "pF_true_z",
        "pF_hat_x",
        "pF_hat_y",
        "pF_hat_z",
        "pf_cov_xx",
        "pf_cov_xy",
        "pf_cov_xz",
        "pf_cov_yy",
        "pf_cov_yz",
        "pf_cov_zz",
    )
    maximum_difference = {name: 0.0 for name in comparison_fields}
    final_noise_cursor_state: Dict[str, int] = {}
    try:
        for step in range(steps):
            action = np.asarray(
                [
                    trace["applied_action_speed"][step],
                    trace["applied_action_yaw"][step],
                    trace["applied_action_pitch"][step],
                ],
                dtype=np.float32,
            )
            env.step(action)
            replay_values = (
                *np.asarray(env.pF, dtype=np.float64),
                *np.asarray(env.pf.mean, dtype=np.float64),
                float(env.pf.cov[0, 0]),
                float(env.pf.cov[0, 1]),
                float(env.pf.cov[0, 2]),
                float(env.pf.cov[1, 1]),
                float(env.pf.cov[1, 2]),
                float(env.pf.cov[2, 2]),
            )
            for name, replay_value in zip(comparison_fields, replay_values):
                difference = abs(float(trace[name][step]) - float(replay_value))
                maximum_difference[name] = max(maximum_difference[name], difference)
                if difference > float(replay_tolerance):
                    raise RuntimeError(
                        f"V19 replay diverged at step {step + 1}, field {name}: "
                        f"difference {difference:.3e} exceeds {replay_tolerance:.3e}"
                    )
        cursor = env._v11_noise_cursor
        if cursor is None:
            raise RuntimeError("environment did not retain the attached noise tape")
        final_noise_cursor_state = dict(cursor.state_dict())
        expected_noise_cursor_state = {
            "substep_index": int(steps * round(cfg.action_dt / cfg.sub_dt)),
            "measurement_index": int(round(steps * cfg.action_dt / cfg.s_meas_period)),
        }
        if final_noise_cursor_state != expected_noise_cursor_state:
            raise RuntimeError(
                "replay consumed an unexpected amount of exogenous noise: "
                f"{final_noise_cursor_state} != {expected_noise_cursor_state}"
            )
    finally:
        env.close()

    online = OnlineDopplerHistory(
        t_s=np.asarray(measurement_time, dtype=np.float64),
        dead_reckoned_displacement_m=np.asarray(dr_displacement, dtype=np.float64),
        leader_position_m=np.asarray(leader_position, dtype=np.float64),
        leader_velocity_mps=np.asarray(leader_velocity, dtype=np.float64),
        follower_velocity_measured_mps=np.asarray(
            follower_velocity_measured, dtype=np.float64
        ),
        doppler_measured_mps=np.asarray(doppler_measured, dtype=np.float64),
        historical_pf_gate_factor=np.asarray(gate_factor, dtype=np.float64),
    )
    truth = ReplayTruthDiagnostics(
        initial_follower_position_m=initial_follower,
        initial_leader_centroid_m=initial_centroid,
        true_displacement_m=np.asarray(true_displacement, dtype=np.float64),
        follower_velocity_true_mps=np.asarray(
            follower_velocity_true, dtype=np.float64
        ),
    )
    final_duration = float(online.t_s[-1])
    expected_measurement_time = np.arange(
        float(cfg.s_meas_period),
        float(cfg.max_steps * cfg.action_dt) + 0.5 * float(cfg.s_meas_period),
        float(cfg.s_meas_period),
        dtype=np.float64,
    )
    if not np.allclose(
        online.t_s, expected_measurement_time, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("captured Doppler time grid differs from frozen contract")
    leader_baseline_initial = float(
        np.linalg.norm(
            online.leader_position_m[0, 1] - online.leader_position_m[0, 0]
        )
    )
    leader_baseline_final = float(
        np.linalg.norm(
            online.leader_position_m[-1, 1] - online.leader_position_m[-1, 0]
        )
    )
    initial_offset = initial_follower - initial_centroid
    initial_radius = float(np.linalg.norm(initial_offset))
    leader_initial_velocity = online.leader_velocity_mps[0]
    leader_initial_course_deg = np.degrees(
        np.arctan2(leader_initial_velocity[:, 1], leader_initial_velocity[:, 0])
    )
    scenario_metrics = {
        "leader_course_difference_deg": circular_difference_deg(
            float(leader_initial_course_deg[0]), float(leader_initial_course_deg[1])
        ),
        "leader_speed_difference_mps": abs(
            float(np.linalg.norm(leader_initial_velocity[0]))
            - float(np.linalg.norm(leader_initial_velocity[1]))
        ),
        "leader_relative_speed_mps": float(
            np.linalg.norm(
                online.leader_velocity_mps[0, 1]
                - online.leader_velocity_mps[0, 0]
            )
        ),
        "leader_baseline_initial_m": leader_baseline_initial,
        "leader_baseline_final_m": leader_baseline_final,
        "truth_initial_radius_m": initial_radius,
        "truth_initial_elevation_deg": float(
            np.degrees(np.arcsin(np.clip(initial_offset[2] / initial_radius, -1.0, 1.0)))
        ),
        "duration_s": final_duration,
    }
    replay_integrity = {
        "passed": True,
        "tolerance": float(replay_tolerance),
        "measurement_count": online.measurement_count,
        "action_count": steps,
        "noise_cursor_state": final_noise_cursor_state,
        "maximum_absolute_difference": maximum_difference,
    }
    provenance = {
        "evaluation_metadata_path": str(metadata_path),
        "evaluation_metadata_sha256": sha256_file(metadata_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "noise_tape_path": str(tape_path),
        "noise_tape_file_sha256": tape_file_sha256,
        "noise_tape_content_sha256": tape_content_sha256,
        "frozen_environment_source_sha256": checked_source_hashes,
        "environment_config_source": "frozen V18.1 metadata",
        "environment_config_overrides": {
            "v11_controller_id": "pid_track_exc direct applied-action path",
            "v11_generate_noise_tape": False,
        },
    }
    return ReplayCapture(
        online=online,
        truth=truth,
        episode_index=int(episode_index),
        episode_seed=int(episode_seed),
        controller=str(controller),
        support_radius_min_m=support_min,
        support_radius_max_m=support_max,
        initial_pf_metrics=initial_pf_metrics,
        frozen_pf_endpoint=_trace_endpoint_values(trace),
        scenario_metrics=scenario_metrics,
        replay_integrity=replay_integrity,
        provenance=provenance,
    )


def estimate_diagnostic_errors(
    estimate: BatchEstimate,
    history: OnlineDopplerHistory,
    truth: ReplayTruthDiagnostics,
) -> Dict[str, float]:
    best = estimate.best.initial_position_m
    initial_error = float(np.linalg.norm(best - truth.initial_follower_position_m))
    endpoint_estimate = best + history.dead_reckoned_displacement_m[-1]
    endpoint_measurement_index = int(round(float(history.t_s[-1]))) - 1
    if (
        endpoint_measurement_index < 0
        or endpoint_measurement_index >= truth.true_displacement_m.shape[0]
        or not math.isclose(
            float(history.t_s[-1]),
            float(endpoint_measurement_index + 1),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("history endpoint does not map to the one-second truth grid")
    endpoint_truth = (
        truth.initial_follower_position_m
        + truth.true_displacement_m[endpoint_measurement_index]
    )
    endpoint_error = float(np.linalg.norm(endpoint_estimate - endpoint_truth))
    result = {
        "initial_position_error_m": initial_error,
        "endpoint_position_error_m": endpoint_error,
    }
    if estimate.alternative_mode_index is not None:
        alternative = estimate.modes[estimate.alternative_mode_index]
        result["alternative_initial_error_m"] = float(
            np.linalg.norm(
                alternative.initial_position_m - truth.initial_follower_position_m
            )
        )
    return result


def config_to_dict(config: BatchEstimatorConfig) -> Dict[str, Any]:
    return asdict(config)


__all__ = [
    "VERSION",
    "SEALED_FINAL_SEED_START",
    "SEALED_FINAL_SEED_END",
    "V181_DEV_SEED_START",
    "V181_DEV_SEED_END",
    "PRIMARY_CONTROLLER",
    "REPLAY_TOLERANCE",
    "OnlineDopplerHistory",
    "ReplayTruthDiagnostics",
    "ReplayCapture",
    "BatchEstimatorConfig",
    "BatchMode",
    "BatchEstimate",
    "assert_seed_is_not_sealed_final",
    "initial_leader_centroid_from_history",
    "predict_doppler",
    "predict_doppler_many",
    "residual_and_jacobian",
    "deterministic_shell_candidates",
    "project_to_shell",
    "refine_damped_gauss_newton",
    "estimate_initial_position_multistart",
    "structural_noiseless_history",
    "capture_v181_episode",
    "save_online_history",
    "load_online_history",
    "save_truth_diagnostics",
    "load_truth_diagnostics",
    "save_replay_capture",
    "load_replay_capture",
    "estimate_diagnostic_errors",
    "sha256_file",
    "config_to_dict",
]
