#!/usr/bin/env python3
"""Paired one- versus two-reference Doppler ablation.

The module is append-only with respect to the frozen V19/V22/V24 method.  It
uses the same audited ACQUIRE->TRACK gate and the same post-lock PID task, but
places an explicit information boundary in front of the estimator and active
planner.  A one-leader arm receives only the selected leader state and Doppler
row for measurement prediction.  Both physical leader broadcasts remain a
common mission reference for the prior shell, fixed S-turn, candidate-action
anchor, and formation target.  Thus the intervention changes the Doppler
references used for inference, not the underlying two-leader task.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from uuv_v11_rng import ExogenousNoiseTape
from uuv_v18_resampling_guard import UUV3DConfig, UUVTwoLeader3DPFEnv
import uuv_v19_observability as v19
import uuv_v20_positioning_ablation as v20
import uuv_v21_causal_lock as v21
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24


VERSION = "v38_leader_source_ablation_1.1"
SOURCE_L1 = "leader1_only"
SOURCE_L2 = "leader2_only"
SOURCE_BOTH = "both_leaders"
SOURCE_MASKS: Mapping[str, Tuple[bool, bool]] = {
    SOURCE_L1: (True, False),
    SOURCE_L2: (False, True),
    SOURCE_BOTH: (True, True),
}
POLICY_FIXED = "fixed_s_turn"
POLICY_ACTIVE = "belief_active"
POLICIES: Tuple[str, ...] = (POLICY_FIXED, POLICY_ACTIVE)
SOURCE_NAMES: Tuple[str, ...] = (SOURCE_L1, SOURCE_L2, SOURCE_BOTH)
RESERVED_START = 49_900
FINAL_END = 50_999
COMMON_PRE_RELEASE_CHECKPOINT_S = 60.0


def assert_seed_allowed(seed: int) -> None:
    value = int(seed)
    if RESERVED_START <= value <= FINAL_END:
        raise PermissionError(
            f"V38 refuses reserved/final seed {value}; "
            f"{RESERVED_START}..{FINAL_END} remain closed"
        )


def arm_name(source_name: str, policy_name: str) -> str:
    if source_name not in SOURCE_MASKS:
        raise ValueError(f"unknown source arm {source_name!r}")
    if policy_name not in POLICIES:
        raise ValueError(f"unknown acquisition policy {policy_name!r}")
    return f"{source_name}__{policy_name}"


def arm_pairs() -> Tuple[Tuple[str, str], ...]:
    return tuple(
        (source, policy)
        for source in SOURCE_NAMES
        for policy in POLICIES
    )


def _mask_array(mask: Sequence[bool]) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.shape != (2,) or not np.any(value):
        raise ValueError("leader source mask must contain one or two active links")
    return value


@dataclass(frozen=True)
class MissionSupport:
    """Common reset-time two-leader support shared by all six arms."""

    center_m: np.ndarray
    radius_min_m: float
    radius_max_m: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("mission support center must contain three finite values")
        if (
            not math.isfinite(float(self.radius_min_m))
            or not math.isfinite(float(self.radius_max_m))
            or float(self.radius_min_m) < 0.0
            or float(self.radius_max_m) <= float(self.radius_min_m)
        ):
            raise ValueError("invalid mission support radii")
        object.__setattr__(self, "center_m", center.copy())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_m": self.center_m.tolist(),
            "radius_min_m": float(self.radius_min_m),
            "radius_max_m": float(self.radius_max_m),
            "source": (
                "shared_reset_time_leader_centroid_and_frozen_mission_radii"
            ),
        }


def mission_support_from_initial_leaders(
    cfg: UUV3DConfig,
    leader1_position_m: Sequence[float],
    leader2_position_m: Sequence[float],
) -> MissionSupport:
    """Return the frozen method's common reset-time support shell.

    Both physical leader broadcasts are available in every arm as common
    mission information.  The only experimental intervention is which
    leader-state/Doppler column enters measurement prediction, evidence, and
    active-planner scoring.
    """

    leader1 = np.asarray(leader1_position_m, dtype=np.float64)
    leader2 = np.asarray(leader2_position_m, dtype=np.float64)
    if (
        leader1.shape != (3,)
        or leader2.shape != (3,)
        or not np.all(np.isfinite(leader1))
        or not np.all(np.isfinite(leader2))
    ):
        raise ValueError("initial leader positions must contain finite 3-D values")
    return MissionSupport(
        center_m=0.5 * (leader1 + leader2),
        radius_min_m=float(cfg.start_rho_min),
        radius_max_m=float(cfg.start_rho_max),
    )


@dataclass(frozen=True)
class MaskedDopplerHistory:
    """Deployable history after physically removing excluded leader columns."""

    source_mask: Tuple[bool, bool]
    source_indices: Tuple[int, ...]
    t_s: np.ndarray
    dead_reckoned_displacement_m: np.ndarray
    leader_position_m: np.ndarray
    leader_velocity_mps: np.ndarray
    follower_velocity_measured_mps: np.ndarray
    doppler_measured_mps: np.ndarray
    historical_pf_gate_factor: np.ndarray

    def __post_init__(self) -> None:
        mask = tuple(bool(value) for value in self.source_mask)
        mask_array = _mask_array(mask)
        indices = tuple(int(value) for value in self.source_indices)
        expected = tuple(int(value) for value in np.flatnonzero(mask_array))
        if indices != expected:
            raise ValueError("source indices do not match source mask")
        t_s = np.asarray(self.t_s, dtype=np.float64)
        if (
            t_s.ndim != 1
            or t_s.size == 0
            or not np.all(np.isfinite(t_s))
            or np.any(np.diff(t_s) <= 0.0)
        ):
            raise ValueError("measurement times must be finite and increasing")
        n = int(t_s.size)
        k = len(indices)
        expected_shapes = {
            "dead_reckoned_displacement_m": (n, 3),
            "leader_position_m": (n, k, 3),
            "leader_velocity_mps": (n, k, 3),
            "follower_velocity_measured_mps": (n, 3),
            "doppler_measured_mps": (n, k),
            "historical_pf_gate_factor": (n, k),
        }
        values: Dict[str, np.ndarray] = {}
        for name, shape in expected_shapes.items():
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            values[name] = array.copy()
        if np.any(values["historical_pf_gate_factor"] < 0.0) or np.any(
            values["historical_pf_gate_factor"] > 1.0
        ):
            raise ValueError("gate factors must lie in [0,1]")
        object.__setattr__(self, "source_mask", mask)
        object.__setattr__(self, "source_indices", indices)
        object.__setattr__(self, "t_s", t_s.copy())
        for name, array in values.items():
            object.__setattr__(self, name, array)

    @classmethod
    def from_full(
        cls,
        history: v19.OnlineDopplerHistory,
        source_mask: Sequence[bool],
    ) -> "MaskedDopplerHistory":
        mask = _mask_array(source_mask)
        indices = np.flatnonzero(mask)
        return cls(
            source_mask=tuple(bool(value) for value in mask),
            source_indices=tuple(int(value) for value in indices),
            t_s=history.t_s.copy(),
            dead_reckoned_displacement_m=history.dead_reckoned_displacement_m.copy(),
            leader_position_m=history.leader_position_m[:, indices, :].copy(),
            leader_velocity_mps=history.leader_velocity_mps[:, indices, :].copy(),
            follower_velocity_measured_mps=(
                history.follower_velocity_measured_mps.copy()
            ),
            doppler_measured_mps=history.doppler_measured_mps[:, indices].copy(),
            historical_pf_gate_factor=(
                history.historical_pf_gate_factor[:, indices].copy()
            ),
        )

    @property
    def measurement_count(self) -> int:
        return int(self.t_s.size)

    @property
    def active_source_count(self) -> int:
        return len(self.source_indices)

    @property
    def scalar_measurement_count(self) -> int:
        return self.measurement_count * self.active_source_count

    def take(self, selection: Any) -> "MaskedDopplerHistory":
        indices = np.arange(self.measurement_count)[selection]
        indices = np.atleast_1d(indices)
        return MaskedDopplerHistory(
            source_mask=self.source_mask,
            source_indices=self.source_indices,
            t_s=self.t_s[indices].copy(),
            dead_reckoned_displacement_m=(
                self.dead_reckoned_displacement_m[indices].copy()
            ),
            leader_position_m=self.leader_position_m[indices].copy(),
            leader_velocity_mps=self.leader_velocity_mps[indices].copy(),
            follower_velocity_measured_mps=(
                self.follower_velocity_measured_mps[indices].copy()
            ),
            doppler_measured_mps=self.doppler_measured_mps[indices].copy(),
            historical_pf_gate_factor=(
                self.historical_pf_gate_factor[indices].copy()
            ),
        )

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(VERSION.encode("utf-8"))
        digest.update(np.asarray(self.source_mask, dtype=np.uint8).tobytes())
        for value in (
            self.t_s,
            self.dead_reckoned_displacement_m,
            self.leader_position_m,
            self.leader_velocity_mps,
            self.follower_velocity_measured_mps,
            self.doppler_measured_mps,
            self.historical_pf_gate_factor,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


def _measurement_weights(
    history: MaskedDopplerHistory,
    config: v19.BatchEstimatorConfig,
) -> np.ndarray:
    if config.gate_mode == "raw":
        return np.ones_like(history.doppler_measured_mps, dtype=np.float64)
    return np.clip(history.historical_pf_gate_factor, 1e-3, 1.0)


def predict_doppler(
    initial_position_m: Sequence[float],
    history: MaskedDopplerHistory,
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
    history: MaskedDopplerHistory,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    candidates = np.asarray(initial_positions_m, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("initial positions must have shape (M,3)")
    outputs: List[np.ndarray] = []
    chunk = max(1, int(chunk_size))
    for start in range(0, candidates.shape[0], chunk):
        candidate = candidates[start : start + chunk]
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
    initial_position_m: Sequence[float],
    history: MaskedDopplerHistory,
    config: v19.BatchEstimatorConfig,
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
    jacobian = -(
        relative_velocity - projection[:, :, None] * line_of_sight
    ) / rho[:, :, None]
    sqrt_weight = np.sqrt(_measurement_weights(history, config))
    return (
        (sqrt_weight * residual).reshape(-1),
        (sqrt_weight[:, :, None] * jacobian).reshape(-1, 3),
    )


def common_s_turn_action(
    env: UUVTwoLeader3DPFEnv,
    time_s: float,
) -> np.ndarray:
    """Common position-independent acquisition reference for all six arms."""

    return v20.position_independent_acquisition_action(
        env,
        float(time_s),
    )


def _mode_from_solution(
    position_m: np.ndarray,
    history: MaskedDopplerHistory,
    config: v19.BatchEstimatorConfig,
    iterations: int,
    converged: bool,
    support: MissionSupport,
) -> v19.BatchMode:
    residual, jacobian = residual_and_jacobian(position_m, history, config)
    if residual.size != history.scalar_measurement_count:
        raise RuntimeError("masked residual denominator differs from active scalar count")
    sse = float(residual @ residual)
    rmse = float(math.sqrt(sse / max(residual.size, 1)))
    hessian = jacobian.T @ jacobian
    hessian = 0.5 * (hessian + hessian.T)
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(3, dtype=np.float64)
    largest = float(max(np.max(eigenvalues), 0.0))
    rank_threshold = max(1e-12, 1e-10 * largest)
    rank = int(np.sum(eigenvalues > rank_threshold))
    condition = (
        float(largest / float(eigenvalues[0]))
        if rank == 3 and float(eigenvalues[0]) > 0.0
        else float("inf")
    )
    radius = float(
        np.linalg.norm(
            np.asarray(position_m, dtype=np.float64) - support.center_m
        )
    )
    boundary_tolerance = max(1e-6, 1e-8 * float(support.radius_max_m))
    constrained_boundary = bool(
        abs(radius - float(support.radius_min_m)) <= boundary_tolerance
        or abs(radius - float(support.radius_max_m)) <= boundary_tolerance
    )
    covariance_valid = bool(
        rank == 3
        and np.isfinite(condition)
        and condition <= 1e10
        and not constrained_boundary
    )
    dof = max(int(residual.size) - 3, 1)
    empirical_variance = max(
        sse / dof,
        float(config.measurement_sigma_mps) ** 2,
    )
    covariance = np.full((3, 3), np.nan, dtype=np.float64)
    radius95 = float("inf")
    if covariance_valid:
        try:
            covariance = empirical_variance * np.linalg.inv(hessian)
            covariance = 0.5 * (covariance + covariance.T)
            maximum_variance = float(
                max(np.max(np.linalg.eigvalsh(covariance)), 0.0)
            )
            radius95 = v19.CHI2_3_95_SQRT * math.sqrt(maximum_variance)
            covariance_valid = bool(np.isfinite(radius95))
        except np.linalg.LinAlgError:
            covariance_valid = False
    if not covariance_valid:
        covariance = np.full((3, 3), np.nan, dtype=np.float64)
        radius95 = float("inf")
    return v19.BatchMode(
        initial_position_m=np.asarray(position_m, dtype=np.float64).copy(),
        residual_sse_mps2=sse,
        residual_rmse_mps=rmse,
        iterations=int(iterations),
        converged=bool(converged),
        hessian_eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        hessian_rank=rank,
        hessian_condition_number=condition,
        local_covariance_m2=covariance,
        local_covariance_valid=covariance_valid,
        local_radius95_m=float(radius95),
    )


def refine_damped_gauss_newton(
    initial_position_m: Sequence[float],
    history: MaskedDopplerHistory,
    support: MissionSupport,
    config: v19.BatchEstimatorConfig,
) -> v19.BatchMode:
    position = v19.project_to_shell(
        np.asarray(initial_position_m, dtype=np.float64),
        support.center_m,
        support.radius_min_m,
        support.radius_max_m,
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
        system = normal + damping * np.diag(np.maximum(np.diag(normal), 1e-12))
        try:
            step = np.linalg.solve(system, -gradient)
        except np.linalg.LinAlgError:
            damping = min(damping * 10.0, 1e18)
            continue
        if float(np.linalg.norm(step)) <= float(config.step_tolerance_m):
            converged = True
            break
        trial = v19.project_to_shell(
            position + step,
            support.center_m,
            support.radius_min_m,
            support.radius_max_m,
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
        support,
    )


def _select_starts(
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
        raise RuntimeError("masked coarse search produced fewer than two starts")
    return selected


def _cluster_modes(
    modes: Iterable[v19.BatchMode],
    config: v19.BatchEstimatorConfig,
) -> Tuple[v19.BatchMode, ...]:
    kept: List[v19.BatchMode] = []
    for mode in sorted(modes, key=lambda value: value.residual_sse_mps2):
        if all(
            float(
                np.linalg.norm(
                    mode.initial_position_m - previous.initial_position_m
                )
            )
            >= float(config.mode_cluster_radius_m)
            for previous in kept
        ):
            kept.append(mode)
    if not kept:
        raise RuntimeError("masked refinement produced no modes")
    return tuple(kept)


def estimate_initial_position_multistart(
    history: MaskedDopplerHistory,
    support: MissionSupport,
    config: v19.BatchEstimatorConfig,
    *,
    candidate_seed: int,
) -> v19.BatchEstimate:
    started = time.perf_counter()
    candidates = np.concatenate(
        [
            v19.deterministic_shell_candidates(
                support.center_m,
                support.radius_min_m,
                support.radius_max_m,
                int(config.coarse_candidates),
                int(candidate_seed) + 104_729 * sweep,
                radial_distribution=config.candidate_radial_distribution,
            )
            for sweep in range(int(config.coarse_sweeps))
        ],
        axis=0,
    )
    prediction = predict_doppler_many(candidates, history)
    residual = history.doppler_measured_mps[None, :, :] - prediction
    weights = _measurement_weights(history, config)[None, :, :]
    costs = np.sum(weights * residual * residual, axis=(1, 2))
    starts = _select_starts(
        candidates,
        costs,
        int(config.local_starts),
        float(config.coarse_start_separation_m),
    )
    all_modes = _cluster_modes(
        (
            refine_damped_gauss_newton(start, history, support, config)
            for start in starts
        ),
        config,
    )
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
    reported = list(all_modes[: int(config.maximum_modes)])
    alternative_index: Optional[int] = None
    if alternative_all_index is not None:
        alternative = all_modes[alternative_all_index]
        try:
            alternative_index = next(
                index for index, mode in enumerate(reported) if mode is alternative
            )
        except StopIteration:
            if len(reported) >= int(config.maximum_modes):
                reported[-1] = alternative
                alternative_index = len(reported) - 1
            else:
                reported.append(alternative)
                alternative_index = len(reported) - 1
    coarse_rmse = float(
        math.sqrt(float(np.min(costs)) / history.scalar_measurement_count)
    )
    return v19.BatchEstimate(
        modes=tuple(reported),
        runtime_s=float(time.perf_counter() - started),
        coarse_best_rmse_mps=coarse_rmse,
        candidate_count=int(candidates.shape[0]),
        refined_start_count=len(starts),
        clustered_mode_count=len(all_modes),
        alternative_mode_index=alternative_index,
        alternative_distance_m=alternative_distance,
        alternative_delta_sse_mps2=alternative_delta_sse,
        alternative_delta_chi2=alternative_delta_chi2,
    )


class LeaderMaskedAuditedGate(v24.AuditedCausalLockGate):
    """Frozen audited gate with explicit active-scalar metric provenance."""

    def __init__(
        self,
        config: v24.AuditedLockConfig,
        source_mask: Sequence[bool],
    ) -> None:
        super().__init__(config)
        mask = _mask_array(source_mask)
        self.source_mask = tuple(bool(value) for value in mask)
        self.full_residual_scalar_count = 0
        self.recent_residual_scalar_count = 0
        self.forward_residual_scalar_count = 0

    def set_residual_counts(
        self,
        *,
        full: int,
        recent: int,
        forward: int,
    ) -> None:
        values = (int(full), int(recent), int(forward))
        if values[0] < 1 or values[1] < 1 or values[2] < 0:
            raise ValueError("invalid active residual counts")
        self.full_residual_scalar_count = values[0]
        self.recent_residual_scalar_count = values[1]
        self.forward_residual_scalar_count = values[2]

    def _audit_checks(
        self,
        *,
        local_position_m: np.ndarray,
        evidence: v21.GlobalLockEvidence,
        release: bool,
    ) -> Dict[str, bool]:
        checks = super()._audit_checks(
            local_position_m=local_position_m,
            evidence=evidence,
            release=release,
        )
        self.last_metrics.update(
            {
                "active_source_count": float(sum(self.source_mask)),
                "full_residual_scalar_count": float(
                    self.full_residual_scalar_count
                ),
                "recent_residual_scalar_count": float(
                    self.recent_residual_scalar_count
                ),
                "forward_residual_scalar_count": float(
                    self.forward_residual_scalar_count
                ),
            }
        )
        return checks


class LeaderMaskedStreamingEstimator:
    """V19 streaming schedule and V24 gate evaluated on masked measurements."""

    def __init__(
        self,
        *,
        estimator_config: v19.BatchEstimatorConfig,
        lock_config: v24.AuditedLockConfig,
        source_mask: Sequence[bool],
        support: MissionSupport,
    ) -> None:
        self.estimator_config = estimator_config
        self.source_mask = tuple(bool(value) for value in _mask_array(source_mask))
        self.support = support
        self.support_radius_min_m = float(support.radius_min_m)
        self.support_radius_max_m = float(support.radius_max_m)
        self.gate = LeaderMaskedAuditedGate(lock_config, self.source_mask)
        self.state = v21.CausalEstimatorState()
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
        return v21.CausalStreamingEstimator._scheduled_global(now_s)

    def _forward_residual(
        self,
        history: MaskedDopplerHistory,
    ) -> Tuple[Optional[float], int, int]:
        if (
            self._previous_global_position_m is None
            or self._previous_global_measurement_count is None
        ):
            return None, 0, 0
        start = int(self._previous_global_measurement_count)
        if start >= history.measurement_count:
            return None, 0, 0
        held_out = history.take(slice(start, history.measurement_count))
        residual = (
            held_out.doppler_measured_mps
            - predict_doppler(self._previous_global_position_m, held_out)
        )
        return (
            float(math.sqrt(float(np.mean(residual * residual)))),
            int(held_out.measurement_count),
            int(held_out.scalar_measurement_count),
        )

    @staticmethod
    def _recent_residual(
        history: MaskedDopplerHistory,
        position_m: np.ndarray,
    ) -> Tuple[float, int]:
        recent = history.take(
            slice(max(0, history.measurement_count - 20), history.measurement_count)
        )
        residual = recent.doppler_measured_mps - predict_doppler(position_m, recent)
        return (
            float(math.sqrt(float(np.mean(residual * residual)))),
            int(recent.scalar_measurement_count),
        )

    def update(
        self,
        history: MaskedDopplerHistory,
        *,
        decision_time_s: float,
        current_dead_reckoning_m: Sequence[float],
    ) -> None:
        if history.source_mask != self.source_mask:
            raise RuntimeError("estimator history source mask changed")
        now = float(decision_time_s)
        current_dr = np.asarray(current_dead_reckoning_m, dtype=np.float64)
        if current_dr.shape != (3,) or not np.all(np.isfinite(current_dr)):
            raise ValueError("current dead reckoning must contain three finite values")
        if float(history.t_s[-1]) > now + 1e-9:
            raise RuntimeError("causal history contains a future measurement")
        if now + 1e-9 < v21.ESTIMATION_START_S:
            return
        forced = self.gate.consume_force_global_refresh()
        global_update = bool(
            forced
            or self._scheduled_global(now)
            or self.state.initial_position_m is None
        )
        started = time.perf_counter()
        forward_scalar_count = 0
        if global_update:
            primary_seed = 21_001 + int(round(now))
            confirmation_seed = 1_021_001 + int(round(now))
            primary = estimate_initial_position_multistart(
                history,
                self.support,
                self.estimator_config,
                candidate_seed=primary_seed,
            )
            confirmation = estimate_initial_position_multistart(
                history,
                self.support,
                self.estimator_config,
                candidate_seed=confirmation_seed,
            )
            mode = primary.best
            forward_rmse, forward_count, forward_scalar_count = (
                self._forward_residual(history)
            )
            stability = (
                None
                if self._previous_global_position_m is None
                else float(
                    np.linalg.norm(
                        mode.initial_position_m
                        - self._previous_global_position_m
                    )
                )
            )
            evidence = v21.GlobalLockEvidence(
                time_s=now,
                primary=primary,
                confirmation=confirmation,
                search_agreement_m=float(
                    np.linalg.norm(
                        mode.initial_position_m
                        - confirmation.best.initial_position_m
                    )
                ),
                stability_from_previous_m=stability,
                forward_prediction_rmse_mps=forward_rmse,
                forward_prediction_sample_count=forward_count,
            )
            self.state.latest_global_evidence = evidence
            self._previous_global_position_m = mode.initial_position_m.copy()
            self._previous_global_measurement_count = history.measurement_count
            elapsed = float(time.perf_counter() - started)
            self.state.global_runtime_s += elapsed
            self.state.global_solve_count += 2
            self.global_records.append(
                {
                    "time_s": now,
                    "forced": forced,
                    "source_mask": list(self.source_mask),
                    "measurement_row_count": int(history.measurement_count),
                    "active_scalar_measurement_count": int(
                        history.scalar_measurement_count
                    ),
                    "forward_active_scalar_measurement_count": int(
                        forward_scalar_count
                    ),
                    "primary_candidate_seed": primary_seed,
                    "confirmation_candidate_seed": confirmation_seed,
                    "runtime_s": elapsed,
                    **evidence.to_dict(),
                }
            )
        else:
            if self.state.initial_position_m is None:
                raise RuntimeError("local solve requested before global initialization")
            mode = refine_damped_gauss_newton(
                self.state.initial_position_m,
                history,
                self.support,
                self.estimator_config,
            )
            elapsed = float(time.perf_counter() - started)
            self.state.local_runtime_s += elapsed
            self.state.local_solve_count += 1
            evidence = self.state.latest_global_evidence
            if evidence is not None and evidence.forward_prediction_sample_count > 0:
                forward_scalar_count = int(
                    evidence.forward_prediction_sample_count
                    * history.active_source_count
                )
        self.state.total_runtime_s += elapsed
        self.state.maximum_update_runtime_s = max(
            self.state.maximum_update_runtime_s,
            elapsed,
        )
        self.state.latest_mode = mode
        self.state.initial_position_m = mode.initial_position_m.copy()
        self.state.endpoint_position_m = (
            self.state.initial_position_m + current_dr
        )
        recent_rmse, recent_scalar_count = self._recent_residual(
            history,
            self.state.initial_position_m,
        )
        self.state.recent_residual_rmse_mps = recent_rmse
        self.gate.set_residual_counts(
            full=history.scalar_measurement_count,
            recent=recent_scalar_count,
            forward=forward_scalar_count,
        )
        self.gate.evaluate(
            now_s=now,
            mode=mode,
            initial_position_m=self.state.initial_position_m,
            recent_residual_rmse_mps=recent_rmse,
            global_evidence=self.state.latest_global_evidence,
        )


class MaskedBeliefFIMPlanner(v22.BeliefFIMPlanner):
    """Frozen V22 utility evaluated only for the selected acoustic sources."""

    def __init__(
        self,
        config: v22.ActivePlannerConfig,
        source_mask: Sequence[bool],
    ) -> None:
        super().__init__(config)
        self.source_mask = tuple(bool(value) for value in _mask_array(source_mask))

    def _hypotheses(
        self,
        estimator: LeaderMaskedStreamingEstimator,
        history: MaskedDopplerHistory,
    ) -> np.ndarray:
        state = estimator.state
        if state.initial_position_m is None or state.latest_mode is None:
            raise RuntimeError("planner requested before estimator initialization")
        values: List[np.ndarray] = [state.initial_position_m.copy()]
        evidence = state.latest_global_evidence
        if evidence is not None:
            for estimate in (evidence.primary, evidence.confirmation):
                values.extend(
                    mode.initial_position_m.copy() for mode in estimate.modes
                )
        mode = state.latest_mode
        covariance = np.asarray(mode.local_covariance_m2, dtype=np.float64)
        if (
            mode.local_covariance_valid
            and covariance.shape == (3, 3)
            and np.all(np.isfinite(covariance))
        ):
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(
                    0.5 * (covariance + covariance.T)
                )
                for index in range(3):
                    scale = float(self.config.covariance_axis_scale) * math.sqrt(
                        max(float(eigenvalues[index]), 0.0)
                    )
                    delta = scale * eigenvectors[:, index]
                    for sign in (-1.0, 1.0):
                        values.append(
                            v19.project_to_shell(
                                state.initial_position_m + sign * delta,
                                estimator.support.center_m,
                                estimator.support.radius_min_m,
                                estimator.support.radius_max_m,
                            )
                        )
            except np.linalg.LinAlgError:
                pass
        return v22._cluster_hypotheses(
            values,
            radius_m=self.config.hypothesis_cluster_radius_m,
            maximum=self.config.maximum_hypotheses,
        )

    def action(
        self,
        env: UUVTwoLeader3DPFEnv,
        *,
        history: MaskedDopplerHistory,
        current_dead_reckoning_m: Sequence[float],
        estimator: LeaderMaskedStreamingEstimator,
        time_s: float,
    ) -> v22.PlannerDecision:
        if history.source_mask != self.source_mask:
            raise RuntimeError("planner history source mask changed")
        started = time.perf_counter()
        fallback = common_s_turn_action(env, float(time_s))
        if not estimator.has_solution or estimator.state.latest_mode is None:
            elapsed = float(time.perf_counter() - started)
            self.total_runtime_s += elapsed
            self.maximum_runtime_s = max(self.maximum_runtime_s, elapsed)
            self.decision_count += 1
            self.previous_action = np.asarray(fallback, dtype=np.float64)
            return v22.PlannerDecision(
                action=np.asarray(fallback, dtype=np.float32),
                utility=0.0,
                worst_radius_before_m=float("inf"),
                worst_radius_after_m=float("inf"),
                minimum_pair_chi2=0.0,
                hypothesis_count=0,
                candidate_count=1,
                runtime_s=elapsed,
            )
        hypotheses = self._hypotheses(estimator, history)
        base_information: List[np.ndarray] = []
        for position in hypotheses:
            _, jacobian = residual_and_jacobian(
                position,
                history,
                estimator.estimator_config,
            )
            base_information.append(jacobian.T @ jacobian)
        base = np.asarray(base_information, dtype=np.float64)
        candidates = v22._candidate_actions(fallback)
        current_dr = np.asarray(current_dead_reckoning_m, dtype=np.float64)
        all_leaders = np.asarray([env.pL1, env.pL2], dtype=np.float64)
        all_velocities = np.asarray([env.vL1, env.vL2], dtype=np.float64)
        indices = np.asarray(history.source_indices, dtype=np.int64)
        leaders = all_leaders[indices]
        leader_velocities = all_velocities[indices]
        sigma = float(estimator.estimator_config.measurement_sigma_mps)
        best_index = 0
        best_utility = -float("inf")
        best_metrics = (float("inf"), float("inf"), 0.0)
        for index, candidate in enumerate(candidates):
            velocity = v22._action_velocity(candidate, env)
            before, after, pair = self._evaluate_velocity(
                velocity_mps=velocity,
                hypotheses_p0_m=hypotheses,
                current_dead_reckoning_m=current_dr,
                base_information=base,
                leader_position_m=leaders,
                leader_velocity_mps=leader_velocities,
                sigma_mps=sigma,
            )
            utility = (
                before
                - after
                + float(self.config.pair_weight) * math.log1p(max(pair, 0.0))
                - float(self.config.action_energy_weight)
                * float(candidate @ candidate)
                - float(self.config.action_change_weight)
                * float(np.sum((candidate - self.previous_action) ** 2))
            )
            if math.isfinite(utility) and utility > best_utility:
                best_index = index
                best_utility = float(utility)
                best_metrics = (before, after, pair)
        if not math.isfinite(best_utility):
            raise RuntimeError("masked planner produced no finite utility")
        selected = np.asarray(candidates[best_index], dtype=np.float32)
        self.previous_action = selected.astype(np.float64)
        elapsed = float(time.perf_counter() - started)
        self.total_runtime_s += elapsed
        self.maximum_runtime_s = max(self.maximum_runtime_s, elapsed)
        self.decision_count += 1
        return v22.PlannerDecision(
            action=selected,
            utility=best_utility,
            worst_radius_before_m=float(best_metrics[0]),
            worst_radius_after_m=float(best_metrics[1]),
            minimum_pair_chi2=float(best_metrics[2]),
            hypothesis_count=int(hypotheses.shape[0]),
            candidate_count=int(candidates.shape[0]),
            runtime_s=elapsed,
        )


@dataclass(frozen=True)
class ArmOutcome:
    summary: Mapping[str, Any]
    trace: Mapping[str, np.ndarray]


def _audit_trace_values(gate: LeaderMaskedAuditedGate) -> Dict[str, float]:
    values = v24._audit_trace_values(gate)
    values.update(
        {
            "active_source_count": float(sum(gate.source_mask)),
            "gate_full_residual_scalar_count": float(
                gate.full_residual_scalar_count
            ),
            "gate_recent_residual_scalar_count": float(
                gate.recent_residual_scalar_count
            ),
            "gate_forward_residual_scalar_count": float(
                gate.forward_residual_scalar_count
            ),
        }
    )
    return values


def run_arm(
    *,
    cfg: UUV3DConfig,
    tape: ExogenousNoiseTape,
    episode_seed: int,
    episode_index: int,
    source_name: str,
    policy_name: str,
    estimator_config: v19.BatchEstimatorConfig,
    lock_config: v24.AuditedLockConfig,
    planner_config: v22.ActivePlannerConfig,
) -> ArmOutcome:
    if source_name not in SOURCE_MASKS or policy_name not in POLICIES:
        raise ValueError("unknown V38 arm")
    assert_seed_allowed(episode_seed)
    source_mask = SOURCE_MASKS[source_name]
    env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    env.attach_exogenous_noise_tape(tape)
    env.reset(seed=int(episode_seed))
    initial_truth = np.asarray(env.pF, dtype=np.float64).copy()
    support = mission_support_from_initial_leaders(
        cfg,
        env.pL1,
        env.pL2,
    )
    recorder = v20.OnlineHistoryRecorder(env)
    estimator = LeaderMaskedStreamingEstimator(
        estimator_config=estimator_config,
        lock_config=lock_config,
        source_mask=source_mask,
        support=support,
    )
    planner = MaskedBeliefFIMPlanner(planner_config, source_mask)
    fields = (
        "step", "action_start_time_s", "time_s", "phase_track",
        "gate_locked_after_update", "action_speed", "action_yaw", "action_pitch",
        "truth_x", "truth_y", "truth_z", "estimate_x", "estimate_y", "estimate_z",
        "formation_error_truth_m", "localization_error_m", "batch_local_radius95_m",
        "planner_utility", "planner_worst_radius_before_m",
        "planner_worst_radius_after_m", "planner_minimum_pair_chi2",
        "planner_hypothesis_count", "planner_runtime_s",
        "gate_release_predicate", "gate_release_pass_streak",
        "gate_hold_predicate", "gate_hold_failure_streak",
        "audit_release_checks_pass", "audit_hold_checks_pass",
        "local_to_primary_m", "local_to_confirmation_m",
        "primary_to_confirmation_m", "primary_radius95_m",
        "confirmation_radius95_m", "primary_rmse_mps",
        "confirmation_rmse_mps", "global_evidence_time_s",
        "active_source_count", "gate_full_residual_scalar_count",
        "gate_recent_residual_scalar_count",
        "gate_forward_residual_scalar_count",
        "batch_full_residual_rmse_mps",
        "batch_recent_residual_rmse_mps",
        "estimator_measurement_row_count",
        "current_dead_reckoning_x", "current_dead_reckoning_y",
        "current_dead_reckoning_z",
        "source_mask_l1", "source_mask_l2",
    )
    trace_rows: Dict[str, List[Any]] = {field: [] for field in fields}
    actions: List[np.ndarray] = []
    maximum_combined_runtime_s = 0.0
    try:
        for _ in range(int(cfg.max_steps)):
            action_start = float(env.t)
            track = bool(estimator.gate.is_locked and estimator.has_solution)
            decision: Optional[v22.PlannerDecision] = None
            if track:
                action = v20.pid_action_for_position(
                    env,
                    np.asarray(estimator.state.endpoint_position_m, dtype=np.float64),
                )
            elif (
                policy_name == POLICY_FIXED
                or not estimator.has_solution
            ):
                action = common_s_turn_action(env, action_start)
            else:
                history_before = MaskedDopplerHistory.from_full(
                    recorder.history(),
                    source_mask,
                )
                decision = planner.action(
                    env,
                    history=history_before,
                    current_dead_reckoning_m=(
                        recorder.accumulated_dead_reckoning_m
                    ),
                    estimator=estimator,
                    time_s=action_start,
                )
                action = decision.action
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (3,) or not np.all(np.isfinite(action)):
                raise RuntimeError("V38 produced a non-finite action")
            actions.append(action.astype(np.float64))
            _, _, terminated, truncated, _ = env.step(action)
            masked_history = MaskedDopplerHistory.from_full(
                recorder.history(),
                source_mask,
            )
            update_started = time.perf_counter()
            estimator.update(
                masked_history,
                decision_time_s=float(env.step_count) * float(cfg.action_dt),
                current_dead_reckoning_m=(
                    recorder.accumulated_dead_reckoning_m
                ),
            )
            update_runtime = float(time.perf_counter() - update_started)
            combined_runtime = update_runtime + (
                0.0 if decision is None else float(decision.runtime_s)
            )
            maximum_combined_runtime_s = max(
                maximum_combined_runtime_s,
                combined_runtime,
            )
            truth = np.asarray(env.pF, dtype=np.float64).copy()
            endpoint = (
                np.asarray(
                    estimator.state.endpoint_position_m,
                    dtype=np.float64,
                ).copy()
                if estimator.has_solution
                else np.full(3, np.nan, dtype=np.float64)
            )
            _, desired = env._formation_desired()
            formation = float(np.linalg.norm(truth - np.asarray(desired)))
            localization = float(np.linalg.norm(endpoint - truth))
            mode = estimator.state.latest_mode
            radius95 = (
                float("nan")
                if mode is None
                else float(mode.local_radius95_m)
            )
            values: Dict[str, Any] = {
                "step": int(env.step_count),
                "action_start_time_s": action_start,
                "time_s": float(env.t),
                "phase_track": float(track),
                "gate_locked_after_update": float(estimator.gate.is_locked),
                "action_speed": float(action[0]),
                "action_yaw": float(action[1]),
                "action_pitch": float(action[2]),
                "truth_x": float(truth[0]),
                "truth_y": float(truth[1]),
                "truth_z": float(truth[2]),
                "estimate_x": float(endpoint[0]),
                "estimate_y": float(endpoint[1]),
                "estimate_z": float(endpoint[2]),
                "formation_error_truth_m": formation,
                "localization_error_m": localization,
                "batch_local_radius95_m": radius95,
                "planner_utility": (
                    float("nan") if decision is None else decision.utility
                ),
                "planner_worst_radius_before_m": (
                    float("nan")
                    if decision is None
                    else decision.worst_radius_before_m
                ),
                "planner_worst_radius_after_m": (
                    float("nan")
                    if decision is None
                    else decision.worst_radius_after_m
                ),
                "planner_minimum_pair_chi2": (
                    float("nan")
                    if decision is None
                    else decision.minimum_pair_chi2
                ),
                "planner_hypothesis_count": (
                    0 if decision is None else decision.hypothesis_count
                ),
                "planner_runtime_s": (
                    0.0 if decision is None else decision.runtime_s
                ),
                "batch_full_residual_rmse_mps": (
                    float("nan")
                    if mode is None
                    else float(mode.residual_rmse_mps)
                ),
                "batch_recent_residual_rmse_mps": (
                    float(estimator.state.recent_residual_rmse_mps)
                    if mode is not None
                    else float("nan")
                ),
                "estimator_measurement_row_count": int(
                    masked_history.measurement_count
                ),
                "current_dead_reckoning_x": float(
                    recorder.accumulated_dead_reckoning_m[0]
                ),
                "current_dead_reckoning_y": float(
                    recorder.accumulated_dead_reckoning_m[1]
                ),
                "current_dead_reckoning_z": float(
                    recorder.accumulated_dead_reckoning_m[2]
                ),
                "source_mask_l1": float(source_mask[0]),
                "source_mask_l2": float(source_mask[1]),
            }
            values.update(_audit_trace_values(estimator.gate))
            for field in fields:
                trace_rows[field].append(values[field])
            if terminated or truncated:
                if int(env.step_count) != int(cfg.max_steps):
                    raise RuntimeError("V38 arm ended before fixed horizon")
                break
        cursor = env._v11_noise_cursor
        if cursor is None:
            raise RuntimeError("V38 environment lost its noise tape")
        noise_cursor = dict(cursor.state_dict())
    finally:
        recorder.restore()
        env.close()

    trace = {field: np.asarray(values) for field, values in trace_rows.items()}
    if trace["time_s"].shape != (int(cfg.max_steps),):
        raise RuntimeError("V38 trace does not contain the fixed horizon")
    full_history = recorder.history()
    masked_history = MaskedDopplerHistory.from_full(full_history, source_mask)
    trace.update(
        {
            "online_t_s": full_history.t_s.copy(),
            "online_dead_reckoned_displacement_m": (
                full_history.dead_reckoned_displacement_m.copy()
            ),
            "online_leader_position_m": full_history.leader_position_m.copy(),
            "online_leader_velocity_mps": full_history.leader_velocity_mps.copy(),
            "online_follower_velocity_measured_mps": (
                full_history.follower_velocity_measured_mps.copy()
            ),
            "online_doppler_measured_mps": (
                full_history.doppler_measured_mps.copy()
            ),
            "online_historical_pf_gate_factor": (
                full_history.historical_pf_gate_factor.copy()
            ),
        }
    )
    times = np.asarray(trace["time_s"], dtype=np.float64)
    formation = np.asarray(trace["formation_error_truth_m"], dtype=np.float64)
    localization = np.asarray(trace["localization_error_m"], dtype=np.float64)
    checkpoint_index = int(
        np.argmin(np.abs(times - COMMON_PRE_RELEASE_CHECKPOINT_S))
    )
    if (
        abs(
            float(times[checkpoint_index])
            - float(COMMON_PRE_RELEASE_CHECKPOINT_S)
        )
        > 1e-6
    ):
        raise RuntimeError("common pre-release checkpoint is absent from trace")
    joint = (formation < v21.TERMINAL_FORMATION_GATE_M) & (
        localization < v21.TERMINAL_LOCALIZATION_GATE_M
    )
    exact = v24.exact_trace_score(trace)
    gate_after = np.asarray(trace["gate_locked_after_update"], dtype=bool)
    transitions = gate_after & ~np.concatenate(
        [np.asarray([False]), gate_after[:-1]]
    )
    audit_release_violation_count = int(
        np.sum(
            transitions
            & ~np.asarray(trace["audit_release_checks_pass"], dtype=bool)
        )
    )
    phase = np.asarray(trace["phase_track"], dtype=bool)
    finite_localization = np.isfinite(localization)
    track_errors = localization[phase & finite_localization]
    false_confidence = (
        np.isfinite(trace["batch_local_radius95_m"])
        & (np.asarray(trace["batch_local_radius95_m"]) < 7.0)
        & ((~finite_localization) | (localization >= 7.0))
    )
    tail = joint[-v21.TAIL_WINDOW_ACTIONS :]
    dwell = joint[-v21.DWELL_ACTIONS :]
    actions_array = np.asarray(actions, dtype=np.float64)
    gate_state = estimator.gate.state
    first_track_time = exact["first_locked_action_time_s"]
    summary: Dict[str, Any] = {
        "version": VERSION,
        "arm": arm_name(source_name, policy_name),
        "source_name": source_name,
        "policy_name": policy_name,
        "source_mask": list(source_mask),
        "active_source_count": int(sum(source_mask)),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "noise_tape_sha256": tape.content_sha256(),
        "masked_history_sha256": masked_history.content_sha256(),
        "action_count": int(times.size),
        "measurement_row_count": int(masked_history.measurement_count),
        "active_scalar_measurement_count": int(
            masked_history.scalar_measurement_count
        ),
        "noise_cursor": noise_cursor,
        "initial_truth_m": initial_truth.tolist(),
        "mission_support": support.to_dict(),
        "checkpoint_60s": {
            "time_s": float(times[checkpoint_index]),
            "localization_error_m": float(localization[checkpoint_index]),
            "localization_below_7m": bool(localization[checkpoint_index] < 7.0),
            "formation_error_m": float(formation[checkpoint_index]),
            "phase_track": bool(
                np.asarray(trace["phase_track"], dtype=bool)[checkpoint_index]
            ),
            "gate_locked_after_update": bool(
                np.asarray(
                    trace["gate_locked_after_update"],
                    dtype=bool,
                )[checkpoint_index]
            ),
        },
        "terminal_formation_error_m": float(formation[-1]),
        "terminal_localization_error_m": float(localization[-1]),
        "terminal_joint_success": bool(joint[-1]),
        "dwell15_joint_success": bool(np.all(dwell)),
        "tail50_joint_occupancy": float(np.mean(tail)),
        "tail80_joint_success": bool(float(np.mean(tail)) >= 0.8),
        "time_to_sustained_joint_lock_s": v21._first_sustained_time(times, joint),
        "mean_squared_action": float(
            np.mean(np.sum(actions_array * actions_array, axis=1))
        ),
        "mean_formation_error_after_30_m": float(
            np.mean(formation[times >= 30.0])
        ),
        "maximum_combined_decision_runtime_s": float(
            maximum_combined_runtime_s
        ),
        "robustness": {
            "maximum_localization_error_m": float(
                np.nanmax(localization[finite_localization])
            ),
            "maximum_track_localization_error_m": (
                None if track_errors.size == 0 else float(np.max(track_errors))
            ),
            "false_confidence_action_count": int(np.sum(false_confidence)),
            "terminal_localization_below_7m": bool(localization[-1] < 7.0),
            "terminal_formation_below_8m": bool(formation[-1] < 8.0),
        },
        "gate": {
            "ever_locked": bool(exact["transition_count"] > 0),
            "first_track_action_time_s": first_track_time,
            "lock_count": int(gate_state.lock_count),
            "unlock_count": int(gate_state.unlock_count),
            "audit_release_violation_count": audit_release_violation_count,
            "exact": exact,
            "transitions": list(gate_state.transitions),
            "full_residual_scalar_count": int(
                estimator.gate.full_residual_scalar_count
            ),
            "recent_residual_scalar_count": int(
                estimator.gate.recent_residual_scalar_count
            ),
            "forward_residual_scalar_count": int(
                estimator.gate.forward_residual_scalar_count
            ),
        },
        "batch": {
            "global_solve_count": int(estimator.state.global_solve_count),
            "local_solve_count": int(estimator.state.local_solve_count),
            "total_runtime_s": float(estimator.state.total_runtime_s),
            "maximum_update_runtime_s": float(
                estimator.state.maximum_update_runtime_s
            ),
            "global_records": list(estimator.global_records),
        },
        "planner": {
            "decision_count": int(planner.decision_count),
            "total_runtime_s": float(planner.total_runtime_s),
            "maximum_runtime_s": float(planner.maximum_runtime_s),
        },
        "information_boundary": {
            "estimator_receives_masked_history_only": True,
            "planner_information_model_uses_masked_sources_only": True,
            "likelihood_active_sources": list(source_mask),
            "jacobian_active_sources": list(source_mask),
            "planner_active_sources": list(source_mask),
            "gate_rmse_active_sources": list(source_mask),
            "common_prior_uses_both_initial_leader_broadcasts": True,
            "common_s_turn_uses_both_leader_velocity_broadcasts": True,
            "planner_candidate_anchor_uses_common_s_turn": True,
            "formation_reference_uses_both_leader_broadcasts": True,
            "mission_support_uses_common_initial_leader_broadcasts": True,
            "legacy_pf_excluded_from_controller": True,
            "simulator_truth_scoring_only": True,
        },
        "sealed_seed_range_untouched": [RESERVED_START, FINAL_END],
    }
    return ArmOutcome(summary=summary, trace=trace)


__all__ = [
    "VERSION",
    "SOURCE_L1",
    "SOURCE_L2",
    "SOURCE_BOTH",
    "SOURCE_MASKS",
    "SOURCE_NAMES",
    "POLICY_FIXED",
    "POLICY_ACTIVE",
    "POLICIES",
    "RESERVED_START",
    "FINAL_END",
    "COMMON_PRE_RELEASE_CHECKPOINT_S",
    "MissionSupport",
    "MaskedDopplerHistory",
    "LeaderMaskedAuditedGate",
    "LeaderMaskedStreamingEstimator",
    "MaskedBeliefFIMPlanner",
    "ArmOutcome",
    "arm_name",
    "arm_pairs",
    "assert_seed_allowed",
    "predict_doppler",
    "predict_doppler_many",
    "residual_and_jacobian",
    "common_s_turn_action",
    "mission_support_from_initial_leaders",
    "refine_damped_gauss_newton",
    "estimate_initial_position_multistart",
    "run_arm",
]
