#!/usr/bin/env python3
"""Variable-projected two-link Doppler-bias estimator for V30."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27


VERSION = "v30_profiled_link_bias_1.0"
BASELINE_LOCAL6 = "baseline_local_nls6"
PROFILED_LOCAL6 = "profiled_bias_local6"
PROFILED_GLOBAL = "profiled_bias_global"
ARMS = (BASELINE_LOCAL6, PROFILED_LOCAL6, PROFILED_GLOBAL)


@dataclass(frozen=True)
class ProfiledBiasMode:
    initial_position_m: np.ndarray
    link_bias_mps: np.ndarray
    residual_sse_mps2: float
    residual_rmse_mps: float
    iterations: int
    converged: bool
    position_covariance_m2: Optional[np.ndarray]
    nominal_radius95_m: Optional[float]
    link_bias_standard_error_mps: Optional[np.ndarray]


@dataclass(frozen=True)
class ProfiledBiasEstimate:
    modes: Tuple[ProfiledBiasMode, ...]
    runtime_s: float
    candidate_count: int
    refined_start_count: int
    clustered_mode_count: int
    alternative_distance_m: Optional[float]
    alternative_delta_chi2: Optional[float]

    @property
    def best(self) -> ProfiledBiasMode:
        if not self.modes:
            raise RuntimeError("empty profiled-bias estimate")
        return self.modes[0]


@dataclass(frozen=True)
class BiasEstimatorOutput:
    arm: str
    current_position_m: np.ndarray
    initial_position_m: np.ndarray
    link_bias_mps: Optional[np.ndarray]
    link_bias_standard_error_mps: Optional[np.ndarray]
    covariance_m2: Optional[np.ndarray]
    nominal_radius95_m: Optional[float]
    residual_rmse_mps: float
    runtime_s: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(self.arm)
        for name in ("current_position_m", "initial_position_m"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3-vector")
            object.__setattr__(self, name, value)
        for name in ("link_bias_mps", "link_bias_standard_error_mps"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != (2,) or not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must be a finite 2-vector")
                object.__setattr__(self, name, array)
        if self.covariance_m2 is not None:
            covariance = np.asarray(self.covariance_m2, dtype=np.float64)
            if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
                raise ValueError("covariance must be a finite 3x3 matrix")
            object.__setattr__(self, "covariance_m2", 0.5 * (covariance + covariance.T))
        for name in ("nominal_radius95_m", "residual_rmse_mps", "runtime_s"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"invalid {name}")

    def to_unscored_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "version": VERSION,
            "arm": self.arm,
            "current_position_m": self.current_position_m.tolist(),
            "initial_position_m": self.initial_position_m.tolist(),
            "link_bias_mps": (
                None if self.link_bias_mps is None else self.link_bias_mps.tolist()
            ),
            "link_bias_standard_error_mps": (
                None
                if self.link_bias_standard_error_mps is None
                else self.link_bias_standard_error_mps.tolist()
            ),
            "covariance_m2": (
                None if self.covariance_m2 is None else self.covariance_m2.tolist()
            ),
            "nominal_radius95_m": self.nominal_radius95_m,
            "residual_rmse_mps": float(self.residual_rmse_mps),
            "runtime_s": float(self.runtime_s),
            "diagnostics": dict(self.diagnostics),
        }


def output_from_unscored_dict(value: Mapping[str, Any]) -> BiasEstimatorOutput:
    return BiasEstimatorOutput(
        arm=str(value["arm"]),
        current_position_m=np.asarray(value["current_position_m"], dtype=np.float64),
        initial_position_m=np.asarray(value["initial_position_m"], dtype=np.float64),
        link_bias_mps=(
            None
            if value.get("link_bias_mps") is None
            else np.asarray(value["link_bias_mps"], dtype=np.float64)
        ),
        link_bias_standard_error_mps=(
            None
            if value.get("link_bias_standard_error_mps") is None
            else np.asarray(value["link_bias_standard_error_mps"], dtype=np.float64)
        ),
        covariance_m2=(
            None
            if value.get("covariance_m2") is None
            else np.asarray(value["covariance_m2"], dtype=np.float64)
        ),
        nominal_radius95_m=(
            None
            if value.get("nominal_radius95_m") is None
            else float(value["nominal_radius95_m"])
        ),
        residual_rmse_mps=float(value["residual_rmse_mps"]),
        runtime_s=float(value["runtime_s"]),
        diagnostics=dict(value.get("diagnostics", {})),
    )


def profiled_residual_and_jacobian(
    initial_position_m: np.ndarray,
    history: v19.OnlineDopplerHistory,
    config: v19.BatchEstimatorConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_residual, raw_jacobian = v19.residual_and_jacobian(
        initial_position_m, history, config
    )
    residual = raw_residual.reshape(history.measurement_count, 2)
    jacobian = raw_jacobian.reshape(history.measurement_count, 2, 3)
    bias = np.mean(residual, axis=0)
    profiled_residual = residual - bias[None, :]
    profiled_jacobian = jacobian - np.mean(jacobian, axis=0, keepdims=True)
    return profiled_residual.reshape(-1), profiled_jacobian.reshape(-1, 3), bias


def _mode_from_position(
    position_m: np.ndarray,
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: v19.BatchEstimatorConfig,
    iterations: int,
    converged: bool,
) -> ProfiledBiasMode:
    residual, jacobian, bias = profiled_residual_and_jacobian(
        position_m, history, config
    )
    sse = float(residual @ residual)
    rmse = float(math.sqrt(sse / max(residual.size, 1)))
    information = jacobian.T @ jacobian
    information = 0.5 * (information + information.T)
    rank = int(np.linalg.matrix_rank(information, tol=1e-10 * max(float(np.linalg.norm(information, 2)), 1e-12)))
    solution_radius = float(
        np.linalg.norm(np.asarray(position_m, dtype=np.float64) - np.asarray(center_m))
    )
    boundary_tolerance = max(1e-6, 1e-8 * float(radius_max_m))
    boundary = (
        abs(solution_radius - float(radius_min_m)) <= boundary_tolerance
        or abs(solution_radius - float(radius_max_m)) <= boundary_tolerance
    )
    dof = max(int(residual.size) - 5, 1)
    variance = max(sse / dof, float(config.measurement_sigma_mps) ** 2)
    covariance: Optional[np.ndarray]
    radius: Optional[float]
    bias_se: Optional[np.ndarray]
    if rank == 3 and not boundary:
        try:
            covariance = variance * np.linalg.inv(information)
            covariance = 0.5 * (covariance + covariance.T)
            maximum = max(float(np.max(np.linalg.eigvalsh(covariance))), 0.0)
            radius = float(v19.CHI2_3_95_SQRT * math.sqrt(maximum))

            raw_residual, raw_jacobian = v19.residual_and_jacobian(
                position_m, history, config
            )
            del raw_residual
            bias_design = np.zeros((history.measurement_count, 2, 2), dtype=np.float64)
            bias_design[:, 0, 0] = -1.0
            bias_design[:, 1, 1] = -1.0
            joint = np.column_stack((raw_jacobian, bias_design.reshape(-1, 2)))
            joint_information = joint.T @ joint
            if np.linalg.matrix_rank(joint_information) == 5:
                joint_covariance = variance * np.linalg.inv(joint_information)
                bias_se = np.sqrt(np.maximum(np.diag(joint_covariance)[3:], 0.0))
            else:
                bias_se = None
        except np.linalg.LinAlgError:
            covariance = None
            radius = None
            bias_se = None
    else:
        covariance = None
        radius = None
        bias_se = None
    return ProfiledBiasMode(
        initial_position_m=np.asarray(position_m, dtype=np.float64).copy(),
        link_bias_mps=np.asarray(bias, dtype=np.float64).copy(),
        residual_sse_mps2=sse,
        residual_rmse_mps=rmse,
        iterations=int(iterations),
        converged=bool(converged),
        position_covariance_m2=covariance,
        nominal_radius95_m=radius,
        link_bias_standard_error_mps=bias_se,
    )


def refine_profiled_bias(
    start_m: np.ndarray,
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: v19.BatchEstimatorConfig,
) -> ProfiledBiasMode:
    position = v19.project_to_shell(start_m, center_m, radius_min_m, radius_max_m)
    damping = float(config.initial_damping)
    converged = False
    iterations = 0
    for iterations in range(1, int(config.maximum_iterations) + 1):
        residual, jacobian, _ = profiled_residual_and_jacobian(
            position, history, config
        )
        cost = 0.5 * float(residual @ residual)
        gradient = jacobian.T @ residual
        if float(np.linalg.norm(gradient, ord=np.inf)) <= float(
            config.gradient_tolerance
        ):
            converged = True
            break
        normal = jacobian.T @ jacobian
        scale = np.maximum(np.diag(normal), 1e-12)
        try:
            step = np.linalg.solve(
                normal + damping * np.diag(scale), -gradient
            )
        except np.linalg.LinAlgError:
            damping = min(damping * 10.0, 1e18)
            continue
        if float(np.linalg.norm(step)) <= float(config.step_tolerance_m):
            converged = True
            break
        trial = v19.project_to_shell(
            position + step, center_m, radius_min_m, radius_max_m
        )
        trial_residual, _, _ = profiled_residual_and_jacobian(
            trial, history, config
        )
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
    return _mode_from_position(
        position,
        history,
        center_m,
        radius_min_m,
        radius_max_m,
        config,
        iterations,
        converged,
    )


def _profiled_candidate_costs(
    candidates: np.ndarray, history: v19.OnlineDopplerHistory
) -> np.ndarray:
    prediction = v19.predict_doppler_many(candidates, history)
    residual = history.doppler_measured_mps[None, :, :] - prediction
    residual -= np.mean(residual, axis=1, keepdims=True)
    return np.sum(residual * residual, axis=(1, 2))


def _separated_starts(
    candidates: np.ndarray, costs: np.ndarray, count: int, separation_m: float
) -> List[np.ndarray]:
    selected: List[np.ndarray] = []
    for index in np.argsort(costs):
        candidate = np.asarray(candidates[int(index)], dtype=np.float64)
        if all(
            float(np.linalg.norm(candidate - previous)) >= float(separation_m)
            for previous in selected
        ):
            selected.append(candidate.copy())
        if len(selected) >= int(count):
            break
    if len(selected) < 2:
        raise RuntimeError("insufficient separated profiled-bias starts")
    return selected


def _cluster_modes(
    modes: Sequence[ProfiledBiasMode], cluster_radius_m: float
) -> Tuple[ProfiledBiasMode, ...]:
    kept: List[ProfiledBiasMode] = []
    for mode in sorted(modes, key=lambda item: item.residual_sse_mps2):
        if all(
            float(np.linalg.norm(mode.initial_position_m - previous.initial_position_m))
            >= float(cluster_radius_m)
            for previous in kept
        ):
            kept.append(mode)
    if not kept:
        raise RuntimeError("profiled-bias refinement produced no modes")
    return tuple(kept)


def estimate_profiled_bias_local6(
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: v27.EvaluatorConfig,
) -> ProfiledBiasEstimate:
    started = time.perf_counter()
    midpoint = 0.5 * (float(radius_min_m) + float(radius_max_m))
    axes = np.vstack((np.eye(3), -np.eye(3)))
    starts = np.asarray(center_m)[None, :] + midpoint * axes
    batch = config.batch_config()
    modes = [
        refine_profiled_bias(
            start, history, center_m, radius_min_m, radius_max_m, batch
        )
        for start in starts
    ]
    clustered = _cluster_modes(modes, batch.mode_cluster_radius_m)
    return ProfiledBiasEstimate(
        modes=clustered[: int(batch.maximum_modes)],
        runtime_s=float(time.perf_counter() - started),
        candidate_count=0,
        refined_start_count=6,
        clustered_mode_count=len(clustered),
        alternative_distance_m=None,
        alternative_delta_chi2=None,
    )


def estimate_profiled_bias_global(
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: v27.EvaluatorConfig,
) -> ProfiledBiasEstimate:
    started = time.perf_counter()
    batch = config.batch_config()
    seed = int(config.global_candidate_seed) + int(round(history.t_s[-1]))
    candidates = np.concatenate(
        [
            v19.deterministic_shell_candidates(
                center_m,
                radius_min_m,
                radius_max_m,
                int(config.coarse_candidates),
                seed + 104_729 * sweep,
                radial_distribution="uniform_radius",
            )
            for sweep in range(int(config.coarse_sweeps))
        ],
        axis=0,
    )
    costs = _profiled_candidate_costs(candidates, history)
    starts = _separated_starts(
        candidates,
        costs,
        int(config.local_starts),
        float(batch.coarse_start_separation_m),
    )
    refined = [
        refine_profiled_bias(
            start, history, center_m, radius_min_m, radius_max_m, batch
        )
        for start in starts
    ]
    clustered = _cluster_modes(refined, batch.mode_cluster_radius_m)
    best = clustered[0]
    alternative_distance = None
    alternative_delta = None
    alternative_mode = None
    for mode in clustered[1:]:
        distance = float(np.linalg.norm(mode.initial_position_m - best.initial_position_m))
        if distance >= float(batch.alternative_separation_m):
            alternative_distance = distance
            alternative_delta = float(
                (mode.residual_sse_mps2 - best.residual_sse_mps2)
                / max(float(batch.measurement_sigma_mps) ** 2, 1e-18)
            )
            alternative_mode = mode
            break
    reported = list(clustered[: int(batch.maximum_modes)])
    if alternative_mode is not None and all(mode is not alternative_mode for mode in reported):
        reported[-1] = alternative_mode
    return ProfiledBiasEstimate(
        modes=tuple(reported),
        runtime_s=float(time.perf_counter() - started),
        candidate_count=int(candidates.shape[0]),
        refined_start_count=len(refined),
        clustered_mode_count=len(clustered),
        alternative_distance_m=alternative_distance,
        alternative_delta_chi2=alternative_delta,
    )


def _output_from_profiled(
    arm: str,
    estimate: ProfiledBiasEstimate,
    history: v19.OnlineDopplerHistory,
) -> BiasEstimatorOutput:
    best = estimate.best
    return BiasEstimatorOutput(
        arm=arm,
        current_position_m=(
            best.initial_position_m + history.dead_reckoned_displacement_m[-1]
        ),
        initial_position_m=best.initial_position_m,
        link_bias_mps=best.link_bias_mps,
        link_bias_standard_error_mps=best.link_bias_standard_error_mps,
        covariance_m2=best.position_covariance_m2,
        nominal_radius95_m=best.nominal_radius95_m,
        residual_rmse_mps=best.residual_rmse_mps,
        runtime_s=estimate.runtime_s,
        diagnostics={
            "candidate_count": estimate.candidate_count,
            "refined_start_count": estimate.refined_start_count,
            "reported_mode_count": len(estimate.modes),
            "clustered_mode_count": estimate.clustered_mode_count,
            "best_iterations": best.iterations,
            "best_converged": best.converged,
            "alternative_distance_m": estimate.alternative_distance_m,
            "alternative_delta_chi2": estimate.alternative_delta_chi2,
            "profiled_parameter_count": 2,
        },
    )


def evaluate_arm(
    arm: str,
    history: v19.OnlineDopplerHistory,
    radius_min_m: float,
    radius_max_m: float,
    config: v27.EvaluatorConfig,
) -> BiasEstimatorOutput:
    if arm not in ARMS:
        raise ValueError(arm)
    center = v19.initial_leader_centroid_from_history(history)
    if arm == BASELINE_LOCAL6:
        output = v27.evaluate_arm(
            v27.LOCAL_NLS6, history, radius_min_m, radius_max_m, config
        )
        if output.initial_position_m is None:
            raise RuntimeError("baseline local NLS lacks initial position")
        return BiasEstimatorOutput(
            arm=arm,
            current_position_m=output.current_position_m,
            initial_position_m=output.initial_position_m,
            link_bias_mps=None,
            link_bias_standard_error_mps=None,
            covariance_m2=output.covariance_m2,
            nominal_radius95_m=output.nominal_radius95_m,
            residual_rmse_mps=float(output.residual_rmse_mps),
            runtime_s=float(output.runtime_s),
            diagnostics=dict(output.diagnostics),
        )
    if arm == PROFILED_LOCAL6:
        estimate = estimate_profiled_bias_local6(
            history, center, radius_min_m, radius_max_m, config
        )
    else:
        estimate = estimate_profiled_bias_global(
            history, center, radius_min_m, radius_max_m, config
        )
    return _output_from_profiled(arm, estimate, history)


def deterministic_payload(output: BiasEstimatorOutput) -> Dict[str, Any]:
    value = output.to_unscored_dict()
    value.pop("runtime_s", None)
    return value


__all__ = [
    "VERSION",
    "ARMS",
    "BASELINE_LOCAL6",
    "PROFILED_LOCAL6",
    "PROFILED_GLOBAL",
    "BiasEstimatorOutput",
    "profiled_residual_and_jacobian",
    "refine_profiled_bias",
    "evaluate_arm",
    "output_from_unscored_dict",
    "deterministic_payload",
]
