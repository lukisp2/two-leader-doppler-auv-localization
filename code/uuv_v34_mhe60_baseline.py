#!/usr/bin/env python3
"""Arrival-cost MHE-60 baseline for the append-only V34 campaign.

Estimator-facing functions accept deployable V19 online histories and public
support only.  Truth scoring is deliberately separated from estimation.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27


VERSION = "v34_mhe60_arrival_fej_1.0"
ARM_NAME = "mhe60_arrival_fej"
CHECKPOINTS_S = (30.0, 60.0, 120.0, 240.0, 440.0)
RESERVED_SEED_START = v27.RESERVED_SEED_START
RESERVED_SEED_END = v27.RESERVED_SEED_END
FINAL_SEED_START = v27.FINAL_SEED_START
FINAL_SEED_END = v27.FINAL_SEED_END
CHI2_3_95_SQRT = v19.CHI2_3_95_SQRT


def sha256_file(path: Path) -> str:
    return v27.sha256_file(Path(path))


def assert_v34_development_seed(seed: int) -> None:
    value = int(seed)
    if RESERVED_SEED_START <= value <= FINAL_SEED_END:
        raise PermissionError(
            f"V34 refuses reserved/final seed {value}; "
            f"{RESERVED_SEED_START}..{FINAL_SEED_END} remain closed"
        )


@dataclass(frozen=True)
class MHEConfig:
    measurement_sigma_mps: float = 0.05
    horizon_samples: int = 60
    initialization_time_s: float = 30.0
    coarse_candidates: int = 4096
    coarse_sweeps: int = 2
    local_starts: int = 48
    maximum_modes: int = 12
    maximum_iterations: int = 80
    initial_damping: float = 1e-3
    gradient_tolerance: float = 1e-10
    step_tolerance_m: float = 1e-8
    candidate_seed_base: int = 27_001

    def __post_init__(self) -> None:
        if not math.isclose(
            float(self.initialization_time_s), 30.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("V34 initialization time is frozen at 30 s")
        if int(self.horizon_samples) != 60:
            raise ValueError("V34 horizon is frozen at exactly 60 samples")
        if int(self.coarse_candidates) < 64 or int(self.coarse_sweeps) < 1:
            raise ValueError("invalid V34 initializer search size")
        if int(self.local_starts) < 2 or int(self.maximum_modes) < 2:
            raise ValueError("invalid V34 initializer refinement size")
        if int(self.maximum_iterations) != 80:
            raise ValueError("V34 maximum iterations are frozen at 80")
        positive = (
            self.measurement_sigma_mps,
            self.initial_damping,
            self.gradient_tolerance,
            self.step_tolerance_m,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("V34 numeric constants must be finite and positive")

    def batch_config(self) -> v19.BatchEstimatorConfig:
        return v19.BatchEstimatorConfig(
            coarse_candidates=int(self.coarse_candidates),
            coarse_sweeps=int(self.coarse_sweeps),
            local_starts=int(self.local_starts),
            maximum_modes=int(self.maximum_modes),
            maximum_iterations=int(self.maximum_iterations),
            initial_damping=float(self.initial_damping),
            gradient_tolerance=float(self.gradient_tolerance),
            step_tolerance_m=float(self.step_tolerance_m),
            measurement_sigma_mps=float(self.measurement_sigma_mps),
            gate_mode="raw",
            candidate_radial_distribution="uniform_radius",
        )


@dataclass
class ArrivalCost:
    information: np.ndarray
    information_vector: np.ndarray
    constant: float
    sample_count: int

    @classmethod
    def zero(cls) -> "ArrivalCost":
        return cls(
            information=np.zeros((3, 3), dtype=np.float64),
            information_vector=np.zeros(3, dtype=np.float64),
            constant=0.0,
            sample_count=0,
        )

    def add_linearized_factor(
        self,
        residual: np.ndarray,
        jacobian: np.ndarray,
        linearization_position_m: np.ndarray,
    ) -> None:
        value = np.asarray(residual, dtype=np.float64).reshape(-1)
        derivative = np.asarray(jacobian, dtype=np.float64)
        position = np.asarray(linearization_position_m, dtype=np.float64).reshape(3)
        if value.shape != (2,) or derivative.shape != (2, 3):
            raise ValueError("one V34 outgoing Doppler factor must have shape 2 by 3")
        if not (
            np.all(np.isfinite(value))
            and np.all(np.isfinite(derivative))
            and np.all(np.isfinite(position))
        ):
            raise ValueError("arrival update inputs must be finite")
        offset = value - derivative @ position
        self.information += derivative.T @ derivative
        self.information = 0.5 * (self.information + self.information.T)
        self.information_vector -= derivative.T @ offset
        self.constant += float(offset @ offset)
        self.sample_count += 1

    def quadratic_value(self, position_m: np.ndarray) -> float:
        position = np.asarray(position_m, dtype=np.float64).reshape(3)
        return float(
            position @ self.information @ position
            - 2.0 * self.information_vector @ position
            + self.constant
        )

    def gradient(self, position_m: np.ndarray) -> np.ndarray:
        position = np.asarray(position_m, dtype=np.float64).reshape(3)
        return self.information @ position - self.information_vector


@dataclass(frozen=True)
class MHECheckpointOutput:
    checkpoint_s: float
    current_position_m: np.ndarray
    initial_position_m: np.ndarray
    covariance_m2: Optional[np.ndarray]
    nominal_radius95_m: Optional[float]
    full_prefix_residual_rmse_mps: float
    arrival_objective_rmse_mps: float
    checkpoint_update_runtime_s: float
    cumulative_runtime_s: float
    post_init_update_latency_p99_s: Optional[float]
    post_init_update_latency_max_s: Optional[float]
    post_init_update_latencies_s: Tuple[float, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if float(self.checkpoint_s) not in CHECKPOINTS_S:
            raise ValueError("unknown V34 checkpoint")
        for name in ("current_position_m", "initial_position_m"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite three-vector")
            object.__setattr__(self, name, array)
        if self.covariance_m2 is not None:
            covariance = np.asarray(self.covariance_m2, dtype=np.float64)
            if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
                raise ValueError("covariance_m2 must be a finite 3 by 3 matrix")
            object.__setattr__(self, "covariance_m2", 0.5 * (covariance + covariance.T))
        for name in (
            "nominal_radius95_m",
            "full_prefix_residual_rmse_mps",
            "arrival_objective_rmse_mps",
            "checkpoint_update_runtime_s",
            "cumulative_runtime_s",
            "post_init_update_latency_p99_s",
            "post_init_update_latency_max_s",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when present")
        latencies = tuple(float(value) for value in self.post_init_update_latencies_s)
        if any(not math.isfinite(value) or value < 0.0 for value in latencies):
            raise ValueError("post-init update latencies must be finite and non-negative")
        object.__setattr__(self, "post_init_update_latencies_s", latencies)

    def to_unscored_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "version": VERSION,
            "arm": ARM_NAME,
            "checkpoint_s": float(self.checkpoint_s),
            "current_position_m": self.current_position_m.tolist(),
            "initial_position_m": self.initial_position_m.tolist(),
            "covariance_m2": (
                None if self.covariance_m2 is None else self.covariance_m2.tolist()
            ),
            "nominal_radius95_m": self.nominal_radius95_m,
            "full_prefix_residual_rmse_mps": float(self.full_prefix_residual_rmse_mps),
            "arrival_objective_rmse_mps": float(self.arrival_objective_rmse_mps),
            "checkpoint_update_runtime_s": float(self.checkpoint_update_runtime_s),
            "cumulative_runtime_s": float(self.cumulative_runtime_s),
            "post_init_update_latency_p99_s": self.post_init_update_latency_p99_s,
            "post_init_update_latency_max_s": self.post_init_update_latency_max_s,
            "post_init_update_latencies_s": list(self.post_init_update_latencies_s),
            "diagnostics": dict(self.diagnostics),
        }


def output_from_unscored_dict(value: Mapping[str, Any]) -> MHECheckpointOutput:
    return MHECheckpointOutput(
        checkpoint_s=float(value["checkpoint_s"]),
        current_position_m=np.asarray(value["current_position_m"], dtype=np.float64),
        initial_position_m=np.asarray(value["initial_position_m"], dtype=np.float64),
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
        full_prefix_residual_rmse_mps=float(value["full_prefix_residual_rmse_mps"]),
        arrival_objective_rmse_mps=float(value["arrival_objective_rmse_mps"]),
        checkpoint_update_runtime_s=float(value["checkpoint_update_runtime_s"]),
        cumulative_runtime_s=float(value["cumulative_runtime_s"]),
        post_init_update_latency_p99_s=(
            None
            if value.get("post_init_update_latency_p99_s") is None
            else float(value["post_init_update_latency_p99_s"])
        ),
        post_init_update_latency_max_s=(
            None
            if value.get("post_init_update_latency_max_s") is None
            else float(value["post_init_update_latency_max_s"])
        ),
        post_init_update_latencies_s=tuple(
            float(item) for item in value.get("post_init_update_latencies_s", [])
        ),
        diagnostics=dict(value.get("diagnostics", {})),
    )


def _nearest_rank(values: Sequence[float], probability: float) -> Optional[float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    rank = max(1, int(math.ceil(float(probability) * len(finite))))
    return finite[min(rank - 1, len(finite) - 1)]


def _window_at(history: v19.OnlineDopplerHistory, endpoint_s: float, count: int) -> v19.OnlineDopplerHistory:
    endpoint = float(endpoint_s)
    mask = np.logical_and(
        history.t_s <= endpoint + 1e-12,
        history.t_s > endpoint - float(count) + 1e-12,
    )
    if not np.any(mask):
        raise ValueError("V34 active window is empty")
    return history.take(mask)


def _objective_terms(
    position_m: np.ndarray,
    window: v19.OnlineDopplerHistory,
    arrival: ArrivalCost,
    batch_config: v19.BatchEstimatorConfig,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual, jacobian = v19.residual_and_jacobian(position_m, window, batch_config)
    arrival_value = arrival.quadratic_value(position_m)
    scale = max(1.0, abs(float(arrival.constant)))
    if arrival_value < -1e-8 * scale:
        raise FloatingPointError("arrival quadratic became materially negative")
    arrival_value = max(arrival_value, 0.0)
    cost = 0.5 * (arrival_value + float(residual @ residual))
    gradient = arrival.gradient(position_m) + jacobian.T @ residual
    normal = arrival.information + jacobian.T @ jacobian
    normal = 0.5 * (normal + normal.T)
    return cost, gradient, normal, residual, jacobian


def _refine_mhe(
    start_m: np.ndarray,
    window: v19.OnlineDopplerHistory,
    arrival: ArrivalCost,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: MHEConfig,
) -> Tuple[np.ndarray, int, bool]:
    position = v19.project_to_shell(start_m, center_m, radius_min_m, radius_max_m)
    batch_config = config.batch_config()
    damping = float(config.initial_damping)
    converged = False
    iterations = 0
    for iterations in range(1, int(config.maximum_iterations) + 1):
        cost, gradient, normal, _, _ = _objective_terms(
            position, window, arrival, batch_config
        )
        if float(np.linalg.norm(gradient, ord=np.inf)) <= float(config.gradient_tolerance):
            converged = True
            break
        diagonal = np.maximum(np.diag(normal), 1e-12)
        system = normal + damping * np.diag(diagonal)
        try:
            step = np.linalg.solve(system, -gradient)
        except np.linalg.LinAlgError:
            damping = min(damping * 10.0, 1e18)
            continue
        if float(np.linalg.norm(step)) <= float(config.step_tolerance_m):
            converged = True
            break
        trial = v19.project_to_shell(
            position + step, center_m, radius_min_m, radius_max_m
        )
        trial_cost, _, _, _, _ = _objective_terms(
            trial, window, arrival, batch_config
        )
        if trial_cost < cost:
            improvement = cost - trial_cost
            position = trial
            damping = max(damping * 0.3, 1e-15)
            if improvement <= 1e-14 * max(1.0, cost):
                converged = True
                break
        else:
            damping = min(damping * 10.0, 1e18)
    if not np.all(np.isfinite(position)):
        raise FloatingPointError("V34 MHE produced a non-finite position")
    return np.asarray(position, dtype=np.float64), int(iterations), bool(converged)


def _covariance_diagnostics(
    position_m: np.ndarray,
    window: v19.OnlineDopplerHistory,
    arrival: ArrivalCost,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: MHEConfig,
) -> Tuple[Optional[np.ndarray], Optional[float], float, Dict[str, Any]]:
    _, _, normal, residual, _ = _objective_terms(
        position_m, window, arrival, config.batch_config()
    )
    arrival_value = max(arrival.quadratic_value(position_m), 0.0)
    approximate_sse = float(arrival_value + residual @ residual)
    total_samples = int(arrival.sample_count + window.measurement_count)
    dof = max(2 * total_samples - 3, 1)
    variance = max(
        approximate_sse / float(dof), float(config.measurement_sigma_mps) ** 2
    )
    try:
        eigenvalues = np.linalg.eigvalsh(normal)
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(3, dtype=np.float64)
    largest = float(max(np.max(eigenvalues), 0.0))
    threshold = max(1e-12, 1e-10 * largest)
    rank = int(np.sum(eigenvalues > threshold))
    condition = (
        float(largest / float(eigenvalues[0]))
        if rank == 3 and float(eigenvalues[0]) > 0.0
        else float("inf")
    )
    radius_from_center = float(
        np.linalg.norm(
            np.asarray(position_m, dtype=np.float64)
            - np.asarray(center_m, dtype=np.float64)
        )
    )
    boundary_tolerance = max(1e-6, 1e-8 * float(radius_max_m))
    at_boundary = bool(
        abs(radius_from_center - float(radius_min_m)) <= boundary_tolerance
        or abs(radius_from_center - float(radius_max_m)) <= boundary_tolerance
    )
    valid = bool(
        rank == 3
        and math.isfinite(condition)
        and condition <= 1e10
        and not at_boundary
    )
    covariance: Optional[np.ndarray]
    radius95: Optional[float]
    if valid:
        try:
            covariance = variance * np.linalg.inv(normal)
            covariance = 0.5 * (covariance + covariance.T)
            maximum_variance = float(max(np.max(np.linalg.eigvalsh(covariance)), 0.0))
            radius95 = float(CHI2_3_95_SQRT * math.sqrt(maximum_variance))
            if not np.all(np.isfinite(covariance)) or not math.isfinite(radius95):
                covariance = None
                radius95 = None
                valid = False
        except np.linalg.LinAlgError:
            covariance = None
            radius95 = None
            valid = False
    else:
        covariance = None
        radius95 = None
    diagnostics = {
        "local_information_rank": rank,
        "local_information_condition": condition if math.isfinite(condition) else None,
        "local_covariance_valid": bool(valid),
        "shell_boundary": at_boundary,
        "variance_scale_mps2": float(variance),
        "approximate_sse_mps2": approximate_sse,
        "arrival_information_trace": float(np.trace(arrival.information)),
        "arrival_information_min_eigenvalue": float(
            np.min(np.linalg.eigvalsh(arrival.information))
        ),
    }
    objective_rmse = float(math.sqrt(approximate_sse / max(2 * total_samples, 1)))
    return covariance, radius95, objective_rmse, diagnostics


def _make_output(
    checkpoint_s: float,
    position_m: np.ndarray,
    history: v19.OnlineDopplerHistory,
    window: v19.OnlineDopplerHistory,
    arrival: ArrivalCost,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: MHEConfig,
    checkpoint_runtime_s: float,
    cumulative_runtime_s: float,
    post_init_latencies_s: Sequence[float],
    iterations: int,
    converged: bool,
    initializer_diagnostics: Mapping[str, Any],
) -> MHECheckpointOutput:
    prefix = history.prefix(float(checkpoint_s))
    covariance, radius95, objective_rmse, covariance_diagnostics = (
        _covariance_diagnostics(
            position_m,
            window,
            arrival,
            center_m,
            radius_min_m,
            radius_max_m,
            config,
        )
    )
    full_residual = prefix.doppler_measured_mps - v19.predict_doppler(
        position_m, prefix
    )
    full_rmse = float(math.sqrt(float(np.mean(full_residual * full_residual))))
    current = (
        np.asarray(position_m, dtype=np.float64)
        + prefix.dead_reckoned_displacement_m[-1]
    )
    diagnostics = {
        "window_sample_count": int(window.measurement_count),
        "marginalized_sample_count": int(arrival.sample_count),
        "total_sample_count": int(window.measurement_count + arrival.sample_count),
        "post_init_update_count": int(len(post_init_latencies_s)),
        "last_iterations": int(iterations),
        "last_converged": bool(converged),
        "initializer": dict(initializer_diagnostics),
        **covariance_diagnostics,
    }
    return MHECheckpointOutput(
        checkpoint_s=float(checkpoint_s),
        current_position_m=current,
        initial_position_m=np.asarray(position_m, dtype=np.float64),
        covariance_m2=covariance,
        nominal_radius95_m=radius95,
        full_prefix_residual_rmse_mps=full_rmse,
        arrival_objective_rmse_mps=objective_rmse,
        checkpoint_update_runtime_s=float(checkpoint_runtime_s),
        cumulative_runtime_s=float(cumulative_runtime_s),
        post_init_update_latency_p99_s=_nearest_rank(post_init_latencies_s, 0.99),
        post_init_update_latency_max_s=(
            None if not post_init_latencies_s else float(max(post_init_latencies_s))
        ),
        post_init_update_latencies_s=tuple(float(value) for value in post_init_latencies_s),
        diagnostics=diagnostics,
    )


def evaluate_mhe_history(
    history: v19.OnlineDopplerHistory,
    radius_min_m: float,
    radius_max_m: float,
    config: MHEConfig,
) -> Tuple[MHECheckpointOutput, ...]:
    """Evaluate one causal episode without accepting truth or episode identity."""

    if not (0.0 <= float(radius_min_m) < float(radius_max_m)):
        raise ValueError("invalid V34 public shell support")
    required = np.arange(1.0, 441.0, dtype=np.float64)
    available = history.t_s[history.t_s <= 440.0 + 1e-12]
    if available.shape != required.shape or not np.allclose(
        available, required, rtol=0.0, atol=1e-12
    ):
        raise ValueError("V34 requires the frozen one-hertz history from 1 through 440 s")
    causal = history.prefix(440.0)
    center = v19.initial_leader_centroid_from_history(causal)
    initialization_time = float(config.initialization_time_s)
    initialization_history = causal.prefix(initialization_time)
    estimate = v19.estimate_initial_position_multistart(
        initialization_history,
        center,
        float(radius_min_m),
        float(radius_max_m),
        config.batch_config(),
        candidate_seed=int(config.candidate_seed_base) + int(round(initialization_time)),
    )
    position = np.asarray(estimate.best.initial_position_m, dtype=np.float64)
    initialization_runtime = float(estimate.runtime_s)
    cumulative_runtime = initialization_runtime
    post_init_latencies: list[float] = []
    arrival = ArrivalCost.zero()
    initializer_diagnostics = {
        "candidate_count": int(estimate.candidate_count),
        "refined_start_count": int(estimate.refined_start_count),
        "reported_mode_count": int(len(estimate.modes)),
        "clustered_mode_count": int(estimate.clustered_mode_count),
        "best_iterations": int(estimate.best.iterations),
        "best_converged": bool(estimate.best.converged),
    }
    outputs: Dict[float, MHECheckpointOutput] = {}
    initial_window = _window_at(causal, initialization_time, int(config.horizon_samples))
    outputs[initialization_time] = _make_output(
        initialization_time,
        position,
        causal,
        initial_window,
        arrival,
        center,
        float(radius_min_m),
        float(radius_max_m),
        config,
        initialization_runtime,
        cumulative_runtime,
        post_init_latencies,
        int(estimate.best.iterations),
        bool(estimate.best.converged),
        initializer_diagnostics,
    )

    checkpoint_set = set(CHECKPOINTS_S)
    batch_config = config.batch_config()
    for endpoint_integer in range(31, 441):
        endpoint = float(endpoint_integer)
        if endpoint_integer > int(config.horizon_samples):
            outgoing_time = float(endpoint_integer - int(config.horizon_samples))
            matches = np.flatnonzero(
                np.isclose(causal.t_s, outgoing_time, rtol=0.0, atol=1e-12)
            )
            if matches.size != 1:
                raise ValueError("V34 outgoing factor has no unique sample")
            outgoing = causal.take(int(matches[0]))
            residual, jacobian = v19.residual_and_jacobian(
                position, outgoing, batch_config
            )
            arrival.add_linearized_factor(residual, jacobian, position)
        window = _window_at(causal, endpoint, int(config.horizon_samples))
        started = time.perf_counter()
        position, iterations, converged = _refine_mhe(
            position,
            window,
            arrival,
            center,
            float(radius_min_m),
            float(radius_max_m),
            config,
        )
        update_runtime = float(time.perf_counter() - started)
        cumulative_runtime += update_runtime
        post_init_latencies.append(update_runtime)
        if endpoint in checkpoint_set:
            outputs[endpoint] = _make_output(
                endpoint,
                position,
                causal,
                window,
                arrival,
                center,
                float(radius_min_m),
                float(radius_max_m),
                config,
                update_runtime,
                cumulative_runtime,
                post_init_latencies,
                iterations,
                converged,
                initializer_diagnostics,
            )
    if set(outputs) != set(CHECKPOINTS_S):
        raise RuntimeError("V34 did not produce every frozen checkpoint")
    return tuple(outputs[value] for value in CHECKPOINTS_S)


def score_output(
    output: MHECheckpointOutput,
    truth: v19.ReplayTruthDiagnostics,
    history: v19.OnlineDopplerHistory,
) -> Dict[str, Any]:
    """Score a persisted output after the runner opens diagnostic labels."""

    endpoint_s = float(output.checkpoint_s)
    endpoint_index = int(round(endpoint_s)) - 1
    if endpoint_index < 0 or endpoint_index >= truth.true_displacement_m.shape[0]:
        raise ValueError("V34 checkpoint does not map to the truth grid")
    prefix = history.prefix(endpoint_s)
    current_truth = (
        truth.initial_follower_position_m + truth.true_displacement_m[endpoint_index]
    )
    endpoint_error = float(np.linalg.norm(output.current_position_m - current_truth))
    initial_error = float(
        np.linalg.norm(output.initial_position_m - truth.initial_follower_position_m)
    )
    radius = output.nominal_radius95_m
    return {
        "endpoint_s": endpoint_s,
        "endpoint_position_error_m": endpoint_error,
        "initial_position_error_m": initial_error,
        "success_lt_7m": bool(endpoint_error < 7.0),
        "success_le_7m": bool(endpoint_error <= 7.0),
        "nominal_radius95_covers": (
            None if radius is None else bool(endpoint_error <= float(radius))
        ),
        "false_confidence_radius_lt_7_error_gt_7": bool(
            radius is not None and float(radius) < 7.0 and endpoint_error > 7.0
        ),
        "prefix_sample_count": int(prefix.measurement_count),
    }


def deterministic_payload(output: MHECheckpointOutput) -> Dict[str, Any]:
    payload = output.to_unscored_dict()
    for key in (
        "checkpoint_update_runtime_s",
        "cumulative_runtime_s",
        "post_init_update_latency_p99_s",
        "post_init_update_latency_max_s",
        "post_init_update_latencies_s",
    ):
        payload.pop(key, None)
    return payload


def config_to_dict(config: MHEConfig) -> Dict[str, Any]:
    return asdict(config)


__all__ = [
    "ARM_NAME",
    "ArrivalCost",
    "CHECKPOINTS_S",
    "FINAL_SEED_END",
    "FINAL_SEED_START",
    "MHECheckpointOutput",
    "MHEConfig",
    "RESERVED_SEED_END",
    "RESERVED_SEED_START",
    "VERSION",
    "assert_v34_development_seed",
    "config_to_dict",
    "deterministic_payload",
    "evaluate_mhe_history",
    "output_from_unscored_dict",
    "score_output",
    "sha256_file",
]
