#!/usr/bin/env python3
"""Truth-isolated estimator baselines and ablations for V27.

The estimator-facing functions in this module accept only the V19 online
allowlist plus public support information.  Truth scoring is deliberately a
separate function so the runner can persist unscored outputs before opening
diagnostic labels.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import uuv_v19_observability as v19


VERSION = "v27_publication_estimator_baselines_1.0"

LEGACY_PF = "legacy_pf"
PF_LW_4096 = "pf_lw_4096"
PF_LW_16384 = "pf_lw_16384"
EKF_STATIC = "ekf_static"
LOCAL_NLS6 = "local_nls6"
COARSE_ONLY_8192 = "coarse_only_8192"
GLOBAL_WINDOW60 = "global_window60"
GLOBAL_FULL = "global_full"

ARM_NAMES = (
    LEGACY_PF,
    PF_LW_4096,
    PF_LW_16384,
    EKF_STATIC,
    LOCAL_NLS6,
    COARSE_ONLY_8192,
    GLOBAL_WINDOW60,
    GLOBAL_FULL,
)

RESERVED_SEED_START = 49_900
RESERVED_SEED_END = 49_999
FINAL_SEED_START = 50_000
FINAL_SEED_END = 50_999
CHI2_3_95_SQRT = v19.CHI2_3_95_SQRT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_v27_development_seed(seed: int) -> None:
    value = int(seed)
    if RESERVED_SEED_START <= value <= FINAL_SEED_END:
        raise PermissionError(
            f"V27 refuses reserved/final seed {value}; "
            f"{RESERVED_SEED_START}..{FINAL_SEED_END} remain closed"
        )


@dataclass(frozen=True)
class EvaluatorConfig:
    measurement_sigma_mps: float = 0.05
    coarse_candidates: int = 4096
    coarse_sweeps: int = 2
    local_starts: int = 48
    maximum_modes: int = 12
    window_s: float = 60.0
    pf_particles_small: int = 4096
    pf_particles_large: int = 16384
    pf_ess_fraction: float = 0.5
    pf_kernel_h: float = 0.12
    global_candidate_seed: int = 27_001
    window_candidate_seed: int = 27_001
    pf_design_seed: int = 27_201

    def __post_init__(self) -> None:
        positive = (
            self.measurement_sigma_mps,
            self.window_s,
            self.pf_ess_fraction,
            self.pf_kernel_h,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("V27 numeric constants must be finite and positive")
        if not 0.0 < float(self.pf_ess_fraction) <= 1.0:
            raise ValueError("pf_ess_fraction must lie in (0, 1]")
        if not 0.0 < float(self.pf_kernel_h) < 1.0:
            raise ValueError("pf_kernel_h must lie in (0, 1)")
        if int(self.coarse_candidates) < 64 or int(self.coarse_sweeps) < 1:
            raise ValueError("invalid global-search size")
        if int(self.local_starts) < 2 or int(self.maximum_modes) < 2:
            raise ValueError("invalid local-search size")
        if int(self.pf_particles_small) < 64:
            raise ValueError("small PF must contain at least 64 particles")
        if int(self.pf_particles_large) <= int(self.pf_particles_small):
            raise ValueError("large PF must be larger than small PF")

    def batch_config(self) -> v19.BatchEstimatorConfig:
        return v19.BatchEstimatorConfig(
            coarse_candidates=int(self.coarse_candidates),
            coarse_sweeps=int(self.coarse_sweeps),
            local_starts=int(self.local_starts),
            maximum_modes=int(self.maximum_modes),
            measurement_sigma_mps=float(self.measurement_sigma_mps),
            gate_mode="raw",
            candidate_radial_distribution="uniform_radius",
        )


@dataclass(frozen=True)
class EstimatorOutput:
    arm: str
    current_position_m: np.ndarray
    initial_position_m: Optional[np.ndarray]
    covariance_m2: Optional[np.ndarray]
    nominal_radius95_m: Optional[float]
    residual_rmse_mps: Optional[float]
    runtime_s: Optional[float]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.arm not in ARM_NAMES:
            raise ValueError(f"unknown V27 arm {self.arm!r}")
        current = np.asarray(self.current_position_m, dtype=np.float64)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise ValueError("current_position_m must be a finite three-vector")
        object.__setattr__(self, "current_position_m", current)
        if self.initial_position_m is not None:
            initial = np.asarray(self.initial_position_m, dtype=np.float64)
            if initial.shape != (3,) or not np.all(np.isfinite(initial)):
                raise ValueError("initial_position_m must be a finite three-vector")
            object.__setattr__(self, "initial_position_m", initial)
        if self.covariance_m2 is not None:
            covariance = np.asarray(self.covariance_m2, dtype=np.float64)
            if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
                raise ValueError("covariance_m2 must be a finite 3x3 matrix")
            covariance = 0.5 * (covariance + covariance.T)
            object.__setattr__(self, "covariance_m2", covariance)
        for name, value in (
            ("nominal_radius95_m", self.nominal_radius95_m),
            ("residual_rmse_mps", self.residual_rmse_mps),
            ("runtime_s", self.runtime_s),
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when present")

    def to_unscored_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "version": VERSION,
            "arm": self.arm,
            "current_position_m": self.current_position_m.tolist(),
            "initial_position_m": (
                None
                if self.initial_position_m is None
                else np.asarray(self.initial_position_m).tolist()
            ),
            "covariance_m2": (
                None
                if self.covariance_m2 is None
                else np.asarray(self.covariance_m2).tolist()
            ),
            "nominal_radius95_m": self.nominal_radius95_m,
            "residual_rmse_mps": self.residual_rmse_mps,
            "runtime_s": self.runtime_s,
            "diagnostics": dict(self.diagnostics),
        }


def output_from_unscored_dict(value: Mapping[str, Any]) -> EstimatorOutput:
    return EstimatorOutput(
        arm=str(value["arm"]),
        current_position_m=np.asarray(value["current_position_m"], dtype=np.float64),
        initial_position_m=(
            None
            if value.get("initial_position_m") is None
            else np.asarray(value["initial_position_m"], dtype=np.float64)
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
        residual_rmse_mps=(
            None
            if value.get("residual_rmse_mps") is None
            else float(value["residual_rmse_mps"])
        ),
        runtime_s=None if value.get("runtime_s") is None else float(value["runtime_s"]),
        diagnostics=dict(value.get("diagnostics", {})),
    )


def _radius_from_covariance(covariance_m2: np.ndarray) -> Optional[float]:
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return None
    covariance = 0.5 * (covariance + covariance.T)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        return None
    if float(np.min(eigenvalues)) < -1e-8:
        return None
    return float(CHI2_3_95_SQRT * math.sqrt(max(float(np.max(eigenvalues)), 0.0)))


def _residual_rmse(initial_position_m: np.ndarray, history: v19.OnlineDopplerHistory) -> float:
    residual = history.doppler_measured_mps - v19.predict_doppler(initial_position_m, history)
    return float(math.sqrt(float(np.mean(residual * residual))))


def last_window(history: v19.OnlineDopplerHistory, window_s: float) -> v19.OnlineDopplerHistory:
    end = float(history.t_s[-1])
    # On the frozen one-hertz grid, a 60 s window ending at t contains exactly
    # t-59, ..., t (60 measurements), not the additional sample at t-60.
    mask = history.t_s > end - float(window_s) + 1e-12
    if not np.any(mask):
        raise ValueError("history window is empty")
    return history.take(mask)


def _batch_output(
    arm: str,
    estimate: v19.BatchEstimate,
    endpoint_displacement_m: np.ndarray,
) -> EstimatorOutput:
    best = estimate.best
    covariance = (
        np.asarray(best.local_covariance_m2, dtype=np.float64)
        if bool(best.local_covariance_valid)
        else None
    )
    radius = float(best.local_radius95_m) if bool(best.local_covariance_valid) else None
    initial = np.asarray(best.initial_position_m, dtype=np.float64)
    return EstimatorOutput(
        arm=arm,
        initial_position_m=initial,
        current_position_m=initial + np.asarray(endpoint_displacement_m, dtype=np.float64),
        covariance_m2=covariance,
        nominal_radius95_m=radius,
        residual_rmse_mps=float(best.residual_rmse_mps),
        runtime_s=float(estimate.runtime_s),
        diagnostics={
            "candidate_count": int(estimate.candidate_count),
            "refined_start_count": int(estimate.refined_start_count),
            "reported_mode_count": len(estimate.modes),
            "clustered_mode_count": int(estimate.clustered_mode_count),
            "best_converged": bool(best.converged),
            "best_iterations": int(best.iterations),
            "local_covariance_valid": bool(best.local_covariance_valid),
            "alternative_distance_m": estimate.alternative_distance_m,
            "alternative_delta_chi2": estimate.alternative_delta_chi2,
        },
    )


def _evaluate_global(
    arm: str,
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: EvaluatorConfig,
) -> EstimatorOutput:
    if arm == GLOBAL_WINDOW60:
        estimator_history = last_window(history, float(config.window_s))
        candidate_seed = int(config.window_candidate_seed) + int(round(history.t_s[-1]))
    else:
        estimator_history = history
        candidate_seed = int(config.global_candidate_seed) + int(round(history.t_s[-1]))
    estimate = v19.estimate_initial_position_multistart(
        estimator_history,
        center_m,
        radius_min_m,
        radius_max_m,
        config.batch_config(),
        candidate_seed=candidate_seed,
    )
    return _batch_output(arm, estimate, history.dead_reckoned_displacement_m[-1])


def _evaluate_local_nls6(
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: EvaluatorConfig,
) -> EstimatorOutput:
    started = time.perf_counter()
    midpoint = 0.5 * (float(radius_min_m) + float(radius_max_m))
    axes = np.vstack((np.eye(3, dtype=np.float64), -np.eye(3, dtype=np.float64)))
    starts = np.asarray(center_m, dtype=np.float64)[None, :] + midpoint * axes
    batch_config = config.batch_config()
    modes = [
        v19.refine_damped_gauss_newton(
            start,
            history,
            center_m,
            radius_min_m,
            radius_max_m,
            batch_config,
        )
        for start in starts
    ]
    best = min(modes, key=lambda mode: mode.residual_sse_mps2)
    runtime = float(time.perf_counter() - started)
    covariance = best.local_covariance_m2 if best.local_covariance_valid else None
    initial = np.asarray(best.initial_position_m, dtype=np.float64)
    return EstimatorOutput(
        arm=LOCAL_NLS6,
        current_position_m=initial + history.dead_reckoned_displacement_m[-1],
        initial_position_m=initial,
        covariance_m2=covariance,
        nominal_radius95_m=(
            float(best.local_radius95_m) if best.local_covariance_valid else None
        ),
        residual_rmse_mps=float(best.residual_rmse_mps),
        runtime_s=runtime,
        diagnostics={
            "start_count": 6,
            "converged_count": int(sum(bool(mode.converged) for mode in modes)),
            "best_iterations": int(best.iterations),
            "local_covariance_valid": bool(best.local_covariance_valid),
        },
    )


def _evaluate_coarse_only(
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: EvaluatorConfig,
) -> EstimatorOutput:
    started = time.perf_counter()
    candidates = np.concatenate(
        [
            v19.deterministic_shell_candidates(
                center_m,
                radius_min_m,
                radius_max_m,
                int(config.coarse_candidates),
                int(config.global_candidate_seed)
                + int(round(history.t_s[-1]))
                + 104_729 * sweep,
                radial_distribution="uniform_radius",
            )
            for sweep in range(int(config.coarse_sweeps))
        ],
        axis=0,
    )
    prediction = v19.predict_doppler_many(candidates, history)
    residual = history.doppler_measured_mps[None, :, :] - prediction
    costs = np.sum(residual * residual, axis=(1, 2))
    best_index = int(np.argmin(costs))
    initial = candidates[best_index].copy()
    runtime = float(time.perf_counter() - started)
    return EstimatorOutput(
        arm=COARSE_ONLY_8192,
        current_position_m=initial + history.dead_reckoned_displacement_m[-1],
        initial_position_m=initial,
        covariance_m2=None,
        nominal_radius95_m=None,
        residual_rmse_mps=float(math.sqrt(float(costs[best_index]) / residual.shape[1] / 2.0)),
        runtime_s=runtime,
        diagnostics={"candidate_count": int(candidates.shape[0]), "local_refinement": False},
    )


def _uniform_radius_prior_variance(radius_min_m: float, radius_max_m: float) -> float:
    lower = float(radius_min_m)
    upper = float(radius_max_m)
    expected_radius_squared = (upper**3 - lower**3) / (3.0 * (upper - lower))
    return float(expected_radius_squared / 3.0)


def _evaluate_ekf(
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: EvaluatorConfig,
) -> EstimatorOutput:
    started = time.perf_counter()
    mean = np.asarray(center_m, dtype=np.float64).copy()
    covariance = np.eye(3, dtype=np.float64) * _uniform_radius_prior_variance(
        radius_min_m, radius_max_m
    )
    measurement_covariance = (
        float(config.measurement_sigma_mps) ** 2 * np.eye(2, dtype=np.float64)
    )
    identity = np.eye(3, dtype=np.float64)
    batch_config = config.batch_config()
    innovation_norms = []
    for index in range(history.measurement_count):
        one = history.take(index)
        residual, residual_jacobian = v19.residual_and_jacobian(mean, one, batch_config)
        observation_jacobian = -residual_jacobian
        innovation_covariance = (
            observation_jacobian @ covariance @ observation_jacobian.T
            + measurement_covariance
        )
        try:
            gain = np.linalg.solve(
                innovation_covariance,
                observation_jacobian @ covariance,
            ).T
        except np.linalg.LinAlgError:
            gain = covariance @ observation_jacobian.T @ np.linalg.pinv(
                innovation_covariance
            )
        mean = mean + gain @ residual
        correction = identity - gain @ observation_jacobian
        covariance = (
            correction @ covariance @ correction.T
            + gain @ measurement_covariance @ gain.T
        )
        covariance = 0.5 * (covariance + covariance.T)
        innovation_norms.append(float(np.linalg.norm(residual)))
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise FloatingPointError("static EKF produced non-finite state")
    runtime = float(time.perf_counter() - started)
    radius = _radius_from_covariance(covariance)
    return EstimatorOutput(
        arm=EKF_STATIC,
        current_position_m=mean + history.dead_reckoned_displacement_m[-1],
        initial_position_m=mean,
        covariance_m2=covariance,
        nominal_radius95_m=radius,
        residual_rmse_mps=_residual_rmse(mean, history),
        runtime_s=runtime,
        diagnostics={
            "measurement_updates": int(history.measurement_count),
            "mean_innovation_norm_mps": float(np.mean(innovation_norms)),
            "prior_variance_axis_m2": _uniform_radius_prior_variance(
                radius_min_m, radius_max_m
            ),
        },
    )


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / float(np.sum(normalized))
    positions = (float(rng.random()) + np.arange(normalized.size)) / normalized.size
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def _project_particles_to_shell(
    particles: np.ndarray,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
) -> np.ndarray:
    center = np.asarray(center_m, dtype=np.float64).reshape(1, 3)
    offset = np.asarray(particles, dtype=np.float64) - center
    radii = np.linalg.norm(offset, axis=1)
    zero = radii <= 1e-12
    if np.any(zero):
        offset[zero] = np.array([float(radius_min_m), 0.0, 0.0])
        radii[zero] = float(radius_min_m)
    clipped = np.clip(radii, float(radius_min_m), float(radius_max_m))
    return center + offset * (clipped / radii)[:, None]


def _particle_prediction(
    particles: np.ndarray,
    history: v19.OnlineDopplerHistory,
    index: int,
) -> np.ndarray:
    follower = particles + history.dead_reckoned_displacement_m[index][None, :]
    relative_position = history.leader_position_m[index][None, :, :] - follower[:, None, :]
    ranges = np.maximum(np.linalg.norm(relative_position, axis=2), 1e-12)
    line_of_sight = relative_position / ranges[:, :, None]
    relative_velocity = (
        history.leader_velocity_mps[index][None, :, :]
        - history.follower_velocity_measured_mps[index][None, None, :]
    )
    return -np.sum(line_of_sight * relative_velocity, axis=2)


def _evaluate_particle_filter(
    arm: str,
    particle_count: int,
    history: v19.OnlineDopplerHistory,
    center_m: np.ndarray,
    radius_min_m: float,
    radius_max_m: float,
    config: EvaluatorConfig,
) -> EstimatorOutput:
    started = time.perf_counter()
    count = int(particle_count)
    particles = v19.deterministic_shell_candidates(
        center_m,
        radius_min_m,
        radius_max_m,
        count,
        int(config.pf_design_seed) + count,
        radial_distribution="uniform_radius",
    )
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    rng = np.random.default_rng(int(config.pf_design_seed) + 17 * count)
    sigma = float(config.measurement_sigma_mps)
    h = float(config.pf_kernel_h)
    shrinkage = math.sqrt(1.0 - h * h)
    threshold = float(config.pf_ess_fraction) * count
    resample_count = 0
    minimum_ess = float(count)
    for index in range(history.measurement_count):
        prediction = _particle_prediction(particles, history, index)
        residual = history.doppler_measured_mps[index][None, :] - prediction
        log_likelihood = -0.5 * np.sum((residual / sigma) ** 2, axis=1)
        log_weights = np.log(np.maximum(weights, np.finfo(np.float64).tiny)) + log_likelihood
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        total = float(np.sum(weights))
        if not math.isfinite(total) or total <= 0.0:
            raise FloatingPointError("particle weights collapsed to a non-finite total")
        weights /= total
        ess = float(1.0 / np.sum(weights * weights))
        minimum_ess = min(minimum_ess, ess)
        if ess < threshold and index + 1 < history.measurement_count:
            mean = np.sum(weights[:, None] * particles, axis=0)
            centered = particles - mean[None, :]
            covariance = (centered.T * weights) @ centered
            covariance = 0.5 * (covariance + covariance.T) + 1e-9 * np.eye(3)
            ancestors = _systematic_resample(weights, rng)
            try:
                noise = rng.multivariate_normal(
                    np.zeros(3, dtype=np.float64),
                    h * h * covariance,
                    size=count,
                    check_valid="raise",
                )
            except (ValueError, np.linalg.LinAlgError):
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12)))
                noise = rng.normal(size=(count, 3)) @ (h * root).T
            particles = (
                shrinkage * particles[ancestors]
                + (1.0 - shrinkage) * mean[None, :]
                + noise
            )
            particles = _project_particles_to_shell(
                particles, center_m, radius_min_m, radius_max_m
            )
            weights.fill(1.0 / count)
            resample_count += 1
    mean = np.sum(weights[:, None] * particles, axis=0)
    centered = particles - mean[None, :]
    covariance = (centered.T * weights) @ centered
    covariance = 0.5 * (covariance + covariance.T)
    radius = _radius_from_covariance(covariance)
    runtime = float(time.perf_counter() - started)
    final_ess = float(1.0 / np.sum(weights * weights))
    return EstimatorOutput(
        arm=arm,
        current_position_m=mean + history.dead_reckoned_displacement_m[-1],
        initial_position_m=mean,
        covariance_m2=covariance,
        nominal_radius95_m=radius,
        residual_rmse_mps=_residual_rmse(mean, history),
        runtime_s=runtime,
        diagnostics={
            "particle_count": count,
            "resample_count": int(resample_count),
            "minimum_ess": float(minimum_ess),
            "final_ess": final_ess,
            "ess_fraction": float(config.pf_ess_fraction),
            "liu_west_h": h,
        },
    )


_LEGACY_TRACE_FIELDS = (
    "t_s",
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


def load_legacy_pf_output(
    trace_path: Path,
    expected_sha256: str,
    endpoint_s: float,
) -> EstimatorOutput:
    path = Path(trace_path)
    if sha256_file(path) != str(expected_sha256):
        raise RuntimeError("legacy PF trace hash does not match frozen V19 metadata")
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in _LEGACY_TRACE_FIELDS if name not in archive.files]
        if missing:
            raise ValueError(f"legacy PF trace lacks fields: {missing}")
        times = np.asarray(archive["t_s"], dtype=np.float64)
        matches = np.flatnonzero(np.isclose(times, float(endpoint_s), rtol=0.0, atol=1e-9))
        if matches.size != 1:
            raise ValueError("legacy PF trace has no unique exact checkpoint")
        index = int(matches[0])
        current = np.array(
            [archive["pF_hat_x"][index], archive["pF_hat_y"][index], archive["pF_hat_z"][index]],
            dtype=np.float64,
        )
        covariance = np.array(
            [
                [archive["pf_cov_xx"][index], archive["pf_cov_xy"][index], archive["pf_cov_xz"][index]],
                [archive["pf_cov_xy"][index], archive["pf_cov_yy"][index], archive["pf_cov_yz"][index]],
                [archive["pf_cov_xz"][index], archive["pf_cov_yz"][index], archive["pf_cov_zz"][index]],
            ],
            dtype=np.float64,
        )
    # Trace loading time is not a historical estimator runtime and is not reported.
    _ = time.perf_counter() - started
    return EstimatorOutput(
        arm=LEGACY_PF,
        current_position_m=current,
        initial_position_m=None,
        covariance_m2=covariance,
        nominal_radius95_m=_radius_from_covariance(covariance),
        residual_rmse_mps=None,
        runtime_s=None,
        diagnostics={
            "source": "frozen_v18_1_trace",
            "trace_sha256": str(expected_sha256),
            "trace_checkpoint_s": float(endpoint_s),
        },
    )


def evaluate_arm(
    arm: str,
    history: v19.OnlineDopplerHistory,
    radius_min_m: float,
    radius_max_m: float,
    config: EvaluatorConfig,
    *,
    legacy_trace_path: Optional[Path] = None,
    legacy_trace_sha256: Optional[str] = None,
) -> EstimatorOutput:
    """Evaluate one arm without accepting a truth object or episode seed."""

    if arm not in ARM_NAMES:
        raise ValueError(f"unknown V27 arm {arm!r}")
    center = v19.initial_leader_centroid_from_history(history)
    if arm == LEGACY_PF:
        if legacy_trace_path is None or legacy_trace_sha256 is None:
            raise ValueError("legacy PF requires a frozen trace path and hash")
        return load_legacy_pf_output(
            legacy_trace_path, legacy_trace_sha256, float(history.t_s[-1])
        )
    if arm == PF_LW_4096:
        return _evaluate_particle_filter(
            arm,
            int(config.pf_particles_small),
            history,
            center,
            radius_min_m,
            radius_max_m,
            config,
        )
    if arm == PF_LW_16384:
        return _evaluate_particle_filter(
            arm,
            int(config.pf_particles_large),
            history,
            center,
            radius_min_m,
            radius_max_m,
            config,
        )
    if arm == EKF_STATIC:
        return _evaluate_ekf(
            history, center, radius_min_m, radius_max_m, config
        )
    if arm == LOCAL_NLS6:
        return _evaluate_local_nls6(
            history, center, radius_min_m, radius_max_m, config
        )
    if arm == COARSE_ONLY_8192:
        return _evaluate_coarse_only(
            history, center, radius_min_m, radius_max_m, config
        )
    if arm in {GLOBAL_WINDOW60, GLOBAL_FULL}:
        return _evaluate_global(
            arm, history, center, radius_min_m, radius_max_m, config
        )
    raise AssertionError("unreachable V27 arm dispatch")


def score_output(
    output: EstimatorOutput,
    truth: v19.ReplayTruthDiagnostics,
    history: v19.OnlineDopplerHistory,
) -> Dict[str, Any]:
    """Score a persisted estimator output after the truth boundary is opened."""

    endpoint_s = float(history.t_s[-1])
    endpoint_index = int(round(endpoint_s)) - 1
    if endpoint_index < 0 or endpoint_index >= truth.true_displacement_m.shape[0]:
        raise ValueError("checkpoint does not map to the one-second truth grid")
    if not math.isclose(endpoint_s, float(endpoint_index + 1), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("checkpoint is not on the one-second truth grid")
    truth_current = (
        truth.initial_follower_position_m + truth.true_displacement_m[endpoint_index]
    )
    endpoint_error = float(np.linalg.norm(output.current_position_m - truth_current))
    initial_error = (
        None
        if output.initial_position_m is None
        else float(
            np.linalg.norm(
                np.asarray(output.initial_position_m)
                - truth.initial_follower_position_m
            )
        )
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
    }


def config_to_dict(config: EvaluatorConfig) -> Dict[str, Any]:
    return asdict(config)


def deterministic_payload(output: EstimatorOutput) -> Dict[str, Any]:
    """Return the output payload with wall-clock runtime removed."""

    payload = output.to_unscored_dict()
    payload.pop("runtime_s", None)
    return payload


__all__ = [
    "VERSION",
    "ARM_NAMES",
    "LEGACY_PF",
    "PF_LW_4096",
    "PF_LW_16384",
    "EKF_STATIC",
    "LOCAL_NLS6",
    "COARSE_ONLY_8192",
    "GLOBAL_WINDOW60",
    "GLOBAL_FULL",
    "EvaluatorConfig",
    "EstimatorOutput",
    "assert_v27_development_seed",
    "evaluate_arm",
    "score_output",
    "output_from_unscored_dict",
    "config_to_dict",
    "deterministic_payload",
    "last_window",
    "load_legacy_pf_output",
    "sha256_file",
]
