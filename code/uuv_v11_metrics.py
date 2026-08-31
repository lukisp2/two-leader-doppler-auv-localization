"""Auditable evaluation metrics for the v11 online UUV experiments.

This module deliberately has no dependency on the environment, Stable-Baselines3,
or SciPy.  It keeps three concepts separate:

* ``raw_pf_largest_eigen_std`` is computed only from the PF posterior covariance;
* ``hybrid_sigma_diagnostic`` is a conservative PF/CRLB/ESS diagnostic and must
  never be labelled as a PF posterior standard deviation;
* ground-truth-dependent consistency metrics are evaluation-only diagnostics.

All differences returned by the paired helpers use the explicit convention
``candidate - baseline``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


# 0.95 quantiles of chi-square distributions.  v11 is three-dimensional, but
# keeping the adjacent dimensions here makes the helper difficult to misuse if
# a reduced diagnostic is needed.
CHI2_95_BY_DOF: Dict[int, float] = {
    1: 3.841458820694124,
    2: 5.991464547107979,
    3: 7.814727903251179,
    4: 9.487729036781154,
    5: 11.070497693516351,
    6: 12.591587243743977,
    7: 14.067140449340169,
    8: 15.50731305586545,
    9: 16.918977604620448,
    10: 18.307038053275146,
}


def _finite_vector(value: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("%s must not be empty" % name)
    if not np.all(np.isfinite(vector)):
        raise ValueError("%s must contain only finite values" % name)
    return vector


def _psd_eigendecomposition(
    covariance: ArrayLike,
    *,
    dimension: Optional[int] = None,
    psd_tolerance: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return clipped PSD eigenvalues/eigenvectors of a covariance matrix.

    The matrix is symmetrised.  Tiny negative eigenvalues caused by round-off
    are clipped to zero, while a materially indefinite matrix raises an error.
    """

    covariance_array = np.asarray(covariance, dtype=np.float64)
    if covariance_array.ndim != 2 or covariance_array.shape[0] != covariance_array.shape[1]:
        raise ValueError("covariance must be a finite square matrix")
    if dimension is not None and covariance_array.shape != (dimension, dimension):
        raise ValueError(
            "covariance shape %s does not match state dimension %d"
            % (covariance_array.shape, dimension)
        )
    if not np.all(np.isfinite(covariance_array)):
        raise ValueError("covariance must contain only finite values")
    if psd_tolerance < 0.0 or not np.isfinite(psd_tolerance):
        raise ValueError("psd_tolerance must be finite and non-negative")

    symmetric = 0.5 * (covariance_array + covariance_array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues[0]) < -float(psd_tolerance) * scale:
        raise ValueError(
            "covariance is not positive semidefinite; minimum eigenvalue=%g"
            % float(eigenvalues[0])
        )
    return np.maximum(eigenvalues, 0.0), eigenvectors


def raw_pf_largest_eigen_std(
    pf_covariance: ArrayLike, *, psd_tolerance: float = 1e-9
) -> float:
    """Return ``sqrt(lambda_max(P_pf))`` without CRLB or ESS inflation."""

    eigenvalues, _ = _psd_eigendecomposition(
        pf_covariance, psd_tolerance=psd_tolerance
    )
    return float(np.sqrt(eigenvalues[-1]))


@dataclass(frozen=True)
class HybridSigmaDiagnostic:
    """Components of the conservative hybrid ``sigma_eff`` diagnostic.

    ``sigma_eff_hybrid`` is not a posterior PF standard deviation.  The field
    names intentionally preserve that distinction in CSV/JSON output.
    """

    raw_pf_largest_std: float
    crlb_floor_std: float
    pre_ess_hybrid_std: float
    ess_fraction: Optional[float]
    ess_inflation_factor: float
    uncapped_sigma_eff_hybrid: float
    sigma_eff_hybrid: float
    cap_applied: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hybrid_sigma_diagnostic(
    raw_pf_std: float,
    *,
    crlb_largest_std: Optional[float] = None,
    crlb_multiplier: float = 1.0,
    ess: Optional[float] = None,
    particle_count: Optional[int] = None,
    ess_threshold_fraction: float = 0.5,
    ess_inflation_gain: float = 0.0,
    cap: Optional[float] = None,
) -> HybridSigmaDiagnostic:
    """Build the explicitly labelled PF/CRLB/ESS hybrid diagnostic.

    The formula mirrors the conservative construction used by the earlier
    environment while exposing every component::

        pre_ess = max(raw_pf_std, crlb_multiplier * crlb_largest_std)
        sigma_eff = pre_ess * (1 + gain * ESS_deficit_fraction)

    The optional cap is applied last.  ``ess`` and ``particle_count`` must be
    supplied together.  Omitting both disables ESS inflation.
    """

    raw_pf_std = float(raw_pf_std)
    crlb_multiplier = float(crlb_multiplier)
    ess_threshold_fraction = float(ess_threshold_fraction)
    ess_inflation_gain = float(ess_inflation_gain)
    if not np.isfinite(raw_pf_std) or raw_pf_std < 0.0:
        raise ValueError("raw_pf_std must be finite and non-negative")
    if not np.isfinite(crlb_multiplier) or crlb_multiplier < 0.0:
        raise ValueError("crlb_multiplier must be finite and non-negative")
    if not 0.0 < ess_threshold_fraction <= 1.0:
        raise ValueError("ess_threshold_fraction must be in (0, 1]")
    if not np.isfinite(ess_inflation_gain) or ess_inflation_gain < 0.0:
        raise ValueError("ess_inflation_gain must be finite and non-negative")

    if crlb_largest_std is None:
        crlb_floor_std = 0.0
    else:
        crlb_largest_std = float(crlb_largest_std)
        if not np.isfinite(crlb_largest_std) or crlb_largest_std < 0.0:
            raise ValueError("crlb_largest_std must be finite and non-negative")
        crlb_floor_std = crlb_multiplier * crlb_largest_std

    pre_ess = max(raw_pf_std, crlb_floor_std)
    if (ess is None) != (particle_count is None):
        raise ValueError("ess and particle_count must be supplied together")

    ess_fraction: Optional[float] = None
    ess_factor = 1.0
    if ess is not None and particle_count is not None:
        ess = float(ess)
        particle_count = int(particle_count)
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if not np.isfinite(ess) or ess < 0.0 or ess > float(particle_count) * (1.0 + 1e-12):
            raise ValueError("ess must be finite and lie in [0, particle_count]")
        ess_fraction = float(np.clip(ess / float(particle_count), 0.0, 1.0))
        if ess_fraction < ess_threshold_fraction:
            deficit = (ess_threshold_fraction - ess_fraction) / ess_threshold_fraction
            ess_factor += ess_inflation_gain * float(np.clip(deficit, 0.0, 1.0))

    uncapped = pre_ess * ess_factor
    if cap is None:
        sigma_eff = uncapped
    else:
        cap = float(cap)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError("cap must be finite and positive")
        sigma_eff = min(uncapped, cap)

    return HybridSigmaDiagnostic(
        raw_pf_largest_std=raw_pf_std,
        crlb_floor_std=float(crlb_floor_std),
        pre_ess_hybrid_std=float(pre_ess),
        ess_fraction=ess_fraction,
        ess_inflation_factor=float(ess_factor),
        uncapped_sigma_eff_hybrid=float(uncapped),
        sigma_eff_hybrid=float(sigma_eff),
        cap_applied=bool(sigma_eff < uncapped),
    )


def localization_error(estimate: ArrayLike, truth: ArrayLike) -> float:
    """Euclidean position error; ground truth is used only during evaluation."""

    estimate_vector = _finite_vector(estimate, "estimate")
    truth_vector = _finite_vector(truth, "truth")
    if estimate_vector.shape != truth_vector.shape:
        raise ValueError("estimate and truth must have the same dimension")
    return float(np.linalg.norm(estimate_vector - truth_vector))


def regularized_covariance_pseudoinverse(
    covariance: ArrayLike,
    *,
    regularization: float = 1e-9,
    rcond: float = 1e-12,
    psd_tolerance: float = 1e-9,
) -> np.ndarray:
    """Return a stable pseudoinverse of ``sym(P) + regularization * I``.

    Set ``regularization=0`` for a pure Moore-Penrose pseudoinverse.  Singular
    directions are discarded according to ``rcond``.
    """

    regularization = float(regularization)
    rcond = float(rcond)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if not np.isfinite(rcond) or rcond < 0.0:
        raise ValueError("rcond must be finite and non-negative")

    eigenvalues, eigenvectors = _psd_eigendecomposition(
        covariance, psd_tolerance=psd_tolerance
    )
    regularized = eigenvalues + regularization
    largest = float(regularized[-1])
    inverse_values = np.zeros_like(regularized)
    if largest > 0.0:
        keep = regularized > rcond * largest
        inverse_values[keep] = 1.0 / regularized[keep]
    return (eigenvectors * inverse_values) @ eigenvectors.T


def nees(
    estimate: ArrayLike,
    truth: ArrayLike,
    covariance: ArrayLike,
    *,
    regularization: float = 1e-9,
    rcond: float = 1e-12,
    psd_tolerance: float = 1e-9,
) -> float:
    """Compute normalized estimation error squared, ``e.T @ P^+ @ e``."""

    estimate_vector = _finite_vector(estimate, "estimate")
    truth_vector = _finite_vector(truth, "truth")
    if estimate_vector.shape != truth_vector.shape:
        raise ValueError("estimate and truth must have the same dimension")
    inverse = regularized_covariance_pseudoinverse(
        np.asarray(covariance, dtype=np.float64),
        regularization=regularization,
        rcond=rcond,
        psd_tolerance=psd_tolerance,
    )
    if inverse.shape != (estimate_vector.size, estimate_vector.size):
        raise ValueError("covariance dimension does not match estimate and truth")
    error = estimate_vector - truth_vector
    value = float(error @ inverse @ error)
    # A negative value at this point can only be round-off.
    return max(0.0, value)


def ellipsoid_coverage_95(
    estimate: ArrayLike,
    truth: ArrayLike,
    covariance: ArrayLike,
    *,
    regularization: float = 1e-9,
    rcond: float = 1e-12,
    psd_tolerance: float = 1e-9,
    chi2_threshold: Optional[float] = None,
) -> bool:
    """Whether truth lies inside the nominal 95% covariance ellipsoid."""

    estimate_vector = _finite_vector(estimate, "estimate")
    truth_vector = _finite_vector(truth, "truth")
    if estimate_vector.shape != truth_vector.shape:
        raise ValueError("estimate and truth must have the same dimension")
    if chi2_threshold is None:
        try:
            threshold = CHI2_95_BY_DOF[int(estimate_vector.size)]
        except KeyError as exc:
            raise ValueError(
                "no built-in 95%% chi-square threshold for %d dimensions; "
                "supply chi2_threshold" % estimate_vector.size
            ) from exc
    else:
        threshold = float(chi2_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("chi2_threshold must be finite and positive")
    value = nees(
        estimate_vector,
        truth_vector,
        covariance,
        regularization=regularization,
        rcond=rcond,
        psd_tolerance=psd_tolerance,
    )
    return bool(value <= threshold)


@dataclass(frozen=True)
class ConsistencyMetrics:
    localization_error: float
    nees: float
    covered_by_95pct_ellipsoid: bool
    chi2_threshold_95: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def consistency_metrics(
    estimate: ArrayLike,
    truth: ArrayLike,
    covariance: ArrayLike,
    *,
    regularization: float = 1e-9,
    rcond: float = 1e-12,
    psd_tolerance: float = 1e-9,
) -> ConsistencyMetrics:
    """Return localization error, NEES, and nominal 95% coverage together."""

    estimate_vector = _finite_vector(estimate, "estimate")
    truth_vector = _finite_vector(truth, "truth")
    if estimate_vector.shape != truth_vector.shape:
        raise ValueError("estimate and truth must have the same dimension")
    dimension = int(estimate_vector.size)
    if dimension not in CHI2_95_BY_DOF:
        raise ValueError("no built-in 95%% chi-square threshold for %d dimensions" % dimension)
    threshold = CHI2_95_BY_DOF[dimension]
    nees_value = nees(
        estimate_vector,
        truth_vector,
        covariance,
        regularization=regularization,
        rcond=rcond,
        psd_tolerance=psd_tolerance,
    )
    return ConsistencyMetrics(
        localization_error=localization_error(estimate_vector, truth_vector),
        nees=nees_value,
        covered_by_95pct_ellipsoid=bool(nees_value <= threshold),
        chi2_threshold_95=threshold,
    )


def threshold_success_mask(
    position_errors: ArrayLike,
    position_tolerance: float,
    *,
    uncertainty_values: Optional[ArrayLike] = None,
    uncertainty_tolerance: Optional[float] = None,
    inclusive: bool = False,
) -> np.ndarray:
    """Construct per-step success flags from explicitly named thresholds.

    Non-finite samples fail the criterion.  Callers should pass raw PF
    uncertainty here unless a differently labelled scientific criterion is
    intentionally being studied.
    """

    errors = np.asarray(position_errors, dtype=np.float64).reshape(-1)
    position_tolerance = float(position_tolerance)
    if errors.size == 0:
        raise ValueError("position_errors must not be empty")
    if not np.isfinite(position_tolerance) or position_tolerance <= 0.0:
        raise ValueError("position_tolerance must be finite and positive")
    if np.any(errors[np.isfinite(errors)] < 0.0):
        raise ValueError("position_errors must be non-negative")

    if inclusive:
        mask = np.isfinite(errors) & (errors <= position_tolerance)
    else:
        mask = np.isfinite(errors) & (errors < position_tolerance)

    if (uncertainty_values is None) != (uncertainty_tolerance is None):
        raise ValueError(
            "uncertainty_values and uncertainty_tolerance must be supplied together"
        )
    if uncertainty_values is not None and uncertainty_tolerance is not None:
        uncertainty = np.asarray(uncertainty_values, dtype=np.float64).reshape(-1)
        uncertainty_tolerance = float(uncertainty_tolerance)
        if uncertainty.shape != errors.shape:
            raise ValueError("uncertainty_values must match position_errors")
        if not np.isfinite(uncertainty_tolerance) or uncertainty_tolerance <= 0.0:
            raise ValueError("uncertainty_tolerance must be finite and positive")
        if np.any(uncertainty[np.isfinite(uncertainty)] < 0.0):
            raise ValueError("uncertainty_values must be non-negative")
        if inclusive:
            mask &= np.isfinite(uncertainty) & (uncertainty <= uncertainty_tolerance)
        else:
            mask &= np.isfinite(uncertainty) & (uncertainty < uncertainty_tolerance)
    return mask


@dataclass(frozen=True)
class EpisodeSuccessMetrics:
    ever_success: bool
    terminal_success: bool
    dwell_success: bool
    total_success_time: float
    longest_contiguous_success_time: float
    dwell_fraction: float
    required_dwell_time: float
    first_success_index: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def episode_success_metrics(
    success_flags: Sequence[bool],
    *,
    sample_durations: Union[float, ArrayLike],
    required_dwell_time: float,
) -> EpisodeSuccessMetrics:
    """Separate ever, terminal, and sustained-dwell episode success.

    Each flag is assumed to describe the interval represented by the matching
    positive ``sample_durations`` value.  ``dwell_success`` requires one
    contiguous successful run of at least ``required_dwell_time``.
    """

    flags = np.asarray(success_flags, dtype=bool).reshape(-1)
    if flags.size == 0:
        raise ValueError("success_flags must not be empty")
    required_dwell_time = float(required_dwell_time)
    if not np.isfinite(required_dwell_time) or required_dwell_time <= 0.0:
        raise ValueError("required_dwell_time must be finite and positive")

    durations_input = np.asarray(sample_durations, dtype=np.float64)
    if durations_input.ndim == 0:
        durations = np.full(flags.size, float(durations_input), dtype=np.float64)
    else:
        durations = durations_input.reshape(-1)
        if durations.shape != flags.shape:
            raise ValueError("sample_durations must be scalar or match success_flags")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("sample_durations must contain finite positive values")

    total_duration = float(np.sum(durations))
    total_success = float(np.sum(durations[flags]))
    longest = 0.0
    current = 0.0
    for flag, duration in zip(flags, durations):
        if bool(flag):
            current += float(duration)
            longest = max(longest, current)
        else:
            current = 0.0
    successful_indices = np.flatnonzero(flags)
    first_index = int(successful_indices[0]) if successful_indices.size else None
    return EpisodeSuccessMetrics(
        ever_success=bool(np.any(flags)),
        terminal_success=bool(flags[-1]),
        dwell_success=bool(longest + 1e-12 >= required_dwell_time),
        total_success_time=total_success,
        longest_contiguous_success_time=float(longest),
        dwell_fraction=float(total_success / total_duration),
        required_dwell_time=required_dwell_time,
        first_success_index=first_index,
    )


@dataclass(frozen=True)
class PairedValues:
    pair_ids: Tuple[Hashable, ...]
    candidate: np.ndarray
    baseline: np.ndarray

    def __post_init__(self) -> None:
        if self.candidate.shape != self.baseline.shape:
            raise ValueError("candidate and baseline must have matching shapes")
        if self.candidate.ndim != 1 or len(self.pair_ids) != self.candidate.size:
            raise ValueError("paired values must be one-dimensional and match pair_ids")


def align_paired_values(
    candidate_by_id: Mapping[Hashable, float],
    baseline_by_id: Mapping[Hashable, float],
    *,
    drop_nonfinite: bool = True,
) -> PairedValues:
    """Align two metric mappings on common scenario/episode identifiers."""

    common_ids = sorted(
        set(candidate_by_id).intersection(baseline_by_id), key=lambda value: repr(value)
    )
    kept_ids = []
    candidate_values = []
    baseline_values = []
    for pair_id in common_ids:
        candidate_value = float(candidate_by_id[pair_id])
        baseline_value = float(baseline_by_id[pair_id])
        finite = np.isfinite(candidate_value) and np.isfinite(baseline_value)
        if not finite and drop_nonfinite:
            continue
        kept_ids.append(pair_id)
        candidate_values.append(candidate_value)
        baseline_values.append(baseline_value)
    return PairedValues(
        pair_ids=tuple(kept_ids),
        candidate=np.asarray(candidate_values, dtype=np.float64),
        baseline=np.asarray(baseline_values, dtype=np.float64),
    )


@dataclass(frozen=True)
class PairedSummary:
    """Paired descriptive summary; differences are candidate minus baseline."""

    n_pairs: int
    candidate_mean: float
    baseline_mean: float
    mean_difference: float
    median_difference: float
    std_difference: float
    standard_error_difference: float
    confidence_level: float
    bootstrap_mean_difference_ci_low: float
    bootstrap_mean_difference_ci_high: float
    candidate_wins: int
    ties: int
    baseline_wins: int
    higher_is_better: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def paired_summary(
    candidate: ArrayLike,
    baseline: ArrayLike,
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10_000,
    random_seed: int = 0,
    higher_is_better: bool = False,
    tie_tolerance: float = 0.0,
) -> PairedSummary:
    """Summarize paired outcomes with a percentile bootstrap CI.

    ``higher_is_better`` affects only win/tie/loss counts.  The reported
    numerical difference always remains ``candidate - baseline`` so that its
    sign cannot silently change between metrics.
    """

    candidate_values = np.asarray(candidate, dtype=np.float64).reshape(-1)
    baseline_values = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if candidate_values.shape != baseline_values.shape:
        raise ValueError("candidate and baseline must have matching shapes")
    finite = np.isfinite(candidate_values) & np.isfinite(baseline_values)
    candidate_values = candidate_values[finite]
    baseline_values = baseline_values[finite]
    if candidate_values.size == 0:
        raise ValueError("at least one finite pair is required")
    confidence_level = float(confidence_level)
    tie_tolerance = float(tie_tolerance)
    bootstrap_samples = int(bootstrap_samples)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")

    differences = candidate_values - baseline_values
    n_pairs = int(differences.size)
    if n_pairs > 1:
        std_difference = float(np.std(differences, ddof=1))
        standard_error = std_difference / float(np.sqrt(n_pairs))
    else:
        std_difference = 0.0
        standard_error = 0.0

    if bootstrap_samples == 0:
        ci_low = float("nan")
        ci_high = float("nan")
    elif n_pairs == 1:
        ci_low = ci_high = float(differences[0])
    else:
        generator = np.random.default_rng(int(random_seed))
        bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
        chunk_size = 2048
        for start in range(0, bootstrap_samples, chunk_size):
            stop = min(start + chunk_size, bootstrap_samples)
            indices = generator.integers(0, n_pairs, size=(stop - start, n_pairs))
            bootstrap_means[start:stop] = np.mean(differences[indices], axis=1)
        alpha = 0.5 * (1.0 - confidence_level)
        ci_low, ci_high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
        ci_low = float(ci_low)
        ci_high = float(ci_high)

    signed_for_wins = differences if higher_is_better else -differences
    candidate_wins = int(np.sum(signed_for_wins > tie_tolerance))
    baseline_wins = int(np.sum(signed_for_wins < -tie_tolerance))
    ties = n_pairs - candidate_wins - baseline_wins
    return PairedSummary(
        n_pairs=n_pairs,
        candidate_mean=float(np.mean(candidate_values)),
        baseline_mean=float(np.mean(baseline_values)),
        mean_difference=float(np.mean(differences)),
        median_difference=float(np.median(differences)),
        std_difference=std_difference,
        standard_error_difference=standard_error,
        confidence_level=confidence_level,
        bootstrap_mean_difference_ci_low=ci_low,
        bootstrap_mean_difference_ci_high=ci_high,
        candidate_wins=candidate_wins,
        ties=ties,
        baseline_wins=baseline_wins,
        higher_is_better=bool(higher_is_better),
    )


__all__ = [
    "CHI2_95_BY_DOF",
    "ConsistencyMetrics",
    "EpisodeSuccessMetrics",
    "HybridSigmaDiagnostic",
    "PairedSummary",
    "PairedValues",
    "align_paired_values",
    "consistency_metrics",
    "ellipsoid_coverage_95",
    "episode_success_metrics",
    "hybrid_sigma_diagnostic",
    "localization_error",
    "nees",
    "paired_summary",
    "raw_pf_largest_eigen_std",
    "regularized_covariance_pseudoinverse",
    "threshold_success_mask",
]
