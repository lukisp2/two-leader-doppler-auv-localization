#!/usr/bin/env python3
"""Causal held-out evidence gate for nominal versus profiled-bias models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27
import uuv_v30_profiled_bias as v30


VERSION = "v31_bias_evidence_gate_1.0"
NOMINAL_ONLY = "nominal_only"
BIAS_ALWAYS = "bias_always"
BIAS_EVIDENCE_GATE = "bias_evidence_gate"
REPORTING_ARMS = (NOMINAL_ONLY, BIAS_ALWAYS, BIAS_EVIDENCE_GATE)

TRAIN_FRACTION = 0.75
TRAIN_AMPLITUDE_MIN_MPS = 0.015
VALIDATION_AMPLITUDE_MIN_MPS = 0.010
AMPLITUDE_DRIFT_MAX_MPS = 0.015
RELATIVE_SSE_GAIN_MIN = 0.10
LAG1_ABS_MAX = 0.50


@dataclass(frozen=True)
class BiasGateEvidence:
    checkpoint_s: float
    training_end_s: float
    validation_count: int
    training_link_bias_mps: np.ndarray
    validation_link_mean_mps: np.ndarray
    selected_component: str
    training_component_mps: float
    validation_component_mps: float
    nominal_validation_sse_mps2: float
    bias_validation_sse_mps2: float
    relative_sse_gain: float
    maximum_absolute_lag1: float
    material_training_bias: bool
    material_validation_bias: bool
    sign_consistent: bool
    amplitude_stable: bool
    predictive_gain_valid: bool
    residual_whiteness_valid: bool
    activate_bias_model: bool
    runtime_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "version": VERSION,
            "checkpoint_s": float(self.checkpoint_s),
            "training_end_s": float(self.training_end_s),
            "validation_count": int(self.validation_count),
            "training_link_bias_mps": self.training_link_bias_mps.tolist(),
            "validation_link_mean_mps": self.validation_link_mean_mps.tolist(),
            "selected_component": self.selected_component,
            "training_component_mps": float(self.training_component_mps),
            "validation_component_mps": float(self.validation_component_mps),
            "nominal_validation_sse_mps2": float(self.nominal_validation_sse_mps2),
            "bias_validation_sse_mps2": float(self.bias_validation_sse_mps2),
            "relative_sse_gain": float(self.relative_sse_gain),
            "maximum_absolute_lag1": float(self.maximum_absolute_lag1),
            "material_training_bias": bool(self.material_training_bias),
            "material_validation_bias": bool(self.material_validation_bias),
            "sign_consistent": bool(self.sign_consistent),
            "amplitude_stable": bool(self.amplitude_stable),
            "predictive_gain_valid": bool(self.predictive_gain_valid),
            "residual_whiteness_valid": bool(self.residual_whiteness_valid),
            "activate_bias_model": bool(self.activate_bias_model),
            "runtime_s": float(self.runtime_s),
        }


def _lag1(value: np.ndarray) -> float:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size < 3:
        return 0.0
    left = vector[:-1] - float(np.mean(vector[:-1]))
    right = vector[1:] - float(np.mean(vector[1:]))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _common_differential(link_values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(link_values, dtype=np.float64).reshape(2)
    return float(0.5 * (values[0] + values[1])), float(
        0.5 * (values[0] - values[1])
    )


def evaluate_evidence(
    history: v19.OnlineDopplerHistory,
    checkpoint_s: float,
    radius_min_m: float,
    radius_max_m: float,
    config: v27.EvaluatorConfig,
) -> BiasGateEvidence:
    """Fit on the first 75% and test model evidence on the held-out tail."""
    started = time.perf_counter()
    checkpoint = float(checkpoint_s)
    training_end = TRAIN_FRACTION * checkpoint
    expected_training_end = 90.0 if checkpoint == 120.0 else 330.0 if checkpoint == 440.0 else None
    if expected_training_end is None or not math.isclose(
        training_end, expected_training_end, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("V31 accepts only frozen checkpoints 120 and 440 s")
    causal = history.prefix(checkpoint)
    training = causal.prefix(training_end)
    validation_mask = (causal.t_s > training_end + 1e-12) & (
        causal.t_s <= checkpoint + 1e-12
    )
    if int(np.sum(validation_mask)) < 3:
        raise ValueError("V31 validation tail is too short")
    validation = causal.take(validation_mask)

    nominal_training = v30.evaluate_arm(
        v30.BASELINE_LOCAL6,
        training,
        radius_min_m,
        radius_max_m,
        config,
    )
    bias_training = v30.evaluate_arm(
        v30.PROFILED_LOCAL6,
        training,
        radius_min_m,
        radius_max_m,
        config,
    )
    if bias_training.link_bias_mps is None:
        raise RuntimeError("profiled training estimate lacks link biases")
    training_bias = np.asarray(bias_training.link_bias_mps, dtype=np.float64)
    nominal_residual = validation.doppler_measured_mps - v19.predict_doppler(
        nominal_training.initial_position_m, validation
    )
    raw_bias_validation_residual = (
        validation.doppler_measured_mps
        - v19.predict_doppler(bias_training.initial_position_m, validation)
    )
    validation_link_mean = np.mean(raw_bias_validation_residual, axis=0)
    bias_residual = raw_bias_validation_residual - training_bias[None, :]
    nominal_sse = float(np.sum(nominal_residual * nominal_residual))
    bias_sse = float(np.sum(bias_residual * bias_residual))
    relative_gain = float((nominal_sse - bias_sse) / max(nominal_sse, 1e-18))

    training_common, training_differential = _common_differential(training_bias)
    validation_common, validation_differential = _common_differential(
        validation_link_mean
    )
    if abs(training_common) >= abs(training_differential):
        selected_component = "common"
        training_component = training_common
        validation_component = validation_common
    else:
        selected_component = "differential"
        training_component = training_differential
        validation_component = validation_differential

    maximum_lag1 = float(
        max(abs(_lag1(bias_residual[:, 0])), abs(_lag1(bias_residual[:, 1])))
    )
    material_training = abs(training_component) >= TRAIN_AMPLITUDE_MIN_MPS
    material_validation = abs(validation_component) >= VALIDATION_AMPLITUDE_MIN_MPS
    sign_consistent = bool(training_component * validation_component > 0.0)
    amplitude_stable = (
        abs(training_component - validation_component) <= AMPLITUDE_DRIFT_MAX_MPS
    )
    predictive_gain = relative_gain >= RELATIVE_SSE_GAIN_MIN
    whiteness = maximum_lag1 <= LAG1_ABS_MAX
    activate = bool(
        material_training
        and material_validation
        and sign_consistent
        and amplitude_stable
        and predictive_gain
        and whiteness
    )
    return BiasGateEvidence(
        checkpoint_s=checkpoint,
        training_end_s=training_end,
        validation_count=validation.measurement_count,
        training_link_bias_mps=training_bias.copy(),
        validation_link_mean_mps=np.asarray(validation_link_mean).copy(),
        selected_component=selected_component,
        training_component_mps=training_component,
        validation_component_mps=validation_component,
        nominal_validation_sse_mps2=nominal_sse,
        bias_validation_sse_mps2=bias_sse,
        relative_sse_gain=relative_gain,
        maximum_absolute_lag1=maximum_lag1,
        material_training_bias=material_training,
        material_validation_bias=material_validation,
        sign_consistent=sign_consistent,
        amplitude_stable=amplitude_stable,
        predictive_gain_valid=predictive_gain,
        residual_whiteness_valid=whiteness,
        activate_bias_model=activate,
        runtime_s=float(time.perf_counter() - started),
    )


def deterministic_evidence(evidence: BiasGateEvidence) -> Dict[str, Any]:
    value = evidence.to_dict()
    value.pop("runtime_s", None)
    return value


__all__ = [
    "VERSION",
    "REPORTING_ARMS",
    "NOMINAL_ONLY",
    "BIAS_ALWAYS",
    "BIAS_EVIDENCE_GATE",
    "BiasGateEvidence",
    "evaluate_evidence",
    "deterministic_evidence",
]
