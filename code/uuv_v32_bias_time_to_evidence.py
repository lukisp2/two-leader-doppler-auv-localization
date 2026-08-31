#!/usr/bin/env python3
"""General-checkpoint implementation of the frozen V31 bias evidence gate."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27
import uuv_v28_estimator_stress as v28
import uuv_v30_profiled_bias as v30
import uuv_v31_bias_evidence_gate as v31


VERSION = "v32_bias_time_to_evidence_1.0"
CHECKPOINTS_S = (120.0, 140.0, 160.0, 180.0, 200.0, 240.0, 300.0, 360.0, 440.0)
CONDITIONS = (
    v28.NOMINAL,
    v28.DOPPLER_COMMON_BIAS,
    v28.DOPPLER_DIFFERENTIAL_BIAS,
    v28.COLORED_NOISE,
    v28.DROPOUT,
)


@dataclass(frozen=True)
class TimeSweepEvaluation:
    evidence: v31.BiasGateEvidence
    nominal_full: v30.BiasEstimatorOutput
    bias_full: v30.BiasEstimatorOutput
    selected: v30.BiasEstimatorOutput
    total_runtime_s: float

    def to_unscored_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "version": VERSION,
            "evidence": self.evidence.to_dict(),
            "nominal_full": self.nominal_full.to_unscored_dict(),
            "bias_full": self.bias_full.to_unscored_dict(),
            "selected_source_arm": self.selected.arm,
            "selected": self.selected.to_unscored_dict(),
            "total_runtime_s": float(self.total_runtime_s),
        }


def evaluate_checkpoint(
    history: v19.OnlineDopplerHistory,
    checkpoint_s: float,
    radius_min_m: float,
    radius_max_m: float,
    config: v27.EvaluatorConfig,
) -> TimeSweepEvaluation:
    started = time.perf_counter()
    checkpoint = float(checkpoint_s)
    if checkpoint not in CHECKPOINTS_S:
        raise ValueError(f"unsupported V32 checkpoint {checkpoint:g}")
    causal = history.prefix(checkpoint)
    training_end = v31.TRAIN_FRACTION * checkpoint
    training = causal.prefix(training_end)
    validation_mask = causal.t_s > training_end + 1e-12
    if int(np.sum(validation_mask)) < 3:
        raise ValueError("V32 validation tail is too short")
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
        raise RuntimeError("profiled V32 training estimate lacks biases")
    training_bias = np.asarray(bias_training.link_bias_mps, dtype=np.float64)
    nominal_residual = validation.doppler_measured_mps - v19.predict_doppler(
        nominal_training.initial_position_m, validation
    )
    raw_bias_residual = validation.doppler_measured_mps - v19.predict_doppler(
        bias_training.initial_position_m, validation
    )
    validation_mean = np.mean(raw_bias_residual, axis=0)
    bias_residual = raw_bias_residual - training_bias[None, :]
    nominal_sse = float(np.sum(nominal_residual * nominal_residual))
    bias_sse = float(np.sum(bias_residual * bias_residual))
    relative_gain = float((nominal_sse - bias_sse) / max(nominal_sse, 1e-18))
    train_common, train_differential = v31._common_differential(training_bias)
    valid_common, valid_differential = v31._common_differential(validation_mean)
    if abs(train_common) >= abs(train_differential):
        component = "common"
        train_value = train_common
        valid_value = valid_common
    else:
        component = "differential"
        train_value = train_differential
        valid_value = valid_differential
    lag1 = float(
        max(
            abs(v31._lag1(bias_residual[:, 0])),
            abs(v31._lag1(bias_residual[:, 1])),
        )
    )
    material_train = abs(train_value) >= v31.TRAIN_AMPLITUDE_MIN_MPS
    material_valid = abs(valid_value) >= v31.VALIDATION_AMPLITUDE_MIN_MPS
    sign = bool(train_value * valid_value > 0.0)
    stable = abs(train_value - valid_value) <= v31.AMPLITUDE_DRIFT_MAX_MPS
    predictive = relative_gain >= v31.RELATIVE_SSE_GAIN_MIN
    white = lag1 <= v31.LAG1_ABS_MAX
    activate = bool(
        material_train and material_valid and sign and stable and predictive and white
    )
    evidence_runtime = float(time.perf_counter() - started)
    evidence = v31.BiasGateEvidence(
        checkpoint_s=checkpoint,
        training_end_s=training_end,
        validation_count=validation.measurement_count,
        training_link_bias_mps=training_bias.copy(),
        validation_link_mean_mps=np.asarray(validation_mean).copy(),
        selected_component=component,
        training_component_mps=train_value,
        validation_component_mps=valid_value,
        nominal_validation_sse_mps2=nominal_sse,
        bias_validation_sse_mps2=bias_sse,
        relative_sse_gain=relative_gain,
        maximum_absolute_lag1=lag1,
        material_training_bias=material_train,
        material_validation_bias=material_valid,
        sign_consistent=sign,
        amplitude_stable=stable,
        predictive_gain_valid=predictive,
        residual_whiteness_valid=white,
        activate_bias_model=activate,
        runtime_s=evidence_runtime,
    )
    nominal_full = v30.evaluate_arm(
        v30.BASELINE_LOCAL6,
        causal,
        radius_min_m,
        radius_max_m,
        config,
    )
    bias_full = v30.evaluate_arm(
        v30.PROFILED_LOCAL6,
        causal,
        radius_min_m,
        radius_max_m,
        config,
    )
    selected = bias_full if activate else nominal_full
    operational_runtime = evidence_runtime + float(selected.runtime_s)
    return TimeSweepEvaluation(
        evidence=evidence,
        nominal_full=nominal_full,
        bias_full=bias_full,
        selected=selected,
        total_runtime_s=operational_runtime,
    )


def deterministic_payload(evaluation: TimeSweepEvaluation) -> Dict[str, Any]:
    value = evaluation.to_unscored_dict()
    value.pop("total_runtime_s", None)
    value["evidence"].pop("runtime_s", None)
    for key in ("nominal_full", "bias_full", "selected"):
        value[key].pop("runtime_s", None)
    return value


__all__ = [
    "VERSION",
    "CHECKPOINTS_S",
    "CONDITIONS",
    "TimeSweepEvaluation",
    "evaluate_checkpoint",
    "deterministic_payload",
]
