#!/usr/bin/env python3
"""v14: PID+exc plus a certified online counterfactual information ray.

The learned policy still emits one non-negative scalar gain in ``[0, 1]``.
Before any learned authority reaches the plant, v14 evaluates the *actual
unclipped composite action* and the unmodified PID+exc action from the same
online PF belief.  Learned authority is accepted only when the composite has
strictly better predicted minimum FIM eigenvalue and satisfies the frozen
formation and action certificates.  Every rejection, including gain zero,
applies the inner PID+exc action exactly.

V13 and all earlier sources are imported but never modified.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import uuv_v8_temporal_infofix as _v8
import uuv_v10_info_tracking as _v10
import uuv_v11_online as _v11
import uuv_v12_hybrid as _v12
import uuv_v13_info_ray as _v13


VERSION = "v14_info_ray_1.0"
V14_VARIANT = "info_ray_gain_v14"
CONTROLLER_ARCHITECTURE = (
    "pid_track_exc_plus_certified_online_counterfactual_information_ray_gain"
)
MANIFEST_FILENAME = "v14_run_manifest.json"
MANIFEST_SCHEMA_VERSION = 4

COUNTERFACTUAL_DELTA_EIG_MIN = 1e-8
COUNTERFACTUAL_PRED_ERR_DELTA_MAX_RATIO = 0.25
COUNTERFACTUAL_BONUS_WEIGHT = 0.75
COUNTERFACTUAL_BONUS_EIG_SCALE = 5e-5

# Stable-Baselines3's stdout formatter truncates the portion after ``env/`` to
# 36 characters.  Full v14 trace diagnostics deliberately use descriptive
# names, several of which collide after that truncation.  TensorBoard needs
# only the four decision variables below; TraceCallback and returned ``info``
# retain the complete dictionary.
TB_V14_NUMERIC_ALLOWLIST = frozenset(
    (
        "v14_counterfactual_certificate_ok",
        "v14_counterfactual_delta_eig",
        "v14_counterfactual_bonus",
        "v14_counterfactual_reject_mask",
    )
)

# The mask is deliberately orthogonal to the frozen v13 planner-certificate
# mask, which is reported separately in the inherited v13 diagnostics.
CF_REJECT_BASIC_V13_CERTIFICATE = 1 << 0
CF_REJECT_ZERO_OR_NO_AUTHORITY = 1 << 1
CF_REJECT_NONFINITE = 1 << 2
CF_REJECT_ACTION_BOUNDS = 1 << 3
CF_REJECT_WOULD_CLIP = 1 << 4
CF_REJECT_EVALUATION_FAILED = 1 << 5
CF_REJECT_DELTA_EIG = 1 << 6
CF_REJECT_CANDIDATE_PRED_ERR = 1 << 7
CF_REJECT_PRED_ERR_DEGRADATION = 1 << 8

_CF_REASON_BY_BIT: Tuple[Tuple[int, str], ...] = (
    (CF_REJECT_BASIC_V13_CERTIFICATE, "basic_v13_certificate_failed"),
    (CF_REJECT_ZERO_OR_NO_AUTHORITY, "zero_gain_or_no_authority"),
    (CF_REJECT_NONFINITE, "counterfactual_nonfinite"),
    (CF_REJECT_ACTION_BOUNDS, "candidate_action_out_of_bounds"),
    (CF_REJECT_WOULD_CLIP, "candidate_action_would_clip"),
    (CF_REJECT_EVALUATION_FAILED, "counterfactual_evaluation_failed"),
    (CF_REJECT_DELTA_EIG, "delta_eig_not_strictly_positive"),
    (CF_REJECT_CANDIDATE_PRED_ERR, "candidate_predicted_formation_unsafe"),
    (CF_REJECT_PRED_ERR_DEGRADATION, "predicted_formation_degradation_too_large"),
)


@dataclass
class UUV3DConfig(_v13.UUV3DConfig):
    """Frozen v14 controller, predictor, and reward contract."""

    v14_variant: str = V14_VARIANT
    v14_controller_architecture: str = CONTROLLER_ARCHITECTURE
    v14_policy_action_semantics: str = "direct_nonnegative_information_ray_gain"
    v14_observation_last_action_semantics: str = "applied_certified_plant_action"
    v14_direction_scale_speed: float = 0.30
    v14_direction_scale_yaw: float = 0.50
    v14_direction_scale_pitch: float = 0.40
    v14_counterfactual_delta_eig_min: float = COUNTERFACTUAL_DELTA_EIG_MIN
    v14_counterfactual_pred_err_delta_max_ratio: float = (
        COUNTERFACTUAL_PRED_ERR_DELTA_MAX_RATIO
    )
    v14_counterfactual_bonus_weight: float = COUNTERFACTUAL_BONUS_WEIGHT
    v14_counterfactual_bonus_eig_scale: float = COUNTERFACTUAL_BONUS_EIG_SCALE
    v14_counterfactual_backend: str = "auto"

    # V13 delayed all meaningful authority until one quarter of the
    # curriculum, while SAC's entropy coefficient had already collapsed.
    # V14 makes certified actions identifiable early without bypassing any
    # action or formation certificate.
    info_planner_start_difficulty: float = 0.0
    info_planner_ramp_difficulty: float = 0.10

    def __post_init__(self) -> None:
        # Calling v12 directly intentionally bypasses v13's frozen direction-
        # scale assertion.  The inherited v13 fields remain validated below;
        # v14 uses its own explicitly versioned scales.
        _v12.UUV3DConfig.__post_init__(self)
        if str(self.v13_variant).lower().strip() != _v13.V13_VARIANT:
            raise ValueError("v14 requires the frozen v13 information-ray parent")
        if str(self.v13_controller_architecture) != _v13.CONTROLLER_ARCHITECTURE:
            raise ValueError("v14 requires the frozen v13 parent architecture")
        if str(self.v14_variant).lower().strip() != V14_VARIANT:
            raise ValueError(f"unknown v14 variant {self.v14_variant!r}")
        self.v14_variant = V14_VARIANT
        if str(self.v14_controller_architecture) != CONTROLLER_ARCHITECTURE:
            raise ValueError("v14 controller architecture is frozen")

        expected_scales = np.asarray((0.30, 0.50, 0.40), dtype=np.float32)
        if not np.array_equal(self.direction_scales(), expected_scales):
            raise ValueError("v14 information-ray scales are frozen at [0.30, 0.50, 0.40]")
        frozen = (
            (self.v13_certificate_margin_min, 1e-6),
            (self.v12_track_gate_full_ratio, 0.50),
            (self.v12_track_gate_zero_ratio, 1.00),
            (self.v14_counterfactual_delta_eig_min, COUNTERFACTUAL_DELTA_EIG_MIN),
            (
                self.v14_counterfactual_pred_err_delta_max_ratio,
                COUNTERFACTUAL_PRED_ERR_DELTA_MAX_RATIO,
            ),
            (self.v14_counterfactual_bonus_weight, COUNTERFACTUAL_BONUS_WEIGHT),
            (
                self.v14_counterfactual_bonus_eig_scale,
                COUNTERFACTUAL_BONUS_EIG_SCALE,
            ),
            (self.info_planner_start_difficulty, 0.0),
            (self.info_planner_ramp_difficulty, 0.10),
        )
        if any(not math.isclose(float(got), want, rel_tol=0.0, abs_tol=0.0) for got, want in frozen):
            raise ValueError(
                "v14 parent certificate, tracking gates, counterfactual thresholds, "
                "and planner curriculum are frozen"
            )
        backend = str(self.v14_counterfactual_backend).lower().strip()
        if backend not in {"auto", "python", "numba"}:
            raise ValueError("v14 counterfactual backend must be auto, python, or numba")
        self.v14_counterfactual_backend = backend

    def direction_scales(self) -> np.ndarray:
        return np.asarray(
            (
                self.v14_direction_scale_speed,
                self.v14_direction_scale_yaw,
                self.v14_direction_scale_pitch,
            ),
            dtype=np.float32,
        )

    def action_contract(self) -> Dict[str, Any]:
        return {
            "architecture": CONTROLLER_ARCHITECTURE,
            "variant": V14_VARIANT,
            "candidate_controller_id": "rl",
            "inner_controller": "pid_track_exc",
            "policy_action_semantics": "direct_nonnegative_information_ray_gain",
            "policy_action_shape": [1],
            "policy_action_dim": 1,
            "policy_action_bounds": [0.0, 1.0],
            "policy_action_low": [0.0],
            "policy_action_high": [1.0],
            "plant_action_shape": [3],
            "plant_action_dim": 3,
            "plant_action_low": [-1.0, -1.0, -1.0],
            "plant_action_high": [1.0, 1.0, 1.0],
            "information_direction_scale": [0.30, 0.50, 0.40],
            "track_gate_full_ratio": 0.50,
            "track_gate_zero_ratio": 1.00,
            "information_gate": "max(online_planner_gate, online_uncertainty_need)",
            "basic_certificate": "frozen_v13_online_planner_certificate",
            "counterfactual": (
                "same online PF belief; actual preclip composite versus exact inner; "
                "finite bounded no-clip candidate; delta_eig>1e-8; candidate_pred_err<=tol_pos; "
                "candidate_pred_err-inner_pred_err<=0.25*tol_pos"
            ),
            "composition": (
                "accept(inner_pid_exc + gain*track_gate*information_gate*"
                "diag(0.30,0.50,0.40)*certified_planner_action) else exact inner_pid_exc"
            ),
            "neutral_branch": "exact inner.copy without addition or clipping",
            "observation_last_action_semantics": str(
                self.v14_observation_last_action_semantics
            ),
            "reward_semantics": (
                "v11_full_online plus once-only accepted counterfactual bonus "
                "0.75*tanh(delta_eig/5e-5)"
            ),
            "counterfactual_bonus_weight": COUNTERFACTUAL_BONUS_WEIGHT,
            "counterfactual_bonus_eig_scale": COUNTERFACTUAL_BONUS_EIG_SCALE,
            "counterfactual_delta_eig_min": COUNTERFACTUAL_DELTA_EIG_MIN,
            "counterfactual_pred_err_delta_max_ratio": (
                COUNTERFACTUAL_PRED_ERR_DELTA_MAX_RATIO
            ),
            "planner_start_difficulty": 0.0,
            "planner_ramp_difficulty": 0.10,
            "baseline_action_mode": "direct_3d",
        }


@dataclass(frozen=True)
class CounterfactualPair:
    inner: _v8.InfoPlanEval
    candidate: _v8.InfoPlanEval
    backend: str

    @property
    def delta_eig(self) -> float:
        return float(self.candidate.eigmin - self.inner.eigmin)

    @property
    def pred_err_delta(self) -> float:
        return float(self.candidate.pred_err - self.inner.pred_err)


@dataclass(frozen=True)
class V14ActionComposition:
    policy_gain: float
    basic: _v13.InfoRayActionComposition
    candidate_preclip: np.ndarray
    counterfactual_inner_eig: float
    counterfactual_candidate_eig: float
    counterfactual_delta_eig: float
    counterfactual_inner_pred_err: float
    counterfactual_candidate_pred_err: float
    counterfactual_pred_err_delta: float
    counterfactual_backend: str
    counterfactual_certificate_ok: bool
    counterfactual_reject_mask: int
    counterfactual_reject_reasons: Tuple[str, ...]
    action_applied: np.ndarray
    accepted_delta: np.ndarray


def _eval_from_metrics(action: np.ndarray, metrics: np.ndarray) -> _v8.InfoPlanEval:
    values = np.asarray(metrics, dtype=float).reshape(9)
    return _v8.InfoPlanEval(
        action=np.asarray(action, dtype=np.float32).copy(),
        score=float(values[0]),
        trace_red=float(values[1]),
        eigmin=float(values[2]),
        sens_avg=float(values[3]),
        pred_err=float(values[4]),
        gate_avg=float(values[5]),
        disp_world_first=np.asarray(values[6:9], dtype=float).copy(),
    )


def _eval_is_finite(value: _v8.InfoPlanEval) -> bool:
    scalars = np.asarray(
        (
            value.score,
            value.trace_red,
            value.eigmin,
            value.sens_avg,
            value.pred_err,
            value.gate_avg,
        ),
        dtype=float,
    )
    return bool(
        np.all(np.isfinite(scalars))
        and np.all(np.isfinite(np.asarray(value.action, dtype=float)))
        and np.all(np.isfinite(np.asarray(value.disp_world_first, dtype=float)))
    )


def _numba_pair_metrics(
    env: "UUVTwoLeader3DPFEnv",
    candidates: np.ndarray,
    support_points: np.ndarray,
    support_weights: np.ndarray,
    p0_eff: np.ndarray,
    tol_pos_est: float,
) -> np.ndarray:
    """Evaluate arbitrary actions through the same Numba kernel as v10."""

    cfg = env.cfg
    if not bool(getattr(_v8, "_NUMBA_AVAILABLE", False)):
        raise RuntimeError("Numba counterfactual backend requested but Numba is unavailable")
    _best, _zero, _second, metrics = _v10._v10_fast_info_search(
        np.ascontiguousarray(candidates, dtype=np.float64),
        np.ascontiguousarray(support_points, dtype=np.float64),
        np.ascontiguousarray(support_weights, dtype=np.float64),
        np.ascontiguousarray(p0_eff, dtype=np.float64),
        np.ascontiguousarray(env.pL1, dtype=np.float64),
        np.ascontiguousarray(env.pL2, dtype=np.float64),
        np.ascontiguousarray(env.vL1, dtype=np.float64),
        np.ascontiguousarray(env.vL2, dtype=np.float64),
        float(env.speed_F),
        float(env.yaw_F),
        float(env.pitch_F),
        float(env._difficulty),
        float(env._gate_relax_active),
        float(tol_pos_est),
        float(cfg.action_dt),
        float(cfg.rl_speed_delta_per_step),
        float(cfg.max_yaw_rate_deg_s),
        float(cfg.rl_yaw_per_step_deg),
        float(cfg.max_pitch_rate_deg_s),
        float(cfg.rl_pitch_per_step_deg),
        float(cfg.f_min_speed),
        float(cfg.f_max_speed),
        float(cfg.pitch_min_deg),
        float(cfg.pitch_max_deg),
        float(cfg.info_planner_dt),
        float(cfg.info_planner_horizon_s),
        float(cfg.doppler_rho_min),
        float(cfg.doppler_v_min),
        float(cfg.doppler_v_perp_min_easy),
        float(cfg.doppler_v_perp_min_hard),
        float(cfg.gate_band_rho),
        float(cfg.gate_band_v),
        float(cfg.gate_band_v_perp),
        float(cfg.gate_min_factor),
        float(cfg.gate_relax_v_perp_min_hard),
        float(cfg.gate_relax_band_v_perp),
        float(env.pf.meas_sigma),
        float(getattr(env.pf, "sigma_nis_mult", 1.0)),
        float(cfg.d_back),
        float(cfg.d_right),
        float(cfg.dz_offset),
        float(cfg.info_planner_goal_err0),
        float(cfg.info_planner_trace_red0),
        float(cfg.info_planner_eigmin0),
        float(cfg.sens_norm),
        float(cfg.info_planner_w_trace),
        float(cfg.info_planner_w_eigmin),
        float(cfg.info_planner_w_sens),
        float(cfg.info_planner_w_goal),
        float(cfg.info_planner_w_energy),
    )
    return np.asarray(metrics, dtype=float)


def evaluate_counterfactual_pair(
    env: "UUVTwoLeader3DPFEnv",
    inner_action: Sequence[float],
    candidate_action: Sequence[float],
    tol_pos_est: float,
    *,
    backend: str = "auto",
) -> CounterfactualPair:
    """Pure pre-step comparison from one shared online belief.

    The inherited predictor reads ``speed_F/yaw_F/pitch_F``.  Those simulator
    truth fields are therefore shadowed with dead-reckoning measurements for
    *both* candidates and restored in a ``finally`` block.  Support points are
    constructed once and copied by each evaluator; no RNG is consumed.
    """

    inner = np.asarray(inner_action, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_action, dtype=np.float64).reshape(-1)
    if inner.shape != (3,) or candidate.shape != (3,):
        raise ValueError("counterfactual actions must each have shape (3,)")
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(candidate)):
        raise ValueError("counterfactual actions must be finite")
    tol = float(tol_pos_est)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("counterfactual formation tolerance must be finite and positive")

    selected = str(backend).lower().strip()
    if selected == "auto":
        selected = "numba" if bool(getattr(_v8, "_NUMBA_AVAILABLE", False)) else "python"
    if selected not in {"python", "numba"}:
        raise ValueError("counterfactual backend must be auto, python, or numba")

    speed_truth = float(env.speed_F)
    yaw_truth = float(env.yaw_F)
    pitch_truth = float(env.pitch_F)
    vf_truth = np.asarray(env.vF, dtype=float).copy()
    env.speed_F = float(env._v11_speed_meas)
    env.yaw_F = float(env._v11_yaw_meas)
    env.pitch_F = float(env._v11_pitch_meas)
    env.vF = np.asarray(env._v11_vf_meas, dtype=float).copy()
    try:
        support_points, support_weights, p0_eff = env._info_support_points()
        # Construct the belief once.  Both evaluators receive the identical
        # values; their implementations copy before propagation.
        support_points = np.asarray(support_points, dtype=np.float64).copy()
        support_weights = np.asarray(support_weights, dtype=np.float64).copy()
        p0_eff = np.asarray(p0_eff, dtype=np.float64).copy()
        if selected == "numba":
            actions = np.vstack((inner, candidate))
            metrics = _numba_pair_metrics(
                env,
                actions,
                support_points,
                support_weights,
                p0_eff,
                tol,
            )
            inner_eval = _eval_from_metrics(inner, metrics[0])
            candidate_eval = _eval_from_metrics(candidate, metrics[1])
        else:
            inner_eval = env._info_eval_action(
                inner, support_points, support_weights, p0_eff, tol
            )
            candidate_eval = env._info_eval_action(
                candidate, support_points, support_weights, p0_eff, tol
            )
    finally:
        env.speed_F = speed_truth
        env.yaw_F = yaw_truth
        env.pitch_F = pitch_truth
        env.vF = vf_truth

    return CounterfactualPair(
        inner=inner_eval,
        candidate=candidate_eval,
        backend=selected,
    )


def _nan_pair(backend: str = "none") -> CounterfactualPair:
    def item() -> _v8.InfoPlanEval:
        return _v8.InfoPlanEval(
            action=np.zeros(3, dtype=np.float32),
            score=float("nan"),
            trace_red=float("nan"),
            eigmin=float("nan"),
            sens_avg=float("nan"),
            pred_err=float("nan"),
            gate_avg=float("nan"),
            disp_world_first=np.full(3, float("nan"), dtype=float),
        )

    return CounterfactualPair(inner=item(), candidate=item(), backend=str(backend))


def compose_v14_action(
    env: "UUVTwoLeader3DPFEnv",
    online_info: Mapping[str, Any],
    policy_gain: Sequence[float],
    step_count: int,
) -> V14ActionComposition:
    """Build and certify the action without advancing the environment."""

    cfg = env.cfg
    basic = _v13.compose_info_ray_action(
        online_info, cfg, policy_gain, int(step_count)
    )
    gain = float(basic.policy_gain)
    inner = np.asarray(basic.inner_pid_exc, dtype=np.float32).copy()
    candidate = np.asarray(basic.action_pre_clip, dtype=np.float32).copy()
    mask = 0

    if not bool(basic.certificate_ok):
        mask |= CF_REJECT_BASIC_V13_CERTIFICATE
    if not bool(basic.authority_eligible) or gain <= 0.0:
        mask |= CF_REJECT_ZERO_OR_NO_AUTHORITY
    if not (
        np.all(np.isfinite(inner))
        and np.all(np.isfinite(candidate))
        and np.all(np.isfinite(np.asarray(basic.effective_action, dtype=float)))
    ):
        mask |= CF_REJECT_NONFINITE
    if np.any(candidate < -1.0) or np.any(candidate > 1.0):
        mask |= CF_REJECT_ACTION_BOUNDS
    clipped = np.clip(candidate, -1.0, 1.0).astype(np.float32)
    if not np.array_equal(clipped, candidate) or np.any(basic.clipped_channels > 0.5):
        mask |= CF_REJECT_WOULD_CLIP

    pair = _nan_pair()
    prerequisite_mask = (
        CF_REJECT_BASIC_V13_CERTIFICATE
        | CF_REJECT_ZERO_OR_NO_AUTHORITY
        | CF_REJECT_NONFINITE
        | CF_REJECT_ACTION_BOUNDS
        | CF_REJECT_WOULD_CLIP
    )
    if (mask & prerequisite_mask) == 0:
        try:
            snapshot = _v13.online_pid_exc_snapshot(online_info)
            tol_pos = float(snapshot["tol_pos_est"])
            pair = evaluate_counterfactual_pair(
                env,
                inner,
                candidate,
                tol_pos,
                backend=str(cfg.v14_counterfactual_backend),
            )
        except Exception:
            mask |= CF_REJECT_EVALUATION_FAILED
        else:
            if not (_eval_is_finite(pair.inner) and _eval_is_finite(pair.candidate)):
                mask |= CF_REJECT_NONFINITE
            else:
                delta_eig = float(pair.delta_eig)
                if delta_eig <= float(cfg.v14_counterfactual_delta_eig_min):
                    mask |= CF_REJECT_DELTA_EIG
                if float(pair.candidate.pred_err) > tol_pos:
                    mask |= CF_REJECT_CANDIDATE_PRED_ERR
                if float(pair.pred_err_delta) > (
                    float(cfg.v14_counterfactual_pred_err_delta_max_ratio) * tol_pos
                ):
                    mask |= CF_REJECT_PRED_ERR_DEGRADATION

    accepted = mask == 0
    if accepted:
        applied = candidate.copy()
        accepted_delta = (candidate - inner).astype(np.float32)
    else:
        # Do not add, clip, or round in the neutral branch.
        applied = inner.copy()
        accepted_delta = np.zeros(3, dtype=np.float32)

    reasons = tuple(name for bit, name in _CF_REASON_BY_BIT if mask & bit)
    return V14ActionComposition(
        policy_gain=gain,
        basic=basic,
        candidate_preclip=candidate,
        counterfactual_inner_eig=float(pair.inner.eigmin),
        counterfactual_candidate_eig=float(pair.candidate.eigmin),
        counterfactual_delta_eig=float(pair.delta_eig),
        counterfactual_inner_pred_err=float(pair.inner.pred_err),
        counterfactual_candidate_pred_err=float(pair.candidate.pred_err),
        counterfactual_pred_err_delta=float(pair.pred_err_delta),
        counterfactual_backend=str(pair.backend),
        counterfactual_certificate_ok=accepted,
        counterfactual_reject_mask=int(mask),
        counterfactual_reject_reasons=reasons,
        action_applied=applied,
        accepted_delta=accepted_delta,
    )


class UUVTwoLeader3DPFEnv(_v13.UUVTwoLeader3DPFEnv):
    """V13 plant with v14 pre-step counterfactual authority certification."""

    BASE_OBS_DIM = _v13.UUVTwoLeader3DPFEnv.BASE_OBS_DIM

    def __init__(self, cfg: Optional[UUV3DConfig] = None, render_mode: str = "none"):
        super().__init__(cfg=cfg or UUV3DConfig(), render_mode=render_mode)
        self.cfg: UUV3DConfig

    def _init_info_ray_diagnostics(self) -> None:
        super()._init_info_ray_diagnostics()
        self._init_v14_diagnostics()

    def _init_v14_diagnostics(self) -> None:
        self._v14_counterfactual_certificate_ok = False
        self._v14_counterfactual_delta_eig = float("nan")
        self._v14_counterfactual_inner_eig = float("nan")
        self._v14_counterfactual_candidate_eig = float("nan")
        self._v14_counterfactual_inner_pred_err = float("nan")
        self._v14_counterfactual_candidate_pred_err = float("nan")
        self._v14_counterfactual_pred_err_delta = float("nan")
        self._v14_counterfactual_bonus = 0.0
        self._v14_counterfactual_reject_mask = 0
        self._v14_counterfactual_reject_reasons: Tuple[str, ...] = ()
        self._v14_counterfactual_backend = "none"
        self._v14_counterfactual_candidate_preclip = np.zeros(3, dtype=np.float32)
        self._v14_counterfactual_accepted_delta = np.zeros(3, dtype=np.float32)
        self._v14_counterfactual_evaluated_step = -1
        self._v14_counterfactual_rewarded_step = -1
        self._v14_counterfactual_bonus_pending = False

    def _reset_info_ray_diagnostics(self) -> None:
        super()._reset_info_ray_diagnostics()
        self._init_v14_diagnostics()

    def _record_v14_composition(self, composition: V14ActionComposition) -> None:
        basic = composition.basic
        if composition.counterfactual_certificate_ok:
            reported_basic = replace(
                basic,
                action_applied=composition.action_applied.copy(),
                realized_delta=composition.accepted_delta.copy(),
                clipped_channels=np.zeros(3, dtype=np.float32),
                neutral_branch=False,
            )
        else:
            reported_basic = replace(
                basic,
                authority_eligible=False,
                effective_action=np.zeros(3, dtype=np.float32),
                action_pre_clip=basic.inner_pid_exc.copy(),
                action_applied=basic.inner_pid_exc.copy(),
                realized_delta=np.zeros(3, dtype=np.float32),
                clipped_channels=np.zeros(3, dtype=np.float32),
                intended_alignment_dot=0.0,
                realized_alignment_dot=0.0,
                intended_alignment_cosine=0.0,
                realized_alignment_cosine=0.0,
                intended_alignment_violation=False,
                realized_alignment_violation=False,
                neutral_branch=True,
            )
        super()._record_composition(reported_basic)

        self._v14_counterfactual_certificate_ok = bool(
            composition.counterfactual_certificate_ok
        )
        self._v14_counterfactual_delta_eig = float(
            composition.counterfactual_delta_eig
        )
        self._v14_counterfactual_inner_eig = float(
            composition.counterfactual_inner_eig
        )
        self._v14_counterfactual_candidate_eig = float(
            composition.counterfactual_candidate_eig
        )
        self._v14_counterfactual_inner_pred_err = float(
            composition.counterfactual_inner_pred_err
        )
        self._v14_counterfactual_candidate_pred_err = float(
            composition.counterfactual_candidate_pred_err
        )
        self._v14_counterfactual_pred_err_delta = float(
            composition.counterfactual_pred_err_delta
        )
        self._v14_counterfactual_bonus = 0.0
        self._v14_counterfactual_reject_mask = int(
            composition.counterfactual_reject_mask
        )
        self._v14_counterfactual_reject_reasons = (
            composition.counterfactual_reject_reasons
        )
        self._v14_counterfactual_backend = str(composition.counterfactual_backend)
        self._v14_counterfactual_candidate_preclip = (
            composition.candidate_preclip.copy()
        )
        self._v14_counterfactual_accepted_delta = composition.accepted_delta.copy()
        self._v14_counterfactual_evaluated_step = int(self.step_count)
        self._v14_counterfactual_rewarded_step = -1
        self._v14_counterfactual_bonus_pending = bool(
            composition.counterfactual_certificate_ok
        )

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)
        info.update(
            {
                "v14_version": 14.0,
                "v14_counterfactual_certificate_ok": float(
                    self._v14_counterfactual_certificate_ok
                ),
                "v14_counterfactual_delta_eig": float(
                    self._v14_counterfactual_delta_eig
                ),
                "v14_counterfactual_bonus": float(
                    self._v14_counterfactual_bonus
                ),
                "v14_counterfactual_reject_mask": int(
                    self._v14_counterfactual_reject_mask
                ),
                "v14_counterfactual_reject_reasons": ",".join(
                    self._v14_counterfactual_reject_reasons
                ),
                "v14_counterfactual_inner_eig": float(
                    self._v14_counterfactual_inner_eig
                ),
                "v14_counterfactual_candidate_eig": float(
                    self._v14_counterfactual_candidate_eig
                ),
                "v14_counterfactual_inner_pred_err": float(
                    self._v14_counterfactual_inner_pred_err
                ),
                "v14_counterfactual_candidate_pred_err": float(
                    self._v14_counterfactual_candidate_pred_err
                ),
                "v14_counterfactual_pred_err_delta": float(
                    self._v14_counterfactual_pred_err_delta
                ),
                "v14_counterfactual_backend": str(
                    self._v14_counterfactual_backend
                ),
                "v14_counterfactual_evaluated_step": int(
                    self._v14_counterfactual_evaluated_step
                ),
                "v14_counterfactual_rewarded_step": int(
                    self._v14_counterfactual_rewarded_step
                ),
                "v14_counterfactual_bonus_pending": float(
                    self._v14_counterfactual_bonus_pending
                ),
            }
        )
        for prefix, values in (
            (
                "v14_counterfactual_candidate_preclip",
                self._v14_counterfactual_candidate_preclip,
            ),
            (
                "v14_counterfactual_accepted_delta",
                self._v14_counterfactual_accepted_delta,
            ),
        ):
            for axis, value in zip(("speed", "yaw", "pitch"), values):
                info[f"{prefix}_{axis}"] = float(value)
        return info

    def _compute_reward(
        self,
        pf_stats_last: Optional[_v8.PFStats],
        planner_action_for_reward: Optional[np.ndarray] = None,
        planner_gate_for_reward: Optional[float] = None,
        planner_margin_for_reward: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        base_reward, terms = super()._compute_reward(
            pf_stats_last,
            planner_action_for_reward,
            planner_gate_for_reward,
            planner_margin_for_reward,
        )

        bonus = 0.0
        pending = bool(self._v14_counterfactual_bonus_pending)
        if pending:
            expected_evaluated_step = int(self.step_count) - 1
            if int(self._v14_counterfactual_evaluated_step) != expected_evaluated_step:
                raise RuntimeError("stale v14 pre-step counterfactual reward cache")
            if not bool(self._v14_counterfactual_certificate_ok):
                raise RuntimeError("uncertified v14 counterfactual bonus was pending")
            delta = float(self._v14_counterfactual_delta_eig)
            if not np.isfinite(delta) or delta <= float(
                self.cfg.v14_counterfactual_delta_eig_min
            ):
                raise RuntimeError("invalid v14 delta_eig reached reward cache")
            bonus = float(self.cfg.v14_counterfactual_bonus_weight) * math.tanh(
                delta / float(self.cfg.v14_counterfactual_bonus_eig_scale)
            )
            self._v14_counterfactual_bonus_pending = False
            self._v14_counterfactual_rewarded_step = int(self.step_count)
            self._v14_counterfactual_bonus = float(bonus)

        terms.update(
            {
                "v14_version": 14.0,
                "v14_reward_uses_truth": 0.0,
                "v14_counterfactual_certificate_ok": float(
                    self._v14_counterfactual_certificate_ok
                ),
                "v14_counterfactual_delta_eig": float(
                    self._v14_counterfactual_delta_eig
                ),
                "v14_counterfactual_bonus": float(bonus),
                "v14_counterfactual_reject_mask": float(
                    self._v14_counterfactual_reject_mask
                ),
                "v14_counterfactual_inner_eig": float(
                    self._v14_counterfactual_inner_eig
                ),
                "v14_counterfactual_candidate_eig": float(
                    self._v14_counterfactual_candidate_eig
                ),
                "v14_counterfactual_inner_pred_err": float(
                    self._v14_counterfactual_inner_pred_err
                ),
                "v14_counterfactual_candidate_pred_err": float(
                    self._v14_counterfactual_candidate_pred_err
                ),
                "v14_counterfactual_pred_err_delta": float(
                    self._v14_counterfactual_pred_err_delta
                ),
            }
        )
        if bonus == 0.0:
            # Preserve the inherited reward bit-for-bit on every neutral or
            # rejected branch; do not even perform an addition by zero.
            return float(base_reward), terms
        return float(
            np.clip(
                float(base_reward) + bonus,
                -float(self.cfg.rew_clip),
                float(self.cfg.rew_clip),
            )
        ), terms

    def step(self, action: Sequence[float]):
        if self._policy_uses_info_ray():
            composition = compose_v14_action(
                self,
                self._get_info(),
                action,
                self.step_count,
            )
            self._v13_info_ray_active = True
            self._record_v14_composition(composition)
            # Bypass both v13 and v12 composers.  The selected plant action is
            # already final and the reward override consumes its pre-step cache.
            return _v11.UUVTwoLeader3DPFEnv.step(
                self, composition.action_applied
            )

        # Direct baselines keep their frozen 3-D semantics.  Clear v14-only
        # state before delegating to v13's already-tested direct branch.
        self._init_v14_diagnostics()
        return super().step(action)


def make_env(seed: int, cfg: UUV3DConfig, render: bool, rank: int = 0):
    def _init():
        local_cfg = replace(cfg, v11_env_rank=int(rank))
        env = UUVTwoLeader3DPFEnv(
            cfg=local_cfg,
            render_mode=("human" if render else "none"),
        )
        env._v11_master_seed = int(seed)
        env._v11_episode_index = -1
        return env

    return _init


def _tb_infos_copy(infos: Any) -> Any:
    """Copy callback infos while retaining only collision-free v14 scalars.

    This is intentionally a logger-boundary adapter.  It never mutates the
    environment dictionaries and is not installed for TraceCallback, reward,
    evaluation, or user-facing ``info``.
    """

    if infos is None:
        return None
    copied = []
    for info in infos:
        if not isinstance(info, dict):
            copied.append(info)
            continue
        filtered: Dict[str, Any] = {}
        for key, value in info.items():
            name = str(key)
            if not name.startswith("v14_"):
                filtered[key] = value
                continue
            if name not in TB_V14_NUMERIC_ALLOWLIST:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                filtered[key] = value
        copied.append(filtered)
    return copied


class V14TBInfoCallback(_v8.TBInfoCallback):
    """V8 logger with a non-mutating, collision-free v14 view of ``infos``."""

    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        callback_locals = dict(locals_)
        callback_locals["infos"] = _tb_infos_copy(locals_.get("infos"))
        return super().__call__(callback_locals, globals_)


def build_parser() -> argparse.ArgumentParser:
    parser = _v13.build_parser()
    replacements = {
        "models_3d_v13_info_ray": "models_3d_v14_info_ray",
        "logs_3d_v13_info_ray": "logs_3d_v14_info_ray",
        "tb_3d_v13_info_ray": "tb_3d_v14_info_ray",
        "eval_3d_logs_v13_info_ray": "eval_3d_logs_v14_info_ray",
        "info_maps_v13_info_ray": "info_maps_v14_info_ray",
    }
    for parser_action in _v11._iter_parser_actions(parser):
        default = getattr(parser_action, "default", None)
        if isinstance(default, str):
            for old, new in replacements.items():
                if old in default:
                    parser_action.default = default.replace(old, new)
                    break
    train_parser = _v11._v10._get_subparser(parser, "train")
    if train_parser is not None and not _v11._v10._parser_has_dest(
        train_parser, "v14_variant"
    ):
        train_parser.add_argument(
            "--v14-variant",
            choices=(V14_VARIANT,),
            default=V14_VARIANT,
            help="Frozen v14 online counterfactual information-ray controller.",
        )
    return parser


@contextmanager
def _patched_training_globals(*, patch_env_class: bool):
    v10 = _v11._v10
    v8 = _v11._v8
    old = {
        "UUV3DConfig": v10.UUV3DConfig,
        "UUVTwoLeader3DPFEnv": v10.UUVTwoLeader3DPFEnv,
        "make_env": v10.make_env,
        "v8_UUV3DConfig": v8.UUV3DConfig,
        "v8_UUVTwoLeader3DPFEnv": v8.UUVTwoLeader3DPFEnv,
        "v8_make_env": v8.make_env,
        "v8_TBInfoCallback": v8.TBInfoCallback,
    }
    v10.UUV3DConfig = UUV3DConfig
    v10.make_env = make_env
    v8.TBInfoCallback = V14TBInfoCallback
    if patch_env_class:
        v10.UUVTwoLeader3DPFEnv = UUVTwoLeader3DPFEnv
    try:
        yield
    finally:
        v10.UUV3DConfig = old["UUV3DConfig"]
        v10.UUVTwoLeader3DPFEnv = old["UUVTwoLeader3DPFEnv"]
        v10.make_env = old["make_env"]
        v8.UUV3DConfig = old["v8_UUV3DConfig"]
        v8.UUVTwoLeader3DPFEnv = old["v8_UUVTwoLeader3DPFEnv"]
        v8.make_env = old["v8_make_env"]
        v8.TBInfoCallback = old["v8_TBInfoCallback"]


SOURCE_NAMES: Tuple[str, ...] = (
    "uuv_v14_info_ray.py",
    "uuv_v14_evaluate.py",
    "run_v14_pilot.py",
    "EXPERIMENT_PROTOCOL_V14.md",
    "tests/test_uuv_v14_info_ray.py",
    "tests/test_uuv_v14_evaluate.py",
    "tests/test_run_v14_pilot.py",
    "uuv_v13_info_ray.py",
    "uuv_v13_evaluate.py",
    "run_v13_experiments.py",
    "EXPERIMENT_PROTOCOL_V13.md",
    "tests/test_uuv_v13_info_ray.py",
    "tests/test_uuv_v13_evaluate.py",
    "tests/test_run_v13_experiments.py",
    "uuv_v12_hybrid.py",
    "uuv_v11_online.py",
    "uuv_v11_evaluate.py",
    "uuv_v11_rng.py",
    "uuv_v11_metrics.py",
    "baseline_controllers_v11.py",
    "V10_ARCHIVE_MANIFEST.json",
    "uuv_v10_info_tracking.py",
    "uuv_v8_temporal_infofix.py",
)


def _training_config_preview(args: argparse.Namespace) -> UUV3DConfig:
    total_timesteps = int(args.total_timesteps)
    n_envs = max(1, int(args.n_envs))
    curriculum_frac = float(
        np.clip(float(getattr(args, "curriculum_frac", 0.90)), 0.0, 1.0)
    )
    curriculum_steps = int(
        max(1, round(curriculum_frac * total_timesteps / n_envs))
    )
    return UUV3DConfig(
        curriculum_steps=curriculum_steps,
        difficulty_fixed=(
            float(args.difficulty_fixed)
            if getattr(args, "difficulty_fixed", None) is not None
            else None
        ),
        pf_use_numba=bool(getattr(args, "pf_numba", True)),
        action_dt=float(args.action_dt),
        fim_window_s=float(getattr(args, "fim_window", 30.0)),
        log_truth_diagnostics=bool(getattr(args, "log_truth", False)),
        success_mode=str(getattr(args, "success_mode", "progress")),
        success_require_std=bool(getattr(args, "success_require_std", True)),
        pf_num_particles=int(
            getattr(args, "pf_particles", UUV3DConfig.pf_num_particles)
        ),
        obs_history_len=int(
            getattr(args, "obs_history_len", UUV3DConfig.obs_history_len)
        ),
        info_gate_floor_hard=float(
            getattr(args, "info_gate_floor_hard", UUV3DConfig.info_gate_floor_hard)
        ),
        info_gate_floor_hard_extra=float(
            getattr(
                args,
                "info_gate_floor_hard_extra",
                UUV3DConfig.info_gate_floor_hard_extra,
            )
        ),
        info_gate_floor_start_difficulty=float(
            getattr(
                args,
                "info_gate_floor_start_difficulty",
                UUV3DConfig.info_gate_floor_start_difficulty,
            )
        ),
        info_gate_floor_ramp_difficulty=float(
            getattr(
                args,
                "info_gate_floor_ramp_difficulty",
                UUV3DConfig.info_gate_floor_ramp_difficulty,
            )
        ),
        v11_variant="full_online",
        v12_variant=_v12.V12_VARIANT,
        v13_variant=_v13.V13_VARIANT,
        v14_variant=str(getattr(args, "v14_variant", V14_VARIANT)),
    )


def cmd_train(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resume", False)):
        raise NotImplementedError("scientific resume is disabled")
    if str(getattr(args, "v11_variant", "full_online")) != "full_online":
        raise ValueError("v14 requires --v11-variant full_online")
    if str(getattr(args, "v12_variant", _v12.V12_VARIANT)) != _v12.V12_VARIANT:
        raise ValueError(f"v14 requires --v12-variant {_v12.V12_VARIANT}")
    if str(getattr(args, "v13_variant", _v13.V13_VARIANT)) != _v13.V13_VARIANT:
        raise ValueError(f"v14 requires --v13-variant {_v13.V13_VARIANT}")
    if str(getattr(args, "v14_variant", V14_VARIANT)) != V14_VARIANT:
        raise ValueError(f"v14 requires --v14-variant {V14_VARIANT}")
    if str(getattr(args, "success_mode", "progress")) != "progress":
        raise ValueError("v14 uses the frozen v11 online progress success definition")
    if not math.isclose(float(getattr(args, "action_dt", 2.0)), 2.0, abs_tol=1e-12):
        raise ValueError("the v14 training protocol requires --action-dt 2.0 s")

    models_dir = Path(str(args.models_dir)).expanduser().resolve()
    manifest_path = models_dir / MANIFEST_FILENAME
    if models_dir.is_dir() and any(models_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty v14 model directory: {models_dir}"
        )

    source_dir = Path(__file__).resolve().parent
    snapshot_dir = models_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing required v14 source artifact: {source}")
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    cfg_preview = _training_config_preview(args)
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "version": VERSION,
        "variant": V14_VARIANT,
        "seed": int(getattr(args, "seed", 42)),
        "resume": False,
        "command": [sys.executable] + list(sys.argv),
        "arguments": vars(args),
        "environment_config": asdict(cfg_preview),
        "action_contract": cfg_preview.action_contract(),
        "observation_dim": int(cfg_preview.obs_history_len)
        * int(UUVTwoLeader3DPFEnv.BASE_OBS_DIM),
        "policy_action_dim": 1,
        "plant_action_dim": 3,
        "packages": _v11._package_versions(),
        "git_commit": _v11._git_commit(),
        "source_sha256": {
            name: _v11._sha256_file(source_dir / name) for name in SOURCE_NAMES
        },
        "source_snapshot_dir": str(snapshot_dir),
    }
    _v11._write_json_atomic(manifest_path, manifest)

    previous_variant = os.environ.get("UUV_V11_VARIANT")
    os.environ["UUV_V11_VARIANT"] = "full_online"
    try:
        with _patched_training_globals(patch_env_class=False):
            _v11._v10.cmd_train(args)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["artifacts_sha256"] = {
            name: _v11._sha256_file(models_dir / name)
            for name in ("final_model.zip", "last_model.zip", "vecnormalize.pkl")
        }
        _v11._write_json_atomic(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _v11._write_json_atomic(manifest_path, manifest)
        raise
    finally:
        if previous_variant is None:
            os.environ.pop("UUV_V11_VARIANT", None)
        else:
            os.environ["UUV_V11_VARIANT"] = previous_variant


def cmd_eval(args: argparse.Namespace) -> None:
    del args
    raise RuntimeError("run `python3 uuv_v14_evaluate.py --help` for v14 evaluation")


def cmd_sim(args: argparse.Namespace) -> None:
    with _patched_training_globals(patch_env_class=True):
        _v11._v10.cmd_sim(args)


def cmd_map(args: argparse.Namespace) -> None:
    with _patched_training_globals(patch_env_class=True):
        _v11._v10.cmd_map(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "sim":
        cmd_sim(args)
    elif args.cmd == "map":
        cmd_map(args)
    else:
        raise ValueError(f"unknown command: {args.cmd!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CF_REJECT_ACTION_BOUNDS",
    "CF_REJECT_BASIC_V13_CERTIFICATE",
    "CF_REJECT_CANDIDATE_PRED_ERR",
    "CF_REJECT_DELTA_EIG",
    "CF_REJECT_EVALUATION_FAILED",
    "CF_REJECT_NONFINITE",
    "CF_REJECT_PRED_ERR_DEGRADATION",
    "CF_REJECT_WOULD_CLIP",
    "CF_REJECT_ZERO_OR_NO_AUTHORITY",
    "CONTROLLER_ARCHITECTURE",
    "CounterfactualPair",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "SOURCE_NAMES",
    "TB_V14_NUMERIC_ALLOWLIST",
    "UUV3DConfig",
    "UUVTwoLeader3DPFEnv",
    "V14ActionComposition",
    "V14_VARIANT",
    "V14TBInfoCallback",
    "VERSION",
    "build_parser",
    "compose_v14_action",
    "evaluate_counterfactual_pair",
    "make_env",
]
