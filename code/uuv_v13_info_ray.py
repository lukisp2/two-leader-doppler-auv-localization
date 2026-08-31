#!/usr/bin/env python3
"""v13: PID+exc with a certified, non-negative information-ray gain.

V12 and all earlier sources remain immutable.  The learned policy emits one
scalar gain in ``[0, 1]``.  It may only amplify the current online planner ray;
it cannot rotate or oppose that ray.  A missing/invalid certificate removes
all learned authority and returns the restored PID+exc action exactly.
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
from gymnasium import spaces

import uuv_v11_online as _v11
import uuv_v12_hybrid as _v12
from baseline_controllers_v11 import pid_tracking_action_from_info


VERSION = "v13_info_ray_1.0"
V13_VARIANT = "info_ray_gain"
CONTROLLER_ARCHITECTURE = "pid_track_exc_plus_certified_online_information_ray_gain"
POLICY_CONTROLLER_IDS = frozenset(("policy", "rl"))

CERT_ACTION_NONFINITE = 1 << 0
CERT_ACTION_ZERO = 1 << 1
CERT_ACTION_OUT_OF_BOUNDS = 1 << 2
CERT_INACTIVE = 1 << 3
CERT_GATE_INVALID = 1 << 4
CERT_MARGIN_INVALID = 1 << 5
CERT_TRACE_REDUCTION_INVALID = 1 << 6
CERT_PREDICTED_FORMATION_UNSAFE = 1 << 7

_CERT_REASON_BY_BIT: Tuple[Tuple[int, str], ...] = (
    (CERT_ACTION_NONFINITE, "planner_action_nonfinite"),
    (CERT_ACTION_ZERO, "planner_action_zero"),
    (CERT_ACTION_OUT_OF_BOUNDS, "planner_action_out_of_bounds"),
    (CERT_INACTIVE, "planner_inactive"),
    (CERT_GATE_INVALID, "planner_gate_below_minimum"),
    (CERT_MARGIN_INVALID, "best_zero_margin_not_positive"),
    (CERT_TRACE_REDUCTION_INVALID, "trace_reduction_not_positive"),
    (CERT_PREDICTED_FORMATION_UNSAFE, "predicted_formation_unsafe"),
)


@dataclass
class UUV3DConfig(_v12.UUV3DConfig):
    """Frozen v13 action and safety contract."""

    v13_variant: str = V13_VARIANT
    v13_controller_architecture: str = CONTROLLER_ARCHITECTURE
    v13_policy_action_semantics: str = "direct_nonnegative_information_ray_gain"
    v13_observation_last_action_semantics: str = "applied_composite_plant_action"
    v13_direction_scale_speed: float = 0.15
    v13_direction_scale_yaw: float = 0.30
    v13_direction_scale_pitch: float = 0.25
    v13_certificate_margin_min: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if str(self.v13_variant).lower().strip() != V13_VARIANT:
            raise ValueError(f"unknown v13 variant {self.v13_variant!r}")
        self.v13_variant = V13_VARIANT
        if str(self.v13_controller_architecture) != CONTROLLER_ARCHITECTURE:
            raise ValueError("v13 controller architecture is frozen")
        expected = np.asarray((0.15, 0.30, 0.25), dtype=np.float32)
        if not np.array_equal(self.direction_scales(), expected):
            raise ValueError("v13 information-ray scales are frozen at [0.15, 0.30, 0.25]")
        if not math.isclose(float(self.v13_certificate_margin_min), 1e-6, abs_tol=0.0):
            raise ValueError("v13 certificate margin is frozen at 1e-6")
        if not (
            math.isclose(float(self.v12_track_gate_full_ratio), 0.50, abs_tol=0.0)
            and math.isclose(float(self.v12_track_gate_zero_ratio), 1.00, abs_tol=0.0)
        ):
            raise ValueError("v13 tracking gate is frozen at full=0.5 and zero=1.0")

    def direction_scales(self) -> np.ndarray:
        return np.asarray(
            (
                self.v13_direction_scale_speed,
                self.v13_direction_scale_yaw,
                self.v13_direction_scale_pitch,
            ),
            dtype=np.float32,
        )

    def action_contract(self) -> Dict[str, Any]:
        return {
            "architecture": CONTROLLER_ARCHITECTURE,
            "variant": V13_VARIANT,
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
            "information_direction_scale": [0.15, 0.30, 0.25],
            "track_gate_full_ratio": 0.50,
            "track_gate_zero_ratio": 1.00,
            "information_gate": "max(online_planner_gate, online_uncertainty_need)",
            "certificate": (
                "online aliases only; finite nonzero bounded planner action; active; "
                "gate>=info_planner_gate_min; best-zero>1e-6; trace_red>0; "
                "pred_err<=tol_pos"
            ),
            "composition": (
                "clip(inner_pid_exc + gain * track_gate * information_gate * "
                "diag(0.15,0.30,0.25) * certified_planner_action, -1, 1)"
            ),
            "neutral_branch": "exact inner.copy without addition or clipping",
            "observation_last_action_semantics": str(
                self.v13_observation_last_action_semantics
            ),
            "reward_semantics": "unchanged_v11_full_online_on_applied_plant_action",
            "baseline_action_mode": "direct_3d",
        }


@dataclass(frozen=True)
class InfoRayActionComposition:
    policy_gain: float
    inner_pid_exc: np.ndarray
    planner_action: np.ndarray
    information_direction: np.ndarray
    certificate_ok: bool
    certificate_failure_mask: int
    certificate_failure_reasons: Tuple[str, ...]
    authority_eligible: bool
    track_gate: float
    information_gate: float
    uncertainty_need: float
    formation_error_ratio: float
    effective_action: np.ndarray
    action_pre_clip: np.ndarray
    action_applied: np.ndarray
    realized_delta: np.ndarray
    clipped_channels: np.ndarray
    intended_alignment_dot: float
    realized_alignment_dot: float
    intended_alignment_cosine: float
    realized_alignment_cosine: float
    intended_alignment_violation: bool
    realized_alignment_violation: bool
    neutral_branch: bool


def _smoothstep01(value: float) -> float:
    t = float(np.clip(float(value), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _optional_float(info: Mapping[str, Any], key: str) -> float:
    try:
        return float(info.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a.astype(float), b.astype(float)) / (na * nb))


def online_pid_exc_snapshot(info: Mapping[str, Any]) -> Dict[str, float]:
    """Use the frozen v12 online-only allowlist for the inner PID+exc loop."""

    return _v12.online_pid_exc_snapshot(info)


def _planner_certificate(
    info: Mapping[str, Any], cfg: UUV3DConfig, tol_pos: float
) -> Tuple[np.ndarray, int, Tuple[str, ...]]:
    planner = np.asarray(
        [
            _optional_float(info, "v13_info_plan_a_speed_online"),
            _optional_float(info, "v13_info_plan_a_yaw_online"),
            _optional_float(info, "v13_info_plan_a_pitch_online"),
        ],
        dtype=np.float32,
    )
    mask = 0
    if not np.all(np.isfinite(planner)):
        mask |= CERT_ACTION_NONFINITE
    else:
        if float(np.linalg.norm(planner)) <= 1e-6:
            mask |= CERT_ACTION_ZERO
        if np.any(planner < -1.0) or np.any(planner > 1.0):
            mask |= CERT_ACTION_OUT_OF_BOUNDS

    active = _optional_float(info, "v13_info_plan_active_online")
    if not np.isfinite(active) or active < 0.5:
        mask |= CERT_INACTIVE

    gate = _optional_float(info, "v13_info_plan_gate_online")
    if not np.isfinite(gate) or gate < float(cfg.info_planner_gate_min) or gate > 1.0:
        mask |= CERT_GATE_INVALID

    score_best = _optional_float(info, "v13_info_plan_score_best_online")
    score_zero = _optional_float(info, "v13_info_plan_score_zero_online")
    margin = score_best - score_zero
    if (
        not np.isfinite(score_best)
        or not np.isfinite(score_zero)
        or not np.isfinite(margin)
        or margin <= float(cfg.v13_certificate_margin_min)
    ):
        mask |= CERT_MARGIN_INVALID

    trace_red = _optional_float(info, "v13_info_plan_trace_red_online")
    if not np.isfinite(trace_red) or trace_red <= 0.0:
        mask |= CERT_TRACE_REDUCTION_INVALID

    pred_err = _optional_float(info, "v13_info_plan_pred_err_online")
    if not np.isfinite(pred_err) or pred_err > float(tol_pos):
        mask |= CERT_PREDICTED_FORMATION_UNSAFE

    reasons = tuple(name for bit, name in _CERT_REASON_BY_BIT if mask & bit)
    return planner, int(mask), reasons


def compose_info_ray_action(
    online_info: Mapping[str, Any],
    cfg: UUV3DConfig,
    policy_gain: Sequence[float],
    step_count: int,
) -> InfoRayActionComposition:
    """Compose one deterministic action using online data only."""

    external = np.asarray(policy_gain, dtype=np.float32).reshape(-1)
    if external.shape != (1,) or not np.all(np.isfinite(external)):
        raise ValueError("policy gain must contain exactly one finite value")
    gain = float(external[0])
    if gain < 0.0 or gain > 1.0:
        raise ValueError("policy gain must be in the frozen [0, 1] action range")

    snapshot = online_pid_exc_snapshot(online_info)
    inner = pid_tracking_action_from_info(
        snapshot,
        cfg,
        mode="pid_track_exc",
        step_count=int(step_count),
    ).astype(np.float32)

    p_hat = np.asarray([snapshot[f"pFhat_{axis}"] for axis in "xyz"], dtype=float)
    p_des = np.asarray([snapshot[f"pFdes_{axis}"] for axis in "xyz"], dtype=float)
    tol_pos = float(snapshot["tol_pos_est"])
    error_ratio = float(np.linalg.norm(p_des - p_hat) / tol_pos)
    gate_width = max(
        float(cfg.v12_track_gate_zero_ratio - cfg.v12_track_gate_full_ratio),
        1e-12,
    )
    gate_t = (error_ratio - float(cfg.v12_track_gate_full_ratio)) / gate_width
    track_gate = float(1.0 - _smoothstep01(gate_t))

    tol_std = float(snapshot["tol_std"])
    uncertainty_need = float(
        np.clip(
            max(float(snapshot["pf_std_max_raw"]) - tol_std, 0.0) / tol_std,
            0.0,
            1.0,
        )
    )
    planner_gate_raw = _optional_float(online_info, "v13_info_plan_gate_online")
    planner_gate = float(np.clip(planner_gate_raw, 0.0, 1.0)) if np.isfinite(planner_gate_raw) else 0.0
    information_gate = float(np.clip(max(planner_gate, uncertainty_need), 0.0, 1.0))

    planner, certificate_mask, certificate_reasons = _planner_certificate(
        online_info, cfg, tol_pos
    )
    certificate_ok = certificate_mask == 0
    # A failed certificate never exposes a malformed or uncertified ray to
    # downstream diagnostics.  The raw planner proposal remains separately
    # logged for audit, while the usable information direction is finite zero.
    direction = (
        (cfg.direction_scales() * planner).astype(np.float32)
        if certificate_ok
        else np.zeros(3, dtype=np.float32)
    )
    authority_eligible = bool(
        certificate_ok
        and gain > 0.0
        and track_gate > 0.0
        and information_gate > 0.0
        and np.any(direction != 0.0)
    )

    if not authority_eligible:
        effective = np.zeros(3, dtype=np.float32)
        pre_clip = inner.copy()
        applied = inner.copy()
        realized = np.zeros(3, dtype=np.float32)
        clipped = np.zeros(3, dtype=np.float32)
        neutral_branch = True
    else:
        authority = np.float32(gain * track_gate * information_gate)
        effective = (authority * direction).astype(np.float32)
        pre_clip = (inner + effective).astype(np.float32)
        applied = np.clip(pre_clip, -1.0, 1.0).astype(np.float32)
        realized = (applied - inner).astype(np.float32)
        clipped = (np.abs(pre_clip - applied) > np.float32(1e-7)).astype(np.float32)
        neutral_branch = False

    planner_diagnostic = (
        planner.astype(np.float32, copy=True)
        if np.all(np.isfinite(planner))
        else np.zeros(3, dtype=np.float32)
    )
    intended_dot = float(np.dot(effective.astype(float), direction.astype(float)))
    realized_dot = float(np.dot(realized.astype(float), direction.astype(float)))
    intended_cos = _cosine(effective, direction)
    realized_cos = _cosine(realized, direction)
    alignment_tol = 1e-8
    return InfoRayActionComposition(
        policy_gain=gain,
        inner_pid_exc=inner.copy(),
        planner_action=planner_diagnostic,
        information_direction=direction.copy(),
        certificate_ok=certificate_ok,
        certificate_failure_mask=certificate_mask,
        certificate_failure_reasons=certificate_reasons,
        authority_eligible=authority_eligible,
        track_gate=track_gate,
        information_gate=information_gate,
        uncertainty_need=uncertainty_need,
        formation_error_ratio=error_ratio,
        effective_action=effective.copy(),
        action_pre_clip=pre_clip.copy(),
        action_applied=applied.copy(),
        realized_delta=realized.copy(),
        clipped_channels=clipped.copy(),
        intended_alignment_dot=intended_dot,
        realized_alignment_dot=realized_dot,
        intended_alignment_cosine=intended_cos,
        realized_alignment_cosine=realized_cos,
        intended_alignment_violation=bool(intended_dot < -alignment_tol),
        realized_alignment_violation=bool(realized_dot < -alignment_tol),
        neutral_branch=neutral_branch,
    )


class UUVTwoLeader3DPFEnv(_v12.UUVTwoLeader3DPFEnv):
    """V11 plant with scalar learned action and direct 3-D baselines."""

    BASE_OBS_DIM = _v11.UUVTwoLeader3DPFEnv.BASE_OBS_DIM

    def __init__(self, cfg: Optional[UUV3DConfig] = None, render_mode: str = "none"):
        self._init_info_ray_diagnostics()
        super().__init__(cfg=cfg or UUV3DConfig(), render_mode=render_mode)
        self.cfg: UUV3DConfig
        if self._policy_uses_info_ray():
            self.action_space = spaces.Box(
                low=np.asarray([0.0], dtype=np.float32),
                high=np.asarray([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _policy_uses_info_ray(self) -> bool:
        return str(self.cfg.v11_controller_id) in POLICY_CONTROLLER_IDS

    def _init_info_ray_diagnostics(self) -> None:
        self._v13_policy_gain = 0.0
        self._v13_inner_pid_exc = np.zeros(3, dtype=np.float32)
        self._v13_planner_action = np.zeros(3, dtype=np.float32)
        self._v13_information_direction = np.zeros(3, dtype=np.float32)
        self._v13_effective_action = np.zeros(3, dtype=np.float32)
        self._v13_action_pre_clip = np.zeros(3, dtype=np.float32)
        self._v13_action_applied = np.zeros(3, dtype=np.float32)
        self._v13_realized_delta = np.zeros(3, dtype=np.float32)
        self._v13_clipped_channels = np.zeros(3, dtype=np.float32)
        self._v13_certificate_ok = False
        self._v13_certificate_failure_mask = 0
        self._v13_certificate_failure_reasons: Tuple[str, ...] = ()
        self._v13_authority_eligible = False
        self._v13_track_gate = 0.0
        self._v13_information_gate = 0.0
        self._v13_uncertainty_need = 0.0
        self._v13_formation_error_ratio = 0.0
        self._v13_intended_alignment_dot = 0.0
        self._v13_realized_alignment_dot = 0.0
        self._v13_intended_alignment_cosine = 0.0
        self._v13_realized_alignment_cosine = 0.0
        self._v13_intended_alignment_violation = False
        self._v13_realized_alignment_violation = False
        self._v13_intended_alignment_violation_count = 0
        self._v13_realized_alignment_violation_count = 0
        self._v13_neutral_branch = True
        self._v13_info_ray_active = False

    def _reset_info_ray_diagnostics(self) -> None:
        self._init_info_ray_diagnostics()
        self._v13_info_ray_active = self._policy_uses_info_ray()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._reset_info_ray_diagnostics()
        return super().reset(seed=seed, options=options)

    def _current_online_formation_ratio(self, info: Mapping[str, Any]) -> float:
        try:
            snapshot = online_pid_exc_snapshot(info)
            p_hat = np.asarray([snapshot[f"pFhat_{axis}"] for axis in "xyz"], dtype=float)
            p_des = np.asarray([snapshot[f"pFdes_{axis}"] for axis in "xyz"], dtype=float)
            return float(np.linalg.norm(p_des - p_hat) / float(snapshot["tol_pos_est"]))
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)
        # V13 inherits v12 only to reuse its online snapshot aliases.  Its old
        # residual composer is deliberately bypassed, so do not advertise it
        # as active in v13 traces.
        info["v12_hybrid_active"] = 0.0
        planner = np.asarray(getattr(self, "_info_plan_action", np.zeros(3)), dtype=float)
        if planner.shape != (3,):
            planner = np.full(3, float("nan"), dtype=float)
        aliases = {
            "v13_info_plan_a_speed_online": float(planner[0]),
            "v13_info_plan_a_yaw_online": float(planner[1]),
            "v13_info_plan_a_pitch_online": float(planner[2]),
            "v13_info_plan_active_online": float(getattr(self, "_info_plan_active", 0.0)),
            "v13_info_plan_gate_online": float(getattr(self, "_info_plan_gate", 0.0)),
            "v13_info_plan_score_best_online": float(getattr(self, "_info_plan_score_best", 0.0)),
            "v13_info_plan_score_zero_online": float(getattr(self, "_info_plan_score_zero", 0.0)),
            "v13_info_plan_trace_red_online": float(getattr(self, "_info_plan_trace_red", 0.0)),
            "v13_info_plan_pred_err_online": float(getattr(self, "_info_plan_pred_err", float("inf"))),
        }
        info.update(aliases)
        info.update(
            {
                "v13_version": 13.0,
                "v13_info_ray_active": float(self._v13_info_ray_active),
                "v13_policy_gain": float(self._v13_policy_gain),
                "v13_certificate_ok": float(self._v13_certificate_ok),
                "v13_certificate_failure_mask": int(self._v13_certificate_failure_mask),
                "v13_certificate_failure_reasons": ",".join(self._v13_certificate_failure_reasons),
                "v13_authority_eligible": float(self._v13_authority_eligible),
                "v13_track_gate": float(self._v13_track_gate),
                "v13_information_gate": float(self._v13_information_gate),
                "v13_uncertainty_need": float(self._v13_uncertainty_need),
                "v13_formation_error_ratio": float(self._v13_formation_error_ratio),
                "v13_intended_alignment_dot": float(self._v13_intended_alignment_dot),
                "v13_realized_alignment_dot": float(self._v13_realized_alignment_dot),
                "v13_intended_alignment_cosine": float(self._v13_intended_alignment_cosine),
                "v13_realized_alignment_cosine": float(self._v13_realized_alignment_cosine),
                "v13_intended_alignment_violation": float(self._v13_intended_alignment_violation),
                "v13_realized_alignment_violation": float(self._v13_realized_alignment_violation),
                "v13_intended_alignment_violation_count": int(self._v13_intended_alignment_violation_count),
                "v13_realized_alignment_violation_count": int(self._v13_realized_alignment_violation_count),
                "v13_neutral_branch": float(self._v13_neutral_branch),
                "v13_action_clipped_count": float(np.sum(self._v13_clipped_channels)),
            }
        )
        for prefix, values in (
            ("v13_inner_pid_exc", self._v13_inner_pid_exc),
            ("v13_planner_action", self._v13_planner_action),
            ("v13_information_direction", self._v13_information_direction),
            ("v13_effective", self._v13_effective_action),
            ("v13_action_pre_clip", self._v13_action_pre_clip),
            ("v13_action_applied", self._v13_action_applied),
            ("v13_realized_delta", self._v13_realized_delta),
            ("v13_action_clipped", self._v13_clipped_channels),
        ):
            for axis, value in zip(("speed", "yaw", "pitch"), values):
                info[f"{prefix}_{axis}"] = float(value)
        return info

    def _record_composition(self, composition: InfoRayActionComposition) -> None:
        self._v13_policy_gain = float(composition.policy_gain)
        self._v13_inner_pid_exc = composition.inner_pid_exc.copy()
        self._v13_planner_action = composition.planner_action.copy()
        self._v13_information_direction = composition.information_direction.copy()
        self._v13_effective_action = composition.effective_action.copy()
        self._v13_action_pre_clip = composition.action_pre_clip.copy()
        self._v13_action_applied = composition.action_applied.copy()
        self._v13_realized_delta = composition.realized_delta.copy()
        self._v13_clipped_channels = composition.clipped_channels.copy()
        self._v13_certificate_ok = bool(composition.certificate_ok)
        self._v13_certificate_failure_mask = int(composition.certificate_failure_mask)
        self._v13_certificate_failure_reasons = composition.certificate_failure_reasons
        self._v13_authority_eligible = bool(composition.authority_eligible)
        self._v13_track_gate = float(composition.track_gate)
        self._v13_information_gate = float(composition.information_gate)
        self._v13_uncertainty_need = float(composition.uncertainty_need)
        self._v13_formation_error_ratio = float(composition.formation_error_ratio)
        self._v13_intended_alignment_dot = float(composition.intended_alignment_dot)
        self._v13_realized_alignment_dot = float(composition.realized_alignment_dot)
        self._v13_intended_alignment_cosine = float(composition.intended_alignment_cosine)
        self._v13_realized_alignment_cosine = float(composition.realized_alignment_cosine)
        self._v13_intended_alignment_violation = bool(composition.intended_alignment_violation)
        self._v13_realized_alignment_violation = bool(composition.realized_alignment_violation)
        self._v13_intended_alignment_violation_count += int(composition.intended_alignment_violation)
        self._v13_realized_alignment_violation_count += int(composition.realized_alignment_violation)
        self._v13_neutral_branch = bool(composition.neutral_branch)

    def step(self, action: Sequence[float]):
        if self._policy_uses_info_ray():
            composition = compose_info_ray_action(
                self._get_info(), self.cfg, action, self.step_count
            )
            self._v13_info_ray_active = True
            self._record_composition(composition)
            applied = composition.action_applied
        else:
            external = np.asarray(action, dtype=np.float32).reshape(-1)
            if external.shape != (3,) or not np.all(np.isfinite(external)):
                raise ValueError("direct baseline action must contain exactly three finite values")
            if np.any(external < -1.0) or np.any(external > 1.0):
                raise ValueError("direct baseline action must be in [-1, 1]^3")
            current_info = self._get_info()
            self._v13_info_ray_active = False
            self._v13_policy_gain = 0.0
            self._v13_inner_pid_exc[:] = 0.0
            self._v13_planner_action[:] = 0.0
            self._v13_information_direction[:] = 0.0
            self._v13_effective_action[:] = 0.0
            self._v13_action_pre_clip = external.copy()
            self._v13_action_applied = external.copy()
            self._v13_realized_delta[:] = 0.0
            self._v13_clipped_channels[:] = 0.0
            self._v13_certificate_ok = False
            self._v13_certificate_failure_mask = 0
            self._v13_certificate_failure_reasons = ()
            self._v13_authority_eligible = False
            self._v13_track_gate = 0.0
            self._v13_information_gate = 0.0
            self._v13_uncertainty_need = 0.0
            self._v13_formation_error_ratio = self._current_online_formation_ratio(current_info)
            self._v13_intended_alignment_dot = 0.0
            self._v13_realized_alignment_dot = 0.0
            self._v13_intended_alignment_cosine = 0.0
            self._v13_realized_alignment_cosine = 0.0
            self._v13_intended_alignment_violation = False
            self._v13_realized_alignment_violation = False
            self._v13_neutral_branch = True
            applied = external

        # Deliberately bypass v12.step: otherwise the old 3-D residual composer
        # would wrap this already-composed plant action a second time.
        return _v11.UUVTwoLeader3DPFEnv.step(self, applied)


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


def build_parser() -> argparse.ArgumentParser:
    parser = _v12.build_parser()
    replacements = {
        "models_3d_v12_hybrid": "models_3d_v13_info_ray",
        "logs_3d_v12_hybrid": "logs_3d_v13_info_ray",
        "tb_3d_v12_hybrid": "tb_3d_v13_info_ray",
        "eval_3d_logs_v12_hybrid": "eval_3d_logs_v13_info_ray",
        "info_maps_v12_hybrid": "info_maps_v13_info_ray",
    }
    for action in _v11._iter_parser_actions(parser):
        default = getattr(action, "default", None)
        if isinstance(default, str):
            for old, new in replacements.items():
                if old in default:
                    action.default = default.replace(old, new)
                    break
    train_parser = _v11._v10._get_subparser(parser, "train")
    if train_parser is not None and not _v11._v10._parser_has_dest(train_parser, "v13_variant"):
        train_parser.add_argument(
            "--v13-variant",
            choices=(V13_VARIANT,),
            default=V13_VARIANT,
            help="Frozen v13 certified information-ray controller architecture.",
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
    }
    v10.UUV3DConfig = UUV3DConfig
    v10.make_env = make_env
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


SOURCE_NAMES: Tuple[str, ...] = (
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
    curriculum_frac = float(np.clip(float(getattr(args, "curriculum_frac", 0.90)), 0.0, 1.0))
    curriculum_steps = int(max(1, round(curriculum_frac * total_timesteps / n_envs)))
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
        pf_num_particles=int(getattr(args, "pf_particles", UUV3DConfig.pf_num_particles)),
        obs_history_len=int(getattr(args, "obs_history_len", UUV3DConfig.obs_history_len)),
        info_gate_floor_hard=float(getattr(args, "info_gate_floor_hard", UUV3DConfig.info_gate_floor_hard)),
        info_gate_floor_hard_extra=float(getattr(args, "info_gate_floor_hard_extra", UUV3DConfig.info_gate_floor_hard_extra)),
        info_gate_floor_start_difficulty=float(getattr(args, "info_gate_floor_start_difficulty", UUV3DConfig.info_gate_floor_start_difficulty)),
        info_gate_floor_ramp_difficulty=float(getattr(args, "info_gate_floor_ramp_difficulty", UUV3DConfig.info_gate_floor_ramp_difficulty)),
        v11_variant="full_online",
        v12_variant=_v12.V12_VARIANT,
        v13_variant=str(getattr(args, "v13_variant", V13_VARIANT)),
    )


def cmd_train(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resume", False)):
        raise NotImplementedError("scientific resume is disabled")
    if str(getattr(args, "v11_variant", "full_online")) != "full_online":
        raise ValueError("v13 requires --v11-variant full_online")
    if str(getattr(args, "v12_variant", _v12.V12_VARIANT)) != _v12.V12_VARIANT:
        raise ValueError(f"v13 requires --v12-variant {_v12.V12_VARIANT}")
    if str(getattr(args, "v13_variant", V13_VARIANT)) != V13_VARIANT:
        raise ValueError(f"v13 requires --v13-variant {V13_VARIANT}")
    if str(getattr(args, "success_mode", "progress")) != "progress":
        raise ValueError("v13 uses the frozen v11 online progress success definition")
    if not math.isclose(float(getattr(args, "action_dt", 2.0)), 2.0, abs_tol=1e-12):
        raise ValueError("the v13 training protocol requires --action-dt 2.0 s")

    models_dir = Path(str(args.models_dir)).expanduser().resolve()
    manifest_path = models_dir / "v13_run_manifest.json"
    if models_dir.is_dir() and any(models_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty v13 model directory: {models_dir}")

    source_dir = Path(__file__).resolve().parent
    snapshot_dir = models_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing required v13 source artifact: {source}")
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    cfg_preview = _training_config_preview(args)
    manifest: Dict[str, Any] = {
        "schema_version": 3,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "version": VERSION,
        "variant": V13_VARIANT,
        "seed": int(getattr(args, "seed", 42)),
        "resume": False,
        "command": [sys.executable] + list(sys.argv),
        "arguments": vars(args),
        "environment_config": asdict(cfg_preview),
        "action_contract": cfg_preview.action_contract(),
        "observation_dim": int(cfg_preview.obs_history_len) * int(UUVTwoLeader3DPFEnv.BASE_OBS_DIM),
        "policy_action_dim": 1,
        "plant_action_dim": 3,
        "packages": _v11._package_versions(),
        "git_commit": _v11._git_commit(),
        "source_sha256": {name: _v11._sha256_file(source_dir / name) for name in SOURCE_NAMES},
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
    raise RuntimeError("run `python3 uuv_v13_evaluate.py --help` for v13 evaluation")


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
    "CONTROLLER_ARCHITECTURE",
    "InfoRayActionComposition",
    "SOURCE_NAMES",
    "UUV3DConfig",
    "UUVTwoLeader3DPFEnv",
    "V13_VARIANT",
    "VERSION",
    "build_parser",
    "compose_info_ray_action",
    "make_env",
    "online_pid_exc_snapshot",
]
