#!/usr/bin/env python3
"""v15: contextual certified information-ray gain over PID+EXC.

V15 preserves the v14 plant, inner controller, ray, and fail-closed authority
boundary.  It adds an online-only opportunity grid and aligns the
counterfactual FIM certificate with the future 30 s half-open reporting
window.  No simulator truth or future exogenous noise is policy-facing.
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
import uuv_v14_info_ray as _v14


VERSION = "v15_contextual_gain_1.0"
V15_VARIANT = "contextual_info_ray_gain_v15"
CONTROLLER_ARCHITECTURE = (
    "pid_track_exc_plus_window_certified_contextual_information_ray_gain"
)
MANIFEST_FILENAME = "v15_run_manifest.json"
MANIFEST_SCHEMA_VERSION = 5

V15_GRID_GAINS: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
V15_GRID_SUFFIXES: Tuple[str, ...] = ("g025", "g050", "g075", "g100")
V15_CONTEXT_DIM = 9
V15_GRID_FEATURE_DIM = 3 * len(V15_GRID_GAINS)
V15_EXTRA_OBS_DIM = V15_CONTEXT_DIM + V15_GRID_FEATURE_DIM

V15_FIM_EIG_REFERENCE = 0.30
V15_INFO_SCALE = 5e-5
V15_INFO_WEIGHT = 0.75
V15_FORMATION_WEIGHT = 0.40
V15_AUTHORITY_WEIGHT = 0.20
V15_PLANNER_HORIZON_S = 4.0
V15_FIM_WINDOW_S = 30.0

TB_V15_NUMERIC_ALLOWLIST = frozenset(
    (
        "v15_counterfactual_certificate_ok",
        "v15_counterfactual_window_delta_eig",
        "v15_counterfactual_bonus",
        "v15_counterfactual_reject_mask",
    )
)


@dataclass
class UUV3DConfig(_v14.UUV3DConfig):
    """Frozen v15 observation, window predictor, and reward contract."""

    v15_variant: str = V15_VARIANT
    v15_controller_architecture: str = CONTROLLER_ARCHITECTURE
    v15_fim_eig_reference: float = V15_FIM_EIG_REFERENCE
    v15_info_scale: float = V15_INFO_SCALE
    v15_info_weight: float = V15_INFO_WEIGHT
    v15_formation_weight: float = V15_FORMATION_WEIGHT
    v15_authority_weight: float = V15_AUTHORITY_WEIGHT
    v15_predictor_backend: str = "auto"
    # V10 overrides V8 with the inherited four-step, one-second model.  V15
    # makes that planner and its new window certificate agree exactly.
    info_planner_horizon_s: float = V15_PLANNER_HORIZON_S

    def __post_init__(self) -> None:
        super().__post_init__()
        if str(self.v15_variant).lower().strip() != V15_VARIANT:
            raise ValueError(f"unknown v15 variant {self.v15_variant!r}")
        self.v15_variant = V15_VARIANT
        if str(self.v15_controller_architecture) != CONTROLLER_ARCHITECTURE:
            raise ValueError("v15 controller architecture is frozen")
        frozen = (
            (self.v15_fim_eig_reference, V15_FIM_EIG_REFERENCE),
            (self.v15_info_scale, V15_INFO_SCALE),
            (self.v15_info_weight, V15_INFO_WEIGHT),
            (self.v15_formation_weight, V15_FORMATION_WEIGHT),
            (self.v15_authority_weight, V15_AUTHORITY_WEIGHT),
            (self.info_planner_horizon_s, V15_PLANNER_HORIZON_S),
            (self.info_planner_dt, 1.0),
            (self.fim_window_s, V15_FIM_WINDOW_S),
        )
        if any(
            not math.isclose(float(got), want, rel_tol=0.0, abs_tol=0.0)
            for got, want in frozen
        ):
            raise ValueError("v15 predictor, reward, and FIM window are frozen")
        backend = str(self.v15_predictor_backend).lower().strip()
        if backend not in {"auto", "python", "numba"}:
            raise ValueError("v15 predictor backend must be auto, python, or numba")
        self.v15_predictor_backend = backend

    def action_contract(self) -> Dict[str, Any]:
        contract = dict(super().action_contract())
        contract.update(
            {
                "architecture": CONTROLLER_ARCHITECTURE,
                "variant": V15_VARIANT,
                "counterfactual": (
                    "online retained half-open 30 s FIM at t+4 s plus predicted "
                    "four-second increments; actual continuous composite versus exact inner"
                ),
                "opportunity_grid_gains": list(V15_GRID_GAINS),
                "predictor_horizon_s": V15_PLANNER_HORIZON_S,
                "fim_window_s": V15_FIM_WINDOW_S,
                "retained_window_cutoff": "t + H - T",
                "retained_interval": "(t+H-T, t]",
                "observation_base_dim": 88,
                "observation_context_dim": V15_CONTEXT_DIM,
                "observation_grid_dim": V15_GRID_FEATURE_DIM,
                "reward_semantics": (
                    "inherited online reward plus once-only accepted tradeoff: "
                    "0.75*tanh(max(window_delta_eig,0)/5e-5) "
                    "-0.40*positive_predicted_formation_degradation^2 "
                    "-0.20*realized_authority^2"
                ),
                "neutral_branch": "exact inherited PID+EXC action and reward",
            }
        )
        return contract


@dataclass(frozen=True)
class V15PredictionBatch:
    actions: np.ndarray
    fim_increment: np.ndarray
    retained_fim: np.ndarray
    post_window_eigmin: np.ndarray
    pred_err: np.ndarray
    backend: str


@dataclass(frozen=True)
class V15RewardTradeoff:
    info_benefit: float
    formation_cost: float
    authority_cost: float
    total: float


@dataclass(frozen=True)
class V15OpportunityGrid:
    gains: np.ndarray
    inner_action: np.ndarray
    candidate_actions: np.ndarray
    inner_post_window_eigmin: float
    post_window_eigmin: np.ndarray
    delta_eig: np.ndarray
    inner_pred_err: float
    candidate_pred_err: np.ndarray
    pred_err_delta: np.ndarray
    safe: np.ndarray
    reject_mask: np.ndarray
    utility: np.ndarray
    tol_pos: float
    basic_certificate_ok: bool
    track_gate: float
    information_gate: float
    backend: str


@dataclass(frozen=True)
class V15ActionComposition:
    policy_gain: float
    basic: _v13.InfoRayActionComposition
    candidate_preclip: np.ndarray
    counterfactual_inner_eig: float
    counterfactual_candidate_eig: float
    counterfactual_window_delta_eig: float
    counterfactual_inner_pred_err: float
    counterfactual_candidate_pred_err: float
    counterfactual_pred_err_delta: float
    counterfactual_backend: str
    counterfactual_certificate_ok: bool
    counterfactual_reject_mask: int
    counterfactual_reject_reasons: Tuple[str, ...]
    action_applied: np.ndarray
    accepted_delta: np.ndarray
    realized_authority: float
    tradeoff: V15RewardTradeoff


def score_v15_tradeoff(
    delta_eig: float,
    pred_err_delta: float,
    tol_pos: float,
    realized_authority: float,
    cfg: UUV3DConfig,
) -> V15RewardTradeoff:
    """Bounded online counterfactual benefit minus formation/authority costs."""

    delta = float(delta_eig)
    pred_delta = float(pred_err_delta)
    tol = max(float(tol_pos), 1e-12)
    authority = float(np.clip(realized_authority, 0.0, 1.0))
    info = float(cfg.v15_info_weight) * math.tanh(
        max(delta, 0.0) / float(cfg.v15_info_scale)
    )
    form_ratio = float(
        np.clip(max(pred_delta, 0.0) / (0.25 * tol), 0.0, 1.0)
    )
    formation_cost = float(cfg.v15_formation_weight) * form_ratio * form_ratio
    authority_cost = float(cfg.v15_authority_weight) * authority * authority
    return V15RewardTradeoff(
        info_benefit=info,
        formation_cost=formation_cost,
        authority_cost=authority_cost,
        total=float(info - formation_cost - authority_cost),
    )


def v15_retained_window_cutoff(env: "UUVTwoLeader3DPFEnv") -> float:
    return (
        float(env.t)
        + float(env.cfg.info_planner_horizon_s)
        - float(env.cfg.fim_window_s)
    )


def _retained_online_fim(env: "UUVTwoLeader3DPFEnv") -> np.ndarray:
    """Online FIM events remaining in ``(t+H-T, t]`` before prediction."""

    tracker = env.fim_hat_win
    window = float(env.cfg.fim_window_s)
    horizon = float(env.cfg.info_planner_horizon_s)
    if window <= 0.0:
        return np.asarray(tracker.I_total, dtype=float).reshape(3, 3).copy()
    cutoff = v15_retained_window_cutoff(env)
    retained = np.zeros((3, 3), dtype=np.float64)
    for timestamp, increment in tuple(tracker._events):
        # HalfOpenFIMTracker3D removes events <= left boundary + 1e-12.
        if float(timestamp) > cutoff + 1e-12:
            retained += np.asarray(increment, dtype=np.float64).reshape(3, 3)
    return 0.5 * (retained + retained.T)


@_v8.njit(cache=True)
def _v15_predict_increment_numba(
    actions: np.ndarray,
    support_points: np.ndarray,
    support_weights: np.ndarray,
    p_l1_0: np.ndarray,
    p_l2_0: np.ndarray,
    v_l1: np.ndarray,
    v_l2: np.ndarray,
    speed0: float,
    yaw0: float,
    pitch0: float,
    difficulty: float,
    gate_relax_active: float,
    action_dt: float,
    rl_speed_delta_per_step: float,
    max_yaw_rate_deg_s: float,
    rl_yaw_per_step_deg: float,
    max_pitch_rate_deg_s: float,
    rl_pitch_per_step_deg: float,
    f_min_speed: float,
    f_max_speed: float,
    pitch_min_deg: float,
    pitch_max_deg: float,
    dt: float,
    horizon: float,
    doppler_rho_min: float,
    doppler_v_min: float,
    doppler_v_perp_min_easy: float,
    doppler_v_perp_min_hard: float,
    gate_band_rho: float,
    gate_band_v: float,
    gate_band_v_perp: float,
    gate_min_factor: float,
    gate_relax_v_perp_min_hard: float,
    gate_relax_band_v_perp: float,
    meas_sigma: float,
    sigma_nis_mult: float,
    d_back: float,
    d_right: float,
    dz_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_actions = actions.shape[0]
    fim = np.zeros((n_actions, 3, 3), dtype=np.float64)
    pred_err = np.zeros(n_actions, dtype=np.float64)
    dt_eff = max(dt, 1e-3)
    n_steps = max(1, int(round(max(horizon, dt_eff) / dt_eff)))
    sigma2 = max(meas_sigma * meas_sigma * sigma_nis_mult * sigma_nis_mult, 1e-18)

    for ia in range(n_actions):
        a0, a1, a2 = actions[ia, 0], actions[ia, 1], actions[ia, 2]
        speed_cmd = a0 * rl_speed_delta_per_step / max(action_dt, 1e-6)
        yaw_cmd = a1 * min(max_yaw_rate_deg_s, rl_yaw_per_step_deg / max(action_dt, 1e-6))
        pitch_cmd = a2 * min(max_pitch_rate_deg_s, rl_pitch_per_step_deg / max(action_dt, 1e-6))
        pts = support_points.copy()
        mean_x = 0.0
        mean_y = 0.0
        mean_z = 0.0
        for k in range(pts.shape[0]):
            mean_x += pts[k, 0] * support_weights[k]
            mean_y += pts[k, 1] * support_weights[k]
            mean_z += pts[k, 2] * support_weights[k]
        p1x, p1y, p1z = p_l1_0[0], p_l1_0[1], p_l1_0[2]
        p2x, p2y, p2z = p_l2_0[0], p_l2_0[1], p_l2_0[2]
        speed, yaw, pitch = speed0, yaw0, pitch0

        for _step in range(n_steps):
            speed = _v10._v10_fast_clamp(speed + speed_cmd * dt_eff, f_min_speed, f_max_speed)
            yaw = (yaw + yaw_cmd * dt_eff) % 360.0
            pitch = _v10._v10_fast_clamp(pitch + pitch_cmd * dt_eff, pitch_min_deg, pitch_max_deg)
            vf0, vf1, vf2 = _v10._v10_fast_vel(speed, yaw, pitch)
            dx, dy, dz = vf0 * dt_eff, vf1 * dt_eff, vf2 * dt_eff
            for k in range(pts.shape[0]):
                pts[k, 0] += dx
                pts[k, 1] += dy
                pts[k, 2] += dz
            mean_x += dx
            mean_y += dy
            mean_z += dz
            p1x += v_l1[0] * dt_eff
            p1y += v_l1[1] * dt_eff
            p1z += v_l1[2] * dt_eff
            p2x += v_l2[0] * dt_eff
            p2y += v_l2[1] * dt_eff
            p2z += v_l2[2] * dt_eff

            for leader in range(2):
                if leader == 0:
                    plx, ply, plz = p1x, p1y, p1z
                    vr0, vr1, vr2 = v_l1[0] - vf0, v_l1[1] - vf1, v_l1[2] - vf2
                else:
                    plx, ply, plz = p2x, p2y, p2z
                    vr0, vr1, vr2 = v_l2[0] - vf0, v_l2[1] - vf1, v_l2[2] - vf2
                for k in range(pts.shape[0]):
                    rx, ry, rz = plx - pts[k, 0], ply - pts[k, 1], plz - pts[k, 2]
                    gate = _v10._v10_fast_gate_factor(
                        rx, ry, rz, vr0, vr1, vr2, difficulty,
                        doppler_rho_min, doppler_v_min,
                        doppler_v_perp_min_easy, doppler_v_perp_min_hard,
                        gate_band_rho, gate_band_v, gate_band_v_perp,
                        gate_min_factor, gate_relax_active,
                        gate_relax_v_perp_min_hard, gate_relax_band_v_perp,
                    )
                    if gate <= 0.0:
                        continue
                    rho = _v10._v10_fast_norm3(rx, ry, rz)
                    inv_rho = 1.0 / max(rho, 1e-9)
                    ux, uy, uz = rx * inv_rho, ry * inv_rho, rz * inv_rho
                    proj = ux * vr0 + uy * vr1 + uz * vr2
                    hx = -(vr0 - proj * ux) * inv_rho
                    hy = -(vr1 - proj * uy) * inv_rho
                    hz = -(vr2 - proj * uz) * inv_rho
                    w = support_weights[k] * gate / sigma2
                    fim[ia, 0, 0] += w * hx * hx
                    fim[ia, 0, 1] += w * hx * hy
                    fim[ia, 0, 2] += w * hx * hz
                    fim[ia, 1, 1] += w * hy * hy
                    fim[ia, 1, 2] += w * hy * hz
                    fim[ia, 2, 2] += w * hz * hz

        fim[ia, 1, 0] = fim[ia, 0, 1]
        fim[ia, 2, 0] = fim[ia, 0, 2]
        fim[ia, 2, 1] = fim[ia, 1, 2]
        pcx, pcy, pcz = 0.5 * (p1x + p2x), 0.5 * (p1y + p2y), 0.5 * (p1z + p2z)
        vcx, vcy = 0.5 * (v_l1[0] + v_l2[0]), 0.5 * (v_l1[1] + v_l2[1])
        nf = math.sqrt(vcx * vcx + vcy * vcy)
        if nf <= 1e-12:
            fx, fy = 1.0, 0.0
        else:
            fx, fy = vcx / nf, vcy / nf
        pdesx = pcx - d_back * fx + d_right * fy
        pdesy = pcy - d_back * fy - d_right * fx
        pdesz = pcz + dz_offset
        pred_err[ia] = _v10._v10_fast_norm3(
            mean_x - pdesx, mean_y - pdesy, mean_z - pdesz
        )
    return fim, pred_err


def _prediction_inputs(env: "UUVTwoLeader3DPFEnv") -> Tuple[np.ndarray, np.ndarray]:
    points, weights, _p0 = env._info_support_points()
    return (
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
    )


def _predict_python_from_inputs(
    env: "UUVTwoLeader3DPFEnv",
    actions: np.ndarray,
    support_points: np.ndarray,
    support_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = env.cfg
    fim = np.zeros((actions.shape[0], 3, 3), dtype=np.float64)
    pred_err = np.zeros(actions.shape[0], dtype=np.float64)
    dt = max(float(cfg.info_planner_dt), 1e-3)
    n_steps = max(1, int(round(max(float(cfg.info_planner_horizon_s), dt) / dt)))
    sigma2 = max(
        float(env.pf.meas_sigma * getattr(env.pf, "sigma_nis_mult", 1.0)) ** 2,
        1e-18,
    )
    for ia, action in enumerate(actions):
        action = np.asarray(action, dtype=float).reshape(3)
        speed_cmd = action[0] * float(cfg.rl_speed_delta_per_step) / max(float(cfg.action_dt), 1e-6)
        yaw_cmd = action[1] * min(float(cfg.max_yaw_rate_deg_s), float(cfg.rl_yaw_per_step_deg) / max(float(cfg.action_dt), 1e-6))
        pitch_cmd = action[2] * min(float(cfg.max_pitch_rate_deg_s), float(cfg.rl_pitch_per_step_deg) / max(float(cfg.action_dt), 1e-6))
        pts = support_points.copy()
        mean = np.sum(pts * support_weights[:, None], axis=0)
        p_l1 = np.asarray(env.pL1, dtype=float).copy()
        p_l2 = np.asarray(env.pL2, dtype=float).copy()
        speed, yaw, pitch = float(env.speed_F), float(env.yaw_F), float(env.pitch_F)
        for _ in range(n_steps):
            speed = float(np.clip(speed + speed_cmd * dt, cfg.f_min_speed, cfg.f_max_speed))
            yaw = _v8.wrap360(yaw + yaw_cmd * dt)
            pitch = float(np.clip(pitch + pitch_cmd * dt, cfg.pitch_min_deg, cfg.pitch_max_deg))
            v_f = _v8.vel_from_speed_yaw_pitch(speed, yaw, pitch)
            displacement = v_f * dt
            pts += displacement[None, :]
            mean += displacement
            p_l1 += np.asarray(env.vL1, dtype=float) * dt
            p_l2 += np.asarray(env.vL2, dtype=float) * dt
            for p_l, v_l in ((p_l1, env.vL1), (p_l2, env.vL2)):
                v_rel = np.asarray(v_l, dtype=float) - v_f
                for point, weight in zip(pts, support_weights):
                    gate = float(env._info_gate_factor_single(p_l - point, v_rel))
                    if gate <= 0.0:
                        continue
                    h = _v8.doppler_H_3d(p_l - point, v_rel)
                    fim[ia] += float(weight * gate) * (h.T @ h) / sigma2
        p_c = 0.5 * (p_l1 + p_l2)
        v_c = 0.5 * (np.asarray(env.vL1, dtype=float) + np.asarray(env.vL2, dtype=float))
        f_hat = _v8.normalize_vec(
            np.asarray((v_c[0], v_c[1], 0.0), dtype=float),
            fallback=np.asarray((1.0, 0.0, 0.0), dtype=float),
        )
        side = _v8.normalize_vec(
            np.cross(f_hat, np.asarray((0.0, 0.0, 1.0), dtype=float)),
            fallback=np.asarray((0.0, -1.0, 0.0), dtype=float),
        )
        desired = p_c - float(cfg.d_back) * f_hat + float(cfg.d_right) * side
        desired[2] = p_c[2] + float(cfg.dz_offset)
        pred_err[ia] = float(np.linalg.norm(mean - desired))
    return fim, pred_err


def _prediction_batch(
    actions: np.ndarray,
    fim_increment: np.ndarray,
    pred_err: np.ndarray,
    retained: np.ndarray,
    backend: str,
) -> V15PredictionBatch:
    post_eig = np.asarray(
        [
            _v8.FIMTracker3D.eig_stats(retained + fim_increment[i])[0]
            for i in range(actions.shape[0])
        ],
        dtype=np.float64,
    )
    return V15PredictionBatch(
        actions=np.asarray(actions, dtype=np.float64).copy(),
        fim_increment=np.asarray(fim_increment, dtype=np.float64).copy(),
        retained_fim=np.asarray(retained, dtype=np.float64).copy(),
        post_window_eigmin=post_eig,
        pred_err=np.asarray(pred_err, dtype=np.float64).copy(),
        backend=str(backend),
    )


@contextmanager
def _measured_follower_shadow(env: "UUVTwoLeader3DPFEnv"):
    truth = (
        float(env.speed_F),
        float(env.yaw_F),
        float(env.pitch_F),
        np.asarray(env.vF, dtype=float).copy(),
    )
    env.speed_F = float(env._v11_speed_meas)
    env.yaw_F = float(env._v11_yaw_meas)
    env.pitch_F = float(env._v11_pitch_meas)
    env.vF = np.asarray(env._v11_vf_meas, dtype=float).copy()
    try:
        yield
    finally:
        env.speed_F, env.yaw_F, env.pitch_F = truth[:3]
        env.vF = truth[3]


def _validate_actions(actions: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 1:
        raise ValueError("v15 prediction actions must have shape (N, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("v15 prediction actions must be finite")
    return np.ascontiguousarray(values)


def predict_v15_candidates_python(
    env: "UUVTwoLeader3DPFEnv", actions: Sequence[Sequence[float]]
) -> V15PredictionBatch:
    values = _validate_actions(actions)
    retained = _retained_online_fim(env)
    with _measured_follower_shadow(env):
        points, weights = _prediction_inputs(env)
        fim, pred_err = _predict_python_from_inputs(env, values, points, weights)
    return _prediction_batch(values, fim, pred_err, retained, "python")


def predict_v15_candidates_numba(
    env: "UUVTwoLeader3DPFEnv", actions: Sequence[Sequence[float]]
) -> V15PredictionBatch:
    if not bool(getattr(_v8, "_NUMBA_AVAILABLE", False)):
        raise RuntimeError("Numba v15 predictor requested but Numba is unavailable")
    values = _validate_actions(actions)
    cfg = env.cfg
    retained = _retained_online_fim(env)
    with _measured_follower_shadow(env):
        points, weights = _prediction_inputs(env)
        fim, pred_err = _v15_predict_increment_numba(
            values,
            points,
            weights,
            np.ascontiguousarray(env.pL1, dtype=np.float64),
            np.ascontiguousarray(env.pL2, dtype=np.float64),
            np.ascontiguousarray(env.vL1, dtype=np.float64),
            np.ascontiguousarray(env.vL2, dtype=np.float64),
            float(env.speed_F),
            float(env.yaw_F),
            float(env.pitch_F),
            float(env._difficulty),
            float(env._gate_relax_active),
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
        )
    return _prediction_batch(values, fim, pred_err, retained, "numba")


def predict_v15_candidates(
    env: "UUVTwoLeader3DPFEnv",
    actions: Sequence[Sequence[float]],
    backend: str = "auto",
) -> V15PredictionBatch:
    selected = str(backend).lower().strip()
    if selected == "auto":
        selected = "numba" if bool(getattr(_v8, "_NUMBA_AVAILABLE", False)) else "python"
    if selected == "python":
        return predict_v15_candidates_python(env, actions)
    if selected == "numba":
        return predict_v15_candidates_numba(env, actions)
    raise ValueError("v15 predictor backend must be auto, python, or numba")


def _v15_prerequisite_mask(basic: _v13.InfoRayActionComposition) -> int:
    inner = np.asarray(basic.inner_pid_exc, dtype=np.float32)
    candidate = np.asarray(basic.action_pre_clip, dtype=np.float32)
    mask = 0
    if not bool(basic.certificate_ok):
        mask |= _v14.CF_REJECT_BASIC_V13_CERTIFICATE
    if not bool(basic.authority_eligible) or float(basic.policy_gain) <= 0.0:
        mask |= _v14.CF_REJECT_ZERO_OR_NO_AUTHORITY
    if not (
        np.all(np.isfinite(inner))
        and np.all(np.isfinite(candidate))
        and np.all(np.isfinite(np.asarray(basic.effective_action, dtype=float)))
    ):
        mask |= _v14.CF_REJECT_NONFINITE
    if np.any(candidate < -1.0) or np.any(candidate > 1.0):
        mask |= _v14.CF_REJECT_ACTION_BOUNDS
    clipped = np.clip(candidate, -1.0, 1.0).astype(np.float32)
    if not np.array_equal(clipped, candidate) or np.any(basic.clipped_channels > 0.5):
        mask |= _v14.CF_REJECT_WOULD_CLIP
    return int(mask)


def _v15_metric_mask(
    mask: int,
    inner_eig: float,
    candidate_eig: float,
    inner_pred_err: float,
    candidate_pred_err: float,
    tol_pos: float,
    cfg: UUV3DConfig,
) -> int:
    values = np.asarray(
        (inner_eig, candidate_eig, inner_pred_err, candidate_pred_err),
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        return int(mask | _v14.CF_REJECT_NONFINITE)
    delta = float(candidate_eig - inner_eig)
    pred_delta = float(candidate_pred_err - inner_pred_err)
    if delta <= float(cfg.v14_counterfactual_delta_eig_min):
        mask |= _v14.CF_REJECT_DELTA_EIG
    if float(candidate_pred_err) > float(tol_pos):
        mask |= _v14.CF_REJECT_CANDIDATE_PRED_ERR
    if pred_delta > float(cfg.v14_counterfactual_pred_err_delta_max_ratio) * float(tol_pos):
        mask |= _v14.CF_REJECT_PRED_ERR_DEGRADATION
    return int(mask)


def _zero_tradeoff() -> V15RewardTradeoff:
    return V15RewardTradeoff(0.0, 0.0, 0.0, 0.0)


def evaluate_v15_opportunity_grid(
    env: "UUVTwoLeader3DPFEnv",
    online_info: Optional[Mapping[str, Any]] = None,
    step_count: Optional[int] = None,
    backend: str = "auto",
) -> V15OpportunityGrid:
    """Pure online-only fixed-gain opportunity probe for the current state."""

    if online_info is None:
        online_info = _v14.UUVTwoLeader3DPFEnv._get_info(env)
    step = int(env.step_count if step_count is None else step_count)
    cfg = env.cfg
    basics = tuple(
        _v13.compose_info_ray_action(online_info, cfg, [gain], step)
        for gain in V15_GRID_GAINS
    )
    snapshot = _v13.online_pid_exc_snapshot(online_info)
    tol_pos = float(snapshot["tol_pos_est"])
    inner = np.asarray(basics[0].inner_pid_exc, dtype=np.float32).copy()
    candidates = np.asarray(
        [np.asarray(item.action_pre_clip, dtype=np.float32) for item in basics],
        dtype=np.float32,
    )
    actions = np.vstack((inner[None, :], candidates)).astype(np.float64)
    selected = str(backend).lower().strip()
    if selected == "auto":
        selected = str(cfg.v15_predictor_backend)
    masks = np.asarray([_v15_prerequisite_mask(item) for item in basics], dtype=np.int64)
    try:
        prediction = predict_v15_candidates(env, actions, backend=selected)
    except Exception:
        masks |= np.int64(_v14.CF_REJECT_EVALUATION_FAILED)
        inner_eig = float("nan")
        eig = np.full(len(V15_GRID_GAINS), float("nan"), dtype=float)
        inner_pred = float("nan")
        candidate_pred = np.full(len(V15_GRID_GAINS), float("nan"), dtype=float)
        used_backend = "failed"
    else:
        inner_eig = float(prediction.post_window_eigmin[0])
        eig = np.asarray(prediction.post_window_eigmin[1:], dtype=float)
        inner_pred = float(prediction.pred_err[0])
        candidate_pred = np.asarray(prediction.pred_err[1:], dtype=float)
        used_backend = str(prediction.backend)
        for idx in range(len(V15_GRID_GAINS)):
            masks[idx] = _v15_metric_mask(
                int(masks[idx]),
                inner_eig,
                float(eig[idx]),
                inner_pred,
                float(candidate_pred[idx]),
                tol_pos,
                cfg,
            )

    delta = eig - inner_eig
    pred_delta = candidate_pred - inner_pred
    safe = masks == 0
    utilities = np.zeros(len(V15_GRID_GAINS), dtype=float)
    track_gate = float(basics[-1].track_gate)
    information_gate = float(basics[-1].information_gate)
    for idx, gain in enumerate(V15_GRID_GAINS):
        if safe[idx]:
            utilities[idx] = score_v15_tradeoff(
                float(delta[idx]),
                float(pred_delta[idx]),
                tol_pos,
                float(gain) * track_gate * information_gate,
                cfg,
            ).total
    return V15OpportunityGrid(
        gains=np.asarray(V15_GRID_GAINS, dtype=np.float32),
        inner_action=inner,
        candidate_actions=candidates.copy(),
        inner_post_window_eigmin=inner_eig,
        post_window_eigmin=eig.copy(),
        delta_eig=delta.copy(),
        inner_pred_err=inner_pred,
        candidate_pred_err=candidate_pred.copy(),
        pred_err_delta=pred_delta.copy(),
        safe=safe.copy(),
        reject_mask=masks.copy(),
        utility=utilities,
        tol_pos=tol_pos,
        basic_certificate_ok=bool(basics[-1].certificate_ok),
        track_gate=track_gate,
        information_gate=information_gate,
        backend=used_backend,
    )


def build_v15_probe_grid(
    env: "UUVTwoLeader3DPFEnv",
    online_info: Optional[Mapping[str, Any]] = None,
    step_count: Optional[int] = None,
    backend: str = "auto",
) -> V15OpportunityGrid:
    """Public evaluator API; alias with an explicitly probe-oriented name."""

    return evaluate_v15_opportunity_grid(
        env,
        online_info=online_info,
        step_count=step_count,
        backend=backend,
    )


def compose_v15_action(
    env: "UUVTwoLeader3DPFEnv",
    online_info: Mapping[str, Any],
    policy_gain: Sequence[float],
    step_count: int,
    *,
    backend: str = "auto",
) -> V15ActionComposition:
    """Compose and certify the actual continuous actor command pre-step."""

    cfg = env.cfg
    basic = _v13.compose_info_ray_action(
        online_info, cfg, policy_gain, int(step_count)
    )
    gain = float(basic.policy_gain)
    inner = np.asarray(basic.inner_pid_exc, dtype=np.float32).copy()
    candidate = np.asarray(basic.action_pre_clip, dtype=np.float32).copy()
    mask = _v15_prerequisite_mask(basic)
    inner_eig = candidate_eig = float("nan")
    inner_pred = candidate_pred = float("nan")
    used_backend = "none"
    prerequisite = (
        _v14.CF_REJECT_BASIC_V13_CERTIFICATE
        | _v14.CF_REJECT_ZERO_OR_NO_AUTHORITY
        | _v14.CF_REJECT_NONFINITE
        | _v14.CF_REJECT_ACTION_BOUNDS
        | _v14.CF_REJECT_WOULD_CLIP
    )
    if (mask & prerequisite) == 0:
        selected = str(backend).lower().strip()
        if selected == "auto":
            selected = str(cfg.v15_predictor_backend)
        try:
            prediction = predict_v15_candidates(
                env, np.vstack((inner, candidate)), backend=selected
            )
        except Exception:
            mask |= _v14.CF_REJECT_EVALUATION_FAILED
        else:
            inner_eig = float(prediction.post_window_eigmin[0])
            candidate_eig = float(prediction.post_window_eigmin[1])
            inner_pred = float(prediction.pred_err[0])
            candidate_pred = float(prediction.pred_err[1])
            used_backend = str(prediction.backend)
            snapshot = _v13.online_pid_exc_snapshot(online_info)
            mask = _v15_metric_mask(
                mask,
                inner_eig,
                candidate_eig,
                inner_pred,
                candidate_pred,
                float(snapshot["tol_pos_est"]),
                cfg,
            )
    accepted = mask == 0
    delta = float(candidate_eig - inner_eig)
    pred_delta = float(candidate_pred - inner_pred)
    if accepted:
        applied = candidate.copy()
        accepted_delta = (candidate - inner).astype(np.float32)
        authority = float(
            np.clip(gain * float(basic.track_gate) * float(basic.information_gate), 0.0, 1.0)
        )
        tol_pos = float(_v13.online_pid_exc_snapshot(online_info)["tol_pos_est"])
        tradeoff = score_v15_tradeoff(delta, pred_delta, tol_pos, authority, cfg)
    else:
        applied = inner.copy()
        accepted_delta = np.zeros(3, dtype=np.float32)
        authority = 0.0
        tradeoff = _zero_tradeoff()
    reasons = tuple(
        name for bit, name in _v14._CF_REASON_BY_BIT if int(mask) & int(bit)
    )
    return V15ActionComposition(
        policy_gain=gain,
        basic=basic,
        candidate_preclip=candidate,
        counterfactual_inner_eig=inner_eig,
        counterfactual_candidate_eig=candidate_eig,
        counterfactual_window_delta_eig=delta,
        counterfactual_inner_pred_err=inner_pred,
        counterfactual_candidate_pred_err=candidate_pred,
        counterfactual_pred_err_delta=pred_delta,
        counterfactual_backend=used_backend,
        counterfactual_certificate_ok=accepted,
        counterfactual_reject_mask=int(mask),
        counterfactual_reject_reasons=reasons,
        action_applied=applied,
        accepted_delta=accepted_delta,
        realized_authority=authority,
        tradeoff=tradeoff,
    )


def _scaled_fim_eigenvalues(env: "UUVTwoLeader3DPFEnv") -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(env.fim_hat_win.I_win, dtype=float).reshape(3, 3)
    try:
        eig = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        eig = np.zeros(3, dtype=float)
    eig = np.maximum(np.where(np.isfinite(eig), eig, 0.0), 0.0)
    scaled = np.clip(
        np.log10((eig + 1e-12) / V15_FIM_EIG_REFERENCE), -6.0, 2.0
    ) / 6.0
    return eig.astype(np.float64), scaled.astype(np.float32)


class UUVTwoLeader3DPFEnv(_v14.UUVTwoLeader3DPFEnv):
    """V14 plant with contextual window-aware information authority."""

    BASE_OBS_DIM = _v14.UUVTwoLeader3DPFEnv.BASE_OBS_DIM + V15_EXTRA_OBS_DIM

    def _init_info_ray_diagnostics(self) -> None:
        super()._init_info_ray_diagnostics()
        self._init_v15_diagnostics()

    def _reset_info_ray_diagnostics(self) -> None:
        super()._reset_info_ray_diagnostics()
        self._init_v15_diagnostics()

    def _init_v15_diagnostics(self) -> None:
        self._v15_previous_requested_gain = 0.0
        self._v15_previous_accepted = False
        self._v15_grid: Optional[V15OpportunityGrid] = None
        self._v15_grid_episode = -1
        self._v15_grid_step = -1
        self._v15_context_fim_eig = np.zeros(3, dtype=float)
        self._v15_counterfactual_certificate_ok = False
        self._v15_counterfactual_window_delta_eig = float("nan")
        self._v15_counterfactual_inner_eig = float("nan")
        self._v15_counterfactual_candidate_eig = float("nan")
        self._v15_counterfactual_inner_pred_err = float("nan")
        self._v15_counterfactual_candidate_pred_err = float("nan")
        self._v15_counterfactual_pred_err_delta = float("nan")
        self._v15_counterfactual_reject_mask = 0
        self._v15_counterfactual_reject_reasons: Tuple[str, ...] = ()
        self._v15_counterfactual_backend = "none"
        self._v15_counterfactual_realized_authority = 0.0
        self._v15_counterfactual_info_benefit = 0.0
        self._v15_counterfactual_formation_cost = 0.0
        self._v15_counterfactual_authority_cost = 0.0
        self._v15_counterfactual_bonus = 0.0
        self._v15_counterfactual_bonus_pending = False
        self._v15_counterfactual_evaluated_step = -1
        self._v15_counterfactual_rewarded_step = -1

    def __init__(self, cfg: Optional[UUV3DConfig] = None, render_mode: str = "none"):
        # V13 invokes the virtual diagnostic initializer during construction.
        super().__init__(cfg=cfg or UUV3DConfig(), render_mode=render_mode)
        self.cfg: UUV3DConfig

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # The inherited reset calls the virtual diagnostic reset before it
        # constructs the first contextual observation.
        return super().reset(seed=seed, options=options)

    def _get_obs_base(self) -> np.ndarray:
        inherited = super()._get_obs_base().astype(np.float32)
        if inherited.shape != (_v14.UUVTwoLeader3DPFEnv.BASE_OBS_DIM,):
            raise RuntimeError("unexpected inherited v14 observation width")
        online_info = _v14.UUVTwoLeader3DPFEnv._get_info(self)
        grid = evaluate_v15_opportunity_grid(
            self,
            online_info=online_info,
            step_count=int(self.step_count),
            backend=str(self.cfg.v15_predictor_backend),
        )
        self._v15_grid = grid
        self._v15_grid_episode = int(getattr(self, "_v11_episode_index", -1))
        self._v15_grid_step = int(self.step_count)
        eig_raw, eig_scaled = _scaled_fim_eigenvalues(self)
        self._v15_context_fim_eig = eig_raw.copy()
        headroom = 0.0
        if np.isfinite(grid.inner_pred_err) and np.isfinite(grid.tol_pos) and grid.tol_pos > 0.0:
            headroom = float(
                np.clip((grid.tol_pos - grid.inner_pred_err) / grid.tol_pos, -1.0, 1.0)
            )
        context = np.asarray(
            [
                eig_scaled[0],
                eig_scaled[1],
                eig_scaled[2],
                headroom,
                float(grid.basic_certificate_ok),
                float(np.clip(grid.track_gate, 0.0, 1.0)),
                float(np.clip(grid.information_gate, 0.0, 1.0)),
                float(np.clip(self._v15_previous_requested_gain, 0.0, 1.0)),
                float(self._v15_previous_accepted),
            ],
            dtype=np.float32,
        )
        grid_features = []
        for delta, pred_delta, safe in zip(
            grid.delta_eig, grid.pred_err_delta, grid.safe
        ):
            info_feature = (
                math.tanh(float(delta) / V15_INFO_SCALE)
                if np.isfinite(delta)
                else 0.0
            )
            formation_feature = (
                math.tanh(float(pred_delta) / (0.25 * max(grid.tol_pos, 1e-12)))
                if np.isfinite(pred_delta)
                else 0.0
            )
            grid_features.extend((info_feature, formation_feature, float(safe)))
        extra = np.concatenate(
            (context, np.asarray(grid_features, dtype=np.float32)), axis=0
        )
        if extra.shape != (V15_EXTRA_OBS_DIM,) or not np.all(np.isfinite(extra)):
            raise RuntimeError("invalid v15 contextual observation")
        return np.concatenate((inherited, extra), axis=0).astype(np.float32)

    def _record_v15_composition(self, composition: V15ActionComposition) -> None:
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
        _v13.UUVTwoLeader3DPFEnv._record_composition(self, reported_basic)
        # Keep inherited v14 diagnostics collision-free but do not arm its old
        # future-increment-only reward cache.
        self._v14_counterfactual_certificate_ok = bool(
            composition.counterfactual_certificate_ok
        )
        self._v14_counterfactual_delta_eig = float(
            composition.counterfactual_window_delta_eig
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
        self._v14_counterfactual_accepted_delta = (
            composition.accepted_delta.copy()
        )
        self._v14_counterfactual_bonus = 0.0
        self._v14_counterfactual_bonus_pending = False

        self._v15_previous_requested_gain = float(composition.policy_gain)
        self._v15_previous_accepted = bool(
            composition.counterfactual_certificate_ok
        )
        self._v15_counterfactual_certificate_ok = bool(
            composition.counterfactual_certificate_ok
        )
        self._v15_counterfactual_window_delta_eig = float(
            composition.counterfactual_window_delta_eig
        )
        self._v15_counterfactual_inner_eig = float(
            composition.counterfactual_inner_eig
        )
        self._v15_counterfactual_candidate_eig = float(
            composition.counterfactual_candidate_eig
        )
        self._v15_counterfactual_inner_pred_err = float(
            composition.counterfactual_inner_pred_err
        )
        self._v15_counterfactual_candidate_pred_err = float(
            composition.counterfactual_candidate_pred_err
        )
        self._v15_counterfactual_pred_err_delta = float(
            composition.counterfactual_pred_err_delta
        )
        self._v15_counterfactual_reject_mask = int(
            composition.counterfactual_reject_mask
        )
        self._v15_counterfactual_reject_reasons = (
            composition.counterfactual_reject_reasons
        )
        self._v15_counterfactual_backend = str(composition.counterfactual_backend)
        self._v15_counterfactual_realized_authority = float(
            composition.realized_authority
        )
        self._v15_counterfactual_info_benefit = float(
            composition.tradeoff.info_benefit
        )
        self._v15_counterfactual_formation_cost = float(
            composition.tradeoff.formation_cost
        )
        self._v15_counterfactual_authority_cost = float(
            composition.tradeoff.authority_cost
        )
        self._v15_counterfactual_bonus = 0.0
        self._v15_counterfactual_bonus_pending = bool(
            composition.counterfactual_certificate_ok
        )
        self._v15_counterfactual_evaluated_step = int(self.step_count)
        self._v15_counterfactual_rewarded_step = -1

    def _compute_reward(
        self,
        pf_stats_last: Optional[_v8.PFStats],
        planner_action_for_reward: Optional[np.ndarray] = None,
        planner_gate_for_reward: Optional[float] = None,
        planner_margin_for_reward: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        # Explicitly bypass v14's obsolete future-increment-only bonus.
        base_reward, terms = _v11.UUVTwoLeader3DPFEnv._compute_reward(
            self,
            pf_stats_last,
            planner_action_for_reward,
            planner_gate_for_reward,
            planner_margin_for_reward,
        )
        bonus = 0.0
        if bool(self._v15_counterfactual_bonus_pending):
            if int(self._v15_counterfactual_evaluated_step) != int(self.step_count) - 1:
                raise RuntimeError("stale v15 pre-step reward cache")
            if not bool(self._v15_counterfactual_certificate_ok):
                raise RuntimeError("uncertified v15 tradeoff reached reward cache")
            bonus = float(
                self._v15_counterfactual_info_benefit
                - self._v15_counterfactual_formation_cost
                - self._v15_counterfactual_authority_cost
            )
            self._v15_counterfactual_bonus_pending = False
            self._v15_counterfactual_rewarded_step = int(self.step_count)
            self._v15_counterfactual_bonus = bonus
        terms.update(
            {
                "v15_version": 15.0,
                "v15_reward_uses_truth": 0.0,
                "v15_counterfactual_certificate_ok": float(
                    self._v15_counterfactual_certificate_ok
                ),
                "v15_counterfactual_window_delta_eig": float(
                    self._v15_counterfactual_window_delta_eig
                ),
                "v15_counterfactual_pred_err_delta": float(
                    self._v15_counterfactual_pred_err_delta
                ),
                "v15_counterfactual_info_benefit": float(
                    self._v15_counterfactual_info_benefit
                ),
                "v15_counterfactual_formation_cost": float(
                    self._v15_counterfactual_formation_cost
                ),
                "v15_counterfactual_authority_cost": float(
                    self._v15_counterfactual_authority_cost
                ),
                "v15_counterfactual_bonus": float(bonus),
                "v15_counterfactual_reject_mask": float(
                    self._v15_counterfactual_reject_mask
                ),
            }
        )
        if bonus == 0.0:
            return float(base_reward), terms
        return float(
            np.clip(
                float(base_reward) + bonus,
                -float(self.cfg.rew_clip),
                float(self.cfg.rew_clip),
            )
        ), terms

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)
        info.update(
            {
                "v15_version": 15.0,
                "v15_context_fim_eig_0": float(self._v15_context_fim_eig[0]),
                "v15_context_fim_eig_1": float(self._v15_context_fim_eig[1]),
                "v15_context_fim_eig_2": float(self._v15_context_fim_eig[2]),
                "v15_previous_requested_gain": float(
                    self._v15_previous_requested_gain
                ),
                "v15_previous_accepted": float(self._v15_previous_accepted),
                "v15_counterfactual_certificate_ok": float(
                    self._v15_counterfactual_certificate_ok
                ),
                "v15_counterfactual_window_delta_eig": float(
                    self._v15_counterfactual_window_delta_eig
                ),
                "v15_counterfactual_pred_err_delta": float(
                    self._v15_counterfactual_pred_err_delta
                ),
                "v15_counterfactual_bonus": float(
                    self._v15_counterfactual_bonus
                ),
                "v15_counterfactual_info_benefit": float(
                    self._v15_counterfactual_info_benefit
                ),
                "v15_counterfactual_formation_cost": float(
                    self._v15_counterfactual_formation_cost
                ),
                "v15_counterfactual_authority_cost": float(
                    self._v15_counterfactual_authority_cost
                ),
                "v15_counterfactual_realized_authority": float(
                    self._v15_counterfactual_realized_authority
                ),
                "v15_counterfactual_reject_mask": int(
                    self._v15_counterfactual_reject_mask
                ),
                "v15_counterfactual_reject_reasons": ",".join(
                    self._v15_counterfactual_reject_reasons
                ),
                "v15_counterfactual_backend": str(
                    self._v15_counterfactual_backend
                ),
                "v15_counterfactual_inner_eig": float(
                    self._v15_counterfactual_inner_eig
                ),
                "v15_counterfactual_candidate_eig": float(
                    self._v15_counterfactual_candidate_eig
                ),
                "v15_counterfactual_inner_pred_err": float(
                    self._v15_counterfactual_inner_pred_err
                ),
                "v15_counterfactual_candidate_pred_err": float(
                    self._v15_counterfactual_candidate_pred_err
                ),
                "v15_counterfactual_evaluated_step": int(
                    self._v15_counterfactual_evaluated_step
                ),
                "v15_counterfactual_rewarded_step": int(
                    self._v15_counterfactual_rewarded_step
                ),
                "v15_counterfactual_bonus_pending": float(
                    self._v15_counterfactual_bonus_pending
                ),
                "v15_predictor_horizon_s": float(
                    self.cfg.info_planner_horizon_s
                ),
                "v15_fim_window_s": float(self.cfg.fim_window_s),
                "v15_retained_window_cutoff": float(
                    v15_retained_window_cutoff(self)
                ),
            }
        )
        grid = self._v15_grid
        if grid is not None:
            headroom = (
                float(np.clip((grid.tol_pos - grid.inner_pred_err) / grid.tol_pos, -1.0, 1.0))
                if np.isfinite(grid.inner_pred_err) and grid.tol_pos > 0.0
                else 0.0
            )
            info.update(
                {
                    "v15_context_inner_headroom": headroom,
                    "v15_context_basic_certificate_ok": float(
                        grid.basic_certificate_ok
                    ),
                    "v15_context_track_gate": float(grid.track_gate),
                    "v15_context_information_gate": float(grid.information_gate),
                    "v15_grid_episode": int(self._v15_grid_episode),
                    "v15_grid_step": int(self._v15_grid_step),
                }
            )
            for idx, suffix in enumerate(V15_GRID_SUFFIXES):
                info.update(
                    {
                        f"v15_grid_{suffix}_gain": float(grid.gains[idx]),
                        f"v15_grid_{suffix}_delta_eig": float(grid.delta_eig[idx]),
                        f"v15_grid_{suffix}_pred_err_delta": float(grid.pred_err_delta[idx]),
                        f"v15_grid_{suffix}_safe": float(grid.safe[idx]),
                        f"v15_grid_{suffix}_reject_mask": int(grid.reject_mask[idx]),
                        f"v15_grid_{suffix}_candidate_pred_err": float(grid.candidate_pred_err[idx]),
                        f"v15_grid_{suffix}_post_window_eigmin": float(grid.post_window_eigmin[idx]),
                        f"v15_grid_{suffix}_utility": float(grid.utility[idx]),
                    }
                )
        return info

    def step(self, action: Sequence[float]):
        if self._policy_uses_info_ray():
            online_info = self._get_info()
            composition = compose_v15_action(
                self,
                online_info,
                action,
                int(self.step_count),
                backend=str(self.cfg.v15_predictor_backend),
            )
            self._v13_info_ray_active = True
            self._record_v15_composition(composition)
            return _v11.UUVTwoLeader3DPFEnv.step(
                self, composition.action_applied
            )
        # Frozen direct baselines retain v14 semantics and see no v15 bonus.
        self._v15_counterfactual_bonus_pending = False
        self._v15_previous_requested_gain = 0.0
        self._v15_previous_accepted = False
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
            if not name.startswith("v15_"):
                filtered[key] = value
            elif name in TB_V15_NUMERIC_ALLOWLIST and isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                filtered[key] = value
        copied.append(filtered)
    return _v14._tb_infos_copy(copied)


class V15TBInfoCallback(_v8.TBInfoCallback):
    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        callback_locals = dict(locals_)
        callback_locals["infos"] = _tb_infos_copy(locals_.get("infos"))
        return super().__call__(callback_locals, globals_)


def build_parser() -> argparse.ArgumentParser:
    parser = _v14.build_parser()
    replacements = {
        "models_3d_v14_info_ray": "models_3d_v15_contextual_gain",
        "logs_3d_v14_info_ray": "logs_3d_v15_contextual_gain",
        "tb_3d_v14_info_ray": "tb_3d_v15_contextual_gain",
        "eval_3d_logs_v14_info_ray": "eval_3d_logs_v15_contextual_gain",
        "info_maps_v14_info_ray": "info_maps_v15_contextual_gain",
    }
    for parser_action in _v11._iter_parser_actions(parser):
        default = getattr(parser_action, "default", None)
        if isinstance(default, str):
            for old, new in replacements.items():
                if old in default:
                    parser_action.default = default.replace(old, new)
                    break
    train_parser = _v10._get_subparser(parser, "train")
    if train_parser is not None and not _v10._parser_has_dest(
        train_parser, "v15_variant"
    ):
        train_parser.add_argument(
            "--v15-variant",
            choices=(V15_VARIANT,),
            default=V15_VARIANT,
            help="Frozen v15 contextual window-aware information-ray controller.",
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
        "v8_TBInfoCallback": v8.TBInfoCallback,
    }
    v10.UUV3DConfig = UUV3DConfig
    v10.make_env = make_env
    v8.TBInfoCallback = V15TBInfoCallback
    if patch_env_class:
        v10.UUVTwoLeader3DPFEnv = UUVTwoLeader3DPFEnv
    try:
        yield
    finally:
        v10.UUV3DConfig = old["UUV3DConfig"]
        v10.UUVTwoLeader3DPFEnv = old["UUVTwoLeader3DPFEnv"]
        v10.make_env = old["make_env"]
        v8.TBInfoCallback = old["v8_TBInfoCallback"]


SOURCE_NAMES: Tuple[str, ...] = (
    "uuv_v15_contextual_gain.py",
    "uuv_v15_evaluate.py",
    "run_v15_training.py",
    "EXPERIMENT_PROTOCOL_V15.md",
    "tests/test_uuv_v15_contextual_gain.py",
    "tests/test_uuv_v15_evaluate.py",
    "tests/test_run_v15_training.py",
) + tuple(_v14.SOURCE_NAMES)


def _training_config_preview(args: argparse.Namespace) -> UUV3DConfig:
    total_timesteps = int(args.total_timesteps)
    n_envs = max(1, int(args.n_envs))
    curriculum_frac = float(
        np.clip(float(getattr(args, "curriculum_frac", 0.90)), 0.0, 1.0)
    )
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
        fim_window_s=float(getattr(args, "fim_window", V15_FIM_WINDOW_S)),
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
        v14_variant=_v14.V14_VARIANT,
        v15_variant=str(getattr(args, "v15_variant", V15_VARIANT)),
    )


def cmd_train(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resume", False)):
        raise NotImplementedError("scientific resume is disabled")
    requirements = (
        ("v11_variant", "full_online"),
        ("v12_variant", _v12.V12_VARIANT),
        ("v13_variant", _v13.V13_VARIANT),
        ("v14_variant", _v14.V14_VARIANT),
        ("v15_variant", V15_VARIANT),
    )
    for name, expected in requirements:
        if str(getattr(args, name, expected)) != expected:
            raise ValueError(f"v15 requires --{name.replace('_', '-')} {expected}")
    if str(getattr(args, "success_mode", "progress")) != "progress":
        raise ValueError("v15 requires the frozen online progress success definition")
    if not math.isclose(float(getattr(args, "action_dt", 2.0)), 2.0, abs_tol=1e-12):
        raise ValueError("v15 training requires --action-dt 2.0 s")
    if not math.isclose(
        float(getattr(args, "fim_window", V15_FIM_WINDOW_S)),
        V15_FIM_WINDOW_S,
        abs_tol=0.0,
    ):
        raise ValueError("v15 training requires --fim-window 30.0 s")

    models_dir = Path(str(args.models_dir)).expanduser().resolve()
    manifest_path = models_dir / MANIFEST_FILENAME
    if models_dir.is_dir() and any(models_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty v15 model directory: {models_dir}"
        )
    source_dir = Path(__file__).resolve().parent
    snapshot_dir = models_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing required v15 source artifact: {source}")
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    cfg_preview = _training_config_preview(args)
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "version": VERSION,
        "variant": V15_VARIANT,
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
    raise RuntimeError("run `python3 uuv_v15_evaluate.py --help` for v15 evaluation")


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
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "SOURCE_NAMES",
    "TB_V15_NUMERIC_ALLOWLIST",
    "UUV3DConfig",
    "UUVTwoLeader3DPFEnv",
    "V15ActionComposition",
    "V15OpportunityGrid",
    "V15PredictionBatch",
    "V15RewardTradeoff",
    "V15TBInfoCallback",
    "V15_AUTHORITY_WEIGHT",
    "V15_FIM_WINDOW_S",
    "V15_FORMATION_WEIGHT",
    "V15_GRID_GAINS",
    "V15_INFO_SCALE",
    "V15_INFO_WEIGHT",
    "V15_PLANNER_HORIZON_S",
    "V15_VARIANT",
    "VERSION",
    "build_parser",
    "build_v15_probe_grid",
    "compose_v15_action",
    "evaluate_v15_opportunity_grid",
    "make_env",
    "predict_v15_candidates",
    "predict_v15_candidates_numba",
    "predict_v15_candidates_python",
    "score_v15_tradeoff",
    "v15_retained_window_cutoff",
]
