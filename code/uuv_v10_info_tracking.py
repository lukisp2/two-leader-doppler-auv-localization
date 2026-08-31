# -*- coding: utf-8 -*-
"""
v10 information-tracking variant built on top of uuv_v8_temporal_infofix.

Purpose of this variant:
- train fixed-horizon behaviour instead of terminal 'reach formation' behaviour,
- maximize Doppler information quality: lambda_min(FIM_win) and low tr(CRLB_win),
- reward increases of information over time, not only absolute information level,
- still penalize formation-position error,
- encourage the follower to move in the same horizontal direction as the leaders,
- expose the v8/v9 information-direction planner to the policy.

The observation extends v8/v9 planner features with 3 log-scaled tracking-ratio
features so the policy can distinguish "bad" from "catastrophically off-formation".
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join("/tmp", "uuv_matplotlib_cache")
    try:
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    except Exception:
        pass

for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np

import uuv_v8_temporal_infofix as _v8
from uuv_v8_temporal_infofix import *  # noqa: F401,F403

try:
    import uuv_v10_info_tracking_cy as _v10_cy
    _V10_CYTHON_AVAILABLE = True
except Exception:
    _v10_cy = None
    _V10_CYTHON_AVAILABLE = False


VERSION = "v10_info_tracking_guarded"
_V8Env = _v8.UUVTwoLeader3DPFEnv


@njit(cache=True)
def _v10_fast_clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True)
def _v10_fast_smoothstep01(t: float) -> float:
    t = _v10_fast_clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def _v10_fast_soft_gate(x: float, x_min: float, band: float) -> float:
    if band <= 1e-12:
        return 1.0 if x > x_min else 0.0
    return _v10_fast_smoothstep01((x - x_min) / band)


@njit(cache=True)
def _v10_fast_sat_ratio(x: float, x0: float) -> float:
    if x < 0.0:
        x = 0.0
    if x0 < 1e-12:
        x0 = 1e-12
    return x / (x + x0)


@njit(cache=True)
def _v10_fast_norm3(x0: float, x1: float, x2: float) -> float:
    return math.sqrt(x0 * x0 + x1 * x1 + x2 * x2)


@njit(cache=True)
def _v10_fast_vel(speed: float, yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
    yaw = yaw_deg * math.pi / 180.0
    pitch = pitch_deg * math.pi / 180.0
    cp = math.cos(pitch)
    return speed * cp * math.cos(yaw), speed * cp * math.sin(yaw), speed * math.sin(pitch)


@njit(cache=True)
def _v10_fast_gate_factor(
    rx: float,
    ry: float,
    rz: float,
    vx: float,
    vy: float,
    vz: float,
    difficulty: float,
    doppler_rho_min: float,
    doppler_v_min: float,
    doppler_v_perp_min_easy: float,
    doppler_v_perp_min_hard: float,
    gate_band_rho: float,
    gate_band_v: float,
    gate_band_v_perp: float,
    gate_min_factor: float,
    gate_relax_active: float,
    gate_relax_v_perp_min_hard: float,
    gate_relax_band_v_perp: float,
) -> float:
    rho = _v10_fast_norm3(rx, ry, rz)
    vnorm = _v10_fast_norm3(vx, vy, vz)
    if rho <= 1e-9 or vnorm <= 1e-9:
        return 0.0

    inv_rho = 1.0 / rho
    rhatx = rx * inv_rho
    rhaty = ry * inv_rho
    rhatz = rz * inv_rho
    proj = rhatx * vx + rhaty * vy + rhatz * vz
    vpx = vx - proj * rhatx
    vpy = vy - proj * rhaty
    vpz = vz - proj * rhatz
    v_perp = _v10_fast_norm3(vpx, vpy, vpz)

    d = _v10_fast_clamp(difficulty, 0.0, 1.0)
    v_perp_min = (1.0 - d) * doppler_v_perp_min_easy + d * doppler_v_perp_min_hard
    band_v_perp = gate_band_v_perp
    if gate_relax_active > 0.5:
        if gate_relax_v_perp_min_hard < v_perp_min:
            v_perp_min = gate_relax_v_perp_min_hard
        if gate_relax_band_v_perp > band_v_perp:
            band_v_perp = gate_relax_band_v_perp

    g = (
        _v10_fast_soft_gate(rho, doppler_rho_min, gate_band_rho)
        * _v10_fast_soft_gate(vnorm, doppler_v_min, gate_band_v)
        * _v10_fast_soft_gate(v_perp, v_perp_min, band_v_perp)
    )
    if g < gate_min_factor:
        return 0.0
    return _v10_fast_clamp(g, 0.0, 1.0)


@njit(cache=True)
def _v10_fast_eig_min_sym3(
    a00: float,
    a01: float,
    a02: float,
    a11: float,
    a12: float,
    a22: float,
) -> float:
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    if p1 <= 1e-30:
        mn = a00
        if a11 < mn:
            mn = a11
        if a22 < mn:
            mn = a22
        return mn if mn > 0.0 else 0.0

    q = (a00 + a11 + a22) / 3.0
    b00 = a00 - q
    b11 = a11 - q
    b22 = a22 - q
    p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1
    p = math.sqrt(max(p2, 0.0) / 6.0)
    if p <= 1e-30:
        return q if q > 0.0 else 0.0

    c00 = b00 / p
    c01 = a01 / p
    c02 = a02 / p
    c11 = b11 / p
    c12 = a12 / p
    c22 = b22 / p
    det_c = (
        c00 * (c11 * c22 - c12 * c12)
        - c01 * (c01 * c22 - c12 * c02)
        + c02 * (c01 * c12 - c11 * c02)
    )
    r = _v10_fast_clamp(det_c / 2.0, -1.0, 1.0)
    phi = math.acos(r) / 3.0
    eig1 = q + 2.0 * p * math.cos(phi)
    eig3 = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    eig2 = 3.0 * q - eig1 - eig3
    mn = eig1
    if eig2 < mn:
        mn = eig2
    if eig3 < mn:
        mn = eig3
    return mn if mn > 0.0 else 0.0


@njit(cache=True)
def _v10_fast_inv3_trace_reg(
    a00: float,
    a01: float,
    a02: float,
    a11: float,
    a12: float,
    a22: float,
    eps: float,
) -> float:
    a00 += eps
    a11 += eps
    a22 += eps
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c11 = a00 * a22 - a02 * a02
    c12 = a01 * a02 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02
    if abs(det) <= 1e-24:
        # Conservative fallback for near-singular matrices.
        return 1.0 / max(a00, eps) + 1.0 / max(a11, eps) + 1.0 / max(a22, eps)
    return (c00 + c11 + c22) / det


@njit(cache=True)
def _v10_fast_info_search(
    candidates: np.ndarray,
    support_points: np.ndarray,
    support_weights: np.ndarray,
    p0_eff: np.ndarray,
    p_l1_0: np.ndarray,
    p_l2_0: np.ndarray,
    v_l1: np.ndarray,
    v_l2: np.ndarray,
    speed0: float,
    yaw0: float,
    pitch0: float,
    difficulty: float,
    gate_relax_active: float,
    tol_pos_est: float,
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
    info_planner_dt: float,
    info_planner_horizon_s: float,
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
    info_planner_goal_err0: float,
    info_planner_trace_red0: float,
    info_planner_eigmin0: float,
    sens_norm: float,
    info_planner_w_trace: float,
    info_planner_w_eigmin: float,
    info_planner_w_sens: float,
    info_planner_w_goal: float,
    info_planner_w_energy: float,
) -> Tuple[int, int, float, np.ndarray]:
    n_actions = candidates.shape[0]
    metrics = np.zeros((n_actions, 9), dtype=np.float64)
    best_idx = 0
    zero_idx = 0
    best_score = -1e300
    second_best_score = -1e300
    zero_found = False

    dt = max(1e-3, info_planner_dt)
    horizon = max(dt, info_planner_horizon_s)
    n_steps = max(1, int(round(horizon / dt)))
    sigma2 = max(meas_sigma * meas_sigma * sigma_nis_mult * sigma_nis_mult, 1e-18)

    p0_00 = p0_eff[0, 0]
    p0_01 = 0.5 * (p0_eff[0, 1] + p0_eff[1, 0])
    p0_02 = 0.5 * (p0_eff[0, 2] + p0_eff[2, 0])
    p0_11 = p0_eff[1, 1]
    p0_12 = 0.5 * (p0_eff[1, 2] + p0_eff[2, 1])
    p0_22 = p0_eff[2, 2]
    p0_trace = p0_00 + p0_11 + p0_22

    for ia in range(n_actions):
        a0 = candidates[ia, 0]
        a1 = candidates[ia, 1]
        a2 = candidates[ia, 2]
        a_norm = _v10_fast_norm3(a0, a1, a2)
        if (not zero_found) and a_norm <= 1e-12:
            zero_idx = ia
            zero_found = True

        speed_cmd = a0 * (rl_speed_delta_per_step / max(action_dt, 1e-6))
        yaw_lim = min(max_yaw_rate_deg_s, rl_yaw_per_step_deg / max(action_dt, 1e-6))
        pitch_lim = min(max_pitch_rate_deg_s, rl_pitch_per_step_deg / max(action_dt, 1e-6))
        yaw_cmd = a1 * yaw_lim
        pitch_cmd = a2 * pitch_lim

        n_pts = support_points.shape[0]
        pts = support_points.copy()
        mean_x = 0.0
        mean_y = 0.0
        mean_z = 0.0
        for k in range(n_pts):
            wk = support_weights[k]
            mean_x += pts[k, 0] * wk
            mean_y += pts[k, 1] * wk
            mean_z += pts[k, 2] * wk

        p_l1x = p_l1_0[0]
        p_l1y = p_l1_0[1]
        p_l1z = p_l1_0[2]
        p_l2x = p_l2_0[0]
        p_l2y = p_l2_0[1]
        p_l2z = p_l2_0[2]
        speed = speed0
        yaw = yaw0
        pitch = pitch0
        fim00 = 0.0
        fim01 = 0.0
        fim02 = 0.0
        fim11 = 0.0
        fim12 = 0.0
        fim22 = 0.0
        gate_sum = 0.0
        gate_cnt = 0.0
        sens_sum = 0.0
        disp0 = 0.0
        disp1 = 0.0
        disp2 = 0.0

        for step_idx in range(n_steps):
            speed = _v10_fast_clamp(speed + speed_cmd * dt, f_min_speed, f_max_speed)
            yaw = (yaw + yaw_cmd * dt) % 360.0
            pitch = _v10_fast_clamp(pitch + pitch_cmd * dt, pitch_min_deg, pitch_max_deg)
            vf0, vf1, vf2 = _v10_fast_vel(speed, yaw, pitch)
            dx = vf0 * dt
            dy = vf1 * dt
            dz = vf2 * dt
            if step_idx == 0:
                disp0 = dx
                disp1 = dy
                disp2 = dz
            for k in range(n_pts):
                pts[k, 0] += dx
                pts[k, 1] += dy
                pts[k, 2] += dz
            mean_x += dx
            mean_y += dy
            mean_z += dz

            p_l1x += v_l1[0] * dt
            p_l1y += v_l1[1] * dt
            p_l1z += v_l1[2] * dt
            p_l2x += v_l2[0] * dt
            p_l2y += v_l2[1] * dt
            p_l2z += v_l2[2] * dt

            for leader_idx in range(2):
                if leader_idx == 0:
                    plx = p_l1x
                    ply = p_l1y
                    plz = p_l1z
                    vrel0 = v_l1[0] - vf0
                    vrel1 = v_l1[1] - vf1
                    vrel2 = v_l1[2] - vf2
                else:
                    plx = p_l2x
                    ply = p_l2y
                    plz = p_l2z
                    vrel0 = v_l2[0] - vf0
                    vrel1 = v_l2[1] - vf1
                    vrel2 = v_l2[2] - vf2

                for k in range(n_pts):
                    rx = plx - pts[k, 0]
                    ry = ply - pts[k, 1]
                    rz = plz - pts[k, 2]
                    g = _v10_fast_gate_factor(
                        rx,
                        ry,
                        rz,
                        vrel0,
                        vrel1,
                        vrel2,
                        difficulty,
                        doppler_rho_min,
                        doppler_v_min,
                        doppler_v_perp_min_easy,
                        doppler_v_perp_min_hard,
                        gate_band_rho,
                        gate_band_v,
                        gate_band_v_perp,
                        gate_min_factor,
                        gate_relax_active,
                        gate_relax_v_perp_min_hard,
                        gate_relax_band_v_perp,
                    )
                    if g <= 0.0:
                        continue
                    rho = _v10_fast_norm3(rx, ry, rz)
                    inv_rho = 1.0 / max(rho, 1e-9)
                    rhatx = rx * inv_rho
                    rhaty = ry * inv_rho
                    rhatz = rz * inv_rho
                    proj = rhatx * vrel0 + rhaty * vrel1 + rhatz * vrel2
                    hx = -(vrel0 - proj * rhatx) * inv_rho
                    hy = -(vrel1 - proj * rhaty) * inv_rho
                    hz = -(vrel2 - proj * rhatz) * inv_rho
                    w_info = support_weights[k] * g / sigma2
                    fim00 += w_info * hx * hx
                    fim01 += w_info * hx * hy
                    fim02 += w_info * hx * hz
                    fim11 += w_info * hy * hy
                    fim12 += w_info * hy * hz
                    fim22 += w_info * hz * hz

                    vpx = vrel0 - proj * rhatx
                    vpy = vrel1 - proj * rhaty
                    vpz = vrel2 - proj * rhatz
                    sens_sum += support_weights[k] * g * _v10_fast_norm3(vpx, vpy, vpz)
                    gate_sum += g
                    gate_cnt += 1.0

        pcx = 0.5 * (p_l1x + p_l2x)
        pcy = 0.5 * (p_l1y + p_l2y)
        pcz = 0.5 * (p_l1z + p_l2z)
        vcx = 0.5 * (v_l1[0] + v_l2[0])
        vcy = 0.5 * (v_l1[1] + v_l2[1])
        nf = math.sqrt(vcx * vcx + vcy * vcy)
        if nf <= 1e-12:
            fx = 1.0
            fy = 0.0
        else:
            fx = vcx / nf
            fy = vcy / nf
        sx = fy
        sy = -fx
        pdes_x = pcx - d_back * fx + d_right * sx
        pdes_y = pcy - d_back * fy + d_right * sy
        pdes_z = pcz + dz_offset
        pred_err = _v10_fast_norm3(mean_x - pdes_x, mean_y - pdes_y, mean_z - pdes_z)

        # Trace of inv(inv(P0) + FIM). Compute full symmetric inverse of P0 here.
        q00 = p0_00 + 1e-6
        q01 = p0_01
        q02 = p0_02
        q11 = p0_11 + 1e-6
        q12 = p0_12
        q22 = p0_22 + 1e-6
        c00 = q11 * q22 - q12 * q12
        c01 = q02 * q12 - q01 * q22
        c02 = q01 * q12 - q02 * q11
        c11 = q00 * q22 - q02 * q02
        c12 = q01 * q02 - q00 * q12
        c22 = q00 * q11 - q01 * q01
        det = q00 * c00 + q01 * c01 + q02 * c02
        if abs(det) <= 1e-24:
            pinv00 = 1.0 / max(q00, 1e-6)
            pinv01 = 0.0
            pinv02 = 0.0
            pinv11 = 1.0 / max(q11, 1e-6)
            pinv12 = 0.0
            pinv22 = 1.0 / max(q22, 1e-6)
        else:
            inv_det = 1.0 / det
            pinv00 = c00 * inv_det
            pinv01 = c01 * inv_det
            pinv02 = c02 * inv_det
            pinv11 = c11 * inv_det
            pinv12 = c12 * inv_det
            pinv22 = c22 * inv_det

        post_trace = _v10_fast_inv3_trace_reg(
            pinv00 + fim00,
            pinv01 + fim01,
            pinv02 + fim02,
            pinv11 + fim11,
            pinv12 + fim12,
            pinv22 + fim22,
            1e-6,
        )
        trace_red = p0_trace - post_trace
        if trace_red < 0.0:
            trace_red = 0.0
        eigmin = _v10_fast_eig_min_sym3(fim00, fim01, fim02, fim11, fim12, fim22)
        sens_avg = sens_sum / max(gate_cnt, 1.0)
        gate_avg = gate_sum / max(gate_cnt, 1.0)
        goal_scale = max(info_planner_goal_err0, 1.5 * tol_pos_est, 1e-6)
        trace_feat = _v10_fast_sat_ratio(trace_red, max(info_planner_trace_red0, 1e-6))
        eig_feat = _v10_fast_sat_ratio(math.log1p(max(eigmin, 0.0)), math.log1p(max(info_planner_eigmin0, 1e-6)))
        sens_feat = _v10_fast_sat_ratio(sens_avg, max(sens_norm, 1e-6))
        goal_pen = _v10_fast_sat_ratio(pred_err, goal_scale)
        energy_pen = _v10_fast_sat_ratio(a_norm, 1.0)
        score = (
            info_planner_w_trace * trace_feat
            + info_planner_w_eigmin * eig_feat
            + info_planner_w_sens * sens_feat
            + 0.10 * gate_avg
            - info_planner_w_goal * goal_pen
            - info_planner_w_energy * energy_pen
        )

        metrics[ia, 0] = score
        metrics[ia, 1] = trace_red
        metrics[ia, 2] = eigmin
        metrics[ia, 3] = sens_avg
        metrics[ia, 4] = pred_err
        metrics[ia, 5] = gate_avg
        metrics[ia, 6] = disp0
        metrics[ia, 7] = disp1
        metrics[ia, 8] = disp2

        if score > best_score:
            if best_score > second_best_score:
                second_best_score = best_score
            best_score = score
            best_idx = ia
        elif score > second_best_score:
            second_best_score = score

    return int(best_idx), int(zero_idx), float(second_best_score), metrics


@dataclass
class UUV3DConfig(_v8.UUV3DConfig):
    # ---------------------------------------------------------------------
    # Main semantic change versus v8/v9:
    # This variant is not a reach-and-stop task. Episodes run to max_steps.
    # Success is kept only as a diagnostic, not as a termination condition.
    # ---------------------------------------------------------------------
    terminal_bonus: float = 0.0

    # Keep v8/v9 information directions available to the policy. The horizon is
    # shorter than v9/v10_temporal for speed, but the planner is not zeroed.
    info_planner_enabled: bool = True
    info_planner_horizon_s: float = 4.0
    info_planner_dt: float = 1.0
    info_planner_support_axes: int = 2
    info_planner_mean_weight: float = 0.40
    info_planner_start_difficulty: float = 0.25
    info_planner_ramp_difficulty: float = 0.25
    info_planner_trace_red0: float = 45.0
    info_planner_eigmin0: float = 12.0
    info_planner_goal_err0: float = 60.0
    info_planner_w_trace: float = 1.35
    info_planner_w_eigmin: float = 1.85
    info_planner_w_sens: float = 0.45
    info_planner_w_goal: float = 0.18
    info_planner_w_energy: float = 0.04
    info_planner_margin_gain: float = 2.5
    info_planner_gate_min: float = 0.01
    info_planner_align_reward: float = 0.0
    v10_info_planner_numba: bool = True
    v10_info_planner_backend: str = "auto"

    # Planner directions should remain visible in the information-tracking task
    # even when the filter is already confident, otherwise the new task loses
    # exactly the guidance signal it is meant to learn from.
    v10_info_dir_start_difficulty: float = 0.20
    v10_info_dir_ramp_difficulty: float = 0.30
    v10_info_dir_gate_floor: float = 0.35

    # Do not force the old v8 information gate floor. v10 information reward
    # is always active through true FIM/CRLB terms, so an additional gate floor
    # is unnecessary and can distort behaviour.
    info_gate_floor_hard: float = 0.0
    info_gate_floor_hard_extra: float = 0.0
    info_gate_floor_start_difficulty: float = 0.75
    info_gate_floor_ramp_difficulty: float = 0.20

    # Targets calibrated around the PID/PID+Exc reference reported for v8/v9.
    # FIM is maximized; CRLB trace is minimized.
    v10_target_fim_min: float = 1.80
    v10_target_crlb_trace: float = 0.75
    v10_bad_crlb_trace: float = 3.0
    v10_crlb_trace_cap: float = 25.0

    # Position / tracking reference scales.
    v10_pos_err0_m: float = 60.0
    v10_pos_gain0_m: float = 4.0
    v10_speed_err0: float = 1.0
    v10_std0_m: float = 20.0

    # Measurement / sensitivity reference scales.
    v10_target_meas_used_frac: float = 0.90
    v10_target_sens_avg: float = 3.0

    # Delta shaping scales. These make information improvement valuable while
    # keeping the reward bounded and stable.
    v10_fim_delta0: float = 0.08
    v10_crlb_delta0: float = 0.08

    # Tracking guard: information is still rewarded, but only fully when the
    # follower remains in a controllable formation corridor.
    v10_track_guard_soft_ratio: float = 2.0
    v10_track_guard_hard_ratio: float = 5.0
    v10_info_guard_floor: float = 0.15
    v10_track_guard_ratio0: float = 1.5
    v10_track_near_ratio: float = 2.5
    v10_est_guard_ratio0: float = 4.0
    v10_speed_excess_ratio: float = 1.25
    v10_speed_excess_ratio0: float = 0.50
    v10_obs_ratio_cap: float = 25.0

    # Reward weights. Information remains important, but formation/stability
    # now dominates when the policy leaves the success corridor.
    v10_w_fim_abs: float = 1.40
    v10_w_crlb_abs: float = 1.25
    v10_w_fim_gain: float = 0.55
    v10_w_crlb_gain: float = 0.55
    v10_w_meas: float = 0.45
    v10_w_sens: float = 0.35
    v10_w_pos_est: float = 2.40
    v10_w_pos_true: float = 1.80
    v10_w_pos_gain: float = 0.75
    v10_w_track_guard: float = 3.00
    v10_w_est_guard: float = 0.85
    v10_w_track_near: float = 1.20
    v10_w_success_step: float = 2.00
    v10_w_heading: float = 1.20
    v10_w_speed_match: float = 0.80
    v10_w_speed_excess: float = 0.90
    v10_w_info_dir_align: float = 0.35
    v10_w_std: float = 1.20
    # Do not penalize steering effort in this experiment: excitation maneuvers
    # are often exactly what improves Doppler observability.
    v10_w_energy: float = 0.0
    v10_w_time: float = 0.0



class UUVTwoLeader3DPFEnv(_V8Env):
    """v10 env: fixed-horizon information tracking with leader-direction following."""

    BASE_OBS_DIM = _V8Env.BASE_OBS_DIM + 3

    def _info_activation_gate(self, std_max_eff: float, tol_std: float) -> float:
        base_gate = float(super()._info_activation_gate(std_max_eff, tol_std))
        cfg = self.cfg
        d = float(_v8.clamp(self._difficulty, 0.0, 1.0))
        t = float(
            _v8.smoothstep01(
                (d - float(cfg.v10_info_dir_start_difficulty))
                / max(float(cfg.v10_info_dir_ramp_difficulty), 1e-6)
            )
        )
        floor = float(cfg.v10_info_dir_gate_floor) * t
        return float(_v8.clamp(max(base_gate, floor), 0.0, 1.0))

    def _get_obs_base(self) -> np.ndarray:
        obs = super()._get_obs_base().astype(np.float32)
        cfg = self.cfg
        _, pF_des = self._formation_desired()
        tol_pos_est, tol_std, tol_pos_true = self._current_tolerances()
        err_est = float(np.linalg.norm(np.asarray(self.pf.mean, dtype=float).reshape(3) - pF_des))
        err_true = float(np.linalg.norm(np.asarray(self.pF, dtype=float).reshape(3) - pF_des))
        std_max = float(self.std_max_eff_step) if np.isfinite(self.std_max_eff_step) else float(self.pf.std_max())

        cap = max(float(cfg.v10_obs_ratio_cap), 1.0)
        denom = math.log1p(cap)

        def _log_ratio_feature(x: float) -> float:
            if not np.isfinite(x):
                x = cap
            x = float(_v8.clamp(x, 0.0, cap))
            return float(math.log1p(x) / denom)

        obs_extra = np.array(
            [
                _log_ratio_feature(err_est / max(float(tol_pos_est), 1e-6)),
                _log_ratio_feature(err_true / max(float(tol_pos_true), 1e-6)),
                _log_ratio_feature(std_max / max(float(tol_std), 1e-6)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs, obs_extra], axis=0).astype(np.float32)

    # ------------------------- reset / diagnostics -------------------------

    def reset(self, *args, **kwargs):
        self._v10_prev_log_fim = 0.0
        self._v10_prev_log_crlb = math.log1p(float(getattr(self.cfg, "v10_crlb_trace_cap", 25.0)))
        self._v10_prev_err_est = 0.0
        self._v10_ever_progress_success = False
        self._v10_ever_true_success = False

        out = super().reset(*args, **kwargs)

        # Initialize deltas from the reset state so that the first reward is not
        # dominated by arbitrary previous-episode values.
        try:
            m = self._v10_window_info_metrics()
            self._v10_prev_log_fim = math.log1p(max(float(m["fim_win_eig_min"]), 0.0))
            crlb = min(max(float(m["crlb_win_trace"]), 0.0), float(self.cfg.v10_crlb_trace_cap))
            self._v10_prev_log_crlb = math.log1p(crlb)
            _, pF_des = self._formation_desired()
            self._v10_prev_err_est = float(np.linalg.norm(self.pf.mean - pF_des))
        except Exception:
            pass

        return out

    # -------------------------- reward utilities ---------------------------

    def _v10_window_info_metrics(self) -> Dict[str, float]:
        cfg = self.cfg

        I_win = np.asarray(self.fim_win.I_win, dtype=float).reshape(3, 3)
        fim_eig_min = float(self.fim_win.eig_stats(I_win)[0])
        fim_trace = float(np.trace(I_win)) if np.all(np.isfinite(I_win)) else 0.0

        C_win = self.fim_win.crlb(use_window=True)
        crlb_trace = (
            float(np.trace(C_win))
            if np.all(np.isfinite(C_win))
            else float(cfg.v10_bad_crlb_trace)
        )
        if not np.isfinite(crlb_trace):
            crlb_trace = float(cfg.v10_bad_crlb_trace)

        meas_frac = (
            float(self.meas_used_step / max(self.meas_total_step, 1))
            if int(self.meas_total_step) > 0
            else 0.0
        )
        gate_avg = float(self.gate_avg_step) if np.isfinite(self.gate_avg_step) else 0.0

        # v8 computes sens_avg inside its reward. v10 does not call v8 reward,
        # so compute and store it here.
        sens_avg = float(self.sens_accum / max(float(self.sens_count), 1.0))
        self.sens_avg_step = float(sens_avg)

        return {
            "fim_win_eig_min": max(fim_eig_min, 0.0) if np.isfinite(fim_eig_min) else 0.0,
            "fim_win_trace": max(fim_trace, 0.0) if np.isfinite(fim_trace) else 0.0,
            "crlb_win_trace": max(crlb_trace, 0.0) if np.isfinite(crlb_trace) else float(cfg.v10_bad_crlb_trace),
            "meas_used_frac": float(_v8.clamp(meas_frac, 0.0, 1.0)),
            "gate_avg": float(_v8.clamp(gate_avg, 0.0, 1.0)),
            "sens_avg": max(sens_avg, 0.0) if np.isfinite(sens_avg) else 0.0,
        }

    def _v10_heading_metrics(self) -> Tuple[float, float, float, float]:
        """Return heading_cos, r_heading, speed_err, r_speed_match."""
        cfg = self.cfg
        vC = 0.5 * (np.asarray(self.vL1, dtype=float) + np.asarray(self.vL2, dtype=float))
        vF = np.asarray(self.vF, dtype=float)

        vC_xy = np.array([vC[0], vC[1]], dtype=float)
        vF_xy = np.array([vF[0], vF[1]], dtype=float)
        nC = float(np.linalg.norm(vC_xy))
        nF = float(np.linalg.norm(vF_xy))

        if nC <= 1e-9 or nF <= 1e-9:
            heading_cos = 0.0
        else:
            heading_cos = float(np.dot(vF_xy, vC_xy) / max(nF * nC, 1e-9))
            heading_cos = float(_v8.clamp(heading_cos, -1.0, 1.0))

        # Signed reward: +1 same horizontal direction, 0 perpendicular, -1 opposite.
        r_heading = heading_cos
        speed_err = abs(nF - nC)
        r_speed_match = -_v8.sat_ratio(speed_err, max(float(cfg.v10_speed_err0), 1e-6))
        return float(heading_cos), float(r_heading), float(speed_err), float(r_speed_match)

    def _v10_info_dir_alignment(
        self,
        planner_action_for_reward: Optional[np.ndarray],
        planner_gate_for_reward: Optional[float],
        planner_margin_for_reward: Optional[float],
    ) -> Tuple[float, float, float, float]:
        cfg = self.cfg
        a_ref = np.asarray(
            self._info_plan_action if planner_action_for_reward is None else planner_action_for_reward,
            dtype=float,
        ).reshape(3)
        gate = float(self._info_plan_gate if planner_gate_for_reward is None else planner_gate_for_reward)
        margin = float(self._info_plan_margin if planner_margin_for_reward is None else planner_margin_for_reward)
        gate = float(_v8.clamp(gate, 0.0, 1.0))
        margin_gate = math.tanh(float(cfg.info_planner_margin_gain) * max(margin, 0.0))

        a = np.asarray(self._last_action_raw, dtype=float).reshape(3)
        na = float(np.linalg.norm(a))
        nr = float(np.linalg.norm(a_ref))
        align_cos = 0.0
        if na > 1e-6 and nr > 1e-6:
            align_cos = float(np.dot(a, a_ref) / max(na * nr, 1e-6))
            align_cos = float(_v8.clamp(align_cos, -1.0, 1.0))

        r_info_dir = gate * margin_gate * max(align_cos, 0.0)
        return float(align_cos), float(gate), float(margin), float(r_info_dir)

    def _info_search_action(
        self,
        support_points: np.ndarray,
        support_weights: np.ndarray,
        p0_eff: np.ndarray,
        tol_pos_est: float,
    ) -> Tuple[Optional[_v8.InfoPlanEval], Optional[_v8.InfoPlanEval], float]:
        cfg = self.cfg
        backend = str(
            os.environ.get(
                "UUV_V10_INFO_BACKEND",
                getattr(cfg, "v10_info_planner_backend", "numba"),
            )
        ).lower().strip()
        if backend in {"base", "python", "off"}:
            return super()._info_search_action(support_points, support_weights, p0_eff, tol_pos_est)
        if backend == "auto":
            backend = "cython" if _V10_CYTHON_AVAILABLE else "numba"
        if backend == "cython":
            if not _V10_CYTHON_AVAILABLE or _v10_cy is None:
                raise RuntimeError(
                    "Wybrano --v10-info-planner-backend cython, ale modul "
                    "uuv_v10_info_tracking_cy nie jest zbudowany."
                )
            kernel = _v10_cy.info_search
        elif backend == "numba":
            if not (bool(getattr(cfg, "v10_info_planner_numba", True)) and bool(getattr(_v8, "_NUMBA_AVAILABLE", False))):
                return super()._info_search_action(support_points, support_weights, p0_eff, tol_pos_est)
            kernel = _v10_fast_info_search
        else:
            raise ValueError(f"Nieznany backend v10 info planner: {backend!r}")

        candidates = np.ascontiguousarray(self.INFO_ACTION_CANDIDATES, dtype=np.float64)
        try:
            best_idx, zero_idx, second_best_score, metrics = kernel(
                candidates,
                np.ascontiguousarray(support_points, dtype=np.float64),
                np.ascontiguousarray(support_weights, dtype=np.float64),
                np.ascontiguousarray(p0_eff, dtype=np.float64),
                np.ascontiguousarray(self.pL1, dtype=np.float64),
                np.ascontiguousarray(self.pL2, dtype=np.float64),
                np.ascontiguousarray(self.vL1, dtype=np.float64),
                np.ascontiguousarray(self.vL2, dtype=np.float64),
                float(self.speed_F),
                float(self.yaw_F),
                float(self.pitch_F),
                float(self._difficulty),
                float(self._gate_relax_active),
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
                float(self.pf.meas_sigma),
                float(getattr(self.pf, "sigma_nis_mult", 1.0)),
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
        except Exception:
            if backend == "cython":
                raise
            return super()._info_search_action(support_points, support_weights, p0_eff, tol_pos_est)

        def _make_eval(idx: int) -> _v8.InfoPlanEval:
            idx = int(idx)
            return _v8.InfoPlanEval(
                action=np.asarray(candidates[idx], dtype=np.float32),
                score=float(metrics[idx, 0]),
                trace_red=float(metrics[idx, 1]),
                eigmin=float(metrics[idx, 2]),
                sens_avg=float(metrics[idx, 3]),
                pred_err=float(metrics[idx, 4]),
                gate_avg=float(metrics[idx, 5]),
                disp_world_first=np.asarray(metrics[idx, 6:9], dtype=float),
            )

        return _make_eval(int(best_idx)), _make_eval(int(zero_idx)), float(second_best_score)

    def _v10_success_flags(self, err_est: float, err_true_form: float, std_max: float) -> Tuple[bool, bool, bool]:
        tol_pos_est, tol_std, tol_pos_true = self._current_tolerances()
        success_progress_now = bool(float(err_est) < float(tol_pos_est))
        success_true_now = bool(float(err_true_form) < float(tol_pos_true))
        if bool(self.cfg.success_require_std):
            success_progress_now = bool(success_progress_now and (float(std_max) < float(tol_std)))
            success_true_now = bool(success_true_now and (float(std_max) < float(tol_std)))

        self._v10_ever_progress_success = bool(
            getattr(self, "_v10_ever_progress_success", False) or success_progress_now
        )
        self._v10_ever_true_success = bool(
            getattr(self, "_v10_ever_true_success", False) or success_true_now
        )

        mode = str(getattr(self.cfg, "success_mode", "progress")).lower().strip()
        success_now = success_true_now if mode == "true" else success_progress_now
        return bool(success_progress_now), bool(success_true_now), bool(success_now)

    # ------------------------------- reward --------------------------------

    def _compute_reward(
        self,
        pf_stats_last: Optional[_v8.PFStats],
        planner_action_for_reward: Optional[np.ndarray] = None,
        planner_gate_for_reward: Optional[float] = None,
        planner_margin_for_reward: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        del pf_stats_last

        cfg = self.cfg
        _, pF_des = self._formation_desired()
        tol_pos_est, tol_std, tol_pos_true = self._current_tolerances()
        pF_hat = np.asarray(self.pf.mean, dtype=float).reshape(3)
        err_est = float(np.linalg.norm(pF_hat - pF_des))
        err_true_form = float(np.linalg.norm(np.asarray(self.pF, dtype=float).reshape(3) - pF_des))

        sx = float(self.std_x_eff_step)
        sy = float(self.std_y_eff_step)
        sz = float(self.std_z_eff_step)
        std_max = float(self.std_max_eff_step)
        std_infl = float(self.std_infl_step)
        if (not np.isfinite(std_max)) or (std_max <= 0.0):
            sx_raw, sy_raw, sz_raw = self.pf.stds()
            sx, sy, sz, std_max, std_infl = self._compute_conservative_std(sx_raw, sy_raw, sz_raw, self.pf.cov)
            self.std_x_eff_step = float(sx)
            self.std_y_eff_step = float(sy)
            self.std_z_eff_step = float(sz)
            self.std_max_eff_step = float(std_max)
            self.std_infl_step = float(std_infl)

        m = self._v10_window_info_metrics()
        fim = float(m["fim_win_eig_min"])
        crlb_raw = float(m["crlb_win_trace"])
        crlb = float(min(max(crlb_raw, 0.0), float(cfg.v10_crlb_trace_cap)))
        meas_frac = float(m["meas_used_frac"])
        gate_avg = float(m["gate_avg"])
        sens_avg = float(m["sens_avg"])

        # Absolute information quality. Both are scaled so that value around 1
        # corresponds roughly to the desired target. Values above target are
        # still rewarded, but are bounded.
        fim_abs = 2.0 * fim / (fim + max(float(cfg.v10_target_fim_min), 1e-9)) if fim > 0.0 else 0.0
        fim_abs = float(_v8.clamp(fim_abs, 0.0, 2.0))
        crlb_abs = 2.0 * float(cfg.v10_target_crlb_trace) / (
            crlb + max(float(cfg.v10_target_crlb_trace), 1e-9)
        )
        crlb_abs = float(_v8.clamp(crlb_abs, 0.0, 2.0))

        # Delta shaping: positive when FIM improves or CRLB trace decreases.
        fim_log = math.log1p(max(fim, 0.0))
        crlb_log = math.log1p(max(crlb, 0.0))
        prev_fim_log = float(getattr(self, "_v10_prev_log_fim", fim_log))
        prev_crlb_log = float(getattr(self, "_v10_prev_log_crlb", crlb_log))
        fim_gain_raw = fim_log - prev_fim_log
        crlb_gain_raw = prev_crlb_log - crlb_log
        self._v10_prev_log_fim = float(fim_log)
        self._v10_prev_log_crlb = float(crlb_log)
        r_fim_gain = math.tanh(fim_gain_raw / max(float(cfg.v10_fim_delta0), 1e-9))
        r_crlb_gain = math.tanh(crlb_gain_raw / max(float(cfg.v10_crlb_delta0), 1e-9))

        # Position tracking is a continuous penalty, not a terminal objective.
        r_pos_est = -_v8.sat_ratio(err_est, max(float(cfg.v10_pos_err0_m), 1e-6))
        r_pos_true = -_v8.sat_ratio(err_true_form, max(float(cfg.v10_pos_err0_m), 1e-6))
        prev_err_est = float(getattr(self, "_v10_prev_err_est", err_est))
        pos_gain_raw = prev_err_est - err_est
        self._v10_prev_err_est = float(err_est)
        r_pos_gain = math.tanh(pos_gain_raw / max(float(cfg.v10_pos_gain0_m), 1e-9))

        # Keep the follower moving with the leaders instead of maximizing info
        # by turning into arbitrary excitation patterns.
        heading_cos, r_heading, speed_err, r_speed_match = self._v10_heading_metrics()
        info_dir_cos, info_dir_gate, info_dir_margin, r_info_dir = self._v10_info_dir_alignment(
            planner_action_for_reward,
            planner_gate_for_reward,
            planner_margin_for_reward,
        )

        est_ratio = err_est / max(float(tol_pos_est), 1e-6)
        true_ratio = err_true_form / max(float(tol_pos_true), 1e-6)
        std_ratio = std_max / max(float(tol_std), 1e-6)
        guard_ratio = max(float(true_ratio), float(std_ratio))
        soft_ratio = max(float(cfg.v10_track_guard_soft_ratio), 1.0)
        hard_ratio = max(float(cfg.v10_track_guard_hard_ratio), soft_ratio + 1e-6)
        guard_t = _v8.smoothstep01((guard_ratio - soft_ratio) / max(hard_ratio - soft_ratio, 1e-6))
        info_guard = float(cfg.v10_info_guard_floor) + (1.0 - float(cfg.v10_info_guard_floor)) * (1.0 - guard_t)
        info_guard = float(_v8.clamp(info_guard, 0.0, 1.0))

        track_excess = max(0.0, guard_ratio - 1.0)
        est_excess = max(0.0, float(est_ratio) - 1.0)
        r_track_guard = -_v8.sat_ratio(track_excess, max(float(cfg.v10_track_guard_ratio0), 1e-6))
        r_est_guard = -_v8.sat_ratio(est_excess, max(float(cfg.v10_est_guard_ratio0), 1e-6))
        near_ratio = max(float(cfg.v10_track_near_ratio), 1.0 + 1e-6)
        r_track_near = _v8.smoothstep01((near_ratio - guard_ratio) / max(near_ratio - 1.0, 1e-6))

        vC = 0.5 * (np.asarray(self.vL1, dtype=float) + np.asarray(self.vL2, dtype=float))
        vC_speed = float(np.linalg.norm(vC))
        speed_ratio = float(self.speed_F) / max(vC_speed, 1e-6)
        speed_excess = max(0.0, speed_ratio - float(cfg.v10_speed_excess_ratio))
        r_speed_excess = -_v8.sat_ratio(speed_excess, max(float(cfg.v10_speed_excess_ratio0), 1e-6))

        success_progress_now, success_true_now, success_now = self._v10_success_flags(err_est, err_true_form, std_max)
        r_success_step = 0.5 * float(1.0 if success_progress_now else 0.0) + 0.5 * float(1.0 if success_true_now else 0.0)

        # PF/measurement health terms. These are secondary because FIM/CRLB are
        # the main information target.
        r_meas = _v8.smoothstep01(meas_frac / max(float(cfg.v10_target_meas_used_frac), 1e-9))
        r_sens = 2.0 * sens_avg / (sens_avg + max(float(cfg.v10_target_sens_avg), 1e-9)) if sens_avg > 0.0 else 0.0
        r_sens = float(_v8.clamp(r_sens, 0.0, 2.0))
        r_std = -_v8.sat_ratio(std_max, max(float(cfg.v10_std0_m), float(tol_std), 1e-6))
        r_energy = -_v8.sat_ratio(float(np.linalg.norm(self.a_cmd)), 1.0)
        r_time = -1.0

        reward = 0.0
        reward += float(cfg.v10_w_fim_abs) * float(info_guard) * float(fim_abs)
        reward += float(cfg.v10_w_crlb_abs) * float(info_guard) * float(crlb_abs)
        reward += float(cfg.v10_w_fim_gain) * float(info_guard) * float(r_fim_gain)
        reward += float(cfg.v10_w_crlb_gain) * float(info_guard) * float(r_crlb_gain)
        reward += float(cfg.v10_w_meas) * float(r_meas)
        reward += float(cfg.v10_w_sens) * float(info_guard) * float(r_sens)
        reward += float(cfg.v10_w_pos_est) * float(r_pos_est)
        reward += float(cfg.v10_w_pos_true) * float(r_pos_true)
        reward += float(cfg.v10_w_pos_gain) * float(r_pos_gain)
        reward += float(cfg.v10_w_track_guard) * float(r_track_guard)
        reward += float(cfg.v10_w_est_guard) * float(r_est_guard)
        reward += float(cfg.v10_w_track_near) * float(r_track_near)
        reward += float(cfg.v10_w_success_step) * float(r_success_step)
        reward += float(cfg.v10_w_heading) * float(r_heading)
        reward += float(cfg.v10_w_speed_match) * float(r_speed_match)
        reward += float(cfg.v10_w_speed_excess) * float(r_speed_excess)
        reward += float(cfg.v10_w_info_dir_align) * float(info_guard) * float(r_info_dir)
        reward += float(cfg.v10_w_std) * float(r_std)
        reward += float(cfg.v10_w_energy) * float(r_energy)
        reward += float(cfg.v10_w_time) * float(r_time)
        reward = float(np.clip(reward, -float(cfg.rew_clip), float(cfg.rew_clip)))

        terms: Dict[str, float] = {
            "r_total": float(reward),
            "v10_version": 10.1,
            "v10_r_fim_abs": float(fim_abs),
            "v10_r_crlb_abs": float(crlb_abs),
            "v10_r_fim_gain": float(r_fim_gain),
            "v10_r_crlb_gain": float(r_crlb_gain),
            "v10_r_fim_abs_guarded": float(info_guard * fim_abs),
            "v10_r_crlb_abs_guarded": float(info_guard * crlb_abs),
            "v10_r_fim_gain_guarded": float(info_guard * r_fim_gain),
            "v10_r_crlb_gain_guarded": float(info_guard * r_crlb_gain),
            "v10_r_pos_est": float(r_pos_est),
            "v10_r_pos_true": float(r_pos_true),
            "v10_r_pos_gain": float(r_pos_gain),
            "v10_r_track_guard": float(r_track_guard),
            "v10_r_est_guard": float(r_est_guard),
            "v10_r_track_near": float(r_track_near),
            "v10_r_success_step": float(r_success_step),
            "v10_r_heading": float(r_heading),
            "v10_r_speed_match": float(r_speed_match),
            "v10_r_speed_excess": float(r_speed_excess),
            "v10_r_info_dir": float(r_info_dir),
            "v10_r_info_dir_guarded": float(info_guard * r_info_dir),
            "v10_r_meas": float(r_meas),
            "v10_r_sens": float(r_sens),
            "v10_r_sens_guarded": float(info_guard * r_sens),
            "v10_r_std": float(r_std),
            "v10_r_energy": float(r_energy),
            "v10_r_time": float(r_time),
            "v10_info_guard": float(info_guard),
            "v10_guard_ratio": float(guard_ratio),
            "v10_est_ratio": float(est_ratio),
            "v10_true_ratio": float(true_ratio),
            "v10_std_ratio": float(std_ratio),
            "v10_speed_ratio": float(speed_ratio),
            "v10_fim_win_eig_min": float(fim),
            "v10_fim_win_trace": float(m["fim_win_trace"]),
            "v10_crlb_win_trace": float(crlb_raw),
            "v10_crlb_win_trace_capped": float(crlb),
            "v10_fim_gain_raw": float(fim_gain_raw),
            "v10_crlb_gain_raw": float(crlb_gain_raw),
            "v10_meas_used_frac": float(meas_frac),
            "v10_gate_avg": float(gate_avg),
            "v10_sens_avg": float(sens_avg),
            "v10_heading_cos_leaders": float(heading_cos),
            "v10_speed_err_leaders": float(speed_err),
            "v10_info_dir_align_cos": float(info_dir_cos),
            "v10_info_dir_gate": float(info_dir_gate),
            "v10_info_dir_margin": float(info_dir_margin),
            "err_est": float(err_est),
            "err_true_form": float(err_true_form),
            "std_max": float(std_max),
            "std_infl": float(std_infl),
            "tol_pos_est": float(tol_pos_est),
            "tol_pos_true": float(tol_pos_true),
            "tol_std": float(tol_std),
            "success_progress_now": float(1.0 if success_progress_now else 0.0),
            "success_true_now": float(1.0 if success_true_now else 0.0),
            "tracking_success_now": float(1.0 if success_now else 0.0),
            "tracking_success_progress_ever": float(1.0 if getattr(self, "_v10_ever_progress_success", False) else 0.0),
            "tracking_success_true_ever": float(1.0 if getattr(self, "_v10_ever_true_success", False) else 0.0),
            # Kept for compatibility with v8/v9 logging. There is no terminal
            # goal bonus in v10.
            "bonus_goal": 0.0,
            "terminal_bonus": 0.0,
        }
        return reward, terms

    # ------------------------------ termination ----------------------------

    def _check_done(self) -> Tuple[bool, bool, str]:
        # Fixed-horizon task: never terminate because formation was reached.
        # Reaching formation is only diagnostic; information tracking continues.
        _, pF_des = self._formation_desired()
        tol_pos_est, tol_std, tol_pos_true = self._current_tolerances()
        err_est = float(np.linalg.norm(self.pf.mean - pF_des))
        err_true_form = float(np.linalg.norm(self.pF - pF_des))
        std_max = float(self.std_max_eff_step) if np.isfinite(self.std_max_eff_step) else float(self.pf.std_max())

        success_progress_now = bool(err_est < tol_pos_est)
        success_true_now = bool(err_true_form < tol_pos_true)
        if bool(self.cfg.success_require_std):
            success_progress_now = bool(success_progress_now and (std_max < tol_std))
            success_true_now = bool(success_true_now and (std_max < tol_std))

        self._episode_progress_success = bool(success_progress_now)
        self._episode_true_success = bool(success_true_now)
        self._v10_ever_progress_success = bool(
            getattr(self, "_v10_ever_progress_success", False) or success_progress_now
        )
        self._v10_ever_true_success = bool(
            getattr(self, "_v10_ever_true_success", False) or success_true_now
        )

        # Keep streak only as a diagnostic counter; do not use it to terminate.
        mode = str(getattr(self.cfg, "success_mode", "progress")).lower().strip()
        success_now = success_true_now if mode == "true" else success_progress_now
        if success_now:
            self.success_streak += 1
        else:
            self.success_streak = 0

        truncated = bool(self.step_count >= int(self.cfg.max_steps))
        if truncated:
            return False, True, "max_steps"
        return False, False, "running"

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)

        mode = str(getattr(self.cfg, "success_mode", "progress")).lower().strip()
        ever_progress = bool(getattr(self, "_v10_ever_progress_success", False))
        ever_true = bool(getattr(self, "_v10_ever_true_success", False))
        ever_selected = ever_true if mode == "true" else ever_progress
        episode_done = bool(getattr(self, "_last_truncated", False) or getattr(self, "_last_terminated", False))

        # v8 eval/TB callbacks look at is_success only when done. In v10, this
        # is a diagnostic: did the fixed-horizon episode ever satisfy the old
        # tracking success condition? It does not terminate the episode.
        info["tracking_success_progress_ever"] = float(1.0 if ever_progress else 0.0)
        info["tracking_success_true_ever"] = float(1.0 if ever_true else 0.0)
        info["tracking_success_ever"] = float(1.0 if ever_selected else 0.0)
        info["is_success"] = float(1.0 if (episode_done and ever_selected) else 0.0)
        info["success"] = info["is_success"]
        info["is_success_progress_terminal"] = float(1.0 if (episode_done and ever_progress) else 0.0)
        info["is_success_true_terminal"] = float(1.0 if (episode_done and ever_true) else 0.0)
        return info


# =============================================================================
# v8 CLI patching
# =============================================================================


def make_env(seed: int, cfg: UUV3DConfig, render: bool, rank: int = 0):
    env_cls = UUVTwoLeader3DPFEnv

    def _init():
        env = env_cls(cfg=cfg, render_mode=("human" if render else "none"))
        env.reset(seed=seed + rank)
        return env

    return _init


def _iter_parser_actions(parser: argparse.ArgumentParser):
    for action in parser._actions:
        yield action
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            for subparser in choices.values():
                if isinstance(subparser, argparse.ArgumentParser):
                    yield from _iter_parser_actions(subparser)


def _retarget_parser_defaults(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    path_defaults = {
        "models_3d_v8": "models_3d_v10_info_tracking_guarded",
        "logs_3d_v8": "logs_3d_v10_info_tracking_guarded",
        "tb_3d_v8": "tb_3d_v10_info_tracking_guarded",
        "eval_3d_logs_v8": "eval_3d_logs_v10_info_tracking_guarded",
        "info_maps_v8": "info_maps_v10_info_tracking_guarded",
    }
    config_defaults = {
        "total_timesteps": 10_000_000,
        "n_envs": 24,
        "eval_freq": 100_000,
        "eval_episodes": 10,
        "save_freq": 50_000,
        "tb_info_freq": 25_000,
        "trace_freq": 100_000,
        "curriculum_frac": 0.90,
        "replay_reset_difficulty": 0.80,
        "success_mode": "progress",
        "info_gate_floor_hard": UUV3DConfig.info_gate_floor_hard,
        "info_gate_floor_hard_extra": UUV3DConfig.info_gate_floor_hard_extra,
        "info_gate_floor_start_difficulty": UUV3DConfig.info_gate_floor_start_difficulty,
        "info_gate_floor_ramp_difficulty": UUV3DConfig.info_gate_floor_ramp_difficulty,
    }
    for action in _iter_parser_actions(parser):
        # Safer than v9: action.default can theoretically be unhashable.
        if isinstance(action.default, str) and action.default in path_defaults:
            action.default = path_defaults[action.default]
        if action.dest in config_defaults:
            action.default = config_defaults[action.dest]
    return parser


def _get_subparser(parser: argparse.ArgumentParser, name: str) -> Optional[argparse.ArgumentParser]:
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and name in choices:
            subparser = choices.get(name)
            if isinstance(subparser, argparse.ArgumentParser):
                return subparser
    return None


def _parser_has_dest(parser: argparse.ArgumentParser, dest: str) -> bool:
    return any(getattr(action, "dest", None) == dest for action in parser._actions)


def _add_v10_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    train_parser = _get_subparser(parser, "train")
    if train_parser is None:
        return parser
    if not _parser_has_dest(train_parser, "gradient_steps"):
        train_parser.add_argument(
            "--gradient-steps",
            type=int,
            default=None,
            help="SAC gradient steps per rollout step. Default auto-scales with --n-envs.",
        )
    if not _parser_has_dest(train_parser, "batch_size"):
        train_parser.add_argument("--batch-size", type=int, default=None, help="SAC replay batch size override.")
    if not _parser_has_dest(train_parser, "train_freq"):
        train_parser.add_argument("--train-freq", type=int, default=None, help="SAC train_freq override.")
    if not _parser_has_dest(train_parser, "learning_starts"):
        train_parser.add_argument("--learning-starts", type=int, default=None, help="SAC learning_starts override.")
    if not _parser_has_dest(train_parser, "v10_info_planner_backend"):
        train_parser.add_argument(
            "--v10-info-planner-backend",
            choices=("numba", "cython", "auto", "python"),
            default=None,
            help="Backend for the v10 information planner hot loop. Default keeps the dataclass setting.",
        )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return _add_v10_train_args(_retarget_parser_defaults(_v8.build_parser()))


def _patch_v8_entrypoints(*, patch_env_class: bool = False) -> None:
    _v8.UUV3DConfig = UUV3DConfig
    _v8.make_env = make_env
    if patch_env_class:
        _v8.UUVTwoLeader3DPFEnv = UUVTwoLeader3DPFEnv
    else:
        _v8.UUVTwoLeader3DPFEnv = _V8Env


def cmd_train(args: argparse.Namespace) -> None:
    _patch_v8_entrypoints(patch_env_class=False)
    backend_override = getattr(args, "v10_info_planner_backend", None)
    if backend_override:
        os.environ["UUV_V10_INFO_BACKEND"] = str(backend_override).lower().strip()
    requested_backend = str(
        os.environ.get("UUV_V10_INFO_BACKEND", UUV3DConfig.v10_info_planner_backend)
    ).lower().strip()
    if requested_backend == "cython" and not _V10_CYTHON_AVAILABLE:
        raise RuntimeError(
            "Wybrano --v10-info-planner-backend cython, ale rozszerzenie Cython "
            "nie jest zbudowane. Uruchom: python setup_uuv_v10_cython.py build_ext --inplace"
        )
    effective_backend = "cython" if requested_backend == "auto" and _V10_CYTHON_AVAILABLE else requested_backend
    print(
        f"[V10] info_planner_backend={requested_backend} "
        f"effective={effective_backend} cython_available={_V10_CYTHON_AVAILABLE}"
    )
    try:
        import stable_baselines3 as sb3
        from stable_baselines3.common import callbacks as sb3_callbacks
    except Exception:
        sb3 = None
        sb3_callbacks = None

    original_checkpoint_callback = None
    original_sac_cls = None
    if sb3_callbacks is not None:
        original_checkpoint_callback = sb3_callbacks.CheckpointCallback

        class V10CheckpointCallback(original_checkpoint_callback):
            def __init__(self, *cb_args, **cb_kwargs):
                cb_kwargs["save_vecnormalize"] = True
                super().__init__(*cb_args, **cb_kwargs)

        sb3_callbacks.CheckpointCallback = V10CheckpointCallback

    if sb3 is not None and str(getattr(args, "algo", "sac")).lower() == "sac":
        original_sac_cls = sb3.SAC
        if getattr(args, "gradient_steps", None) is None:
            # Keep roughly the same update/sample ratio as the earlier 24-env runs.
            args.gradient_steps = max(1, int(round(float(max(int(getattr(args, "n_envs", 1)), 1)) / 24.0)))

        sac_overrides: Dict[str, Any] = {}
        for arg_name, kw_name in (
            ("gradient_steps", "gradient_steps"),
            ("batch_size", "batch_size"),
            ("train_freq", "train_freq"),
            ("learning_starts", "learning_starts"),
        ):
            value = getattr(args, arg_name, None)
            if value is not None:
                sac_overrides[kw_name] = int(value)

        class V10SAC(original_sac_cls):
            def __init__(self, *model_args, **model_kwargs):
                for k, v in sac_overrides.items():
                    model_kwargs[k] = v
                super().__init__(*model_args, **model_kwargs)

            @classmethod
            def load(cls, path, *load_args, **load_kwargs):
                model = original_sac_cls.load(path, *load_args, **load_kwargs)
                for k, v in sac_overrides.items():
                    if k != "train_freq":
                        setattr(model, k, v)
                return model

        sb3.SAC = V10SAC
        print(f"[V10] SAC overrides: {sac_overrides}")

    try:
        _v8.cmd_train(args)
    finally:
        if sb3_callbacks is not None and original_checkpoint_callback is not None:
            sb3_callbacks.CheckpointCallback = original_checkpoint_callback
        if sb3 is not None and original_sac_cls is not None:
            sb3.SAC = original_sac_cls


def cmd_eval(args: argparse.Namespace) -> None:
    _patch_v8_entrypoints(patch_env_class=True)
    try:
        _v8.cmd_eval(args)
    finally:
        _patch_v8_entrypoints(patch_env_class=False)


def cmd_sim(args: argparse.Namespace) -> None:
    _patch_v8_entrypoints(patch_env_class=True)
    try:
        _v8.cmd_sim(args)
    finally:
        _patch_v8_entrypoints(patch_env_class=False)


def cmd_map(args: argparse.Namespace) -> None:
    _patch_v8_entrypoints(patch_env_class=True)
    try:
        _v8.cmd_map(args)
    finally:
        _patch_v8_entrypoints(patch_env_class=False)


def main() -> None:
    _patch_v8_entrypoints(patch_env_class=False)
    args = build_parser().parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "sim":
        cmd_sim(args)
    elif args.cmd == "map":
        cmd_map(args)
    else:
        raise ValueError("Unknown command")


_patch_v8_entrypoints(patch_env_class=False)


if __name__ == "__main__":
    main()
