# -*- coding: utf-8 -*-
"""
UUV RL (3D) – v8 temporal info-fix + info-guidance

Standalone v8 file created by merging the stable v7 environment with the
short-horizon information-guidance wrapper. This version keeps the full v7
training / eval / sim CLI and adds:
- observation redesign to reduce hidden non-stationarity,
- formation-frame (leader-centric) state representation,
- PF-health / Doppler-quality features,
- online short-horizon information planner,
- guidance action / direction / score features in the observation,
- optional reward shaping toward the planner recommendation,
- offline helper to export a frozen-scenario 3D information map.

Important:
- This script is standalone; it does not depend on `wrapper_v7_temporal_infofix.py`.
- Final observation size is 64 base features (55 core + 9 info-guidance features).
- Models trained with older versions are NOT compatible with this file.

Typical training:
python Skrypty/UUV_ART2/uuv_v8_temporal_infofix.py train --algo sac --n-envs 20 --eval-freq 250000 --eval-episodes 20 --tb-info-freq 25000 --total-timesteps 100000000 --tb-log tb_3d_v8 --log-dir logs_3d_v8 --models-dir models_3d_v8 --net-arch 512,512,512 --activation relu --success-mode progress --obs-history-len 4 --info-gate-floor-hard 0.30 --info-gate-floor-hard-extra 0.25

Typical info-map export:
python Skrypty/UUV_ART2/uuv_v8_temporal_infofix.py map --seed 42 --out-dir info_maps_v8
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except Exception:
    pygame = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# =============================================================================
# Helpers
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def lerp(a: float, b: float, t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return (1.0 - t) * float(a) + t * float(b)


def sat_ratio(x: float, x0: float) -> float:
    x = float(max(0.0, x))
    x0 = float(max(1e-12, x0))
    return x / (x + x0)


def smoothstep01(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_gate(x: float, x_min: float, band: float) -> float:
    x = float(x)
    x_min = float(x_min)
    band = float(max(0.0, band))
    if band <= 1e-12:
        return 1.0 if x > x_min else 0.0
    t = (x - x_min) / band
    return smoothstep01(t)


def deg2rad(a: float) -> float:
    return float(a) * math.pi / 180.0


def rad2deg(a: float) -> float:
    return float(a) * 180.0 / math.pi


def wrap360(a_deg: float) -> float:
    a = float(a_deg) % 360.0
    if a < 0:
        a += 360.0
    return a


def wrap180(a_deg: float) -> float:
    return (float(a_deg) + 180.0) % 360.0 - 180.0


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    N = int(w.shape[0])
    positions = (rng.random() + np.arange(N)) / N
    cumulative_sum = np.cumsum(w)
    idx = np.zeros(N, dtype=int)
    i = 0
    j = 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
            if j >= N:
                j = N - 1
    return idx


def yaw_pitch_to_dir(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array([cp * cy, cp * sy, sp], dtype=float)


def vel_from_speed_yaw_pitch(speed: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    return float(speed) * yaw_pitch_to_dir(yaw_deg, pitch_deg)


def radial_speed(r_vec: np.ndarray, v_rel: np.ndarray) -> float:
    r_vec = np.asarray(r_vec, dtype=float).reshape(3)
    v_rel = np.asarray(v_rel, dtype=float).reshape(3)
    rho = float(np.linalg.norm(r_vec))
    if rho <= 1e-12:
        return 0.0
    rhat = r_vec / rho
    return -float(np.dot(rhat, v_rel))


def doppler_H_3d(r_vec: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
    r_vec = np.asarray(r_vec, dtype=float).reshape(3)
    v_rel = np.asarray(v_rel, dtype=float).reshape(3)
    rho = float(np.linalg.norm(r_vec))
    if rho <= 1e-12:
        return np.zeros((1, 3), dtype=float)
    rhat = r_vec / rho
    proj = float(np.dot(rhat, v_rel))
    H = -(v_rel - proj * rhat) / rho
    return H.reshape(1, 3)


def safe_inv_sym(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    M = 0.5 * (M + M.T)
    try:
        return np.linalg.inv(M + eps * np.eye(M.shape[0], dtype=float))
    except np.linalg.LinAlgError:
        return np.full_like(M, np.nan, dtype=float)


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _flatten(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def normalize_vec(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        if fallback is None:
            out = np.zeros_like(v)
            out[0] = 1.0
            return out
        fb = np.asarray(fallback, dtype=float).reshape(-1)
        nf = float(np.linalg.norm(fb))
        if nf <= 1e-12:
            out = np.zeros_like(v)
            out[0] = 1.0
            return out
        return fb / nf
    return v / n


def formation_axes_from_vel(vL1: np.ndarray, vL2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vL1 = np.asarray(vL1, dtype=float).reshape(3)
    vL2 = np.asarray(vL2, dtype=float).reshape(3)
    vC = 0.5 * (vL1 + vL2)
    f_hat = normalize_vec(np.array([vC[0], vC[1], 0.0], dtype=float), fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    zhat = np.array([0.0, 0.0, 1.0], dtype=float)
    side_hat = np.cross(f_hat, zhat)
    side_hat = normalize_vec(side_hat, fallback=np.array([0.0, 1.0, 0.0], dtype=float))
    return f_hat, side_hat, zhat


def form_rotation_matrix(f_hat: np.ndarray, side_hat: np.ndarray, zhat: np.ndarray) -> np.ndarray:
    return np.vstack([
        np.asarray(f_hat, dtype=float).reshape(3),
        np.asarray(side_hat, dtype=float).reshape(3),
        np.asarray(zhat, dtype=float).reshape(3),
    ])


def project_vec_form(v_world: np.ndarray, R_form: np.ndarray) -> np.ndarray:
    return np.asarray(R_form, dtype=float).reshape(3, 3) @ np.asarray(v_world, dtype=float).reshape(3)


def project_cov_form(C_world: np.ndarray, R_form: np.ndarray) -> np.ndarray:
    R = np.asarray(R_form, dtype=float).reshape(3, 3)
    C = np.asarray(C_world, dtype=float).reshape(3, 3)
    C_form = R @ C @ R.T
    return 0.5 * (C_form + C_form.T)


def cov_to_corrs(C: np.ndarray) -> Tuple[float, float, float]:
    C = np.asarray(C, dtype=float).reshape(3, 3)
    s0 = math.sqrt(max(float(C[0, 0]), 0.0))
    s1 = math.sqrt(max(float(C[1, 1]), 0.0))
    s2 = math.sqrt(max(float(C[2, 2]), 0.0))
    def _corr(a: float, b: float, c: float) -> float:
        denom = max(a * b, 1e-9)
        return float(clamp(c / denom, -1.0, 1.0))
    return _corr(s0, s1, float(C[0, 1])), _corr(s0, s2, float(C[0, 2])), _corr(s1, s2, float(C[1, 2]))


def dominant_unc_axis_abs(C_form: np.ndarray) -> np.ndarray:
    C = np.asarray(C_form, dtype=float).reshape(3, 3)
    C = 0.5 * (C + C.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = int(np.argmax(eigvals))
        v = np.abs(np.asarray(eigvecs[:, idx], dtype=float).reshape(3))
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return v / n
    except np.linalg.LinAlgError:
        return np.array([1.0, 0.0, 0.0], dtype=float)


def resolve_torch_device(device_arg: str) -> str:
    device = str(device_arg).lower().strip()
    if device != "auto":
        return device
    try:
        import torch as th
        if bool(th.cuda.is_available()):
            return "cuda"
        mps_ok = bool(
            hasattr(th.backends, "mps")
            and hasattr(th.backends.mps, "is_available")
            and th.backends.mps.is_available()
        )
        if mps_ok:
            return "mps"
    except Exception:
        pass
    return "cpu"


def _pid_action_from_info(info: Dict[str, Any], cfg: "UUV3DBaseConfig", mode: str, step_count: int) -> np.ndarray:
    mode = str(mode).lower().strip()
    pF_hat = np.array([
        float(info.get("pFhat_x", 0.0)),
        float(info.get("pFhat_y", 0.0)),
        float(info.get("pFhat_z", 0.0)),
    ], dtype=float)
    pF_des = np.array([
        float(info.get("pFdes_x", 0.0)),
        float(info.get("pFdes_y", 0.0)),
        float(info.get("pFdes_z", 0.0)),
    ], dtype=float)
    yaw_F = float(info.get("yaw_F", 0.0))
    pitch_F = float(info.get("pitch_F", 0.0))

    e = pF_des - pF_hat
    ex, ey, ez = e
    e_xy = math.hypot(ex, ey)
    e_norm = float(np.linalg.norm(e))

    yaw_des = yaw_F if e_xy <= 1e-9 else wrap360(rad2deg(math.atan2(ey, ex)))
    pitch_des = rad2deg(math.atan2(ez, max(e_xy, 1e-6)))

    yaw_err = wrap180(yaw_des - yaw_F)
    pitch_err = pitch_des - pitch_F
    a_speed = clamp(sat_ratio(max(e_norm - 8.0, 0.0), 40.0), 0.0, 1.0)
    a_yaw = clamp(yaw_err / max(float(cfg.rl_yaw_per_step_deg), 1e-6), -1.0, 1.0)
    a_pitch = clamp(pitch_err / max(float(cfg.rl_pitch_per_step_deg), 1e-6), -1.0, 1.0)

    if mode == "pid_exc":
        std_max = float(info.get("std_max_eff", 0.0))
        tol_std = float(lerp(float(cfg.tol_std_easy), float(cfg.tol_std_hard), float(cfg.difficulty_fixed or 1.0)))
        excite = sat_ratio(max(std_max - tol_std, 0.0), max(tol_std, 1e-6))
        a_yaw += 0.20 * math.sin(0.10 * float(step_count)) * excite
        a_pitch += 0.15 * math.cos(0.12 * float(step_count)) * excite

    return np.array([
        float(clamp(a_speed, -1.0, 1.0)),
        float(clamp(a_yaw, -1.0, 1.0)),
        float(clamp(a_pitch, -1.0, 1.0)),
    ], dtype=np.float32)


@njit(cache=True)
def _pf_recompute_stats_numba(p: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = p.shape[0]
    mean = np.zeros(3, dtype=np.float64)
    for i in range(n):
        wi = w[i]
        mean[0] += p[i, 0] * wi
        mean[1] += p[i, 1] * wi
        mean[2] += p[i, 2] * wi

    cov = np.zeros((3, 3), dtype=np.float64)
    for i in range(n):
        wi = w[i]
        dx0 = p[i, 0] - mean[0]
        dx1 = p[i, 1] - mean[1]
        dx2 = p[i, 2] - mean[2]
        cov[0, 0] += wi * dx0 * dx0
        cov[0, 1] += wi * dx0 * dx1
        cov[0, 2] += wi * dx0 * dx2
        cov[1, 1] += wi * dx1 * dx1
        cov[1, 2] += wi * dx1 * dx2
        cov[2, 2] += wi * dx2 * dx2

    cov[1, 0] = cov[0, 1]
    cov[2, 0] = cov[0, 2]
    cov[2, 1] = cov[1, 2]
    cov[0, 0] += 1e-9
    cov[1, 1] += 1e-9
    cov[2, 2] += 1e-9
    return mean, cov


@njit(cache=True)
def _pf_update_logw_numba(
    p: np.ndarray,
    w: np.ndarray,
    pL_mat: np.ndarray,
    vL_mat: np.ndarray,
    vF_meas: np.ndarray,
    s_vals: np.ndarray,
    s_valid: np.ndarray,
    gate_f: np.ndarray,
    gate_count_thr: float,
    gate_min_factor: float,
    sigma2_base: float,
    loglik_clip: float,
    likelihood_temp: float,
    mean: np.ndarray,
) -> Tuple[np.ndarray, int, int, int, float, int, float, float]:
    n = p.shape[0]
    m = pL_mat.shape[0]

    logw = np.empty(n, dtype=np.float64)
    for k in range(n):
        wk = w[k]
        if wk < 1e-300:
            wk = 1e-300
        logw[k] = math.log(wk)

    used_soft = 0
    used_hard = 0
    total = 0
    nis_acc = 0.0
    nis_count = 0
    gate_sum = 0.0
    gate_min = 1e300
    gate_cnt = 0

    for i in range(m):
        if s_valid[i] < 0.5:
            continue
        total += 1
        g = gate_f[i]
        if g < gate_min_factor:
            continue

        used_soft += 1
        gate_sum += g
        gate_cnt += 1
        if g < gate_min:
            gate_min = g
        if g >= gate_count_thr:
            used_hard += 1

        gg = g
        if gg < 1e-3:
            gg = 1e-3
        sigma2_eff = sigma2_base / gg
        inv_sigma2_eff = 1.0 / sigma2_eff
        log_norm = math.log(2.0 * math.pi * sigma2_eff)

        vr0 = vL_mat[i, 0] - vF_meas[0]
        vr1 = vL_mat[i, 1] - vF_meas[1]
        vr2 = vL_mat[i, 2] - vF_meas[2]
        s = s_vals[i]

        for k in range(n):
            rx = pL_mat[i, 0] - p[k, 0]
            ry = pL_mat[i, 1] - p[k, 1]
            rz = pL_mat[i, 2] - p[k, 2]
            rho = math.sqrt(rx * rx + ry * ry + rz * rz)
            if rho < 1e-9:
                rho = 1e-9
            rhatx = rx / rho
            rhaty = ry / rho
            rhatz = rz / rho
            h = -(rhatx * vr0 + rhaty * vr1 + rhatz * vr2)
            y = s - h
            ll = -0.5 * (log_norm + (y * y) * inv_sigma2_eff)
            if ll > loglik_clip:
                ll = loglik_clip
            elif ll < -loglik_clip:
                ll = -loglik_clip
            logw[k] += ll / likelihood_temp

        rxm = pL_mat[i, 0] - mean[0]
        rym = pL_mat[i, 1] - mean[1]
        rzm = pL_mat[i, 2] - mean[2]
        rho_m = math.sqrt(rxm * rxm + rym * rym + rzm * rzm)
        if rho_m > 1e-6:
            rhatmx = rxm / rho_m
            rhatmy = rym / rho_m
            rhatmz = rzm / rho_m
            hm = -(rhatmx * vr0 + rhatmy * vr1 + rhatmz * vr2)
            ym = s - hm
            nis_acc += (ym * ym) * inv_sigma2_eff
            nis_count += 1

    gate_avg = gate_sum / gate_cnt if gate_cnt > 0 else np.nan
    gate_min_out = gate_min if gate_cnt > 0 else np.nan
    return logw, used_soft, used_hard, total, nis_acc, nis_count, gate_avg, gate_min_out


def warmup_numba_pf() -> None:
    if not _NUMBA_AVAILABLE:
        return
    p = np.zeros((32, 3), dtype=np.float64)
    w = np.ones(32, dtype=np.float64) / 32.0
    _pf_recompute_stats_numba(p, w)
    pL = np.zeros((2, 3), dtype=np.float64)
    vL = np.zeros((2, 3), dtype=np.float64)
    vF = np.zeros(3, dtype=np.float64)
    s_vals = np.zeros(2, dtype=np.float64)
    s_valid = np.array([1.0, 1.0], dtype=np.float64)
    gate_f = np.ones(2, dtype=np.float64)
    _pf_update_logw_numba(
        p, w, pL, vL, vF, s_vals, s_valid, gate_f,
        0.5, 0.02, 1e-4, 80.0, 1.0, np.zeros(3, dtype=np.float64),
    )

# =============================================================================
# FIM / CRLB tracker
# =============================================================================

class FIMTracker3D:
    def __init__(self, window_s: Optional[float], reg_eps: float = 1e-9):
        self.window_s = None if window_s is None else float(max(0.0, window_s))
        self.reg_eps = float(max(0.0, reg_eps))
        self.I_total = np.zeros((3, 3), dtype=float)
        self.I_win = np.zeros((3, 3), dtype=float)
        self._events: List[Tuple[float, np.ndarray]] = []

    def reset(self) -> None:
        self.I_total[:] = 0.0
        self.I_win[:] = 0.0
        self._events = []

    def add_I(self, t_meas: float, I_inc: np.ndarray) -> None:
        t_meas = float(t_meas)
        I_inc = np.asarray(I_inc, dtype=float).reshape(3, 3)
        self.I_total += I_inc

        if self.window_s is None or self.window_s <= 0.0:
            self.I_win = self.I_total.copy()
            return

        self._events.append((t_meas, I_inc))
        self.I_win += I_inc
        t_min = t_meas - self.window_s
        while self._events and self._events[0][0] < t_min:
            _, I_old = self._events.pop(0)
            self.I_win -= I_old
        self.I_win = 0.5 * (self.I_win + self.I_win.T)

    def crlb(self, use_window: bool) -> np.ndarray:
        I = self.I_win if use_window else self.I_total
        return safe_inv_sym(I, eps=self.reg_eps)

    @staticmethod
    def eig_stats(I: np.ndarray) -> Tuple[float, float, float]:
        I = 0.5 * (I + I.T)
        try:
            eig = np.linalg.eigvalsh(I)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), float("nan")
        eig = np.maximum(eig, 0.0)
        eig_min = float(np.min(eig))
        eig_max = float(np.max(eig))
        if eig_min <= 1e-18:
            cond = float("inf") if eig_max > 0 else float("nan")
        else:
            cond = float(eig_max / eig_min)
        return eig_min, eig_max, cond


# =============================================================================
# Particle filter
# =============================================================================

@dataclass
class PFStats:
    used_meas: int = 0
    used_meas_soft: int = 0
    meas_total: int = 0
    gate_avg: float = float("nan")
    gate_min: float = float("nan")
    ess: float = 0.0
    w_max: float = 0.0
    resampled: int = 0
    nis: float = float("nan")
    nis_ratio: float = float("nan")
    consistency_infl: float = 1.0
    sigma_nis_mult: float = 1.0
    injected: int = 0


class ParticleFilter3D:
    def __init__(
        self,
        num_particles: int,
        pos_std0: float,
        process_std: float,
        meas_sigma: float,
        rng: np.random.Generator,
        *,
        ess_frac_resample: float = 0.5,
        jitter_std: float = 2.0,
        roughen_k: float = 0.0,
        roughen_max_std: float = 25.0,
        likelihood_temp: float = 1.0,
        loglik_clip: float = 80.0,
        nis_ratio_ref: float = 1.0,
        nis_infl_gain: float = 0.5,
        nis_infl_cap: float = 3.0,
        nis_sigma_adapt_alpha: float = 0.08,
        nis_sigma_adapt_decay: float = 0.02,
        nis_sigma_adapt_min: float = 1.0,
        nis_sigma_adapt_max: float = 3.0,
        weight_floor: float = 1e-12,
        use_numba: bool = True,
    ) -> None:
        self.N = int(max(16, num_particles))
        self.pos_std0 = float(max(1e-6, pos_std0))
        self.process_std = float(max(0.0, process_std))
        self.meas_sigma = float(max(1e-9, meas_sigma))
        self.rng = rng

        self.ess_frac_resample = float(clamp(ess_frac_resample, 0.01, 0.99))
        self.jitter_std = float(max(0.0, jitter_std))
        self.roughen_k = float(max(0.0, roughen_k))
        self.roughen_max_std = float(max(0.0, roughen_max_std))
        self.likelihood_temp = float(max(1e-6, likelihood_temp))
        self.loglik_clip = float(max(1.0, loglik_clip))
        self.nis_ratio_ref = float(max(1e-6, nis_ratio_ref))
        self.nis_infl_gain = float(max(0.0, nis_infl_gain))
        self.nis_infl_cap = float(max(1.0, nis_infl_cap))
        self.nis_sigma_adapt_alpha = float(clamp(nis_sigma_adapt_alpha, 0.0, 1.0))
        self.nis_sigma_adapt_decay = float(clamp(nis_sigma_adapt_decay, 0.0, 1.0))
        self.nis_sigma_adapt_min = float(max(1.0, nis_sigma_adapt_min))
        self.nis_sigma_adapt_max = float(max(self.nis_sigma_adapt_min, nis_sigma_adapt_max))
        self.weight_floor = float(max(0.0, weight_floor))
        self.use_numba = bool(use_numba and _NUMBA_AVAILABLE)

        self.p = np.zeros((self.N, 3), dtype=float)
        self.w = np.ones(self.N, dtype=float) / self.N
        self.mean = np.zeros(3, dtype=float)
        self.cov = np.eye(3, dtype=float) * (self.pos_std0 ** 2)
        self.sigma_nis_mult = 1.0

    def reset_gaussian(self, mean: np.ndarray, std: float) -> None:
        mean = np.asarray(mean, dtype=float).reshape(3)
        std = float(max(1e-9, std))
        self.p = mean[None, :] + self.rng.normal(0.0, std, size=(self.N, 3))
        self.w[:] = 1.0 / self.N
        self._recompute_stats()

    def reset_sphere_shell_band(self, center: np.ndarray, rho_min: float, rho_max: float, *, cos_phi_max: float = 1.0) -> None:
        c = np.asarray(center, dtype=float).reshape(3)
        rho_min = float(max(0.0, rho_min))
        rho_max = float(max(rho_min + 1e-6, rho_max))
        cos_phi_max = float(clamp(cos_phi_max, 0.0, 1.0))

        theta = 2.0 * math.pi * self.rng.random(self.N)
        cos_phi = self.rng.uniform(-cos_phi_max, cos_phi_max, size=self.N)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        sin_phi = np.sqrt(np.maximum(1.0 - cos_phi ** 2, 0.0))
        dirs = np.stack([
            np.cos(theta) * sin_phi,
            np.sin(theta) * sin_phi,
            cos_phi,
        ], axis=1)

        ur = self.rng.random(self.N)
        a3 = rho_min ** 3
        b3 = rho_max ** 3
        r = (a3 + ur * (b3 - a3)) ** (1.0 / 3.0)
        self.p = c[None, :] + dirs * r[:, None]
        self.w[:] = 1.0 / self.N
        self._recompute_stats()

    def predict(self, vF_meas: np.ndarray, dt: float) -> None:
        vF_meas = np.asarray(vF_meas, dtype=float).reshape(3)
        dt = float(max(1e-6, dt))
        self.p += dt * vF_meas[None, :]
        if self.process_std > 0.0:
            q = self.process_std * math.sqrt(dt)
            self.p += self.rng.normal(0.0, q, size=self.p.shape)
        self._recompute_stats()

    def update_doppler(
        self,
        pL_list: List[np.ndarray],
        vL_list: List[np.ndarray],
        vF_meas: np.ndarray,
        s_meas_list: List[Optional[float]],
        *,
        gate_factors: Optional[List[float]] = None,
        gate_mask: Optional[List[bool]] = None,
        gate_count_thr: float = 0.5,
        gate_min_factor: float = 0.02,
    ) -> PFStats:
        vF_meas = np.asarray(vF_meas, dtype=float).reshape(3)
        assert len(pL_list) == len(vL_list) == len(s_meas_list)
        m = len(pL_list)

        if gate_factors is None:
            if gate_mask is not None:
                gate_factors = [1.0 if bool(b) else 0.0 for b in gate_mask]
            else:
                gate_factors = [1.0] * m
        if len(gate_factors) != m:
            raise ValueError("gate_factors must have the same length as pL_list")

        gate_f = [clamp(float(g), 0.0, 1.0) for g in gate_factors]
        gate_count_thr = float(clamp(gate_count_thr, 0.0, 1.0))
        gate_min_factor = float(max(0.0, gate_min_factor))

        used_soft = 0
        used_hard = 0
        total = 0
        gate_used_vals: List[float] = []
        logw = np.log(np.maximum(self.w, 1e-300))
        nis_acc = 0.0
        nis_count = 0
        sigma2_base = float(max((self.meas_sigma * self.sigma_nis_mult) ** 2, 1e-18))
        gate_avg = float("nan")
        gate_min = float("nan")

        if self.use_numba:
            pL_mat = np.asarray(pL_list, dtype=np.float64).reshape(m, 3)
            vL_mat = np.asarray(vL_list, dtype=np.float64).reshape(m, 3)
            s_vals = np.zeros(m, dtype=np.float64)
            s_valid = np.zeros(m, dtype=np.float64)
            for i in range(m):
                si = s_meas_list[i]
                if si is None:
                    continue
                s_vals[i] = float(si)
                s_valid[i] = 1.0
            gate_arr = np.asarray(gate_f, dtype=np.float64).reshape(m)
            (
                logw,
                used_soft,
                used_hard,
                total,
                nis_acc,
                nis_count,
                gate_avg,
                gate_min,
            ) = _pf_update_logw_numba(
                self.p,
                self.w,
                pL_mat,
                vL_mat,
                vF_meas.astype(np.float64),
                s_vals,
                s_valid,
                gate_arr,
                float(gate_count_thr),
                float(gate_min_factor),
                float(sigma2_base),
                float(self.loglik_clip),
                float(self.likelihood_temp),
                self.mean.astype(np.float64),
            )
        else:
            for i in range(m):
                s = s_meas_list[i]
                if s is None:
                    continue
                total += 1
                g = float(gate_f[i])
                if g < gate_min_factor:
                    continue
                used_soft += 1
                gate_used_vals.append(g)
                if g >= gate_count_thr:
                    used_hard += 1
                sigma2_eff = sigma2_base / max(g, 1e-3)
                inv_sigma2_eff = 1.0 / max(sigma2_eff, 1e-18)
                pL = np.asarray(pL_list[i], dtype=float).reshape(3)
                vL = np.asarray(vL_list[i], dtype=float).reshape(3)
                v_rel = vL - vF_meas
                r = pL[None, :] - self.p
                rho = np.linalg.norm(r, axis=1)
                rho_safe = np.maximum(rho, 1e-9)
                rhat = r / rho_safe[:, None]
                h = -np.einsum("ij,j->i", rhat, v_rel)
                y = float(s) - h
                ll = -0.5 * (math.log(2.0 * math.pi * sigma2_eff) + (y * y) * inv_sigma2_eff)
                ll = np.clip(ll, -self.loglik_clip, self.loglik_clip)
                logw += ll / self.likelihood_temp
                r_mean = pL - self.mean
                rho_m = float(np.linalg.norm(r_mean))
                if rho_m > 1e-6:
                    rhat_m = r_mean / rho_m
                    h_m = -float(np.dot(rhat_m, v_rel))
                    y_m = float(s) - h_m
                    nis_acc += (y_m * y_m) * inv_sigma2_eff
                    nis_count += 1
            gate_avg = float(np.mean(gate_used_vals)) if gate_used_vals else float("nan")
            gate_min = float(np.min(gate_used_vals)) if gate_used_vals else float("nan")

        logw -= float(np.max(logw))
        w_new = np.exp(logw)
        ssum = float(np.sum(w_new))
        if (not np.isfinite(ssum)) or ssum <= 1e-300:
            self.w[:] = 1.0 / self.N
        else:
            w_new /= ssum
            if self.weight_floor > 0.0:
                w_new = np.maximum(w_new, self.weight_floor)
                w_new /= float(np.sum(w_new))
            self.w = w_new

        ess = float(1.0 / np.sum(np.square(self.w)))
        w_max = float(np.max(self.w))

        resampled = 0
        if ess < self.ess_frac_resample * self.N:
            idx = systematic_resample(self.w, self.rng)
            self.p = self.p[idx, :]
            self.w[:] = 1.0 / self.N
            resampled = 1
            if self.jitter_std > 0.0:
                self.p += self.rng.normal(0.0, self.jitter_std, size=self.p.shape)
            self._roughen()

        self._recompute_stats()

        nis_sum = float(nis_acc) if nis_count > 0 else float("nan")
        nis_ratio = float("nan")
        consistency_infl = 1.0
        if np.isfinite(nis_sum):
            expect = float(self.nis_ratio_ref) * float(max(int(nis_count), 1))
            nis_ratio = float(nis_sum / max(expect, 1e-12))
            if nis_ratio > 1.0:
                target_mult = self.sigma_nis_mult * math.sqrt(nis_ratio)
                target_mult = clamp(target_mult, self.nis_sigma_adapt_min, self.nis_sigma_adapt_max)
                a = self.nis_sigma_adapt_alpha
                self.sigma_nis_mult = (1.0 - a) * self.sigma_nis_mult + a * float(target_mult)
            else:
                b = self.nis_sigma_adapt_decay
                self.sigma_nis_mult = (1.0 - b) * self.sigma_nis_mult + b * 1.0
            self.sigma_nis_mult = clamp(self.sigma_nis_mult, self.nis_sigma_adapt_min, self.nis_sigma_adapt_max)
            if nis_ratio > 1.0 and self.nis_infl_gain > 0.0:
                consistency_infl = float(1.0 + self.nis_infl_gain * (nis_ratio - 1.0))
                consistency_infl = min(consistency_infl, self.nis_infl_cap)
            if consistency_infl > 1.0001:
                mean = np.asarray(self.mean, dtype=float).reshape(1, 3)
                self.p = mean + (self.p - mean) * math.sqrt(consistency_infl)
                self._recompute_stats()

        return PFStats(
            used_meas=int(used_hard),
            used_meas_soft=int(used_soft),
            meas_total=int(total),
            gate_avg=float(gate_avg),
            gate_min=float(gate_min),
            ess=float(ess),
            w_max=float(w_max),
            resampled=int(resampled),
            nis=float(nis_sum),
            nis_ratio=float(nis_ratio),
            consistency_infl=float(consistency_infl),
            sigma_nis_mult=float(self.sigma_nis_mult),
        )

    def _roughen(self) -> None:
        if self.roughen_k <= 0.0:
            return
        d = 3.0
        n_pow = float(max(self.N, 1)) ** (-1.0 / d)
        ranges = np.ptp(self.p, axis=0)
        sigma = self.roughen_k * ranges * n_pow
        sigma = np.clip(sigma, 0.0, self.roughen_max_std)
        if float(np.max(sigma)) <= 0.0:
            return
        self.p += self.rng.normal(0.0, sigma[None, :], size=self.p.shape)

    def inject_sphere_shell_band(
        self,
        center: np.ndarray,
        rho_min: float,
        rho_max: float,
        cos_phi_max: float,
        frac: float,
        mass: float = 0.02,
    ) -> int:
        frac = float(np.clip(frac, 0.0, 1.0))
        if frac <= 0.0:
            return 0
        n_inj = int(round(frac * self.N))
        if n_inj <= 0:
            return 0
        center = np.asarray(center, dtype=np.float64).reshape(3)
        rho_min = float(max(1e-6, rho_min))
        rho_max = float(max(rho_min * 1.001, rho_max))
        cos_phi_max = float(np.clip(cos_phi_max, 0.0, 1.0))

        idx = self.rng.choice(self.N, size=n_inj, replace=False)
        theta = 2.0 * np.pi * self.rng.random(n_inj)
        cos_phi = self.rng.uniform(-cos_phi_max, cos_phi_max, size=n_inj)
        sin_phi = np.sqrt(np.clip(1.0 - cos_phi * cos_phi, 0.0, 1.0))
        dirs = np.stack([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi], axis=1)
        a3 = rho_min ** 3
        b3 = rho_max ** 3
        ur = self.rng.random(n_inj)
        r = (a3 + ur * (b3 - a3)) ** (1.0 / 3.0)
        self.p[idx, :] = center[None, :] + dirs * r[:, None]

        mass = float(np.clip(mass, 0.0, 0.5))
        if mass > 0.0:
            w = self.w.copy()
            w[idx] = 0.0
            s_rest = float(np.sum(w))
            if s_rest <= 1e-18:
                w[:] = (1.0 - mass) / max(1, (self.N - n_inj))
            else:
                w *= (1.0 - mass) / s_rest
            w[idx] = mass / float(n_inj)
            self.w = w
        else:
            s = float(np.sum(self.w))
            if s > 1e-18:
                self.w /= s

        self._recompute_stats()
        return int(n_inj)

    def _recompute_stats(self) -> None:
        if self.use_numba:
            mean, cov = _pf_recompute_stats_numba(self.p, self.w)
            self.mean = mean
            self.cov = cov
            return
        w = self.w
        self.mean = np.sum(self.p * w[:, None], axis=0)
        dp = self.p - self.mean[None, :]
        cov = (dp * w[:, None]).T @ dp
        cov = 0.5 * (cov + cov.T)
        cov += 1e-9 * np.eye(3)
        self.cov = cov

    def stds(self) -> Tuple[float, float, float]:
        return (
            float(math.sqrt(max(self.cov[0, 0], 0.0))),
            float(math.sqrt(max(self.cov[1, 1], 0.0))),
            float(math.sqrt(max(self.cov[2, 2], 0.0))),
        )

    def std_max(self) -> float:
        try:
            eig = np.linalg.eigvalsh(self.cov)
            return float(math.sqrt(max(np.max(eig), 0.0)))
        except np.linalg.LinAlgError:
            sx, sy, sz = self.stds()
            return max(sx, sy, sz)

# =============================================================================
# Config
# =============================================================================

@dataclass
class UUV3DBaseConfig:
    screen_w: int = 1200
    screen_h: int = 800
    render_fps: int = 30

    f_min_speed: float = 0.2
    f_max_speed: float = 5.0
    max_yaw_rate_deg_s: float = 60.0
    max_pitch_rate_deg_s: float = 45.0
    pitch_min_deg: float = -45.0
    pitch_max_deg: float = 45.0

    rl_yaw_per_step_deg: float = 20.0
    rl_pitch_per_step_deg: float = 14.0
    rl_speed_delta_per_step: float = 0.4

    manual_yaw_per_step_deg: float = 7.0
    manual_pitch_per_step_deg: float = 6.0
    manual_speed_delta_per_step: float = 0.08
    manual_fine_scale: float = 0.25

    leader_speed_min: float = 1.0
    leader_speed_max: float = 3.0
    leader_yaw_min_deg: float = 0.0
    leader_yaw_max_deg: float = 360.0
    leader1_depth_range: Tuple[float, float] = (-40.0, -20.0)
    leader2_depth_range: Tuple[float, float] = (-90.0, -60.0)
    leaders_sep_y: float = 100.0

    start_rho_min: float = 120.0
    start_rho_max: float = 350.0
    init_dir_band_deg_easy: float = 25.0

    d_back: float = 120.0
    d_right: float = 0.0
    dz_offset: float = 0.0

    action_dt: float = 2.0
    sub_dt: float = 0.1
    max_steps: int = 220

    sigma_s_true: float = 0.02
    s_meas_period: float = 1.0
    sigma_speed: float = 0.02
    sigma_yaw_deg: float = 0.8
    sigma_pitch_deg: float = 0.8

    pf_num_particles: int = 1024
    pf_init_pos_std: float = 200.0
    pf_process_std: float = 0.20
    pf_ess_frac_resample: float = 0.45
    pf_jitter_std_easy: float = 3.0
    pf_jitter_std_hard: float = 2.0
    pf_likelihood_temp_easy: float = 1.6
    pf_likelihood_temp_hard: float = 1.2
    pf_loglik_clip: float = 80.0
    pf_nis_ratio_ref: float = 1.0
    pf_nis_infl_gain: float = 0.60
    pf_nis_infl_cap: float = 3.0
    pf_nis_sigma_adapt_alpha: float = 0.08
    pf_nis_sigma_adapt_decay: float = 0.02
    pf_nis_sigma_adapt_min: float = 1.0
    pf_nis_sigma_adapt_max: float = 3.0
    pf_meas_sigma_mult_easy: float = 3.0
    pf_meas_sigma_mult_hard: float = 2.5
    pf_roughen_k_easy: float = 0.12
    pf_roughen_k_hard: float = 0.08
    pf_roughen_max_std: float = 25.0
    pf_use_numba: bool = True

    pf_inject_frac_easy: float = 0.05
    pf_inject_frac_hard: float = 0.02
    pf_inject_mass: float = 0.02
    pf_inject_rho_min: float = 20.0
    pf_inject_rho_max_mult: float = 2.0

    success_mode: str = "true"
    success_require_std: bool = True
    tol_pos_true_easy: float = 30.0
    tol_pos_true_hard: float = 20.0

    use_conservative_std: bool = True
    cons_use_crlb_floor: bool = True
    cons_crlb_mult: float = 1.0
    cons_use_ess_inflation: bool = True
    cons_ess_thr_frac: float = 0.40
    cons_ess_k: float = 2.0
    cons_std_cap: float = 500.0

    doppler_rho_min: float = 30.0
    doppler_v_min: float = 0.05
    doppler_v_perp_min_easy: float = 0.05
    doppler_v_perp_min_hard: float = 0.10
    gate_band_rho: float = 30.0
    gate_band_v: float = 0.20
    gate_band_v_perp: float = 1.00
    gate_min_factor: float = 0.02
    gate_count_thr: float = 0.50
    gate_relax_difficulty_thr: float = 0.70
    gate_relax_meas_used_thr: int = 2
    gate_relax_gateavg_thr: float = 0.60
    gate_relax_v_perp_min_hard: float = 0.08
    gate_relax_band_v_perp: float = 1.20

    tol_pos_est_hard: float = 8.0
    tol_std_hard: float = 7.0
    tol_pos_est_easy: float = 20.0
    tol_std_easy: float = 20.0
    success_hold_steps: int = 3

    w_pos: float = 2.0
    w_info: float = 1.0
    w_sens: float = 0.40
    w_nis: float = 0.35
    w_close: float = 0.4
    w_stall: float = 0.20
    w_worst: float = 0.30
    stdmax_red0_m: float = 0.25
    w_fim_min: float = 0.65
    w_fim_trace: float = 0.20
    fim_min0: float = 1.0
    fim_trace0: float = 8.0
    close_goal_pos_mult: float = 2.5
    info_sens_boost: float = 3.0
    unc_gap0_m: float = 2.0
    obs_boost_k: float = 2.0
    pos_relax_k: float = 1.5
    w_time: float = 0.05
    w_prog: float = 0.60
    pos_prog0_m: float = 2.0
    obs_boost: bool = True
    obs_gate_pos_mult: float = 1.5
    obs_gate_band_mult: float = 0.6
    obs_gate_need_info_only: bool = True
    obs_gate_start_difficulty: float = 0.65
    obs_gate_ramp_difficulty: float = 0.20

    # Temporal observation memory (frame stack of compact state features).
    # This addresses partial observability of Doppler-only 3D localization.
    obs_history_len: int = 4

    # In hard curriculum regime, do not let information-seeking rewards vanish completely.
    # A floor on the information gate keeps pressure on geometry / observability.
    info_gate_floor_hard: float = 0.30
    info_gate_floor_hard_extra: float = 0.25
    info_gate_floor_start_difficulty: float = 0.55
    info_gate_floor_ramp_difficulty: float = 0.25
    info_gate_floor_meas_used_frac_thr: float = 0.75
    info_gate_floor_gateavg_thr: float = 0.65
    w_std: float = 1.0
    w_fim: float = 0.65
    w_crlb: float = 0.20
    w_energy: float = 0.10
    terminal_bonus: float = 40.0
    rew_clip: float = 50.0
    sens_norm: float = 0.6

    rho_soft_min: float = 20.0
    nis_95: float = 5.99
    nis_99: float = 9.21

    pos_scale: float = 300.0
    std_scale: float = 200.0
    vel_scale: float = 4.0
    s_scale: float = 3.0
    fim_window_s: float = 30.0
    fim_reg_eps: float = 1e-9
    fim_use_gating: bool = True

    difficulty_fixed: Optional[float] = None
    curriculum_steps: int = 30000
    log_truth_diagnostics: bool = False


# =============================================================================
# Camera
# =============================================================================

class Camera2D:
    def __init__(self, w: int, h: int, scale_init: float = 2.0, scale_min: float = 0.3, scale_max: float = 8.0):
        self.w = int(w)
        self.h = int(h)
        self.center = np.zeros(2, dtype=float)
        self.scale = float(scale_init)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    def update(self, points_xy: List[np.ndarray]) -> None:
        pts = np.vstack(points_xy)
        c = pts.mean(axis=0)
        d = np.linalg.norm(pts - c[None, :], axis=1)
        dmax = float(max(d.max(), 10.0))
        target_px = 0.40 * min(self.w, self.h)
        target_scale = target_px / dmax
        target_scale = clamp(target_scale, self.scale_min, self.scale_max)
        alpha = 0.15
        self.center = (1 - alpha) * self.center + alpha * c
        self.scale = (1 - alpha) * self.scale + alpha * target_scale

    def world_to_screen(self, p_xy: np.ndarray) -> Tuple[int, int]:
        p_xy = np.asarray(p_xy, dtype=float).reshape(2)
        rel = p_xy - self.center
        sx = self.w / 2 + rel[0] * self.scale
        sy = self.h / 2 - rel[1] * self.scale
        return int(sx), int(sy)


# =============================================================================
# Environment
# =============================================================================

class UUVTwoLeader3DPFBaseEnv(gym.Env):
    """
    Core v7 observation redesign (55 dims before the v8 extension):
      0-2   e_form = R_form (pF_hat - pF_des)
      3-5   rel_form = R_form (pF_hat - pC)
      6-8   conservative stds in formation frame
      9-11  correlations in formation frame
      12    follower speed estimate
      13-14 cos/sin(yaw_F relative to formation forward)
      15    sin(pitch_F)
      16-17 last Dopplers s1,s2
      18-20 rel_L1_form
      21-23 uLOS_L1_form
      24    rho_L1
      25-27 rel_L2_form
      28-30 uLOS_L2_form
      31    rho_L2
      32    centroid speed
      33    ESS fraction
      34    max weight
      35    gate_avg_step
      36    meas_used_frac
      37    sigma_nis_mult norm
      38    NIS-ratio feature
      39    resampled flag
      40    injected fraction
      41    gate_relax_active
      42    difficulty
      43    err_est / tol_pos_est
      44    std_max / tol_std
      45    sensitivity feature
      46    FIM min-eig feature
      47-49 abs dominant uncertainty axis in formation frame
      50-51 current soft gate factors for L1/L2
      52-54 previous normalized action

    Returned observation = frame stack of the last `obs_history_len` base observations.
    This is a standard way to handle partial observability without switching the policy to RNN.
    The public v8 environment appends 9 planner features on top of this base vector.
    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}
    BASE_OBS_DIM = 55

    def __init__(self, cfg: Optional[UUV3DBaseConfig] = None, render_mode: str = "none"):
        super().__init__()
        self.cfg = cfg or UUV3DBaseConfig()
        self.render_mode = str(render_mode)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.obs_history_len = int(max(1, getattr(self.cfg, "obs_history_len", 1)))
        self.obs_dim = int(self.BASE_OBS_DIM * self.obs_history_len)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.rng = np.random.default_rng(0)
        self.total_env_steps = 0
        self._difficulty = 0.0
        self.t = 0.0
        self.step_count = 0
        self.success_streak = 0
        self._episode_progress_success = False
        self._episode_true_success = False

        self.pL1 = np.zeros(3, dtype=float)
        self.pL2 = np.zeros(3, dtype=float)
        self.pF = np.zeros(3, dtype=float)
        self.vL1 = np.zeros(3, dtype=float)
        self.vL2 = np.zeros(3, dtype=float)
        self.vF = np.zeros(3, dtype=float)

        self.leader1_speed = 2.0
        self.leader2_speed = 2.0
        self.yaw_L1 = 0.0
        self.yaw_L2 = 0.0
        self.speed_F = 2.0
        self.yaw_F = 0.0
        self.pitch_F = 0.0

        self.s1_last = 0.0
        self.s2_last = 0.0
        self._has_s1 = False
        self._has_s2 = False
        self.next_s_time = self.cfg.s_meas_period

        self.pf = ParticleFilter3D(
            num_particles=self.cfg.pf_num_particles,
            pos_std0=self.cfg.pf_init_pos_std,
            process_std=self.cfg.pf_process_std,
            meas_sigma=self.cfg.sigma_s_true,
            rng=self.rng,
            ess_frac_resample=self.cfg.pf_ess_frac_resample,
            jitter_std=self.cfg.pf_jitter_std_easy,
            likelihood_temp=self.cfg.pf_likelihood_temp_easy,
            loglik_clip=self.cfg.pf_loglik_clip,
            nis_ratio_ref=float(self.cfg.pf_nis_ratio_ref),
            nis_infl_gain=float(self.cfg.pf_nis_infl_gain),
            nis_infl_cap=float(self.cfg.pf_nis_infl_cap),
            nis_sigma_adapt_alpha=float(self.cfg.pf_nis_sigma_adapt_alpha),
            nis_sigma_adapt_decay=float(self.cfg.pf_nis_sigma_adapt_decay),
            nis_sigma_adapt_min=float(self.cfg.pf_nis_sigma_adapt_min),
            nis_sigma_adapt_max=float(self.cfg.pf_nis_sigma_adapt_max),
            use_numba=bool(self.cfg.pf_use_numba),
        )

        self.prev_unc_metric: Optional[float] = None
        self.prev_err_est: Optional[float] = None
        self.no_update_steps: int = 0
        self._err_est_prev: float = 0.0
        self._unc_metric_prev: float = 0.0
        self.sens_accum: float = 0.0
        self.sens_count: float = 0.0
        self.sens_avg_step: float = 0.0
        self.meas_total_step: int = 0
        self.meas_used_step: int = 0
        self.pf_ess_step: float = float("nan")
        self.pf_wmax_step: float = float("nan")
        self.pf_resampled_step: int = 0
        self.pf_injected_step: int = 0
        self.pf_sigma_nis_mult_step: float = 1.0
        self.pf_nis_ratio_step: float = float("nan")
        self.pf_consistency_infl_step: float = 1.0
        self.std_x_eff_step: float = float("nan")
        self.std_y_eff_step: float = float("nan")
        self.std_z_eff_step: float = float("nan")
        self.std_max_eff_step: float = float("nan")
        self.std_infl_step: float = float("nan")
        self._gate_relax_active: float = 0.0
        self._cos_phi_max_pf: float = 1.0
        self._pf_inject_frac: float = 0.0
        self.nis_step: float = float("nan")
        self.gate_avg_step: float = float("nan")
        self.gate_min_step: float = float("nan")
        self._last_gate_pf_current = [0.0, 0.0]
        self.obs_gate = 1.0
        self._last_action_raw = np.zeros(3, dtype=np.float32)
        self._last_base_obs = np.zeros(self.BASE_OBS_DIM, dtype=np.float32)
        self._obs_hist: deque[np.ndarray] = deque(maxlen=self.obs_history_len)

        self.fim_total = FIMTracker3D(window_s=None, reg_eps=self.cfg.fim_reg_eps)
        self.fim_win = FIMTracker3D(window_s=self.cfg.fim_window_s, reg_eps=self.cfg.fim_reg_eps)
        self.fim_hat_win = FIMTracker3D(window_s=self.cfg.fim_window_s, reg_eps=self.cfg.fim_reg_eps)
        self.fim_hat_total = FIMTracker3D(window_s=None, reg_eps=self.cfg.fim_reg_eps)
        self.fim_step_meas = 0
        self.fim_step_used = 0
        self.fim_hat_step_used = 0

        self.prev_std_max: Optional[float] = None
        self.prev_eigmin_hat: Optional[float] = None
        self._last_terminated = False
        self._last_truncated = False
        self._last_term_reason = "reset"
        self.manual_override = False
        self._last_speed_cmd = 0.0
        self._last_yaw_rate_cmd = 0.0
        self._last_pitch_rate_cmd = 0.0
        self.a_cmd = np.array([0.0, 0.0, 0.0], dtype=float)
        self._request_reset = False
        self._request_quit = False
        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._font = None
        self._cam = Camera2D(self.cfg.screen_w, self.cfg.screen_h)
        self._renderer3d = None

    # ----------------- gym api -----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self.pf.rng = self.rng

        if self.cfg.difficulty_fixed is not None:
            self._difficulty = clamp(float(self.cfg.difficulty_fixed), 0.0, 1.0)
        else:
            self._difficulty = clamp(self.total_env_steps / max(1, int(self.cfg.curriculum_steps)), 0.0, 1.0)

        d_pf = float(self._difficulty)
        self.pf.jitter_std = lerp(float(self.cfg.pf_jitter_std_easy), float(self.cfg.pf_jitter_std_hard), d_pf)
        self.pf.likelihood_temp = lerp(float(self.cfg.pf_likelihood_temp_easy), float(self.cfg.pf_likelihood_temp_hard), d_pf)
        self.pf.meas_sigma = float(self.cfg.sigma_s_true) * lerp(float(self.cfg.pf_meas_sigma_mult_easy), float(self.cfg.pf_meas_sigma_mult_hard), d_pf)
        self.pf.roughen_k = lerp(float(self.cfg.pf_roughen_k_easy), float(self.cfg.pf_roughen_k_hard), d_pf)
        self.pf.roughen_max_std = float(self.cfg.pf_roughen_max_std)
        self._pf_inject_frac = lerp(float(self.cfg.pf_inject_frac_easy), float(self.cfg.pf_inject_frac_hard), d_pf)

        self.t = 0.0
        self.step_count = 0
        self.success_streak = 0
        self._request_reset = False
        self._request_quit = False
        self.fim_total.reset()
        self.fim_win.reset()
        self.fim_hat_win.reset()
        self.fim_hat_total.reset()
        self.fim_step_meas = 0
        self.fim_step_used = 0
        self.fim_hat_step_used = 0
        self._last_terminated = False
        self._last_truncated = False
        self._last_term_reason = "reset"

        d = float(self._difficulty)
        sep_y = float(self.cfg.leaders_sep_y)
        z1 = float(self.rng.uniform(self.cfg.leader1_depth_range[0], self.cfg.leader1_depth_range[1]))
        z2 = float(self.rng.uniform(self.cfg.leader2_depth_range[0], self.cfg.leader2_depth_range[1]))
        self.pL1[:] = np.array([0.0, +0.5 * sep_y, z1], dtype=float)
        self.pL2[:] = np.array([0.0, -0.5 * sep_y, z2], dtype=float)

        self.leader1_speed = float(self.rng.uniform(self.cfg.leader_speed_min, self.cfg.leader_speed_max))
        self.leader2_speed = float(self.rng.uniform(self.cfg.leader_speed_min, self.cfg.leader_speed_max))
        self.yaw_L1 = float(self.rng.uniform(self.cfg.leader_yaw_min_deg, self.cfg.leader_yaw_max_deg))
        yaw2_spread = lerp(5.0, 25.0, d)
        self.yaw_L2 = wrap360(self.yaw_L1 + float(self.rng.normal(0.0, yaw2_spread)))
        self.vL1 = vel_from_speed_yaw_pitch(self.leader1_speed, self.yaw_L1, 0.0)
        self.vL2 = vel_from_speed_yaw_pitch(self.leader2_speed, self.yaw_L2, 0.0)

        pC = 0.5 * (self.pL1 + self.pL2)
        rho0 = float(self.rng.uniform(self.cfg.start_rho_min, self.cfg.start_rho_max))
        theta = 2.0 * math.pi * float(self.rng.random())
        band_deg = lerp(self.cfg.init_dir_band_deg_easy, 90.0, d)
        cos_phi_max = math.sin(deg2rad(band_deg))
        cos_phi = float(self.rng.uniform(-cos_phi_max, cos_phi_max))
        cos_phi = clamp(cos_phi, -1.0, 1.0)
        sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))
        dir0 = np.array([math.cos(theta) * sin_phi, math.sin(theta) * sin_phi, cos_phi], dtype=float)
        self.pF[:] = pC - dir0 * rho0

        self.yaw_F = float(self.rng.uniform(0.0, 360.0))
        self.pitch_F = float(self.rng.uniform(-10.0, 10.0))
        self.speed_F = float(np.clip(2.0 + self.rng.normal(0.0, 0.2), self.cfg.f_min_speed, self.cfg.f_max_speed))
        self.vF = vel_from_speed_yaw_pitch(self.speed_F, self.yaw_F, self.pitch_F)

        rho_min = lerp(60.0, self.cfg.start_rho_min, d)
        rho_max = lerp(140.0, self.cfg.start_rho_max, d)
        cos_phi_max_pf = math.sin(deg2rad(lerp(self.cfg.init_dir_band_deg_easy, 90.0, d)))
        self._cos_phi_max_pf = float(cos_phi_max_pf)
        self.pf.reset_sphere_shell_band(center=pC, rho_min=rho_min, rho_max=rho_max, cos_phi_max=cos_phi_max_pf)

        self.s1_last = 0.0
        self.s2_last = 0.0
        self._has_s1 = False
        self._has_s2 = False
        self.next_s_time = float(self.cfg.s_meas_period)

        stds = self.pf.stds()
        self.prev_unc_metric = float(sum(stds))
        _, pF_des = self._formation_desired()
        self.prev_err_est = float(np.linalg.norm(self.pf.mean - pF_des))
        self._err_est_prev = float(self.prev_err_est)
        self._unc_metric_prev = float(self.prev_unc_metric)
        self.prev_std_max = float(self.pf.std_max())
        self.prev_eigmin_hat = float(self.fim_hat_win.eig_stats(self.fim_hat_win.I_win)[0])
        self.no_update_steps = 0
        self._reset_step_accums()
        self._last_action_raw = np.zeros(3, dtype=np.float32)
        base_obs = self._get_obs_base()
        self._reset_obs_history(base_obs)
        obs = self._stack_obs_history()
        info = self._get_info(extra={"difficulty": float(self._difficulty), "term_reason": "reset"})
        return obs, info

    def step(self, action: np.ndarray):
        if self.render_mode == "human" and self._pygame_inited and pygame is not None:
            self._process_pygame_events()

        if self._request_quit:
            self._request_quit = False
            if len(self._obs_hist) == 0:
                self._reset_obs_history(self._get_obs_base())
            obs = self._stack_obs_history()
            info = self._get_info(extra={"term_reason": "quit"})
            return obs, 0.0, False, True, info

        if self._request_reset:
            self._request_reset = False
            if len(self._obs_hist) == 0:
                self._reset_obs_history(self._get_obs_base())
            obs = self._stack_obs_history()
            info = self._get_info(extra={"term_reason": "manual_reset"})
            return obs, 0.0, False, True, info

        self._reset_step_accums()

        if self.render_mode == "human" and self.manual_override and self._pygame_inited and pygame is not None:
            a = self._keyboard_action()
            a_raw = np.asarray(a[:3], dtype=np.float32).reshape(-1)
            speed_cmd = float(a_raw[0]) * (self.cfg.manual_speed_delta_per_step / max(self.cfg.action_dt, 1e-6))
            yaw_cmd = float(a_raw[1]) * (self.cfg.manual_yaw_per_step_deg / max(self.cfg.action_dt, 1e-6))
            pitch_cmd = float(a_raw[2]) * (self.cfg.manual_pitch_per_step_deg / max(self.cfg.action_dt, 1e-6))
            fine = float(a[3])
            if fine > 0.5:
                speed_cmd *= self.cfg.manual_fine_scale
                yaw_cmd *= self.cfg.manual_fine_scale
                pitch_cmd *= self.cfg.manual_fine_scale
        else:
            a_raw = np.asarray(action, dtype=np.float32).reshape(-1)
            a_raw = np.clip(a_raw, -1.0, 1.0)
            speed_cmd = float(a_raw[0]) * (float(self.cfg.rl_speed_delta_per_step) / max(self.cfg.action_dt, 1e-6))
            yaw_cmd = float(a_raw[1]) * min(self.cfg.max_yaw_rate_deg_s, float(self.cfg.rl_yaw_per_step_deg) / max(self.cfg.action_dt, 1e-6))
            pitch_cmd = float(a_raw[2]) * min(self.cfg.max_pitch_rate_deg_s, float(self.cfg.rl_pitch_per_step_deg) / max(self.cfg.action_dt, 1e-6))

        self._last_action_raw = np.asarray(a_raw, dtype=np.float32).copy()
        self._last_speed_cmd = float(speed_cmd)
        self._last_yaw_rate_cmd = float(yaw_cmd)
        self._last_pitch_rate_cmd = float(pitch_cmd)
        self.a_cmd = np.array([self._last_speed_cmd, self._last_yaw_rate_cmd, self._last_pitch_rate_cmd], dtype=float)

        n_sub = max(1, int(round(self.cfg.action_dt / self.cfg.sub_dt)))
        dt = float(self.cfg.action_dt / n_sub)
        pf_stats_last: Optional[PFStats] = None
        for _ in range(n_sub):
            pf_stats_last = self._sim_substep(speed_cmd, yaw_cmd, pitch_cmd, dt)

        self.step_count += 1
        self.total_env_steps += 1
        self._update_gate_relax_state()

        base_obs = self._get_obs_base()
        self._push_obs_history(base_obs)
        obs = self._stack_obs_history()
        reward, terms = self._compute_reward(pf_stats_last)
        terminated, truncated, reason = self._check_done()
        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)
        self._last_term_reason = str(reason)

        if terminated and reason == "success":
            reward += float(self.cfg.terminal_bonus)
            terms["terminal_bonus"] = float(self.cfg.terminal_bonus)
        else:
            terms["terminal_bonus"] = 0.0

        info = self._get_info(extra=terms | {"term_reason": reason, "manual_override": int(self.manual_override)})
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ----------------- environment internals -----------------

    def _reset_step_accums(self) -> None:
        self.sens_accum = 0.0
        self.sens_count = 0.0
        self.sens_avg_step = 0.0
        self.meas_total_step = 0
        self.meas_used_step = 0
        self.pf_ess_step = float("nan")
        self.pf_wmax_step = float("nan")
        self.pf_resampled_step = 0
        self.pf_injected_step = 0
        self.pf_sigma_nis_mult_step = 1.0
        self.pf_nis_ratio_step = float("nan")
        self.pf_consistency_infl_step = 1.0
        self.nis_step = float("nan")
        self.gate_avg_step = float("nan")
        self.gate_min_step = float("nan")
        self.fim_step_meas = 0
        self.fim_step_used = 0
        self.fim_hat_step_used = 0
        self._gate_relax_active = 0.0
        self._last_gate_pf_current = [0.0, 0.0]

    def _update_gate_relax_state(self) -> None:
        gate_condition = False
        if float(self._difficulty) >= float(self.cfg.gate_relax_difficulty_thr):
            gate_condition = (
                (int(self.meas_used_step) <= int(self.cfg.gate_relax_meas_used_thr))
                and np.isfinite(self.gate_avg_step)
                and (self.gate_avg_step < float(self.cfg.gate_relax_gateavg_thr))
            )
        self._gate_relax_active = 1.0 if gate_condition else 0.0

    def _current_tolerances(self) -> Tuple[float, float, float]:
        d = float(self._difficulty)
        tol_pos_est = lerp(float(self.cfg.tol_pos_est_easy), float(self.cfg.tol_pos_est_hard), d)
        tol_std = lerp(float(self.cfg.tol_std_easy), float(self.cfg.tol_std_hard), d)
        tol_pos_true = lerp(float(self.cfg.tol_pos_true_easy), float(self.cfg.tol_pos_true_hard), d)
        return float(tol_pos_est), float(tol_std), float(tol_pos_true)

    def _formation_axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        f_hat, side_hat, zhat = formation_axes_from_vel(self.vL1, self.vL2)
        R_form = form_rotation_matrix(f_hat, side_hat, zhat)
        return f_hat, side_hat, zhat, R_form

    def _reset_obs_history(self, base_obs: np.ndarray) -> None:
        base_obs = np.asarray(base_obs, dtype=np.float32).reshape(-1)
        self._obs_hist.clear()
        for _ in range(self.obs_history_len):
            self._obs_hist.append(base_obs.copy())
        self._last_base_obs = base_obs.copy()

    def _push_obs_history(self, base_obs: np.ndarray) -> None:
        base_obs = np.asarray(base_obs, dtype=np.float32).reshape(-1)
        if len(self._obs_hist) == 0:
            self._reset_obs_history(base_obs)
            return
        self._obs_hist.append(base_obs.copy())
        self._last_base_obs = base_obs.copy()

    def _stack_obs_history(self) -> np.ndarray:
        if len(self._obs_hist) == 0:
            self._reset_obs_history(np.zeros(self.BASE_OBS_DIM, dtype=np.float32))
        return np.concatenate(list(self._obs_hist), axis=0).astype(np.float32)

    def _compute_conservative_std(self, sx: float, sy: float, sz: float, cov_raw: Optional[np.ndarray] = None) -> Tuple[float, float, float, float, float]:
        sx = float(max(0.0, sx))
        sy = float(max(0.0, sy))
        sz = float(max(0.0, sz))
        std_max_raw = max(sx, sy, sz, 0.0)
        if cov_raw is not None:
            try:
                C = np.asarray(cov_raw, dtype=float).reshape(3, 3)
                C = 0.5 * (C + C.T)
                eig = np.linalg.eigvalsh(C)
                std_max_raw = float(math.sqrt(max(float(np.max(eig)), 0.0)))
            except Exception:
                pass
        if (not bool(self.cfg.use_conservative_std)) or (std_max_raw <= 0.0):
            return sx, sy, sz, std_max_raw, 1.0

        std_max_eff = std_max_raw
        if bool(self.cfg.cons_use_crlb_floor):
            try:
                crlb_hat = self.fim_hat_win.crlb(use_window=True)
                eig = np.linalg.eigvalsh(crlb_hat)
                std_crlb_max = math.sqrt(max(float(eig[-1]), 0.0))
                std_max_eff = max(std_max_eff, float(self.cfg.cons_crlb_mult) * std_crlb_max)
            except Exception:
                pass

        if bool(self.cfg.cons_use_ess_inflation):
            ess = self.pf_ess_step
            if not np.isfinite(ess):
                w = self.pf.w
                s = float(np.sum(w * w))
                ess = (1.0 / s) if s > 1e-18 else float(self.pf.N)
            ess_frac = float(ess) / float(self.pf.N)
            thr = float(self.cfg.cons_ess_thr_frac)
            if thr > 1e-6 and ess_frac < thr:
                t = float(np.clip((thr - ess_frac) / thr, 0.0, 1.0))
                std_max_eff *= (1.0 + float(self.cfg.cons_ess_k) * t)

        std_max_eff = float(min(std_max_eff, float(self.cfg.cons_std_cap)))
        infl = std_max_eff / max(std_max_raw, 1e-6)
        return sx * infl, sy * infl, sz * infl, std_max_eff, infl

    def _formation_desired(self) -> Tuple[np.ndarray, np.ndarray]:
        pC = 0.5 * (self.pL1 + self.pL2)
        vC = 0.5 * (self.vL1 + self.vL2)
        v_xy = np.array([vC[0], vC[1], 0.0], dtype=float)
        f_hat = normalize_vec(v_xy, fallback=np.array([1.0, 0.0, 0.0], dtype=float))
        zhat = np.array([0.0, 0.0, 1.0], dtype=float)
        side_hat = normalize_vec(np.cross(f_hat, zhat), fallback=np.array([0.0, 1.0, 0.0], dtype=float))
        pF_des = pC - float(self.cfg.d_back) * f_hat + float(self.cfg.d_right) * side_hat
        pF_des[2] = float(pC[2] + self.cfg.dz_offset)
        return pC, pF_des

    def _doppler_gate_truth(self, r_true: np.ndarray, v_rel_true: np.ndarray) -> bool:
        rho = float(np.linalg.norm(r_true))
        vnorm = float(np.linalg.norm(v_rel_true))
        if rho <= self.cfg.doppler_rho_min or vnorm <= self.cfg.doppler_v_min:
            return False
        rhat = r_true / max(rho, 1e-9)
        proj = float(np.dot(rhat, v_rel_true))
        v_perp = float(np.linalg.norm(v_rel_true - proj * rhat))
        v_perp_min = lerp(self.cfg.doppler_v_perp_min_easy, self.cfg.doppler_v_perp_min_hard, float(self._difficulty))
        return v_perp > float(v_perp_min)

    def _doppler_gate_pf_factors(self, vF_meas: np.ndarray) -> List[float]:
        vF_meas = np.asarray(vF_meas, dtype=float).reshape(3)
        pF_hat = self.pf.mean
        d = float(self._difficulty)
        v_perp_min = lerp(self.cfg.doppler_v_perp_min_easy, self.cfg.doppler_v_perp_min_hard, d)
        gate_band_v_perp = float(self.cfg.gate_band_v_perp)
        if self._gate_relax_active > 0.5:
            v_perp_min = min(float(v_perp_min), float(self.cfg.gate_relax_v_perp_min_hard))
            gate_band_v_perp = max(gate_band_v_perp, float(self.cfg.gate_relax_band_v_perp))

        factors: List[float] = []
        for pL, vL in ((self.pL1, self.vL1), (self.pL2, self.vL2)):
            r = np.asarray(pL - pF_hat, dtype=float)
            rho = float(np.linalg.norm(r))
            v_rel = np.asarray(vL - vF_meas, dtype=float)
            vnorm = float(np.linalg.norm(v_rel))
            if rho <= 1e-9 or vnorm <= 1e-9:
                factors.append(0.0)
                continue
            rhat = r / max(rho, 1e-9)
            proj = float(np.dot(rhat, v_rel))
            v_perp = float(np.linalg.norm(v_rel - proj * rhat))
            g_rho = soft_gate(rho, float(self.cfg.doppler_rho_min), float(self.cfg.gate_band_rho))
            g_v = soft_gate(vnorm, float(self.cfg.doppler_v_min), float(self.cfg.gate_band_v))
            g_vp = soft_gate(v_perp, float(v_perp_min), float(gate_band_v_perp))
            g = float(g_rho * g_v * g_vp)
            if g < float(self.cfg.gate_min_factor):
                g = 0.0
            factors.append(float(clamp(g, 0.0, 1.0)))
        return factors

    def _accum_sens(self, pL: np.ndarray, vL: np.ndarray, vF: np.ndarray, *, weight: float = 1.0) -> None:
        w = float(max(0.0, weight))
        if w <= 0.0:
            return
        pF_hat = self.pf.mean
        r_hat = np.asarray(pL - pF_hat, dtype=float)
        rho = float(np.linalg.norm(r_hat))
        if rho <= 1e-6:
            return
        rhat = r_hat / rho
        v_rel = np.asarray(vL - vF, dtype=float)
        proj = float(np.dot(rhat, v_rel))
        v_perp = float(np.linalg.norm(v_rel - proj * rhat))
        self.sens_accum += w * v_perp
        self.sens_count += w

    def _sim_substep(self, speed_cmd: float, yaw_rate_cmd: float, pitch_rate_cmd: float, dt: float) -> PFStats:
        dt = float(max(1e-6, dt))
        self.speed_F = float(np.clip(self.speed_F + speed_cmd * dt, self.cfg.f_min_speed, self.cfg.f_max_speed))
        self.yaw_F = wrap360(self.yaw_F + yaw_rate_cmd * dt)
        self.pitch_F = clamp(self.pitch_F + pitch_rate_cmd * dt, self.cfg.pitch_min_deg, self.cfg.pitch_max_deg)
        self.vF = vel_from_speed_yaw_pitch(self.speed_F, self.yaw_F, self.pitch_F)

        self.vL1 = vel_from_speed_yaw_pitch(self.leader1_speed, self.yaw_L1, 0.0)
        self.vL2 = vel_from_speed_yaw_pitch(self.leader2_speed, self.yaw_L2, 0.0)
        self.pL1 = self.pL1 + self.vL1 * dt
        self.pL2 = self.pL2 + self.vL2 * dt
        self.pF = self.pF + self.vF * dt
        self.t += dt

        speed_meas = float(self.speed_F + self.rng.normal(0.0, self.cfg.sigma_speed))
        yaw_meas = float(self.yaw_F + self.rng.normal(0.0, self.cfg.sigma_yaw_deg))
        pitch_meas = float(self.pitch_F + self.rng.normal(0.0, self.cfg.sigma_pitch_deg))
        vF_meas = vel_from_speed_yaw_pitch(speed_meas, yaw_meas, pitch_meas)
        self.pf.predict(vF_meas=vF_meas, dt=dt)

        pf_stats = PFStats()
        while self.t + 1e-12 >= self.next_s_time:
            t_meas = float(self.next_s_time)
            r1_true = self.pL1 - self.pF
            r2_true = self.pL2 - self.pF
            vrel1_true = self.vL1 - self.vF
            vrel2_true = self.vL2 - self.vF
            s1_true = radial_speed(r1_true, vrel1_true)
            s2_true = radial_speed(r2_true, vrel2_true)
            s1 = float(s1_true + self.rng.normal(0.0, self.cfg.sigma_s_true))
            s2 = float(s2_true + self.rng.normal(0.0, self.cfg.sigma_s_true))
            self.s1_last = s1
            self.s2_last = s2
            self._has_s1 = True
            self._has_s2 = True

            gate1 = self._doppler_gate_truth(r1_true, vrel1_true)
            gate2 = self._doppler_gate_truth(r2_true, vrel2_true)
            self.fim_step_meas += 2
            if (not self.cfg.fim_use_gating) or gate1:
                H1 = doppler_H_3d(r1_true, vrel1_true)
                I1 = (H1.T @ H1) / max(self.cfg.sigma_s_true ** 2, 1e-18)
                self.fim_total.add_I(t_meas, I1)
                self.fim_win.add_I(t_meas, I1)
                self.fim_step_used += 1
            if (not self.cfg.fim_use_gating) or gate2:
                H2 = doppler_H_3d(r2_true, vrel2_true)
                I2 = (H2.T @ H2) / max(self.cfg.sigma_s_true ** 2, 1e-18)
                self.fim_total.add_I(t_meas, I2)
                self.fim_win.add_I(t_meas, I2)
                self.fim_step_used += 1

            gate_factors = self._doppler_gate_pf_factors(vF_meas=vF_meas)
            self._last_gate_pf_current = [float(gate_factors[0]), float(gate_factors[1])]

            pF_hat_pred = self.pf.mean
            sigma_hat = float(self.pf.meas_sigma * getattr(self.pf, "sigma_nis_mult", 1.0))
            sigma2_hat = max(sigma_hat * sigma_hat, 1e-18)
            for i, (pL, vL) in enumerate(((self.pL1, self.vL1), (self.pL2, self.vL2))):
                g = float(gate_factors[i]) if i < len(gate_factors) else 0.0
                if g <= 0.0:
                    continue
                r_hat = np.asarray(pL - pF_hat_pred, dtype=float)
                v_rel_hat = np.asarray(vL - vF_meas, dtype=float)
                H_hat = doppler_H_3d(r_hat, v_rel_hat)
                I_inc = (H_hat.T @ H_hat) / sigma2_hat
                self.fim_hat_total.add_I(t_meas, g * I_inc)
                self.fim_hat_win.add_I(t_meas, g * I_inc)
                if g >= float(self.cfg.gate_count_thr):
                    self.fim_hat_step_used += 1

            pf_stats = self.pf.update_doppler(
                pL_list=[self.pL1, self.pL2],
                vL_list=[self.vL1, self.vL2],
                vF_meas=vF_meas,
                s_meas_list=[s1, s2],
                gate_factors=gate_factors,
                gate_count_thr=float(self.cfg.gate_count_thr),
                gate_min_factor=float(self.cfg.gate_min_factor),
            )

            injected = 0
            if (self._pf_inject_frac > 0.0) and (int(pf_stats.resampled) == 1):
                rho_max_inj = float(self.cfg.start_rho_max) * float(self.cfg.pf_inject_rho_max_mult)
                pC = 0.5 * (self.pL1 + self.pL2)
                injected = self.pf.inject_sphere_shell_band(
                    center=pC,
                    rho_min=float(self.cfg.pf_inject_rho_min),
                    rho_max=rho_max_inj,
                    cos_phi_max=float(self._cos_phi_max_pf),
                    frac=float(self._pf_inject_frac),
                    mass=float(self.cfg.pf_inject_mass),
                )
            pf_stats.injected = int(injected)
            self.pf_injected_step += int(injected)

            self.meas_total_step += int(pf_stats.meas_total)
            self.meas_used_step += int(pf_stats.used_meas)
            if len(gate_factors) > 0 and gate_factors[0] > 0.0:
                self._accum_sens(pL=self.pL1, vL=self.vL1, vF=vF_meas, weight=float(gate_factors[0]))
            if len(gate_factors) > 1 and gate_factors[1] > 0.0:
                self._accum_sens(pL=self.pL2, vL=self.vL2, vF=vF_meas, weight=float(gate_factors[1]))

            self.pf_ess_step = float(pf_stats.ess)
            self.pf_wmax_step = float(pf_stats.w_max)
            self.pf_resampled_step = int(pf_stats.resampled)
            self.pf_nis_ratio_step = float(pf_stats.nis_ratio) if np.isfinite(pf_stats.nis_ratio) else float("nan")
            self.pf_consistency_infl_step = float(getattr(pf_stats, "consistency_infl", 1.0))
            self.pf_sigma_nis_mult_step = float(getattr(pf_stats, "sigma_nis_mult", 1.0))
            self.nis_step = float(pf_stats.nis)
            self.gate_avg_step = float(pf_stats.gate_avg) if np.isfinite(pf_stats.gate_avg) else float("nan")
            self.gate_min_step = float(pf_stats.gate_min) if np.isfinite(pf_stats.gate_min) else float("nan")
            self.next_s_time += float(self.cfg.s_meas_period)

        return pf_stats

    def _compute_reward(self, pf_stats_last: Optional[PFStats]) -> Tuple[float, Dict[str, float]]:
        cfg = self.cfg
        _, pF_des = self._formation_desired()
        tol_pos_est, tol_std, tol_pos_true = self._current_tolerances()
        pF_hat = self.pf.mean
        err_est = float(np.linalg.norm(pF_hat - pF_des))

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

        unc_metric = float(sx + sy + sz)
        prog = float(self._err_est_prev - err_est)
        self._err_est_prev = float(err_est)
        r_prog = 2.0 * sat_ratio(prog, 10.0)
        unc_reduction_pos = float(self._unc_metric_prev - unc_metric)
        self._unc_metric_prev = float(unc_metric)
        r_info = sat_ratio(unc_reduction_pos, 5.0)

        unc_gap = max(0.0, std_max - tol_std)
        unc_need = sat_ratio(unc_gap, max(float(cfg.unc_gap0_m), 1e-6))
        pos_penalty_scale = 1.0 / (1.0 + float(cfg.pos_relax_k) * unc_need)

        r_pos = -err_est / 40.0
        r_worst = -0.2 * sat_ratio(max(0.0, err_est - tol_pos_est), 100.0)
        r_std = -sat_ratio(std_max, 35.0)

        I_hat_win = self.fim_hat_win.I_win
        fim_hat_eig_min, fim_hat_eig_max, fim_hat_cond = self.fim_hat_win.eig_stats(I_hat_win)
        fim_hat_trace = float(np.trace(I_hat_win))
        crlb_hat = self.fim_hat_win.crlb(use_window=True)
        crlb_hat_trace = float(np.trace(crlb_hat))
        sens_avg = float(self.sens_accum / max(self.sens_count, 1.0))
        self.sens_avg_step = float(sens_avg)
        r_sens = sat_ratio(sens_avg, float(cfg.sens_norm))
        r_fimmin = sat_ratio(math.log1p(max(0.0, fim_hat_eig_min)), math.log1p(10.0))
        r_crlb = -sat_ratio(math.log1p(max(crlb_hat_trace, 0.0)), math.log1p(150.0))
        r_energy = -sat_ratio(float(np.linalg.norm(self.a_cmd)), 1.0)
        obs_gate = float(self.obs_gate)

        meas_used_frac = float(self.meas_used_step / max(self.meas_total_step, 1)) if self.meas_total_step > 0 else 0.0
        gate_avg = float(self.gate_avg_step) if np.isfinite(self.gate_avg_step) else 0.0
        hard_regime_t = smoothstep01((float(self._difficulty) - float(cfg.info_gate_floor_start_difficulty)) / max(float(cfg.info_gate_floor_ramp_difficulty), 1e-6))
        weak_meas_t = smoothstep01((float(cfg.info_gate_floor_meas_used_frac_thr) - meas_used_frac) / max(float(cfg.info_gate_floor_meas_used_frac_thr), 1e-6))
        weak_gate_t = smoothstep01((float(cfg.info_gate_floor_gateavg_thr) - gate_avg) / max(float(cfg.info_gate_floor_gateavg_thr), 1e-6))
        weak_obs_t = max(float(unc_need), float(weak_meas_t), float(weak_gate_t))
        info_gate_floor = hard_regime_t * (float(cfg.info_gate_floor_hard) + float(cfg.info_gate_floor_hard_extra) * weak_obs_t)
        info_gate = float(max(obs_gate, clamp(info_gate_floor, 0.0, 1.0)))

        obs_boost = 1.0
        if bool(cfg.obs_boost) and (info_gate > 0.0):
            obs_boost = 1.0 + float(cfg.obs_boost_k) * info_gate * sat_ratio(max(0.0, std_max - tol_std), max(tol_std, 1e-6))
        obs_hard_mult = 1.0 + 0.5 * hard_regime_t
        obs_unc_mult = 1.0 + float(cfg.w_fim_min) * unc_need
        close_goal_radius = float(cfg.close_goal_pos_mult) * float(tol_pos_est)
        close_ratio = sat_ratio(max(close_goal_radius - err_est, 0.0), max(close_goal_radius, 1e-6))
        obs_close_mult = 1.0 + (0.25 * float(cfg.info_sens_boost) * close_ratio if std_max > tol_std else 0.0)
        obs_info_mult = obs_boost * info_gate * obs_hard_mult * obs_unc_mult * obs_close_mult

        if self.prev_std_max is None:
            self.prev_std_max = float(std_max)
        std_gain = float(self.prev_std_max - std_max)
        self.prev_std_max = float(std_max)
        r_stdmax_gain = sat_ratio(std_gain, float(cfg.stdmax_red0_m))

        success_progress_now = bool(err_est < tol_pos_est)
        if bool(cfg.success_require_std):
            success_progress_now = bool(success_progress_now and (std_max < tol_std))
        err_true_form = float(np.linalg.norm(self.pF - pF_des))
        success_true_now = bool(err_true_form < tol_pos_true)
        if bool(cfg.success_require_std):
            success_true_now = bool(success_true_now and (std_max < tol_std))
        mode = str(getattr(cfg, "success_mode", "progress")).lower().strip()
        success_now = success_true_now if mode == "true" else success_progress_now
        bonus_goal = 5.0 if success_now else 0.0

        r_time = -1.0
        r_total = 0.0
        r_total += float(cfg.w_pos) * pos_penalty_scale * r_pos + pos_penalty_scale * r_worst
        r_total += float(cfg.w_std) * r_std
        r_total += float(cfg.w_sens) * obs_info_mult * r_sens
        r_total += float(cfg.w_fim) * obs_info_mult * r_fimmin
        r_total += float(cfg.w_crlb) * obs_info_mult * r_crlb
        r_total += float(cfg.w_worst) * obs_info_mult * r_stdmax_gain
        r_total += float(cfg.w_prog) * r_prog
        r_total += float(cfg.w_info) * r_info
        r_total += float(cfg.w_energy) * r_energy
        r_total += float(cfg.w_time) * r_time
        r_total += bonus_goal
        r_total = float(np.clip(r_total, -float(cfg.rew_clip), float(cfg.rew_clip)))

        nis = float(pf_stats_last.nis) if (pf_stats_last is not None) else float("nan")
        nis_ratio = float(pf_stats_last.nis_ratio) if (pf_stats_last is not None and np.isfinite(pf_stats_last.nis_ratio)) else float("nan")
        consistency_infl = float(pf_stats_last.consistency_infl) if (pf_stats_last is not None and np.isfinite(pf_stats_last.consistency_infl)) else 1.0
        terms = {
            "r_total": r_total,
            "r_pos": r_pos,
            "r_worst": r_worst,
            "r_std": r_std,
            "r_sens": r_sens,
            "r_fimmin": r_fimmin,
            "r_crlb": r_crlb,
            "r_stdmax_gain": r_stdmax_gain,
            "r_prog": r_prog,
            "r_info": r_info,
            "r_energy": r_energy,
            "r_time": r_time,
            "bonus_goal": bonus_goal,
            "err_est": err_est,
            "std_max": std_max,
            "std_gain": std_gain,
            "std_infl": std_infl,
            "unc_metric": unc_metric,
            "sens_avg": sens_avg,
            "fim_hat_eig_min": fim_hat_eig_min,
            "fim_hat_eig_max": fim_hat_eig_max,
            "fim_hat_cond": fim_hat_cond,
            "fim_hat_trace": fim_hat_trace,
            "crlb_hat_trace": crlb_hat_trace,
            "obs_gate": obs_gate,
            "info_gate": info_gate,
            "info_gate_floor": info_gate_floor,
            "weak_meas_t": weak_meas_t,
            "weak_gate_t": weak_gate_t,
            "obs_boost": obs_boost,
            "obs_hard_mult": obs_hard_mult,
            "obs_unc_mult": obs_unc_mult,
            "obs_close_mult": obs_close_mult,
            "obs_info_mult": obs_info_mult,
            "tol_pos_est": float(tol_pos_est),
            "tol_pos_true": float(tol_pos_true),
            "tol_std": float(tol_std),
            "success_progress_now": float(1.0 if success_progress_now else 0.0),
            "success_true_now": float(1.0 if success_true_now else 0.0),
            "success_mode_true": 1.0 if mode == "true" else 0.0,
            "nis": nis,
            "nis_ratio": nis_ratio,
            "pf_consistency_infl": consistency_infl,
            "pf_injected_step": float(self.pf_injected_step),
        }
        if bool(cfg.log_truth_diagnostics):
            terms["err_true_form"] = err_true_form
            terms["err_true_rel_est"] = float(np.linalg.norm(self.pF - pF_hat))
        return r_total, terms

    def _check_done(self) -> Tuple[bool, bool, str]:
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
        mode = str(getattr(self.cfg, "success_mode", "progress")).lower().strip()
        success_now = success_true_now if mode == "true" else success_progress_now
        self._episode_progress_success = bool(success_progress_now)
        self._episode_true_success = bool(success_true_now)
        if success_now:
            self.success_streak += 1
        else:
            self.success_streak = 0
        terminated = bool(self.success_streak >= int(self.cfg.success_hold_steps))
        truncated = bool(self.step_count >= int(self.cfg.max_steps))
        if terminated:
            return True, False, "success"
        if truncated:
            return False, True, "max_steps"
        return False, False, "running"

    def _get_obs_base(self) -> np.ndarray:
        pC, pF_des = self._formation_desired()
        pF_hat = self.pf.mean
        f_hat, side_hat, zhat, R_form = self._formation_axes()

        e_form = project_vec_form(pF_hat - pF_des, R_form)
        rel_form = project_vec_form(pF_hat - pC, R_form)
        rel_l1_form = project_vec_form(self.pL1 - pF_hat, R_form)
        rel_l2_form = project_vec_form(self.pL2 - pF_hat, R_form)
        rho_l1 = float(np.linalg.norm(self.pL1 - pF_hat))
        rho_l2 = float(np.linalg.norm(self.pL2 - pF_hat))
        u_l1_form = rel_l1_form / max(rho_l1, 1e-9)
        u_l2_form = rel_l2_form / max(rho_l2, 1e-9)

        sx_raw, sy_raw, sz_raw = self.pf.stds()
        P_raw = self.pf.cov
        sx_eff, sy_eff, sz_eff, std_max_eff, std_infl = self._compute_conservative_std(sx_raw, sy_raw, sz_raw, P_raw)
        P_eff = np.asarray(P_raw, dtype=float) * (std_infl ** 2)
        P_form = project_cov_form(P_eff, R_form)
        std_form = np.sqrt(np.maximum(np.diag(P_form), 0.0))
        corr01, corr02, corr12 = cov_to_corrs(P_form)
        dom_unc = dominant_unc_axis_abs(P_form)

        self.std_x_eff_step = float(sx_eff)
        self.std_y_eff_step = float(sy_eff)
        self.std_z_eff_step = float(sz_eff)
        self.std_max_eff_step = float(std_max_eff)
        self.std_infl_step = float(std_infl)

        speed_meas = float(self.speed_F)
        yaw_form_deg = wrap360(rad2deg(math.atan2(f_hat[1], f_hat[0])))
        yaw_rel_deg = wrap180(self.yaw_F - yaw_form_deg)
        pitch_meas = float(self.pitch_F)
        vC_speed = float(np.linalg.norm(0.5 * (self.vL1 + self.vL2)))

        ess = self.pf_ess_step
        if not np.isfinite(ess):
            w = self.pf.w
            s = float(np.sum(w * w))
            ess = (1.0 / s) if s > 1e-18 else float(self.pf.N)
        ess_frac = float(ess / max(self.pf.N, 1))

        tol_pos_est, tol_std, _ = self._current_tolerances()
        err_est = float(np.linalg.norm(pF_hat - pF_des))
        need_info = bool(std_max_eff > tol_std)
        denom = max(float(self.cfg.obs_gate_band_mult) * float(tol_pos_est), 1e-6)
        x = (float(self.cfg.obs_gate_pos_mult) * float(tol_pos_est) - err_est) / denom
        gate_pos = smoothstep01(x)
        d_now = float(clamp(self._difficulty, 0.0, 1.0))
        d0 = float(clamp(self.cfg.obs_gate_start_difficulty, 0.0, 1.0))
        dr = float(max(0.0, self.cfg.obs_gate_ramp_difficulty))
        if dr <= 1e-9:
            alpha = 1.0 if d_now >= d0 else 0.0
        else:
            alpha = smoothstep01((d_now - d0) / dr)
        gate_blend = float(lerp(1.0, gate_pos, alpha))
        self.obs_gate = 1.0 if need_info else gate_blend

        sens_avg = float(self.sens_accum / max(self.sens_count, 1.0))
        sens_feat = sat_ratio(sens_avg, float(self.cfg.sens_norm))
        fim_hat_eig_min = float(self.fim_hat_win.eig_stats(self.fim_hat_win.I_win)[0])
        fim_feat = sat_ratio(math.log1p(max(0.0, fim_hat_eig_min)), math.log1p(10.0))

        meas_used_frac = float(self.meas_used_step / max(self.meas_total_step, 1)) if self.meas_total_step > 0 else 0.0
        gate_avg = float(self.gate_avg_step) if np.isfinite(self.gate_avg_step) else 0.0
        nis_ratio_feat = 0.0
        if np.isfinite(self.pf_nis_ratio_step):
            nis_ratio_feat = float(math.tanh(math.log(max(float(self.pf_nis_ratio_step), 1e-6))))
        sigma_nis_norm = float(self.pf_sigma_nis_mult_step / max(float(self.cfg.pf_nis_sigma_adapt_max), 1e-6))
        w_max = float(self.pf_wmax_step) if np.isfinite(self.pf_wmax_step) else float(np.max(self.pf.w))
        injected_frac = float(self.pf_injected_step / max(self.pf.N, 1))
        gate_cur = self._doppler_gate_pf_factors(vel_from_speed_yaw_pitch(speed_meas, self.yaw_F, pitch_meas))
        if len(gate_cur) < 2:
            gate_cur = [0.0, 0.0]
        self._last_gate_pf_current = [float(gate_cur[0]), float(gate_cur[1])]

        obs = np.array([
            e_form[0] / self.cfg.pos_scale,
            e_form[1] / self.cfg.pos_scale,
            e_form[2] / self.cfg.pos_scale,
            rel_form[0] / self.cfg.pos_scale,
            rel_form[1] / self.cfg.pos_scale,
            rel_form[2] / self.cfg.pos_scale,
            std_form[0] / self.cfg.std_scale,
            std_form[1] / self.cfg.std_scale,
            std_form[2] / self.cfg.std_scale,
            corr01,
            corr02,
            corr12,
            speed_meas / self.cfg.vel_scale,
            math.cos(deg2rad(yaw_rel_deg)),
            math.sin(deg2rad(yaw_rel_deg)),
            math.sin(deg2rad(pitch_meas)),
            (self.s1_last if self._has_s1 else 0.0) / self.cfg.s_scale,
            (self.s2_last if self._has_s2 else 0.0) / self.cfg.s_scale,
            rel_l1_form[0] / self.cfg.pos_scale,
            rel_l1_form[1] / self.cfg.pos_scale,
            rel_l1_form[2] / self.cfg.pos_scale,
            u_l1_form[0],
            u_l1_form[1],
            u_l1_form[2],
            rho_l1 / self.cfg.pos_scale,
            rel_l2_form[0] / self.cfg.pos_scale,
            rel_l2_form[1] / self.cfg.pos_scale,
            rel_l2_form[2] / self.cfg.pos_scale,
            u_l2_form[0],
            u_l2_form[1],
            u_l2_form[2],
            rho_l2 / self.cfg.pos_scale,
            vC_speed / max(self.cfg.leader_speed_max, 1e-6),
            ess_frac,
            w_max,
            gate_avg,
            meas_used_frac,
            sigma_nis_norm,
            nis_ratio_feat,
            float(self.pf_resampled_step),
            injected_frac,
            float(self._gate_relax_active),
            float(self._difficulty),
            clamp(err_est / max(tol_pos_est, 1e-6), 0.0, 5.0),
            clamp(std_max_eff / max(tol_std, 1e-6), 0.0, 5.0),
            sens_feat,
            fim_feat,
            dom_unc[0],
            dom_unc[1],
            dom_unc[2],
            float(gate_cur[0]),
            float(gate_cur[1]),
            float(self._last_action_raw[0]),
            float(self._last_action_raw[1]),
            float(self._last_action_raw[2]),
        ], dtype=np.float32)
        return obs

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pC, pF_des = self._formation_desired()
        pF_hat = self.pf.mean
        P = self.pf.cov
        gate_cur = self._last_gate_pf_current if len(self._last_gate_pf_current) >= 2 else [0.0, 0.0]
        vC_speed = float(np.linalg.norm(0.5 * (self.vL1 + self.vL2)))

        info: Dict[str, Any] = {
            "t": float(self.t),
            "step": int(self.step_count),
            "difficulty": float(self._difficulty),
            "pL1_x": float(self.pL1[0]), "pL1_y": float(self.pL1[1]), "pL1_z": float(self.pL1[2]),
            "pL2_x": float(self.pL2[0]), "pL2_y": float(self.pL2[1]), "pL2_z": float(self.pL2[2]),
            "leader1_speed": float(self.leader1_speed),
            "leader2_speed": float(self.leader2_speed),
            "yaw_L1": float(self.yaw_L1),
            "yaw_L2": float(self.yaw_L2),
            "pF_x": float(self.pF[0]), "pF_y": float(self.pF[1]), "pF_z": float(self.pF[2]),
            "speed_F": float(self.speed_F),
            "yaw_F": float(self.yaw_F),
            "pitch_F": float(self.pitch_F),
            "acc_cmd": float(self._last_speed_cmd),
            "yaw_rate_cmd": float(self._last_yaw_rate_cmd),
            "pitch_rate_cmd": float(self._last_pitch_rate_cmd),
            "pFhat_x": float(pF_hat[0]), "pFhat_y": float(pF_hat[1]), "pFhat_z": float(pF_hat[2]),
            "P_xx": float(P[0, 0]), "P_xy": float(P[0, 1]), "P_xz": float(P[0, 2]),
            "P_yy": float(P[1, 1]), "P_yz": float(P[1, 2]), "P_zz": float(P[2, 2]),
            "pFdes_x": float(pF_des[0]), "pFdes_y": float(pF_des[1]), "pFdes_z": float(pF_des[2]),
            "s1_last": float(self.s1_last) if self._has_s1 else float("nan"),
            "s2_last": float(self.s2_last) if self._has_s2 else float("nan"),
            "meas_total_step": float(self.meas_total_step),
            "meas_used_step": float(self.meas_used_step),
            "gate_avg_step": float(self.gate_avg_step) if np.isfinite(self.gate_avg_step) else float("nan"),
            "gate_min_step": float(self.gate_min_step) if np.isfinite(self.gate_min_step) else float("nan"),
            "gate_relax_active": float(self._gate_relax_active),
            "gate_cur_l1": float(gate_cur[0]),
            "gate_cur_l2": float(gate_cur[1]),
            "vC_speed": float(vC_speed),
            "fim_step_meas": float(self.fim_step_meas),
            "fim_step_used": float(self.fim_step_used),
            "fim_hat_step_used": float(self.fim_hat_step_used),
        }

        err_true_form = float(np.linalg.norm(self.pF - pF_des))
        err_true_rel_est = float(np.linalg.norm(self.pf.mean - self.pF))
        info["err_true_form"] = err_true_form
        info["err_true_rel_est"] = err_true_rel_est

        I_win = self.fim_win.I_win
        C_win = self.fim_win.crlb(use_window=True)
        eigmin, eigmax, cond = self.fim_win.eig_stats(I_win)
        info.update({
            "fim_win_eig_min": float(eigmin),
            "fim_win_eig_max": float(eigmax),
            "fim_win_cond": float(cond),
            "crlb_win_trace": float(np.trace(C_win)) if np.all(np.isfinite(C_win)) else float("nan"),
        })

        mode = str(getattr(self.cfg, "success_mode", "progress")).lower().strip()
        info["success_mode"] = mode
        info["is_success_progress"] = float(1.0 if getattr(self, "_episode_progress_success", False) else 0.0)
        info["is_success_true"] = float(1.0 if getattr(self, "_episode_true_success", False) else 0.0)
        info["is_success"] = float(1.0 if (self._last_terminated and self._last_term_reason == "success") else 0.0)
        info["is_success_progress_terminal"] = float(1.0 if (self._last_terminated and self._last_term_reason == "success" and getattr(self, "_episode_progress_success", False)) else 0.0)
        info["is_success_true_terminal"] = float(1.0 if (self._last_terminated and self._last_term_reason == "success" and getattr(self, "_episode_true_success", False)) else 0.0)
        info["success"] = float(1.0 if (self._last_term_reason == "success") else 0.0)

        info["std_x_eff"] = float(self.std_x_eff_step) if np.isfinite(self.std_x_eff_step) else float("nan")
        info["std_y_eff"] = float(self.std_y_eff_step) if np.isfinite(self.std_y_eff_step) else float("nan")
        info["std_z_eff"] = float(self.std_z_eff_step) if np.isfinite(self.std_z_eff_step) else float("nan")
        info["std_max_eff"] = float(self.std_max_eff_step) if np.isfinite(self.std_max_eff_step) else float("nan")
        info["std_infl"] = float(self.std_infl_step) if np.isfinite(self.std_infl_step) else float("nan")
        info["pf_nis_ratio_step"] = float(self.pf_nis_ratio_step) if np.isfinite(self.pf_nis_ratio_step) else float("nan")
        info["pf_consistency_infl_step"] = float(self.pf_consistency_infl_step) if np.isfinite(self.pf_consistency_infl_step) else 1.0
        info["pf_sigma_nis_mult_step"] = float(self.pf_sigma_nis_mult_step)
        info["pf_wmax_step"] = float(self.pf_wmax_step) if np.isfinite(self.pf_wmax_step) else float("nan")
        info["pf_injected_step"] = float(self.pf_injected_step)

        if extra:
            info.update(extra)
        return info

    # ----------------- render / manual control -----------------

    def render(self):
        if self.render_mode != "human":
            return
        if pygame is None:
            raise RuntimeError("pygame is not installed. pip install pygame")
        if not self._pygame_inited:
            pygame.init()
            self._screen = pygame.display.set_mode((self.cfg.screen_w, self.cfg.screen_h))
            pygame.display.set_caption("UUV 3D PF – Two Leaders (formation-frame obs)")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 16)
            self._pygame_inited = True
        assert self._screen is not None and self._clock is not None and self._font is not None
        self._process_pygame_events()
        if self._request_quit:
            return
        screen = self._screen
        font = self._font
        try:
            from uuv_3d_render_pygame import UUV3DRenderer
            if self._renderer3d is None:
                self._renderer3d = UUV3DRenderer(cfg=self.cfg)
            self._renderer3d.draw(env=self, screen=screen, font=font)
        except Exception as e:
            screen.fill((10, 10, 25))
            txt = font.render(f"[render fallback] {type(e).__name__}: {e}", True, (235, 235, 235))
            screen.blit(txt, (10, 10))
        pygame.display.flip()
        self._clock.tick(self.cfg.render_fps)

    def close(self):
        if self._pygame_inited and pygame is not None:
            pygame.quit()
        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._font = None

    def _process_pygame_events(self) -> None:
        if pygame is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._request_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._request_quit = True
                elif event.key == pygame.K_BACKSPACE:
                    self._request_reset = True
                elif event.key == pygame.K_m:
                    self.manual_override = not self.manual_override
        pygame.event.pump()

    def _keyboard_action(self) -> np.ndarray:
        if pygame is None:
            return np.zeros(4, dtype=np.float32)
        keys = pygame.key.get_pressed()
        a_speed = 0.0
        if keys[pygame.K_UP]:
            a_speed += 1.0
        if keys[pygame.K_DOWN]:
            a_speed -= 1.0
        a_yaw = 0.0
        if keys[pygame.K_LEFT]:
            a_yaw += 1.0
        if keys[pygame.K_RIGHT]:
            a_yaw -= 1.0
        a_pitch = 0.0
        if keys[pygame.K_w]:
            a_pitch += 1.0
        if keys[pygame.K_s]:
            a_pitch -= 1.0
        fine = 1.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 0.0
        return np.array([
            clamp(a_speed, -1.0, 1.0),
            clamp(a_yaw, -1.0, 1.0),
            clamp(a_pitch, -1.0, 1.0),
            fine,
        ], dtype=np.float32)


# =============================================================================
# v8 Info-Guidance Extension
# =============================================================================

# Compatibility aliases for the merged v7/v8 layout.
BaseUUV3DConfig = UUV3DBaseConfig
BaseUUVTwoLeader3DPFEnv = UUVTwoLeader3DPFBaseEnv


@dataclass
class UUV3DConfig(UUV3DBaseConfig):
    info_planner_enabled: bool = True
    info_planner_horizon_s: float = 3.0
    info_planner_dt: float = 1.0
    info_planner_support_sigma: float = 1.0
    info_planner_support_axes: int = 2
    info_planner_mean_weight: float = 0.45
    info_planner_start_difficulty: float = 0.72
    info_planner_ramp_difficulty: float = 0.18
    info_planner_unc_gap0_m: float = 2.0
    info_planner_trace_red0: float = 30.0
    info_planner_eigmin0: float = 8.0
    info_planner_goal_err0: float = 35.0
    info_planner_w_trace: float = 1.15
    info_planner_w_eigmin: float = 0.95
    info_planner_w_sens: float = 0.25
    info_planner_w_goal: float = 0.65
    info_planner_w_energy: float = 0.08
    info_planner_margin_gain: float = 2.0
    info_planner_gate_min: float = 0.05
    info_planner_align_reward: float = 0.30


@dataclass
class InfoPlanEval:
    action: np.ndarray
    score: float
    trace_red: float
    eigmin: float
    sens_avg: float
    pred_err: float
    gate_avg: float
    disp_world_first: np.ndarray


class UUVTwoLeader3DPFEnv(UUVTwoLeader3DPFBaseEnv):
    """
    v8 extends the 55-dim core observation with 9 information-guidance features:
      55-57  recommended action in RL action space (speed/yaw/pitch)
      58-60  recommended displacement direction in formation frame
      61     planner best-score feature
      62     planner score margin feature (best - neutral)
      63     planner activation gate
    """

    BASE_OBS_DIM = UUVTwoLeader3DPFBaseEnv.BASE_OBS_DIM + 9

    INFO_ACTION_CANDIDATES = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [+0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, +0.5, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, +1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, +0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, +1.0],
            [0.0, 0.0, -1.0],
            [+0.5, +1.0, 0.0],
            [+0.5, -1.0, 0.0],
            [+0.5, 0.0, +1.0],
            [+0.5, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, cfg: Optional[UUV3DConfig] = None, render_mode: str = "none"):
        cfg = cfg or UUV3DConfig()
        super().__init__(cfg=cfg, render_mode=render_mode)
        self.cfg: UUV3DConfig
        self._info_plan_action = np.zeros(3, dtype=np.float32)
        self._info_plan_dir_form = np.zeros(3, dtype=np.float32)
        self._info_plan_score_best = 0.0
        self._info_plan_score_zero = 0.0
        self._info_plan_margin = 0.0
        self._info_plan_gate = 0.0
        self._info_plan_trace_red = 0.0
        self._info_plan_eigmin = 0.0
        self._info_plan_pred_err = 0.0
        self._info_plan_sens_avg = 0.0
        self._info_plan_gate_avg = 0.0
        self._info_plan_active = 0.0

    def _info_reset(self) -> None:
        self._info_plan_action[:] = 0.0
        self._info_plan_dir_form[:] = 0.0
        self._info_plan_score_best = 0.0
        self._info_plan_score_zero = 0.0
        self._info_plan_margin = 0.0
        self._info_plan_gate = 0.0
        self._info_plan_trace_red = 0.0
        self._info_plan_eigmin = 0.0
        self._info_plan_pred_err = 0.0
        self._info_plan_sens_avg = 0.0
        self._info_plan_gate_avg = 0.0
        self._info_plan_active = 0.0

    def _info_support_points_from_cov(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean, dtype=float).reshape(3)
        cov = np.asarray(cov, dtype=float).reshape(3, 3)
        cov = 0.5 * (cov + cov.T)

        mean_w = float(clamp(self.cfg.info_planner_mean_weight, 0.05, 0.95))
        points: List[np.ndarray] = [mean.copy()]
        weights: List[float] = [mean_w]

        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return mean[None, :], np.array([1.0], dtype=float)

        axes = int(max(0, min(3, int(self.cfg.info_planner_support_axes))))
        if axes <= 0:
            return mean[None, :], np.array([1.0], dtype=float)

        idx = np.argsort(eigvals)[::-1][:axes]
        sigma_scale = float(max(0.0, self.cfg.info_planner_support_sigma))
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for j in idx:
            s = sigma_scale * math.sqrt(max(float(eigvals[j]), 0.0))
            if s <= 1e-6:
                continue
            v = np.asarray(eigvecs[:, j], dtype=float).reshape(3)
            pairs.append((mean + s * v, mean - s * v))

        if not pairs:
            return mean[None, :], np.array([1.0], dtype=float)

        side_w = max(1e-6, 1.0 - mean_w) / float(2 * len(pairs))
        for p_pos, p_neg in pairs:
            points.extend([p_pos, p_neg])
            weights.extend([side_w, side_w])

        w = np.asarray(weights, dtype=float)
        w /= float(np.sum(w))
        return np.asarray(points, dtype=float), w

    def _info_support_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.asarray(self.pf.mean, dtype=float).reshape(3)
        infl = float(self.std_infl_step) if np.isfinite(self.std_infl_step) and self.std_infl_step > 0.0 else 1.0
        p_eff = np.asarray(self.pf.cov, dtype=float).reshape(3, 3) * (infl ** 2)
        pts, w = self._info_support_points_from_cov(mean, p_eff)
        return pts, w, p_eff

    def _info_activation_gate(self, std_max_eff: float, tol_std: float) -> float:
        if not bool(self.cfg.info_planner_enabled):
            return 0.0
        d = float(clamp(self._difficulty, 0.0, 1.0))
        hard_t = smoothstep01(
            (d - float(self.cfg.info_planner_start_difficulty))
            / max(float(self.cfg.info_planner_ramp_difficulty), 1e-6)
        )
        meas_used_frac = float(self.meas_used_step / max(self.meas_total_step, 1)) if self.meas_total_step > 0 else 0.0
        gate_avg = float(self.gate_avg_step) if np.isfinite(self.gate_avg_step) else 0.0
        weak_meas_t = smoothstep01(
            (float(self.cfg.info_gate_floor_meas_used_frac_thr) - meas_used_frac)
            / max(float(self.cfg.info_gate_floor_meas_used_frac_thr), 1e-6)
        )
        weak_gate_t = smoothstep01(
            (float(self.cfg.info_gate_floor_gateavg_thr) - gate_avg)
            / max(float(self.cfg.info_gate_floor_gateavg_thr), 1e-6)
        )
        unc_gap = max(0.0, float(std_max_eff) - float(tol_std))
        unc_t = sat_ratio(unc_gap, max(float(self.cfg.info_planner_unc_gap0_m), 1e-6))
        gate = hard_t * max(unc_t, weak_meas_t, weak_gate_t)
        return float(clamp(gate, 0.0, 1.0))

    def _info_gate_factor_single(self, r_vec: np.ndarray, v_rel: np.ndarray) -> float:
        r_vec = np.asarray(r_vec, dtype=float).reshape(3)
        v_rel = np.asarray(v_rel, dtype=float).reshape(3)
        rho = float(np.linalg.norm(r_vec))
        vnorm = float(np.linalg.norm(v_rel))
        if rho <= 1e-9 or vnorm <= 1e-9:
            return 0.0
        rhat = r_vec / max(rho, 1e-9)
        proj = float(np.dot(rhat, v_rel))
        v_perp = float(np.linalg.norm(v_rel - proj * rhat))
        v_perp_min = lerp(self.cfg.doppler_v_perp_min_easy, self.cfg.doppler_v_perp_min_hard, float(self._difficulty))
        band_v_perp = float(self.cfg.gate_band_v_perp)
        if self._gate_relax_active > 0.5:
            v_perp_min = min(float(v_perp_min), float(self.cfg.gate_relax_v_perp_min_hard))
            band_v_perp = max(float(band_v_perp), float(self.cfg.gate_relax_band_v_perp))
        g_rho = soft_gate(rho, float(self.cfg.doppler_rho_min), float(self.cfg.gate_band_rho))
        g_v = soft_gate(vnorm, float(self.cfg.doppler_v_min), float(self.cfg.gate_band_v))
        g_vp = soft_gate(v_perp, float(v_perp_min), float(band_v_perp))
        g = float(g_rho * g_v * g_vp)
        if g < float(self.cfg.gate_min_factor):
            return 0.0
        return float(clamp(g, 0.0, 1.0))

    def _info_eval_action(
        self,
        action_norm: np.ndarray,
        support_points: np.ndarray,
        support_weights: np.ndarray,
        p0_eff: np.ndarray,
        tol_pos_est: float,
    ) -> InfoPlanEval:
        cfg = self.cfg
        a = np.asarray(action_norm, dtype=float).reshape(3)
        dt = float(max(1e-3, cfg.info_planner_dt))
        horizon = float(max(dt, cfg.info_planner_horizon_s))
        n_steps = max(1, int(round(horizon / dt)))

        speed_cmd = float(a[0]) * (float(cfg.rl_speed_delta_per_step) / max(float(cfg.action_dt), 1e-6))
        yaw_cmd = float(a[1]) * min(float(cfg.max_yaw_rate_deg_s), float(cfg.rl_yaw_per_step_deg) / max(float(cfg.action_dt), 1e-6))
        pitch_cmd = float(a[2]) * min(float(cfg.max_pitch_rate_deg_s), float(cfg.rl_pitch_per_step_deg) / max(float(cfg.action_dt), 1e-6))

        pts = np.asarray(support_points, dtype=float).copy()
        wk = np.asarray(support_weights, dtype=float).reshape(-1)
        mean_p = np.sum(pts * wk[:, None], axis=0)

        p_l1 = np.asarray(self.pL1, dtype=float).copy()
        p_l2 = np.asarray(self.pL2, dtype=float).copy()
        v_l1 = np.asarray(self.vL1, dtype=float).copy()
        v_l2 = np.asarray(self.vL2, dtype=float).copy()
        speed = float(self.speed_F)
        yaw = float(self.yaw_F)
        pitch = float(self.pitch_F)

        sigma_mult = float(getattr(self.pf, "sigma_nis_mult", 1.0))
        sigma2 = max(float(self.pf.meas_sigma) ** 2 * sigma_mult * sigma_mult, 1e-18)

        fim_inc = np.zeros((3, 3), dtype=float)
        gate_sum = 0.0
        gate_cnt = 0.0
        sens_sum = 0.0
        disp_world_first = np.zeros(3, dtype=float)

        for step_idx in range(n_steps):
            speed = float(np.clip(speed + speed_cmd * dt, cfg.f_min_speed, cfg.f_max_speed))
            yaw = wrap360(yaw + yaw_cmd * dt)
            pitch = clamp(pitch + pitch_cmd * dt, cfg.pitch_min_deg, cfg.pitch_max_deg)
            v_f = vel_from_speed_yaw_pitch(speed, yaw, pitch)
            disp = v_f * dt
            if step_idx == 0:
                disp_world_first = disp.copy()
            pts = pts + disp[None, :]
            mean_p = mean_p + disp
            p_l1 = p_l1 + v_l1 * dt
            p_l2 = p_l2 + v_l2 * dt

            for p_l, v_l in ((p_l1, v_l1), (p_l2, v_l2)):
                v_rel = np.asarray(v_l - v_f, dtype=float)
                for k in range(pts.shape[0]):
                    g = self._info_gate_factor_single(p_l - pts[k], v_rel)
                    if g <= 0.0:
                        continue
                    h_k = doppler_H_3d(p_l - pts[k], v_rel)
                    fim_inc += float(wk[k] * g) * (h_k.T @ h_k) / sigma2
                    rho = float(np.linalg.norm(p_l - pts[k]))
                    if rho > 1e-9:
                        rhat = (p_l - pts[k]) / rho
                        proj = float(np.dot(rhat, v_rel))
                        v_perp = float(np.linalg.norm(v_rel - proj * rhat))
                        sens_sum += float(wk[k] * g) * v_perp
                    gate_sum += float(g)
                    gate_cnt += 1.0

        p_c = 0.5 * (p_l1 + p_l2)
        v_c = 0.5 * (v_l1 + v_l2)
        f_hat = normalize_vec(np.array([v_c[0], v_c[1], 0.0], dtype=float), fallback=np.array([1.0, 0.0, 0.0], dtype=float))
        zhat = np.array([0.0, 0.0, 1.0], dtype=float)
        side_hat = normalize_vec(np.cross(f_hat, zhat), fallback=np.array([0.0, 1.0, 0.0], dtype=float))
        p_f_des = p_c - float(cfg.d_back) * f_hat + float(cfg.d_right) * side_hat
        p_f_des[2] = float(p_c[2] + cfg.dz_offset)
        pred_err = float(np.linalg.norm(mean_p - p_f_des))

        p_inv = safe_inv_sym(np.asarray(p0_eff, dtype=float).reshape(3, 3), eps=1e-6)
        if not np.all(np.isfinite(p_inv)):
            s = max(float(np.trace(p0_eff)) / 3.0, 1e-6)
            p_inv = np.eye(3, dtype=float) / s
        c_post = safe_inv_sym(p_inv + fim_inc, eps=1e-6)
        trace_red = 0.0
        if np.all(np.isfinite(c_post)):
            trace_red = max(float(np.trace(p0_eff) - np.trace(c_post)), 0.0)
        eigmin, _eigmax, _cond = FIMTracker3D.eig_stats(fim_inc)
        sens_avg = float(sens_sum / max(gate_cnt, 1.0))
        gate_avg = float(gate_sum / max(gate_cnt, 1.0))

        goal_scale = max(float(cfg.info_planner_goal_err0), 1.5 * float(tol_pos_est), 1e-6)
        trace_feat = sat_ratio(trace_red, max(float(cfg.info_planner_trace_red0), 1e-6))
        eig_feat = sat_ratio(math.log1p(max(float(eigmin), 0.0)), math.log1p(max(float(cfg.info_planner_eigmin0), 1e-6)))
        sens_feat = sat_ratio(sens_avg, max(float(cfg.sens_norm), 1e-6))
        goal_pen = sat_ratio(pred_err, goal_scale)
        energy_pen = sat_ratio(float(np.linalg.norm(a)), 1.0)
        score = 0.0
        score += float(cfg.info_planner_w_trace) * trace_feat
        score += float(cfg.info_planner_w_eigmin) * eig_feat
        score += float(cfg.info_planner_w_sens) * sens_feat
        score += 0.10 * gate_avg
        score -= float(cfg.info_planner_w_goal) * goal_pen
        score -= float(cfg.info_planner_w_energy) * energy_pen

        return InfoPlanEval(
            action=np.asarray(a, dtype=np.float32),
            score=float(score),
            trace_red=float(trace_red),
            eigmin=float(eigmin),
            sens_avg=float(sens_avg),
            pred_err=float(pred_err),
            gate_avg=float(gate_avg),
            disp_world_first=np.asarray(disp_world_first, dtype=float),
        )

    def _info_search_action(
        self,
        support_points: np.ndarray,
        support_weights: np.ndarray,
        p0_eff: np.ndarray,
        tol_pos_est: float,
    ) -> Tuple[Optional[InfoPlanEval], Optional[InfoPlanEval], float]:
        best: Optional[InfoPlanEval] = None
        zero: Optional[InfoPlanEval] = None
        second_best_score = -float("inf")
        for a in self.INFO_ACTION_CANDIDATES:
            ev = self._info_eval_action(a, support_points, support_weights, p0_eff, tol_pos_est)
            if zero is None and float(np.linalg.norm(a)) <= 1e-12:
                zero = ev
            if best is None or ev.score > best.score:
                if best is not None:
                    second_best_score = max(second_best_score, best.score)
                best = ev
            else:
                second_best_score = max(second_best_score, ev.score)
        if zero is None:
            zero = best
        if best is None:
            second_best_score = -float("inf")
        return best, zero, float(second_best_score)

    def _compute_info_guidance(self) -> None:
        self._info_reset()
        if not bool(self.cfg.info_planner_enabled):
            return

        tol_pos_est, tol_std, _ = self._current_tolerances()
        std_max_eff = float(self.std_max_eff_step) if np.isfinite(self.std_max_eff_step) else float(self.pf.std_max())
        assist_gate = self._info_activation_gate(std_max_eff, tol_std)
        self._info_plan_gate = float(assist_gate)
        if assist_gate < float(self.cfg.info_planner_gate_min):
            return

        support_points, support_weights, p0_eff = self._info_support_points()
        best, zero, second_best_score = self._info_search_action(support_points, support_weights, p0_eff, tol_pos_est)
        if best is None:
            return

        _, _, _, r_form = self._formation_axes()
        d_world = np.asarray(best.disp_world_first, dtype=float).reshape(3)
        d_form = project_vec_form(d_world, r_form)
        n_d = float(np.linalg.norm(d_form))
        if n_d > 1e-9:
            d_form = d_form / n_d
        else:
            d_form = np.zeros(3, dtype=float)

        zero_score = float(zero.score) if zero is not None else 0.0
        margin = max(float(best.score) - zero_score, 0.0)

        self._info_plan_action = np.asarray(best.action, dtype=np.float32).copy()
        self._info_plan_dir_form = np.asarray(d_form, dtype=np.float32).copy()
        self._info_plan_score_best = float(best.score)
        self._info_plan_score_zero = float(zero_score)
        self._info_plan_margin = float(margin)
        self._info_plan_trace_red = float(best.trace_red)
        self._info_plan_eigmin = float(best.eigmin)
        self._info_plan_pred_err = float(best.pred_err)
        self._info_plan_sens_avg = float(best.sens_avg)
        self._info_plan_gate_avg = float(best.gate_avg)
        self._info_plan_active = 1.0 if np.isfinite(second_best_score) else 0.0

    def _get_obs_base(self) -> np.ndarray:
        obs_base = super()._get_obs_base().astype(np.float32)
        self._compute_info_guidance()
        score_feat = float(math.tanh(self._info_plan_score_best))
        margin_feat = float(math.tanh(float(self.cfg.info_planner_margin_gain) * max(self._info_plan_margin, 0.0)))
        obs_info = np.array(
            [
                float(self._info_plan_action[0]),
                float(self._info_plan_action[1]),
                float(self._info_plan_action[2]),
                float(self._info_plan_dir_form[0]),
                float(self._info_plan_dir_form[1]),
                float(self._info_plan_dir_form[2]),
                score_feat,
                margin_feat,
                float(self._info_plan_gate),
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs_base, obs_info], axis=0).astype(np.float32)

    def _compute_reward(
        self,
        pf_stats_last: Optional[PFStats],
        planner_action_for_reward: Optional[np.ndarray] = None,
        planner_gate_for_reward: Optional[float] = None,
        planner_margin_for_reward: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        reward, terms = super()._compute_reward(pf_stats_last)

        a_ref = np.asarray(
            self._info_plan_action if planner_action_for_reward is None else planner_action_for_reward,
            dtype=float,
        ).reshape(3)
        gate = float(self._info_plan_gate if planner_gate_for_reward is None else planner_gate_for_reward)
        margin = float(self._info_plan_margin if planner_margin_for_reward is None else planner_margin_for_reward)

        a = np.asarray(self._last_action_raw, dtype=float).reshape(3)
        na = float(np.linalg.norm(a))
        nr = float(np.linalg.norm(a_ref))
        align_cos = 0.0
        if na > 1e-6 and nr > 1e-6:
            align_cos = float(np.dot(a, a_ref) / max(na * nr, 1e-6))
        align_bonus = 0.0
        if gate > 0.0 and margin > 0.0:
            align_bonus = float(self.cfg.info_planner_align_reward) * gate * math.tanh(
                float(self.cfg.info_planner_margin_gain) * margin
            ) * max(align_cos, 0.0)

        reward = float(np.clip(reward + align_bonus, -float(self.cfg.rew_clip), float(self.cfg.rew_clip)))
        terms["r_info_align"] = float(align_bonus)
        terms["info_align_cos"] = float(align_cos)
        terms["info_plan_gate_prev"] = float(gate)
        terms["info_plan_margin_prev"] = float(margin)
        return reward, terms

    def step(self, action: np.ndarray):
        planner_action_prev = np.asarray(self._info_plan_action, dtype=np.float32).copy()
        planner_gate_prev = float(self._info_plan_gate)
        planner_margin_prev = float(self._info_plan_margin)

        if self.render_mode == "human" and self._pygame_inited and pygame is not None:
            self._process_pygame_events()

        if self._request_quit:
            self._request_quit = False
            if len(self._obs_hist) == 0:
                self._reset_obs_history(self._get_obs_base())
            obs = self._stack_obs_history()
            info = self._get_info(extra={"term_reason": "quit"})
            return obs, 0.0, False, True, info

        if self._request_reset:
            self._request_reset = False
            if len(self._obs_hist) == 0:
                self._reset_obs_history(self._get_obs_base())
            obs = self._stack_obs_history()
            info = self._get_info(extra={"term_reason": "manual_reset"})
            return obs, 0.0, False, True, info

        self._reset_step_accums()

        if self.render_mode == "human" and self.manual_override and self._pygame_inited and pygame is not None:
            a = self._keyboard_action()
            a_raw = np.asarray(a[:3], dtype=np.float32).reshape(-1)
            speed_cmd = float(a_raw[0]) * (self.cfg.manual_speed_delta_per_step / max(self.cfg.action_dt, 1e-6))
            yaw_cmd = float(a_raw[1]) * (self.cfg.manual_yaw_per_step_deg / max(self.cfg.action_dt, 1e-6))
            pitch_cmd = float(a_raw[2]) * (self.cfg.manual_pitch_per_step_deg / max(self.cfg.action_dt, 1e-6))
            fine = float(a[3])
            if fine > 0.5:
                speed_cmd *= self.cfg.manual_fine_scale
                yaw_cmd *= self.cfg.manual_fine_scale
                pitch_cmd *= self.cfg.manual_fine_scale
        else:
            a_raw = np.asarray(action, dtype=np.float32).reshape(-1)
            a_raw = np.clip(a_raw, -1.0, 1.0)
            speed_cmd = float(a_raw[0]) * (float(self.cfg.rl_speed_delta_per_step) / max(self.cfg.action_dt, 1e-6))
            yaw_cmd = float(a_raw[1]) * min(float(self.cfg.max_yaw_rate_deg_s), float(self.cfg.rl_yaw_per_step_deg) / max(self.cfg.action_dt, 1e-6))
            pitch_cmd = float(a_raw[2]) * min(float(self.cfg.max_pitch_rate_deg_s), float(self.cfg.rl_pitch_per_step_deg) / max(self.cfg.action_dt, 1e-6))

        self._last_action_raw = np.asarray(a_raw, dtype=np.float32).copy()
        self._last_speed_cmd = float(speed_cmd)
        self._last_yaw_rate_cmd = float(yaw_cmd)
        self._last_pitch_rate_cmd = float(pitch_cmd)
        self.a_cmd = np.array([self._last_speed_cmd, self._last_yaw_rate_cmd, self._last_pitch_rate_cmd], dtype=float)

        n_sub = max(1, int(round(self.cfg.action_dt / self.cfg.sub_dt)))
        dt = float(self.cfg.action_dt / n_sub)
        pf_stats_last: Optional[PFStats] = None
        for _ in range(n_sub):
            pf_stats_last = self._sim_substep(speed_cmd, yaw_cmd, pitch_cmd, dt)

        self.step_count += 1
        self.total_env_steps += 1
        self._update_gate_relax_state()

        base_obs = self._get_obs_base()
        self._push_obs_history(base_obs)
        obs = self._stack_obs_history()
        reward, terms = self._compute_reward(
            pf_stats_last,
            planner_action_for_reward=planner_action_prev,
            planner_gate_for_reward=planner_gate_prev,
            planner_margin_for_reward=planner_margin_prev,
        )
        terminated, truncated, reason = self._check_done()
        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)
        self._last_term_reason = str(reason)

        if terminated and reason == "success":
            reward += float(self.cfg.terminal_bonus)
            terms["terminal_bonus"] = float(self.cfg.terminal_bonus)
        else:
            terms["terminal_bonus"] = 0.0

        info = self._get_info(extra=terms | {"term_reason": reason, "manual_override": int(self.manual_override)})
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)
        info.update(
            {
                "info_plan_a_speed": float(self._info_plan_action[0]),
                "info_plan_a_yaw": float(self._info_plan_action[1]),
                "info_plan_a_pitch": float(self._info_plan_action[2]),
                "info_plan_dir_fx": float(self._info_plan_dir_form[0]),
                "info_plan_dir_fy": float(self._info_plan_dir_form[1]),
                "info_plan_dir_fz": float(self._info_plan_dir_form[2]),
                "info_plan_score_best": float(self._info_plan_score_best),
                "info_plan_score_zero": float(self._info_plan_score_zero),
                "info_plan_margin": float(self._info_plan_margin),
                "info_plan_gate": float(self._info_plan_gate),
                "info_plan_trace_red": float(self._info_plan_trace_red),
                "info_plan_eigmin": float(self._info_plan_eigmin),
                "info_plan_pred_err": float(self._info_plan_pred_err),
                "info_plan_sens_avg": float(self._info_plan_sens_avg),
                "info_plan_gate_avg": float(self._info_plan_gate_avg),
                "info_plan_active": float(self._info_plan_active),
            }
        )
        return info


def export_info_map(
    env: UUVTwoLeader3DPFEnv,
    out_prefix: str,
    *,
    nx: int = 25,
    ny: int = 25,
    nz: int = 11,
    x_min: float = -280.0,
    x_max: float = 60.0,
    y_min: float = -220.0,
    y_max: float = 220.0,
    z_min: float = -120.0,
    z_max: float = 20.0,
) -> Dict[str, str]:
    """
    Export a 3D information map in formation frame for the current frozen scenario.
    """
    if not isinstance(env, UUVTwoLeader3DPFEnv):
        raise TypeError("env must be UUVTwoLeader3DPFEnv")

    _ensure_dir(os.path.dirname(out_prefix) or ".")

    p_c, _ = env._formation_desired()
    _, _, _, r_form = env._formation_axes()
    infl = float(env.std_infl_step) if np.isfinite(env.std_infl_step) and env.std_infl_step > 0.0 else 1.0
    p0_eff = np.asarray(env.pf.cov, dtype=float).reshape(3, 3) * (infl ** 2)
    tol_pos_est, _tol_std, _tol_pos_true = env._current_tolerances()

    xs = np.linspace(float(x_min), float(x_max), int(max(2, nx)))
    ys = np.linspace(float(y_min), float(y_max), int(max(2, ny)))
    zs = np.linspace(float(z_min), float(z_max), int(max(2, nz)))

    score = np.full((xs.size, ys.size, zs.size), np.nan, dtype=np.float32)
    action = np.zeros((xs.size, ys.size, zs.size, 3), dtype=np.float32)
    margin = np.full((xs.size, ys.size, zs.size), np.nan, dtype=np.float32)
    rows: List[Dict[str, Any]] = []

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                mean_form = np.array([x, y, z], dtype=float)
                mean_world = np.asarray(p_c, dtype=float).reshape(3) + (r_form.T @ mean_form)
                support_points, support_weights = env._info_support_points_from_cov(mean_world, p0_eff)
                best, zero, _second = env._info_search_action(support_points, support_weights, p0_eff, tol_pos_est)
                if best is None:
                    continue
                zero_score = float(zero.score) if zero is not None else 0.0
                m = max(float(best.score) - zero_score, 0.0)
                score[ix, iy, iz] = float(best.score)
                action[ix, iy, iz, :] = np.asarray(best.action, dtype=np.float32)
                margin[ix, iy, iz] = float(m)
                rows.append(
                    {
                        "x_form": float(x),
                        "y_form": float(y),
                        "z_form": float(z),
                        "score_best": float(best.score),
                        "score_zero": float(zero_score),
                        "margin": float(m),
                        "best_a_speed": float(best.action[0]),
                        "best_a_yaw": float(best.action[1]),
                        "best_a_pitch": float(best.action[2]),
                        "trace_red": float(best.trace_red),
                        "eigmin": float(best.eigmin),
                        "sens_avg": float(best.sens_avg),
                        "pred_err": float(best.pred_err),
                        "gate_avg": float(best.gate_avg),
                    }
                )

    csv_path = out_prefix + ".csv"
    npz_path = out_prefix + ".npz"
    save_csv(csv_path, rows)
    save_npz(
        npz_path,
        {
            "x_form": xs.astype(np.float32),
            "y_form": ys.astype(np.float32),
            "z_form": zs.astype(np.float32),
            "score_best": score,
            "score_margin": margin,
            "best_action": action,
        },
    )
    return {"csv": csv_path, "npz": npz_path}


# =============================================================================
# SB3 helpers / callbacks
# =============================================================================

def make_env(seed: int, cfg: UUV3DConfig, render: bool, rank: int = 0):
    def _init():
        env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode=("human" if render else "none"))
        env.reset(seed=seed + rank)
        return env
    return _init


class TBInfoCallback:
    def __init__(self, log_freq: int = 1000, prefix: str = "env/"):
        self.log_freq = int(max(1, log_freq))
        self.prefix = str(prefix)
        self._episodes = 0
        self._successes = 0
        self._win_size = 200
        self._recent_success = deque(maxlen=self._win_size)
        self._sum: Dict[str, float] = {}
        self._count: Dict[str, int] = {}
        self._ep_returns: Optional[List[float]] = None
        self._ep_lengths: Optional[List[int]] = None
        self._ep_return_hist: List[float] = []
        self._ep_len_hist: List[int] = []

    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        self_ = locals_["self"]
        num_timesteps = int(getattr(self_, "num_timesteps", 0))
        dones = locals_.get("dones", None)
        rewards = locals_.get("rewards", None)
        if rewards is None:
            rewards = locals_.get("reward", None)
        infos = locals_.get("infos", None)
        if infos is None:
            return True

        reward_vals = None
        if rewards is not None:
            try:
                reward_vals = np.asarray(rewards, dtype=float).reshape(-1)
            except Exception:
                try:
                    reward_vals = np.array([float(rewards)], dtype=float)
                except Exception:
                    reward_vals = None

        for info in infos:
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if not np.isfinite(fv):
                        continue
                    self._sum[k] = float(self._sum.get(k, 0.0)) + fv
                    self._count[k] = int(self._count.get(k, 0)) + 1

        if dones is not None:
            for done, info in zip(dones, infos):
                if bool(done):
                    self._episodes += 1
                    s = 1 if (isinstance(info, dict) and float(info.get("is_success", 0.0)) > 0.5) else 0
                    self._successes += int(s)
                    self._recent_success.append(int(s))

        if dones is not None and reward_vals is not None:
            done_arr = np.asarray(dones, dtype=bool).reshape(-1)
            if done_arr.size > 0:
                n = min(done_arr.size, reward_vals.size, len(infos))
                if n > 0:
                    if self._ep_returns is None or len(self._ep_returns) != n:
                        self._ep_returns = [0.0 for _ in range(n)]
                        self._ep_lengths = [0 for _ in range(n)]
                    for i in range(n):
                        r = reward_vals[i]
                        if np.isfinite(r):
                            self._ep_returns[i] += float(r)
                        self._ep_lengths[i] += 1
                        if bool(done_arr[i]):
                            self._ep_return_hist.append(float(self._ep_returns[i]))
                            self._ep_len_hist.append(int(self._ep_lengths[i]))
                            self._ep_returns[i] = 0.0
                            self._ep_lengths[i] = 0

        if reward_vals is not None:
            finite_mask = np.isfinite(reward_vals)
            if np.any(finite_mask):
                self._sum["reward"] = float(self._sum.get("reward", 0.0)) + float(np.sum(reward_vals[finite_mask]))
                self._count["reward"] = int(self._count.get("reward", 0)) + int(np.sum(finite_mask))

        if num_timesteps % self.log_freq != 0:
            return True

        logger = getattr(self_, "logger", None)
        if logger is None:
            self._sum.clear()
            self._count.clear()
            return True

        for k in list(self._sum.keys()):
            c = int(self._count.get(k, 0))
            if c <= 0:
                continue
            m = float(self._sum[k] / c)
            logger.record(self.prefix + k, m)
            if k == "reward":
                logger.record("rollout/reward", m)
        self._sum.clear()
        self._count.clear()

        if self._episodes > 0:
            sr_cum = float(self._successes / self._episodes)
            logger.record(self.prefix + "success_rate_cum", sr_cum)
            logger.record(self.prefix + "episodes", float(self._episodes))
            if len(self._recent_success) > 0:
                sr_win = float(np.mean(self._recent_success))
                logger.record(self.prefix + "success_rate_win", sr_win)
        if self._ep_return_hist:
            logger.record("rollout/ep_rew_mean", float(np.mean(self._ep_return_hist)))
            logger.record("rollout/ep_len_mean", float(np.mean(self._ep_len_hist)) if self._ep_len_hist else float("nan"))
            self._ep_return_hist.clear()
            self._ep_len_hist.clear()
        return True


class TraceCallback:
    def __init__(self, out_dir: str, trace_freq: int = 10000, max_steps: int = 5000):
        self.out_dir = str(out_dir)
        _ensure_dir(self.out_dir)
        self.trace_freq = int(max(1, trace_freq))
        self.max_steps = int(max(10, max_steps))
        self._t0 = _timestamp()
        self._buf: Dict[str, List[float]] = {}
        self._step_buf = 0
        self._trace_idx = 0

    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        self_ = locals_["self"]
        num_timesteps = int(getattr(self_, "num_timesteps", 0))
        infos = locals_.get("infos", None)
        if not infos or not isinstance(infos, (list, tuple)):
            return True
        info0 = infos[0] if infos else None
        if isinstance(info0, dict):
            for k, v in info0.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    fv = float(v)
                    if np.isfinite(fv):
                        self._buf.setdefault(k, []).append(fv)
        self._step_buf += 1
        if self._step_buf >= self.max_steps:
            for k in list(self._buf.keys()):
                self._buf[k] = self._buf[k][-self.max_steps:]
            self._step_buf = self.max_steps
        if num_timesteps % self.trace_freq != 0:
            return True
        path = os.path.join(self.out_dir, f"trace_train_{self._t0}_{self._trace_idx:04d}_t{num_timesteps}.npz")
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in self._buf.items() if v}
        np.savez_compressed(path, **arrays)
        self._trace_idx += 1
        return True


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_npz(path: str, data: Dict[str, Any]) -> None:
    arrays = {}
    for k, v in data.items():
        try:
            arrays[k] = np.asarray(v)
        except Exception:
            pass
    np.savez_compressed(path, **arrays)



# =============================================================================
# Commands
# =============================================================================

def cmd_train(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
    except Exception as e:
        raise RuntimeError("Install stable-baselines3[extra].") from e

    algo = str(args.algo).lower()
    total_timesteps = int(args.total_timesteps)
    seed = int(args.seed)
    n_envs = int(args.n_envs)
    models_dir = str(args.models_dir)
    log_dir = str(args.log_dir)
    tb_log = str(args.tb_log)
    _ensure_dir(models_dir)
    _ensure_dir(log_dir)
    _ensure_dir(tb_log)
    device = resolve_torch_device(str(args.device))

    curriculum_frac = clamp(float(getattr(args, "curriculum_frac", 0.75)), 0.0, 1.0)
    steps_per_env = float(total_timesteps) / float(max(1, n_envs))
    curric_steps = int(max(1, round(curriculum_frac * steps_per_env)))

    cfg = UUV3DConfig(
        curriculum_steps=curric_steps,
        difficulty_fixed=(float(args.difficulty_fixed) if getattr(args, "difficulty_fixed", None) is not None else None),
        pf_use_numba=bool(getattr(args, "pf_numba", True)),
        action_dt=float(args.action_dt),
        fim_window_s=float(getattr(args, "fim_window", 30.0)),
        log_truth_diagnostics=bool(getattr(args, "log_truth", False)),
        success_mode=str(getattr(args, "success_mode", "true")),
        success_require_std=bool(getattr(args, "success_require_std", True)),
        pf_num_particles=int(getattr(args, "pf_particles", UUV3DConfig.pf_num_particles)),
        obs_history_len=int(getattr(args, "obs_history_len", UUV3DConfig.obs_history_len)),
        info_gate_floor_hard=float(getattr(args, "info_gate_floor_hard", UUV3DConfig.info_gate_floor_hard)),
        info_gate_floor_hard_extra=float(getattr(args, "info_gate_floor_hard_extra", UUV3DConfig.info_gate_floor_hard_extra)),
        info_gate_floor_start_difficulty=float(getattr(args, "info_gate_floor_start_difficulty", UUV3DConfig.info_gate_floor_start_difficulty)),
        info_gate_floor_ramp_difficulty=float(getattr(args, "info_gate_floor_ramp_difficulty", UUV3DConfig.info_gate_floor_ramp_difficulty)),
    )
    cfg_eval = UUV3DConfig(**asdict(cfg))
    cfg_eval.difficulty_fixed = 1.0

    if bool(getattr(args, "numba_warmup", True)) and bool(cfg.pf_use_numba):
        t0 = time.perf_counter()
        warmup_numba_pf()
        print(f"[TRAIN] Numba warmup done in {time.perf_counter() - t0:.2f}s (available={_NUMBA_AVAILABLE})")

    env_fns = [make_env(seed, cfg, render=False, rank=i) for i in range(n_envs)]
    if n_envs == 1:
        venv = DummyVecEnv(env_fns)
    else:
        venv = SubprocVecEnv(env_fns, start_method="spawn")

    vn_path = os.path.join(models_dir, "vecnormalize.pkl")
    if os.path.exists(vn_path) and bool(args.resume):
        env = VecNormalize.load(vn_path, venv)
        env.training = True
        env.norm_reward = True
        vecnorm = env
    else:
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        vecnorm = env

    eval_env = DummyVecEnv([make_env(seed + 10_000, cfg_eval, render=False, rank=0)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.obs_rms = vecnorm.obs_rms

    def _parse_net_arch(s: str) -> List[int]:
        s = str(s).strip()
        if not s:
            return []
        out: List[int] = []
        for p in [q.strip() for q in s.replace(";", ",").split(",") if q.strip()]:
            try:
                n = int(p)
            except Exception:
                continue
            if n > 0:
                out.append(n)
        return out

    net_arch = _parse_net_arch(getattr(args, "net_arch", ""))
    act_name = str(getattr(args, "activation", "relu")).lower().strip()
    policy_kwargs: Dict[str, Any] = {}
    if net_arch:
        policy_kwargs["net_arch"] = net_arch
    try:
        import torch as th
        act_map = {"relu": th.nn.ReLU, "tanh": th.nn.Tanh, "elu": th.nn.ELU, "leaky_relu": th.nn.LeakyReLU}
        if act_name in act_map:
            policy_kwargs["activation_fn"] = act_map[act_name]
    except Exception:
        pass

    ent_coef_arg = str(getattr(args, "ent_coef", "auto_0.2")).strip()
    ent_coef: Any = ent_coef_arg
    if not ent_coef_arg.lower().startswith("auto"):
        try:
            ent_coef = float(ent_coef_arg)
        except Exception:
            ent_coef = ent_coef_arg

    target_entropy_arg = str(getattr(args, "target_entropy", "auto")).strip()
    target_entropy: Any = target_entropy_arg
    if not target_entropy_arg.lower().startswith("auto"):
        try:
            target_entropy = float(target_entropy_arg)
        except Exception:
            target_entropy = target_entropy_arg

    ModelCls = SAC if algo == "sac" else PPO if algo == "ppo" else None
    if ModelCls is None:
        raise ValueError("algo must be sac or ppo")

    model_path_resume = os.path.join(models_dir, "last_model.zip")
    if bool(args.resume) and os.path.exists(model_path_resume):
        model = ModelCls.load(model_path_resume, env=env, device=device, tensorboard_log=tb_log, print_system_info=False)
        print(f"[TRAIN] Resumed from {model_path_resume}")
    else:
        model_kwargs: Dict[str, Any] = dict(
            device=device,
            tensorboard_log=tb_log,
            verbose=1,
            seed=seed,
            policy_kwargs=(policy_kwargs if policy_kwargs else None),
        )
        if algo == "sac":
            model_kwargs["ent_coef"] = ent_coef
            model_kwargs["target_entropy"] = target_entropy
        model = ModelCls("MlpPolicy", env, **model_kwargs)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=log_dir,
        eval_freq=int(args.eval_freq),
        n_eval_episodes=int(args.eval_episodes),
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=int(args.save_freq),
        save_path=models_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    tb_info_cb = TBInfoCallback(log_freq=int(args.tb_info_freq), prefix="env/")
    trace_cb = TraceCallback(out_dir=log_dir, trace_freq=int(args.trace_freq), max_steps=5000)

    class _FuncCallback(BaseCallback):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn
        def _on_step(self) -> bool:
            return bool(self.fn(self.locals, self.globals))

    class _TrainProgressCallback(BaseCallback):
        def __init__(self, total_timesteps: int):
            super().__init__()
            self.total_timesteps = int(total_timesteps)
            self._start_timesteps = None
            self._last_seen = 0
            self._episode_success_num = 0
            self._episode_count = 0
            self._diff_sum = 0.0
            self._diff_count = 0
            self._pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step", dynamic_ncols=True) if tqdm is not None else None
        def _on_training_start(self) -> None:
            if self._pbar is not None:
                self._start_timesteps = int(getattr(self.model, "num_timesteps", 0))
        def _on_step(self) -> bool:
            if self._pbar is None or self._start_timesteps is None:
                return True
            infos = self.locals.get("infos", None)
            dones = self.locals.get("dones", None)
            if infos and isinstance(infos, (list, tuple)):
                for idx, info in enumerate(infos):
                    if not isinstance(info, dict):
                        continue
                    try:
                        d = float(info.get("difficulty", np.nan))
                        if np.isfinite(d):
                            self._diff_sum += d
                            self._diff_count += 1
                    except Exception:
                        pass
                    done = False
                    if dones is not None and idx < len(dones):
                        try:
                            done = bool(dones[idx])
                        except Exception:
                            done = False
                    if done:
                        try:
                            succ = float(info.get("is_success", 0.0))
                        except Exception:
                            succ = 0.0
                        self._episode_success_num += int(succ > 0.5)
                        self._episode_count += 1
                avg_diff = self._diff_sum / max(1.0, float(self._diff_count)) if self._diff_count > 0 else float("nan")
                success_txt = f"{100.0 * self._episode_success_num / max(1, self._episode_count):.1f}% ({self._episode_success_num}/{self._episode_count})" if self._episode_count > 0 else "n/a"
                self._pbar.set_postfix({"difficulty": "n/a" if np.isnan(avg_diff) else f"{avg_diff:.3f}", "succ": success_txt}, refresh=False)
            current = int(getattr(self.model, "num_timesteps", 0)) - self._start_timesteps
            delta = current - self._last_seen
            if delta > 0:
                self._pbar.update(delta)
                self._last_seen = current
            return True
        def _on_training_end(self) -> None:
            if self._pbar is not None:
                if self._last_seen < self.total_timesteps:
                    self._pbar.update(self.total_timesteps - self._last_seen)
                self._pbar.close()

    class ReplayBufferResetCallback(BaseCallback):
        def __init__(self, reset_difficulty: float):
            super().__init__()
            self.reset_difficulty = float(reset_difficulty)
            self._done = False
        def _on_step(self) -> bool:
            if self._done or self.reset_difficulty <= 0.0:
                return True
            infos = self.locals.get("infos", None)
            if not infos:
                return True
            diffs: List[float] = []
            for info in infos:
                if not isinstance(info, dict):
                    continue
                try:
                    fd = float(info.get("difficulty", np.nan))
                except Exception:
                    continue
                if np.isfinite(fd):
                    diffs.append(fd)
            if not diffs:
                return True
            d_mean = float(np.mean(diffs))
            if d_mean >= self.reset_difficulty:
                rb = getattr(self.model, "replay_buffer", None)
                if rb is not None:
                    try:
                        rb.reset()
                        print(f"[TRAIN] Replay buffer reset at mean difficulty={d_mean:.3f} (threshold={self.reset_difficulty:.3f})")
                    except Exception as e:
                        print(f"[TRAIN] Replay buffer reset failed: {type(e).__name__}: {e}")
                self._done = True
            return True

    cb_list = [eval_callback, checkpoint_callback, _FuncCallback(tb_info_cb), _FuncCallback(trace_cb), _TrainProgressCallback(total_timesteps=total_timesteps)]
    if algo == "sac":
        reset_thr = float(getattr(args, "replay_reset_difficulty", 0.0))
        if reset_thr > 0.0:
            cb_list.append(ReplayBufferResetCallback(reset_difficulty=reset_thr))
    callback = CallbackList(cb_list)

    print(f"[TRAIN] algo={algo} total_timesteps={total_timesteps} n_envs={n_envs} vecnorm=True curriculum_steps(per-env)={curric_steps} (frac={curriculum_frac}) net_arch={net_arch if net_arch else 'default'} ent_coef={ent_coef if algo == 'sac' else 'n/a'} target_entropy={target_entropy if algo == 'sac' else 'n/a'} replay_reset_diff={float(getattr(args, 'replay_reset_difficulty', 0.0)) if algo == 'sac' else 0.0} device={device} pf_numba={bool(cfg.pf_use_numba)} numba_available={_NUMBA_AVAILABLE}")
    print(f"[TRAIN] models_dir={models_dir} log_dir={log_dir} tb_log={tb_log}")
    model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=f"{algo}_3d_seed{seed}")
    final_path = os.path.join(models_dir, "final_model.zip")
    model.save(final_path)
    vecnorm.save(vn_path)
    model.save(os.path.join(models_dir, "last_model.zip"))
    print("[TRAIN] saved:", final_path)
    print("[TRAIN] saved:", vn_path)
    env.close()
    eval_env.close()


def cmd_eval(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except Exception as e:
        raise RuntimeError("Install stable-baselines3[extra].") from e

    allowed_controllers = ("rl", "random", "pid", "pid_exc")
    ctrl_arg = str(args.controller).lower().strip()
    controllers_arg = str(getattr(args, "controllers", "")).strip().lower()
    if not controllers_arg:
        controllers = list(allowed_controllers) if ctrl_arg == "all" else [ctrl_arg]
    else:
        controllers = [c.strip() for c in controllers_arg.split(",") if c.strip()]
        if "all" in controllers:
            controllers = list(allowed_controllers)
    controllers = [c for c in controllers if c in allowed_controllers]
    if not controllers:
        raise ValueError("No valid controllers in --controller/--controllers")

    algo = str(args.algo).lower()
    device = resolve_torch_device(str(args.device))
    cfg = UUV3DConfig(
        action_dt=float(args.action_dt),
        render_fps=int(args.render_fps),
        fim_window_s=float(args.fim_window),
        fim_reg_eps=float(args.fim_reg_eps),
        fim_use_gating=bool(args.fim_use_gating),
        log_truth_diagnostics=True,
        success_mode=str(getattr(args, "success_mode", "true")),
        success_require_std=bool(getattr(args, "success_require_std", True)),
        difficulty_fixed=float(getattr(args, "difficulty_fixed", 1.0)),
        pf_use_numba=bool(getattr(args, "pf_numba", True)),
        obs_history_len=int(getattr(args, "obs_history_len", UUV3DConfig.obs_history_len)),
        info_gate_floor_hard=float(getattr(args, "info_gate_floor_hard", UUV3DConfig.info_gate_floor_hard)),
        info_gate_floor_hard_extra=float(getattr(args, "info_gate_floor_hard_extra", UUV3DConfig.info_gate_floor_hard_extra)),
        info_gate_floor_start_difficulty=float(getattr(args, "info_gate_floor_start_difficulty", UUV3DConfig.info_gate_floor_start_difficulty)),
        info_gate_floor_ramp_difficulty=float(getattr(args, "info_gate_floor_ramp_difficulty", UUV3DConfig.info_gate_floor_ramp_difficulty)),
    )

    vn_path = os.path.join(str(args.models_dir), "vecnormalize.pkl")
    has_vecnorm = bool(os.path.exists(vn_path))
    vecenv_for_rl = None
    if "rl" in controllers:
        base_env = DummyVecEnv([make_env(int(args.seed), cfg, render=False, rank=0)])
        if has_vecnorm:
            vecenv_for_rl = VecNormalize.load(vn_path, base_env)
            vecenv_for_rl.training = False
            vecenv_for_rl.norm_reward = False
        else:
            vecenv_for_rl = base_env

    model_rl = None
    if "rl" in controllers:
        model_path = str(args.model)
        if not model_path:
            raise ValueError("--model is required when controller includes rl")
        custom_objects = {"learning_rate": 3e-4, "learning_rate_schedule": None, "lr_schedule": (lambda _progress: 3e-4)}
        if algo == "sac":
            model_rl = SAC.load(model_path, env=vecenv_for_rl, device=device, custom_objects=custom_objects)
        elif algo == "ppo":
            model_rl = PPO.load(model_path, env=vecenv_for_rl, device=device, custom_objects=custom_objects)
        else:
            raise ValueError("algo must be sac or ppo")

    out_dir = str(args.out_dir)
    _ensure_dir(out_dir)
    ts0 = _timestamp()
    compare_meta = {
        "timestamp": ts0,
        "controllers": controllers,
        "algo": algo,
        "model": str(args.model),
        "vecnormalize": has_vecnorm,
        "cfg": asdict(cfg),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "render": bool(args.render),
        "fim_window": float(args.fim_window),
        "fim_reg_eps": float(args.fim_reg_eps),
        "fim_use_gating": bool(args.fim_use_gating),
    }
    compare_meta_path = os.path.join(out_dir, f"eval_compare_meta_{ts0}.json")
    with open(compare_meta_path, "w", encoding="utf-8") as f:
        json.dump(compare_meta, f, indent=2)
    print("[EVAL] compare_meta:", compare_meta_path)

    def _render_single(venv):
        try:
            if hasattr(venv, "envs") and len(getattr(venv, "envs")) > 0:
                venv.envs[0].render()
                return
        except Exception:
            pass
        try:
            if hasattr(venv, "venv") and hasattr(venv.venv, "envs") and len(getattr(venv.venv, "envs")) > 0:
                venv.venv.envs[0].render()
                return
        except Exception:
            pass

    def _run_single_controller(controller: str) -> Dict[str, Any]:
        controller = str(controller).lower().strip()
        env = vecenv_for_rl if controller == "rl" else UUVTwoLeader3DPFEnv(cfg=cfg, render_mode=("human" if bool(args.render) else "none"))
        summary_rows: List[Dict[str, Any]] = []
        for ep in range(int(args.episodes)):
            ep_seed = int(args.seed) + ep
            ctrl_rng = np.random.default_rng(ep_seed + 1_000_000)
            if controller == "rl":
                env.seed(ep_seed)
                obs = env.reset()
                info0: Dict[str, Any] = {}
            else:
                obs, info0 = env.reset(seed=ep_seed)
            done = False
            ep_return = 0.0
            step_idx = 0
            term_reason = ""
            t_end = float("nan")
            success = 0
            success_true = 0
            success_progress = 0
            rows: List[Dict[str, Any]] = []
            npz: Dict[str, Any] = {"t": [], "reward": [], "done": [], "action": [], "obs_norm": [], "obs_raw": []}
            while not done:
                if controller == "rl":
                    action, _ = model_rl.predict(obs, deterministic=True)
                    action_arr = np.asarray(action, dtype=np.float32)
                    if action_arr.ndim == 1:
                        action_arr = action_arr.reshape(1, -1)
                    obs2, reward, dones, infos = env.step(action_arr)
                    done = bool(dones[0])
                    info0 = infos[0] if infos else {}
                    used_action = action_arr[0]
                    obs_norm = np.array(obs, copy=True)
                    obs_raw = obs_norm
                    if hasattr(env, "unnormalize_obs"):
                        try:
                            obs_raw = env.unnormalize_obs(obs_norm)
                        except Exception:
                            pass
                    reward_scalar = float(reward[0])
                else:
                    if controller == "random":
                        used_action = ctrl_rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                    else:
                        used_action = _pid_action_from_info(info0, cfg, controller, step_idx)
                    obs2, reward_scalar, terminated, truncated, step_info = env.step(used_action)
                    info0 = step_info if isinstance(step_info, dict) else info0
                    done = bool(terminated or truncated)
                    obs_norm = np.array([obs2], copy=True)
                    obs_raw = obs_norm
                ep_return += float(reward_scalar)
                if bool(args.render):
                    try:
                        if controller == "rl":
                            _render_single(env)
                        else:
                            env.render()
                    except Exception:
                        pass
                if done:
                    term_reason = str(info0.get("term_reason", ""))
                    success = int(float(info0.get("is_success", 0.0)) > 0.5)
                    success_true = int(float(info0.get("is_success_true_terminal", 0.0)) > 0.5)
                    success_progress = int(float(info0.get("is_success_progress_terminal", 0.0)) > 0.5)
                t_end = float(info0.get("t", np.nan))
                row: Dict[str, Any] = {
                    "episode": ep + 1,
                    "step": step_idx,
                    "t": float(info0.get("t", np.nan)),
                    "reward": float(reward_scalar),
                    "done": int(done),
                    "term_reason": str(info0.get("term_reason", "")),
                    "a_speed": float(used_action.reshape(-1)[0]),
                    "a_yaw": float(used_action.reshape(-1)[1]),
                    "a_pitch": float(used_action.reshape(-1)[2]),
                    "controller": controller,
                }
                for k, v in info0.items():
                    if k in row:
                        continue
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        row[k] = float(v)
                    else:
                        row[k] = str(v)
                obs_raw_flat = _flatten(np.array(obs_raw))
                obs_norm_flat = _flatten(np.array(obs_norm))
                for i, v in enumerate(obs_raw_flat):
                    row[f"obs_raw_{i}"] = float(v)
                for i, v in enumerate(obs_norm_flat):
                    row[f"obs_norm_{i}"] = float(v)
                rows.append(row)
                npz["t"].append(float(info0.get("t", np.nan)))
                npz["reward"].append(float(reward_scalar))
                npz["done"].append(int(done))
                npz["action"].append(_flatten(used_action).astype(np.float32))
                npz["obs_norm"].append(obs_norm_flat.astype(np.float32))
                npz["obs_raw"].append(obs_raw_flat.astype(np.float32))
                obs = obs2
                step_idx += 1
            ts = _timestamp()
            base = os.path.join(out_dir, f"eval_trace_{controller}_{ts}_seed{int(args.seed)}_ep{ep+1:03d}")
            csv_path = base + ".csv"
            npz_path = base + ".npz"
            save_csv(csv_path, rows)
            save_npz(npz_path, npz)
            print(f"[EVAL] controller={controller} ep {ep+1}/{args.episodes} return={ep_return:.2f} term={term_reason} success={success} (true={success_true}, prog={success_progress}) t_end={t_end:.1f}s")
            summary_rows.append({"episode": ep + 1, "seed": ep_seed, "return": float(ep_return), "term_reason": term_reason, "success": int(success), "success_true": int(success_true), "success_progress": int(success_progress), "t_end": float(t_end), "trace_csv": os.path.basename(csv_path), "trace_npz": os.path.basename(npz_path)})
        succ_rate = float(np.mean([r["success"] for r in summary_rows])) if summary_rows else float("nan")
        succ_rate_true = float(np.mean([r.get("success_true", 0) for r in summary_rows])) if summary_rows else float("nan")
        succ_rate_progress = float(np.mean([r.get("success_progress", 0) for r in summary_rows])) if summary_rows else float("nan")
        ctrl_summary = {"controller": controller, "episodes": summary_rows, "success_rate": succ_rate, "success_rate_true": succ_rate_true, "success_rate_progress": succ_rate_progress, "mean_return": float(np.mean([r.get("return", 0.0) for r in summary_rows])) if summary_rows else float("nan"), "mean_t_end": float(np.mean([r.get("t_end", 0.0) for r in summary_rows])) if summary_rows else float("nan"), "term_counts": {}}
        for r in summary_rows:
            tr = str(r.get("term_reason", ""))
            ctrl_summary["term_counts"][tr] = ctrl_summary["term_counts"].get(tr, 0) + 1
        print(f"[EVAL] controller={controller} success_rate={succ_rate:.3f} true={succ_rate_true:.3f} progress={succ_rate_progress:.3f} mean_return={ctrl_summary['mean_return']:.2f} mean_t={ctrl_summary['mean_t_end']:.1f}s")
        if controller != "rl":
            try:
                env.close()
            except Exception:
                pass
        return ctrl_summary

    compare_rows: List[Dict[str, Any]] = []
    for c in controllers:
        compare_rows.append(_run_single_controller(c))
    summary_path = os.path.join(out_dir, f"eval_compare_summary_{ts0}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"meta": compare_meta, "controllers": compare_rows}, f, indent=2)
    print("[EVAL] summary:", summary_path)
    for row in compare_rows:
        ctrl = str(row.get("controller"))
        print(f"  - {ctrl}: success={row['success_rate']:.3f} true={row['success_rate_true']:.3f} progress={row['success_rate_progress']:.3f} mean_return={row['mean_return']:.2f} mean_t={row['mean_t_end']:.1f}s")
    if vecenv_for_rl is not None:
        vecenv_for_rl.close()


def cmd_sim(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except Exception:
        PPO = None
        SAC = None
        DummyVecEnv = None
        VecNormalize = None

    controller = str(args.controller).lower().strip()
    algo = str(args.algo).lower().strip()
    cfg = UUV3DConfig(
        action_dt=float(args.action_dt),
        render_fps=int(args.render_fps),
        fim_window_s=float(args.fim_window),
        log_truth_diagnostics=True,
        difficulty_fixed=1.0,
        success_mode=str(getattr(args, "success_mode", "true")),
        success_require_std=bool(getattr(args, "success_require_std", True)),
        pf_use_numba=bool(getattr(args, "pf_numba", True)),
        obs_history_len=int(getattr(args, "obs_history_len", UUV3DConfig.obs_history_len)),
        info_gate_floor_hard=float(getattr(args, "info_gate_floor_hard", UUV3DConfig.info_gate_floor_hard)),
        info_gate_floor_hard_extra=float(getattr(args, "info_gate_floor_hard_extra", UUV3DConfig.info_gate_floor_hard_extra)),
        info_gate_floor_start_difficulty=float(getattr(args, "info_gate_floor_start_difficulty", UUV3DConfig.info_gate_floor_start_difficulty)),
        info_gate_floor_ramp_difficulty=float(getattr(args, "info_gate_floor_ramp_difficulty", UUV3DConfig.info_gate_floor_ramp_difficulty)),
    )
    base_env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="human")
    model = None
    runner_env = None
    if controller == "rl":
        if PPO is None:
            raise RuntimeError("stable-baselines3 not installed, cannot run controller=rl")
        model_path = str(args.model)
        if not model_path:
            raise ValueError("--model is required for sim --controller rl")
        runner_env = DummyVecEnv([lambda: base_env])
        vn_path = os.path.join(str(args.models_dir), "vecnormalize.pkl")
        if os.path.exists(vn_path) and VecNormalize is not None:
            runner_env = VecNormalize.load(vn_path, runner_env)
            runner_env.training = False
            runner_env.norm_reward = False
        device = resolve_torch_device(str(args.device))
        if algo == "sac":
            model = SAC.load(model_path, env=runner_env, device=device)
        elif algo == "ppo":
            model = PPO.load(model_path, env=runner_env, device=device)
        else:
            raise ValueError("algo must be sac or ppo")
        base_env.manual_override = False
    else:
        base_env.manual_override = True

    for ep in range(int(args.episodes)):
        if runner_env is not None:
            obs = runner_env.reset()
        else:
            obs_raw, _info = base_env.reset(seed=int(args.seed) + ep)
            obs = obs_raw.reshape(1, -1)
        done = False
        while not done:
            if controller == "rl" and runner_env is not None and model is not None:
                if base_env.manual_override:
                    action = np.zeros((1, 3), dtype=np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = np.asarray(action, dtype=np.float32)
                    if action.ndim == 1:
                        action = action.reshape(1, -1)
                obs, _reward, dones, _infos = runner_env.step(action)
                done = bool(dones[0])
                base_env.render()
            else:
                _obs2, _r, terminated, truncated, _info = base_env.step(np.zeros(3, dtype=np.float32))
                done = bool(terminated or truncated)
                base_env.render()
            if base_env._request_quit:
                base_env.close()
                return
            if base_env._request_reset:
                base_env._request_reset = False
                done = True
    base_env.close()


def cmd_map(args: argparse.Namespace) -> None:
    cfg = UUV3DConfig(
        action_dt=float(args.action_dt),
        fim_window_s=float(args.fim_window),
        log_truth_diagnostics=True,
        success_mode=str(getattr(args, "success_mode", "true")),
        success_require_std=bool(getattr(args, "success_require_std", True)),
        difficulty_fixed=(float(args.difficulty_fixed) if getattr(args, "difficulty_fixed", None) is not None else 1.0),
        pf_use_numba=bool(getattr(args, "pf_numba", True)),
        obs_history_len=int(getattr(args, "obs_history_len", UUV3DConfig.obs_history_len)),
        info_gate_floor_hard=float(getattr(args, "info_gate_floor_hard", UUV3DConfig.info_gate_floor_hard)),
        info_gate_floor_hard_extra=float(getattr(args, "info_gate_floor_hard_extra", UUV3DConfig.info_gate_floor_hard_extra)),
        info_gate_floor_start_difficulty=float(getattr(args, "info_gate_floor_start_difficulty", UUV3DConfig.info_gate_floor_start_difficulty)),
        info_gate_floor_ramp_difficulty=float(getattr(args, "info_gate_floor_ramp_difficulty", UUV3DConfig.info_gate_floor_ramp_difficulty)),
    )

    out_dir = str(args.out_dir)
    _ensure_dir(out_dir)
    ts0 = _timestamp()
    out_prefix = str(getattr(args, "out_prefix", "")).strip()
    if not out_prefix:
        out_prefix = os.path.join(out_dir, f"info_map_seed{int(args.seed)}_{ts0}")

    env = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    try:
        _obs, info = env.reset(seed=int(args.seed))
        paths = export_info_map(
            env,
            out_prefix,
            nx=int(args.nx),
            ny=int(args.ny),
            nz=int(args.nz),
            x_min=float(args.x_min),
            x_max=float(args.x_max),
            y_min=float(args.y_min),
            y_max=float(args.y_max),
            z_min=float(args.z_min),
            z_max=float(args.z_max),
        )

        meta_info: Dict[str, Any] = {}
        for k, v in info.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                meta_info[k] = float(v)
            else:
                meta_info[k] = str(v)

        meta = {
            "timestamp": ts0,
            "seed": int(args.seed),
            "paths": paths,
            "cfg": asdict(cfg),
            "grid": {
                "nx": int(args.nx),
                "ny": int(args.ny),
                "nz": int(args.nz),
                "x_min": float(args.x_min),
                "x_max": float(args.x_max),
                "y_min": float(args.y_min),
                "y_max": float(args.y_max),
                "z_min": float(args.z_min),
                "z_max": float(args.z_max),
            },
            "reset_info": meta_info,
        }
        meta_path = out_prefix + "_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print("[MAP] csv:", paths["csv"])
        print("[MAP] npz:", paths["npz"])
        print("[MAP] meta:", meta_path)
    finally:
        env.close()


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train a 3D agent (SAC/PPO) with 2 leaders + PF.")
    pt.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    pt.add_argument("--total-timesteps", type=int, default=6_000_000)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--n-envs", type=int, default=64)
    pt.add_argument("--models-dir", type=str, default="models_3d_v8")
    pt.add_argument("--log-dir", type=str, default="logs_3d_v8")
    pt.add_argument("--tb-log", type=str, default="tb_3d_v8")
    pt.add_argument("--eval-freq", type=int, default=200_000)
    pt.add_argument("--eval-episodes", type=int, default=5)
    pt.add_argument("--save-freq", type=int, default=200_000)
    pt.add_argument("--tb-info-freq", type=int, default=5_000)
    pt.add_argument("--trace-freq", type=int, default=200_000)
    pt.add_argument("--curriculum-frac", type=float, default=0.75)
    pt.add_argument("--difficulty-fixed", type=float, default=None)
    pt.add_argument("--resume", action="store_true")
    pt.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    pt.add_argument("--action-dt", type=float, default=2.0)
    pt.add_argument("--fim-window", type=float, default=30.0)
    pt.add_argument("--pf-particles", type=int, default=1024)
    pt.add_argument("--pf-numba", action=argparse.BooleanOptionalAction, default=True)
    pt.add_argument("--numba-warmup", action=argparse.BooleanOptionalAction, default=True)
    pt.add_argument("--log-truth", action="store_true")
    pt.add_argument("--success-mode", choices=["progress", "true"], default="true")
    pt.add_argument("--success-require-std", action=argparse.BooleanOptionalAction, default=True)
    pt.add_argument("--net-arch", type=str, default="256,256,256")
    pt.add_argument("--activation", choices=["relu", "tanh", "elu", "leaky_relu"], default="relu")
    pt.add_argument("--ent-coef", type=str, default="auto_0.2")
    pt.add_argument("--target-entropy", type=str, default="auto")
    pt.add_argument("--replay-reset-difficulty", type=float, default=0.85)
    pt.add_argument("--obs-history-len", type=int, default=4,
                    help="Frame stack length for compact observations (handles partial observability).")
    pt.add_argument("--info-gate-floor-hard", type=float, default=0.30,
                    help="Minimum information-gate floor in hard regime.")
    pt.add_argument("--info-gate-floor-hard-extra", type=float, default=0.25,
                    help="Extra information-gate floor when observability indicators are weak.")
    pt.add_argument("--info-gate-floor-start-difficulty", type=float, default=0.55,
                    help="Difficulty where information-gate floor starts ramping up.")
    pt.add_argument("--info-gate-floor-ramp-difficulty", type=float, default=0.25,
                    help="Difficulty ramp width for information-gate floor.")

    pe = sub.add_parser("eval", help="Evaluate a trained model; export CSV/NPZ + meta/summary.")
    pe.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    pe.add_argument("--controller", choices=["rl", "random", "pid", "pid_exc", "all"], default="rl")
    pe.add_argument("--controllers", type=str, default="")
    pe.add_argument("--model", type=str, default="")
    pe.add_argument("--models-dir", type=str, default="models_3d_v8")
    pe.add_argument("--episodes", type=int, default=30)
    pe.add_argument("--seed", type=int, default=42)
    pe.add_argument("--action-dt", type=float, default=2.0)
    pe.add_argument("--render-fps", type=int, default=30)
    pe.add_argument("--render", action="store_true")
    pe.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    pe.add_argument("--difficulty-fixed", type=float, default=1.0)
    pe.add_argument("--out-dir", type=str, default="eval_3d_logs_v8")
    pe.add_argument("--fim-window", type=float, default=30.0)
    pe.add_argument("--fim-reg-eps", type=float, default=1e-9)
    pe.add_argument("--fim-use-gating", action="store_true")
    pe.add_argument("--fim-no-gating", dest="fim_use_gating", action="store_false")
    pe.set_defaults(fim_use_gating=True)
    pe.add_argument("--success-mode", choices=["progress", "true"], default="true")
    pe.add_argument("--success-require-std", action=argparse.BooleanOptionalAction, default=True)
    pe.add_argument("--pf-numba", action=argparse.BooleanOptionalAction, default=True)
    pe.add_argument("--obs-history-len", type=int, default=4,
                    help="Frame stack length for compact observations.")
    pe.add_argument("--info-gate-floor-hard", type=float, default=0.30)
    pe.add_argument("--info-gate-floor-hard-extra", type=float, default=0.25)
    pe.add_argument("--info-gate-floor-start-difficulty", type=float, default=0.55)
    pe.add_argument("--info-gate-floor-ramp-difficulty", type=float, default=0.25)

    ps = sub.add_parser("sim", help="Interactive simulation (keyboard) in 3D.")
    ps.add_argument("--controller", choices=["manual", "rl"], default="manual")
    ps.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ps.add_argument("--model", type=str, default="")
    ps.add_argument("--models-dir", type=str, default="models_3d_v8")
    ps.add_argument("--episodes", type=int, default=1)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--action-dt", type=float, default=2.0)
    ps.add_argument("--render-fps", type=int, default=30)
    ps.add_argument("--fim-window", type=float, default=30.0)
    ps.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    ps.add_argument("--success-mode", choices=["progress", "true"], default="true")
    ps.add_argument("--success-require-std", action=argparse.BooleanOptionalAction, default=True)
    ps.add_argument("--pf-numba", action=argparse.BooleanOptionalAction, default=True)
    ps.add_argument("--obs-history-len", type=int, default=4,
                    help="Frame stack length for compact observations.")
    ps.add_argument("--info-gate-floor-hard", type=float, default=0.30)
    ps.add_argument("--info-gate-floor-hard-extra", type=float, default=0.25)
    ps.add_argument("--info-gate-floor-start-difficulty", type=float, default=0.55)
    ps.add_argument("--info-gate-floor-ramp-difficulty", type=float, default=0.25)

    pm = sub.add_parser("map", help="Export a frozen-scenario 3D information map.")
    pm.add_argument("--seed", type=int, default=42)
    pm.add_argument("--difficulty-fixed", type=float, default=1.0)
    pm.add_argument("--out-dir", type=str, default="info_maps_v8")
    pm.add_argument("--out-prefix", type=str, default="")
    pm.add_argument("--nx", type=int, default=25)
    pm.add_argument("--ny", type=int, default=25)
    pm.add_argument("--nz", type=int, default=11)
    pm.add_argument("--x-min", type=float, default=-280.0)
    pm.add_argument("--x-max", type=float, default=60.0)
    pm.add_argument("--y-min", type=float, default=-220.0)
    pm.add_argument("--y-max", type=float, default=220.0)
    pm.add_argument("--z-min", type=float, default=-120.0)
    pm.add_argument("--z-max", type=float, default=20.0)
    pm.add_argument("--action-dt", type=float, default=2.0)
    pm.add_argument("--fim-window", type=float, default=30.0)
    pm.add_argument("--success-mode", choices=["progress", "true"], default="true")
    pm.add_argument("--success-require-std", action=argparse.BooleanOptionalAction, default=True)
    pm.add_argument("--pf-numba", action=argparse.BooleanOptionalAction, default=True)
    pm.add_argument("--obs-history-len", type=int, default=4)
    pm.add_argument("--info-gate-floor-hard", type=float, default=0.30)
    pm.add_argument("--info-gate-floor-hard-extra", type=float, default=0.25)
    pm.add_argument("--info-gate-floor-start-difficulty", type=float, default=0.55)
    pm.add_argument("--info-gate-floor-ramp-difficulty", type=float, default=0.25)

    return p


def main() -> None:
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


if __name__ == "__main__":
    main()
