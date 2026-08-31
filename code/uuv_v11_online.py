# -*- coding: utf-8 -*-
"""v11: deployable, online-only information-tracking environment.

This module intentionally leaves ``uuv_v10_info_tracking.py`` untouched.  It
keeps v10's action space and observation width, while enforcing the following
scientific contract:

* ground-truth follower position is diagnostic only;
* the actor, reward, guard, planner, and primary success use online quantities;
* reward and planner share one online Jacobian/gate/noise convention;
* raw PF spread is kept separate from the legacy PF/CRLB/ESS hybrid diagnostic;
* exogenous and algorithm-internal random streams are independent;
* fixed-horizon success is terminal/dwell based, never merely "hit once".

The truth-based FIM is still accumulated so it can be reported as an offline
oracle diagnostic, but it cannot influence the policy or its reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import uuv_v8_temporal_infofix as _v8
import uuv_v10_info_tracking as _v10
from uuv_v11_metrics import (
    consistency_metrics,
    hybrid_sigma_diagnostic,
    raw_pf_largest_eigen_std,
)
from uuv_v11_rng import (
    EpisodeRNGStreams,
    EpisodeSeedPlan,
    ExogenousNoiseCursor,
    ExogenousNoiseTape,
)


VERSION = "v11_online_1.0"
_V8Env = _v8.UUVTwoLeader3DPFEnv
_V10Env = _v10.UUVTwoLeader3DPFEnv


class HalfOpenFIMTracker3D(_v8.FIMTracker3D):
    """Window tracker for the explicitly documented interval ``(t-T, t]``."""

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
        # The left boundary is excluded.  With a 30 s window and 1 Hz samples,
        # exactly 30 timestamps remain after the window is full.
        while self._events and self._events[0][0] <= t_min + 1e-12:
            _, I_old = self._events.pop(0)
            self.I_win -= I_old
        self.I_win = 0.5 * (self.I_win + self.I_win.T)


class OnlineParticleFilter3D(_v8.ParticleFilter3D):
    """PF with a predictive NIS diagnostic consistent with the manuscript.

    The likelihood update remains the tested v8 implementation.  NIS is
    recomputed *before* that update from the particle-predictive measurement
    mean and variance, including sensor variance.  In v11 NIS-based adaptive
    covariance inflation is disabled by configuration, so the old point-mean
    NIS cannot feed back into the estimator.
    """

    def _predictive_nis(
        self,
        pL_list: List[np.ndarray],
        vL_list: List[np.ndarray],
        vF_meas: np.ndarray,
        s_meas_list: List[Optional[float]],
        gate_factors: Optional[List[float]],
        gate_min_factor: float,
    ) -> Tuple[float, int]:
        v_f = np.asarray(vF_meas, dtype=float).reshape(3)
        factors = gate_factors if gate_factors is not None else [1.0] * len(s_meas_list)
        w = np.asarray(self.w, dtype=float).reshape(-1)
        w_sum = float(np.sum(w))
        if (not np.isfinite(w_sum)) or w_sum <= 1e-18:
            w = np.full(self.N, 1.0 / max(self.N, 1), dtype=float)
        else:
            w = w / w_sum

        sigma2_base = float(max((self.meas_sigma * self.sigma_nis_mult) ** 2, 1e-18))
        nis_sum = 0.0
        count = 0
        for i, s in enumerate(s_meas_list):
            if s is None:
                continue
            g = float(np.clip(factors[i] if i < len(factors) else 0.0, 0.0, 1.0))
            if g < float(gate_min_factor):
                continue
            p_l = np.asarray(pL_list[i], dtype=float).reshape(3)
            v_rel = np.asarray(vL_list[i], dtype=float).reshape(3) - v_f
            r = p_l[None, :] - np.asarray(self.p, dtype=float)
            rho = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
            h = -np.einsum("ij,j->i", r / rho[:, None], v_rel)
            h_mean = float(np.sum(w * h))
            h_var = float(np.sum(w * np.square(h - h_mean)))
            pred_var = max(h_var + sigma2_base / max(g, 1e-3), 1e-18)
            nis_sum += float((float(s) - h_mean) ** 2 / pred_var)
            count += 1
        return float(nis_sum) if count else float("nan"), int(count)

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
    ) -> _v8.PFStats:
        if gate_factors is None and gate_mask is not None:
            gate_factors = [1.0 if bool(x) else 0.0 for x in gate_mask]
        nis_sum, nis_count = self._predictive_nis(
            pL_list,
            vL_list,
            vF_meas,
            s_meas_list,
            gate_factors,
            gate_min_factor,
        )
        stats = super().update_doppler(
            pL_list=pL_list,
            vL_list=vL_list,
            vF_meas=vF_meas,
            s_meas_list=s_meas_list,
            gate_factors=gate_factors,
            gate_mask=gate_mask,
            gate_count_thr=gate_count_thr,
            gate_min_factor=gate_min_factor,
        )
        stats.nis = float(nis_sum)
        stats.nis_ratio = float(nis_sum / nis_count) if nis_count > 0 and np.isfinite(nis_sum) else float("nan")
        return stats


@dataclass
class UUV3DConfig(_v10.UUV3DConfig):
    """v11 defaults; all policy-facing quantities are available online."""

    # Policy and primary endpoints use raw PF covariance.  The legacy hybrid
    # remains available under an explicitly diagnostic name in ``info``.
    use_conservative_std: bool = False
    cons_use_crlb_floor: bool = False
    cons_use_ess_inflation: bool = False
    success_mode: str = "progress"

    # Disable feedback from the legacy point-mean NIS implementation.  v11
    # reports a predictive particle NIS instead.
    pf_nis_infl_gain: float = 0.0
    pf_nis_sigma_adapt_alpha: float = 0.0
    pf_nis_sigma_adapt_decay: float = 0.0
    pf_nis_sigma_adapt_min: float = 1.0
    pf_nis_sigma_adapt_max: float = 1.0

    # No truth-dependent reward term.  These equal tolerances also ensure the
    # inherited reward's shadowed online guard has one unambiguous threshold.
    v10_w_pos_true: float = 0.0
    tol_pos_true_easy: float = 20.0
    tol_pos_true_hard: float = 8.0

    # The online FIM uses PF effective sigma (0.05 m/s in the hard regime), not
    # the oracle 0.02 m/s.  Targets are rescaled before the pilot experiment.
    v10_target_fim_min: float = 0.30
    v10_target_crlb_trace: float = 5.0
    v10_bad_crlb_trace: float = 25.0
    v10_crlb_trace_cap: float = 100.0

    # Primary fixed-horizon reporting.
    v11_dwell_steps: int = 15          # 30 s for action_dt=2 s
    v11_tail_window_steps: int = 50    # final 100 s
    v11_dwell_seconds: float = 30.0
    v11_tail_window_seconds: float = 100.0
    v11_tail_success_fraction: float = 0.80
    v11_success_require_raw_std: bool = True

    # Parameters used only to reproduce and label the old hybrid diagnostic.
    v11_diag_crlb_mult: float = 1.0
    v11_diag_ess_thr_frac: float = 0.40
    v11_diag_ess_k: float = 2.0
    v11_diag_std_cap: float = 500.0

    # Stable stream identity and index-addressed exogenous noise.  The tape is
    # inexpensive (~0.12 MB for a 440 s episode) and is therefore also enabled
    # during training; final evaluation additionally saves it as an artifact.
    v11_env_rank: int = 0
    v11_controller_id: str = "policy"
    v11_generate_noise_tape: bool = True
    v11_variant: str = "auto"

    def __post_init__(self) -> None:
        requested = str(self.v11_variant).lower().strip()
        if requested in {"", "auto"}:
            requested = str(os.environ.get("UUV_V11_VARIANT", "full_online")).lower().strip()
        aliases = {
            "full": "full_online",
            "no_info": "no_fim_crlb",
            "tracking": "tracking_only",
        }
        requested = aliases.get(requested, requested)
        allowed = {"full_online", "no_fim_crlb", "no_planner", "no_guard", "tracking_only"}
        if requested not in allowed:
            raise ValueError(f"unknown v11 variant {requested!r}; expected one of {sorted(allowed)}")
        self.v11_variant = requested
        # Success duration semantics remain invariant if action_dt is changed in
        # a smoke test. Production is fixed at 2 s by the experiment runner.
        self.v11_dwell_steps = max(1, int(math.ceil(self.v11_dwell_seconds / max(self.action_dt, 1e-9))))
        self.v11_tail_window_steps = max(
            1, int(math.ceil(self.v11_tail_window_seconds / max(self.action_dt, 1e-9)))
        )
        # Keep the inherited CLI flag authoritative; there is no second hidden
        # switch for the primary online uncertainty criterion.
        self.v11_success_require_raw_std = bool(self.success_require_std)

        if requested in {"no_fim_crlb", "tracking_only"}:
            self.v10_w_fim_abs = 0.0
            self.v10_w_crlb_abs = 0.0
            self.v10_w_fim_gain = 0.0
            self.v10_w_crlb_gain = 0.0
        if requested == "no_planner":
            self.info_planner_enabled = False
            self.v10_w_info_dir_align = 0.0
        if requested == "no_guard":
            self.v10_info_guard_floor = 1.0
            self.v10_w_track_guard = 0.0
            self.v10_w_est_guard = 0.0
            self.v10_w_track_near = 0.0
        if requested == "tracking_only":
            self.info_planner_enabled = False
            self.v10_w_meas = 0.0
            self.v10_w_sens = 0.0
            self.v10_w_info_dir_align = 0.0
            self.v10_w_std = 0.0
            self.v10_w_track_guard = 0.0
            self.v10_w_track_near = 0.0
            self.v10_w_success_step = 0.0


def _seed_plan_words(plan: EpisodeSeedPlan, controller_id: str) -> Dict[str, int]:
    """Exact 64-bit provenance values for each independently keyed stream."""

    out: Dict[str, int] = {}
    for name in ("scenario", "sensor", "dead_reckoning", "pf"):
        words = plan.seed_sequence(name).generate_state(2, dtype=np.uint32)
        out[name] = int(words[0]) | (int(words[1]) << 32)
    words = plan.seed_sequence("controller", controller_id=str(controller_id)).generate_state(2, dtype=np.uint32)
    out["controller"] = int(words[0]) | (int(words[1]) << 32)
    return out


class UUVTwoLeader3DPFEnv(_V10Env):
    """Fixed-horizon v11 environment with a deployable online actor."""

    # 64 v8 online features + 3 log-scaled online risk features.  This preserves
    # v10's 67 features/frame and 268 features for the default four-frame stack.
    BASE_OBS_DIM = _V8Env.BASE_OBS_DIM + 3

    def __init__(self, cfg: Optional[UUV3DConfig] = None, render_mode: str = "none"):
        super().__init__(cfg=cfg or UUV3DConfig(), render_mode=render_mode)
        self.cfg: UUV3DConfig

        # Correct the window convention without changing legacy modules.
        self.fim_total = HalfOpenFIMTracker3D(window_s=None, reg_eps=self.cfg.fim_reg_eps)
        self.fim_win = HalfOpenFIMTracker3D(window_s=self.cfg.fim_window_s, reg_eps=self.cfg.fim_reg_eps)
        self.fim_hat_win = HalfOpenFIMTracker3D(window_s=self.cfg.fim_window_s, reg_eps=self.cfg.fim_reg_eps)
        self.fim_hat_total = HalfOpenFIMTracker3D(window_s=None, reg_eps=self.cfg.fim_reg_eps)

        # The subclass adds no storage, therefore changing the class preserves
        # all already-configured PF parameters and arrays.
        self.pf.__class__ = OnlineParticleFilter3D
        self.pf.nis_infl_gain = 0.0
        self.pf.nis_sigma_adapt_alpha = 0.0
        self.pf.nis_sigma_adapt_decay = 0.0
        self.pf.nis_sigma_adapt_min = 1.0
        self.pf.nis_sigma_adapt_max = 1.0
        self.pf.sigma_nis_mult = 1.0

        self._v11_master_seed = 0
        self._v11_episode_index = -1
        self._v11_seed_plan = EpisodeSeedPlan(0, 0, int(self.cfg.v11_env_rank))
        self._v11_rng_streams = EpisodeRNGStreams(self._v11_seed_plan, str(self.cfg.v11_controller_id))
        self._v11_stream_seeds = _seed_plan_words(self._v11_seed_plan, str(self.cfg.v11_controller_id))
        self.rng_scenario = self._v11_rng_streams.scenario
        self.rng_doppler = self._v11_rng_streams.sensor
        self.rng_dead_reckoning = self._v11_rng_streams.dead_reckoning
        self.rng_particle_filter = self._v11_rng_streams.pf
        self.rng_controller = self._v11_rng_streams.controller
        self._v11_external_noise_tape: Optional[ExogenousNoiseTape] = None
        self._v11_noise_tape: Optional[ExogenousNoiseTape] = None
        self._v11_noise_cursor: Optional[ExogenousNoiseCursor] = None
        self._v11_last_noise: Dict[str, float] = {}
        self._v11_speed_meas = float(self.speed_F)
        self._v11_yaw_meas = float(self.yaw_F)
        self._v11_pitch_meas = float(self.pitch_F)
        self._v11_vf_meas = np.asarray(self.vF, dtype=float).copy()

        self._v11_ever_online_success = False
        self._v11_dwell_success = False
        self._v11_success_streak = 0
        self._v11_last_recorded_step = -1
        self._v11_success_history: deque[bool] = deque(maxlen=max(1, int(self.cfg.v11_tail_window_steps)))
        self._info_plan_margin_zero = 0.0

    def _info_reset(self) -> None:
        super()._info_reset()
        self._info_plan_margin_zero = 0.0

    def _compute_info_guidance(self) -> None:
        """v8 planner guidance with the documented best-vs-second margin."""

        self._info_reset()
        if not bool(self.cfg.info_planner_enabled):
            return
        tol_pos_est, tol_std, _ = self._current_tolerances()
        raw_std = float(self.pf.std_max())
        assist_gate = self._info_activation_gate(raw_std, tol_std)
        self._info_plan_gate = float(assist_gate)
        if assist_gate < float(self.cfg.info_planner_gate_min):
            return
        support_points, support_weights, p0_eff = self._info_support_points()
        best, zero, second_best_score = self._info_search_action(
            support_points, support_weights, p0_eff, tol_pos_est
        )
        if best is None:
            return

        _, _, _, r_form = self._formation_axes()
        d_form = _v8.project_vec_form(np.asarray(best.disp_world_first, dtype=float), r_form)
        norm = float(np.linalg.norm(d_form))
        d_form = d_form / norm if norm > 1e-9 else np.zeros(3, dtype=float)
        zero_score = float(zero.score) if zero is not None else 0.0
        margin_second = (
            max(float(best.score) - float(second_best_score), 0.0)
            if np.isfinite(second_best_score)
            else 0.0
        )

        self._info_plan_action = np.asarray(best.action, dtype=np.float32).copy()
        self._info_plan_dir_form = np.asarray(d_form, dtype=np.float32).copy()
        self._info_plan_score_best = float(best.score)
        self._info_plan_score_zero = float(zero_score)
        self._info_plan_margin = float(margin_second)
        self._info_plan_margin_zero = max(float(best.score) - zero_score, 0.0)
        self._info_plan_trace_red = float(best.trace_red)
        self._info_plan_eigmin = float(best.eigmin)
        self._info_plan_pred_err = float(best.pred_err)
        self._info_plan_sens_avg = float(best.sens_avg)
        self._info_plan_gate_avg = float(best.gate_avg)
        self._info_plan_active = float(np.isfinite(second_best_score))

    # ------------------------------------------------------------------ RNG

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._v11_master_seed = int(seed)
            self._v11_episode_index = 0
        else:
            self._v11_episode_index = max(0, int(self._v11_episode_index) + 1)

        plan = EpisodeSeedPlan(
            root_seed=int(self._v11_master_seed),
            episode_index=int(self._v11_episode_index),
            env_rank=int(self.cfg.v11_env_rank),
        )
        streams = EpisodeRNGStreams(plan, str(self.cfg.v11_controller_id))
        self._v11_seed_plan = plan
        self._v11_rng_streams = streams
        self._v11_stream_seeds = _seed_plan_words(plan, str(self.cfg.v11_controller_id))

        # Passing seed=None is deliberate: the superclass then consumes the
        # exact named scenario generator installed here.  Its later PF draw is
        # discarded and redrawn from the independent PF stream below.
        self.rng = streams.scenario
        self.pf.rng = streams.scenario
        super().reset(seed=None, options=options)
        self.rng_scenario = self.rng
        self.rng_doppler = streams.sensor
        self.rng_dead_reckoning = streams.dead_reckoning
        self.rng_particle_filter = streams.pf
        self.rng_controller = streams.controller
        self.rng = self.rng_doppler  # compatibility alias; v11 substeps use named streams
        self.pf.rng = self.rng_particle_filter
        self.pf.sigma_nis_mult = 1.0

        d = float(self._difficulty)
        rho_min = _v8.lerp(60.0, self.cfg.start_rho_min, d)
        rho_max = _v8.lerp(140.0, self.cfg.start_rho_max, d)
        cos_phi_max = math.sin(_v8.deg2rad(_v8.lerp(self.cfg.init_dir_band_deg_easy, 90.0, d)))
        self._cos_phi_max_pf = float(cos_phi_max)
        self.pf.reset_sphere_shell_band(
            center=0.5 * (self.pL1 + self.pL2),
            rho_min=rho_min,
            rho_max=rho_max,
            cos_phi_max=cos_phi_max,
        )

        tape = self._v11_external_noise_tape
        if tape is not None and tape.seed_plan != plan:
            raise ValueError("attached exogenous noise tape does not match this episode seed plan")
        if tape is None and bool(self.cfg.v11_generate_noise_tape):
            n_substeps = int(self.cfg.max_steps) * max(1, int(round(self.cfg.action_dt / self.cfg.sub_dt)))
            duration = float(self.cfg.max_steps) * float(self.cfg.action_dt)
            n_measurements = int(math.ceil(duration / max(float(self.cfg.s_meas_period), 1e-9))) + 2
            tape = ExogenousNoiseTape.generate(
                plan,
                n_substeps=n_substeps,
                n_doppler_measurements=n_measurements,
            )
        self._v11_noise_tape = tape
        self._v11_noise_cursor = ExogenousNoiseCursor(tape) if tape is not None else None

        # Recompute all reset diagnostics after the independent PF draw.
        stds = self.pf.stds()
        self.prev_unc_metric = float(sum(stds))
        _, p_f_des = self._formation_desired()
        self.prev_err_est = float(np.linalg.norm(self.pf.mean - p_f_des))
        self._err_est_prev = float(self.prev_err_est)
        self._unc_metric_prev = float(self.prev_unc_metric)
        self.prev_std_max = float(self.pf.std_max())
        self.prev_eigmin_hat = float(self.fim_hat_win.eig_stats(self.fim_hat_win.I_win)[0])
        self._v10_prev_err_est = float(self.prev_err_est)
        self._v10_prev_log_fim = 0.0
        self._v10_prev_log_crlb = math.log1p(float(self.cfg.v10_crlb_trace_cap))

        self._v11_ever_online_success = False
        self._v11_dwell_success = False
        self._v11_success_streak = 0
        self._v11_last_recorded_step = -1
        self._v11_success_history = deque(maxlen=max(1, int(self.cfg.v11_tail_window_steps)))
        self._v11_last_noise = {}
        self._v11_speed_meas = float(self.speed_F)
        self._v11_yaw_meas = float(self.yaw_F)
        self._v11_pitch_meas = float(self.pitch_F)
        self._v11_vf_meas = np.asarray(self.vF, dtype=float).copy()
        self._info_reset()
        self._reset_step_accums()
        self._last_action_raw = np.zeros(3, dtype=np.float32)
        base_obs = self._get_obs_base()
        self._reset_obs_history(base_obs)
        obs = self._stack_obs_history()
        info = self._get_info(extra={"difficulty": float(self._difficulty), "term_reason": "reset"})
        return obs, info

    def attach_exogenous_noise_tape(self, tape: Optional[ExogenousNoiseTape]) -> None:
        """Use a saved common-random-number tape on the next reset."""

        if tape is not None and not isinstance(tape, ExogenousNoiseTape):
            raise TypeError("tape must be ExogenousNoiseTape or None")
        self._v11_external_noise_tape = tape

    @property
    def exogenous_noise_tape(self) -> Optional[ExogenousNoiseTape]:
        return self._v11_noise_tape

    # ----------------------------------------------------------- observation

    @staticmethod
    def _log_ratio_feature(value: float, cap: float) -> float:
        cap = max(float(cap), 1.0)
        if not np.isfinite(value):
            value = cap
        return float(math.log1p(float(np.clip(value, 0.0, cap))) / math.log1p(cap))

    def _online_crlb_std_max(self) -> float:
        try:
            c = np.asarray(self.fim_hat_win.crlb(use_window=True), dtype=float).reshape(3, 3)
            eig = np.linalg.eigvalsh(0.5 * (c + c.T))
            return float(math.sqrt(max(float(eig[-1]), 0.0)))
        except Exception:
            return float("inf")

    def _get_obs_base(self) -> np.ndarray:
        # Calling v8 directly bypasses v10's ground-truth tracking-ratio slot.
        speed_truth, yaw_truth, pitch_truth = float(self.speed_F), float(self.yaw_F), float(self.pitch_F)
        vf_truth = np.asarray(self.vF, dtype=float).copy()
        self.speed_F = float(self._v11_speed_meas)
        self.yaw_F = float(self._v11_yaw_meas)
        self.pitch_F = float(self._v11_pitch_meas)
        self.vF = np.asarray(self._v11_vf_meas, dtype=float).copy()
        try:
            # This also makes the embedded planner use the latest onboard
            # dead-reckoning measurement rather than simulator kinematic truth.
            obs = _V8Env._get_obs_base(self).astype(np.float32)
        finally:
            self.speed_F, self.yaw_F, self.pitch_F = speed_truth, yaw_truth, pitch_truth
            self.vF = vf_truth
        _, p_f_des = self._formation_desired()
        tol_pos_est, tol_std, _ = self._current_tolerances()
        err_est = float(np.linalg.norm(np.asarray(self.pf.mean) - p_f_des))
        raw_std = float(self.pf.std_max())
        crlb_std = self._online_crlb_std_max()
        cap = float(self.cfg.v10_obs_ratio_cap)
        extra = np.asarray(
            [
                self._log_ratio_feature(err_est / max(tol_pos_est, 1e-6), cap),
                self._log_ratio_feature(raw_std / max(tol_std, 1e-6), cap),
                self._log_ratio_feature(crlb_std / max(tol_std, 1e-6), cap),
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs, extra], axis=0).astype(np.float32)

    # --------------------------------------------------------------- metrics

    def _v10_window_info_metrics(self) -> Dict[str, float]:
        """Compatibility hook: v10 reward now receives the online FIM only."""

        i_win = np.asarray(self.fim_hat_win.I_win, dtype=float).reshape(3, 3)
        fim_eig_min = float(self.fim_hat_win.eig_stats(i_win)[0])
        fim_trace = float(np.trace(i_win)) if np.all(np.isfinite(i_win)) else 0.0
        c_win = self.fim_hat_win.crlb(use_window=True)
        crlb_trace = float(np.trace(c_win)) if np.all(np.isfinite(c_win)) else float(self.cfg.v10_bad_crlb_trace)
        meas_frac = float(self.meas_used_step / max(self.meas_total_step, 1)) if self.meas_total_step > 0 else 0.0
        gate_avg = float(self.gate_avg_step) if np.isfinite(self.gate_avg_step) else 0.0
        sens_avg = float(self.sens_accum / max(float(self.sens_count), 1.0))
        self.sens_avg_step = float(sens_avg)
        return {
            "fim_win_eig_min": max(fim_eig_min, 0.0) if np.isfinite(fim_eig_min) else 0.0,
            "fim_win_trace": max(fim_trace, 0.0) if np.isfinite(fim_trace) else 0.0,
            "crlb_win_trace": max(crlb_trace, 0.0) if np.isfinite(crlb_trace) else float(self.cfg.v10_bad_crlb_trace),
            "meas_used_frac": float(np.clip(meas_frac, 0.0, 1.0)),
            "gate_avg": float(np.clip(gate_avg, 0.0, 1.0)),
            "sens_avg": max(sens_avg, 0.0) if np.isfinite(sens_avg) else 0.0,
        }

    def _legacy_hybrid_sigma_diagnostic(self) -> Tuple[float, float, float]:
        raw = float(raw_pf_largest_eigen_std(self.pf.cov))
        crlb = float(self._online_crlb_std_max())
        w = np.asarray(self.pf.w, dtype=float)
        denom = float(np.sum(np.square(w)))
        ess = float(1.0 / denom) if denom > 1e-18 else float(self.pf.N)
        diagnostic = hybrid_sigma_diagnostic(
            raw,
            crlb_largest_std=crlb,
            crlb_multiplier=float(self.cfg.v11_diag_crlb_mult),
            ess=ess,
            particle_count=int(self.pf.N),
            ess_threshold_fraction=float(self.cfg.v11_diag_ess_thr_frac),
            ess_inflation_gain=float(self.cfg.v11_diag_ess_k),
            cap=float(self.cfg.v11_diag_std_cap),
        )
        return raw, crlb, float(diagnostic.sigma_eff_hybrid)

    # --------------------------------------------------------------- success

    def _online_success_components(self) -> Tuple[bool, bool, bool, float, float]:
        _, p_f_des = self._formation_desired()
        tol_pos_est, tol_std, _ = self._current_tolerances()
        err_est = float(np.linalg.norm(np.asarray(self.pf.mean) - p_f_des))
        raw_std = float(self.pf.std_max())
        formation_ok = bool(err_est < tol_pos_est)
        uncertainty_ok = bool(raw_std < tol_std)
        success = formation_ok and (uncertainty_ok if bool(self.cfg.v11_success_require_raw_std) else True)
        return formation_ok, uncertainty_ok, bool(success), err_est, raw_std

    def _record_online_success(self, success: bool) -> None:
        if int(self._v11_last_recorded_step) == int(self.step_count):
            return
        self._v11_last_recorded_step = int(self.step_count)
        self._v11_success_history.append(bool(success))
        self._v11_ever_online_success = bool(self._v11_ever_online_success or success)
        self._v11_success_streak = int(self._v11_success_streak + 1) if success else 0
        if self._v11_success_streak >= max(1, int(self.cfg.v11_dwell_steps)):
            self._v11_dwell_success = True

    def _v10_success_flags(self, err_est: float, err_true_form: float, std_max: float) -> Tuple[bool, bool, bool]:
        del err_est, err_true_form, std_max
        formation_ok, uncertainty_ok, success, _, _ = self._online_success_components()
        self._record_online_success(success)
        # Both compatibility flags intentionally refer to the same online-only
        # primary task inside the reward.  Truth success is logged separately.
        return bool(success), bool(success), bool(success)

    # ---------------------------------------------------------------- reward

    def _compute_reward(
        self,
        pf_stats_last: Optional[_v8.PFStats],
        planner_action_for_reward: Optional[np.ndarray] = None,
        planner_gate_for_reward: Optional[float] = None,
        planner_margin_for_reward: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        # v10's well-tested shaping can be reused safely by shadowing the only
        # truth position it reads.  FIM dispatch is already redirected above,
        # std_max is raw in the v11 config, and the truth weight is exactly zero.
        p_f_truth = np.asarray(self.pF, dtype=float).copy()
        speed_truth, yaw_truth, pitch_truth = float(self.speed_F), float(self.yaw_F), float(self.pitch_F)
        vf_truth = np.asarray(self.vF, dtype=float).copy()
        self.pF = np.asarray(self.pf.mean, dtype=float).copy()
        self.speed_F = float(self._v11_speed_meas)
        self.yaw_F = float(self._v11_yaw_meas)
        self.pitch_F = float(self._v11_pitch_meas)
        self.vF = np.asarray(self._v11_vf_meas, dtype=float).copy()
        try:
            reward, terms = super()._compute_reward(
                pf_stats_last,
                planner_action_for_reward,
                planner_gate_for_reward,
                planner_margin_for_reward,
            )
        finally:
            self.pF = p_f_truth
            self.speed_F, self.yaw_F, self.pitch_F = speed_truth, yaw_truth, pitch_truth
            self.vF = vf_truth

        _, p_f_des = self._formation_desired()
        true_form_error = float(np.linalg.norm(p_f_truth - p_f_des))
        localization_error = float(np.linalg.norm(np.asarray(self.pf.mean) - p_f_truth))
        formation_ok, uncertainty_ok, success, err_est, raw_std = self._online_success_components()

        terms.pop("v10_r_pos_true", None)
        terms.pop("v10_true_ratio", None)
        terms.pop("tracking_success_true_ever", None)
        terms["v11_version"] = 11.0
        terms["v11_reward_uses_truth"] = 0.0
        terms["v11_information_source_online"] = 1.0
        terms["v11_err_est"] = float(err_est)
        terms["v11_pf_std_max_raw"] = float(raw_std)
        terms["v11_success_formation_now"] = float(formation_ok)
        terms["v11_success_uncertainty_now"] = float(uncertainty_ok)
        terms["v11_success_online_now"] = float(success)
        terms["v11_guard_est_ratio"] = float(terms.get("v10_est_ratio", 0.0))
        terms["truth_form_error_diagnostic"] = true_form_error
        terms["truth_localization_error_diagnostic"] = localization_error
        # Preserve the old field for plotting, but its name no longer implies it
        # entered the reward.
        terms["err_true_form"] = true_form_error
        _, tol_std, tol_pos_true = self._current_tolerances()
        truth_success = bool(true_form_error < tol_pos_true and localization_error < tol_std)
        terms["success_true_now"] = float(truth_success)
        terms["success_truth_diagnostic_now"] = float(truth_success)
        return float(reward), terms

    # -------------------------------------------------------------- dynamics

    def _sim_substep(self, speed_cmd: float, yaw_rate_cmd: float, pitch_rate_cmd: float, dt: float) -> _v8.PFStats:
        """v8 dynamics with independent dead-reckoning, Doppler, and PF RNGs."""

        dt = float(max(1e-6, dt))
        self.speed_F = float(np.clip(self.speed_F + speed_cmd * dt, self.cfg.f_min_speed, self.cfg.f_max_speed))
        self.yaw_F = _v8.wrap360(self.yaw_F + yaw_rate_cmd * dt)
        self.pitch_F = _v8.clamp(self.pitch_F + pitch_rate_cmd * dt, self.cfg.pitch_min_deg, self.cfg.pitch_max_deg)
        self.vF = _v8.vel_from_speed_yaw_pitch(self.speed_F, self.yaw_F, self.pitch_F)

        self.vL1 = _v8.vel_from_speed_yaw_pitch(self.leader1_speed, self.yaw_L1, 0.0)
        self.vL2 = _v8.vel_from_speed_yaw_pitch(self.leader2_speed, self.yaw_L2, 0.0)
        self.pL1 = self.pL1 + self.vL1 * dt
        self.pL2 = self.pL2 + self.vL2 * dt
        self.pF = self.pF + self.vF * dt
        self.t += dt

        if self._v11_noise_cursor is not None:
            dr_noise = self._v11_noise_cursor.next_dead_reckoning(
                [self.cfg.sigma_speed, self.cfg.sigma_yaw_deg, self.cfg.sigma_pitch_deg]
            )
            n_speed, n_yaw, n_pitch = (float(x) for x in dr_noise)
        else:
            n_speed = float(self.rng_dead_reckoning.normal(0.0, self.cfg.sigma_speed))
            n_yaw = float(self.rng_dead_reckoning.normal(0.0, self.cfg.sigma_yaw_deg))
            n_pitch = float(self.rng_dead_reckoning.normal(0.0, self.cfg.sigma_pitch_deg))
        self._v11_last_noise.update({"dr_speed": n_speed, "dr_yaw": n_yaw, "dr_pitch": n_pitch})
        speed_meas = float(self.speed_F + n_speed)
        yaw_meas = float(self.yaw_F + n_yaw)
        pitch_meas = float(self.pitch_F + n_pitch)
        v_f_meas = _v8.vel_from_speed_yaw_pitch(speed_meas, yaw_meas, pitch_meas)
        self._v11_speed_meas = speed_meas
        self._v11_yaw_meas = yaw_meas
        self._v11_pitch_meas = pitch_meas
        self._v11_vf_meas = np.asarray(v_f_meas, dtype=float).copy()
        self.pf.predict(vF_meas=v_f_meas, dt=dt)

        pf_stats = _v8.PFStats()
        while self.t + 1e-12 >= self.next_s_time:
            t_meas = float(self.next_s_time)
            r1_true = self.pL1 - self.pF
            r2_true = self.pL2 - self.pF
            vrel1_true = self.vL1 - self.vF
            vrel2_true = self.vL2 - self.vF
            s1_true = _v8.radial_speed(r1_true, vrel1_true)
            s2_true = _v8.radial_speed(r2_true, vrel2_true)
            if self._v11_noise_cursor is not None:
                doppler_noise = self._v11_noise_cursor.next_doppler(self.cfg.sigma_s_true)
                n_s1, n_s2 = (float(x) for x in doppler_noise)
            else:
                n_s1 = float(self.rng_doppler.normal(0.0, self.cfg.sigma_s_true))
                n_s2 = float(self.rng_doppler.normal(0.0, self.cfg.sigma_s_true))
            self._v11_last_noise.update({"doppler_l1": n_s1, "doppler_l2": n_s2})
            s1 = float(s1_true + n_s1)
            s2 = float(s2_true + n_s2)
            self.s1_last, self.s2_last = s1, s2
            self._has_s1 = self._has_s2 = True

            # Oracle geometry: diagnostic only.
            gate1 = self._doppler_gate_truth(r1_true, vrel1_true)
            gate2 = self._doppler_gate_truth(r2_true, vrel2_true)
            self.fim_step_meas += 2
            if (not self.cfg.fim_use_gating) or gate1:
                h1 = _v8.doppler_H_3d(r1_true, vrel1_true)
                i1 = (h1.T @ h1) / max(self.cfg.sigma_s_true ** 2, 1e-18)
                self.fim_total.add_I(t_meas, i1)
                self.fim_win.add_I(t_meas, i1)
                self.fim_step_used += 1
            if (not self.cfg.fim_use_gating) or gate2:
                h2 = _v8.doppler_H_3d(r2_true, vrel2_true)
                i2 = (h2.T @ h2) / max(self.cfg.sigma_s_true ** 2, 1e-18)
                self.fim_total.add_I(t_meas, i2)
                self.fim_win.add_I(t_meas, i2)
                self.fim_step_used += 1

            # Online historical geometry used by reward/observation/reporting.
            # The predictive planner is a distinct belief-support surrogate but
            # uses the same Jacobian, soft-gate, and noise convention.
            gate_factors = self._doppler_gate_pf_factors(vF_meas=v_f_meas)
            self._last_gate_pf_current = [float(gate_factors[0]), float(gate_factors[1])]
            p_f_hat_pred = np.asarray(self.pf.mean, dtype=float).copy()
            sigma_hat = float(self.pf.meas_sigma * self.pf.sigma_nis_mult)
            sigma2_hat = max(sigma_hat * sigma_hat, 1e-18)
            for i, (p_l, v_l) in enumerate(((self.pL1, self.vL1), (self.pL2, self.vL2))):
                g = float(gate_factors[i]) if i < len(gate_factors) else 0.0
                if g <= 0.0:
                    continue
                r_hat = np.asarray(p_l - p_f_hat_pred, dtype=float)
                v_rel_hat = np.asarray(v_l - v_f_meas, dtype=float)
                h_hat = _v8.doppler_H_3d(r_hat, v_rel_hat)
                i_inc = (h_hat.T @ h_hat) / sigma2_hat
                self.fim_hat_total.add_I(t_meas, g * i_inc)
                self.fim_hat_win.add_I(t_meas, g * i_inc)
                if g >= float(self.cfg.gate_count_thr):
                    self.fim_hat_step_used += 1

            pf_stats = self.pf.update_doppler(
                pL_list=[self.pL1, self.pL2],
                vL_list=[self.vL1, self.vL2],
                vF_meas=v_f_meas,
                s_meas_list=[s1, s2],
                gate_factors=gate_factors,
                gate_count_thr=float(self.cfg.gate_count_thr),
                gate_min_factor=float(self.cfg.gate_min_factor),
            )

            injected = 0
            if self._pf_inject_frac > 0.0 and int(pf_stats.resampled) == 1:
                injected = self.pf.inject_sphere_shell_band(
                    center=0.5 * (self.pL1 + self.pL2),
                    rho_min=float(self.cfg.pf_inject_rho_min),
                    rho_max=float(self.cfg.start_rho_max) * float(self.cfg.pf_inject_rho_max_mult),
                    cos_phi_max=float(self._cos_phi_max_pf),
                    frac=float(self._pf_inject_frac),
                    mass=float(self.cfg.pf_inject_mass),
                )
            pf_stats.injected = int(injected)
            self.pf_injected_step += int(injected)

            self.meas_total_step += int(pf_stats.meas_total)
            self.meas_used_step += int(pf_stats.used_meas)
            if gate_factors[0] > 0.0:
                self._accum_sens(self.pL1, self.vL1, v_f_meas, weight=float(gate_factors[0]))
            if gate_factors[1] > 0.0:
                self._accum_sens(self.pL2, self.vL2, v_f_meas, weight=float(gate_factors[1]))

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

    # ---------------------------------------------------------- termination

    def _check_done(self) -> Tuple[bool, bool, str]:
        formation_ok, uncertainty_ok, success, _, _ = self._online_success_components()
        self._episode_progress_success = bool(formation_ok)
        self._episode_true_success = bool(success)  # compatibility: online primary
        self._record_online_success(success)
        self.success_streak = int(self._v11_success_streak)
        if self.step_count >= int(self.cfg.max_steps):
            return False, True, "max_steps"
        return False, False, "running"

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)
        # Fair controller-facing legacy fields: PID baselines historically read
        # these names, so in v11 they carry the same onboard measurements seen
        # by the RL actor. Simulator truth is retained only under explicit names.
        speed_truth = float(info.get("speed_F", self.speed_F))
        yaw_truth = float(info.get("yaw_F", self.yaw_F))
        pitch_truth = float(info.get("pitch_F", self.pitch_F))
        info["speed_F_truth_diagnostic"] = speed_truth
        info["yaw_F_truth_diagnostic"] = yaw_truth
        info["pitch_F_truth_diagnostic"] = pitch_truth
        info["speed_F"] = float(self._v11_speed_meas)
        info["yaw_F"] = float(self._v11_yaw_meas)
        info["pitch_F"] = float(self._v11_pitch_meas)
        raw_std, crlb_std, hybrid = self._legacy_hybrid_sigma_diagnostic()
        formation_ok, uncertainty_ok, success, err_est, _ = self._online_success_components()
        done = bool(getattr(self, "_last_terminated", False) or getattr(self, "_last_truncated", False))
        hist = list(self._v11_success_history)
        tail_occupancy = float(np.mean(hist)) if hist else 0.0
        tail_ready = len(hist) >= max(1, int(self.cfg.v11_tail_window_steps))
        tail_success = bool(tail_ready and tail_occupancy >= float(self.cfg.v11_tail_success_fraction))
        _, p_f_des = self._formation_desired()
        _, tol_std, tol_pos_true = self._current_tolerances()
        truth_form_error = float(np.linalg.norm(np.asarray(self.pF) - p_f_des))
        truth_localization_error = float(np.linalg.norm(np.asarray(self.pf.mean) - np.asarray(self.pF)))
        truth_success = bool(truth_form_error < tol_pos_true and truth_localization_error < tol_std)

        i_online = np.asarray(self.fim_hat_win.I_win, dtype=float)
        eig_online = self.fim_hat_win.eig_stats(i_online)
        c_online = self.fim_hat_win.crlb(use_window=True)
        i_oracle = np.asarray(self.fim_win.I_win, dtype=float)
        eig_oracle = self.fim_win.eig_stats(i_oracle)

        info.update(
            {
                "v11_version": 11.0,
                "policy_uses_truth": 0.0,
                "reward_uses_truth": 0.0,
                "err_est_online": float(err_est),
                "pf_std_max_raw": float(raw_std),
                "crlb_online_std_max_surrogate": float(crlb_std),
                "sigma_eff_hybrid_diagnostic": float(hybrid),
                "fim_online_win_eig_min": float(eig_online[0]),
                "fim_online_win_eig_max": float(eig_online[1]),
                "fim_online_win_cond": float(eig_online[2]),
                "crlb_online_win_trace_surrogate": float(np.trace(c_online)) if np.all(np.isfinite(c_online)) else float("nan"),
                "fim_oracle_win_eig_min_diagnostic": float(eig_oracle[0]),
                "fim_oracle_win_eig_max_diagnostic": float(eig_oracle[1]),
                "success_formation_online_now": float(formation_ok),
                "success_uncertainty_raw_now": float(uncertainty_ok),
                "success_online_now": float(success),
                "success_online_terminal": float(done and success),
                "success_online_ever_diagnostic": float(self._v11_ever_online_success),
                "success_online_dwell": float(self._v11_dwell_success),
                "success_online_tail_occupancy": float(tail_occupancy),
                "success_online_tail80": float(tail_success),
                "is_success": float(done and success),
                "success": float(done and success),
                "is_success_progress_terminal": float(done and formation_ok),
                "is_success_true_terminal": float(done and truth_success),
                "success_truth_diagnostic_now": float(truth_success),
                "success_truth_diagnostic_terminal": float(done and truth_success),
                "rng_master_seed": float(self._v11_master_seed),
                "rng_episode_index": float(self._v11_episode_index),
                "rng_env_rank": float(self.cfg.v11_env_rank),
                "rng_noise_tape_active": float(self._v11_noise_tape is not None),
                "rng_controller_id": str(self.cfg.v11_controller_id),
                "speed_F_online_meas": float(self._v11_speed_meas),
                "yaw_F_online_meas": float(self._v11_yaw_meas),
                "pitch_F_online_meas": float(self._v11_pitch_meas),
                "info_plan_margin_best_second": float(self._info_plan_margin),
                "info_plan_margin_best_zero_diagnostic": float(self._info_plan_margin_zero),
            }
        )
        try:
            consistency = consistency_metrics(self.pf.mean, self.pF, self.pf.cov)
            info.update(
                {
                    "localization_error_diagnostic": float(consistency.localization_error),
                    "nees_diagnostic": float(consistency.nees),
                    "coverage_95_diagnostic": float(consistency.covered_by_95pct_ellipsoid),
                }
            )
        except Exception:
            info.update(
                {
                    "localization_error_diagnostic": float("nan"),
                    "nees_diagnostic": float("nan"),
                    "coverage_95_diagnostic": float("nan"),
                }
            )
        for name, value in self._v11_stream_seeds.items():
            # Use strings for exact 64-bit provenance; float would lose bits.
            info[f"rng_seed_{name}"] = str(value)
        return info


# =============================================================================
# CLI compatibility with v10/v8
# =============================================================================


def make_env(seed: int, cfg: UUV3DConfig, render: bool, rank: int = 0):
    def _init():
        local_cfg = replace(cfg, v11_env_rank=int(rank))
        env = UUVTwoLeader3DPFEnv(cfg=local_cfg, render_mode=("human" if render else "none"))
        # VecEnv performs the first reset.  Priming the master seed without an
        # otherwise discarded reset keeps its first scientific episode at index 0.
        env._v11_master_seed = int(seed)
        env._v11_episode_index = -1
        return env

    return _init


def _iter_parser_actions(parser: argparse.ArgumentParser):
    yield from _v10._iter_parser_actions(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = _v10.build_parser()
    replacements = {
        "models_3d_v10_info_tracking_guarded": "models_3d_v11_online",
        "logs_3d_v10_info_tracking_guarded": "logs_3d_v11_online",
        "tb_3d_v10_info_tracking_guarded": "tb_3d_v11_online",
        "eval_3d_logs_v10_info_tracking_guarded": "eval_3d_logs_v11_online",
        "info_maps_v10_info_tracking_guarded": "info_maps_v11_online",
    }
    for action in _iter_parser_actions(parser):
        default = getattr(action, "default", None)
        if isinstance(default, str):
            for old, new in replacements.items():
                if old in default:
                    action.default = default.replace(old, new)
                    break
        production_defaults = {
            "total_timesteps": 25_000_000,
            "n_envs": 24,
            "net_arch": "512,512,512",
            "success_mode": "progress",
            "action_dt": 2.0,
            "pf_particles": 1024,
        }
        if getattr(action, "dest", None) in production_defaults:
            action.default = production_defaults[action.dest]
        if getattr(action, "dest", None) == "success_mode":
            action.choices = ("progress",)
    train_parser = _v10._get_subparser(parser, "train")
    if train_parser is not None and not _v10._parser_has_dest(train_parser, "v11_variant"):
        train_parser.add_argument(
            "--v11-variant",
            choices=("full_online", "no_fim_crlb", "no_planner", "no_guard", "tracking_only"),
            default="full_online",
            help="Frozen v11 ablation variant.",
        )
    return parser


@contextmanager
def _patched_v10_globals(*, patch_env_class: bool):
    old = {
        "UUV3DConfig": _v10.UUV3DConfig,
        "UUVTwoLeader3DPFEnv": _v10.UUVTwoLeader3DPFEnv,
        "make_env": _v10.make_env,
        "v8_UUV3DConfig": _v8.UUV3DConfig,
        "v8_UUVTwoLeader3DPFEnv": _v8.UUVTwoLeader3DPFEnv,
        "v8_make_env": _v8.make_env,
    }
    _v10.UUV3DConfig = UUV3DConfig
    _v10.make_env = make_env
    # Do not replace the v10 class symbol during SubprocVecEnv construction.
    # Cloudpickle otherwise resolves the zero-argument-super __class__ cell to
    # the v11 symbol in spawned workers. Training only needs our make_env.
    if patch_env_class:
        _v10.UUVTwoLeader3DPFEnv = UUVTwoLeader3DPFEnv
    try:
        yield
    finally:
        _v10.UUV3DConfig = old["UUV3DConfig"]
        _v10.UUVTwoLeader3DPFEnv = old["UUVTwoLeader3DPFEnv"]
        _v10.make_env = old["make_env"]
        _v8.UUV3DConfig = old["v8_UUV3DConfig"]
        _v8.UUVTwoLeader3DPFEnv = old["v8_UUVTwoLeader3DPFEnv"]
        _v8.make_env = old["v8_make_env"]


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


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
        info_gate_floor_hard_extra=float(
            getattr(args, "info_gate_floor_hard_extra", UUV3DConfig.info_gate_floor_hard_extra)
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
        v11_variant=str(getattr(args, "v11_variant", "full_online")),
    )


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for module_name in ("stable_baselines3", "torch", "gymnasium"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = None
    return versions


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def cmd_train(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resume", False)) and "v10" in str(getattr(args, "models_dir", "")).lower():
        raise ValueError("v11 cannot resume a v10 model: observation semantics changed")
    if bool(getattr(args, "resume", False)):
        raise NotImplementedError(
            "scientific --resume is disabled in v11: legacy training does not save/load "
            "the SAC replay buffer and complete environment/curriculum state"
        )
    if str(getattr(args, "success_mode", "progress")) != "progress":
        raise ValueError("v11 has one primary online success definition; --success-mode must be progress")
    if not math.isclose(float(getattr(args, "action_dt", 2.0)), 2.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the frozen v11 training protocol requires --action-dt 2.0 s")
    models_dir = Path(str(args.models_dir)).expanduser().resolve()
    manifest_path = models_dir / "v11_run_manifest.json"
    final_model_path = models_dir / "final_model.zip"
    resume = bool(getattr(args, "resume", False))
    variant = str(getattr(args, "v11_variant", "full_online"))
    seed = int(getattr(args, "seed", 42))
    existing_training_artifacts = [
        path for path in (final_model_path, models_dir / "last_model.zip", models_dir / "vecnormalize.pkl")
        if path.exists()
    ]
    if models_dir.is_dir() and any(models_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"refusing to write into non-empty v11 model directory: {models_dir}"
        )
    if existing_training_artifacts and not resume:
        raise FileExistsError(
            "refusing to overwrite existing run artifacts: "
            + ", ".join(str(path) for path in existing_training_artifacts)
            + "; choose a new directory or use --resume"
        )
    if resume and not manifest_path.exists():
        raise FileNotFoundError("v11 resume requires an existing v11_run_manifest.json")
    if resume:
        with manifest_path.open("r", encoding="utf-8") as handle:
            old_manifest = json.load(handle)
        if str(old_manifest.get("variant")) != variant or int(old_manifest.get("seed", -1)) != seed:
            raise ValueError("resume manifest does not match requested v11 variant/seed")

    source_dir = Path(__file__).resolve().parent
    source_names = (
        "uuv_v11_online.py",
        "uuv_v11_rng.py",
        "uuv_v11_metrics.py",
        "baseline_controllers_v11.py",
        "EXPERIMENT_PROTOCOL_V11.md",
        "V10_ARCHIVE_MANIFEST.json",
        "uuv_v10_info_tracking.py",
        "uuv_v8_temporal_infofix.py",
    )
    source_snapshot_dir = models_dir / "source_snapshot"
    source_snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in source_names:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, source_snapshot_dir / name)
    cfg_preview = _training_config_preview(args)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "version": VERSION,
        "variant": variant,
        "seed": seed,
        "resume": resume,
        "command": [sys.executable] + list(sys.argv),
        "arguments": vars(args),
        "environment_config": asdict(cfg_preview),
        "packages": _package_versions(),
        "git_commit": _git_commit(),
        "source_sha256": {
            name: _sha256_file(source_dir / name)
            for name in source_names
        },
        "source_snapshot_dir": str(source_snapshot_dir),
    }
    _write_json_atomic(manifest_path, manifest)

    previous_variant = os.environ.get("UUV_V11_VARIANT")
    os.environ["UUV_V11_VARIANT"] = variant
    try:
        with _patched_v10_globals(patch_env_class=False):
            _v10.cmd_train(args)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["artifacts_sha256"] = {
            name: _sha256_file(models_dir / name)
            for name in ("final_model.zip", "last_model.zip", "vecnormalize.pkl")
        }
        _write_json_atomic(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(manifest_path, manifest)
        raise
    finally:
        if previous_variant is None:
            os.environ.pop("UUV_V11_VARIANT", None)
        else:
            os.environ["UUV_V11_VARIANT"] = previous_variant


def cmd_eval(args: argparse.Namespace) -> None:
    del args
    raise RuntimeError(
        "the inherited v8/v10 evaluator is disabled for v11 because it uses legacy "
        "controllers, success semantics, and RNG; run `python3 uuv_v11_evaluate.py --help`"
    )


def cmd_sim(args: argparse.Namespace) -> None:
    with _patched_v10_globals(patch_env_class=True):
        _v10.cmd_sim(args)


def cmd_map(args: argparse.Namespace) -> None:
    with _patched_v10_globals(patch_env_class=True):
        _v10.cmd_map(args)


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
        raise ValueError(f"unknown command: {args.cmd!r}")


if __name__ == "__main__":
    main()
