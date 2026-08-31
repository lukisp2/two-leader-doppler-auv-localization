#!/usr/bin/env python3
"""v16: temporal greedy-floor escalation over certified PID+EXC.

V16 keeps the complete v15 online predictor, plant, PID+EXC inner loop, and
fail-closed action certificate.  The deterministic v15 greedy-grid decision is
the minimum requested information gain. A scalar RL action in ``[0, 1]``
chooses only how far to escalate toward the largest full-safe grid gain.

The actor additionally observes online-only episode timing, terminal-window
urgency, the zero-gain post-window FIM eigenvalue, and imminent FIM expiry.
No simulator truth or future exogenous-noise sample is policy- or reward-facing.
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
import uuv_v15_contextual_gain as _v15


VERSION = "v16_temporal_escalation_1.0"
V16_VARIANT = "temporal_greedy_escalation_v16"
CONTROLLER_ARCHITECTURE = (
    "pid_track_exc_plus_window_certified_greedy_floor_plus_rl_escalation"
)
MANIFEST_FILENAME = "v16_run_manifest.json"
MANIFEST_SCHEMA_VERSION = 6

V16_TEMPORAL_CONTEXT_DIM = 10
V16_EXTRA_OBS_DIM = V16_TEMPORAL_CONTEXT_DIM
V16_TAIL_WINDOW_S = 100.0
V16_FORMATION_BARRIER_START = 0.65
V16_FORMATION_WEIGHT = 0.60
V16_AUTHORITY_WEIGHT = 0.08
V16_URGENCY_WEIGHT = 0.45
V16_URGENCY_AUTHORITY_DISCOUNT = 0.75
V16_FALLBACK_WEIGHT = 0.05

TB_V16_NUMERIC_ALLOWLIST = frozenset(
    (
        "v16_actor_escalation",
        "v16_reference_gain",
        "v16_requested_gain",
        "v16_tail_phase",
        "v16_uncertainty_need",
        "v16_urgency",
        "v16_counterfactual_certificate_ok",
        "v16_counterfactual_window_delta_eig",
        "v16_counterfactual_bonus",
        "v16_counterfactual_reject_mask",
    )
)


@dataclass
class UUV3DConfig(_v15.UUV3DConfig):
    """Frozen v16 timing, observation, escalation, and reward contract."""

    v16_variant: str = V16_VARIANT
    v16_controller_architecture: str = CONTROLLER_ARCHITECTURE
    v16_tail_window_s: float = V16_TAIL_WINDOW_S
    v16_formation_barrier_start: float = V16_FORMATION_BARRIER_START
    v16_formation_weight: float = V16_FORMATION_WEIGHT
    v16_authority_weight: float = V16_AUTHORITY_WEIGHT
    v16_urgency_weight: float = V16_URGENCY_WEIGHT
    v16_urgency_authority_discount: float = V16_URGENCY_AUTHORITY_DISCOUNT
    v16_fallback_weight: float = V16_FALLBACK_WEIGHT

    def __post_init__(self) -> None:
        super().__post_init__()
        if str(self.v16_variant).lower().strip() != V16_VARIANT:
            raise ValueError(f"unknown v16 variant {self.v16_variant!r}")
        self.v16_variant = V16_VARIANT
        if str(self.v16_controller_architecture) != CONTROLLER_ARCHITECTURE:
            raise ValueError("v16 controller architecture is frozen")
        frozen = (
            (self.v16_tail_window_s, V16_TAIL_WINDOW_S),
            (self.v16_formation_barrier_start, V16_FORMATION_BARRIER_START),
            (self.v16_formation_weight, V16_FORMATION_WEIGHT),
            (self.v16_authority_weight, V16_AUTHORITY_WEIGHT),
            (self.v16_urgency_weight, V16_URGENCY_WEIGHT),
            (
                self.v16_urgency_authority_discount,
                V16_URGENCY_AUTHORITY_DISCOUNT,
            ),
            (self.v16_fallback_weight, V16_FALLBACK_WEIGHT),
        )
        if any(
            not math.isclose(float(got), want, rel_tol=0.0, abs_tol=0.0)
            for got, want in frozen
        ):
            raise ValueError("v16 temporal escalation and reward are frozen")

    def action_contract(self) -> Dict[str, Any]:
        contract = dict(super().action_contract())
        contract.update(
            {
                "architecture": CONTROLLER_ARCHITECTURE,
                "variant": V16_VARIANT,
                "policy_action_semantics": (
                    "scalar escalation e in [0,1]; requested_gain = "
                    "greedy_reference + e*(maximum_safe_grid_gain-greedy_reference)"
                ),
                "greedy_reference": (
                    "lowest-gain argmax of strictly positive frozen v15 utility "
                    "among exact full-safe grid candidates; otherwise zero"
                ),
                "observation_base_dim": 98,
                "observation_v15_dim": _v15.UUVTwoLeader3DPFEnv.BASE_OBS_DIM,
                "observation_v16_temporal_dim": V16_TEMPORAL_CONTEXT_DIM,
                "tail_window_s": V16_TAIL_WINDOW_S,
                "reward_semantics": (
                    "inherited online reward plus once-only accepted v16 tradeoff: "
                    "v15 information benefit + late uncertainty urgency benefit "
                    "- predicted formation barrier - urgency-discounted authority cost"
                ),
                "formation_barrier_start_ratio": V16_FORMATION_BARRIER_START,
                "rejected_escalation_fallback": (
                    "apply the exact re-certified greedy reference; PID+EXC if no "
                    "positive reference or if reference re-certification fails"
                ),
                "fallback_penalty_weight": V16_FALLBACK_WEIGHT,
                "neutral_branch": (
                    "exact inherited PID+EXC action and reward when requested gain "
                    "is zero or the full certificate rejects it"
                ),
                "truth_or_future_noise_inputs": False,
            }
        )
        return contract


@dataclass(frozen=True)
class V16TemporalContext:
    progress: float
    remaining_fraction: float
    tail_active: float
    tail_phase: float
    uncertainty_need: float
    urgency: float
    inner_post_window_eigmin: float
    scaled_inner_post_window_eigmin: float
    fim_expiry_feature: float
    reference_gain: float
    maximum_safe_gain: float


@dataclass(frozen=True)
class V16RewardTradeoff:
    info_benefit: float
    urgency_benefit: float
    formation_cost: float
    authority_cost: float
    fallback_cost: float
    total: float


@dataclass(frozen=True)
class V16ActionComposition:
    actor_escalation: float
    reference_gain: float
    maximum_safe_gain: float
    requested_gain: float
    applied_gain: float
    escalation_accepted: bool
    reference_fallback: bool
    temporal_context: V16TemporalContext
    requested_evidence: _v15.V15ActionComposition
    inherited: _v15.V15ActionComposition
    tradeoff: V16RewardTradeoff


def _episode_horizon_s(env: "UUVTwoLeader3DPFEnv") -> float:
    return max(
        float(env.cfg.action_dt),
        float(env.cfg.max_steps) * float(env.cfg.action_dt),
    )


def _scaled_eigmin(value: float) -> float:
    eig = max(float(value), 0.0) if np.isfinite(value) else 0.0
    return float(
        np.clip(
            math.log10((eig + 1e-12) / float(_v15.V15_FIM_EIG_REFERENCE)),
            -6.0,
            2.0,
        )
        / 6.0
    )


def select_v16_reference_gain(grid: _v15.V15OpportunityGrid) -> float:
    """Return the deterministic v15 greedy reference with a zero fallback."""

    gains = np.asarray(grid.gains, dtype=float)
    safe = np.asarray(grid.safe, dtype=bool)
    utility = np.asarray(grid.utility, dtype=float)
    if gains.shape != (len(_v15.V15_GRID_GAINS),):
        raise ValueError("v16 reference received an invalid gain grid")
    if safe.shape != gains.shape or utility.shape != gains.shape:
        raise ValueError("v16 reference grid arrays have inconsistent shapes")
    eligible = safe & np.isfinite(utility) & (utility > 0.0)
    if not np.any(eligible):
        return 0.0
    candidates = np.flatnonzero(eligible)
    # np.argmax is stable and therefore implements the frozen lowest-gain tie.
    local = int(np.argmax(utility[candidates]))
    return float(gains[int(candidates[local])])


def build_v16_temporal_context(
    env: "UUVTwoLeader3DPFEnv",
    grid: _v15.V15OpportunityGrid,
    online_info: Optional[Mapping[str, Any]] = None,
) -> V16TemporalContext:
    """Construct the online-only timing and post-window opportunity context."""

    if online_info is None:
        online_info = _v14.UUVTwoLeader3DPFEnv._get_info(env)
    horizon = _episode_horizon_s(env)
    progress = float(np.clip(float(env.t) / horizon, 0.0, 1.0))
    remaining_s = max(horizon - float(env.t), 0.0)
    remaining_fraction = float(np.clip(remaining_s / horizon, 0.0, 1.0))
    tail_window = float(env.cfg.v16_tail_window_s)
    tail_active = float(remaining_s <= tail_window + 1e-12)
    tail_phase = float(
        np.clip((tail_window - remaining_s) / max(tail_window, 1e-12), 0.0, 1.0)
    )
    snapshot = _v13.online_pid_exc_snapshot(online_info)
    tol_std = max(float(snapshot["tol_std"]), 1e-12)
    uncertainty_need = float(
        np.clip(
            (float(snapshot["pf_std_max_raw"]) - tol_std) / tol_std,
            0.0,
            1.0,
        )
    )
    urgency = float(tail_phase * uncertainty_need)
    current_eig = float(_v15._scaled_fim_eigenvalues(env)[0][0])
    inner_post = float(grid.inner_post_window_eigmin)
    expiry = (
        math.tanh((current_eig - inner_post) / float(_v15.V15_INFO_SCALE))
        if np.isfinite(inner_post)
        else 0.0
    )
    safe_gains = np.asarray(grid.gains, dtype=float)[np.asarray(grid.safe, dtype=bool)]
    maximum_safe = float(np.max(safe_gains)) if safe_gains.size else 0.0
    return V16TemporalContext(
        progress=progress,
        remaining_fraction=remaining_fraction,
        tail_active=tail_active,
        tail_phase=tail_phase,
        uncertainty_need=uncertainty_need,
        urgency=urgency,
        inner_post_window_eigmin=inner_post,
        scaled_inner_post_window_eigmin=_scaled_eigmin(inner_post),
        fim_expiry_feature=float(np.clip(expiry, -1.0, 1.0)),
        reference_gain=select_v16_reference_gain(grid),
        maximum_safe_gain=maximum_safe,
    )


def score_v16_tradeoff(
    *,
    delta_eig: float,
    candidate_pred_err: float,
    tol_pos: float,
    realized_authority: float,
    tail_phase: float,
    uncertainty_need: float,
    fallback_fraction: float = 0.0,
    cfg: UUV3DConfig,
) -> V16RewardTradeoff:
    """Score an accepted action using only online counterfactual quantities."""

    tol = max(float(tol_pos), 1e-12)
    authority = float(np.clip(realized_authority, 0.0, 1.0))
    urgency = float(
        np.clip(float(tail_phase), 0.0, 1.0)
        * np.clip(float(uncertainty_need), 0.0, 1.0)
    )
    info = float(cfg.v15_info_weight) * math.tanh(
        max(float(delta_eig), 0.0) / float(cfg.v15_info_scale)
    )
    urgency_benefit = float(cfg.v16_urgency_weight) * urgency * authority
    predicted_ratio = float(
        np.clip(float(candidate_pred_err) / tol, 0.0, 1.0)
    )
    barrier_start = float(cfg.v16_formation_barrier_start)
    barrier = float(
        np.clip(
            (predicted_ratio - barrier_start) / max(1.0 - barrier_start, 1e-12),
            0.0,
            1.0,
        )
    )
    formation_cost = float(cfg.v16_formation_weight) * barrier * barrier
    authority_discount = 1.0 - float(cfg.v16_urgency_authority_discount) * urgency
    authority_cost = (
        float(cfg.v16_authority_weight)
        * float(np.clip(authority_discount, 0.0, 1.0))
        * authority
        * authority
    )
    fallback_cost = float(cfg.v16_fallback_weight) * float(
        np.clip(fallback_fraction, 0.0, 1.0)
    )
    return V16RewardTradeoff(
        info_benefit=info,
        urgency_benefit=urgency_benefit,
        formation_cost=formation_cost,
        authority_cost=authority_cost,
        fallback_cost=fallback_cost,
        total=float(
            info
            + urgency_benefit
            - formation_cost
            - authority_cost
            - fallback_cost
        ),
    )


def compose_v16_action(
    env: "UUVTwoLeader3DPFEnv",
    online_info: Mapping[str, Any],
    actor_escalation: Sequence[float],
    step_count: int,
    *,
    grid: Optional[_v15.V15OpportunityGrid] = None,
    backend: str = "auto",
) -> V16ActionComposition:
    """Compose the greedy-floor plus RL escalation and apply the v15 certificate."""

    raw = np.asarray(actor_escalation, dtype=np.float32).reshape(-1)
    if raw.shape != (1,) or not np.all(np.isfinite(raw)):
        raise ValueError("v16 actor escalation must contain one finite value")
    escalation = float(raw[0])
    if escalation < 0.0 or escalation > 1.0:
        raise ValueError("v16 actor escalation must be in [0, 1]")
    if grid is None:
        grid = _v15.evaluate_v15_opportunity_grid(
            env,
            online_info=online_info,
            step_count=int(step_count),
            backend=backend,
        )
    temporal = build_v16_temporal_context(env, grid, online_info)
    reference = float(temporal.reference_gain)
    maximum_safe = max(float(temporal.maximum_safe_gain), reference)
    requested = float(reference + escalation * (maximum_safe - reference))
    requested_evidence = _v15.compose_v15_action(
        env,
        online_info,
        [requested],
        int(step_count),
        backend=backend,
    )
    inherited = requested_evidence
    reference_fallback = False
    if (
        not requested_evidence.counterfactual_certificate_ok
        and reference > 0.0
        and requested > reference + 1e-12
    ):
        reference_evidence = _v15.compose_v15_action(
            env,
            online_info,
            [reference],
            int(step_count),
            backend=backend,
        )
        if reference_evidence.counterfactual_certificate_ok:
            inherited = reference_evidence
            reference_fallback = True
    escalation_accepted = bool(
        requested_evidence.counterfactual_certificate_ok
        and not reference_fallback
    )
    if inherited.counterfactual_certificate_ok:
        tradeoff = score_v16_tradeoff(
            delta_eig=float(inherited.counterfactual_window_delta_eig),
            candidate_pred_err=float(inherited.counterfactual_candidate_pred_err),
            tol_pos=float(_v13.online_pid_exc_snapshot(online_info)["tol_pos_est"]),
            realized_authority=float(inherited.realized_authority),
            tail_phase=float(temporal.tail_phase),
            uncertainty_need=float(temporal.uncertainty_need),
            fallback_fraction=(escalation if reference_fallback else 0.0),
            cfg=env.cfg,
        )
    else:
        tradeoff = V16RewardTradeoff(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return V16ActionComposition(
        actor_escalation=escalation,
        reference_gain=reference,
        maximum_safe_gain=maximum_safe,
        requested_gain=requested,
        applied_gain=(
            float(inherited.policy_gain)
            if inherited.counterfactual_certificate_ok
            else 0.0
        ),
        escalation_accepted=escalation_accepted,
        reference_fallback=reference_fallback,
        temporal_context=temporal,
        requested_evidence=requested_evidence,
        inherited=inherited,
        tradeoff=tradeoff,
    )


class UUVTwoLeader3DPFEnv(_v15.UUVTwoLeader3DPFEnv):
    """V15 plant with temporal greedy-floor RL escalation."""

    BASE_OBS_DIM = _v15.UUVTwoLeader3DPFEnv.BASE_OBS_DIM + V16_EXTRA_OBS_DIM

    def _init_info_ray_diagnostics(self) -> None:
        super()._init_info_ray_diagnostics()
        self._init_v16_diagnostics()

    def _reset_info_ray_diagnostics(self) -> None:
        super()._reset_info_ray_diagnostics()
        self._init_v16_diagnostics()

    def _init_v16_diagnostics(self) -> None:
        self._v16_actor_escalation = 0.0
        self._v16_reference_gain = 0.0
        self._v16_maximum_safe_gain = 0.0
        self._v16_requested_gain = 0.0
        self._v16_applied_gain = 0.0
        self._v16_escalation_accepted = False
        self._v16_reference_fallback = False
        self._v16_requested_reject_mask = 0
        self._v16_temporal_context: Optional[V16TemporalContext] = None
        self._v16_action_temporal_context: Optional[V16TemporalContext] = None
        self._v16_tradeoff = V16RewardTradeoff(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._v16_counterfactual_bonus_pending = False
        self._v16_counterfactual_bonus = 0.0
        self._v16_counterfactual_evaluated_step = -1
        self._v16_counterfactual_rewarded_step = -1

    def __init__(self, cfg: Optional[UUV3DConfig] = None, render_mode: str = "none"):
        super().__init__(cfg=cfg or UUV3DConfig(), render_mode=render_mode)
        self.cfg: UUV3DConfig

    def _policy_uses_v16_escalation(self) -> bool:
        return str(self.cfg.v11_controller_id) in _v13.POLICY_CONTROLLER_IDS

    def _get_obs_base(self) -> np.ndarray:
        inherited = super()._get_obs_base().astype(np.float32)
        if inherited.shape != (_v15.UUVTwoLeader3DPFEnv.BASE_OBS_DIM,):
            raise RuntimeError("unexpected inherited v15 observation width")
        grid = self._v15_grid
        if grid is None or int(self._v15_grid_step) != int(self.step_count):
            raise RuntimeError("v16 observation lacks a current v15 opportunity grid")
        online_info = _v14.UUVTwoLeader3DPFEnv._get_info(self)
        temporal = build_v16_temporal_context(self, grid, online_info)
        self._v16_temporal_context = temporal
        extra = np.asarray(
            [
                temporal.progress,
                temporal.remaining_fraction,
                temporal.tail_active,
                temporal.tail_phase,
                temporal.uncertainty_need,
                temporal.urgency,
                temporal.scaled_inner_post_window_eigmin,
                temporal.fim_expiry_feature,
                temporal.reference_gain,
                temporal.maximum_safe_gain,
            ],
            dtype=np.float32,
        )
        if extra.shape != (V16_EXTRA_OBS_DIM,) or not np.all(np.isfinite(extra)):
            raise RuntimeError("invalid v16 temporal observation")
        return np.concatenate((inherited, extra), axis=0).astype(np.float32)

    def _record_v16_composition(self, composition: V16ActionComposition) -> None:
        super()._record_v15_composition(composition.inherited)
        # V16 replaces, rather than stacks with, the v15 tradeoff reward.
        self._v15_counterfactual_bonus_pending = False
        self._v16_actor_escalation = float(composition.actor_escalation)
        self._v16_reference_gain = float(composition.reference_gain)
        self._v16_maximum_safe_gain = float(composition.maximum_safe_gain)
        self._v16_requested_gain = float(composition.requested_gain)
        self._v16_applied_gain = float(composition.applied_gain)
        self._v16_escalation_accepted = bool(composition.escalation_accepted)
        self._v16_reference_fallback = bool(composition.reference_fallback)
        self._v16_requested_reject_mask = int(
            composition.requested_evidence.counterfactual_reject_mask
        )
        self._v16_action_temporal_context = composition.temporal_context
        self._v16_tradeoff = composition.tradeoff
        self._v16_counterfactual_bonus = 0.0
        self._v16_counterfactual_bonus_pending = bool(
            composition.inherited.counterfactual_certificate_ok
        )
        self._v16_counterfactual_evaluated_step = int(self.step_count)
        self._v16_counterfactual_rewarded_step = -1

    def _compute_reward(
        self,
        pf_stats_last: Optional[_v8.PFStats],
        planner_action_for_reward: Optional[np.ndarray] = None,
        planner_gate_for_reward: Optional[float] = None,
        planner_margin_for_reward: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        base_reward, terms = _v11.UUVTwoLeader3DPFEnv._compute_reward(
            self,
            pf_stats_last,
            planner_action_for_reward,
            planner_gate_for_reward,
            planner_margin_for_reward,
        )
        bonus = 0.0
        if self._v16_counterfactual_bonus_pending:
            if int(self._v16_counterfactual_evaluated_step) != int(self.step_count) - 1:
                raise RuntimeError("stale v16 pre-step reward cache")
            if not bool(self._v15_counterfactual_certificate_ok):
                raise RuntimeError("uncertified v16 tradeoff reached reward cache")
            bonus = float(self._v16_tradeoff.total)
            self._v16_counterfactual_bonus_pending = False
            self._v16_counterfactual_rewarded_step = int(self.step_count)
            self._v16_counterfactual_bonus = bonus
        temporal = self._v16_action_temporal_context
        terms.update(
            {
                "v16_version": 16.0,
                "v16_reward_uses_truth": 0.0,
                "v16_actor_escalation": float(self._v16_actor_escalation),
                "v16_reference_gain": float(self._v16_reference_gain),
                "v16_maximum_safe_gain": float(self._v16_maximum_safe_gain),
                "v16_requested_gain": float(self._v16_requested_gain),
                "v16_applied_gain": float(self._v16_applied_gain),
                "v16_escalation_accepted": float(self._v16_escalation_accepted),
                "v16_reference_fallback": float(self._v16_reference_fallback),
                "v16_tail_phase": float(temporal.tail_phase if temporal else 0.0),
                "v16_uncertainty_need": float(
                    temporal.uncertainty_need if temporal else 0.0
                ),
                "v16_urgency": float(temporal.urgency if temporal else 0.0),
                "v16_counterfactual_certificate_ok": float(
                    self._v15_counterfactual_certificate_ok
                ),
                "v16_counterfactual_window_delta_eig": float(
                    self._v15_counterfactual_window_delta_eig
                ),
                "v16_counterfactual_info_benefit": float(
                    self._v16_tradeoff.info_benefit
                ),
                "v16_counterfactual_urgency_benefit": float(
                    self._v16_tradeoff.urgency_benefit
                ),
                "v16_counterfactual_formation_cost": float(
                    self._v16_tradeoff.formation_cost
                ),
                "v16_counterfactual_authority_cost": float(
                    self._v16_tradeoff.authority_cost
                ),
                "v16_counterfactual_fallback_cost": float(
                    self._v16_tradeoff.fallback_cost
                ),
                "v16_counterfactual_bonus": float(bonus),
                "v16_counterfactual_reject_mask": float(
                    self._v15_counterfactual_reject_mask
                ),
            }
        )
        if bonus == 0.0:
            return float(base_reward), terms
        return (
            float(
                np.clip(
                    float(base_reward) + bonus,
                    -float(self.cfg.rew_clip),
                    float(self.cfg.rew_clip),
                )
            ),
            terms,
        )

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)
        temporal = self._v16_action_temporal_context or self._v16_temporal_context
        info.update(
            {
                "v16_version": 16.0,
                "v16_actor_escalation": float(self._v16_actor_escalation),
                "v16_reference_gain": float(self._v16_reference_gain),
                "v16_maximum_safe_gain": float(self._v16_maximum_safe_gain),
                "v16_requested_gain": float(self._v16_requested_gain),
                "v16_applied_gain": float(self._v16_applied_gain),
                "v16_escalation_accepted": float(self._v16_escalation_accepted),
                "v16_reference_fallback": float(self._v16_reference_fallback),
                "v16_requested_reject_mask": int(self._v16_requested_reject_mask),
                "v16_progress": float(temporal.progress if temporal else 0.0),
                "v16_remaining_fraction": float(
                    temporal.remaining_fraction if temporal else 1.0
                ),
                "v16_tail_active": float(temporal.tail_active if temporal else 0.0),
                "v16_tail_phase": float(temporal.tail_phase if temporal else 0.0),
                "v16_uncertainty_need": float(
                    temporal.uncertainty_need if temporal else 0.0
                ),
                "v16_urgency": float(temporal.urgency if temporal else 0.0),
                "v16_inner_post_window_eigmin": float(
                    temporal.inner_post_window_eigmin if temporal else 0.0
                ),
                "v16_fim_expiry_feature": float(
                    temporal.fim_expiry_feature if temporal else 0.0
                ),
                "v16_maximum_safe_gain": float(
                    temporal.maximum_safe_gain if temporal else 0.0
                ),
                "v16_counterfactual_certificate_ok": float(
                    self._v15_counterfactual_certificate_ok
                ),
                "v16_counterfactual_window_delta_eig": float(
                    self._v15_counterfactual_window_delta_eig
                ),
                "v16_counterfactual_candidate_pred_err": float(
                    self._v15_counterfactual_candidate_pred_err
                ),
                "v16_counterfactual_realized_authority": float(
                    self._v15_counterfactual_realized_authority
                ),
                "v16_counterfactual_info_benefit": float(
                    self._v16_tradeoff.info_benefit
                ),
                "v16_counterfactual_urgency_benefit": float(
                    self._v16_tradeoff.urgency_benefit
                ),
                "v16_counterfactual_formation_cost": float(
                    self._v16_tradeoff.formation_cost
                ),
                "v16_counterfactual_authority_cost": float(
                    self._v16_tradeoff.authority_cost
                ),
                "v16_counterfactual_fallback_cost": float(
                    self._v16_tradeoff.fallback_cost
                ),
                "v16_counterfactual_bonus": float(
                    self._v16_counterfactual_bonus
                ),
                "v16_counterfactual_reject_mask": int(
                    self._v15_counterfactual_reject_mask
                ),
                "v16_counterfactual_reject_reasons": ",".join(
                    self._v15_counterfactual_reject_reasons
                ),
                "v16_counterfactual_evaluated_step": int(
                    self._v16_counterfactual_evaluated_step
                ),
                "v16_counterfactual_rewarded_step": int(
                    self._v16_counterfactual_rewarded_step
                ),
            }
        )
        return info

    def step(self, action: Sequence[float]):
        if self._policy_uses_v16_escalation():
            online_info = self._get_info()
            grid = self._v15_grid
            if grid is None or int(self._v15_grid_step) != int(self.step_count):
                grid = _v15.evaluate_v15_opportunity_grid(
                    self,
                    online_info=online_info,
                    step_count=int(self.step_count),
                    backend=str(self.cfg.v15_predictor_backend),
                )
            composition = compose_v16_action(
                self,
                online_info,
                action,
                int(self.step_count),
                grid=grid,
                backend=str(self.cfg.v15_predictor_backend),
            )
            self._v13_info_ray_active = True
            self._record_v16_composition(composition)
            return _v11.UUVTwoLeader3DPFEnv.step(
                self, composition.inherited.action_applied
            )
        # Exact fixed-gain and direct baselines retain their inherited meaning.
        self._v16_counterfactual_bonus_pending = False
        self._v16_actor_escalation = 0.0
        self._v16_reference_gain = 0.0
        self._v16_maximum_safe_gain = 0.0
        self._v16_requested_gain = 0.0
        self._v16_applied_gain = 0.0
        self._v16_escalation_accepted = False
        self._v16_reference_fallback = False
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
            if not name.startswith("v16_"):
                filtered[key] = value
            elif name in TB_V16_NUMERIC_ALLOWLIST and isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                filtered[key] = value
        copied.append(filtered)
    return _v15._tb_infos_copy(copied)


class V16TBInfoCallback(_v8.TBInfoCallback):
    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        callback_locals = dict(locals_)
        callback_locals["infos"] = _tb_infos_copy(locals_.get("infos"))
        return super().__call__(callback_locals, globals_)


def build_parser() -> argparse.ArgumentParser:
    parser = _v15.build_parser()
    replacements = {
        "models_3d_v15_contextual_gain": "models_3d_v16_temporal_escalation",
        "logs_3d_v15_contextual_gain": "logs_3d_v16_temporal_escalation",
        "tb_3d_v15_contextual_gain": "tb_3d_v16_temporal_escalation",
        "eval_3d_logs_v15_contextual_gain": "eval_3d_logs_v16_temporal_escalation",
        "info_maps_v15_contextual_gain": "info_maps_v16_temporal_escalation",
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
        train_parser, "v16_variant"
    ):
        train_parser.add_argument(
            "--v16-variant",
            choices=(V16_VARIANT,),
            default=V16_VARIANT,
            help="Frozen v16 temporal greedy-floor escalation controller.",
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
    v8.TBInfoCallback = V16TBInfoCallback
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
    "uuv_v16_temporal_escalation.py",
    "uuv_v16_evaluate.py",
    "run_v16_training.py",
    "EXPERIMENT_PROTOCOL_V16.md",
    "tests/test_uuv_v16_temporal_escalation.py",
    "tests/test_uuv_v16_evaluate.py",
    "tests/test_run_v16_training.py",
) + tuple(_v15.SOURCE_NAMES)


def _training_config_preview(args: argparse.Namespace) -> UUV3DConfig:
    inherited = asdict(_v15._training_config_preview(args))
    inherited["v16_variant"] = str(
        getattr(args, "v16_variant", V16_VARIANT)
    )
    return UUV3DConfig(**inherited)


def cmd_train(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resume", False)):
        raise NotImplementedError("scientific resume is disabled")
    requirements = (
        ("v11_variant", "full_online"),
        ("v12_variant", _v12.V12_VARIANT),
        ("v13_variant", _v13.V13_VARIANT),
        ("v14_variant", _v14.V14_VARIANT),
        ("v15_variant", _v15.V15_VARIANT),
        ("v16_variant", V16_VARIANT),
    )
    for name, expected in requirements:
        if str(getattr(args, name, expected)) != expected:
            raise ValueError(f"v16 requires --{name.replace('_', '-')} {expected}")
    if str(getattr(args, "success_mode", "progress")) != "progress":
        raise ValueError("v16 requires the frozen online progress success definition")
    if not math.isclose(float(getattr(args, "action_dt", 2.0)), 2.0, abs_tol=1e-12):
        raise ValueError("v16 training requires --action-dt 2.0 s")
    if not math.isclose(
        float(getattr(args, "fim_window", _v15.V15_FIM_WINDOW_S)),
        _v15.V15_FIM_WINDOW_S,
        abs_tol=0.0,
    ):
        raise ValueError("v16 training requires --fim-window 30.0 s")

    models_dir = Path(str(args.models_dir)).expanduser().resolve()
    manifest_path = models_dir / MANIFEST_FILENAME
    if models_dir.is_dir() and any(models_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty v16 model directory: {models_dir}"
        )
    source_dir = Path(__file__).resolve().parent
    snapshot_dir = models_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing required v16 source artifact: {source}")
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    cfg_preview = _training_config_preview(args)
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "version": VERSION,
        "variant": V16_VARIANT,
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
    raise RuntimeError("run `python3 uuv_v16_evaluate.py --help` for v16 evaluation")


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
    "UUV3DConfig",
    "UUVTwoLeader3DPFEnv",
    "V16ActionComposition",
    "V16RewardTradeoff",
    "V16TemporalContext",
    "V16_VARIANT",
    "VERSION",
    "build_parser",
    "build_v16_temporal_context",
    "compose_v16_action",
    "make_env",
    "score_v16_tradeoff",
    "select_v16_reference_gain",
]
