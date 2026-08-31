#!/usr/bin/env python3
"""V18: PF-resampling-aware guard around the complete V17 gain.

V18 changes exactly one scientific mechanism from V17.  When the minimum
online, pre-resampling particle-filter ESS observed in the preceding action
is at or below 0.50, or a resampling occurred during one of the previous 15
action intervals, both the deterministic greedy reference and the learned
ceiling are capped at gain 0.50.  Outside that guard V17 is bit-for-bit
inherited.  The particle filter, reward, temporal context, predictor,
certificate, plant and one-grid-step learned increment are otherwise
unchanged.
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

import uuv_v17_bounded_escalation as _v17


VERSION = "v18_resampling_guard_1.0"
V18_VARIANT = "resampling_risk_guard_v18"
CONTROLLER_ARCHITECTURE = (
    "pid_track_exc_plus_resampling_guarded_greedy_floor_plus_"
    "one_grid_step_rl_escalation"
)
MANIFEST_FILENAME = "v18_run_manifest.json"
MANIFEST_SCHEMA_VERSION = 8
V18_MAX_ESCALATION_DELTA = _v17.V17_MAX_ESCALATION_DELTA
V18_ESS_GUARD_THRESHOLD = 0.50
V18_RESAMPLE_COOLDOWN_ACTIONS = 15
V18_GUARDED_TOTAL_GAIN_CAP = 0.50
# Backward-compatible public helper; V18 additionally applies
# ``guarded_gain_bounds`` around these inherited V17 bounds.
escalation_ceiling_gain = _v17.escalation_ceiling_gain

TB_V18_NUMERIC_ALLOWLIST = frozenset(
    (
        "v18_actor_escalation",
        "v18_reference_gain",
        "v18_full_safe_gain",
        "v18_escalation_ceiling_gain",
        "v18_requested_gain",
        "v18_applied_gain",
        "v18_escalation_accepted",
        "v18_reference_fallback",
        "v18_counterfactual_certificate_ok",
        "v18_guard_active",
        "v18_guard_trigger_low_ess",
        "v18_guard_trigger_recent_resample",
        "v18_decision_ess_fraction",
        "v18_previous_action_resampled_any",
        "v18_last_resample_age_steps",
        "v18_unguarded_reference_gain",
        "v18_unguarded_ceiling_gain",
        "v18_reference_reduction",
        "v18_ceiling_reduction",
        "v18_outcome_had_measurement",
        "v18_outcome_ess_min_pre_resample_fraction",
        "v18_pf_resampled_any_action",
        "v18_outcome_injected_count",
    )
)


@dataclass
class UUV3DConfig(_v17.UUV3DConfig):
    """Frozen V18 resampling-risk guard contract."""

    v18_variant: str = V18_VARIANT
    v18_controller_architecture: str = CONTROLLER_ARCHITECTURE
    v18_max_escalation_delta: float = V18_MAX_ESCALATION_DELTA
    v18_ess_guard_threshold: float = V18_ESS_GUARD_THRESHOLD
    v18_resample_cooldown_actions: int = V18_RESAMPLE_COOLDOWN_ACTIONS
    v18_guarded_total_gain_cap: float = V18_GUARDED_TOTAL_GAIN_CAP

    def __post_init__(self) -> None:
        super().__post_init__()
        if str(self.v18_variant).lower().strip() != V18_VARIANT:
            raise ValueError(f"unknown v18 variant {self.v18_variant!r}")
        self.v18_variant = V18_VARIANT
        if str(self.v18_controller_architecture) != CONTROLLER_ARCHITECTURE:
            raise ValueError("v18 controller architecture is frozen")
        if not math.isclose(
            float(self.v18_max_escalation_delta),
            V18_MAX_ESCALATION_DELTA,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("v18 escalation delta is frozen at one grid step")
        if not math.isclose(
            float(self.v18_ess_guard_threshold),
            V18_ESS_GUARD_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("v18 ESS guard threshold is frozen at 0.50")
        if int(self.v18_resample_cooldown_actions) != V18_RESAMPLE_COOLDOWN_ACTIONS:
            raise ValueError("v18 resampling cooldown is frozen at 15 actions")
        if not math.isclose(
            float(self.v18_guarded_total_gain_cap),
            V18_GUARDED_TOTAL_GAIN_CAP,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("v18 guarded total gain cap is frozen at 0.50")

    def action_contract(self) -> Dict[str, Any]:
        contract = dict(super().action_contract())
        contract.update(
            {
                "architecture": CONTROLLER_ARCHITECTURE,
                "variant": V18_VARIANT,
                "policy_action_semantics": (
                    "inherit V17 scalar escalation; when the minimum online "
                    "pre-resampling ESS/N from the preceding action is <= "
                    "0.50, or resampling occurred in the previous 15 action "
                    "intervals, cap both greedy reference and actionable "
                    "ceiling at 0.50"
                ),
                "maximum_escalation_above_reference": V18_MAX_ESCALATION_DELTA,
                "maximum_escalation_basis": (
                    "one interval of the frozen V15 gain grid"
                ),
                "resampling_guard_ess_fraction_threshold": (
                    V18_ESS_GUARD_THRESHOLD
                ),
                "resampling_guard_cooldown_actions": (
                    V18_RESAMPLE_COOLDOWN_ACTIONS
                ),
                "resampling_guard_total_gain_cap": V18_GUARDED_TOTAL_GAIN_CAP,
                "resampling_guard_scope": "greedy reference and learned ceiling",
                "resampling_guard_ess_timing": (
                    "minimum pre-resampling ESS across all measurement "
                    "updates in the immediately preceding action; reset uses "
                    "current particle weights; non-finite evidence fails closed"
                ),
                "observation_base_dim": 98,
                "observation_v18_extra_dim": 0,
                "observation_last_feature": "actionable escalation ceiling gain",
                "unchanged_from_v17": [
                    "one-grid-step actor increment outside the guard",
                    "temporal context",
                    "reward",
                    "four-second predictor",
                    "30-second FIM window",
                    "full action certificate",
                    "plant and particle filter",
                ],
                "truth_or_future_noise_inputs": False,
            }
        )
        return contract


@dataclass(frozen=True)
class V18ActionComposition:
    actor_escalation: float
    reference_gain: float
    full_safe_gain: float
    escalation_ceiling_gain: float
    requested_gain: float
    applied_gain: float
    escalation_accepted: bool
    reference_fallback: bool
    guard: "V18GuardDecision"
    inherited_v17: _v17.V17ActionComposition


@dataclass(frozen=True)
class V18GuardDecision:
    state_step: int
    decision_ess_fraction: float
    previous_action_resampled_any: bool
    last_resample_age_steps: int
    trigger_low_ess: bool
    trigger_recent_resample: bool
    active: bool
    unguarded_reference_gain: float
    unguarded_ceiling_gain: float
    guarded_reference_gain: float
    guarded_ceiling_gain: float


def guarded_gain_bounds(
    unguarded_reference_gain: float,
    full_safe_gain: float,
    *,
    guard_active: bool,
    max_delta: float = V18_MAX_ESCALATION_DELTA,
    total_gain_cap: float = V18_GUARDED_TOTAL_GAIN_CAP,
) -> Tuple[float, float, float]:
    """Return guarded reference, ceiling and the unguarded V17 ceiling."""

    reference = float(unguarded_reference_gain)
    full_safe = max(float(full_safe_gain), reference)
    delta = float(max_delta)
    cap = float(total_gain_cap)
    if not all(np.isfinite((reference, full_safe, delta, cap))):
        raise ValueError("v18 gain bounds must be finite")
    if not (0.0 <= reference <= full_safe <= 1.0):
        raise ValueError("v18 inherited gain bounds are outside [0, 1]")
    if delta < 0.0 or not (0.0 <= cap <= 1.0):
        raise ValueError("v18 guard constants are outside the gain contract")
    unguarded_ceiling = _v17.escalation_ceiling_gain(
        reference, full_safe, delta
    )
    if not bool(guard_active):
        return reference, unguarded_ceiling, unguarded_ceiling
    guarded_reference = min(reference, cap)
    guarded_ceiling = min(
        full_safe, guarded_reference + delta, cap
    )
    if not (0.0 <= guarded_reference <= guarded_ceiling <= cap + 1e-12):
        raise RuntimeError("v18 guarded gain bounds are inconsistent")
    return float(guarded_reference), float(guarded_ceiling), unguarded_ceiling


def build_v18_temporal_context(
    env: "UUVTwoLeader3DPFEnv",
    grid: _v17._v15.V15OpportunityGrid,
    online_info: Optional[Mapping[str, Any]] = None,
) -> Tuple[_v17._v16.V16TemporalContext, float, V18GuardDecision]:
    """Return V17 temporal context after the online resampling-risk guard."""

    original, full_safe = _v17.build_v17_temporal_context(
        env, grid, online_info
    )
    unguarded_reference = float(original.reference_gain)
    unguarded_ceiling = float(original.maximum_safe_gain)
    guard_inputs = env._v18_guard_inputs()
    reference, ceiling, recomputed_unguarded_ceiling = guarded_gain_bounds(
        unguarded_reference,
        full_safe,
        guard_active=bool(guard_inputs["active"]),
        max_delta=float(env.cfg.v18_max_escalation_delta),
        total_gain_cap=float(env.cfg.v18_guarded_total_gain_cap),
    )
    if not math.isclose(
        unguarded_ceiling, recomputed_unguarded_ceiling, abs_tol=1e-12
    ):
        raise RuntimeError("v18 inherited V17 ceiling is inconsistent")
    decision = V18GuardDecision(
        state_step=int(env.step_count),
        decision_ess_fraction=float(guard_inputs["ess_fraction"]),
        previous_action_resampled_any=bool(
            guard_inputs["previous_action_resampled_any"]
        ),
        last_resample_age_steps=int(guard_inputs["last_resample_age_steps"]),
        trigger_low_ess=bool(guard_inputs["trigger_low_ess"]),
        trigger_recent_resample=bool(guard_inputs["trigger_recent_resample"]),
        active=bool(guard_inputs["active"]),
        unguarded_reference_gain=unguarded_reference,
        unguarded_ceiling_gain=unguarded_ceiling,
        guarded_reference_gain=reference,
        guarded_ceiling_gain=ceiling,
    )
    return (
        replace(original, reference_gain=reference, maximum_safe_gain=ceiling),
        float(full_safe),
        decision,
    )


def compose_v18_action(
    env: "UUVTwoLeader3DPFEnv",
    online_info: Mapping[str, Any],
    actor_escalation: Sequence[float],
    step_count: int,
    *,
    grid: Optional[_v17._v15.V15OpportunityGrid] = None,
    backend: str = "auto",
) -> V18ActionComposition:
    """Compose, certify and apply the bounded V18 escalation."""

    raw = np.asarray(actor_escalation, dtype=np.float32).reshape(-1)
    if raw.shape != (1,) or not np.all(np.isfinite(raw)):
        raise ValueError("v18 actor escalation must contain one finite value")
    escalation = float(raw[0])
    if escalation < 0.0 or escalation > 1.0:
        raise ValueError("v18 actor escalation must be in [0, 1]")
    if grid is None:
        grid = _v17._v15.evaluate_v15_opportunity_grid(
            env,
            online_info=online_info,
            step_count=int(step_count),
            backend=backend,
        )

    temporal, full_safe, guard = build_v18_temporal_context(
        env, grid, online_info
    )
    if not guard.active:
        inherited_v17 = _v17.compose_v17_action(
            env,
            online_info,
            actor_escalation,
            int(step_count),
            grid=grid,
            backend=backend,
        )
        return V18ActionComposition(
            actor_escalation=float(inherited_v17.actor_escalation),
            reference_gain=float(inherited_v17.reference_gain),
            full_safe_gain=float(inherited_v17.full_safe_gain),
            escalation_ceiling_gain=float(
                inherited_v17.escalation_ceiling_gain
            ),
            requested_gain=float(inherited_v17.requested_gain),
            applied_gain=float(inherited_v17.applied_gain),
            escalation_accepted=bool(inherited_v17.escalation_accepted),
            reference_fallback=bool(inherited_v17.reference_fallback),
            guard=guard,
            inherited_v17=inherited_v17,
        )
    reference = float(temporal.reference_gain)
    ceiling = float(temporal.maximum_safe_gain)
    requested = float(reference + escalation * (ceiling - reference))
    requested_evidence = _v17._v15.compose_v15_action(
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
        reference_evidence = _v17._v15.compose_v15_action(
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
        tradeoff = _v17._v16.score_v16_tradeoff(
            delta_eig=float(inherited.counterfactual_window_delta_eig),
            candidate_pred_err=float(inherited.counterfactual_candidate_pred_err),
            tol_pos=float(
                _v17._v16._v13.online_pid_exc_snapshot(online_info)["tol_pos_est"]
            ),
            realized_authority=float(inherited.realized_authority),
            tail_phase=float(temporal.tail_phase),
            uncertainty_need=float(temporal.uncertainty_need),
            fallback_fraction=(escalation if reference_fallback else 0.0),
            cfg=env.cfg,
        )
    else:
        tradeoff = _v17._v16.V16RewardTradeoff(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    inherited_v16 = _v17._v16.V16ActionComposition(
        actor_escalation=escalation,
        reference_gain=reference,
        maximum_safe_gain=ceiling,
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
    inherited_v17 = _v17.V17ActionComposition(
        actor_escalation=escalation,
        reference_gain=reference,
        # From the inherited V17 layer's perspective V18 has constrained the
        # safe set itself; this preserves all V17 formula invariants.
        full_safe_gain=min(
            float(full_safe), float(env.cfg.v18_guarded_total_gain_cap)
        ),
        escalation_ceiling_gain=ceiling,
        requested_gain=requested,
        applied_gain=float(inherited_v16.applied_gain),
        escalation_accepted=escalation_accepted,
        reference_fallback=reference_fallback,
        inherited_v16=inherited_v16,
    )
    return V18ActionComposition(
        actor_escalation=escalation,
        reference_gain=reference,
        full_safe_gain=full_safe,
        escalation_ceiling_gain=ceiling,
        requested_gain=requested,
        applied_gain=float(inherited_v16.applied_gain),
        escalation_accepted=escalation_accepted,
        reference_fallback=reference_fallback,
        guard=guard,
        inherited_v17=inherited_v17,
    )


class UUVTwoLeader3DPFEnv(_v17.UUVTwoLeader3DPFEnv):
    """V17 with a causal guard around the complete candidate gain."""

    BASE_OBS_DIM = _v17.UUVTwoLeader3DPFEnv.BASE_OBS_DIM

    def _init_info_ray_diagnostics(self) -> None:
        super()._init_info_ray_diagnostics()
        self._init_v18_action_diagnostics()
        self._init_v18_guard_state()

    def _reset_info_ray_diagnostics(self) -> None:
        super()._reset_info_ray_diagnostics()
        self._init_v18_action_diagnostics()
        self._init_v18_guard_state()

    def _init_v18_action_diagnostics(self) -> None:
        self._v18_actor_escalation = 0.0
        self._v18_reference_gain = 0.0
        self._v18_full_safe_gain = 0.0
        self._v18_escalation_ceiling_gain = 0.0
        self._v18_requested_gain = 0.0
        self._v18_applied_gain = 0.0
        self._v18_escalation_accepted = False
        self._v18_reference_fallback = False
        self._v18_action_guard: Optional[V18GuardDecision] = None

    def _init_v18_guard_state(self) -> None:
        # A state-step cache makes repeated observation/info calls idempotent.
        self._v18_guard_cache_step = -1
        self._v18_guard_cache: Optional[Dict[str, Any]] = None
        self._v18_last_resample_state_step = -1
        self._v18_outcome_had_measurement = False
        self._v18_outcome_ess_min_pre_resample_fraction = 1.0
        self._v18_outcome_resampled_any = False

    def __init__(self, cfg: Optional[UUV3DConfig] = None, render_mode: str = "none"):
        super().__init__(cfg=cfg or UUV3DConfig(), render_mode=render_mode)
        self.cfg: UUV3DConfig

    def _reset_step_accums(self) -> None:
        super()._reset_step_accums()
        self._v18_outcome_had_measurement = False
        self._v18_outcome_ess_min_pre_resample_fraction = 1.0
        self._v18_outcome_resampled_any = False

    def _sim_substep(
        self,
        speed_cmd: float,
        yaw_rate_cmd: float,
        pitch_rate_cmd: float,
        dt: float,
    ):
        stats = super()._sim_substep(
            speed_cmd, yaw_rate_cmd, pitch_rate_cmd, dt
        )
        if int(getattr(stats, "meas_total", 0)) > 0:
            self._v18_outcome_had_measurement = True
            ess = float(getattr(stats, "ess", float("nan")))
            denom = float(max(int(self.cfg.pf_num_particles), 1))
            fraction = ess / denom
            if not np.isfinite(fraction):
                fraction = 0.0  # fail closed on malformed online evidence
            self._v18_outcome_ess_min_pre_resample_fraction = min(
                float(self._v18_outcome_ess_min_pre_resample_fraction),
                float(np.clip(fraction, 0.0, 1.0)),
            )
            if int(getattr(stats, "resampled", 0)) == 1:
                self._v18_outcome_resampled_any = True
        return stats

    def _current_pf_ess_fraction(self) -> float:
        try:
            weights = np.asarray(self.pf.w, dtype=float).reshape(-1)
            if weights.size != int(self.cfg.pf_num_particles):
                return 0.0
            sum_sq = float(np.sum(np.square(weights)))
            if not np.isfinite(sum_sq) or sum_sq <= 0.0:
                return 0.0
            value = (1.0 / sum_sq) / float(max(weights.size, 1))
            return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0
        except Exception:
            return 0.0

    def _v18_guard_inputs(self) -> Dict[str, Any]:
        state_step = int(self.step_count)
        cached = self._v18_guard_cache
        if (
            int(getattr(self, "_v18_guard_cache_step", -1)) == state_step
            and isinstance(cached, dict)
        ):
            return dict(cached)

        previous_resampled = bool(
            getattr(self, "_v18_outcome_resampled_any", False)
        )
        # Cross-check the sticky signal against the inherited whole-action
        # injection accumulator.  The latter is non-zero iff a resampling was
        # followed by the frozen V11 particle injection.
        previous_resampled = previous_resampled or bool(
            int(getattr(self, "pf_injected_step", 0)) > 0
        )
        if state_step > 0 and previous_resampled:
            self._v18_last_resample_state_step = state_step

        last_step = int(getattr(self, "_v18_last_resample_state_step", -1))
        age = state_step - last_step if last_step >= 0 else -1
        recent = bool(0 <= age < int(self.cfg.v18_resample_cooldown_actions))

        if bool(getattr(self, "_v18_outcome_had_measurement", False)):
            ess_fraction = float(
                self._v18_outcome_ess_min_pre_resample_fraction
            )
        else:
            ess_fraction = self._current_pf_ess_fraction()
        if not np.isfinite(ess_fraction):
            ess_fraction = 0.0
        ess_fraction = float(np.clip(ess_fraction, 0.0, 1.0))
        low_ess = bool(ess_fraction <= float(self.cfg.v18_ess_guard_threshold))

        result: Dict[str, Any] = {
            "state_step": state_step,
            "ess_fraction": ess_fraction,
            "previous_action_resampled_any": previous_resampled,
            "last_resample_age_steps": age,
            "trigger_low_ess": low_ess,
            "trigger_recent_resample": recent,
            "active": bool(low_ess or recent),
        }
        self._v18_guard_cache_step = state_step
        self._v18_guard_cache = dict(result)
        return result

    def _get_obs_base(self) -> np.ndarray:
        observation = super()._get_obs_base().astype(np.float32)
        if observation.shape != (self.BASE_OBS_DIM,):
            raise RuntimeError("unexpected inherited v17 observation width")
        grid = self._v15_grid
        if grid is None or int(self._v15_grid_step) != int(self.step_count):
            raise RuntimeError("v18 observation lacks a current opportunity grid")
        online_info = _v17._v16._v14.UUVTwoLeader3DPFEnv._get_info(self)
        temporal, _full_safe, _guard = build_v18_temporal_context(
            self, grid, online_info
        )
        self._v16_temporal_context = temporal
        # V18 caps the complete gain, so both V16 temporal gain slots change.
        observation[-2] = np.float32(temporal.reference_gain)
        observation[-1] = np.float32(temporal.maximum_safe_gain)
        # Action diagnostics are written only before the plant transition.
        return observation

    def _record_v18_composition(self, composition: V18ActionComposition) -> None:
        super()._record_v17_composition(composition.inherited_v17)
        self._v18_actor_escalation = float(composition.actor_escalation)
        self._v18_reference_gain = float(composition.reference_gain)
        self._v18_full_safe_gain = float(composition.full_safe_gain)
        self._v18_escalation_ceiling_gain = float(
            composition.escalation_ceiling_gain
        )
        self._v18_requested_gain = float(composition.requested_gain)
        self._v18_applied_gain = float(composition.applied_gain)
        self._v18_escalation_accepted = bool(composition.escalation_accepted)
        self._v18_reference_fallback = bool(composition.reference_fallback)
        self._v18_action_guard = composition.guard

    def _v18_diagnostic_fields(self) -> Dict[str, Any]:
        guard = self._v18_action_guard
        return {
            "v18_version": 18.0,
            "v18_reward_uses_truth": 0.0,
            "v18_actor_escalation": float(self._v18_actor_escalation),
            "v18_reference_gain": float(self._v18_reference_gain),
            "v18_full_safe_gain": float(self._v18_full_safe_gain),
            "v18_escalation_ceiling_gain": float(
                self._v18_escalation_ceiling_gain
            ),
            "v18_requested_gain": float(self._v18_requested_gain),
            "v18_applied_gain": float(self._v18_applied_gain),
            "v18_escalation_accepted": float(self._v18_escalation_accepted),
            "v18_reference_fallback": float(self._v18_reference_fallback),
            "v18_counterfactual_certificate_ok": float(
                self._v15_counterfactual_certificate_ok
            ),
            "v18_policy_uses_truth": 0.0,
            "v18_max_escalation_delta": V18_MAX_ESCALATION_DELTA,
            "v18_ess_guard_threshold": V18_ESS_GUARD_THRESHOLD,
            "v18_resample_cooldown_actions": V18_RESAMPLE_COOLDOWN_ACTIONS,
            "v18_guarded_total_gain_cap": V18_GUARDED_TOTAL_GAIN_CAP,
            "v18_guard_active": float(guard.active if guard else False),
            "v18_guard_trigger_low_ess": float(
                guard.trigger_low_ess if guard else False
            ),
            "v18_guard_trigger_recent_resample": float(
                guard.trigger_recent_resample if guard else False
            ),
            "v18_decision_state_step": int(guard.state_step if guard else -1),
            "v18_decision_ess_fraction": float(
                guard.decision_ess_fraction if guard else 1.0
            ),
            "v18_previous_action_resampled_any": float(
                guard.previous_action_resampled_any if guard else False
            ),
            "v18_last_resample_age_steps": int(
                guard.last_resample_age_steps if guard else -1
            ),
            "v18_unguarded_reference_gain": float(
                guard.unguarded_reference_gain if guard else 0.0
            ),
            "v18_unguarded_ceiling_gain": float(
                guard.unguarded_ceiling_gain if guard else 0.0
            ),
            "v18_reference_reduction": float(
                (guard.unguarded_reference_gain - guard.guarded_reference_gain)
                if guard
                else 0.0
            ),
            "v18_ceiling_reduction": float(
                (guard.unguarded_ceiling_gain - guard.guarded_ceiling_gain)
                if guard
                else 0.0
            ),
            "v18_outcome_had_measurement": float(
                self._v18_outcome_had_measurement
            ),
            "v18_outcome_ess_min_pre_resample_fraction": float(
                self._v18_outcome_ess_min_pre_resample_fraction
            ),
            "v18_pf_resampled_any_action": float(
                self._v18_outcome_resampled_any
                or int(getattr(self, "pf_injected_step", 0)) > 0
            ),
            "v18_outcome_injected_count": int(
                getattr(self, "pf_injected_step", 0)
            ),
        }

    def _compute_reward(self, *args: Any, **kwargs: Any):
        reward, terms = super()._compute_reward(*args, **kwargs)
        terms.update(self._v18_diagnostic_fields())
        return reward, terms

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)
        info.update(self._v18_diagnostic_fields())
        return info

    def step(self, action: Sequence[float]):
        if self._policy_uses_v16_escalation():
            online_info = self._get_info()
            grid = self._v15_grid
            if grid is None or int(self._v15_grid_step) != int(self.step_count):
                grid = _v17._v15.evaluate_v15_opportunity_grid(
                    self,
                    online_info=online_info,
                    step_count=int(self.step_count),
                    backend=str(self.cfg.v15_predictor_backend),
                )
            composition = compose_v18_action(
                self,
                online_info,
                action,
                int(self.step_count),
                grid=grid,
                backend=str(self.cfg.v15_predictor_backend),
            )
            self._v13_info_ray_active = True
            self._record_v18_composition(composition)
            return _v17._v16._v11.UUVTwoLeader3DPFEnv.step(
                self,
                composition.inherited_v17.inherited_v16.inherited.action_applied,
            )
        # Fixed/direct baselines retain their inherited, unguarded meaning.
        self._init_v18_action_diagnostics()
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
            if not name.startswith("v18_"):
                filtered[key] = value
            elif name in TB_V18_NUMERIC_ALLOWLIST and isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                filtered[key] = value
        copied.append(filtered)
    return _v17._tb_infos_copy(copied)


class V18TBInfoCallback(_v17.V17TBInfoCallback):
    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        callback_locals = dict(locals_)
        callback_locals["infos"] = _tb_infos_copy(locals_.get("infos"))
        return super().__call__(callback_locals, globals_)


def build_parser() -> argparse.ArgumentParser:
    parser = _v17.build_parser()
    replacements = {
        "models_3d_v17_bounded_escalation": "models_3d_v18_resampling_guard",
        "logs_3d_v17_bounded_escalation": "logs_3d_v18_resampling_guard",
        "tb_3d_v17_bounded_escalation": "tb_3d_v18_resampling_guard",
        "eval_3d_logs_v17_bounded_escalation": "eval_3d_logs_v18_resampling_guard",
        "info_maps_v17_bounded_escalation": "info_maps_v18_resampling_guard",
    }
    for action in _v17._v16._v11._iter_parser_actions(parser):
        default = getattr(action, "default", None)
        if isinstance(default, str):
            for old, new in replacements.items():
                if old in default:
                    action.default = default.replace(old, new)
                    break
    train_parser = _v17._v16._v10._get_subparser(parser, "train")
    if train_parser is not None and not _v17._v16._v10._parser_has_dest(
        train_parser, "v18_variant"
    ):
        train_parser.add_argument(
            "--v18-variant",
            choices=(V18_VARIANT,),
            default=V18_VARIANT,
            help="Frozen V18 resampling-risk guard around the complete gain.",
        )
    return parser


@contextmanager
def _patched_training_globals(*, patch_env_class: bool):
    v10 = _v17._v16._v11._v10
    v8 = _v17._v16._v11._v8
    old = {
        "UUV3DConfig": v10.UUV3DConfig,
        "UUVTwoLeader3DPFEnv": v10.UUVTwoLeader3DPFEnv,
        "make_env": v10.make_env,
        "v8_TBInfoCallback": v8.TBInfoCallback,
    }
    v10.UUV3DConfig = UUV3DConfig
    v10.make_env = make_env
    v8.TBInfoCallback = V18TBInfoCallback
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
    "uuv_v18_resampling_guard.py",
    "uuv_v18_evaluate.py",
    "run_v18_sequential_training.py",
    "EXPERIMENT_PROTOCOL_V18.md",
    "tests/test_uuv_v18_resampling_guard.py",
    "tests/test_uuv_v18_evaluate.py",
    "tests/test_run_v18_sequential_training.py",
) + tuple(_v17.SOURCE_NAMES)


def _training_config_preview(args: argparse.Namespace) -> UUV3DConfig:
    inherited = asdict(_v17._training_config_preview(args))
    inherited["v18_variant"] = str(getattr(args, "v18_variant", V18_VARIANT))
    inherited["v18_controller_architecture"] = CONTROLLER_ARCHITECTURE
    inherited["v18_max_escalation_delta"] = V18_MAX_ESCALATION_DELTA
    inherited["v18_ess_guard_threshold"] = V18_ESS_GUARD_THRESHOLD
    inherited["v18_resample_cooldown_actions"] = V18_RESAMPLE_COOLDOWN_ACTIONS
    inherited["v18_guarded_total_gain_cap"] = V18_GUARDED_TOTAL_GAIN_CAP
    return UUV3DConfig(**inherited)


def cmd_train(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resume", False)):
        raise NotImplementedError("scientific resume is disabled")
    requirements = (
        ("v11_variant", "full_online"),
        ("v12_variant", _v17._v16._v12.V12_VARIANT),
        ("v13_variant", _v17._v16._v13.V13_VARIANT),
        ("v14_variant", _v17._v16._v14.V14_VARIANT),
        ("v15_variant", _v17._v15.V15_VARIANT),
        ("v16_variant", _v17._v16.V16_VARIANT),
        ("v17_variant", _v17.V17_VARIANT),
        ("v18_variant", V18_VARIANT),
    )
    for name, expected in requirements:
        if str(getattr(args, name, expected)) != expected:
            raise ValueError(f"v18 requires --{name.replace('_', '-')} {expected}")
    if str(getattr(args, "success_mode", "progress")) != "progress":
        raise ValueError("v18 requires the frozen online progress success definition")
    if not math.isclose(float(getattr(args, "action_dt", 2.0)), 2.0, abs_tol=1e-12):
        raise ValueError("v18 training requires --action-dt 2.0 s")
    if not math.isclose(
        float(getattr(args, "fim_window", _v17._v15.V15_FIM_WINDOW_S)),
        _v17._v15.V15_FIM_WINDOW_S,
        abs_tol=0.0,
    ):
        raise ValueError("v18 training requires --fim-window 30.0 s")

    models_dir = Path(str(args.models_dir)).expanduser().resolve()
    manifest_path = models_dir / MANIFEST_FILENAME
    if models_dir.is_dir() and any(models_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty v18 model directory: {models_dir}"
        )
    source_dir = Path(__file__).resolve().parent
    snapshot_dir = models_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing required v18 source artifact: {source}")
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    cfg_preview = _training_config_preview(args)
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "version": VERSION,
        "variant": V18_VARIANT,
        "seed": int(getattr(args, "seed", 42)),
        "resume": False,
        "command": [sys.executable] + list(sys.argv),
        "arguments": vars(args),
        "environment_config": asdict(cfg_preview),
        "action_contract": cfg_preview.action_contract(),
        "observation_dim": int(cfg_preview.obs_history_len) * int(self_dim()),
        "policy_action_dim": 1,
        "plant_action_dim": 3,
        "packages": _v17._v16._v11._package_versions(),
        "git_commit": _v17._v16._v11._git_commit(),
        "source_sha256": {
            name: _v17._v16._v11._sha256_file(source_dir / name)
            for name in SOURCE_NAMES
        },
        "source_snapshot_dir": str(snapshot_dir),
    }
    _v17._v16._v11._write_json_atomic(manifest_path, manifest)

    previous_variant = os.environ.get("UUV_V11_VARIANT")
    os.environ["UUV_V11_VARIANT"] = "full_online"
    try:
        with _patched_training_globals(patch_env_class=False):
            _v17._v16._v11._v10.cmd_train(args)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["artifacts_sha256"] = {
            name: _v17._v16._v11._sha256_file(models_dir / name)
            for name in ("final_model.zip", "last_model.zip", "vecnormalize.pkl")
        }
        _v17._v16._v11._write_json_atomic(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _v17._v16._v11._write_json_atomic(manifest_path, manifest)
        raise
    finally:
        if previous_variant is None:
            os.environ.pop("UUV_V11_VARIANT", None)
        else:
            os.environ["UUV_V11_VARIANT"] = previous_variant


def self_dim() -> int:
    return int(UUVTwoLeader3DPFEnv.BASE_OBS_DIM)


def cmd_eval(args: argparse.Namespace) -> None:
    del args
    raise RuntimeError("run `python3 uuv_v18_evaluate.py --help` for v18 evaluation")


def cmd_sim(args: argparse.Namespace) -> None:
    with _patched_training_globals(patch_env_class=True):
        _v17._v16._v11._v10.cmd_sim(args)


def cmd_map(args: argparse.Namespace) -> None:
    with _patched_training_globals(patch_env_class=True):
        _v17._v16._v11._v10.cmd_map(args)


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
    "V18ActionComposition",
    "V18GuardDecision",
    "V18_ESS_GUARD_THRESHOLD",
    "V18_GUARDED_TOTAL_GAIN_CAP",
    "V18_MAX_ESCALATION_DELTA",
    "V18_RESAMPLE_COOLDOWN_ACTIONS",
    "V18_VARIANT",
    "VERSION",
    "build_parser",
    "build_v18_temporal_context",
    "compose_v18_action",
    "escalation_ceiling_gain",
    "guarded_gain_bounds",
    "make_env",
]
