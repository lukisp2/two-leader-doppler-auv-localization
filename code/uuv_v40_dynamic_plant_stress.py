#!/usr/bin/env python3
"""Paired policy, command-execution, and current robustness experiment.

The submitted estimator, belief-information planner, audited ACQUIRE--TRACK
gate, and post-lock PID are not modified. The experiment crosses acquisition
policy, low-order command execution, and a bottom-track-visible horizontal
current. Simulator truth remains scoring-only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import uuv_v11_online as v11
from uuv_v18_resampling_guard import UUV3DConfig, UUVTwoLeader3DPFEnv
import uuv_v38_leader_source_ablation as v38


VERSION = "v40_dynamic_current_factorial_1.0"

POLICY_FIXED = v38.POLICY_FIXED
POLICY_ACTIVE = v38.POLICY_ACTIVE
POLICIES: Tuple[str, ...] = (POLICY_FIXED, POLICY_ACTIVE)

DYNAMICS_KINEMATIC = "kinematic"
DYNAMICS_LOW_ORDER = "low_order_dynamic"
DYNAMICS: Tuple[str, ...] = (DYNAMICS_KINEMATIC, DYNAMICS_LOW_ORDER)

CURRENT_NONE = "no_current"
CURRENT_VISIBLE = "bottom_track_visible"
CURRENTS: Tuple[str, ...] = (CURRENT_NONE, CURRENT_VISIBLE)

SMOKE_SEEDS = (49_566, 49_591)
QUALIFICATION_START = 49_900
QUALIFICATION_END = 49_999
FINAL_START = 50_000
FINAL_END = 50_999

CURRENT_STEADY_SPEED_MPS = 0.30
CURRENT_GM_COMPONENT_STD_MPS = 0.05
CURRENT_GM_TIME_CONSTANT_S = 120.0
CURRENT_GM_SIGMA_MPS = CURRENT_GM_COMPONENT_STD_MPS
CURRENT_GM_TAU_S = CURRENT_GM_TIME_CONSTANT_S
CURRENT_STREAM_TAG = 40_120_300


@dataclass(frozen=True)
class PlantParameters:
    command_delay_actions: int
    surge_acceleration_time_constant_s: float
    yaw_rate_time_constant_s: float
    pitch_rate_time_constant_s: float

    @property
    def is_low_order(self) -> bool:
        return bool(
            self.command_delay_actions > 0
            or self.surge_acceleration_time_constant_s > 0.0
            or self.yaw_rate_time_constant_s > 0.0
            or self.pitch_rate_time_constant_s > 0.0
        )

    @property
    def has_first_order_response(self) -> bool:
        return bool(
            self.surge_acceleration_time_constant_s > 0.0
            or self.yaw_rate_time_constant_s > 0.0
            or self.pitch_rate_time_constant_s > 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_delay_actions": int(self.command_delay_actions),
            "command_delay_s": float(2.0 * self.command_delay_actions),
            "surge_acceleration_time_constant_s": float(
                self.surge_acceleration_time_constant_s
            ),
            "yaw_rate_time_constant_s": float(self.yaw_rate_time_constant_s),
            "pitch_rate_time_constant_s": float(
                self.pitch_rate_time_constant_s
            ),
        }


PLANT_PARAMETERS: Mapping[str, PlantParameters] = {
    DYNAMICS_KINEMATIC: PlantParameters(0, 0.0, 0.0, 0.0),
    DYNAMICS_LOW_ORDER: PlantParameters(1, 5.0, 2.0, 3.0),
}


@dataclass(frozen=True)
class ArmSpec:
    policy: str
    dynamics: str
    current: str

    @property
    def name(self) -> str:
        return f"{self.policy}__{self.dynamics}__{self.current}"

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "policy": self.policy,
            "dynamics": self.dynamics,
            "current": self.current,
        }


ARM_SPECS: Tuple[ArmSpec, ...] = tuple(
    ArmSpec(policy, dynamics, current)
    for policy in POLICIES
    for dynamics in DYNAMICS
    for current in CURRENTS
)
ARM_BY_NAME: Mapping[str, ArmSpec] = {arm.name: arm for arm in ARM_SPECS}


def arm_specs() -> Tuple[ArmSpec, ...]:
    return ARM_SPECS


def assert_seed_allowed(seed: int, *, smoke: bool = False) -> None:
    value = int(seed)
    if FINAL_START <= value <= FINAL_END:
        raise PermissionError(f"final holdout seed {value} remains sealed")
    if smoke:
        if value not in SMOKE_SEEDS:
            raise PermissionError("smoke is restricted to previously opened seeds")
        return
    if not QUALIFICATION_START <= value <= QUALIFICATION_END:
        raise PermissionError("qualification uses exactly seeds 49900..49999")


def _assert_seed_allowed_automatic(seed: int) -> None:
    value = int(seed)
    assert_seed_allowed(value, smoke=value in SMOKE_SEEDS)


def _first_order(previous: float, requested: float, tau_s: float, dt_s: float) -> float:
    if tau_s <= 0.0:
        return float(requested)
    alpha = -math.expm1(-float(dt_s) / float(tau_s))
    return float(previous + alpha * (requested - previous))


def _velocity_to_speed_yaw_pitch(
    velocity_mps: Sequence[float],
) -> Tuple[float, float, float]:
    velocity = np.asarray(velocity_mps, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-15:
        return 0.0, 0.0, 0.0
    yaw = v11._v8.wrap360(
        math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
    )
    pitch = math.degrees(
        math.asin(float(np.clip(velocity[2] / speed, -1.0, 1.0)))
    )
    return speed, float(yaw), float(pitch)


@dataclass(frozen=True)
class CurrentTape:
    """Action-independent horizontal current sampled at the plant interval."""

    episode_seed: int
    dt_s: float
    steady_horizontal_velocity_mps: np.ndarray
    gauss_markov_velocity_mps: np.ndarray

    def __post_init__(self) -> None:
        steady = np.asarray(self.steady_horizontal_velocity_mps, dtype=np.float64)
        gm = np.asarray(self.gauss_markov_velocity_mps, dtype=np.float64)
        if steady.shape != (2,) or gm.ndim != 2 or gm.shape[1] != 2:
            raise ValueError("invalid current-tape shape")
        if not np.all(np.isfinite(steady)) or not np.all(np.isfinite(gm)):
            raise ValueError("current tape contains non-finite values")
        object.__setattr__(self, "steady_horizontal_velocity_mps", steady.copy())
        object.__setattr__(self, "gauss_markov_velocity_mps", gm.copy())

    @classmethod
    def generate(
        cls,
        seed: int,
        *,
        horizon_s: float,
        dt_s: float,
    ) -> "CurrentTape":
        if horizon_s < 0.0 or dt_s <= 0.0:
            raise ValueError("invalid current-tape horizon")
        sample_count = int(round(float(horizon_s) / float(dt_s))) + 1
        direction_rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([CURRENT_STREAM_TAG, int(seed), 0])
            )
        )
        process_rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([CURRENT_STREAM_TAG, int(seed), 1])
            )
        )
        direction = float(direction_rng.uniform(0.0, 2.0 * math.pi))
        steady = np.asarray(
            [
                CURRENT_STEADY_SPEED_MPS * math.cos(direction),
                CURRENT_STEADY_SPEED_MPS * math.sin(direction),
            ],
            dtype=np.float64,
        )
        alpha = math.exp(-float(dt_s) / CURRENT_GM_TIME_CONSTANT_S)
        innovation_std = CURRENT_GM_COMPONENT_STD_MPS * math.sqrt(
            1.0 - alpha * alpha
        )
        gm = np.zeros((sample_count, 2), dtype=np.float64)
        gm[0] = process_rng.normal(0.0, CURRENT_GM_COMPONENT_STD_MPS, size=2)
        if sample_count > 1:
            innovations = process_rng.normal(
                0.0,
                innovation_std,
                size=(sample_count - 1, 2),
            )
            for index in range(1, sample_count):
                gm[index] = (
                    alpha * gm[index - 1] + innovations[index - 1]
                )
        return cls(
            episode_seed=int(seed),
            dt_s=float(dt_s),
            steady_horizontal_velocity_mps=steady,
            gauss_markov_velocity_mps=gm,
        )

    @property
    def horizontal_velocity_mps(self) -> np.ndarray:
        return (
            self.gauss_markov_velocity_mps
            + self.steady_horizontal_velocity_mps[None, :]
        )

    @property
    def vectors_mps(self) -> np.ndarray:
        horizontal = self.horizontal_velocity_mps
        return np.column_stack(
            [horizontal, np.zeros(horizontal.shape[0], dtype=np.float64)]
        )

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        metadata = {
            "episode_seed": int(self.episode_seed),
            "dt_s": float(self.dt_s),
            "steady_speed_mps": CURRENT_STEADY_SPEED_MPS,
            "gm_component_std_mps": CURRENT_GM_COMPONENT_STD_MPS,
            "gm_time_constant_s": CURRENT_GM_TIME_CONSTANT_S,
            "stream_tag": CURRENT_STREAM_TAG,
            "shape": list(self.gauss_markov_velocity_mps.shape),
        }
        digest.update(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(
            np.ascontiguousarray(
                self.steady_horizontal_velocity_mps,
                dtype="<f8",
            ).tobytes()
        )
        digest.update(
            np.ascontiguousarray(
                self.gauss_markov_velocity_mps,
                dtype="<f8",
            ).tobytes()
        )
        return digest.hexdigest()


class DynamicCurrentEnv(UUVTwoLeader3DPFEnv):
    """Frozen environment with optional actuator lag/delay and visible current."""

    def __init__(
        self,
        cfg: Optional[UUV3DConfig] = None,
        render_mode: str = "none",
        *,
        dynamics: str = DYNAMICS_KINEMATIC,
        current: str = CURRENT_NONE,
        episode_seed: int = 0,
    ):
        if dynamics not in DYNAMICS or current not in CURRENTS:
            raise ValueError("unknown V40 factor level")
        self.v40_dynamics = str(dynamics)
        self.v40_current = str(current)
        self.v40_episode_seed = int(episode_seed)
        self._v40_action_queue: Deque[np.ndarray] = deque()
        self._v40_rate_state = np.zeros(3, dtype=np.float64)
        self._v40_body_speed = 0.0
        self._v40_body_yaw = 0.0
        self._v40_body_pitch = 0.0
        self._v40_current_tape: Optional[CurrentTape] = None
        self._v40_substep_index = 0
        self._v40_last_current = np.zeros(3, dtype=np.float64)
        self._v40_requested_actions: list[np.ndarray] = []
        self._v40_delivered_actions: list[np.ndarray] = []
        self._v40_rate_state_end: list[np.ndarray] = []
        self._v40_current_end: list[np.ndarray] = []
        self._v40_body_velocity_end: list[np.ndarray] = []
        self._v40_ground_velocity_end: list[np.ndarray] = []
        super().__init__(cfg=cfg, render_mode=render_mode)

    @property
    def plant_parameters(self) -> PlantParameters:
        return PLANT_PARAMETERS[self.v40_dynamics]

    def _reset_v40_state(self) -> None:
        delay = int(self.plant_parameters.command_delay_actions)
        self._v40_action_queue = deque(
            np.zeros(3, dtype=np.float32) for _ in range(delay)
        )
        self._v40_rate_state = np.zeros(3, dtype=np.float64)
        self._v40_body_speed = float(self.speed_F)
        self._v40_body_yaw = float(self.yaw_F)
        self._v40_body_pitch = float(self.pitch_F)
        self._v40_current_tape = CurrentTape.generate(
            seed=self.v40_episode_seed,
            horizon_s=float(self.cfg.max_steps) * float(self.cfg.action_dt),
            dt_s=float(self.cfg.sub_dt),
        )
        self._v40_substep_index = 0
        self._v40_last_current = np.zeros(3, dtype=np.float64)
        self._v40_requested_actions = []
        self._v40_delivered_actions = []
        self._v40_rate_state_end = []
        self._v40_current_end = []
        self._v40_body_velocity_end = []
        self._v40_ground_velocity_end = []

        if self.v40_current == CURRENT_VISIBLE:
            current = self._v40_current_tape.vectors_mps[0]
            body_velocity = v11._v8.vel_from_speed_yaw_pitch(
                self._v40_body_speed,
                self._v40_body_yaw,
                self._v40_body_pitch,
            )
            self._v40_last_current = np.asarray(current, dtype=np.float64).copy()
            self.vF = np.asarray(body_velocity, dtype=np.float64) + current
            speed, yaw, pitch = _velocity_to_speed_yaw_pitch(self.vF)
            self._v11_speed_meas = speed
            self._v11_yaw_meas = yaw
            self._v11_pitch_meas = pitch
            self._v11_vf_meas = np.asarray(self.vF, dtype=np.float64).copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.v40_episode_seed = int(seed)
        result = super().reset(seed=seed, options=options)
        self._reset_v40_state()
        return result

    def _delivered_action(self, requested: np.ndarray) -> np.ndarray:
        delay = int(self.plant_parameters.command_delay_actions)
        if delay <= 0:
            return requested.copy()
        self._v40_action_queue.append(requested.copy())
        delivered = np.asarray(self._v40_action_queue.popleft(), dtype=np.float32)
        if len(self._v40_action_queue) != delay:
            raise RuntimeError("command-delay queue lost its frozen length")
        return delivered

    def step(self, action: Sequence[float]):
        if (
            self.v40_dynamics == DYNAMICS_KINEMATIC
            and self.v40_current == CURRENT_NONE
        ):
            requested = np.asarray(action, dtype=np.float64).reshape(3).copy()
            self._v40_requested_actions.append(requested)
            self._v40_delivered_actions.append(requested.copy())
            result = super().step(action)
            self._v40_body_speed = float(self.speed_F)
            self._v40_body_yaw = float(self.yaw_F)
            self._v40_body_pitch = float(self.pitch_F)
            self._v40_rate_state_end.append(np.zeros(3, dtype=np.float64))
            self._v40_current_end.append(np.zeros(3, dtype=np.float64))
            self._v40_body_velocity_end.append(np.asarray(self.vF).copy())
            self._v40_ground_velocity_end.append(np.asarray(self.vF).copy())
            return result
        requested = np.clip(
            np.asarray(action, dtype=np.float32).reshape(3),
            -1.0,
            1.0,
        )
        delivered = self._delivered_action(requested)
        self._v40_requested_actions.append(requested.astype(np.float64))
        self._v40_delivered_actions.append(delivered.astype(np.float64))
        result = super().step(delivered)
        body_velocity = v11._v8.vel_from_speed_yaw_pitch(
            self._v40_body_speed,
            self._v40_body_yaw,
            self._v40_body_pitch,
        )
        self._v40_rate_state_end.append(self._v40_rate_state.copy())
        self._v40_current_end.append(self._v40_last_current.copy())
        self._v40_body_velocity_end.append(
            np.asarray(body_velocity, dtype=np.float64)
        )
        self._v40_ground_velocity_end.append(
            np.asarray(self.vF, dtype=np.float64).copy()
        )
        return result

    def _current_vector(self) -> np.ndarray:
        tape = self._v40_current_tape
        if tape is None:
            raise RuntimeError("current tape is not initialized")
        if self._v40_substep_index >= tape.gauss_markov_velocity_mps.shape[0]:
            raise RuntimeError("current tape was exhausted")
        if self.v40_current == CURRENT_NONE:
            return np.zeros(3, dtype=np.float64)
        return tape.vectors_mps[self._v40_substep_index].copy()

    def _record_v18_measurement_outcome(self, pf_stats: Any) -> None:
        if int(getattr(pf_stats, "meas_total", 0)) <= 0:
            return
        self._v18_outcome_had_measurement = True
        ess = float(getattr(pf_stats, "ess", float("nan")))
        denominator = float(max(int(self.cfg.pf_num_particles), 1))
        fraction = ess / denominator
        if not np.isfinite(fraction):
            fraction = 0.0
        self._v18_outcome_ess_min_pre_resample_fraction = min(
            float(self._v18_outcome_ess_min_pre_resample_fraction),
            float(np.clip(fraction, 0.0, 1.0)),
        )
        if int(getattr(pf_stats, "resampled", 0)) == 1:
            self._v18_outcome_resampled_any = True

    def _sim_substep(
        self,
        speed_cmd: float,
        yaw_rate_cmd: float,
        pitch_rate_cmd: float,
        dt: float,
    ):
        """V11 named-noise dynamics with a separate through-water body state."""

        if (
            self.v40_dynamics == DYNAMICS_KINEMATIC
            and self.v40_current == CURRENT_NONE
        ):
            stats = super()._sim_substep(
                speed_cmd,
                yaw_rate_cmd,
                pitch_rate_cmd,
                dt,
            )
            self._v40_substep_index += 1
            return stats

        dt = float(max(1e-6, dt))
        parameters = self.plant_parameters
        requested_rates = np.asarray(
            [speed_cmd, yaw_rate_cmd, pitch_rate_cmd],
            dtype=np.float64,
        )
        self._v40_rate_state[0] = _first_order(
            self._v40_rate_state[0],
            requested_rates[0],
            parameters.surge_acceleration_time_constant_s,
            dt,
        )
        self._v40_rate_state[1] = _first_order(
            self._v40_rate_state[1],
            requested_rates[1],
            parameters.yaw_rate_time_constant_s,
            dt,
        )
        self._v40_rate_state[2] = _first_order(
            self._v40_rate_state[2],
            requested_rates[2],
            parameters.pitch_rate_time_constant_s,
            dt,
        )
        self._v40_body_speed = float(
            np.clip(
                self._v40_body_speed + self._v40_rate_state[0] * dt,
                self.cfg.f_min_speed,
                self.cfg.f_max_speed,
            )
        )
        self._v40_body_yaw = v11._v8.wrap360(
            self._v40_body_yaw + self._v40_rate_state[1] * dt
        )
        self._v40_body_pitch = v11._v8.clamp(
            self._v40_body_pitch + self._v40_rate_state[2] * dt,
            self.cfg.pitch_min_deg,
            self.cfg.pitch_max_deg,
        )
        body_velocity = v11._v8.vel_from_speed_yaw_pitch(
            self._v40_body_speed,
            self._v40_body_yaw,
            self._v40_body_pitch,
        )
        current = self._current_vector()
        self._v40_last_current = current.copy()
        ground_velocity = np.asarray(body_velocity, dtype=np.float64) + current

        self.speed_F = float(self._v40_body_speed)
        self.yaw_F = float(self._v40_body_yaw)
        self.pitch_F = float(self._v40_body_pitch)
        self.vF = ground_velocity.copy()

        self.vL1 = v11._v8.vel_from_speed_yaw_pitch(
            self.leader1_speed,
            self.yaw_L1,
            0.0,
        )
        self.vL2 = v11._v8.vel_from_speed_yaw_pitch(
            self.leader2_speed,
            self.yaw_L2,
            0.0,
        )
        self.pL1 = self.pL1 + self.vL1 * dt
        self.pL2 = self.pL2 + self.vL2 * dt
        self.pF = self.pF + self.vF * dt
        self.t += dt

        if self._v11_noise_cursor is not None:
            dr_noise = self._v11_noise_cursor.next_dead_reckoning(
                [
                    self.cfg.sigma_speed,
                    self.cfg.sigma_yaw_deg,
                    self.cfg.sigma_pitch_deg,
                ]
            )
            n_speed, n_yaw, n_pitch = (float(x) for x in dr_noise)
        else:
            n_speed = float(
                self.rng_dead_reckoning.normal(0.0, self.cfg.sigma_speed)
            )
            n_yaw = float(
                self.rng_dead_reckoning.normal(0.0, self.cfg.sigma_yaw_deg)
            )
            n_pitch = float(
                self.rng_dead_reckoning.normal(0.0, self.cfg.sigma_pitch_deg)
            )
        self._v11_last_noise.update(
            {"dr_speed": n_speed, "dr_yaw": n_yaw, "dr_pitch": n_pitch}
        )
        ground_speed, ground_yaw, ground_pitch = _velocity_to_speed_yaw_pitch(
            ground_velocity
        )
        speed_meas = float(ground_speed + n_speed)
        yaw_meas = float(ground_yaw + n_yaw)
        pitch_meas = float(ground_pitch + n_pitch)
        v_f_meas = v11._v8.vel_from_speed_yaw_pitch(
            speed_meas,
            yaw_meas,
            pitch_meas,
        )
        self._v11_speed_meas = speed_meas
        self._v11_yaw_meas = yaw_meas
        self._v11_pitch_meas = pitch_meas
        self._v11_vf_meas = np.asarray(v_f_meas, dtype=float).copy()
        self.pf.predict(vF_meas=v_f_meas, dt=dt)

        pf_stats = v11._v8.PFStats()
        while self.t + 1e-12 >= self.next_s_time:
            t_meas = float(self.next_s_time)
            r1_true = self.pL1 - self.pF
            r2_true = self.pL2 - self.pF
            vrel1_true = self.vL1 - self.vF
            vrel2_true = self.vL2 - self.vF
            s1_true = v11._v8.radial_speed(r1_true, vrel1_true)
            s2_true = v11._v8.radial_speed(r2_true, vrel2_true)
            if self._v11_noise_cursor is not None:
                doppler_noise = self._v11_noise_cursor.next_doppler(
                    self.cfg.sigma_s_true
                )
                n_s1, n_s2 = (float(x) for x in doppler_noise)
            else:
                n_s1 = float(
                    self.rng_doppler.normal(0.0, self.cfg.sigma_s_true)
                )
                n_s2 = float(
                    self.rng_doppler.normal(0.0, self.cfg.sigma_s_true)
                )
            self._v11_last_noise.update(
                {"doppler_l1": n_s1, "doppler_l2": n_s2}
            )
            s1 = float(s1_true + n_s1)
            s2 = float(s2_true + n_s2)
            self.s1_last, self.s2_last = s1, s2
            self._has_s1 = self._has_s2 = True

            gate1 = self._doppler_gate_truth(r1_true, vrel1_true)
            gate2 = self._doppler_gate_truth(r2_true, vrel2_true)
            self.fim_step_meas += 2
            if (not self.cfg.fim_use_gating) or gate1:
                h1 = v11._v8.doppler_H_3d(r1_true, vrel1_true)
                i1 = (h1.T @ h1) / max(
                    self.cfg.sigma_s_true ** 2,
                    1e-18,
                )
                self.fim_total.add_I(t_meas, i1)
                self.fim_win.add_I(t_meas, i1)
                self.fim_step_used += 1
            if (not self.cfg.fim_use_gating) or gate2:
                h2 = v11._v8.doppler_H_3d(r2_true, vrel2_true)
                i2 = (h2.T @ h2) / max(
                    self.cfg.sigma_s_true ** 2,
                    1e-18,
                )
                self.fim_total.add_I(t_meas, i2)
                self.fim_win.add_I(t_meas, i2)
                self.fim_step_used += 1

            gate_factors = self._doppler_gate_pf_factors(vF_meas=v_f_meas)
            self._last_gate_pf_current = [
                float(gate_factors[0]),
                float(gate_factors[1]),
            ]
            p_f_hat_pred = np.asarray(self.pf.mean, dtype=float).copy()
            sigma_hat = float(self.pf.meas_sigma * self.pf.sigma_nis_mult)
            sigma2_hat = max(sigma_hat * sigma_hat, 1e-18)
            for index, (p_l, v_l) in enumerate(
                ((self.pL1, self.vL1), (self.pL2, self.vL2))
            ):
                gate_factor = float(gate_factors[index])
                if gate_factor <= 0.0:
                    continue
                r_hat = np.asarray(p_l - p_f_hat_pred, dtype=float)
                v_rel_hat = np.asarray(v_l - v_f_meas, dtype=float)
                h_hat = v11._v8.doppler_H_3d(r_hat, v_rel_hat)
                information = (h_hat.T @ h_hat) / sigma2_hat
                self.fim_hat_total.add_I(
                    t_meas,
                    gate_factor * information,
                )
                self.fim_hat_win.add_I(
                    t_meas,
                    gate_factor * information,
                )
                if gate_factor >= float(self.cfg.gate_count_thr):
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
                    rho_max=float(self.cfg.start_rho_max)
                    * float(self.cfg.pf_inject_rho_max_mult),
                    cos_phi_max=float(self._cos_phi_max_pf),
                    frac=float(self._pf_inject_frac),
                    mass=float(self.cfg.pf_inject_mass),
                )
            pf_stats.injected = int(injected)
            self.pf_injected_step += int(injected)
            self.meas_total_step += int(pf_stats.meas_total)
            self.meas_used_step += int(pf_stats.used_meas)
            if gate_factors[0] > 0.0:
                self._accum_sens(
                    self.pL1,
                    self.vL1,
                    v_f_meas,
                    weight=float(gate_factors[0]),
                )
            if gate_factors[1] > 0.0:
                self._accum_sens(
                    self.pL2,
                    self.vL2,
                    v_f_meas,
                    weight=float(gate_factors[1]),
                )

            self.pf_ess_step = float(pf_stats.ess)
            self.pf_wmax_step = float(pf_stats.w_max)
            self.pf_resampled_step = int(pf_stats.resampled)
            self.pf_nis_ratio_step = (
                float(pf_stats.nis_ratio)
                if np.isfinite(pf_stats.nis_ratio)
                else float("nan")
            )
            self.pf_consistency_infl_step = float(
                getattr(pf_stats, "consistency_infl", 1.0)
            )
            self.pf_sigma_nis_mult_step = float(
                getattr(pf_stats, "sigma_nis_mult", 1.0)
            )
            self.nis_step = float(pf_stats.nis)
            self.gate_avg_step = (
                float(pf_stats.gate_avg)
                if np.isfinite(pf_stats.gate_avg)
                else float("nan")
            )
            self.gate_min_step = (
                float(pf_stats.gate_min)
                if np.isfinite(pf_stats.gate_min)
                else float("nan")
            )
            self.next_s_time += float(self.cfg.s_meas_period)

        self._v40_substep_index += 1
        self._record_v18_measurement_outcome(pf_stats)
        return pf_stats

    def plant_trace(self) -> Dict[str, np.ndarray]:
        expected = int(self.step_count)
        values = {
            "plant_requested_action": np.asarray(
                self._v40_requested_actions,
                dtype=np.float64,
            ),
            "plant_delivered_action": np.asarray(
                self._v40_delivered_actions,
                dtype=np.float64,
            ),
            "plant_executed_rate_state": np.asarray(
                self._v40_rate_state_end,
                dtype=np.float64,
            ),
            "water_current_mps": np.asarray(
                self._v40_current_end,
                dtype=np.float64,
            ),
            "body_velocity_through_water_mps": np.asarray(
                self._v40_body_velocity_end,
                dtype=np.float64,
            ),
            "ground_velocity_mps": np.asarray(
                self._v40_ground_velocity_end,
                dtype=np.float64,
            ),
        }
        for name, value in values.items():
            if value.shape != (expected, 3) or not np.all(np.isfinite(value)):
                raise RuntimeError(f"invalid V40 trace {name}: {value.shape}")
        return values

    @property
    def current_tape_sha256(self) -> str:
        if self._v40_current_tape is None:
            raise RuntimeError("current tape is unavailable")
        return self._v40_current_tape.content_sha256()


def run_factorial_arm(
    *,
    arm: ArmSpec,
    cfg: UUV3DConfig,
    tape: Any,
    episode_seed: int,
    episode_index: int,
    estimator_config: Any,
    lock_config: Any,
    planner_config: Any,
) -> v38.ArmOutcome:
    """Run one two-leader policy through one frozen execution condition."""

    if arm.name not in ARM_BY_NAME:
        raise ValueError("unknown V40 arm")
    _assert_seed_allowed_automatic(episode_seed)
    holder: Dict[str, DynamicCurrentEnv] = {}
    original_factory = v38.UUVTwoLeader3DPFEnv
    original_seed_guard = v38.assert_seed_allowed

    def factory(
        *,
        cfg: UUV3DConfig,
        render_mode: str = "none",
    ) -> DynamicCurrentEnv:
        env = DynamicCurrentEnv(
            cfg=cfg,
            render_mode=render_mode,
            dynamics=arm.dynamics,
            current=arm.current,
            episode_seed=int(episode_seed),
        )
        holder["env"] = env
        return env

    v38.UUVTwoLeader3DPFEnv = factory  # type: ignore[assignment]
    v38.assert_seed_allowed = _assert_seed_allowed_automatic  # type: ignore[assignment]
    try:
        base = v38.run_arm(
            cfg=cfg,
            tape=tape,
            episode_seed=int(episode_seed),
            episode_index=int(episode_index),
            source_name=v38.SOURCE_BOTH,
            policy_name=arm.policy,
            estimator_config=estimator_config,
            lock_config=lock_config,
            planner_config=planner_config,
        )
    finally:
        v38.UUVTwoLeader3DPFEnv = original_factory
        v38.assert_seed_allowed = original_seed_guard

    env = holder.get("env")
    if env is None:
        raise RuntimeError("V40 environment factory was not used")
    trace = dict(base.trace)
    trace.update(env.plant_trace())
    summary = dict(base.summary)
    summary.update(
        {
            "version": VERSION,
            "arm": arm.name,
            "policy_name": arm.policy,
            "dynamics_name": arm.dynamics,
            "current_name": arm.current,
            "current_tape_sha256": env.current_tape_sha256,
            "plant": {
                "parameters": env.plant_parameters.to_dict(),
                "bottom_track_current_visible": arm.current == CURRENT_VISIBLE,
                "planner_uses_execution_model": False,
                "planner_uses_current_model": False,
                "requested_delivered_action_rms": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                trace["plant_requested_action"]
                                - trace["plant_delivered_action"]
                            )
                        )
                    )
                ),
                "executed_rate_rms": np.sqrt(
                    np.mean(
                        np.square(trace["plant_executed_rate_state"]),
                        axis=0,
                    )
                ).tolist(),
                "current_speed_rms_mps": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                np.square(trace["water_current_mps"]),
                                axis=1,
                            )
                        )
                    )
                ),
                "minimum_ground_speed_mps": float(
                    np.min(np.linalg.norm(trace["ground_velocity_mps"], axis=1))
                ),
                "ground_speed_below_0p05_action_count": int(
                    np.sum(
                        np.linalg.norm(trace["ground_velocity_mps"], axis=1)
                        < 0.05
                    )
                ),
            },
        }
    )
    return v38.ArmOutcome(summary=summary, trace=trace)


def condition_contract() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "design": "paired_2x2x2",
        "policies": list(POLICIES),
        "dynamics": list(DYNAMICS),
        "currents": list(CURRENTS),
        "arms": [arm.to_dict() for arm in ARM_SPECS],
        "plant_parameters": {
            name: value.to_dict() for name, value in PLANT_PARAMETERS.items()
        },
        "current_parameters": {
            "steady_horizontal_speed_mps": CURRENT_STEADY_SPEED_MPS,
            "direction": "uniform_per_episode_from_frozen_current_stream",
            "gauss_markov_component_std_mps": CURRENT_GM_COMPONENT_STD_MPS,
            "gauss_markov_time_constant_s": CURRENT_GM_TIME_CONSTANT_S,
            "vertical_component_mps": 0.0,
            "measurement_convention": "bottom_track_velocity_over_ground",
        },
        "qualification_seeds": [QUALIFICATION_START, QUALIFICATION_END],
        "runs": 100 * len(ARM_SPECS),
        "retuning_allowed": False,
        "sealed_final_range": [FINAL_START, FINAL_END],
    }


__all__ = [
    "VERSION",
    "POLICY_FIXED",
    "POLICY_ACTIVE",
    "POLICIES",
    "DYNAMICS_KINEMATIC",
    "DYNAMICS_LOW_ORDER",
    "DYNAMICS",
    "CURRENT_NONE",
    "CURRENT_VISIBLE",
    "CURRENTS",
    "ARM_SPECS",
    "ARM_BY_NAME",
    "ArmSpec",
    "PlantParameters",
    "CurrentTape",
    "DynamicCurrentEnv",
    "arm_specs",
    "assert_seed_allowed",
    "run_factorial_arm",
    "condition_contract",
]
