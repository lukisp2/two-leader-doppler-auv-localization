# -*- coding: utf-8 -*-
"""Reference controllers for the v11 evaluation protocol.

The two tracking controllers in this module are a source-level restoration of
the controllers used for the final v10 paper evaluation.  Their original
source was overwritten, but its CPython 3.9 bytecode is still present in::

    __pycache__/uuv_v10_info_tracking.cpython-39.pyc

The restored implementation was reconstructed from the code object named
``_v10_tracking_pid_action_from_info`` (original source lines 1224--1338) and
checked instruction-by-instruction against that object.  The v10 evaluation
metadata independently records the same names and aliases.  The bytecode has
SHA-256 ``a946f82c6d1d109d83a6d829f355173362a17c64c4fb1261bace5ac6ca06b2a1``.

``planner_only`` is different: no such controller was found in the v10 source,
bytecode, or evaluation metadata.  It is an explicitly new v11 baseline that
applies the already-computed model-based information-planner action directly.
It must not be described as one of the controllers used for the old results.

This file deliberately depends only on NumPy and the small config protocol
documented below.  It can therefore be tested without importing the simulator
or Stable-Baselines3.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


RESTORED_V10_CONTROLLERS: Tuple[str, ...] = ("pid_track", "pid_track_exc")
NEW_V11_CONTROLLERS: Tuple[str, ...] = ("planner_only",)
SUPPORTED_CONTROLLERS: Tuple[str, ...] = RESTORED_V10_CONTROLLERS + NEW_V11_CONTROLLERS

# These aliases are exactly the aliases stored by the final v10 .pyc and its
# evaluation metadata.  ``planner`` is a new convenience alias for v11.
CONTROLLER_ALIASES: Dict[str, str] = {
    "pid_exc": "pid_track_exc",
    "pid_follow": "pid_track",
    "pid_follow_exc": "pid_track_exc",
    "pid_track_excite": "pid_track_exc",
    "planner": "planner_only",
}

PROVENANCE: Dict[str, Dict[str, Any]] = {
    "pid_track": {
        "status": "restored_exactly_from_v10_cpython39_bytecode",
        "artifact": "__pycache__/uuv_v10_info_tracking.cpython-39.pyc",
        "artifact_sha256": "a946f82c6d1d109d83a6d829f355173362a17c64c4fb1261bace5ac6ca06b2a1",
        "code_object": "_v10_tracking_pid_action_from_info",
        "original_source_lines": (1224, 1338),
    },
    "pid_track_exc": {
        "status": "restored_exactly_from_v10_cpython39_bytecode",
        "artifact": "__pycache__/uuv_v10_info_tracking.cpython-39.pyc",
        "artifact_sha256": "a946f82c6d1d109d83a6d829f355173362a17c64c4fb1261bace5ac6ca06b2a1",
        "code_object": "_v10_tracking_pid_action_from_info",
        "original_source_lines": (1224, 1338),
    },
    "planner_only": {
        "status": "new_v11_baseline_not_present_in_v10",
        "input": "info_plan_a_speed/info_plan_a_yaw/info_plan_a_pitch",
        "fallback": "configurable bounded action; default [0, 0, 0]",
    },
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), float(lo)), float(hi))


def _wrap360(deg: float) -> float:
    return float(deg) % 360.0


def _wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _rad2deg(rad: float) -> float:
    return float(rad) * 180.0 / math.pi


def _sat_ratio(x: float, x0: float) -> float:
    x = max(float(x), 0.0)
    x0 = max(float(x0), 1e-12)
    return x / (x + x0)


def _smoothstep01(t: float) -> float:
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _vel_from_speed_yaw_pitch(speed: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    cp = math.cos(pitch)
    return np.array(
        [
            float(speed) * cp * math.cos(yaw),
            float(speed) * cp * math.sin(yaw),
            float(speed) * math.sin(pitch),
        ],
        dtype=float,
    )


def _info_float(info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read one finite scalar from an environment ``info`` mapping.

    This mirrors the recovered v10 helper: malformed or non-finite values fall
    back to ``default``.  NaN is preserved only when the caller intentionally
    passes ``default=nan``.
    """

    try:
        value = float(info.get(key, default))
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _info_vec3(info: Mapping[str, Any], prefix: str) -> np.ndarray:
    return np.array(
        [
            _info_float(info, f"{prefix}_x", 0.0),
            _info_float(info, f"{prefix}_y", 0.0),
            _info_float(info, f"{prefix}_z", 0.0),
        ],
        dtype=float,
    )


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def resolve_controller_name(controller: str) -> str:
    name = str(controller).lower().strip()
    return CONTROLLER_ALIASES.get(name, name)


def pid_tracking_action_from_info(
    info: Mapping[str, Any],
    cfg: Any,
    mode: str = "pid_track",
    step_count: int = 0,
) -> np.ndarray:
    """Return the restored v10 PID-track or PID-track-excitation action.

    Required config fields are ``f_min_speed``, ``f_max_speed``,
    ``pitch_min_deg``, ``pitch_max_deg``, ``rl_speed_delta_per_step``,
    ``rl_yaw_per_step_deg`` and ``rl_pitch_per_step_deg``.  The historical
    defaults are used only for ``leader_speed_max``, ``tol_std_hard`` and
    ``tol_pos_est_hard``, exactly as in the recovered function.

    The returned action follows the simulator convention
    ``[speed increment, yaw increment, pitch increment]`` normalized to
    ``[-1, 1]``.
    """

    mode = resolve_controller_name(mode)
    if mode not in RESTORED_V10_CONTROLLERS:
        raise ValueError(f"mode must be one of {RESTORED_V10_CONTROLLERS}, got {mode!r}")

    pF_hat = _info_vec3(info, "pFhat")
    pF_des = _info_vec3(info, "pFdes")
    yaw_F = _info_float(info, "yaw_F", 0.0)
    pitch_F = _info_float(info, "pitch_F", 0.0)
    speed_F = _info_float(info, "speed_F", 0.0)

    leader1_speed = _info_float(info, "leader1_speed", 0.0)
    leader2_speed = _info_float(info, "leader2_speed", 0.0)
    yaw_L1 = _info_float(info, "yaw_L1", yaw_F)
    yaw_L2 = _info_float(info, "yaw_L2", yaw_F)
    vL1 = _vel_from_speed_yaw_pitch(leader1_speed, yaw_L1, 0.0)
    vL2 = _vel_from_speed_yaw_pitch(leader2_speed, yaw_L2, 0.0)
    vC = 0.5 * (np.asarray(vL1, dtype=float) + np.asarray(vL2, dtype=float))
    vC_xy = np.array([vC[0], vC[1]], dtype=float)
    vC_speed_xy = float(np.linalg.norm(vC_xy))

    e = pF_des - pF_hat
    e_xy = np.array([e[0], e[1]], dtype=float)
    if vC_speed_xy > 1e-9:
        f_hat = vC_xy / max(vC_speed_xy, 1e-9)
    else:
        e_xy_norm = float(np.linalg.norm(e_xy))
        if e_xy_norm > 1e-9:
            f_hat = e_xy / e_xy_norm
        else:
            yaw_rad = math.radians(yaw_F)
            f_hat = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=float)

    side_hat = np.array([f_hat[1], -f_hat[0]], dtype=float)
    along_err = float(np.dot(e_xy, f_hat))
    side_err = float(np.dot(e_xy, side_hat))
    z_err = float(e[2])

    # Recovered gains and correction caps.  These are intentionally literal:
    # changing them would define a new baseline rather than restore v10.
    k_along = 0.055
    k_side = 0.065
    k_z = 0.055
    max_xy_corr = max(0.75, 0.55 * _cfg_float(cfg, "leader_speed_max", 3.0))
    max_z_corr = 1.2
    along_corr = float(_clamp(k_along * along_err, -max_xy_corr, max_xy_corr))
    side_corr = float(_clamp(k_side * side_err, -max_xy_corr, max_xy_corr))
    vz_des = float(_clamp(k_z * z_err, -max_z_corr, max_z_corr))

    v_des_xy = vC_xy + along_corr * f_hat + side_corr * side_hat
    speed_xy_des = float(np.linalg.norm(v_des_xy))
    if speed_xy_des <= 1e-9:
        speed_xy_des = max(vC_speed_xy, _cfg_float(cfg, "f_min_speed", 0.2))
        v_des_xy = speed_xy_des * f_hat

    yaw_des = _wrap360(_rad2deg(math.atan2(float(v_des_xy[1]), float(v_des_xy[0]))))
    pitch_des = _rad2deg(math.atan2(vz_des, max(speed_xy_des, 1e-6)))
    pitch_des = float(
        _clamp(
            pitch_des,
            _cfg_float(cfg, "pitch_min_deg", -45.0),
            _cfg_float(cfg, "pitch_max_deg", 45.0),
        )
    )
    speed_des = float(
        np.clip(
            math.hypot(speed_xy_des, vz_des),
            _cfg_float(cfg, "f_min_speed", 0.2),
            _cfg_float(cfg, "f_max_speed", 5.0),
        )
    )

    yaw_err = _wrap180(yaw_des - yaw_F)
    pitch_err = pitch_des - pitch_F
    a_speed = (speed_des - speed_F) / max(_cfg_float(cfg, "rl_speed_delta_per_step", 0.4), 1e-6)
    a_yaw = yaw_err / max(_cfg_float(cfg, "rl_yaw_per_step_deg", 20.0), 1e-6)
    a_pitch = pitch_err / max(_cfg_float(cfg, "rl_pitch_per_step_deg", 14.0), 1e-6)
    action = np.array(
        [
            float(_clamp(a_speed, -1.0, 1.0)),
            float(_clamp(a_yaw, -1.0, 1.0)),
            float(_clamp(a_pitch, -1.0, 1.0)),
        ],
        dtype=np.float32,
    )

    if mode == "pid_track_exc":
        std_max = _info_float(info, "std_max_eff", _info_float(info, "std_max", 0.0))
        tol_std = _info_float(info, "tol_std", _cfg_float(cfg, "tol_std_hard", 7.0))
        tol_pos_est = _info_float(info, "tol_pos_est", _cfg_float(cfg, "tol_pos_est_hard", 8.0))
        err_norm = float(np.linalg.norm(e))
        unc_need = _sat_ratio(max(0.0, std_max - tol_std), max(tol_std, 1e-6))
        track_ratio = err_norm / max(tol_pos_est, 1e-6)
        track_guard = 1.0 - _smoothstep01((track_ratio - 2.0) / 3.0)

        planner_action = np.array(
            [
                _info_float(info, "info_plan_a_speed", float("nan")),
                _info_float(info, "info_plan_a_yaw", float("nan")),
                _info_float(info, "info_plan_a_pitch", float("nan")),
            ],
            dtype=float,
        )
        planner_ok = bool(np.all(np.isfinite(planner_action)) and np.linalg.norm(planner_action) > 1e-6)
        gate = _info_float(info, "info_plan_gate", _info_float(info, "v10_info_dir_gate", 0.0))
        gate = float(_clamp(gate, 0.0, 1.0))
        excite = float(_clamp(unc_need * track_guard, 0.0, 1.0))

        if planner_ok:
            action = action + 0.35 * excite * max(0.25, gate) * planner_action.astype(np.float32)
        else:
            action[1] += float(0.20 * math.sin(0.10 * float(step_count)) * excite)
            action[2] += float(0.15 * math.cos(0.12 * float(step_count)) * excite)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

    return action


def planner_only_action_from_info(
    info: Mapping[str, Any],
    *,
    fallback: Sequence[float] = (0.0, 0.0, 0.0),
    minimum_gate: Optional[float] = None,
) -> np.ndarray:
    """Apply the model-based information planner directly (new v11 baseline).

    The planner already scores formation error, information, sensitivity, and
    energy.  This controller therefore applies its normalized candidate action
    without adding the restored PID.  If the planner did not produce a finite
    non-zero candidate, ``fallback`` is used.  ``minimum_gate`` is optional and
    disabled by default; when supplied, a planner action below that diagnostic
    gate also falls back.

    ``fallback`` must itself be finite.  Both planner and fallback actions are
    clipped to the environment action bounds.
    """

    fallback_action = np.asarray(fallback, dtype=float).reshape(-1)
    if fallback_action.shape != (3,) or not np.all(np.isfinite(fallback_action)):
        raise ValueError("fallback must contain exactly three finite values")

    planner_action = np.array(
        [
            _info_float(info, "info_plan_a_speed", float("nan")),
            _info_float(info, "info_plan_a_yaw", float("nan")),
            _info_float(info, "info_plan_a_pitch", float("nan")),
        ],
        dtype=float,
    )
    planner_ok = bool(np.all(np.isfinite(planner_action)) and np.linalg.norm(planner_action) > 1e-6)
    if minimum_gate is not None:
        threshold = _clamp(float(minimum_gate), 0.0, 1.0)
        gate = _info_float(info, "info_plan_gate", _info_float(info, "v10_info_dir_gate", 0.0))
        planner_ok = bool(planner_ok and np.isfinite(gate) and gate >= threshold)

    selected = planner_action if planner_ok else fallback_action
    return np.clip(selected, -1.0, 1.0).astype(np.float32)


def reference_action_from_info(
    controller: str,
    info: Mapping[str, Any],
    cfg: Any,
    step_count: int,
    *,
    planner_fallback: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Common dispatch function for the restored and new v11 baselines."""

    name = resolve_controller_name(controller)
    if name in RESTORED_V10_CONTROLLERS:
        return pid_tracking_action_from_info(info, cfg, name, step_count)
    if name == "planner_only":
        return planner_only_action_from_info(info, fallback=planner_fallback)
    raise ValueError(f"unsupported baseline controller {controller!r}; expected one of {SUPPORTED_CONTROLLERS}")


__all__ = [
    "CONTROLLER_ALIASES",
    "NEW_V11_CONTROLLERS",
    "PROVENANCE",
    "RESTORED_V10_CONTROLLERS",
    "SUPPORTED_CONTROLLERS",
    "pid_tracking_action_from_info",
    "planner_only_action_from_info",
    "reference_action_from_info",
    "resolve_controller_name",
]
