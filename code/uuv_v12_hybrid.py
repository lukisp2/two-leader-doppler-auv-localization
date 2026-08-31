#!/usr/bin/env python3
"""v12 residual-RL controller with a restored PID+exc inner loop.

This module deliberately subclasses the frozen v11 implementation instead of
editing it.  Existing v11 models validate the SHA-256 of their source files;
keeping those files byte-identical preserves the already collected evidence.

The policy action is a *normalized information residual*.  The action applied
to the vehicle is

    clip(PID+exc + tracking_gate * information_gate * scale * residual, -1, 1)

where every gate input is available online.  Once the estimated formation
error reaches the success-corridor boundary, residual authority is exactly
zero and the inner PID+exc loop has full control.
"""

from __future__ import annotations

import argparse
import json
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

import uuv_v11_online as _v11
from baseline_controllers_v11 import pid_tracking_action_from_info


VERSION = "v12_hybrid_1.0"
V12_VARIANT = "hybrid_pid_exc_rl"
CONTROLLER_ARCHITECTURE = "pid_track_exc_plus_bounded_rl_information_residual"
POLICY_CONTROLLER_IDS = frozenset(("policy", "rl"))


@dataclass
class UUV3DConfig(_v11.UUV3DConfig):
    """Frozen v12 hybrid-action contract.

    The v11 observation, reward, estimator, RNG and success semantics remain
    unchanged.  Only the mapping from the policy output to the plant action is
    new in the first v12 pilot.
    """

    v11_variant: str = "full_online"
    v12_variant: str = V12_VARIANT
    v12_controller_architecture: str = CONTROLLER_ARCHITECTURE
    v12_inner_controller: str = "pid_track_exc"
    v12_policy_action_semantics: str = "normalized_information_residual"
    v12_observation_last_action_semantics: str = "applied_composite_action"
    v12_residual_scale_speed: float = 0.15
    v12_residual_scale_yaw: float = 0.30
    v12_residual_scale_pitch: float = 0.25
    v12_track_gate_full_ratio: float = 0.50
    v12_track_gate_zero_ratio: float = 1.00

    def __post_init__(self) -> None:
        requested_v11 = str(self.v11_variant).lower().strip()
        if requested_v11 not in {"", "auto", "full", "full_online"}:
            raise ValueError("v12 hybrid requires the frozen v11 full_online reward")
        self.v11_variant = "full_online"
        super().__post_init__()

        if str(self.v12_variant).lower().strip() != V12_VARIANT:
            raise ValueError(f"unknown v12 variant {self.v12_variant!r}")
        self.v12_variant = V12_VARIANT
        if str(self.v12_inner_controller) != "pid_track_exc":
            raise ValueError("v12 inner controller must be pid_track_exc")
        if str(self.v12_controller_architecture) != CONTROLLER_ARCHITECTURE:
            raise ValueError("v12 controller architecture is frozen")
        if bool(self.use_conservative_std):
            raise ValueError("v12 inner loop requires the v11 raw online PF uncertainty")
        if not bool(self.info_planner_enabled):
            raise ValueError("v12 PID+exc inner loop requires the online information planner")

        scales = self.residual_scales()
        if not np.all(np.isfinite(scales)) or np.any(scales < 0.0) or np.any(scales > 1.0):
            raise ValueError("v12 residual scales must be finite values in [0, 1]")
        full = float(self.v12_track_gate_full_ratio)
        zero = float(self.v12_track_gate_zero_ratio)
        if not (np.isfinite(full) and np.isfinite(zero) and 0.0 <= full < zero):
            raise ValueError("v12 tracking-gate ratios must satisfy 0 <= full < zero")

    def residual_scales(self) -> np.ndarray:
        return np.asarray(
            (
                self.v12_residual_scale_speed,
                self.v12_residual_scale_yaw,
                self.v12_residual_scale_pitch,
            ),
            dtype=np.float32,
        )

    def action_contract(self) -> Dict[str, Any]:
        return {
            "architecture": CONTROLLER_ARCHITECTURE,
            "candidate_controller_id": "rl",
            "inner_controller": "pid_track_exc",
            "policy_action_semantics": "normalized_information_residual",
            "policy_action_shape": [3],
            "policy_action_bounds": [-1.0, 1.0],
            "residual_scale": [
                float(self.v12_residual_scale_speed),
                float(self.v12_residual_scale_yaw),
                float(self.v12_residual_scale_pitch),
            ],
            "track_gate_full_ratio": float(self.v12_track_gate_full_ratio),
            "track_gate_zero_ratio": float(self.v12_track_gate_zero_ratio),
            "information_gate": "max(online_planner_gate, online_uncertainty_need)",
            "composition": "clip(inner_pid_exc + track_gate * information_gate * scale * residual, -1, 1)",
            "observation_last_action_semantics": str(
                self.v12_observation_last_action_semantics
            ),
            "reward_semantics": "unchanged_v11_full_online_on_applied_composite_action",
            "baseline_action_mode": "direct",
        }


@dataclass(frozen=True)
class HybridActionComposition:
    policy_residual: np.ndarray
    inner_pid_exc: np.ndarray
    track_gate: float
    information_gate: float
    uncertainty_need: float
    residual_effective: np.ndarray
    action_pre_clip: np.ndarray
    action_applied: np.ndarray
    clipped_channels: np.ndarray
    formation_error_ratio: float


def _finite_required(info: Mapping[str, Any], key: str) -> float:
    if key not in info:
        raise KeyError(f"missing required online controller field {key!r}")
    value = float(info[key])
    if not np.isfinite(value):
        raise ValueError(f"online controller field {key!r} must be finite")
    return value


def _smoothstep01(value: float) -> float:
    t = float(np.clip(float(value), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def online_pid_exc_snapshot(info: Mapping[str, Any]) -> Dict[str, float]:
    """Build a fail-closed allowlist for the inner controller.

    In particular this function never forwards the legacy oracle FIM/CRLB
    aliases that survive v11's evaluation denylist.  Follower kinematics are
    sourced only from explicit dead-reckoning fields and uncertainty from the
    explicit online v12 alias populated by the environment.
    """

    out: Dict[str, float] = {}
    for prefix in ("pFhat", "pFdes"):
        for axis in "xyz":
            key = f"{prefix}_{axis}"
            out[key] = _finite_required(info, key)

    for key in ("leader1_speed", "leader2_speed", "yaw_L1", "yaw_L2"):
        out[key] = _finite_required(info, key)

    for target, source in (
        ("speed_F", "speed_F_online_meas"),
        ("yaw_F", "yaw_F_online_meas"),
        ("pitch_F", "pitch_F_online_meas"),
    ):
        out[target] = _finite_required(info, source)

    std_online = _finite_required(info, "v12_std_max_eff_online")
    if std_online < 0.0:
        raise ValueError("online PF uncertainty must be non-negative")
    out["std_max_eff"] = std_online
    out["std_max"] = std_online
    pf_std_raw = _finite_required(info, "pf_std_max_raw")
    if pf_std_raw < 0.0:
        raise ValueError("raw PF uncertainty must be non-negative")
    out["pf_std_max_raw"] = pf_std_raw
    tol_std = _finite_required(info, "v12_tol_std_online")
    tol_pos_est = _finite_required(info, "v12_tol_pos_est_online")
    if tol_std <= 0.0 or tol_pos_est <= 0.0:
        raise ValueError("online tolerances must be strictly positive")
    out["tol_std"] = tol_std
    out["tol_pos_est"] = tol_pos_est

    for key in ("info_plan_a_speed", "info_plan_a_yaw", "info_plan_a_pitch"):
        try:
            value = float(info.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        out[key] = value
    planner_gate = float(info.get("info_plan_gate", 0.0))
    if not np.isfinite(planner_gate):
        planner_gate = 0.0
    out["info_plan_gate"] = float(np.clip(planner_gate, 0.0, 1.0))
    out["v10_info_dir_gate"] = out["info_plan_gate"]
    return out


def compose_hybrid_action(
    online_info: Mapping[str, Any],
    cfg: UUV3DConfig,
    policy_residual: Sequence[float],
    step_count: int,
) -> HybridActionComposition:
    """Compose one deterministic, online-only hybrid action."""

    residual = np.asarray(policy_residual, dtype=np.float32).reshape(-1)
    if residual.shape != (3,) or not np.all(np.isfinite(residual)):
        raise ValueError("policy residual must contain exactly three finite values")
    residual = np.clip(residual, -1.0, 1.0).astype(np.float32)

    snapshot = online_pid_exc_snapshot(online_info)
    inner = pid_tracking_action_from_info(
        snapshot,
        cfg,
        mode="pid_track_exc",
        step_count=int(step_count),
    ).astype(np.float32)

    p_hat = np.asarray([snapshot[f"pFhat_{a}"] for a in "xyz"], dtype=float)
    p_des = np.asarray([snapshot[f"pFdes_{a}"] for a in "xyz"], dtype=float)
    tol_pos = float(snapshot["tol_pos_est"])
    error_ratio = float(np.linalg.norm(p_des - p_hat) / tol_pos)
    gate_width = max(
        float(cfg.v12_track_gate_zero_ratio - cfg.v12_track_gate_full_ratio),
        1e-12,
    )
    track_t = (error_ratio - float(cfg.v12_track_gate_full_ratio)) / gate_width
    track_gate = float(1.0 - _smoothstep01(track_t))

    tol_std = float(snapshot["tol_std"])
    uncertainty_need = float(
        np.clip(max(float(snapshot["pf_std_max_raw"]) - tol_std, 0.0) / tol_std, 0.0, 1.0)
    )
    information_gate = float(
        np.clip(max(float(snapshot["info_plan_gate"]), uncertainty_need), 0.0, 1.0)
    )
    effective = (
        residual * cfg.residual_scales() * np.float32(track_gate * information_gate)
    ).astype(np.float32)
    pre_clip = (inner + effective).astype(np.float32)
    applied = np.clip(pre_clip, -1.0, 1.0).astype(np.float32)
    clipped = (np.abs(pre_clip - applied) > np.float32(1e-7)).astype(np.float32)
    return HybridActionComposition(
        policy_residual=residual.copy(),
        inner_pid_exc=inner.copy(),
        track_gate=track_gate,
        information_gate=information_gate,
        uncertainty_need=uncertainty_need,
        residual_effective=effective.copy(),
        action_pre_clip=pre_clip.copy(),
        action_applied=applied.copy(),
        clipped_channels=clipped.copy(),
        formation_error_ratio=error_ratio,
    )


class UUVTwoLeader3DPFEnv(_v11.UUVTwoLeader3DPFEnv):
    """v11 plant/estimator with per-controller hybrid action semantics."""

    BASE_OBS_DIM = _v11.UUVTwoLeader3DPFEnv.BASE_OBS_DIM

    def __init__(self, cfg: Optional[UUV3DConfig] = None, render_mode: str = "none"):
        self._v12_policy_residual = np.zeros(3, dtype=np.float32)
        self._v12_inner_pid_exc = np.zeros(3, dtype=np.float32)
        self._v12_residual_effective = np.zeros(3, dtype=np.float32)
        self._v12_action_pre_clip = np.zeros(3, dtype=np.float32)
        self._v12_action_applied = np.zeros(3, dtype=np.float32)
        self._v12_clipped_channels = np.zeros(3, dtype=np.float32)
        self._v12_track_gate = 0.0
        self._v12_information_gate = 0.0
        self._v12_uncertainty_need = 0.0
        self._v12_error_ratio = float("inf")
        self._v12_hybrid_active = False
        super().__init__(cfg=cfg or UUV3DConfig(), render_mode=render_mode)
        self.cfg: UUV3DConfig

    def _policy_uses_hybrid_action(self) -> bool:
        return str(self.cfg.v11_controller_id) in POLICY_CONTROLLER_IDS

    def _reset_hybrid_diagnostics(self) -> None:
        self._v12_policy_residual[:] = 0.0
        self._v12_inner_pid_exc[:] = 0.0
        self._v12_residual_effective[:] = 0.0
        self._v12_action_pre_clip[:] = 0.0
        self._v12_action_applied[:] = 0.0
        self._v12_clipped_channels[:] = 0.0
        self._v12_track_gate = 0.0
        self._v12_information_gate = 0.0
        self._v12_uncertainty_need = 0.0
        self._v12_error_ratio = float("inf")
        self._v12_hybrid_active = self._policy_uses_hybrid_action()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._reset_hybrid_diagnostics()
        return super().reset(seed=seed, options=options)

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)
        std_online = float(info.get("std_max_eff", self.pf.std_max()))
        if not np.isfinite(std_online):
            std_online = float(self.pf.std_max())
        tol_pos_est, tol_std, _ = self._current_tolerances()
        info.update(
            {
                "v12_version": 12.0,
                "v12_hybrid_active": float(self._v12_hybrid_active),
                "v12_std_max_eff_online": std_online,
                "v12_tol_pos_est_online": float(tol_pos_est),
                "v12_tol_std_online": float(tol_std),
                "v12_track_gate": float(self._v12_track_gate),
                "v12_information_gate": float(self._v12_information_gate),
                "v12_uncertainty_need": float(self._v12_uncertainty_need),
                "v12_formation_error_ratio": float(self._v12_error_ratio),
                "v12_action_clipped_count": float(np.sum(self._v12_clipped_channels)),
            }
        )
        for prefix, values in (
            ("v12_policy_residual", self._v12_policy_residual),
            ("v12_inner_pid_exc", self._v12_inner_pid_exc),
            ("v12_residual_effective", self._v12_residual_effective),
            ("v12_action_pre_clip", self._v12_action_pre_clip),
            ("v12_action_applied", self._v12_action_applied),
            ("v12_action_clipped", self._v12_clipped_channels),
        ):
            for axis, value in zip(("speed", "yaw", "pitch"), values):
                info[f"{prefix}_{axis}"] = float(value)
        return info

    def step(self, action: Sequence[float]):
        external = np.asarray(action, dtype=np.float32).reshape(-1)
        if external.shape != (3,) or not np.all(np.isfinite(external)):
            raise ValueError("environment action must contain exactly three finite values")
        external = np.clip(external, -1.0, 1.0).astype(np.float32)

        if self._policy_uses_hybrid_action():
            self._v12_hybrid_active = True
            composition = compose_hybrid_action(
                self._get_info(),
                self.cfg,
                external,
                self.step_count,
            )
            self._v12_policy_residual = composition.policy_residual.copy()
            self._v12_inner_pid_exc = composition.inner_pid_exc.copy()
            self._v12_residual_effective = composition.residual_effective.copy()
            self._v12_action_pre_clip = composition.action_pre_clip.copy()
            self._v12_action_applied = composition.action_applied.copy()
            self._v12_clipped_channels = composition.clipped_channels.copy()
            self._v12_track_gate = float(composition.track_gate)
            self._v12_information_gate = float(composition.information_gate)
            self._v12_uncertainty_need = float(composition.uncertainty_need)
            self._v12_error_ratio = float(composition.formation_error_ratio)
            applied = composition.action_applied
        else:
            # Reference controllers remain direct.  This prevents a second PID
            # from being added around the PID baseline during paired evaluation.
            self._v12_hybrid_active = False
            self._v12_policy_residual[:] = 0.0
            self._v12_inner_pid_exc[:] = 0.0
            self._v12_residual_effective[:] = 0.0
            self._v12_action_pre_clip = external.copy()
            self._v12_action_applied = external.copy()
            self._v12_clipped_channels[:] = 0.0
            self._v12_track_gate = 0.0
            self._v12_information_gate = 0.0
            self._v12_uncertainty_need = 0.0
            self._v12_error_ratio = float("nan")
            applied = external
        return super().step(applied)


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
    parser = _v11.build_parser()
    replacements = {
        "models_3d_v11_online": "models_3d_v12_hybrid",
        "logs_3d_v11_online": "logs_3d_v12_hybrid",
        "tb_3d_v11_online": "tb_3d_v12_hybrid",
        "eval_3d_logs_v11_online": "eval_3d_logs_v12_hybrid",
        "info_maps_v11_online": "info_maps_v12_hybrid",
    }
    for action in _v11._iter_parser_actions(parser):
        default = getattr(action, "default", None)
        if isinstance(default, str):
            for old, new in replacements.items():
                if old in default:
                    action.default = default.replace(old, new)
                    break
        if getattr(action, "dest", None) == "v11_variant":
            action.choices = ("full_online",)
            action.default = "full_online"
    train_parser = _v11._v10._get_subparser(parser, "train")
    if train_parser is not None and not _v11._v10._parser_has_dest(train_parser, "v12_variant"):
        train_parser.add_argument(
            "--v12-variant",
            choices=(V12_VARIANT,),
            default=V12_VARIANT,
            help="Frozen v12 hybrid controller architecture.",
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
    "uuv_v12_hybrid.py",
    "uuv_v12_evaluate.py",
    "run_v12_experiments.py",
    "EXPERIMENT_PROTOCOL_V12.md",
    "tests/test_uuv_v12_hybrid.py",
    "tests/test_uuv_v12_evaluate.py",
    "tests/test_run_v12_experiments.py",
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
        info_gate_floor_hard=float(
            getattr(args, "info_gate_floor_hard", UUV3DConfig.info_gate_floor_hard)
        ),
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
        v11_variant="full_online",
        v12_variant=str(getattr(args, "v12_variant", V12_VARIANT)),
    )


def cmd_train(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resume", False)):
        raise NotImplementedError(
            "scientific resume is disabled: replay buffer and complete environment state are not saved"
        )
    if str(getattr(args, "v11_variant", "full_online")) != "full_online":
        raise ValueError("v12 hybrid requires --v11-variant full_online")
    if str(getattr(args, "v12_variant", V12_VARIANT)) != V12_VARIANT:
        raise ValueError(f"v12 requires --v12-variant {V12_VARIANT}")
    if str(getattr(args, "success_mode", "progress")) != "progress":
        raise ValueError("v12 uses the frozen v11 online progress success definition")
    if not math.isclose(float(getattr(args, "action_dt", 2.0)), 2.0, abs_tol=1e-12):
        raise ValueError("the v12 training protocol requires --action-dt 2.0 s")

    models_dir = Path(str(args.models_dir)).expanduser().resolve()
    manifest_path = models_dir / "v12_run_manifest.json"
    if models_dir.is_dir() and any(models_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty v12 model directory: {models_dir}")

    source_dir = Path(__file__).resolve().parent
    snapshot_dir = models_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing required v12 source artifact: {source}")
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    cfg_preview = _training_config_preview(args)
    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "version": VERSION,
        "variant": V12_VARIANT,
        "seed": int(getattr(args, "seed", 42)),
        "resume": False,
        "command": [sys.executable] + list(sys.argv),
        "arguments": vars(args),
        "environment_config": asdict(cfg_preview),
        "action_contract": cfg_preview.action_contract(),
        "observation_dim": int(cfg_preview.obs_history_len) * int(UUVTwoLeader3DPFEnv.BASE_OBS_DIM),
        "policy_action_dim": 3,
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
    raise RuntimeError("run `python3 uuv_v12_evaluate.py --help` for v12 evaluation")


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
    "HybridActionComposition",
    "SOURCE_NAMES",
    "UUV3DConfig",
    "UUVTwoLeader3DPFEnv",
    "V12_VARIANT",
    "VERSION",
    "build_parser",
    "compose_hybrid_action",
    "make_env",
    "online_pid_exc_snapshot",
]
