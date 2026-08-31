#!/usr/bin/env python3
"""V17: one-grid-step bounded temporal escalation.

V17 changes exactly one scientific mechanism from V16.  The deterministic
greedy reference remains the floor, but the learned scalar escalation may move
at most one frozen V15 grid interval (0.25) above that reference.  The V16
temporal context, reward, predictor, plant and complete fail-closed certificate
are inherited unchanged.
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

import uuv_v15_contextual_gain as _v15
import uuv_v16_temporal_escalation as _v16


VERSION = "v17_bounded_escalation_1.0"
V17_VARIANT = "bounded_temporal_escalation_v17"
CONTROLLER_ARCHITECTURE = (
    "pid_track_exc_plus_window_certified_greedy_floor_plus_"
    "one_grid_step_rl_escalation"
)
MANIFEST_FILENAME = "v17_run_manifest.json"
MANIFEST_SCHEMA_VERSION = 7
V17_MAX_ESCALATION_DELTA = 0.25

TB_V17_NUMERIC_ALLOWLIST = frozenset(
    (
        "v17_actor_escalation",
        "v17_reference_gain",
        "v17_full_safe_gain",
        "v17_escalation_ceiling_gain",
        "v17_requested_gain",
        "v17_applied_gain",
        "v17_escalation_accepted",
        "v17_reference_fallback",
        "v17_counterfactual_certificate_ok",
    )
)


@dataclass
class UUV3DConfig(_v16.UUV3DConfig):
    """Frozen V17 bounded-escalation contract."""

    v17_variant: str = V17_VARIANT
    v17_controller_architecture: str = CONTROLLER_ARCHITECTURE
    v17_max_escalation_delta: float = V17_MAX_ESCALATION_DELTA

    def __post_init__(self) -> None:
        super().__post_init__()
        if str(self.v17_variant).lower().strip() != V17_VARIANT:
            raise ValueError(f"unknown v17 variant {self.v17_variant!r}")
        self.v17_variant = V17_VARIANT
        if str(self.v17_controller_architecture) != CONTROLLER_ARCHITECTURE:
            raise ValueError("v17 controller architecture is frozen")
        if not math.isclose(
            float(self.v17_max_escalation_delta),
            V17_MAX_ESCALATION_DELTA,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("v17 escalation delta is frozen at one grid step")

    def action_contract(self) -> Dict[str, Any]:
        contract = dict(super().action_contract())
        contract.update(
            {
                "architecture": CONTROLLER_ARCHITECTURE,
                "variant": V17_VARIANT,
                "policy_action_semantics": (
                    "scalar escalation e in [0,1]; ceiling_gain = "
                    "min(full_safe_gain, greedy_reference + 0.25); "
                    "requested_gain = greedy_reference + "
                    "e*(ceiling_gain-greedy_reference)"
                ),
                "maximum_escalation_above_reference": V17_MAX_ESCALATION_DELTA,
                "maximum_escalation_basis": (
                    "one interval of the frozen V15 gain grid"
                ),
                "observation_base_dim": 98,
                "observation_v17_extra_dim": 0,
                "observation_last_feature": "actionable escalation ceiling gain",
                "unchanged_from_v16": [
                    "greedy reference",
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
class V17ActionComposition:
    actor_escalation: float
    reference_gain: float
    full_safe_gain: float
    escalation_ceiling_gain: float
    requested_gain: float
    applied_gain: float
    escalation_accepted: bool
    reference_fallback: bool
    inherited_v16: _v16.V16ActionComposition


def escalation_ceiling_gain(
    reference_gain: float,
    full_safe_gain: float,
    max_delta: float = V17_MAX_ESCALATION_DELTA,
) -> float:
    """Return the actionable ceiling, bounded to one frozen grid interval."""

    reference = float(reference_gain)
    full_safe = max(float(full_safe_gain), reference)
    delta = float(max_delta)
    if not all(np.isfinite((reference, full_safe, delta))):
        raise ValueError("v17 escalation bounds must be finite")
    if reference < 0.0 or full_safe > 1.0 or delta < 0.0:
        raise ValueError("v17 escalation bounds are outside the policy contract")
    return float(min(full_safe, reference + delta))


def build_v17_temporal_context(
    env: "UUVTwoLeader3DPFEnv",
    grid: _v15.V15OpportunityGrid,
    online_info: Optional[Mapping[str, Any]] = None,
) -> Tuple[_v16.V16TemporalContext, float]:
    """Return the V16 context with its full-safe gain replaced by the ceiling."""

    original = _v16.build_v16_temporal_context(env, grid, online_info)
    full_safe = max(float(original.maximum_safe_gain), float(original.reference_gain))
    ceiling = escalation_ceiling_gain(
        original.reference_gain,
        full_safe,
        float(env.cfg.v17_max_escalation_delta),
    )
    return replace(original, maximum_safe_gain=ceiling), full_safe


def compose_v17_action(
    env: "UUVTwoLeader3DPFEnv",
    online_info: Mapping[str, Any],
    actor_escalation: Sequence[float],
    step_count: int,
    *,
    grid: Optional[_v15.V15OpportunityGrid] = None,
    backend: str = "auto",
) -> V17ActionComposition:
    """Compose, certify and apply the bounded V17 escalation."""

    raw = np.asarray(actor_escalation, dtype=np.float32).reshape(-1)
    if raw.shape != (1,) or not np.all(np.isfinite(raw)):
        raise ValueError("v17 actor escalation must contain one finite value")
    escalation = float(raw[0])
    if escalation < 0.0 or escalation > 1.0:
        raise ValueError("v17 actor escalation must be in [0, 1]")
    if grid is None:
        grid = _v15.evaluate_v15_opportunity_grid(
            env,
            online_info=online_info,
            step_count=int(step_count),
            backend=backend,
        )

    temporal, full_safe = build_v17_temporal_context(env, grid, online_info)
    reference = float(temporal.reference_gain)
    ceiling = float(temporal.maximum_safe_gain)
    requested = float(reference + escalation * (ceiling - reference))
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
        tradeoff = _v16.score_v16_tradeoff(
            delta_eig=float(inherited.counterfactual_window_delta_eig),
            candidate_pred_err=float(inherited.counterfactual_candidate_pred_err),
            tol_pos=float(
                _v16._v13.online_pid_exc_snapshot(online_info)["tol_pos_est"]
            ),
            realized_authority=float(inherited.realized_authority),
            tail_phase=float(temporal.tail_phase),
            uncertainty_need=float(temporal.uncertainty_need),
            fallback_fraction=(escalation if reference_fallback else 0.0),
            cfg=env.cfg,
        )
    else:
        tradeoff = _v16.V16RewardTradeoff(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    inherited_v16 = _v16.V16ActionComposition(
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
    return V17ActionComposition(
        actor_escalation=escalation,
        reference_gain=reference,
        full_safe_gain=full_safe,
        escalation_ceiling_gain=ceiling,
        requested_gain=requested,
        applied_gain=float(inherited_v16.applied_gain),
        escalation_accepted=escalation_accepted,
        reference_fallback=reference_fallback,
        inherited_v16=inherited_v16,
    )


class UUVTwoLeader3DPFEnv(_v16.UUVTwoLeader3DPFEnv):
    """The V16 environment with one-grid-step learned escalation authority."""

    BASE_OBS_DIM = _v16.UUVTwoLeader3DPFEnv.BASE_OBS_DIM

    def _init_info_ray_diagnostics(self) -> None:
        super()._init_info_ray_diagnostics()
        self._init_v17_diagnostics()

    def _reset_info_ray_diagnostics(self) -> None:
        super()._reset_info_ray_diagnostics()
        self._init_v17_diagnostics()

    def _init_v17_diagnostics(self) -> None:
        self._v17_actor_escalation = 0.0
        self._v17_reference_gain = 0.0
        self._v17_full_safe_gain = 0.0
        self._v17_escalation_ceiling_gain = 0.0
        self._v17_requested_gain = 0.0
        self._v17_applied_gain = 0.0
        self._v17_escalation_accepted = False
        self._v17_reference_fallback = False

    def __init__(self, cfg: Optional[UUV3DConfig] = None, render_mode: str = "none"):
        super().__init__(cfg=cfg or UUV3DConfig(), render_mode=render_mode)
        self.cfg: UUV3DConfig

    def _get_obs_base(self) -> np.ndarray:
        observation = super()._get_obs_base().astype(np.float32)
        if observation.shape != (self.BASE_OBS_DIM,):
            raise RuntimeError("unexpected inherited v16 observation width")
        grid = self._v15_grid
        if grid is None or int(self._v15_grid_step) != int(self.step_count):
            raise RuntimeError("v17 observation lacks a current opportunity grid")
        online_info = _v16._v14.UUVTwoLeader3DPFEnv._get_info(self)
        temporal, _full_safe = build_v17_temporal_context(self, grid, online_info)
        self._v16_temporal_context = temporal
        observation[-1] = np.float32(temporal.maximum_safe_gain)
        # Action diagnostics are written only by _record_v17_composition().
        # _get_obs_base() also runs after the plant step, for state t+1; writing
        # them here would mix the action from t with bounds from the next state.
        return observation

    def _record_v17_composition(self, composition: V17ActionComposition) -> None:
        super()._record_v16_composition(composition.inherited_v16)
        self._v17_actor_escalation = float(composition.actor_escalation)
        self._v17_reference_gain = float(composition.reference_gain)
        self._v17_full_safe_gain = float(composition.full_safe_gain)
        self._v17_escalation_ceiling_gain = float(
            composition.escalation_ceiling_gain
        )
        self._v17_requested_gain = float(composition.requested_gain)
        self._v17_applied_gain = float(composition.applied_gain)
        self._v17_escalation_accepted = bool(composition.escalation_accepted)
        self._v17_reference_fallback = bool(composition.reference_fallback)

    def _compute_reward(self, *args: Any, **kwargs: Any):
        reward, terms = super()._compute_reward(*args, **kwargs)
        terms.update(
            {
                "v17_version": 17.0,
                "v17_reward_uses_truth": 0.0,
                "v17_actor_escalation": float(self._v17_actor_escalation),
                "v17_reference_gain": float(self._v17_reference_gain),
                "v17_full_safe_gain": float(self._v17_full_safe_gain),
                "v17_escalation_ceiling_gain": float(
                    self._v17_escalation_ceiling_gain
                ),
                "v17_requested_gain": float(self._v17_requested_gain),
                "v17_applied_gain": float(self._v17_applied_gain),
            }
        )
        return reward, terms

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = super()._get_info(extra=extra)
        info.update(
            {
                "v17_version": 17.0,
                "v17_reward_uses_truth": 0.0,
                "v17_actor_escalation": float(self._v17_actor_escalation),
                "v17_reference_gain": float(self._v17_reference_gain),
                "v17_full_safe_gain": float(self._v17_full_safe_gain),
                "v17_escalation_ceiling_gain": float(
                    self._v17_escalation_ceiling_gain
                ),
                "v17_requested_gain": float(self._v17_requested_gain),
                "v17_applied_gain": float(self._v17_applied_gain),
                "v17_escalation_accepted": float(self._v17_escalation_accepted),
                "v17_reference_fallback": float(self._v17_reference_fallback),
                "v17_counterfactual_certificate_ok": float(
                    self._v15_counterfactual_certificate_ok
                ),
                "v17_policy_uses_truth": 0.0,
                "v17_max_escalation_delta": V17_MAX_ESCALATION_DELTA,
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
            composition = compose_v17_action(
                self,
                online_info,
                action,
                int(self.step_count),
                grid=grid,
                backend=str(self.cfg.v15_predictor_backend),
            )
            self._v13_info_ray_active = True
            self._record_v17_composition(composition)
            return _v16._v11.UUVTwoLeader3DPFEnv.step(
                self, composition.inherited_v16.inherited.action_applied
            )
        self._init_v17_diagnostics()
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
            if not name.startswith("v17_"):
                filtered[key] = value
            elif name in TB_V17_NUMERIC_ALLOWLIST and isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                filtered[key] = value
        copied.append(filtered)
    return _v16._tb_infos_copy(copied)


class V17TBInfoCallback(_v16._v8.TBInfoCallback):
    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        callback_locals = dict(locals_)
        callback_locals["infos"] = _tb_infos_copy(locals_.get("infos"))
        return super().__call__(callback_locals, globals_)


def build_parser() -> argparse.ArgumentParser:
    parser = _v16.build_parser()
    replacements = {
        "models_3d_v16_temporal_escalation": "models_3d_v17_bounded_escalation",
        "logs_3d_v16_temporal_escalation": "logs_3d_v17_bounded_escalation",
        "tb_3d_v16_temporal_escalation": "tb_3d_v17_bounded_escalation",
        "eval_3d_logs_v16_temporal_escalation": "eval_3d_logs_v17_bounded_escalation",
        "info_maps_v16_temporal_escalation": "info_maps_v17_bounded_escalation",
    }
    for action in _v16._v11._iter_parser_actions(parser):
        default = getattr(action, "default", None)
        if isinstance(default, str):
            for old, new in replacements.items():
                if old in default:
                    action.default = default.replace(old, new)
                    break
    train_parser = _v16._v10._get_subparser(parser, "train")
    if train_parser is not None and not _v16._v10._parser_has_dest(
        train_parser, "v17_variant"
    ):
        train_parser.add_argument(
            "--v17-variant",
            choices=(V17_VARIANT,),
            default=V17_VARIANT,
            help="Frozen V17 one-grid-step bounded temporal escalation.",
        )
    return parser


@contextmanager
def _patched_training_globals(*, patch_env_class: bool):
    v10 = _v16._v11._v10
    v8 = _v16._v11._v8
    old = {
        "UUV3DConfig": v10.UUV3DConfig,
        "UUVTwoLeader3DPFEnv": v10.UUVTwoLeader3DPFEnv,
        "make_env": v10.make_env,
        "v8_TBInfoCallback": v8.TBInfoCallback,
    }
    v10.UUV3DConfig = UUV3DConfig
    v10.make_env = make_env
    v8.TBInfoCallback = V17TBInfoCallback
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
    "uuv_v17_bounded_escalation.py",
    "uuv_v17_evaluate.py",
    "run_v17_sequential_training.py",
    "EXPERIMENT_PROTOCOL_V17.md",
    "tests/test_uuv_v17_bounded_escalation.py",
    "tests/test_uuv_v17_evaluate.py",
    "tests/test_run_v17_sequential_training.py",
) + tuple(_v16.SOURCE_NAMES)


def _training_config_preview(args: argparse.Namespace) -> UUV3DConfig:
    inherited = asdict(_v16._training_config_preview(args))
    inherited["v17_variant"] = str(getattr(args, "v17_variant", V17_VARIANT))
    inherited["v17_controller_architecture"] = CONTROLLER_ARCHITECTURE
    inherited["v17_max_escalation_delta"] = V17_MAX_ESCALATION_DELTA
    return UUV3DConfig(**inherited)


def cmd_train(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resume", False)):
        raise NotImplementedError("scientific resume is disabled")
    requirements = (
        ("v11_variant", "full_online"),
        ("v12_variant", _v16._v12.V12_VARIANT),
        ("v13_variant", _v16._v13.V13_VARIANT),
        ("v14_variant", _v16._v14.V14_VARIANT),
        ("v15_variant", _v15.V15_VARIANT),
        ("v16_variant", _v16.V16_VARIANT),
        ("v17_variant", V17_VARIANT),
    )
    for name, expected in requirements:
        if str(getattr(args, name, expected)) != expected:
            raise ValueError(f"v17 requires --{name.replace('_', '-')} {expected}")
    if str(getattr(args, "success_mode", "progress")) != "progress":
        raise ValueError("v17 requires the frozen online progress success definition")
    if not math.isclose(float(getattr(args, "action_dt", 2.0)), 2.0, abs_tol=1e-12):
        raise ValueError("v17 training requires --action-dt 2.0 s")
    if not math.isclose(
        float(getattr(args, "fim_window", _v15.V15_FIM_WINDOW_S)),
        _v15.V15_FIM_WINDOW_S,
        abs_tol=0.0,
    ):
        raise ValueError("v17 training requires --fim-window 30.0 s")

    models_dir = Path(str(args.models_dir)).expanduser().resolve()
    manifest_path = models_dir / MANIFEST_FILENAME
    if models_dir.is_dir() and any(models_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty v17 model directory: {models_dir}"
        )
    source_dir = Path(__file__).resolve().parent
    snapshot_dir = models_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing required v17 source artifact: {source}")
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    cfg_preview = _training_config_preview(args)
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "version": VERSION,
        "variant": V17_VARIANT,
        "seed": int(getattr(args, "seed", 42)),
        "resume": False,
        "command": [sys.executable] + list(sys.argv),
        "arguments": vars(args),
        "environment_config": asdict(cfg_preview),
        "action_contract": cfg_preview.action_contract(),
        "observation_dim": int(cfg_preview.obs_history_len) * int(self_dim()),
        "policy_action_dim": 1,
        "plant_action_dim": 3,
        "packages": _v16._v11._package_versions(),
        "git_commit": _v16._v11._git_commit(),
        "source_sha256": {
            name: _v16._v11._sha256_file(source_dir / name) for name in SOURCE_NAMES
        },
        "source_snapshot_dir": str(snapshot_dir),
    }
    _v16._v11._write_json_atomic(manifest_path, manifest)

    previous_variant = os.environ.get("UUV_V11_VARIANT")
    os.environ["UUV_V11_VARIANT"] = "full_online"
    try:
        with _patched_training_globals(patch_env_class=False):
            _v16._v11._v10.cmd_train(args)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["artifacts_sha256"] = {
            name: _v16._v11._sha256_file(models_dir / name)
            for name in ("final_model.zip", "last_model.zip", "vecnormalize.pkl")
        }
        _v16._v11._write_json_atomic(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _v16._v11._write_json_atomic(manifest_path, manifest)
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
    raise RuntimeError("run `python3 uuv_v17_evaluate.py --help` for v17 evaluation")


def cmd_sim(args: argparse.Namespace) -> None:
    with _patched_training_globals(patch_env_class=True):
        _v16._v11._v10.cmd_sim(args)


def cmd_map(args: argparse.Namespace) -> None:
    with _patched_training_globals(patch_env_class=True):
        _v16._v11._v10.cmd_map(args)


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
    "V17ActionComposition",
    "V17_MAX_ESCALATION_DELTA",
    "V17_VARIANT",
    "VERSION",
    "build_parser",
    "build_v17_temporal_context",
    "compose_v17_action",
    "escalation_ceiling_gain",
    "make_env",
]
