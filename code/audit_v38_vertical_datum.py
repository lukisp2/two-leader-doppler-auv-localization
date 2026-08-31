#!/usr/bin/env python3
"""Audit a physical vertical datum and vertical-translation invariance for V38.

The V38 simulator uses a local Cartesian coordinate system and contains no
surface or seabed interaction.  This independent, append-only audit checks
whether every saved trajectory can be embedded in a prespecified deep-water
column without changing relative geometry:

    z_phys = -1000 m + (z_local - c_z0),

where ``c_z0`` is the reset-time two-leader centroid stored in the common
mission support.  It also tests the model's expected invariance to translating
all absolute positions by a constant vertical offset.

This file does not modify or replace any V38 source, campaign result, trace, or
noise tape.  Full campaign clearance is intentionally refused until all 600
prespecified traces are complete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

import run_v20_positioning_ablation as runner20
from uuv_v11_rng import EpisodeSeedPlan, ExogenousNoiseTape
from uuv_v18_resampling_guard import UUVTwoLeader3DPFEnv
import uuv_v19_observability as v19
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24
import uuv_v38_leader_source_ablation as v38


AUDITOR_VERSION = "v38_vertical_datum_auditor_1.0"
SCHEMA_VERSION = 1

EXPECTED_RUNS = 600
EXPECTED_ARMS = tuple(
    v38.arm_name(source, policy)
    for source, policy in v38.arm_pairs()
)
EXPECTED_RUNS_PER_ARM = 100

PHYSICAL_CENTROID_Z_M = -1000.0
WATER_COLUMN_MIN_Z_M = -2000.0
WATER_COLUMN_MAX_Z_M = 0.0
MINIMUM_CLEARANCE_M = 100.0
TRANSLATION_OFFSET_M = -1000.0

DEFAULT_INVARIANCE_SEED = 48_981
DEFAULT_PLANT_STEPS = 40

COORDINATE_TOLERANCE_M = 2.0e-5
RELATIVE_TOLERANCE_M = 2.0e-5
ACTION_TOLERANCE = 1.0e-6
DOPPLER_TOLERANCE_MPS = 1.0e-10
CLOSED_LOOP_SIGNAL_TOLERANCE = 1.0e-6
PLANNER_TOLERANCE = 1.0e-3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_text_atomic(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def physical_z(
    local_z_m: np.ndarray | Sequence[float] | float,
    initial_centroid_z_m: float,
    *,
    physical_centroid_z_m: float = PHYSICAL_CENTROID_Z_M,
) -> np.ndarray:
    """Map local vertical coordinates to the prespecified physical datum."""

    local = np.asarray(local_z_m, dtype=np.float64)
    center = float(initial_centroid_z_m)
    datum = float(physical_centroid_z_m)
    if not math.isfinite(center) or not math.isfinite(datum):
        raise ValueError("vertical datum values must be finite")
    if not np.all(np.isfinite(local)):
        raise ValueError("local vertical coordinates must be finite")
    return datum + (local - center)


def _range(values: Iterable[np.ndarray]) -> Tuple[float, float, int]:
    minimum = float("inf")
    maximum = -float("inf")
    count = 0
    for value in values:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("vertical audit received empty or non-finite values")
        minimum = min(minimum, float(np.min(array)))
        maximum = max(maximum, float(np.max(array)))
        count += int(array.size)
    if count == 0:
        raise ValueError("vertical audit received no values")
    return minimum, maximum, count


def _clearance_record(
    minimum_z_m: float,
    maximum_z_m: float,
    *,
    sample_count: int,
) -> Dict[str, Any]:
    lower = float(WATER_COLUMN_MIN_Z_M)
    upper = float(WATER_COLUMN_MAX_Z_M)
    minimum = float(minimum_z_m)
    maximum = float(maximum_z_m)
    surface_clearance = upper - maximum
    seabed_clearance = minimum - lower
    minimum_clearance = min(surface_clearance, seabed_clearance)
    return {
        "minimum_z_phys_m": minimum,
        "maximum_z_phys_m": maximum,
        "vertical_sample_count": int(sample_count),
        "surface_clearance_m": float(surface_clearance),
        "seabed_clearance_m": float(seabed_clearance),
        "minimum_clearance_m": float(minimum_clearance),
        "inside_water_column": bool(minimum >= lower and maximum <= upper),
        "passes_minimum_clearance": bool(
            surface_clearance >= MINIMUM_CLEARANCE_M
            and seabed_clearance >= MINIMUM_CLEARANCE_M
        ),
    }


def _campaign_expected_entries(
    contract: Mapping[str, Any],
) -> List[Tuple[int, int, str]]:
    seeds = [int(value) for value in contract.get("seeds", [])]
    episode_start = int(contract.get("episode_start", 0))
    arms_raw = contract.get("arms", [])
    arms = [
        str(value["arm"]) if isinstance(value, Mapping) else str(value)
        for value in arms_raw
    ]
    if not seeds or not arms:
        raise RuntimeError("campaign contract lacks seeds or arms")
    return [
        (episode_start + local_index, seed, arm)
        for local_index, seed in enumerate(seeds)
        for arm in arms
    ]


def _entry_paths(
    campaign: Path,
    episode_index: int,
    seed: int,
    arm: str,
) -> Tuple[Path, Path]:
    return (
        campaign
        / "episode_results"
        / f"episode_{episode_index:04d}_seed_{seed}_{arm}.json",
        campaign
        / "traces_npz"
        / arm
        / f"episode_{episode_index:04d}_seed_{seed}.npz",
    )


def audit_campaign_clearance(
    campaign_dir: Path,
    *,
    expected_runs: int = EXPECTED_RUNS,
    expected_runs_per_arm: int = EXPECTED_RUNS_PER_ARM,
    require_complete_progress: bool = True,
) -> Dict[str, Any]:
    """Audit all prespecified V38 follower and leader vertical trajectories."""

    campaign = Path(campaign_dir).expanduser().resolve()
    contract_path = campaign / "control" / "campaign_contract.json"
    progress_path = campaign / "control" / "progress.json"
    if not contract_path.is_file() or not progress_path.is_file():
        raise FileNotFoundError("campaign contract or progress record is missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))

    contract_expected = int(contract.get("expected_runs", -1))
    if contract_expected != int(expected_runs):
        raise RuntimeError(
            f"vertical audit requires {expected_runs} runs; "
            f"contract declares {contract_expected}"
        )
    if require_complete_progress and (
        str(progress.get("status")) != "complete"
        or int(progress.get("completed_runs", -1)) != int(expected_runs)
        or int(progress.get("total_runs", -1)) != int(expected_runs)
    ):
        raise RuntimeError(
            "full vertical clearance audit is refused until the campaign "
            f"is complete ({progress.get('completed_runs')}/{expected_runs}, "
            f"status={progress.get('status')!r})"
        )

    expected_entries = _campaign_expected_entries(contract)
    if len(expected_entries) != int(expected_runs):
        raise RuntimeError("campaign contract does not expand to the expected run count")
    contract_arms = tuple(
        str(value["arm"]) if isinstance(value, Mapping) else str(value)
        for value in contract["arms"]
    )
    if int(expected_runs) == EXPECTED_RUNS and set(contract_arms) != set(EXPECTED_ARMS):
        raise RuntimeError("campaign arm set differs from the frozen six-arm design")

    per_arm_follower: MutableMapping[str, List[np.ndarray]] = defaultdict(list)
    per_arm_leaders: MutableMapping[str, List[np.ndarray]] = defaultdict(list)
    per_arm_combined: MutableMapping[str, List[np.ndarray]] = defaultdict(list)
    run_counts: Counter[str] = Counter()
    seed_centers: MutableMapping[int, List[float]] = defaultdict(list)
    seed_arm_counts: Counter[int] = Counter()
    missing: List[str] = []
    trace_action_counts: List[int] = []
    source_files: List[str] = []

    for episode_index, seed, arm in expected_entries:
        result_path, trace_path = _entry_paths(
            campaign, episode_index, seed, arm
        )
        if not result_path.is_file() or not trace_path.is_file():
            missing.append(
                str(result_path if not result_path.is_file() else trace_path)
            )
            continue
        summary = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            int(summary.get("episode_index", -1)) != episode_index
            or int(summary.get("episode_seed", -1)) != seed
            or str(summary.get("arm")) != arm
        ):
            raise RuntimeError(f"summary identity mismatch: {result_path}")
        support = summary.get("mission_support")
        if not isinstance(support, Mapping):
            raise RuntimeError(f"summary lacks mission support: {result_path}")
        center = np.asarray(support.get("center_m"), dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise RuntimeError(f"invalid mission-support center: {result_path}")
        center_z = float(center[2])
        initial_truth = np.asarray(summary.get("initial_truth_m"), dtype=np.float64)
        if initial_truth.shape != (3,) or not np.all(np.isfinite(initial_truth)):
            raise RuntimeError(f"invalid initial follower truth: {result_path}")

        with np.load(trace_path, allow_pickle=False) as archive:
            required = {
                "truth_z",
                "online_leader_position_m",
                "time_s",
            }
            missing_keys = required - set(archive.files)
            if missing_keys:
                raise RuntimeError(
                    f"trace lacks {sorted(missing_keys)}: {trace_path}"
                )
            truth_z = np.asarray(archive["truth_z"], dtype=np.float64)
            leader_position = np.asarray(
                archive["online_leader_position_m"],
                dtype=np.float64,
            )
            time_s = np.asarray(archive["time_s"], dtype=np.float64)
        if truth_z.ndim != 1 or time_s.shape != truth_z.shape:
            raise RuntimeError(f"invalid follower trace shape: {trace_path}")
        if (
            leader_position.ndim != 3
            or leader_position.shape[1:] != (2, 3)
        ):
            raise RuntimeError(f"invalid leader-history shape: {trace_path}")
        if not (
            np.all(np.isfinite(truth_z))
            and np.all(np.isfinite(leader_position))
            and np.all(np.isfinite(time_s))
        ):
            raise RuntimeError(f"non-finite vertical trace: {trace_path}")

        follower_local = np.concatenate(
            (
                np.asarray([initial_truth[2]], dtype=np.float64),
                truth_z,
            )
        )
        follower_phys = physical_z(follower_local, center_z)
        leaders_phys = physical_z(leader_position[:, :, 2], center_z)
        combined = np.concatenate(
            (follower_phys.reshape(-1), leaders_phys.reshape(-1))
        )

        per_arm_follower[arm].append(follower_phys)
        per_arm_leaders[arm].append(leaders_phys)
        per_arm_combined[arm].append(combined)
        run_counts[arm] += 1
        seed_arm_counts[seed] += 1
        seed_centers[seed].append(center_z)
        trace_action_counts.append(int(truth_z.size))
        source_files.extend((str(result_path), str(trace_path)))

    if missing:
        raise RuntimeError(
            f"vertical audit found {len(missing)} missing run artifacts; "
            f"first={missing[:3]}"
        )
    if sum(run_counts.values()) != int(expected_runs):
        raise RuntimeError("vertical audit did not load the expected number of runs")
    for arm in contract_arms:
        if run_counts[arm] != int(expected_runs_per_arm):
            raise RuntimeError(
                f"arm {arm} has {run_counts[arm]} runs, "
                f"expected {expected_runs_per_arm}"
            )
    expected_arms_per_seed = len(contract_arms)
    if any(
        count != expected_arms_per_seed
        for count in seed_arm_counts.values()
    ):
        raise RuntimeError("at least one seed lacks complete six-arm pairing")
    maximum_center_disagreement = 0.0
    for centers in seed_centers.values():
        maximum_center_disagreement = max(
            maximum_center_disagreement,
            float(np.max(centers) - np.min(centers)),
        )
    if maximum_center_disagreement > 1.0e-12:
        raise RuntimeError("paired arms use different initial vertical centroids")

    per_arm: Dict[str, Any] = {}
    all_follower: List[np.ndarray] = []
    all_leaders: List[np.ndarray] = []
    all_combined: List[np.ndarray] = []
    for arm in contract_arms:
        follower_min, follower_max, follower_count = _range(
            per_arm_follower[arm]
        )
        leader_min, leader_max, leader_count = _range(per_arm_leaders[arm])
        combined_min, combined_max, combined_count = _range(
            per_arm_combined[arm]
        )
        per_arm[arm] = {
            "run_count": int(run_counts[arm]),
            "follower": _clearance_record(
                follower_min,
                follower_max,
                sample_count=follower_count,
            ),
            "leaders": _clearance_record(
                leader_min,
                leader_max,
                sample_count=leader_count,
            ),
            "all_vehicles": _clearance_record(
                combined_min,
                combined_max,
                sample_count=combined_count,
            ),
        }
        all_follower.extend(per_arm_follower[arm])
        all_leaders.extend(per_arm_leaders[arm])
        all_combined.extend(per_arm_combined[arm])

    follower_min, follower_max, follower_count = _range(all_follower)
    leader_min, leader_max, leader_count = _range(all_leaders)
    combined_min, combined_max, combined_count = _range(all_combined)
    overall = {
        "follower": _clearance_record(
            follower_min,
            follower_max,
            sample_count=follower_count,
        ),
        "leaders": _clearance_record(
            leader_min,
            leader_max,
            sample_count=leader_count,
        ),
        "all_vehicles": _clearance_record(
            combined_min,
            combined_max,
            sample_count=combined_count,
        ),
    }
    valid = bool(
        overall["all_vehicles"]["inside_water_column"]
        and overall["all_vehicles"]["passes_minimum_clearance"]
        and all(
            value["all_vehicles"]["passes_minimum_clearance"]
            for value in per_arm.values()
        )
    )
    return {
        "valid": valid,
        "campaign_dir": str(campaign),
        "campaign_contract": str(contract_path),
        "progress_record": str(progress_path),
        "run_count": int(sum(run_counts.values())),
        "seed_count": int(len(seed_arm_counts)),
        "run_counts_by_arm": dict(run_counts),
        "trace_action_count_min": int(min(trace_action_counts)),
        "trace_action_count_max": int(max(trace_action_counts)),
        "maximum_paired_center_disagreement_m": float(
            maximum_center_disagreement
        ),
        "datum": {
            "mapping": "z_phys = -1000 m + (z_local - c_z0)",
            "physical_initial_leader_centroid_z_m": PHYSICAL_CENTROID_Z_M,
            "water_column_min_z_m": WATER_COLUMN_MIN_Z_M,
            "water_column_max_z_m": WATER_COLUMN_MAX_Z_M,
            "minimum_required_clearance_m": MINIMUM_CLEARANCE_M,
            "c_z0_source": "reset-time common mission-support centroid",
        },
        "overall": overall,
        "per_arm": per_arm,
        "source_artifact_count": int(len(source_files)),
        "interpretation": (
            "Deep-water embedding of an unconstrained local relative-navigation "
            "model; this is not a surface/seabed avoidance test."
        ),
    }


def _noise_tape(cfg: Any, seed: int) -> ExogenousNoiseTape:
    plan = EpisodeSeedPlan(root_seed=int(seed), episode_index=0, env_rank=0)
    substeps = int(cfg.max_steps) * int(
        round(float(cfg.action_dt) / float(cfg.sub_dt))
    )
    measurements = int(
        math.ceil(
            float(cfg.max_steps * cfg.action_dt)
            / float(cfg.s_meas_period)
        )
    ) + 2
    return ExogenousNoiseTape.generate(
        plan,
        n_substeps=substeps,
        n_doppler_measurements=measurements,
    )


def translate_environment_state(
    env: UUVTwoLeader3DPFEnv,
    offset_m: Sequence[float],
) -> None:
    """Translate all absolute position state immediately after reset."""

    offset = np.asarray(offset_m, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("translation offset must contain three finite values")
    for name in ("pL1", "pL2", "pF"):
        value = np.asarray(getattr(env, name), dtype=np.float64)
        if value.shape != (3,):
            raise RuntimeError(f"environment position {name} has invalid shape")
        setattr(env, name, value + offset)
    particles = np.asarray(env.pf.p, dtype=np.float64)
    mean = np.asarray(env.pf.mean, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[1] != 3 or mean.shape != (3,):
        raise RuntimeError("particle-filter position state has invalid shape")
    env.pf.p = particles + offset[None, :]
    env.pf.mean = mean + offset


def _max_abs_difference(
    left: np.ndarray | Sequence[float],
    right: np.ndarray | Sequence[float],
) -> Tuple[float, bool]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf"), False
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf"), False
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isinf(a), np.isinf(b)):
        return float("inf"), False
    if not np.any(finite):
        return 0.0, True
    return float(np.max(np.abs(a[finite] - b[finite]))), True


def run_plant_invariance_smoke(
    cfg: Any,
    *,
    seed: int = DEFAULT_INVARIANCE_SEED,
    steps: int = DEFAULT_PLANT_STEPS,
    translation_z_m: float = TRANSLATION_OFFSET_M,
) -> Dict[str, Any]:
    """Run two paired plant/PF instances under a constant vertical shift."""

    count = int(steps)
    if count < 1 or count > int(cfg.max_steps):
        raise ValueError("plant invariance step count is outside the horizon")
    v38.assert_seed_allowed(int(seed))
    tape = _noise_tape(cfg, int(seed))
    baseline = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    shifted = UUVTwoLeader3DPFEnv(cfg=cfg, render_mode="none")
    baseline.attach_exogenous_noise_tape(tape)
    shifted.attach_exogenous_noise_tape(tape)
    baseline.reset(seed=int(seed))
    shifted.reset(seed=int(seed))
    offset = np.asarray([0.0, 0.0, float(translation_z_m)])
    translate_environment_state(shifted, offset)

    maxima = {
        "relative_leader1_position_m": 0.0,
        "relative_leader2_position_m": 0.0,
        "doppler_mps": 0.0,
        "pf_relative_position_m": 0.0,
        "observation": 0.0,
        "reward": 0.0,
        "formation_error_m": 0.0,
        "localization_error_m": 0.0,
        "absolute_translation_error_m": 0.0,
    }
    categorical_equal = True
    started = time.perf_counter()
    try:
        action_rng = np.random.default_rng(int(seed) + 9_973)
        for _ in range(count):
            action = action_rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
            obs_a, reward_a, terminated_a, truncated_a, _ = baseline.step(action)
            obs_b, reward_b, terminated_b, truncated_b, _ = shifted.step(action)
            maxima["relative_leader1_position_m"] = max(
                maxima["relative_leader1_position_m"],
                float(
                    np.max(
                        np.abs(
                            (baseline.pL1 - baseline.pF)
                            - (shifted.pL1 - shifted.pF)
                        )
                    )
                ),
            )
            maxima["relative_leader2_position_m"] = max(
                maxima["relative_leader2_position_m"],
                float(
                    np.max(
                        np.abs(
                            (baseline.pL2 - baseline.pF)
                            - (shifted.pL2 - shifted.pF)
                        )
                    )
                ),
            )
            maxima["doppler_mps"] = max(
                maxima["doppler_mps"],
                abs(float(baseline.s1_last - shifted.s1_last)),
                abs(float(baseline.s2_last - shifted.s2_last)),
            )
            maxima["pf_relative_position_m"] = max(
                maxima["pf_relative_position_m"],
                float(
                    np.max(
                        np.abs(
                            (baseline.pf.mean - baseline.pF)
                            - (shifted.pf.mean - shifted.pF)
                        )
                    )
                ),
            )
            maxima["observation"] = max(
                maxima["observation"],
                float(np.max(np.abs(np.asarray(obs_a) - np.asarray(obs_b)))),
            )
            maxima["reward"] = max(
                maxima["reward"],
                abs(float(reward_a) - float(reward_b)),
            )
            _, desired_a = baseline._formation_desired()
            _, desired_b = shifted._formation_desired()
            maxima["formation_error_m"] = max(
                maxima["formation_error_m"],
                abs(
                    float(np.linalg.norm(baseline.pF - desired_a))
                    - float(np.linalg.norm(shifted.pF - desired_b))
                ),
            )
            maxima["localization_error_m"] = max(
                maxima["localization_error_m"],
                abs(
                    float(np.linalg.norm(baseline.pf.mean - baseline.pF))
                    - float(np.linalg.norm(shifted.pf.mean - shifted.pF))
                ),
            )
            for name in ("pL1", "pL2", "pF"):
                maxima["absolute_translation_error_m"] = max(
                    maxima["absolute_translation_error_m"],
                    float(
                        np.max(
                            np.abs(
                                (np.asarray(getattr(shifted, name))
                                 - np.asarray(getattr(baseline, name)))
                                - offset
                            )
                        )
                    ),
                )
            maxima["absolute_translation_error_m"] = max(
                maxima["absolute_translation_error_m"],
                float(
                    np.max(
                        np.abs(
                            (shifted.pf.mean - baseline.pf.mean) - offset
                        )
                    )
                ),
            )
            categorical_equal = categorical_equal and (
                bool(terminated_a) == bool(terminated_b)
                and bool(truncated_a) == bool(truncated_b)
            )
        cursor_equal = (
            baseline._v11_noise_cursor is not None
            and shifted._v11_noise_cursor is not None
            and baseline._v11_noise_cursor.state_dict()
            == shifted._v11_noise_cursor.state_dict()
        )
    finally:
        baseline.close()
        shifted.close()
    valid = bool(
        categorical_equal
        and cursor_equal
        and maxima["doppler_mps"] <= DOPPLER_TOLERANCE_MPS
        and maxima["relative_leader1_position_m"] <= RELATIVE_TOLERANCE_M
        and maxima["relative_leader2_position_m"] <= RELATIVE_TOLERANCE_M
        and maxima["pf_relative_position_m"] <= RELATIVE_TOLERANCE_M
        and maxima["formation_error_m"] <= RELATIVE_TOLERANCE_M
        and maxima["localization_error_m"] <= RELATIVE_TOLERANCE_M
        and maxima["absolute_translation_error_m"] <= COORDINATE_TOLERANCE_M
        and maxima["observation"] <= ACTION_TOLERANCE
        and maxima["reward"] <= RELATIVE_TOLERANCE_M
    )
    return {
        "valid": valid,
        "seed": int(seed),
        "step_count": count,
        "translation_m": offset.tolist(),
        "maximum_absolute_differences": maxima,
        "termination_and_truncation_equal": bool(categorical_equal),
        "noise_cursor_equal": bool(cursor_equal),
        "elapsed_wall_s": float(time.perf_counter() - started),
    }


def _shifted_environment_class(
    base_class: type[UUVTwoLeader3DPFEnv],
    offset: np.ndarray,
) -> type[UUVTwoLeader3DPFEnv]:
    class ShiftedEnvironment(base_class):
        def reset(self, *args: Any, **kwargs: Any) -> Any:
            result = super().reset(*args, **kwargs)
            translate_environment_state(self, offset)
            return result

    ShiftedEnvironment.__name__ = "VerticallyShiftedV38Environment"
    return ShiftedEnvironment


def _coordinate_shift_error(
    baseline: np.ndarray,
    shifted: np.ndarray,
    expected_shift: float,
) -> Tuple[float, bool]:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(shifted, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf"), False
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return float("inf"), False
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return 0.0, True
    return (
        float(np.max(np.abs((right[finite] - left[finite]) - expected_shift))),
        True,
    )


def run_closed_loop_invariance_smoke(
    cfg: Any,
    *,
    seed: int = DEFAULT_INVARIANCE_SEED,
    translation_z_m: float = TRANSLATION_OFFSET_M,
    coarse_candidates: int = 4096,
    coarse_sweeps: int = 2,
    local_starts: int = 48,
) -> Dict[str, Any]:
    """Run a complete paired both-link active episode under a z translation."""

    v38.assert_seed_allowed(int(seed))
    tape = _noise_tape(cfg, int(seed))
    estimator_config = v19.BatchEstimatorConfig(
        coarse_candidates=int(coarse_candidates),
        coarse_sweeps=int(coarse_sweeps),
        local_starts=int(local_starts),
        gate_mode="raw",
        candidate_radial_distribution="uniform_radius",
    )
    lock_config = v24.AuditedLockConfig()
    planner_config = v22.ActivePlannerConfig()
    offset = np.asarray([0.0, 0.0, float(translation_z_m)])
    run_kwargs = {
        "cfg": cfg,
        "tape": tape,
        "episode_seed": int(seed),
        "episode_index": 0,
        "source_name": v38.SOURCE_BOTH,
        "policy_name": v38.POLICY_ACTIVE,
        "estimator_config": estimator_config,
        "lock_config": lock_config,
        "planner_config": planner_config,
    }
    started = time.perf_counter()
    baseline = v38.run_arm(**run_kwargs)
    original_class = v38.UUVTwoLeader3DPFEnv
    shifted_class = _shifted_environment_class(original_class, offset)
    try:
        v38.UUVTwoLeader3DPFEnv = shifted_class
        shifted = v38.run_arm(**run_kwargs)
    finally:
        v38.UUVTwoLeader3DPFEnv = original_class

    numeric_fields = {
        "action_speed": ACTION_TOLERANCE,
        "action_yaw": ACTION_TOLERANCE,
        "action_pitch": ACTION_TOLERANCE,
        "formation_error_truth_m": RELATIVE_TOLERANCE_M,
        "localization_error_m": RELATIVE_TOLERANCE_M,
        "batch_local_radius95_m": RELATIVE_TOLERANCE_M,
        "planner_utility": PLANNER_TOLERANCE,
        "planner_worst_radius_before_m": PLANNER_TOLERANCE,
        "planner_worst_radius_after_m": PLANNER_TOLERANCE,
        "planner_minimum_pair_chi2": PLANNER_TOLERANCE,
        "online_dead_reckoned_displacement_m": RELATIVE_TOLERANCE_M,
        # The isolated plant test above applies the strict Doppler tolerance.
        # Here a translated solve can perturb an action by a few ulps; that
        # feedback legitimately produces O(1e-8) later signal differences.
        "online_follower_velocity_measured_mps": CLOSED_LOOP_SIGNAL_TOLERANCE,
        "online_doppler_measured_mps": CLOSED_LOOP_SIGNAL_TOLERANCE,
    }
    numeric_differences: Dict[str, float] = {}
    numeric_shapes_and_nan_equal = True
    numeric_within_tolerance = True
    for field, tolerance in numeric_fields.items():
        difference, availability_equal = _max_abs_difference(
            baseline.trace[field],
            shifted.trace[field],
        )
        numeric_differences[field] = difference
        numeric_shapes_and_nan_equal = (
            numeric_shapes_and_nan_equal and availability_equal
        )
        numeric_within_tolerance = (
            numeric_within_tolerance
            and availability_equal
            and difference <= tolerance
        )

    categorical_fields = (
        "phase_track",
        "gate_locked_after_update",
        "gate_release_predicate",
        "gate_hold_predicate",
        "audit_release_checks_pass",
        "audit_hold_checks_pass",
        "source_mask_l1",
        "source_mask_l2",
    )
    categorical_equal = all(
        np.array_equal(
            np.asarray(baseline.trace[field]),
            np.asarray(shifted.trace[field]),
        )
        for field in categorical_fields
    )

    coordinate_errors: Dict[str, float] = {}
    coordinate_availability_equal = True
    for field, expected in (
        ("truth_x", 0.0),
        ("truth_y", 0.0),
        ("truth_z", float(translation_z_m)),
        ("estimate_x", 0.0),
        ("estimate_y", 0.0),
        ("estimate_z", float(translation_z_m)),
    ):
        error, availability_equal = _coordinate_shift_error(
            baseline.trace[field],
            shifted.trace[field],
            expected,
        )
        coordinate_errors[field] = error
        coordinate_availability_equal = (
            coordinate_availability_equal and availability_equal
        )
    leader_expected = np.asarray([0.0, 0.0, float(translation_z_m)])
    leader_left = np.asarray(
        baseline.trace["online_leader_position_m"],
        dtype=np.float64,
    )
    leader_right = np.asarray(
        shifted.trace["online_leader_position_m"],
        dtype=np.float64,
    )
    leader_translation_error = float(
        np.max(
            np.abs(
                (leader_right - leader_left)
                - leader_expected[None, None, :]
            )
        )
    )
    coordinate_errors["online_leader_position_m"] = leader_translation_error

    summary_fields = (
        "terminal_joint_success",
        "tail80_joint_success",
        "dwell15_joint_success",
    )
    summary_categories_equal = all(
        bool(baseline.summary[field]) == bool(shifted.summary[field])
        for field in summary_fields
    ) and (
        bool(baseline.summary["gate"]["ever_locked"])
        == bool(shifted.summary["gate"]["ever_locked"])
        and int(baseline.summary["gate"]["lock_count"])
        == int(shifted.summary["gate"]["lock_count"])
        and int(baseline.summary["gate"]["unlock_count"])
        == int(shifted.summary["gate"]["unlock_count"])
    )
    summary_numeric_fields = (
        "terminal_formation_error_m",
        "terminal_localization_error_m",
        "tail50_joint_occupancy",
        "mean_squared_action",
        "mean_formation_error_after_30_m",
    )
    summary_numeric_differences = {
        field: abs(
            float(baseline.summary[field])
            - float(shifted.summary[field])
        )
        for field in summary_numeric_fields
    }
    summary_numeric_equal = all(
        difference <= RELATIVE_TOLERANCE_M
        for difference in summary_numeric_differences.values()
    )
    full_horizon = bool(
        int(baseline.summary["action_count"]) == int(cfg.max_steps)
        and int(shifted.summary["action_count"]) == int(cfg.max_steps)
    )
    coordinates_valid = bool(
        coordinate_availability_equal
        and all(
            error <= COORDINATE_TOLERANCE_M
            for error in coordinate_errors.values()
        )
    )
    valid = bool(
        full_horizon
        and numeric_shapes_and_nan_equal
        and numeric_within_tolerance
        and categorical_equal
        and coordinates_valid
        and summary_categories_equal
        and summary_numeric_equal
    )
    return {
        "valid": valid,
        "seed": int(seed),
        "translation_m": offset.tolist(),
        "full_horizon_actions": int(cfg.max_steps),
        "publication_estimator_settings": bool(
            int(coarse_candidates) == 4096
            and int(coarse_sweeps) == 2
            and int(local_starts) == 48
        ),
        "estimator_settings": {
            "coarse_candidates": int(coarse_candidates),
            "coarse_sweeps": int(coarse_sweeps),
            "local_starts": int(local_starts),
        },
        "numeric_maximum_absolute_differences": numeric_differences,
        "numeric_shapes_and_nan_patterns_equal": bool(
            numeric_shapes_and_nan_equal
        ),
        "categorical_trace_fields_equal": bool(categorical_equal),
        "coordinate_translation_errors_m": coordinate_errors,
        "summary_categories_equal": bool(summary_categories_equal),
        "summary_numeric_absolute_differences": summary_numeric_differences,
        "full_horizon_completed": full_horizon,
        "elapsed_wall_s": float(time.perf_counter() - started),
    }


def run_invariance_audit(
    cfg: Any,
    *,
    seed: int = DEFAULT_INVARIANCE_SEED,
    plant_steps: int = DEFAULT_PLANT_STEPS,
    coarse_candidates: int = 4096,
    coarse_sweeps: int = 2,
    local_starts: int = 48,
) -> Dict[str, Any]:
    plant = run_plant_invariance_smoke(
        cfg,
        seed=int(seed),
        steps=int(plant_steps),
    )
    closed_loop = run_closed_loop_invariance_smoke(
        cfg,
        seed=int(seed),
        coarse_candidates=int(coarse_candidates),
        coarse_sweeps=int(coarse_sweeps),
        local_starts=int(local_starts),
    )
    return {
        "valid": bool(plant["valid"] and closed_loop["valid"]),
        "translation_offset_z_m": TRANSLATION_OFFSET_M,
        "tolerances": {
            "coordinate_m": COORDINATE_TOLERANCE_M,
            "relative_m": RELATIVE_TOLERANCE_M,
            "action": ACTION_TOLERANCE,
            "doppler_mps": DOPPLER_TOLERANCE_MPS,
            "closed_loop_signal": CLOSED_LOOP_SIGNAL_TOLERANCE,
            "planner": PLANNER_TOLERANCE,
        },
        "plant_level": plant,
        "full_both_link_active_episode": closed_loop,
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# V38 vertical datum and translation-invariance audit",
        "",
        f"- Auditor: `{report['auditor_version']}`",
        f"- Created: {report['created_at_utc']}",
        f"- Executed components valid: **{str(report['valid']).upper()}**",
        f"- Complete audit: **{str(report['complete_audit']).upper()}**",
        "",
        "## Prespecified physical interpretation",
        "",
        "The V38 positions are interpreted in an episode-local Cartesian frame.",
        "The reset-time two-leader centroid is embedded at `z = -1000 m`:",
        "",
        "`z_phys = -1000 m + (z_local - c_z0)`",
        "",
        "The audited water column is `[-2000, 0] m`, with a required clearance",
        "of at least `100 m` from both the surface and seabed. This audit does",
        "not claim surface or seabed avoidance; it checks that those boundaries",
        "are inactive for the saved relative-navigation experiment.",
        "",
    ]
    clearance = report.get("clearance")
    if isinstance(clearance, Mapping):
        overall = clearance["overall"]["all_vehicles"]
        lines.extend(
            [
                "## Full campaign clearance",
                "",
                f"- Runs audited: {clearance['run_count']}",
                f"- Seeds audited: {clearance['seed_count']}",
                (
                    "- Physical vertical range, all vehicles: "
                    f"`[{overall['minimum_z_phys_m']:.3f}, "
                    f"{overall['maximum_z_phys_m']:.3f}] m`"
                ),
                (
                    "- Minimum surface clearance: "
                    f"`{overall['surface_clearance_m']:.3f} m`"
                ),
                (
                    "- Minimum seabed clearance: "
                    f"`{overall['seabed_clearance_m']:.3f} m`"
                ),
                (
                    "- Clearance pass: "
                    f"**{str(overall['passes_minimum_clearance']).upper()}**"
                ),
                "",
                "| Arm | Runs | Minimum z (m) | Maximum z (m) | "
                "Surface clearance (m) | Seabed clearance (m) | Pass |",
                "|---|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for arm, value in clearance["per_arm"].items():
            item = value["all_vehicles"]
            lines.append(
                f"| `{arm}` | {value['run_count']} | "
                f"{item['minimum_z_phys_m']:.3f} | "
                f"{item['maximum_z_phys_m']:.3f} | "
                f"{item['surface_clearance_m']:.3f} | "
                f"{item['seabed_clearance_m']:.3f} | "
                f"{'PASS' if item['passes_minimum_clearance'] else 'FAIL'} |"
            )
        lines.append("")
    invariance = report.get("translation_invariance")
    if isinstance(invariance, Mapping):
        plant = invariance["plant_level"]
        closed = invariance["full_both_link_active_episode"]
        lines.extend(
            [
                "## Translation invariance",
                "",
                (
                    "- Plant-level paired smoke: "
                    f"**{str(plant['valid']).upper()}**, "
                    f"{plant['step_count']} steps"
                ),
                (
                    "- Full both-link active episode: "
                    f"**{str(closed['valid']).upper()}**, "
                    f"{closed['full_horizon_actions']} actions"
                ),
                (
                    "- Publication estimator settings: "
                    f"**{str(closed['publication_estimator_settings']).upper()}**"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            (
                "The local-coordinate campaign admits the prespecified "
                "deep-water interpretation."
                if report["valid"] and report["complete_audit"]
                else (
                    "The executed translation-invariance component passed; "
                    "full campaign clearance remains pending."
                    if report["valid"]
                    else "The prespecified vertical-datum interpretation failed."
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaign",
        nargs="?",
        type=Path,
        default=root / "experiments_v38_leader_source_ablation_dev100",
    )
    parser.add_argument("--environment-metadata", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--skip-clearance", action="store_true")
    parser.add_argument("--skip-invariance", action="store_true")
    parser.add_argument(
        "--invariance-seed",
        type=int,
        default=DEFAULT_INVARIANCE_SEED,
    )
    parser.add_argument("--plant-steps", type=int, default=DEFAULT_PLANT_STEPS)
    parser.add_argument("--coarse-candidates", type=int, default=4096)
    parser.add_argument("--coarse-sweeps", type=int, default=2)
    parser.add_argument("--local-starts", type=int, default=48)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    if args.skip_clearance and args.skip_invariance:
        raise ValueError("at least one audit component must be enabled")
    campaign = Path(args.campaign).expanduser().resolve()
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json is not None
        else campaign / "vertical_datum_audit.json"
    )
    output_md = (
        Path(args.output_md).expanduser().resolve()
        if args.output_md is not None
        else campaign / "vertical_datum_audit.md"
    )
    started = time.perf_counter()
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "auditor_version": AUDITOR_VERSION,
        "created_at_utc": _utc_now(),
        "campaign_dir": str(campaign),
        "clearance": None,
        "translation_invariance": None,
    }
    if not args.skip_clearance:
        report["clearance"] = audit_campaign_clearance(campaign)
    if not args.skip_invariance:
        metadata = (
            Path(args.environment_metadata).expanduser().resolve()
            if args.environment_metadata is not None
            else runner20._default_metadata(Path(__file__).resolve().parent)
        )
        cfg = runner20._load_environment_config(metadata)
        report["environment_metadata"] = str(metadata)
        report["environment_config"] = asdict(cfg)
        report["translation_invariance"] = run_invariance_audit(
            cfg,
            seed=int(args.invariance_seed),
            plant_steps=int(args.plant_steps),
            coarse_candidates=int(args.coarse_candidates),
            coarse_sweeps=int(args.coarse_sweeps),
            local_starts=int(args.local_starts),
        )
    component_validity = [
        bool(value["valid"])
        for value in (
            report.get("clearance"),
            report.get("translation_invariance"),
        )
        if isinstance(value, Mapping)
    ]
    report["valid"] = bool(component_validity and all(component_validity))
    report["complete_audit"] = bool(
        isinstance(report.get("clearance"), Mapping)
        and isinstance(report.get("translation_invariance"), Mapping)
    )
    report["elapsed_wall_s"] = float(time.perf_counter() - started)
    report["output_json"] = str(output_json)
    report["output_md"] = str(output_md)
    _write_json_atomic(output_json, report)
    _write_text_atomic(output_md, _markdown_report(report))
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
