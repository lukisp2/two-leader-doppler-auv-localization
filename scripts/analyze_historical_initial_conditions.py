#!/usr/bin/env python3
"""Reanalyse two frozen campaigns without running or changing any simulation.

The error strata (<=200, (200,300], >300 m) and heading strata (<=90,
>90 degrees) are specified before outcomes are loaded. This is a descriptive
post-hoc reanalysis, not a new confirmatory test or a retuning exercise.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('UUV_INITIAL_ANALYSIS_OUTPUT', ROOT / 'results/initial_condition_reanalysis'))
ERROR_BOUNDS_M = (200.0, 300.0)
HEADING_BOUND_DEG = 90.0
ARCHIVE = Path(os.environ.get('UUV_DATA_ARCHIVE', ROOT / 'data/downloads/two-leader-doppler-auv-data-v1.0.0'))
NOMINAL_LOCAL_MIRROR = Path(os.environ.get('UUV_NOMINAL_MIRROR', ARCHIVE / 'data/raw/leader_source_ablation'))
CONTROLLER = Path(os.environ.get('UUV_CONTROLLER_CAMPAIGN', ROOT / 'data/raw/controller_repair/campaign'))
ARCHIVE_HASHES = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
                  for line in (ARCHIVE / "SHA256SUMS").read_text().splitlines() if line.strip()}
NOMINAL_ARMS = {
    "leader1_only__fixed_s_turn": (7, 6, 18),
    "leader1_only__belief_active": (89, 82, 100),
    "leader2_only__fixed_s_turn": (11, 7, 22),
    "leader2_only__belief_active": (96, 89, 100),
    "both_leaders__fixed_s_turn": (39, 27, 74),
    "both_leaders__belief_active": (100, 99, 100),
}
CONTROLLER_ARMS = {
    "belief_active__low_order_dynamic__baseline_pid__no_current": (38, 2, 0),
    "belief_active__low_order_dynamic__delay_aware__no_current": (95, 91, 95),
    "belief_active__low_order_dynamic__baseline_pid__bottom_track_visible": (26, 6, 6),
    "belief_active__low_order_dynamic__delay_aware__bottom_track_visible": (95, 92, 94),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_label(path):
    for base, label in ((CONTROLLER, 'controller_repair'), (ARCHIVE, 'archive'), (ROOT, 'repository')):
        try:
            return label + '/' + path.relative_to(base).as_posix()
        except ValueError:
            pass
    return path.name


def load_json(path: Path):
    return json.loads(path.read_text())


def local_read_path(path: Path, campaign: Path) -> Path:
    """Use an exact local mirror to avoid rehydrating archived Drive objects."""
    if campaign == ARCHIVE / "data/raw/leader_source_ablation" and NOMINAL_LOCAL_MIRROR.is_dir():
        mirror = NOMINAL_LOCAL_MIRROR / path.relative_to(campaign)
        archive_key = str(path.relative_to(ARCHIVE))
        assert mirror.is_file() and sha(mirror) == ARCHIVE_HASHES[archive_key]
        return mirror
    return path


def load_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    if isinstance(value, bool):
        return value
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"Unexpected boolean: {value!r}")


def initial_state(seed: int, cfg: dict, seed_class):
    """Replay only reset-time RNG draws, not the environment or an episode.

    Exact order is frozen in uuv_v8_temporal_infofix.py:1340-1365, after
    uuv_v11_online.py:384-406 installs the keyed scenario RNG. Reconstructed
    positions, leader velocities and prior center are verified against every
    archived episode before the unlogged initial heading is used.
    """
    assert cfg["difficulty_fixed"] == 1.0
    rng = seed_class(seed, 0, cfg.get("v11_env_rank", 0)).generator("scenario")
    z1 = rng.uniform(*cfg["leader1_depth_range"])
    z2 = rng.uniform(*cfg["leader2_depth_range"])
    s1 = rng.uniform(cfg["leader_speed_min"], cfg["leader_speed_max"])
    s2 = rng.uniform(cfg["leader_speed_min"], cfg["leader_speed_max"])
    y1 = rng.uniform(cfg["leader_yaw_min_deg"], cfg["leader_yaw_max_deg"])
    y2 = (y1 + rng.normal(0.0, 25.0)) % 360.0
    leader_p0 = np.array([[0, cfg["leaders_sep_y"] / 2, z1],
                          [0, -cfg["leaders_sep_y"] / 2, z2]], dtype=float)
    leader_v = np.array([[s * math.cos(math.radians(y)),
                          s * math.sin(math.radians(y)), 0.0]
                         for s, y in ((s1, y1), (s2, y2))])
    centroid = leader_p0.mean(axis=0)
    rho = rng.uniform(cfg["start_rho_min"], cfg["start_rho_max"])
    azimuth = 2.0 * math.pi * rng.random()
    cos_polar = rng.uniform(-1.0, 1.0)
    sin_polar = math.sqrt(max(0.0, 1.0 - cos_polar ** 2))
    p0 = centroid - rho * np.array([math.cos(azimuth) * sin_polar,
                                   math.sin(azimuth) * sin_polar, cos_polar])
    yaw = float(rng.uniform(0.0, 360.0))
    pitch = float(rng.uniform(-10.0, 10.0))
    forward = leader_v.mean(axis=0)
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    desired0 = centroid - cfg["d_back"] * forward
    desired0 += cfg["d_right"] * np.cross(forward, [0, 0, 1])
    desired0[2] = centroid[2] + cfg["dz_offset"]
    formation_yaw = math.degrees(math.atan2(forward[1], forward[0]))
    heading_error = abs((yaw - formation_yaw + 180.0) % 360.0 - 180.0)
    return p0, leader_p0, leader_v, centroid, desired0, yaw, pitch, heading_error


def quantiles(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"available_n": 0, "min": None, "median": None, "p95": None, "max": None}
    return {"available_n": len(finite), "min": min(finite),
            "median": float(np.median(finite)), "p95": float(np.percentile(finite, 95)),
            "max": max(finite)}


def aggregate(rows):
    n = len(rows)
    sat = [r["post_track_saturation_fraction"] for r in rows
           if r["post_track_saturation_fraction"] is not None]
    curvature = [r["post_track_action_curvature_rms"] for r in rows
                 if r["post_track_action_curvature_rms"] is not None]
    result = {
        "n": n,
        "terminal_success_count": sum(r["terminal_joint_success"] for r in rows),
        "tail80_success_count": sum(r["tail80_joint_success"] for r in rows),
        "dwell15_success_count": sum(r["dwell15_joint_success"] for r in rows),
        "ever_track_count": sum(r["ever_locked"] for r in rows),
        "nonfinite_terminal_count": sum(r["nonfinite_terminal"] for r in rows),
        "initial_formation_error_m": quantiles(r["initial_formation_error_m"] for r in rows),
        "initial_centroid_distance_m": quantiles(r["initial_centroid_distance_m"] for r in rows),
        "initial_heading_error_deg": quantiles(r["initial_heading_error_deg"] for r in rows),
        "terminal_formation_error_m": quantiles(r["terminal_formation_error_m"] for r in rows),
        "terminal_localization_error_m": quantiles(r["terminal_localization_error_m"] for r in rows),
        "first_track_time_s_conditional_on_track": quantiles(r["first_track_time_s"] for r in rows),
        "post_track_saturation_mean": float(np.mean(sat)) if sat else None,
        "post_track_saturation_available_n": len(sat),
        "post_track_action_curvature_mean": float(np.mean(curvature)) if curvature else None,
        "post_track_action_curvature_available_n": len(curvature),
    }
    for name in ("terminal", "tail80", "dwell15"):
        result[name + "_success_fraction"] = result[name + "_success_count"] / n if n else None
    return result


def process_campaign(name, campaign, cfg, seed_class, expected, source_records):
    table = load_csv(campaign / "episode_arm_summary.csv")
    keys = [(int(r["episode_seed"]), r["arm"]) for r in table]
    assert len(keys) == len(set(keys)) == 100 * len(expected)
    assert set(r["arm"] for r in table) == set(expected)
    seeds = sorted({s for s, _ in keys})
    assert len(seeds) == 100
    rows, pairing = [], {}
    for raw in sorted(table, key=lambda r: (int(r["episode_seed"]), r["arm"])):
        seed, idx, arm = int(raw["episode_seed"]), int(raw["episode_index"]), raw["arm"]
        stem = f"episode_{idx:04d}_seed_{seed}"
        if name == "nominal_link_acquisition":
            jp = campaign / "episode_results" / f"{stem}_{arm}.json"
        else:
            jp = campaign / "episode_results" / arm / f"{stem}.json"
        tp = campaign / "traces_npz" / arm / f"{stem}.npz"
        read_jp, read_tp = local_read_path(jp, campaign), local_read_path(tp, campaign)
        data = load_json(read_jp)
        assert (data["episode_seed"], data["episode_index"], data["arm"]) == (seed, idx, arm)
        with np.load(read_tp, allow_pickle=False) as trace:
            assert len(trace["time_s"]) == data["action_count"] == 220
            assert np.isclose(trace["time_s"][-1], 440.0, rtol=0, atol=1e-7)
            p0, lp0, lv, center, desired0, yaw, pitch, yaw_error = initial_state(seed, cfg, seed_class)
            assert np.allclose(p0, data["initial_truth_m"], rtol=0, atol=1e-11)
            assert np.allclose(center, data["mission_support"]["center_m"], rtol=0, atol=1e-11)
            times = trace["online_t_s"]
            lv_saved = trace["online_leader_velocity_mps"]
            lp_saved = trace["online_leader_position_m"]
            assert np.allclose(lv_saved, lv, rtol=0, atol=1e-11)
            # The first recorded broadcast is at exactly one second; use this
            # frame to validate reset-time leader positions, not later row labels.
            assert math.isclose(float(times[0]), 1.0, abs_tol=1e-12)
            assert np.allclose(lp_saved[0], lp0 + times[0] * lv,
                               rtol=0, atol=1e-10)
            ep, ef = trace["localization_error_m"], trace["formation_error_truth_m"]
            joint = np.isfinite(ep) & np.isfinite(ef) & (ep < 7.0) & (ef < 8.0)
            scores = {"terminal_joint_success": bool(joint[-1]),
                      "tail80_joint_success": bool(np.sum(joint[-50:]) >= 40),
                      "dwell15_joint_success": bool(np.all(joint[-15:]))}
            for key, value in scores.items():
                assert value == data[key] == boolean(raw[key]), (name, seed, arm, key)
            assert np.isclose(ep[-1], float(raw["terminal_localization_error_m"]), equal_nan=True)
            assert np.isclose(ef[-1], float(raw["terminal_formation_error_m"]), equal_nan=True)
            track = trace["phase_track"].astype(bool)
            first = float(trace["action_start_time_s"][np.flatnonzero(track)[0]]) if track.any() else None
            archived_first = data["gate"].get("first_track_action_time_s")
            assert (first is None) == (archived_first is None)
            if first is not None:
                assert math.isclose(first, archived_first, abs_tol=1e-7)
            assert bool(track.any()) == boolean(raw["ever_locked"])
            action = np.column_stack([trace["action_speed"], trace["action_yaw"], trace["action_pitch"]])
            saturation = float(np.mean(np.any(np.abs(action[track]) >= 0.98, axis=1))) if track.any() else None
            three_track = track[2:] & track[1:-1] & track[:-2]
            second_difference = action[2:] - 2.0 * action[1:-1] + action[:-2]
            curvature = (float(np.sqrt(np.mean(np.sum(second_difference[three_track] ** 2, axis=1))))
                         if three_track.any() else None)
            if "post_track_saturation_fraction" in raw:
                assert saturation is not None and math.isclose(saturation, float(raw["post_track_saturation_fraction"]), abs_tol=1e-12)
                assert curvature is not None and math.isclose(curvature, float(raw["post_track_action_curvature_rms"]), abs_tol=1e-12)
            # Keyed geometry and noise must match all arms for each scenario.
            signature = (data["initial_truth_m"], data["mission_support"],
                         lp_saved[0].tolist(), lv_saved[0].tolist(), data["noise_tape_sha256"])
            if seed in pairing:
                assert signature == pairing[seed]
            else:
                pairing[seed] = signature
            e0 = float(np.linalg.norm(p0 - desired0))
            dist = float(np.linalg.norm(p0 - center))
            assert 120 <= dist <= 350
            row = {"campaign": name, "episode_seed": seed, "episode_index": idx, "arm": arm,
                   "initial_position_m": p0.tolist(), "initial_desired_position_m": desired0.tolist(),
                   "initial_formation_error_m": e0, "initial_centroid_distance_m": dist,
                   "initial_yaw_deg": yaw, "initial_pitch_deg": pitch,
                   "initial_heading_error_deg": yaw_error,
                   "formation_error_stratum": "le200" if e0 <= 200 else "200to300" if e0 <= 300 else "gt300",
                   "heading_error_stratum": "le90" if yaw_error <= 90 else "gt90",
                   **scores, "ever_locked": bool(track.any()), "first_track_time_s": first,
                   "post_track_saturation_fraction": saturation,
                   "post_track_action_curvature_rms": curvature,
                   "terminal_formation_error_m": float(ef[-1]),
                   "terminal_localization_error_m": float(ep[-1]),
                   "nonfinite_terminal": bool(not (np.isfinite(ep[-1]) and np.isfinite(ef[-1]))),
                   "source_json": source_label(jp), "source_json_sha256": sha(read_jp),
                   "source_trace": source_label(tp), "source_trace_sha256": sha(read_tp),
                   "exact_local_mirror_used": read_tp != tp}
            rows.append(row)
            if len(rows) % (10 * len(expected)) == 0:
                print(name, f"{len(rows)}/{len(table)} episode-arms verified", flush=True)
    groups = {}
    for arm in expected:
        selected = [r for r in rows if r["arm"] == arm]
        overall = aggregate(selected)
        observed = (overall["terminal_success_count"], overall["tail80_success_count"],
                    overall["ever_track_count"] if name == "nominal_link_acquisition" else overall["dwell15_success_count"])
        assert observed == expected[arm], (name, arm, observed, expected[arm])
        groups[arm] = {"all": overall,
                       "by_initial_formation_error": {s: aggregate([r for r in selected if r["formation_error_stratum"] == s])
                                                       for s in ("le200", "200to300", "gt300")},
                       "by_initial_heading_error": {s: aggregate([r for r in selected if r["heading_error_stratum"] == s])
                                                     for s in ("le90", "gt90")}}
    source_records[source_label(campaign / "episode_arm_summary.csv")] = sha(campaign / "episode_arm_summary.csv")
    source_records[source_label(campaign / "control/campaign_contract.json")] = sha(campaign / "control/campaign_contract.json")
    return rows, {"scenario_n": 100, "arm_n": len(expected), "row_n": len(rows),
                  "seed_min": min(seeds), "seed_max": max(seeds),
                  "all_rows_included": True, "published_totals_reproduced": True,
                  "all_initial_geometry_and_noise_pairs_equal": True,
                  "all_initial_reconstructions_matched_raw_positions_and_leaders": True,
                  "groups": groups}


def main():
    rng_path = ROOT / "code/uuv_v11_rng.py"
    spec = importlib.util.spec_from_file_location("frozen_rng_for_initial_audit", rng_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg_path = ROOT / "code/experiments_v18_1_guard_ablation_dev_3seed/evaluations/dev100_seed_28001_range_45000_45099/metadata.json"
    cfg = load_json(cfg_path)["environment_config"]
    source_records = {source_label(p): sha(p) for p in (rng_path, cfg_path, Path(__file__).resolve())}
    configurations = [
        ("nominal_link_acquisition", ARCHIVE / "data/raw/leader_source_ablation", NOMINAL_ARMS),
        ("delayed_controller_comparison", CONTROLLER, CONTROLLER_ARMS),
    ]
    all_rows, campaigns = [], {}
    for name, path, expected in configurations:
        rows, summary = process_campaign(name, path, cfg, module.EpisodeSeedPlan, expected, source_records)
        all_rows.extend(rows)
        campaigns[name] = summary
        print(name, "verified", len(rows), "episode-arms")
    public_path = ROOT / "data/tables/controller_repair_episode_rows.csv"
    public = load_csv(public_path)
    raw = load_csv(configurations[1][1] / "episode_arm_summary.csv")
    assert public == raw, "Public controller table differs from raw campaign table"
    source_records[source_label(public_path)] = sha(public_path)
    report = {
        "analysis_date": "2026-11-02",
        "analysis_type": "descriptive_post_hoc_stratification_of_existing_frozen_campaigns",
        "new_simulation_runs": 0,
        "strata_specified_before_loading_outcomes_in_this_reanalysis": True,
        "formation_error_strata_m": {"le200": "e_f(0)<=200", "200to300": "200<e_f(0)<=300", "gt300": "e_f(0)>300"},
        "initial_heading_error": "absolute wrapped difference from leader-centroid velocity heading; not line-of-sight heading",
        "initial_heading_reconstruction": "Exact frozen scenario RNG reset draws; initial position, both leader positions and velocities verified in every raw episode",
        "metric_definitions": {"terminal": "joint success at t=440s", "joint_success": "finite e_p<7m and e_f<8m",
                               "Tail80": "joint success at >=40 of last50 decisions", "Dwell15": "joint success at all last15 decisions",
                               "saturation": "per-episode fraction of TRACK commands with any abs(normalized channel)>=0.98; mean across episodes with TRACK",
                               "first_TRACK": "start time of first executed TRACK command; conditional on TRACK occurrence",
                               "percentile_method": "numpy.percentile default linear; episodes are analysis units"},
        "limits": ["Not a new prespecified qualification or hypothesis test.",
                   "Original prior support 120-350m is correctly specified for every episode.",
                   "Initial centroid range is different from error to formation reference 120m behind centroid.",
                   "No extrapolation outside the observed initial states or to untested physical vehicle dynamics.",
                   "Across-bin differences are descriptive, not randomized effects of initial error.",
                   "The nominal and delayed-controller campaigns use different scenarios and are not paired with each other."],
        "campaigns": campaigns,
        "public_controller_table_exactly_matches_raw": True,
        "source_sha256": source_records,
    }
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "initial_condition_episode_rows.json").write_text(json.dumps(all_rows, indent=2, allow_nan=False) + "\n")
    (OUT / "initial_condition_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, result in campaigns.items():
        for arm, group in result["groups"].items():
            print(name, arm, {s: (v["n"], v["terminal_success_count"], v["tail80_success_count"])
                              for s, v in group["by_initial_formation_error"].items()})


if __name__ == "__main__":
    main()
