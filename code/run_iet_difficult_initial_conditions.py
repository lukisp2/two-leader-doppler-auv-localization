#!/usr/bin/env python3
"""Frozen reviewer-requested difficult-start test; historical methods are unchanged."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
WORKSPACE = ROOT
sys.path.insert(0, str(CODE))
import run_v41_controller_repair as inherited
from tqdm.auto import tqdm

PROTOCOL = ROOT / "protocols" / "EXPERIMENT_PROTOCOL_IET_DIFFICULT_STARTS.txt"
PUBLISHED_SELECTION = ROOT / "data/provenance/iet_difficult_initial_conditions/selected_scenarios.json"
VERSION = "iet_difficult_starts_1.0"
POOL = tuple(range(52000, 53000))
COUNT = 50
SMOKE_SEEDS = (49566,)
ARMS = tuple(inherited.v41.ArmSpec(c, inherited.v40.CURRENT_NONE)
             for c in (inherited.v41.BASELINE_PID, inherited.v41.DELAY_AWARE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    inherited.runner20._write_json_atomic(Path(path), value)


def load_config():
    path = inherited.runner20._default_metadata(CODE)
    return path, inherited.runner20._load_environment_config(path)


def initial_record(env, seed):
    """Truth is used here for prospective scenario selection, never online."""
    env.reset(seed=int(seed))
    center, desired = env._formation_desired()
    velocity = 0.5 * (env.vL1 + env.vL2)
    course = float(np.degrees(np.arctan2(velocity[1], velocity[0])))
    yaw_error = abs((float(env.yaw_F) - course + 180.0) % 360.0 - 180.0)
    formation_error = float(np.linalg.norm(env.pF - desired))
    return {
        "seed": int(seed), "initial_position_m": env.pF.tolist(),
        "leader_positions_m": [env.pL1.tolist(), env.pL2.tolist()],
        "leader_velocities_mps": [env.vL1.tolist(), env.vL2.tolist()],
        "desired_position_m": desired.tolist(),
        "initial_distance_to_centroid_m": float(np.linalg.norm(env.pF-center)),
        "initial_formation_error_m": formation_error,
        "initial_heading_error_deg": float(yaw_error),
        "initial_yaw_deg": float(env.yaw_F),
        "initial_pitch_deg": float(env.pitch_F),
        "initial_speed_mps": float(env.speed_F),
        "eligible": bool(formation_error >= 350.0 and yaw_error >= 120.0),
    }


def screen(cfg):
    env = inherited.v40.DynamicCurrentEnv(
        cfg=cfg, render_mode="none", dynamics=inherited.v40.DYNAMICS_LOW_ORDER,
        current=inherited.v40.CURRENT_NONE, episode_seed=POOL[0])
    try:
        records = [initial_record(env, seed) for seed in POOL]
    finally:
        env.close()
    selected = [row["seed"] for row in records if row["eligible"]][:COUNT]
    if len(selected) != COUNT:
        raise RuntimeError(f"Only {len(selected)} eligible seeds; protocol requires {COUNT}; no simulation permitted")
    return records, selected


def assert_allowed(seed, selected, smoke=False):
    value = int(seed)
    if 50000 <= value <= 50999:
        raise PermissionError("sealed final range is prohibited")
    if smoke:
        if value not in SMOKE_SEEDS:
            raise PermissionError("smoke permits only the previously opened seed 49566")
    elif value not in set(selected) or value not in POOL:
        raise PermissionError("seed is absent from the frozen prospective selection")


@contextmanager
def seed_scope(selected, smoke=False):
    old = inherited.v41._assert_seed_allowed_automatic
    inherited.v41._assert_seed_allowed_automatic = lambda seed: assert_allowed(seed, selected, smoke)
    try:
        yield
    finally:
        inherited.v41._assert_seed_allowed_automatic = old


def local_sources():
    paths = {"revision_runner.py": Path(__file__).resolve(),
             "protocol.txt": PROTOCOL}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.parent == CODE.resolve() and path.suffix == ".py":
                paths[path.name] = path
    metadata, _ = load_config()
    paths["environment_metadata.json"] = metadata
    return paths


def source_hashes():
    return {name: sha(path) for name, path in sorted(local_sources().items())}


def _json_seeds(value, selected):
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"seed", "episode_seed"}:
                try:
                    if int(child) in selected:
                        found.add(int(child))
                except (TypeError, ValueError):
                    pass
            if key in {"seeds", "episode_seeds", "qualification_seeds"} and isinstance(child, list):
                for item in child:
                    if isinstance(item, int) and item in selected:
                        found.add(item)
            found.update(_json_seeds(child, selected))
    elif isinstance(value, list):
        for child in value:
            found.update(_json_seeds(child, selected))
    return found


def freshness_audit(selected):
    """Bounded inventory of actual campaign names, contracts and summary rows.

    No NPZ data hydration is necessary: these archives encode seeds in filenames
    and expose complete episode summaries. Scope is recorded, not universalized.
    """
    roots = [p for base in (CODE, ROOT) if base.exists()
             for p in sorted(base.glob("experiments_*")) if p.is_dir()]
    roots += [ROOT / "data", ROOT / "results"]
    roots = [p for p in roots if p.exists()]
    selected = set(map(int, selected))
    token = re.compile(r"(?:seed[_-]?)(\d{5,})")
    findings, unreadable, inspected = [], [], 0
    for root in roots:
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in {"source_snapshot", "__pycache__", ".git", "models", "logs"}]
            for filename in files:
                path = Path(directory) / filename
                matched = {int(s) for s in token.findall(filename)} & selected
                is_summary = path.suffix == ".csv" and ("summary" in filename or "episode" in filename)
                is_contract = path.suffix == ".json" and (
                    "contract" in filename or "manifest" in filename or "progress" in filename)
                if is_summary or is_contract:
                    inspected += 1
                    try:
                        if is_contract:
                            matched |= _json_seeds(json.loads(path.read_text()), selected)
                        else:
                            with path.open(newline="") as handle:
                                reader = csv.DictReader(handle)
                                keys = set(reader.fieldnames or ()) & {"seed", "episode_seed"}
                                for row in reader:
                                    for key in keys:
                                        if row.get(key) not in (None, ""):
                                            val = int(row[key])
                                            if val in selected:
                                                matched.add(val)
                    except (OSError, ValueError, TypeError) as exc:
                        unreadable.append({"path":str(path), "error":str(exc)})
                if matched:
                    findings.append({"path":str(path), "seeds":sorted(matched)})
    return {"scope_roots": list(map(str, roots)), "summary_contract_files_read": inspected,
            "method":"seed-bearing filenames plus contract/manifest/progress JSON and episode CSV rows",
            "findings":findings,"unreadable":unreadable,"fresh_within_checked_scope":not findings and not unreadable}


def freeze_selection(cfg, output, replicate_published=False):
    records, selected = screen(cfg)
    if replicate_published:
        published = json.loads(PUBLISHED_SELECTION.read_text())
        if selected != published["selected_seeds"] or records != published["screened_initial_states"]:
            raise RuntimeError("Initial-state screening differs from the published frozen cohort")
        audit = {"replication_of_published_cohort": True, "fresh_within_checked_scope": False,
                 "original_freshness_audit": published["freshness_audit"]}
    else:
        audit = freshness_audit(selected)
    manifest = {"version":VERSION,"protocol_sha256":sha(PROTOCOL),
                "environment_config":asdict(cfg), "pool":[POOL[0],POOL[-1]],
                "selection_rule":{"formation_error_min_m":350.0,"heading_error_min_deg":120.0,"count":COUNT},
                "screened_initial_states":records,"selected_seeds":selected,
                "selection_uses_outcomes":False,"freshness_audit":audit}
    if not replicate_published and not audit["fresh_within_checked_scope"]:
        write_json(output / "selection_audit_failed.json", manifest)
        raise RuntimeError("Selected-seed reuse or unreadable relevant summaries; see selection_audit_failed.json")
    path = output / "selected_scenarios.json"
    if path.exists():
        old = json.loads(path.read_text())
        if canonical(old) != canonical(manifest):
            raise RuntimeError("Prospective scenario selection differs from its saved manifest")
    else:
        write_json(path, manifest)
    return manifest


def aggregate(rows, pairings, elapsed):
    by_controller = {c:[r for r in rows if r["controller"] == c] for c in inherited.v41.CONTROLLERS}
    metrics = {c:inherited._metrics(v) for c,v in by_controller.items()}
    paired = {}
    baseline = {r["episode_seed"]:r for r in by_controller[inherited.v41.BASELINE_PID]}
    treatment = {r["episode_seed"]:r for r in by_controller[inherited.v41.DELAY_AWARE]}
    for endpoint in ("terminal_joint_success","tail80_joint_success","dwell15_joint_success"):
        keys = sorted(set(baseline)&set(treatment))
        differences = [int(treatment[s][endpoint])-int(baseline[s][endpoint]) for s in keys]
        paired[endpoint] = {"pairs":len(keys),"treatment_only":differences.count(1),
                            "reference_only":differences.count(-1),
                            "risk_difference":float(np.mean(differences)) if differences else None}
    return {"version":VERSION,"status":"complete","elapsed_wall_s":elapsed,
            "interpretation":"descriptive difficult-start stress; no post-hoc acceptance criterion",
            "method_retuned":False,"sealed_final_range_opened":False,
            "episodes_per_arm":len(baseline),"metrics_by_controller":metrics,
            "paired_binary_differences":paired,"pairing_checks":pairings,
            "integrity_valid":all(p["pretrack_bitwise_equal"] for p in pairings)}


def run(args):
    output = Path(args.output).resolve() if args.output else ROOT / "results" / ("iet_difficult_smoke" if args.smoke else "iet_difficult_replication")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata, cfg = load_config()
    manifest = None
    if not args.smoke:
        manifest = freeze_selection(cfg, output, args.replicate_published)
    if args.prepare_only:
        print(json.dumps({"output":str(output),"selected_count":len(manifest["selected_seeds"]),
                          "first_seed":manifest["selected_seeds"][0],"last_seed":manifest["selected_seeds"][-1]}))
        return 0
    seeds = list(SMOKE_SEEDS) if args.smoke else manifest["selected_seeds"]
    estimator = inherited.v19.BatchEstimatorConfig(coarse_candidates=4096,coarse_sweeps=2,
                    local_starts=48,gate_mode="raw",candidate_radial_distribution="uniform_radius")
    planner = inherited.v22.ActivePlannerConfig()
    gate = inherited.v24.AuditedLockConfig()
    frozen = source_hashes()
    contract = {"version":VERSION,"smoke":args.smoke,"seeds":seeds,
                "expected_runs":len(seeds)*len(ARMS),"environment_config":asdict(cfg),
                "estimator_config":inherited.v19.config_to_dict(estimator),
                "gate_config":asdict(gate),"planner_config":asdict(planner),
                "arms":[a.to_dict() for a in ARMS],"source_sha256":frozen,
                "metadata_sha256":sha(metadata),"python_executable":sys.executable,
                "package_versions":inherited._package_versions(),
                "selection_manifest_sha256":None if args.smoke else sha(output/"selected_scenarios.json")}
    contract_path = output / "control" / "campaign_contract.json"
    if contract_path.exists():
        if not args.resume:
            raise FileExistsError("Campaign already prepared; pass --resume")
        if canonical(json.loads(contract_path.read_text())) != canonical(contract):
            raise RuntimeError("Frozen source/configuration/selection contract changed")
    else:
        write_json(contract_path,contract)
        snapshot = output / "control" / "source_snapshot"
        snapshot.mkdir(parents=True,exist_ok=True)
        for name,path in local_sources().items():
            shutil.copy2(path,snapshot/name)
    total = contract["expected_runs"]
    completed, rows, pairings, durations = 0, [], [], []
    started = time.perf_counter()
    bar = tqdm(total=total,desc="IET difficult starts",unit="run",dynamic_ncols=True,mininterval=1.0)
    inherited._progress(output,status="running",completed=0,total=total,elapsed_s=0,eta_s=None)
    initial_by_seed = {} if args.smoke else {r["seed"]:r for r in manifest["screened_initial_states"]}
    try:
        with seed_scope(seeds,args.smoke):
            for index,seed in enumerate(seeds):
                tape = inherited.runner20._tape_for_episode(output,cfg,seed,index)
                outcomes = {}
                for arm in ARMS:
                    result_path,trace_path = inherited._paths(output,index,seed,arm.name)
                    if result_path.exists() != trace_path.exists():
                        raise RuntimeError("Incomplete saved result/trace pair")
                    bar.set_postfix_str(f"seed={seed} {arm.controller}")
                    if result_path.exists():
                        outcome = inherited._load_outcome(result_path,trace_path)
                    else:
                        t0=time.perf_counter()
                        outcome=inherited.v41.run_controller_arm(arm=arm,cfg=cfg,tape=tape,
                            episode_seed=seed,episode_index=index,estimator_config=estimator,
                            lock_config=gate,planner_config=planner)
                        durations.append(time.perf_counter()-t0)
                        inherited.runner20._write_json_atomic(result_path,outcome.summary)
                        inherited.runner20._write_npz_atomic(trace_path,outcome.trace)
                    inherited._validate_outcome(outcome,arm=arm,tape=tape,cfg=cfg,seed=seed,episode_index=index)
                    if not args.smoke and not np.array_equal(np.asarray(outcome.summary["initial_truth_m"]),
                                    np.asarray(initial_by_seed[seed]["initial_position_m"])):
                        raise RuntimeError("Actual initial position differs from prospective selection")
                    outcomes[arm.controller]=outcome
                    row=inherited._flatten(outcome)
                    if not args.smoke:
                        row.update({k:v for k,v in initial_by_seed[seed].items() if k in
                                   {"initial_formation_error_m","initial_heading_error_deg","initial_distance_to_centroid_m"}})
                    rows.append(row)
                    completed+=1
                    elapsed=time.perf_counter()-started
                    eta=float(np.mean(durations)*(total-completed)) if durations else None
                    inherited._progress(output,status="running",completed=completed,total=total,
                                        elapsed_s=elapsed,eta_s=eta,seed=seed,arm=arm.name)
                    inherited._log(output,"run_completed",completed_runs=completed,total_runs=total,
                                   seed=seed,controller=arm.controller,elapsed_wall_s=elapsed,
                                   estimated_remaining_s=eta)
                    inherited.runner20._write_csv_atomic(output/"episode_arm_summary.csv",rows)
                    if source_hashes()!=frozen:
                        raise RuntimeError("Frozen source changed during campaign")
                    bar.update(1)
                pairings.append(inherited._pretrack_pairing(outcomes[inherited.v41.BASELINE_PID],
                                outcomes[inherited.v41.DELAY_AWARE],inherited.v40.CURRENT_NONE))
                write_json(output/"pairing_checks.json",pairings)
                if args.smoke:
                    reference=inherited.v40.run_factorial_arm(
                        arm=inherited.v40.ArmSpec(inherited.v40.POLICY_ACTIVE,inherited.v40.DYNAMICS_LOW_ORDER,inherited.v40.CURRENT_NONE),
                        cfg=cfg,tape=tape,episode_seed=seed,episode_index=index,estimator_config=estimator,
                        lock_config=gate,planner_config=planner)
                    equivalence=inherited._reference_equivalence(outcomes[inherited.v41.BASELINE_PID],reference)
                    write_json(output/"reference_equivalence.json",equivalence)
                    if not equivalence["bitwise_equal"]:
                        raise RuntimeError("Smoke reference differs from unmodified V40")
        elapsed=time.perf_counter()-started
        summary=aggregate(rows,pairings,elapsed)
        summary["measured_run_durations_s"]=durations
        summary["estimated_100_run_duration_s"]=float(np.mean(durations)*100) if durations else None
        write_json(output/"campaign_summary.json",summary)
        inherited._progress(output,status="complete",completed=completed,total=total,elapsed_s=elapsed,eta_s=0)
        print(f"Complete: {completed}/{total}, elapsed {elapsed:.1f}s; {output}",flush=True)
        return 0
    except BaseException as exc:
        inherited._progress(output,status="failed",completed=completed,total=total,
                            elapsed_s=time.perf_counter()-started,eta_s=None,
                            error=f"{type(exc).__name__}: {exc}",traceback_text=traceback.format_exc())
        raise
    finally:
        bar.close()
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--replicate-published",action="store_true",
                        help="Explicitly reproduce the published 50-scenario cohort; does not claim fresh scenarios")
    args=parser.parse_args()
    if args.smoke and args.prepare_only:
        parser.error("prepare-only selects the prospective campaign, not smoke")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
