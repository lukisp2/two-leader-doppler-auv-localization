#!/usr/bin/env python3
"""Run the frozen V38 campaign on the authorized fresh replication block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import run_v20_positioning_ablation as runner20
import run_v38_leader_source_ablation as frozen
import uuv_v19_observability as v19
import uuv_v22_active_acquisition as v22
import uuv_v24_audited_gate as v24
import uuv_v38_leader_source_ablation as v38


RUNNER_VERSION = "v38_runtime_replication_runner_1.0"
AMENDMENT_NAME = "EXPERIMENT_PROTOCOL_V38_RUNTIME_REPLICATION_AMENDMENT.md"
TEST_NAME = "tests/test_run_v38_leader_source_ablation_replication.py"
REQUIRED_SEED_START = 48_600
CAMPAIGN_EPISODES = 100
RUNTIME_THRESHOLD_S = 2.0
REFERENCE_CAMPAIGN_NAME = "experiments_v38_leader_source_ablation_dev100"
REFERENCE_CAMPAIGN_CONTRACT_SHA256 = (
    "48d906d7ad2c972e371ee86d866ccc601de20900d56d0ec80942c1a0b293db01"
)
REFERENCE_DECISION_SHA256 = (
    "1beb68bbf00896f440a0a528c5dd9fde32e8f752b60e47b02e65acbcd4a2331c"
)
REPLICATION_REASON = (
    "Exact unchanged replication opened only because the reference campaign "
    "had one 2.9143316249828786 s runtime-gate violation; five exact "
    "diagnostic repeats passed the unchanged 2.0 s threshold. No efficacy "
    "tuning was performed."
)


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--preflight-seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument(
        "--reference-contract",
        type=Path,
        help=(
            "Optional reference campaign_contract.json. When supplied or "
            "present at the legacy local path, its SHA256 must match the "
            "frozen published digest."
        ),
    )
    parser.add_argument(
        "--reference-decision",
        type=Path,
        help=(
            "Optional reference decision.json. When supplied or present at "
            "the legacy local path, its SHA256 must match the frozen digest."
        ),
    )
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    seed_start = int(args.seed_start)
    if seed_start != REQUIRED_SEED_START:
        raise ValueError(
            f"this amendment authorizes --seed-start {REQUIRED_SEED_START}"
        )
    campaign_seeds = list(
        range(seed_start, seed_start + CAMPAIGN_EPISODES)
    )
    for seed in campaign_seeds:
        v38.assert_seed_allowed(seed)
    if any(
        v38.RESERVED_START <= seed <= v38.FINAL_END
        for seed in campaign_seeds
    ):
        raise PermissionError("replication block overlaps the closed final range")

    preflight_seed = args.preflight_seed
    if preflight_seed is None:
        seeds = campaign_seeds
        preflight = False
    else:
        value = int(preflight_seed)
        v38.assert_seed_allowed(value)
        if value in campaign_seeds:
            raise ValueError("preflight seed must be outside the campaign block")
        seeds = [value]
        preflight = True

    output = Path(args.output_dir).expanduser().resolve()
    metadata = runner20._default_metadata(root)
    if not Path(metadata).is_file():
        raise FileNotFoundError(metadata)
    default_reference = root / REFERENCE_CAMPAIGN_NAME
    reference_contract = (
        Path(args.reference_contract).expanduser().resolve()
        if args.reference_contract is not None
        else default_reference / "control" / "campaign_contract.json"
    )
    reference_decision = (
        Path(args.reference_decision).expanduser().resolve()
        if args.reference_decision is not None
        else default_reference / "decision.json"
    )
    return {
        "root": root,
        "output": output,
        "metadata": metadata,
        "smoke": preflight,
        "preflight": preflight,
        "resume": bool(args.resume),
        "episode_start": 0,
        "episodes": len(seeds),
        "seeds": seeds,
        "campaign_seed_start": seed_start,
        "campaign_seed_end": campaign_seeds[-1],
        "campaign_seed_count": len(campaign_seeds),
        "campaign_seeds": campaign_seeds,
        "coarse_candidates": 4096,
        "coarse_sweeps": 2,
        "local_starts": 48,
        "publication_settings": True,
        "progress_every": max(1, int(args.progress_every)),
        "reference_contract": reference_contract,
        "reference_decision": reference_decision,
        "reference_contract_explicit": args.reference_contract is not None,
        "reference_decision_explicit": args.reference_decision is not None,
    }


def _reference_artifact_sha256(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
    explicitly_provided: bool,
) -> tuple[str, str]:
    """Verify an available reference artifact or use its frozen digest.

    The public release intentionally omits the private reference campaign.
    Its two provenance hashes are frozen in this runner.  A caller may still
    provide either artifact; an available artifact is always hashed and must
    match, while an explicitly requested but missing file fails closed.
    """

    if path.is_file():
        observed = runner20._sha256(path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"{label} SHA256 mismatch: {observed} != {expected_sha256}"
            )
        return observed, "verified_file"
    if explicitly_provided:
        raise FileNotFoundError(path)
    return expected_sha256, "frozen_expected_sha256_fallback"


def _seed_references(path: Path, selected: set[int]) -> list[int]:
    found: set[int] = set()
    for match in re.finditer(r"seed_(\d+)", path.name):
        value = int(match.group(1))
        if value in selected:
            found.add(value)
    if path.suffix == ".json":
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return sorted(found)
        for match in re.finditer(r'"episode_seed"\s*:\s*(\d+)', text):
            value = int(match.group(1))
            if value in selected:
                found.add(value)
    elif path.suffix == ".csv":
        try:
            with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                if "episode_seed" in (reader.fieldnames or ()):
                    for row in reader:
                        try:
                            value = int(row["episode_seed"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if value in selected:
                            found.add(value)
        except OSError:
            pass
    return sorted(found)


def _fresh_seed_audit(
    root: Path,
    output: Path,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    selected = {int(seed) for seed in seeds}
    ordered = sorted(selected)
    findings: list[Dict[str, Any]] = []
    for directory in sorted(root.glob("experiments_*")):
        if not directory.is_dir():
            continue
        try:
            if directory.resolve() == output.resolve():
                continue
        except OSError:
            continue
        for path in directory.rglob("*"):
            if (
                not path.is_file()
                or "source_snapshot" in path.parts
                or path.suffix not in {".json", ".csv", ".npz"}
            ):
                continue
            references = _seed_references(path, selected)
            if references:
                findings.append(
                    {
                        "path": str(path.relative_to(root)),
                        "seeds": references,
                    }
                )
    consecutive = bool(
        ordered
        and ordered == list(range(ordered[0], ordered[0] + len(ordered)))
    )
    return {
        "range": [ordered[0], ordered[-1]],
        "count": len(ordered),
        "consecutive": consecutive,
        "fresh": not findings,
        "finding_count": len(findings),
        "findings": findings[:100],
    }


def _loaded_local_sources(root: Path) -> Dict[str, str]:
    manifest = frozen._loaded_local_sources(root)
    for name in (
        Path(__file__).name,
        AMENDMENT_NAME,
        TEST_NAME,
    ):
        path = root / name
        if not path.is_file() and name.startswith("tests/"):
            path = root.parent / name
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest[name] = runner20._sha256(path)
    return dict(sorted(manifest.items()))


def _contract(
    settings: Mapping[str, Any],
    cfg: Any,
    estimator_config: v19.BatchEstimatorConfig,
    planner_config: v22.ActivePlannerConfig,
    lock_config: v24.AuditedLockConfig,
    actual_freshness: Mapping[str, Any],
    campaign_freshness: Mapping[str, Any],
) -> Dict[str, Any]:
    contract = frozen._contract(
        settings,
        cfg,
        estimator_config,
        planner_config,
        lock_config,
        actual_freshness,
    )
    manifest = _loaded_local_sources(Path(settings["root"]))
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    reference_contract_sha, reference_contract_source = (
        _reference_artifact_sha256(
            Path(settings["reference_contract"]),
            REFERENCE_CAMPAIGN_CONTRACT_SHA256,
            label="reference campaign contract",
            explicitly_provided=bool(
                settings["reference_contract_explicit"]
            ),
        )
    )
    reference_decision_sha, reference_decision_source = (
        _reference_artifact_sha256(
            Path(settings["reference_decision"]),
            REFERENCE_DECISION_SHA256,
            label="reference decision",
            explicitly_provided=bool(
                settings["reference_decision_explicit"]
            ),
        )
    )
    contract.update(
        {
            "runner_version": RUNNER_VERSION,
            "purpose": (
                "runtime-only exact replication of the frozen paired 3x2 "
                "Doppler-source/acquisition-policy campaign"
            ),
            "replication_amendment": {
                "protocol": AMENDMENT_NAME,
                "reason": REPLICATION_REASON,
                "runner_only_change": True,
                "scientific_method_changed": False,
                "efficacy_tuning_performed": False,
                "runtime_definition_changed": False,
                "runtime_threshold_changed": False,
                "strict_runtime_threshold_s": RUNTIME_THRESHOLD_S,
                "reference_campaign": REFERENCE_CAMPAIGN_NAME,
                "reference_campaign_contract_sha256": reference_contract_sha,
                "reference_campaign_contract_sha256_source": (
                    reference_contract_source
                ),
                "reference_decision_sha256": reference_decision_sha,
                "reference_decision_sha256_source": reference_decision_source,
                "authorized_campaign_seed_range": [
                    int(settings["campaign_seed_start"]),
                    int(settings["campaign_seed_end"]),
                ],
                "authorized_campaign_seed_count": int(
                    settings["campaign_seed_count"]
                ),
                "preflight": bool(settings["preflight"]),
                "preflight_seed": (
                    int(settings["seeds"][0])
                    if settings["preflight"]
                    else None
                ),
            },
            "campaign_fresh_seed_audit": dict(campaign_freshness),
            "reserved_range_audit": {
                "closed_range": [v38.RESERVED_START, v38.FINAL_END],
                "selected_range": [
                    int(settings["campaign_seed_start"]),
                    int(settings["campaign_seed_end"]),
                ],
                "overlap": False,
            },
            "source_sha256": manifest,
            "source_manifest_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        }
    )
    return contract


def _immutable(contract: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(contract)
    value.pop("created_at_utc", None)
    return value


def _prepare(
    output: Path,
    root: Path,
    contract: Mapping[str, Any],
    resume: bool,
) -> None:
    contract_path = output / "control" / "campaign_contract.json"
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if _immutable(previous) != _immutable(contract):
            raise RuntimeError("replication resume contract differs")
        return
    runner20._write_json_atomic(contract_path, contract)
    snapshot = output / "control" / "source_snapshot"
    for name, expected_sha in contract["source_sha256"].items():
        source = root / name
        if not source.is_file() and name.startswith("tests/"):
            source = root.parent / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if runner20._sha256(source) != expected_sha:
            raise RuntimeError(f"source changed before snapshot: {name}")
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def main(argv: Optional[Sequence[str]] = None) -> int:
    settings = _settings(_parse(argv))
    root = Path(settings["root"])
    output = Path(settings["output"])
    actual_freshness = _fresh_seed_audit(root, output, settings["seeds"])
    campaign_freshness = _fresh_seed_audit(
        root,
        output,
        settings["campaign_seeds"],
    )
    if not actual_freshness["fresh"]:
        raise RuntimeError(
            "selected execution seeds are not fresh: "
            f"{actual_freshness['findings'][:10]}"
        )
    if not campaign_freshness["fresh"]:
        raise RuntimeError(
            "authorized campaign block is not fresh: "
            f"{campaign_freshness['findings'][:10]}"
        )

    cfg = runner20._load_environment_config(Path(settings["metadata"]))
    estimator_config = v19.BatchEstimatorConfig(
        coarse_candidates=4096,
        coarse_sweeps=2,
        local_starts=48,
        gate_mode="raw",
        candidate_radial_distribution="uniform_radius",
    )
    planner_config = v22.ActivePlannerConfig()
    lock_config = v24.AuditedLockConfig()
    contract = _contract(
        settings,
        cfg,
        estimator_config,
        planner_config,
        lock_config,
        actual_freshness,
        campaign_freshness,
    )
    _prepare(output, root, contract, bool(settings["resume"]))

    expected_sources = dict(contract["source_sha256"])
    rows: list[Dict[str, Any]] = []
    pairing_records: list[Dict[str, Any]] = []
    started = time.perf_counter()
    completed = 0
    total = int(contract["expected_runs"])
    try:
        for local_index, seed in enumerate(settings["seeds"]):
            episode_index = int(settings["episode_start"]) + local_index
            tape = runner20._tape_for_episode(
                output,
                cfg,
                int(seed),
                episode_index,
            )
            outcomes: Dict[Tuple[str, str], v38.ArmOutcome] = {}
            for source, policy in v38.arm_pairs():
                result_path, trace_path = frozen._paths(
                    output,
                    episode_index,
                    int(seed),
                    source,
                    policy,
                )
                if (
                    settings["resume"]
                    and result_path.is_file()
                    and trace_path.is_file()
                ):
                    summary = json.loads(
                        result_path.read_text(encoding="utf-8")
                    )
                    with np.load(trace_path, allow_pickle=False) as archive:
                        trace = {
                            key: archive[key].copy() for key in archive.files
                        }
                    outcome = v38.ArmOutcome(summary=summary, trace=trace)
                else:
                    outcome = v38.run_arm(
                        cfg=cfg,
                        tape=tape,
                        episode_seed=int(seed),
                        episode_index=episode_index,
                        source_name=source,
                        policy_name=policy,
                        estimator_config=estimator_config,
                        lock_config=lock_config,
                        planner_config=planner_config,
                    )
                    runner20._write_json_atomic(result_path, outcome.summary)
                    runner20._write_npz_atomic(trace_path, outcome.trace)
                if outcome.summary["noise_tape_sha256"] != tape.content_sha256():
                    raise RuntimeError("replication saved the wrong tape hash")
                if int(outcome.summary["action_count"]) != int(cfg.max_steps):
                    raise RuntimeError("replication saved an incomplete trace")
                outcomes[(source, policy)] = outcome
                rows.append(frozen._flatten(outcome.summary))
                completed += 1
                if _loaded_local_sources(root) != expected_sources:
                    raise RuntimeError(
                        "replication source closure changed during execution"
                    )
                elapsed = float(time.perf_counter() - started)
                eta = elapsed / max(completed, 1) * max(total - completed, 0)
                runner20._write_json_atomic(
                    output / "control" / "progress.json",
                    {
                        "status": "running",
                        "runner_version": RUNNER_VERSION,
                        "updated_at_utc": runner20._utc_now(),
                        "completed_runs": completed,
                        "total_runs": total,
                        "elapsed_wall_s": elapsed,
                        "estimated_remaining_s": eta,
                        "last_episode_index": episode_index,
                        "last_seed": int(seed),
                        "last_source": source,
                        "last_policy": policy,
                        "resumable": True,
                    },
                )
                if completed % int(settings["progress_every"]) == 0:
                    print(
                        f"[{runner20._utc_now()}] V38-R {completed}/{total}; "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s; "
                        f"seed={seed} source={source} policy={policy}",
                        flush=True,
                    )
            pairing_records.append(frozen._verify_seed_pairing(outcomes))

        elapsed = float(time.perf_counter() - started)
        runner20._write_csv_atomic(output / "episode_arm_summary.csv", rows)
        summary = frozen._aggregate(
            rows,
            contract,
            elapsed,
            pairing_records,
        )
        summary["runner_version"] = RUNNER_VERSION
        summary["replication_amendment"] = contract[
            "replication_amendment"
        ]
        runner20._write_json_atomic(output / "campaign_summary.json", summary)
        runner20._write_json_atomic(
            output / "decision.json",
            {
                "decision": summary["decision"],
                "claim_decision": summary["claim_decision"],
                "integrity_valid": summary["integrity_valid"],
                "development_only": True,
                "runner_version": RUNNER_VERSION,
                "replication_reason": REPLICATION_REASON,
                "reserved_final_range_untouched": summary[
                    "reserved_final_range_untouched"
                ],
            },
        )
        runner20._write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "complete",
                "runner_version": RUNNER_VERSION,
                "updated_at_utc": runner20._utc_now(),
                "completed_runs": completed,
                "total_runs": total,
                "elapsed_wall_s": elapsed,
                "estimated_remaining_s": 0.0,
                "resumable": True,
            },
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        elapsed = float(time.perf_counter() - started)
        runner20._write_json_atomic(
            output / "control" / "progress.json",
            {
                "status": "failed",
                "runner_version": RUNNER_VERSION,
                "updated_at_utc": runner20._utc_now(),
                "completed_runs": completed,
                "total_runs": total,
                "elapsed_wall_s": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "resumable": True,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
