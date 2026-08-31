#!/usr/bin/env python3
"""Independent structural audit for a completed V40 factorial campaign."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import uuv_v40_dynamic_plant_stress as v40


TRACE_FIELDS = (
    "plant_requested_action",
    "plant_delivered_action",
    "plant_executed_rate_state",
    "water_current_mps",
    "body_velocity_through_water_mps",
    "ground_velocity_mps",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.expanduser().resolve()
    contract = json.loads(
        (root / "control" / "campaign_contract.json").read_text(
            encoding="utf-8"
        )
    )
    expected = int(contract["expected_runs"])
    horizon = int(contract["fixed_horizon_actions"])
    expected_arms = {arm.name for arm in v40.ARM_SPECS}
    result_files = sorted((root / "episode_results").rglob("*.json"))
    trace_files = sorted((root / "traces_npz").rglob("*.npz"))
    if len(result_files) != expected or len(trace_files) != expected:
        raise RuntimeError("run count differs from the frozen contract")

    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    for path in result_files:
        result = json.loads(path.read_text(encoding="utf-8"))
        seed = int(result["episode_seed"])
        arm = str(result["arm"])
        key = (seed, arm)
        if key in seen:
            raise RuntimeError(f"duplicate result {key}")
        seen.add(key)
        if int(result["action_count"]) != horizon:
            raise RuntimeError(f"short trace {key}")
        if arm not in expected_arms:
            raise RuntimeError(f"unknown arm {arm}")
        if v40.FINAL_START <= seed <= v40.FINAL_END:
            raise RuntimeError(f"final holdout was opened: {seed}")
        by_seed[seed].append(result)

    for seed, results in by_seed.items():
        if {str(result["arm"]) for result in results} != expected_arms:
            raise RuntimeError(f"seed {seed} lacks the complete factorial")
        for field in (
            "noise_tape_sha256",
            "current_tape_sha256",
            "initial_truth_m",
            "mission_support",
        ):
            canonical = json.dumps(results[0][field], sort_keys=True)
            if any(
                json.dumps(result[field], sort_keys=True) != canonical
                for result in results[1:]
            ):
                raise RuntimeError(f"seed {seed} is not paired in {field}")

    for path in trace_files:
        with np.load(path, allow_pickle=False) as archive:
            if archive["time_s"].shape != (horizon,):
                raise RuntimeError(f"bad time vector in {path}")
            for field in TRACE_FIELDS:
                value = archive[field]
                if value.shape != (horizon, 3) or not np.all(np.isfinite(value)):
                    raise RuntimeError(f"bad {field} in {path}")

    summary = json.loads(
        (root / "campaign_summary.json").read_text(encoding="utf-8")
    )
    if not bool(summary["integrity_valid"]):
        raise RuntimeError("campaign summary reports an integrity failure")
    if summary["final_holdout_sealed"] != [v40.FINAL_START, v40.FINAL_END]:
        raise RuntimeError("sealed final range differs from the protocol")
    print(
        json.dumps(
            {
                "audit": "PASS",
                "runs": expected,
                "episodes": len(by_seed),
                "arms": sorted(expected_arms),
                "decision": summary["decision"],
                "final_holdout_sealed": summary["final_holdout_sealed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
