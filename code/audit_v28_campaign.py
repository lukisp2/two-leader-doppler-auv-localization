#!/usr/bin/env python3
"""Independent artifact audit for V28 stress/calibration campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27
import uuv_v28_estimator_stress as v28


AUDITOR_VERSION = "v28_campaign_auditor_1.0"
FORBIDDEN_KEYS = ("truth", "error", "success")


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _forbidden(value: Any, prefix: str = "") -> Sequence[str]:
    result = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            if any(token in str(key).lower() for token in FORBIDDEN_KEYS):
                result.append(label)
            result.extend(_forbidden(item, label))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_forbidden(item, f"{prefix}[{index}]"))
    return result


def _manual_score(payload: Mapping[str, Any], truth: v19.ReplayTruthDiagnostics, prefix_s: float) -> Dict[str, Any]:
    index = int(round(float(prefix_s))) - 1
    target = truth.initial_follower_position_m + truth.true_displacement_m[index]
    current = np.asarray(payload["current_position_m"], dtype=np.float64)
    endpoint_error = float(np.linalg.norm(current - target))
    initial = payload.get("initial_position_m")
    initial_error = (
        None
        if initial is None
        else float(
            np.linalg.norm(
                np.asarray(initial, dtype=np.float64)
                - truth.initial_follower_position_m
            )
        )
    )
    radius = payload.get("nominal_radius95_m")
    return {
        "endpoint_s": float(prefix_s),
        "endpoint_position_error_m": endpoint_error,
        "initial_position_error_m": initial_error,
        "success_lt_7m": endpoint_error < 7.0,
        "success_le_7m": endpoint_error <= 7.0,
        "nominal_radius95_covers": (
            None if radius is None else endpoint_error <= float(radius)
        ),
    }


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) is bool(right)
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10)


def audit_campaign(campaign_directory: Path) -> Dict[str, Any]:
    campaign = Path(campaign_directory).expanduser().resolve()
    issues = []
    contract_path = campaign / "control" / "campaign_contract.json"
    summary_path = campaign / "campaign_summary.json"
    if not contract_path.is_file() or not summary_path.is_file():
        return {"auditor_version": AUDITOR_VERSION, "valid": False, "issues": ["missing contract or summary"]}
    contract = _read(contract_path)
    summary = _read(summary_path)
    stored_hash = str(contract.get("contract_sha256", ""))
    unhashed = dict(contract)
    unhashed.pop("contract_sha256", None)
    if _hash_json(unhashed) != stored_hash:
        issues.append("contract hash mismatch")
    if summary.get("contract_sha256") != stored_hash:
        issues.append("summary contract identity mismatch")
    if contract.get("arms") != [v27.GLOBAL_FULL, v27.LOCAL_NLS6, v27.PF_LW_16384]:
        issues.append("arm contract changed")
    if contract.get("stress_contract", {}).get("conditions") != list(v28.CONDITIONS):
        issues.append("stress contract changed")
    if contract.get("reserved_seed_range_untouched") != [49_900, 49_999]:
        issues.append("reserved range declaration changed")
    if contract.get("sealed_final_seed_range_untouched") != [50_000, 50_999]:
        issues.append("final range declaration changed")
    seeds = [int(value) for value in contract.get("episode_seeds", [])]
    if any(49_900 <= value <= 50_999 for value in seeds):
        issues.append("campaign uses a closed seed")

    manifest = contract.get("source_manifest", {})
    for relative, expected in manifest.items() if isinstance(manifest, Mapping) else []:
        path = campaign / "control" / "source_snapshot" / str(relative)
        if not path.is_file() or v27.sha256_file(path) != str(expected):
            issues.append(f"source snapshot mismatch: {relative}")

    episodes = [int(value) for value in contract.get("episode_indices", [])]
    prefixes = [float(value) for value in contract.get("prefixes_s", [])]
    arms = [str(value) for value in contract.get("arms", [])]
    source_rows = {
        int(row["episode_index"]): row
        for row in contract.get("source_episodes", [])
        if isinstance(row, Mapping)
    }
    expected_cells = len(episodes) * len(v28.CONDITIONS) * len(prefixes) * len(arms)
    if len(list((campaign / "unscored").glob("*.json"))) != expected_cells:
        issues.append("unscored file count mismatch")
    if len(list((campaign / "episode_results").glob("*.json"))) != expected_cells:
        issues.append("scored file count mismatch")

    checked = 0
    success_counts: Dict[str, int] = {}
    for episode in episodes:
        source = source_rows.get(episode)
        if source is None:
            issues.append(f"missing source row for episode {episode}")
            continue
        if int(source["episode_seed"]) != 45_000 + episode:
            issues.append(f"source seed mismatch for episode {episode}")
        online_path = Path(str(source["online_path"]))
        truth_path = Path(str(source["truth_path"]))
        if not online_path.is_file() or v27.sha256_file(online_path) != source["online_sha256"]:
            issues.append(f"online source mismatch for episode {episode}")
            continue
        if not truth_path.is_file() or v27.sha256_file(truth_path) != source["truth_sha256"]:
            issues.append(f"truth source mismatch for episode {episode}")
            continue
        online = v19.load_online_history(online_path)
        truth = v19.load_truth_diagnostics(truth_path)
        for condition in v28.CONDITIONS:
            stressed = v28.apply_stress(online, condition, episode)
            stress_hash = v28.history_sha256(stressed)
            for prefix in prefixes:
                for arm in arms:
                    name = (
                        f"episode_{episode:04d}_{condition}_"
                        f"prefix_{int(round(prefix)):04d}_{arm}.json"
                    )
                    upath = campaign / "unscored" / name
                    spath = campaign / "episode_results" / name
                    if not upath.is_file() or not spath.is_file():
                        continue
                    unscored = _read(upath)
                    scored = _read(spath)
                    payload = unscored.get("payload")
                    if not isinstance(payload, Mapping):
                        issues.append(f"missing payload: {name}")
                        continue
                    leaked = _forbidden(payload)
                    if leaked:
                        issues.append(f"truth/scoring leakage in {name}: {leaked[:3]}")
                    identity = {
                        "contract_sha256": stored_hash,
                        "episode_index": episode,
                        "condition": condition,
                        "prefix_s": prefix,
                        "arm": arm,
                        "online_inputs_sha256": source["online_sha256"],
                        "stressed_history_sha256": stress_hash,
                    }
                    for key, expected in identity.items():
                        if unscored.get(key) != expected:
                            issues.append(f"unscored identity mismatch {key}: {name}")
                        if key in scored and scored.get(key) != expected:
                            issues.append(f"scored identity mismatch {key}: {name}")
                    if scored.get("truth_labels_sha256") != source["truth_sha256"]:
                        issues.append(f"truth hash mismatch: {name}")
                    if scored.get("unscored_sha256") != v27.sha256_file(upath):
                        issues.append(f"unscored hash mismatch: {name}")
                    if scored.get("unscored") != payload:
                        issues.append(f"persisted payload mismatch: {name}")
                    expected_score = _manual_score(payload, truth, prefix)
                    actual = scored.get("score", {})
                    for key, expected in expected_score.items():
                        if not _same(actual.get(key), expected):
                            issues.append(f"score mismatch {key}: {name}")
                    summary_key = f"{condition}|{arm}@{int(prefix)}"
                    success_counts[summary_key] = success_counts.get(summary_key, 0) + int(
                        bool(expected_score["success_le_7m"])
                    )
                    checked += 1

    aggregate = summary.get("aggregate", {})
    table = aggregate.get("by_condition_arm_prefix", {}) if isinstance(aggregate, Mapping) else {}
    if int(aggregate.get("cell_count", -1)) != expected_cells:
        issues.append("summary cell count mismatch")
    for key, count in success_counts.items():
        row = table.get(key)
        if not isinstance(row, Mapping) or int(row.get("success_le_7m_count", -1)) != count:
            issues.append(f"summary success mismatch: {key}")

    return {
        "auditor_version": AUDITOR_VERSION,
        "valid": not issues and checked == expected_cells,
        "issues": issues,
        "checked_cells": checked,
        "expected_cells": expected_cells,
        "truth_boundary_checked": True,
        "reserved_and_final_ranges_untouched": not any(49_900 <= value <= 50_999 for value in seeds),
        "contract_sha256": stored_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args(argv)
    result = audit_campaign(args.campaign)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
