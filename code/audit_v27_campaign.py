#!/usr/bin/env python3
"""Independent artifact and truth-boundary audit for V27."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27


AUDITOR_VERSION = "v27_campaign_auditor_1.0"
FORBIDDEN_UNSCORED_KEY_TOKENS = ("truth", "error", "success")


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _forbidden_payload_keys(value: Any, prefix: str = "") -> Sequence[str]:
    issues = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_UNSCORED_KEY_TOKENS):
                issues.append(label)
            issues.extend(_forbidden_payload_keys(item, label))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_forbidden_payload_keys(item, f"{prefix}[{index}]"))
    return issues


def _close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) is bool(right)
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _manual_score(
    payload: Mapping[str, Any],
    truth: v19.ReplayTruthDiagnostics,
    prefix_s: float,
) -> Dict[str, Any]:
    endpoint_index = int(round(float(prefix_s))) - 1
    current_truth = (
        truth.initial_follower_position_m + truth.true_displacement_m[endpoint_index]
    )
    current = np.asarray(payload["current_position_m"], dtype=np.float64)
    endpoint_error = float(np.linalg.norm(current - current_truth))
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
        "success_lt_7m": bool(endpoint_error < 7.0),
        "success_le_7m": bool(endpoint_error <= 7.0),
        "nominal_radius95_covers": (
            None if radius is None else bool(endpoint_error <= float(radius))
        ),
    }


def audit_campaign(campaign_directory: Path) -> Dict[str, Any]:
    campaign = Path(campaign_directory).expanduser().resolve()
    issues = []
    contract_path = campaign / "control" / "campaign_contract.json"
    summary_path = campaign / "campaign_summary.json"
    if not contract_path.is_file() or not summary_path.is_file():
        return {
            "auditor_version": AUDITOR_VERSION,
            "valid": False,
            "issues": ["campaign contract or summary is missing"],
        }
    contract = _read_json(contract_path)
    summary = _read_json(summary_path)
    stored_contract_hash = str(contract.get("contract_sha256", ""))
    unhashed = dict(contract)
    unhashed.pop("contract_sha256", None)
    if _json_hash(unhashed) != stored_contract_hash:
        issues.append("campaign contract hash is invalid")
    if summary.get("contract_sha256") != stored_contract_hash:
        issues.append("summary contract hash mismatch")
    if not bool(contract.get("development_only", False)) or not bool(
        contract.get("no_rl_training", False)
    ):
        issues.append("contract does not preserve the development/no-RL boundary")
    seeds = [int(value) for value in contract.get("episode_seeds", [])]
    if any(v27.RESERVED_SEED_START <= seed <= v27.FINAL_SEED_END for seed in seeds):
        issues.append("contract intersects reserved/final seeds")
    if contract.get("reserved_seed_range_untouched") != [
        v27.RESERVED_SEED_START,
        v27.RESERVED_SEED_END,
    ]:
        issues.append("reserved range declaration changed")
    if contract.get("sealed_final_seed_range_untouched") != [
        v27.FINAL_SEED_START,
        v27.FINAL_SEED_END,
    ]:
        issues.append("final range declaration changed")

    manifest = contract.get("source_manifest", {})
    if not isinstance(manifest, Mapping):
        issues.append("source manifest is missing")
        manifest = {}
    for relative, expected_hash in manifest.items():
        snapshot = campaign / "control" / "source_snapshot" / str(relative)
        if not snapshot.is_file():
            issues.append(f"source snapshot missing: {relative}")
        elif v27.sha256_file(snapshot) != str(expected_hash):
            issues.append(f"source snapshot hash mismatch: {relative}")

    source_rows = {
        int(row["episode_index"]): row
        for row in contract.get("source_episodes", [])
        if isinstance(row, Mapping) and "episode_index" in row
    }
    episode_indices = [int(value) for value in contract.get("episode_indices", [])]
    prefixes = [float(value) for value in contract.get("prefixes_s", [])]
    arms = [str(value) for value in contract.get("arms", [])]
    if arms != list(v27.ARM_NAMES):
        issues.append("arm order or membership changed")
    expected_cells = len(episode_indices) * len(prefixes) * len(arms)
    unscored_files = sorted((campaign / "unscored").glob("*.json"))
    scored_files = sorted((campaign / "episode_results").glob("*.json"))
    if len(unscored_files) != expected_cells:
        issues.append(
            f"unscored cell count mismatch: expected {expected_cells}, found {len(unscored_files)}"
        )
    if len(scored_files) != expected_cells:
        issues.append(
            f"scored cell count mismatch: expected {expected_cells}, found {len(scored_files)}"
        )

    checked_cells = 0
    success_counts: Dict[str, int] = {}
    for episode_index in episode_indices:
        source = source_rows.get(episode_index)
        if source is None:
            issues.append(f"source contract row missing for episode {episode_index}")
            continue
        seed = int(source["episode_seed"])
        if seed != v19.V181_DEV_SEED_START + episode_index:
            issues.append(f"episode seed mismatch for episode {episode_index}")
        try:
            v27.assert_v27_development_seed(seed)
        except PermissionError as exc:
            issues.append(str(exc))
        online_path = Path(str(source["online_path"]))
        truth_path = Path(str(source["truth_path"]))
        if not online_path.is_file() or v27.sha256_file(online_path) != source["online_sha256"]:
            issues.append(f"online source changed for episode {episode_index}")
            continue
        if not truth_path.is_file() or v27.sha256_file(truth_path) != source["truth_sha256"]:
            issues.append(f"truth source changed for episode {episode_index}")
            continue
        truth = v19.load_truth_diagnostics(truth_path)
        for prefix in prefixes:
            for arm in arms:
                stem = f"episode_{episode_index:04d}_prefix_{int(round(prefix)):04d}_{arm}.json"
                unscored_path = campaign / "unscored" / stem
                scored_path = campaign / "episode_results" / stem
                if not unscored_path.is_file() or not scored_path.is_file():
                    continue
                unscored = _read_json(unscored_path)
                scored = _read_json(scored_path)
                payload = unscored.get("payload")
                if not isinstance(payload, Mapping):
                    issues.append(f"unscored payload missing: {stem}")
                    continue
                forbidden = _forbidden_payload_keys(payload)
                if forbidden:
                    issues.append(f"truth/scoring leakage in {stem}: {forbidden[:3]}")
                identity = {
                    "contract_sha256": stored_contract_hash,
                    "episode_index": episode_index,
                    "prefix_s": prefix,
                    "arm": arm,
                    "online_inputs_sha256": source["online_sha256"],
                }
                for key, expected in identity.items():
                    if unscored.get(key) != expected:
                        issues.append(f"unscored identity mismatch {key}: {stem}")
                    if key in scored and scored.get(key) != expected:
                        issues.append(f"scored identity mismatch {key}: {stem}")
                if scored.get("truth_labels_sha256") != source["truth_sha256"]:
                    issues.append(f"truth hash mismatch: {stem}")
                if scored.get("unscored_sha256") != v27.sha256_file(unscored_path):
                    issues.append(f"unscored artifact hash mismatch: {stem}")
                if scored.get("unscored") != payload:
                    issues.append(f"scored payload differs from persisted estimator output: {stem}")
                expected_score = _manual_score(payload, truth, prefix)
                actual_score = scored.get("score")
                if not isinstance(actual_score, Mapping):
                    issues.append(f"score missing: {stem}")
                    continue
                for key, expected in expected_score.items():
                    if not _close(actual_score.get(key), expected):
                        issues.append(f"score mismatch {key}: {stem}")
                key = f"{arm}@{int(round(prefix))}"
                success_counts[key] = success_counts.get(key, 0) + int(
                    bool(expected_score["success_le_7m"])
                )
                checked_cells += 1

    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        issues.append("campaign summary lacks aggregate")
    else:
        if int(aggregate.get("cell_count", -1)) != expected_cells:
            issues.append("summary aggregate cell count mismatch")
        by_arm_prefix = aggregate.get("by_arm_prefix")
        if not isinstance(by_arm_prefix, Mapping):
            issues.append("summary lacks by-arm-prefix table")
        else:
            for key, count in success_counts.items():
                row = by_arm_prefix.get(key)
                if not isinstance(row, Mapping) or int(row.get("success_le_7m_count", -1)) != int(count):
                    issues.append(f"summary success count mismatch: {key}")

    forbidden_seed_pattern = re.compile(r"seed_(49(?:9\d\d)|50\d\d\d)(?:\D|$)")
    forbidden_paths = [
        str(path.relative_to(campaign))
        for path in campaign.rglob("*")
        if forbidden_seed_pattern.search(path.name)
    ]
    if forbidden_paths:
        issues.append(f"reserved/final seed artifacts found: {forbidden_paths[:3]}")

    return {
        "auditor_version": AUDITOR_VERSION,
        "valid": not issues and checked_cells == expected_cells,
        "issues": issues,
        "checked_cells": int(checked_cells),
        "expected_cells": int(expected_cells),
        "truth_boundary_checked": True,
        "reserved_and_final_ranges_untouched": not forbidden_paths,
        "contract_sha256": stored_contract_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a V27 campaign")
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args(argv)
    result = audit_campaign(args.campaign)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
