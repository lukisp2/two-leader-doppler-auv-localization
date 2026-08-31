#!/usr/bin/env python3
"""Independent artifact, scoring and causal-accounting audit for V34."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import uuv_v19_observability as v19
import uuv_v34_mhe60_baseline as v34


AUDITOR_VERSION = "v34_mhe60_campaign_auditor_1.0"
FORBIDDEN_UNSCORED_KEY_TOKENS = ("truth", "error", "success")
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 34_045_000


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
    issues: List[str] = []
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
    checkpoint_s: float,
) -> Dict[str, Any]:
    endpoint_index = int(round(float(checkpoint_s))) - 1
    current_truth = (
        truth.initial_follower_position_m + truth.true_displacement_m[endpoint_index]
    )
    current = np.asarray(payload["current_position_m"], dtype=np.float64)
    initial = np.asarray(payload["initial_position_m"], dtype=np.float64)
    endpoint_error = float(np.linalg.norm(current - current_truth))
    initial_error = float(np.linalg.norm(initial - truth.initial_follower_position_m))
    radius = payload.get("nominal_radius95_m")
    return {
        "endpoint_s": float(checkpoint_s),
        "endpoint_position_error_m": endpoint_error,
        "initial_position_error_m": initial_error,
        "success_lt_7m": bool(endpoint_error < 7.0),
        "success_le_7m": bool(endpoint_error <= 7.0),
        "nominal_radius95_covers": (
            None if radius is None else bool(endpoint_error <= float(radius))
        ),
        "false_confidence_radius_lt_7_error_gt_7": bool(
            radius is not None and float(radius) < 7.0 and endpoint_error > 7.0
        ),
        "prefix_sample_count": int(round(float(checkpoint_s))),
    }


def _bootstrap_mean_ci(values: np.ndarray) -> Dict[str, float]:
    differences = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    samples: List[np.ndarray] = []
    remaining = BOOTSTRAP_RESAMPLES
    while remaining > 0:
        chunk = min(5000, remaining)
        indices = rng.integers(
            0, differences.size, size=(chunk, differences.size), endpoint=False
        )
        samples.append(np.mean(differences[indices], axis=1))
        remaining -= chunk
    distribution = np.concatenate(samples)
    return {
        "mean": float(np.mean(differences)),
        "lower95": float(np.percentile(distribution, 2.5)),
        "upper95": float(np.percentile(distribution, 97.5)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def audit_campaign(campaign_directory: Path) -> Dict[str, Any]:
    campaign = Path(campaign_directory).expanduser().resolve()
    issues: List[str] = []
    contract_path = campaign / "control" / "campaign_contract.json"
    summary_path = campaign / "campaign_summary.json"
    test_report_path = campaign / "unit_test_report.json"
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
    if not bool(contract.get("development_only", False)):
        issues.append("development-only boundary is absent")
    if not bool(contract.get("append_only_v27_unchanged", False)):
        issues.append("append-only V27 boundary is absent")
    if not bool(contract.get("no_rl_training", False)):
        issues.append("no-RL boundary is absent")
    if contract.get("arm") != v34.ARM_NAME:
        issues.append("V34 arm identity changed")
    if [float(value) for value in contract.get("checkpoints_s", [])] != list(
        v34.CHECKPOINTS_S
    ):
        issues.append("V34 checkpoints changed")

    seeds = [int(value) for value in contract.get("episode_seeds", [])]
    if any(v34.RESERVED_SEED_START <= seed <= v34.FINAL_SEED_END for seed in seeds):
        issues.append("campaign intersects reserved/final seeds")
    if contract.get("reserved_seed_range_untouched") != [
        v34.RESERVED_SEED_START,
        v34.RESERVED_SEED_END,
    ]:
        issues.append("reserved seed declaration changed")
    if contract.get("sealed_final_seed_range_untouched") != [
        v34.FINAL_SEED_START,
        v34.FINAL_SEED_END,
    ]:
        issues.append("final seed declaration changed")

    manifest = contract.get("source_manifest", {})
    if not isinstance(manifest, Mapping):
        issues.append("source manifest is missing")
        manifest = {}
    for relative, expected_hash in manifest.items():
        snapshot = campaign / "control" / "source_snapshot" / str(relative)
        if not snapshot.is_file():
            issues.append(f"source snapshot missing: {relative}")
        elif v34.sha256_file(snapshot) != str(expected_hash):
            issues.append(f"source snapshot hash mismatch: {relative}")

    if not test_report_path.is_file():
        issues.append("unit-test report is missing")
    else:
        test_report = _read_json(test_report_path)
        if not bool(test_report.get("passed", False)) or int(
            test_report.get("returncode", -1)
        ) != 0:
            issues.append("unit tests did not pass")

    reference_rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    reference_section = contract.get("v27_reference", {})
    if not isinstance(reference_section, Mapping):
        issues.append("V27 reference section is missing")
        reference_section = {}
    for reference in reference_section.get("global_full_cells", []):
        if not isinstance(reference, Mapping):
            issues.append("invalid V27 reference row")
            continue
        path = Path(str(reference.get("path", "")))
        if not path.is_file() or v34.sha256_file(path) != str(reference.get("sha256", "")):
            issues.append(f"V27 reference hash mismatch: {path}")
            continue
        row = _read_json(path)
        key = (
            int(reference["episode_index"]),
            int(round(float(reference["checkpoint_s"]))),
        )
        if row.get("arm") != "global_full":
            issues.append(f"V27 reference is not global_full: {path}")
        if int(row.get("episode_index", -1)) != key[0] or int(
            round(float(row.get("prefix_s", -1)))
        ) != key[1]:
            issues.append(f"V27 reference identity mismatch: {path}")
        reference_rows[key] = row

    source_rows = {
        int(row["episode_index"]): row
        for row in contract.get("source_episodes", [])
        if isinstance(row, Mapping) and "episode_index" in row
    }
    episode_indices = [int(value) for value in contract.get("episode_indices", [])]
    expected_cells = len(episode_indices) * len(v34.CHECKPOINTS_S)
    if len(reference_rows) != expected_cells:
        issues.append("V27 reference cell count mismatch")
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
    recomputed_rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for episode_index in episode_indices:
        source = source_rows.get(episode_index)
        if source is None:
            issues.append(f"source row missing for episode {episode_index}")
            continue
        seed = int(source["episode_seed"])
        if seed != v19.V181_DEV_SEED_START + episode_index:
            issues.append(f"source seed mismatch for episode {episode_index}")
        try:
            v34.assert_v34_development_seed(seed)
        except PermissionError as exc:
            issues.append(str(exc))
        online_path = Path(str(source["online_path"]))
        truth_path = Path(str(source["truth_path"]))
        if not online_path.is_file() or v34.sha256_file(online_path) != source["online_sha256"]:
            issues.append(f"online source changed for episode {episode_index}")
            continue
        if not truth_path.is_file() or v34.sha256_file(truth_path) != source["truth_sha256"]:
            issues.append(f"truth source changed for episode {episode_index}")
            continue
        truth = v19.load_truth_diagnostics(truth_path)
        for checkpoint in v34.CHECKPOINTS_S:
            endpoint = int(round(checkpoint))
            stem = (
                f"episode_{episode_index:04d}_checkpoint_{endpoint:04d}_{v34.ARM_NAME}.json"
            )
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
                "checkpoint_s": float(checkpoint),
                "arm": v34.ARM_NAME,
                "online_inputs_sha256": source["online_sha256"],
            }
            for key, expected in identity.items():
                if unscored.get(key) != expected:
                    issues.append(f"unscored identity mismatch {key}: {stem}")
                if key in scored and scored.get(key) != expected:
                    issues.append(f"scored identity mismatch {key}: {stem}")
            if scored.get("truth_labels_sha256") != source["truth_sha256"]:
                issues.append(f"truth hash mismatch: {stem}")
            if scored.get("unscored_sha256") != v34.sha256_file(unscored_path):
                issues.append(f"unscored artifact hash mismatch: {stem}")
            if scored.get("unscored") != payload:
                issues.append(f"scored payload differs from unscored artifact: {stem}")
            diagnostics = payload.get("diagnostics", {})
            expected_window = min(endpoint, 60)
            expected_marginalized = max(endpoint - 60, 0)
            expected_updates = endpoint - 30
            if int(diagnostics.get("window_sample_count", -1)) != expected_window:
                issues.append(f"window count mismatch: {stem}")
            if int(diagnostics.get("marginalized_sample_count", -1)) != expected_marginalized:
                issues.append(f"marginalized count mismatch: {stem}")
            if int(diagnostics.get("total_sample_count", -1)) != endpoint:
                issues.append(f"total causal sample count mismatch: {stem}")
            if int(diagnostics.get("post_init_update_count", -1)) != expected_updates:
                issues.append(f"post-init update count mismatch: {stem}")
            latencies = payload.get("post_init_update_latencies_s", [])
            if not isinstance(latencies, list) or len(latencies) != expected_updates:
                issues.append(f"post-init latency list mismatch: {stem}")
            elif any(not math.isfinite(float(value)) or float(value) < 0.0 for value in latencies):
                issues.append(f"invalid post-init latency: {stem}")
            expected_score = _manual_score(payload, truth, float(checkpoint))
            actual_score = scored.get("score")
            if not isinstance(actual_score, Mapping):
                issues.append(f"score missing: {stem}")
                continue
            for key, expected in expected_score.items():
                if not _close(actual_score.get(key), expected):
                    issues.append(f"score mismatch {key}: {stem}")
            recomputed_rows[(episode_index, endpoint)] = expected_score
            checked_cells += 1

    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        issues.append("campaign summary lacks aggregate")
    else:
        if int(aggregate.get("cell_count", -1)) != expected_cells:
            issues.append("aggregate cell count mismatch")
        by_checkpoint = aggregate.get("by_checkpoint", {})
        paired = aggregate.get("paired_mhe_minus_v27_global_full_error_m", {})
        if not isinstance(by_checkpoint, Mapping) or not isinstance(paired, Mapping):
            issues.append("aggregate tables are missing")
        else:
            for checkpoint in v34.CHECKPOINTS_S:
                endpoint = int(round(checkpoint))
                key = str(endpoint)
                rows_at_checkpoint = [
                    score
                    for (episode, stored_endpoint), score in recomputed_rows.items()
                    if stored_endpoint == endpoint
                ]
                summary_row = by_checkpoint.get(key)
                if not isinstance(summary_row, Mapping):
                    issues.append(f"aggregate checkpoint missing: {key}")
                    continue
                success = sum(bool(row["success_le_7m"]) for row in rows_at_checkpoint)
                false_confidence = sum(
                    bool(row["false_confidence_radius_lt_7_error_gt_7"])
                    for row in rows_at_checkpoint
                )
                if int(summary_row.get("success_le_7m_count", -1)) != int(success):
                    issues.append(f"aggregate success mismatch: {key}")
                if int(summary_row.get("false_confidence_count", -1)) != int(false_confidence):
                    issues.append(f"aggregate false-confidence mismatch: {key}")
                if len(rows_at_checkpoint) == len(episode_indices):
                    differences = np.asarray(
                        [
                            float(recomputed_rows[(episode, endpoint)]["endpoint_position_error_m"])
                            - float(reference_rows[(episode, endpoint)]["score"]["endpoint_position_error_m"])
                            for episode in sorted(episode_indices)
                        ],
                        dtype=np.float64,
                    )
                    expected_ci = _bootstrap_mean_ci(differences)
                    actual_ci = paired.get(key)
                    if not isinstance(actual_ci, Mapping):
                        issues.append(f"paired interval missing: {key}")
                    else:
                        for statistic, expected in expected_ci.items():
                            if not _close(actual_ci.get(statistic), expected, tolerance=1e-12):
                                issues.append(f"paired interval mismatch {statistic}: {key}")

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
        "causal_sample_accounting_checked": True,
        "paired_v27_reference_checked": True,
        "reserved_and_final_ranges_untouched": not forbidden_paths,
        "contract_sha256": stored_contract_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a V34 MHE campaign")
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args(argv)
    result = audit_campaign(args.campaign)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

